from __future__ import absolute_import
from future.builtins import object

import numpy as np

from .vectorspace import VectorSpaceMatrices, VectorSpaceHandles
from . import util
from .parallel import parallel_default_instance
_parallel = parallel_default_instance


def compute_BPOD_matrices(
    direct_vecs, adjoint_vecs, direct_mode_indices, adjoint_mode_indices,
    inner_product_weights=None, atol=1e-13, rtol=None, return_all=False):
    """Computes BPOD modes using data stored in matrices, using method of
    snapshots.
        
    Args:
        ``direct_vecs``: Matrix whose columns are direct data vectors 
        (:math:`X`).
    
        ``adjoint_vecs``: Matrix whose columns are adjoint data vectors 
        (:math:`Y`).

        ``direct_mode_indices``: List of indices describing which direct modes
        to compute.  Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 

        ``adjoint_mode_indices``: List of indices describing which adjoint
        modes to compute.  Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 

    Kwargs:
        ``inner_product_weights``: 1D array or matrix of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
        
        ``atol``: Level below which Hankel singular values are truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest 
        Hankel singular values.  Smaller ones are truncated.

        ``return_all``: Return more objects; see below. Default is false.
        
    Returns:
        ``direct_modes``: Matrix whose columns are direct modes.
        
        ``adjoint_modes``: Matrix whose columns are adjoint modes.

        ``sing_vals``: 1D array of Hankel singular values (:math:`E`).
        
        If ``return_all`` is true, then also returns:
        
        ``L_sing_vecs``: Matrix whose columns are left singular vectors of
        Hankel matrix (:math:`U`).
    
        ``R_sing_vecs``: Matrix whose columns are right singular vectors of
        Hankel matrix (:math:`V`).

        ``Hankel_mat``: Hankel matrix (:math:`Y^* W X`).
        
    See also :py:class:`BPODHandles`.
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceMatrices(weights=inner_product_weights)
    direct_vecs = util.make_mat(direct_vecs)
    adjoint_vecs = util.make_mat(adjoint_vecs)
    
    #Hankel_mat = vec_space.compute_inner_product_mat(adjoint_vecs, 
    #    direct_vecs)
    first_adjoint_all_direct = vec_space.compute_inner_product_mat(
        adjoint_vecs[:,0], direct_vecs)
    all_adjoint_last_direct = vec_space.compute_inner_product_mat(
        adjoint_vecs, direct_vecs[:,-1])
    Hankel_mat = util.Hankel(first_adjoint_all_direct, all_adjoint_last_direct)
    L_sing_vecs, sing_vals, R_sing_vecs = util.svd(
        Hankel_mat, atol=atol, rtol=rtol)
    #print 'diff in Hankels',Hankel_mat - Hankel_mat2
    #Hankel_mat = Hankel_mat2
    sing_vals_sqrt_mat = np.mat(np.diag(sing_vals**-0.5))
    direct_build_coeff_mat = R_sing_vecs * sing_vals_sqrt_mat
    direct_mode_array = vec_space.lin_combine(direct_vecs, 
        direct_build_coeff_mat, coeff_mat_col_indices=direct_mode_indices)
     
    adjoint_build_coeff_mat = L_sing_vecs * sing_vals_sqrt_mat
    adjoint_mode_array = vec_space.lin_combine(adjoint_vecs,
        adjoint_build_coeff_mat, coeff_mat_col_indices=adjoint_mode_indices)
    
    if return_all:
        return direct_mode_array, adjoint_mode_array, sing_vals, L_sing_vecs, \
            R_sing_vecs, Hankel_mat
    else:
        return direct_mode_array, adjoint_mode_array, sing_vals
           

class BPODHandles(object):
    """Balanced Proper Orthogonal Decomposition implemented for large datasets.

    Args:    
        ``inner_product``: Function that computes inner product of two vector
        objects.

    Kwargs:
        ``put_mat``: Function to put a matrix out of modred, e.g., write it to
        file.
      	
      	``get_mat``: Function to get a matrix into modred, e.g., load it from
        file.

        ``max_vecs_per_node``: Maximum number of vectors that can be stored in 
        memory, per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.
   
    Computes direct and adjoint BPOD modes from direct and adjoint vector
    objects (or handles).  Uses :py:class:`vectorspace.VectorSpaceHandles` for
    low level functions.
    
    Usage::

      myBPOD = BPODHandles(my_inner_product, max_vecs_per_node=500)
      myBPOD.compute_decomp(direct_vec_handles, adjoint_vec_handles)
      myBPOD.compute_direct_modes(range(50), direct_modes)
      myBPOD.compute_adjoint_modes(range(50), adjoint_modes)

    See also :py:func:`compute_BPOD_matrices` and :mod:`vectors`.
    """
    def __init__(self, inner_product, 
        put_mat=util.save_array_text, get_mat=util.load_array_text,
        max_vecs_per_node=None, verbosity=1):
        """Constructor """
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.L_sing_vecs = None
        self.R_sing_vecs = None
        self.sing_vals = None
        self.Hankel_mat = None
        # Class that contains all of the low-level vec operations
        self.vec_space = VectorSpaceHandles(
            inner_product=inner_product, max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.direct_vec_handles = None
        self.adjoint_vec_handles = None

    def get_decomp(self, sing_vals_src, L_sing_vecs_src, R_sing_vecs_src):
        """Gets the decomposition matrices from sources (memory or file).
        
        Args:
            ``sing_vals_src``: Source from which to retrieve Hankel singular
            values.

            ``L_sing_vecs_src``: Source from which to retrieve left singular
            vectors of Hankel matrix.
                        
            ``R_sing_vecs_src``: Source from which to retrieve right singular
            vectors of Hankel matrix.
        """
        self.sing_vals = np.squeeze(_parallel.call_and_bcast(
            self.get_mat, sing_vals_src))
        self.L_sing_vecs = _parallel.call_and_bcast(
            self.get_mat, L_sing_vecs_src)
        self.R_sing_vecs = _parallel.call_and_bcast(
            self.get_mat, R_sing_vecs_src)
    
    def put_decomp(self, sing_vals_dest, L_sing_vecs_dest, R_sing_vecs_dest):
        """Puts the decomposition matrices in destinations (file or memory).
        
        Args:
            ``sing_vals_dest``: Destination in which to put Hankel singular
            values.

            ``L_sing_vecs_dest``: Destination in which to put left singular
            vectors of Hankel matrix.
           
            ``R_sing_vecs_dest``: Destination in which to put right singular
            vectors of Hankel matrix.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_L_sing_vecs(L_sing_vecs_dest)
        self.put_R_sing_vecs(R_sing_vecs_dest)
        self.put_sing_vals(sing_vals_dest)

    def put_L_sing_vecs(self, dest):
        """Puts left singular vectors of Hankel matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.L_sing_vecs, dest)
        _parallel.barrier()
        
    def put_sing_vals(self, dest):
        """Puts Hankel singular values to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.sing_vals, dest)
        _parallel.barrier()

    def put_R_sing_vecs(self, dest):
        """Puts right singular vectors of Hankel matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.R_sing_vecs, dest)
        _parallel.barrier()
            
    def put_Hankel_mat(self, dest):
        """Puts Hankel matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.Hankel_mat, dest)
        _parallel.barrier()
  
    def put_direct_proj_coeffs(self, dest):
        """Puts direct projection coefficients to ``dest``"""
        if _parallel.is_rank_zero():
            self.put_mat(self.proj_coeffs, dest)
        _parallel.barrier()

    def put_adjoint_proj_coeffs(self, dest):
        """Puts adjoint projection coefficients to ``dest``"""
        if _parallel.is_rank_zero():
            self.put_mat(self.adjoint_proj_coeffs, dest)
        _parallel.barrier()

    def compute_SVD(self, atol=1e-13, rtol=None):
        """Computes singular value decomposition of the Hankel matrix.
       
       Kwargs:
            ``atol``: Level below which Hankel singular values are truncated.
 
            ``rtol``: Maximum relative difference between largest and smallest 
            Hankel singular values.  Smaller ones are truncated.
 
        Useful if you already have the Hankel matrix and want to avoid
        recomputing it.  
        
        Usage::
        
          my_BPOD.Hankel_mat = pre_existing_Hankel_mat
          my_BPOD.compute_SVD()
          my_BPOD.compute_direct_modes(
              range(10), mode_handles, direct_vec_handles=direct_vec_handles)
        """
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = \
            _parallel.call_and_bcast(
            util.svd, self.Hankel_mat, atol=atol, rtol=rtol)

    def sanity_check(self, test_vec_handle):
        """Checks that user-supplied vector handle and vector satisfy 
        requirements.
        
        Args:
            ``test_vec_handle``: A vector handle to test.
        
        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)
    
    def compute_decomp(
        self, direct_vec_handles, adjoint_vec_handles, atol=1e-13, rtol=None):
        """Computes Hankel matrix :math:`H=Y^*X` and its singular value 
        decomposition :math:`UEV^*=H`.
        
        Args:
            ``direct_vec_handles``: List of handles for direct vector objects 
            (:math:`X`).
            
            ``adjoint_vec_handles``: List of handles for adjoint vector objects 
            (:math:`Y`).
       
        Kwargs:
            ``atol``: Level below which Hankel singular values are truncated.
     
            ``rtol``: Maximum relative difference between largest and smallest 
            Hankel singular values.  Smaller ones are truncated.
 
        Returns:
            ``sing_vals``: 1D array of Hankel singular values (:math:`E`).

            ``L_sing_vecs``: Matrix of left singular vectors of Hankel matrix
            (:math:`U`).
            
            ``R_sing_vecs``: Matrix of right singular vectors of Hankel matrix
            (:math:`V`).
        """
        self.direct_vec_handles = direct_vec_handles
        self.adjoint_vec_handles = adjoint_vec_handles
        self.Hankel_mat = self.vec_space.compute_inner_product_mat(
            self.adjoint_vec_handles, self.direct_vec_handles)
        self.compute_SVD(atol=atol, rtol=rtol)
        return self.sing_vals, self.L_sing_vecs, self.R_sing_vecs
            
    def compute_direct_modes(self, mode_indices, mode_handles, 
        direct_vec_handles=None):
        """Computes direct BPOD modes and calls ``put`` on them using mode
        handles.
        
        Args:
            ``mode_indices``: List of indices describing which direct modes to
            compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for direct modes to compute.

        Kwargs:
            ``direct_vec_handles``: List of handles for direct vector objects.
            Optional if given when calling :py:meth:`compute_decomp`. 
        """
        if direct_vec_handles is not None:
            self.direct_vec_handles = util.make_iterable(direct_vec_handles)
        if self.direct_vec_handles is None:
            raise util.UndefinedError('direct_vec_handles undefined')
            
        self.sing_vals = np.squeeze(np.array(self.sing_vals))
        build_coeff_mat = np.dot(
            self.R_sing_vecs, np.diag(self.sing_vals ** -0.5))
        
        self.vec_space.lin_combine(
            mode_handles, self.direct_vec_handles, build_coeff_mat,
            coeff_mat_col_indices=mode_indices)
            
    def compute_adjoint_modes(self, mode_indices, mode_handles, 
        adjoint_vec_handles=None):
        """Computes adjoint BPOD modes and calls ``put`` on them using mode
        handles.

        Args:
            ``mode_indices``: List of indices describing which adjoint modes to
            compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for adjoint modes to compute.

        Kwargs:
            ``adjoint_vec_handles``: List of handles for adjoint vector objects.
            Optional if given when calling :py:meth:`compute_decomp`. 
        """
        if adjoint_vec_handles is not None:
            self.adjoint_vec_handles = util.make_iterable(adjoint_vec_handles)
        if self.adjoint_vec_handles is None:
            raise util.UndefinedError('adjoint_vec_handles undefined')
        
        self.sing_vals = np.squeeze(np.array(self.sing_vals))
        build_coeff_mat = np.dot(
            self.L_sing_vecs, np.diag(self.sing_vals ** -0.5))
        
        self.vec_space.lin_combine(
            mode_handles, self.adjoint_vec_handles, build_coeff_mat, 
            coeff_mat_col_indices=mode_indices)

    def compute_proj_coeffs(self):
        """Computes biorthogonal projection of direct vector objects onto
        direct BPOD modes, using adjoint BPOD modes.  
       
        Returns:
            ``proj_coeffs``: Matrix of projection coefficients for direct
            vector objects, expressed as a linear combination of direct BPOD
            modes.  Columns correspond to direct vector objects, rows
            correspond to direct BPOD modes.
        """
        self.proj_coeffs = ( 
            np.mat(np.diag(self.sing_vals ** 0.5)) * self.R_sing_vecs.H)
        return self.proj_coeffs        

    def compute_adjoint_proj_coeffs(self):
        """Computes biorthogonal projection of adjoint vector objects onto
        adjoint BPOD modes, using direct BPOD modes.  
       
        Returns:
            ``adjoint_proj_coeffs``: Matrix of projection coefficients for
            adjoint vector objects, expressed as a linear combination of
            adjoint BPOD modes.  Columns correspond to adjoint vector objects,
            rows correspond to adjoint BPOD modes.
        """
        self.adjoint_proj_coeffs = ( 
            np.mat(np.diag(self.sing_vals ** 0.5)) * self.L_sing_vecs.H)
        return self.adjoint_proj_coeffs        


