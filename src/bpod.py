
import numpy as N
from vectorspace import VectorSpaceMatrices, VectorSpaceHandles
import util
from parallel import parallel_default_instance
_parallel = parallel_default_instance

def compute_BPOD_matrices(direct_vecs, adjoint_vecs, 
    direct_mode_indices, adjoint_mode_indices, inner_product_weights=None, 
    return_all=False):
    """Computes BPOD modes with data in a matrix.
        
    Args:
        ``direct_vecs``: Matrix with direct vecs as columns (:math:`X`).
    
        ``adjoint_vecs``: Matrix with adjoint vecs as columns (:math:`Y`).

        ``direct_mode_indices``: List of direct mode indices to compute. 
          Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 

        ``adjoint_mode_indices``: List of adjoint mode indices to compute. 
          Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 

    Kwargs:
        ``inner_product_weights``: 1D array or matrix of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
        
        ``return_all``: Return more objects, see below. Default is false.
        
    Returns:
        ``direct_modes``: Matrix with direct modes as columns.
        
        ``adjoint_modes``: Matrix with adjoint modes as columns.

        ``sing_vals``: 1D array of singular values of Hankel mat (:math:`E`).
        
        If ``return_all`` is true, then also returns:
        
        ``L_sing_vecs``: Matrix of left singular vectors of Hankel mat 
        (:math:`U`).
    
        ``R_sing_vecs``: Matrix of right singular vectors of Hankel mat
        (:math:`V`).

        ``Hankel_mat``: Hankel matrix (:math:`Y^* W X`).
        
    See also :py:class:`BPODHandles`.
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceMatrices(weights=inner_product_weights)
    direct_vecs = util.make_mat(direct_vecs)
    adjoint_vecs = util.make_mat(adjoint_vecs)
    
    Hankel_mat = vec_space.compute_inner_product_mat(adjoint_vecs, 
        direct_vecs)
    L_sing_vecs, sing_vals, R_sing_vecs = util.svd(Hankel_mat)
    
    sing_vals_sqrt_mat = N.mat(N.diag(sing_vals**-0.5))
    
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
    """Balanced Proper Orthogonal Decomposition for large data.

    Args:    
        ``inner_product``: Function to take inner products.

    Kwargs:
        ``put_mat``: Function to put a matrix out of modred.
      	
      	``get_mat``: Function to get a matrix into modred.

        ``max_vecs_per_node``: Max number of vectors in memory per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing 
   
    Computes direct and adjoint modes from direct and adjoint vecs.
    It uses :py:class:`vectorspace.VectorSpaceHandles` for low level functions.
    
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
        self.vec_space = VectorSpaceHandles(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.direct_vec_handles = None
        self.adjoint_vec_handles = None

    def get_decomp(self, L_sing_vecs_source, sing_vals_source, 
        R_sing_vecs_source):
        """Gets the decomposition matrices from elsewhere (memory or file).
        
        Args:
            ``L_sing_vecs_source``: Source from which to retrieve left singular
            vectors.
            
            ``sing_vals_source``: Source from which to retrieve singular
            values.
            
            ``R_sing_vecs_source``: Source from which to retrieve right singular
            vectors.
        """
        self.L_sing_vecs = _parallel.call_and_bcast(self.get_mat, 
            L_sing_vecs_source)
        self.sing_vals = N.squeeze(_parallel.call_and_bcast(self.get_mat, 
            sing_vals_source))
        self.R_sing_vecs = _parallel.call_and_bcast(self.get_mat, 
            R_sing_vecs_source)
    
    def put_Hankel_mat(self, dest):
        """Put Hankel mat to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.Hankel_mat, dest)
        _parallel.barrier()
        
    def put_L_sing_vecs(self, dest):
        """Put left singular vectors of SVD to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.L_sing_vecs, dest)
        _parallel.barrier()
        
    def put_R_sing_vecs(self, dest):
        """Put right singular vectors of SVD to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.R_sing_vecs, dest)
        _parallel.barrier()
        
    def put_sing_vals(self, dest):
        """Put singular values of SVD to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.sing_vals, dest)
        _parallel.barrier()
        
    def put_decomp(self, L_sing_vecs_dest, sing_vals_dest, R_sing_vecs_dest):
        """Put the decomposition matrices to destinations.
        
        Args:
            ``L_sing_vecs_dest``: Destination to which to put the left singular
            vectors.
            
            ``sing_vals_dest``: Destination to which to put the singular
            values.
            
            ``R_sing_vecs_dest``: Destination to which to put the right singular
            vectors.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_L_sing_vecs(L_sing_vecs_dest)
        self.put_R_sing_vecs(R_sing_vecs_dest)
        self.put_sing_vals(sing_vals_dest)
        

    def compute_SVD(self):
        """Takes the SVD of the Hankel matrix.
        
        Useful if you already have the Hankel mat and want to skip 
        recomputing it. 
        Instead, set ``self.Hankel_mat``. Usage::
        
          my_BPOD.Hankel_mat = pre_existing_Hankel_mat
          my_BPOD.compute_SVD()
          my_BPOD.compute_direct_modes(range(10), modes, 
              direct_vec_handles=my_direct_vec_handles)

        """
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = \
            _parallel.call_and_bcast(util.svd, self.Hankel_mat)

        
    def sanity_check(self, test_vec_handle):
        """Check that user-supplied vector handle and vector satisfy 
        requirements.
        
        Args:
            ``test_vec_handle``: A vector handle.
        
        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    
    def compute_decomp(self, direct_vec_handles, adjoint_vec_handles):
        """Finds Hankel matrix :math:`H=Y^*X` and its SVD, :math:`UEV^*=H`.
        
        Args:
            ``direct_vec_handles``: List of handles for direct vectors 
            (:math:`X`).
            
            ``adjoint_vec_handles``: List of handles for adjoint vectors 
            (:math:`Y`).
        
        Returns:
            ``L_sing_vecs``: Matrix of left singular vectors (:math:`U`).
        
            ``sing_vals``: 1D array of singular values (:math:`E`).
            
            ``R_sing_vecs``: Matrix of right singular vectors (:math:`V`).
        """
        self.direct_vec_handles = direct_vec_handles
        self.adjoint_vec_handles = adjoint_vec_handles
        self.Hankel_mat = self.vec_space.compute_inner_product_mat(
            self.adjoint_vec_handles, self.direct_vec_handles)
        self.compute_SVD()
        return self.L_sing_vecs, self.sing_vals, self.R_sing_vecs

            
    def compute_direct_modes(self, mode_indices, modes, 
        direct_vec_handles=None):
        """Computes direct modes and calls ``put`` on them.
        
        Args:
          ``mode_indices``: List of mode indices, ``range(10)`` or ``[3, 0, 6]``. 
              
          ``modes``: List of handles for direct modes.
          
        Kwargs:
          ``direct_vec_handles``: List of handles for direct vecs 
          (:math:`X`). 
              Optional if already given when calling :py:meth:`compute_decomp`. 
        """
        if direct_vec_handles is not None:
            self.direct_vec_handles = util.make_list(direct_vec_handles)
        if self.direct_vec_handles is None:
            raise util.UndefinedError('direct_vec_handles undefined')
            
        self.sing_vals = N.squeeze(N.array(self.sing_vals))
        build_coeff_mat = N.dot(self.R_sing_vecs, N.diag(self.sing_vals**-0.5))
        
        self.vec_space.lin_combine(modes, self.direct_vec_handles, 
            build_coeff_mat, coeff_mat_col_indices=mode_indices)
        

            
    def compute_adjoint_modes(self, mode_indices, modes, 
        adjoint_vec_handles=None):
        """Computes adjoint modes, calls ``put`` on them.
        
        Args:
          ``mode_indices``: List of mode numbers to compute. 
              Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 
              
          ``modes``: List of handles for adjoint modes.
          
        Kwargs:
          ``adjoint_vec_handles``: List of handles for adjoint vecs 
          (:math:`Y`). 
              Optional if already given when calling :py:meth:`compute_decomp`.
        """
        if adjoint_vec_handles is not None:
            self.adjoint_vec_handles = util.make_list(adjoint_vec_handles)
        if self.adjoint_vec_handles is None:
            raise util.UndefinedError('adjoint_vec_handles undefined')
        
        self.sing_vals = N.squeeze(N.array(self.sing_vals))
        build_coeff_mat = N.dot(self.L_sing_vecs, N.diag(self.sing_vals**-0.5))
        
        self.vec_space.lin_combine(modes, self.adjoint_vec_handles, 
            build_coeff_mat, coeff_mat_col_indices=mode_indices)
    
