"""DMD class"""

import numpy as N
from vectorspace import VectorSpace
import util
import vectors as V
from parallel import parallel_default_instance
_parallel = parallel_default_instance

class DMD(object):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition.

    Args:
        ``inner_product``: Function to compute inner product.
        
    Kwargs:        
        ``put_mat``: Function to put a matrix out of modred.
      	
      	``get_mat``: Function to get a matrix into modred.
               
        ``max_vecs_per_node``: Max number of vectors in memory per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.
               
    Computes Ritz vectors from vecs.
    
    Usage::
    
      myDMD = DMD(my_inner_product)
      myDMD.compute_decomp(vec_handles)
      myDMD.compute_modes(range(50), mode_handles)
    
    """
    def __init__(self, inner_product, 
        get_mat=util.load_array_text, put_mat=util.save_array_text,
        max_vecs_per_node=None, verbosity=1):
        """Constructor"""
        self.vec_space = VectorSpace(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.ritz_vals = None
        self.build_coeffs = None
        self.mode_norms = None
        self.vec_handles = None
        self.vecs = None



    def sanity_check(self, test_vec_handle):
        """Check user-supplied vector handle.
        
        Args:
            ``test_vec_handle``: A vector handle.
        
        See :py:meth:`vectorspace.VectorSpace.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    def sanity_check_in_memory(self, test_vec):
        """Check user-supplied vector object.
        
        Args:
            ``test_vec``: A vector.
        
        See :py:meth:`vectorspace.VectorSpace.sanity_check_in_memory`.
        """
        self.vec_space.sanity_check_in_memory(test_vec)


    def get_decomp(self, ritz_vals_source, mode_norms_source, 
        build_coeffs_source):
        """Retrieves the decomposition matrices from sources."""        
        self.ritz_vals = N.squeeze(N.array(
            _parallel.call_and_bcast(self.get_mat, ritz_vals_source)))
        self.mode_norms = N.squeeze(N.array(
            _parallel.call_and_bcast(self.get_mat, mode_norms_source)))
        self.build_coeffs = _parallel.call_and_bcast(self.get_mat, 
            build_coeffs_source)
            
    def put_decomp(self, ritz_vals_dest, mode_norms_dest, build_coeffs_dest):
        """Puts the decomposition matrices in destinations."""
        # Don't need to check if rank is zero because the following methods do
        # that check already.  In fact, this will cause the code to hang due to
        # the barriers within those functions.
        self.put_ritz_vals(ritz_vals_dest)
        self.put_mode_norms(mode_norms_dest)
        self.put_build_coeffs(build_coeffs_dest)

    def put_ritz_vals(self, dest):
        """Puts the Ritz values to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.ritz_vals, dest)
        _parallel.barrier()
        
    def put_mode_norms(self, dest):
        """Puts the mode norms to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.mode_norms, dest)
        _parallel.barrier()
        
    def put_build_coeffs(self, dest):
        """Puts the build coeffs to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.build_coeffs, dest)
        _parallel.barrier()
        
    def put_correlation_mat(self, dest):
        """Puts the correlation mat to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, dest)
        _parallel.barrier()
    
            
    def compute_decomp(self, vec_handles):
        """Computes decomposition and returns eigen decomposition matrices.
        
        Args:
            ``vec_handles``: List of handles for the vecs.
                    
        Returns:
            ``ritz_vals``: 1D array of Ritz values.
            
            ``mode_norms``: 1D array of mode norms.
            
            ``build_coeffs``: 2D array of build coefficients for modes (T).
        """
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)
        if self.vec_handles is None:
            raise util.UndefinedError('vec_handles is not given')

        # Compute correlation mat for all vectors.  This is more efficient
        # because only one call is made to the inner product routine, even
        # though we don't need the last row/column yet.  Later we need all but
        # the last element of the last column, so it is faster to compute all
        # of this now.  Only one extra element is computed, since this is a
        # symmetric inner product matrix.
        self.correlation_mat = \
            self.vec_space.compute_symmetric_inner_product_mat(self.vec_handles)
        correlation_mat_evals, correlation_mat_evecs = \
            _parallel.call_and_bcast(util.eigh, self.correlation_mat[:-1, :-1],
                is_positive_definite=True)
        correlation_mat_evals_sqrt = N.mat(N.diag(correlation_mat_evals**-0.5))
        low_order_linear_map = correlation_mat_evals_sqrt *\
            correlation_mat_evecs.H * self.correlation_mat[:-1, 1:] *\
            correlation_mat_evecs * correlation_mat_evals_sqrt
        self.ritz_vals, low_order_evecs = _parallel.call_and_bcast(
            N.linalg.eig, low_order_linear_map)
        self.build_coeffs = correlation_mat_evecs *\
            correlation_mat_evals_sqrt * low_order_evecs *\
            N.diag(N.array(N.array(_parallel.call_and_bcast(N.linalg.inv, 
            low_order_evecs.H * low_order_evecs) * low_order_evecs.H *\
            correlation_mat_evals_sqrt * correlation_mat_evecs.H * 
            self.correlation_mat[:-1, 0]).squeeze(), ndmin=1))
        self.mode_norms = N.diag(self.build_coeffs.H * 
            self.correlation_mat[:-1, :-1] * self.build_coeffs).real
        return self.ritz_vals, self.mode_norms, self.build_coeffs
        
        
    def compute_decomp_in_memory(self, vecs):
        """Same as :py:meth:`compute_decomp` but takes vecs instead of handles."""
        self.vecs = util.make_list(vecs)
        vec_handles = [V.InMemoryVecHandle(v) for v in self.vecs]
        return self.compute_decomp(vec_handles)
        
    
    def compute_modes(self, mode_nums, mode_handles, vec_handles=None, 
        index_from=0):
        """Computes modes and calls ``put`` on them.
        
        Args:
            ``mode_nums``: List of mode numbers, ``range(10)`` or ``[3, 2, 5]``.
            
            ``mode_handles``: List of handles for modes.
            
        Kwargs:
            ``vec_handles``: List of handles for vecs, can omit if given in
            :py:meth:`compute_decomp`.

            ``index_from``: Integer to start numbering modes from, 0, 1, or other.
        """
        if self.build_coeffs is None:
            raise util.UndefinedError('self.build_coeffs is undefined.')
        # User should specify ALL vecs, even though all but last are used
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)
        self.vec_space.compute_modes(mode_nums, mode_handles, 
            self.vec_handles[:-1], self.build_coeffs, index_from=index_from)
        
    def compute_modes_in_memory(self, mode_nums, vecs=None, 
        index_from=0):
        """Computes modes and returns them.
        
        Args:
            ``mode_nums``: List of mode numbers, ``range(10)`` or ``[3, 2, 5]``.
            
        Kwargs:
            ``vecs``: List of vecs.
                Can omit if given in :py:meth:`compute_decomp`.

            ``index_from``: Integer to start numbering modes from, 0, 1, or other.
        
        Returns:
            ``modes``: List of all modes.

        In parallel, each MPI worker returns all modes.
        See :py:meth:`compute_modes`.
        """
        if self.build_coeffs is None:
            raise util.UndefinedError('self.build_coeffs is undefined.')
        # User should specify ALL vecs, even though all but last are used
        if vecs is not None:
            self.vecs = util.make_list(vecs)
        return self.vec_space.compute_modes_in_memory(mode_nums, 
            self.vecs[:-1], self.build_coeffs, index_from=index_from)
 
 
