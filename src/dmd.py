"""DMD class"""

import numpy as N
from vectorspace import *
import util
import vectors as V
from parallel import parallel_default_instance
_parallel = parallel_default_instance

class DMDBase(object):
    """Base class, instantiate :py:class:`DMDArrays` or :py:class:`DMDHandles`."""
    def __init__(self, get_mat=util.load_array_text, 
        put_mat=util.save_array_text, verbosity=1):
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.ritz_vals = None
        self.build_coeffs = None
        self.mode_norms = None
        self.correlation_mat = None
        
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
        # Don't check if rank is zero because the following methods do.
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

    def compute_eigen_decomp(self):
        """Computes eigen decomposition of low-order linear map and associated 
        DMD matrices."""
        correlation_mat_evals_sqrt = N.mat(N.diag(
            self.correlation_mat_evals**-0.5))
        self.ritz_vals, low_order_evecs = _parallel.call_and_bcast(
            N.linalg.eig, self.low_order_linear_map)
        self.build_coeffs = self.correlation_mat_evecs *\
            correlation_mat_evals_sqrt * low_order_evecs *\
            N.diag(N.array(N.array(_parallel.call_and_bcast(N.linalg.inv, 
            low_order_evecs.H * low_order_evecs) * low_order_evecs.H *\
            correlation_mat_evals_sqrt * self.correlation_mat_evecs.H * 
            self.correlation_mat[:, 0]).squeeze(), ndmin=1))
        self.mode_norms = N.diag(self.build_coeffs.H * 
            self.correlation_mat * self.build_coeffs).real


class DMDArrays(DMDBase):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition for small data.

    Kwargs:        
        ``put_mat``: Function to put a matrix out of modred.
      	
      	``get_mat``: Function to get a matrix into modred.
        
        ``inner_product_weights``: 1D or 2D array of weights.
               
    Computes Ritz vectors from vec_array.
    It uses :py:class:`vectorspace.VectorSpaceArrays` for low level functions.

    Usage::
    
      myDMD = DMDArrays()
      ritz_vals, mode_norms, build_coeffs = myDMD.compute_decomp(vec_array)
      modes = myDMD.compute_modes(range(50))
    
    """
    def __init__(self, get_mat=util.load_array_text, 
        put_mat=util.save_array_text,
        inner_product_weights=None, verbosity=1):
        if _parallel.is_distributed():
            raise RuntimeError('Cannot be used in parallel.')
        DMDBase.__init__(self, get_mat=get_mat, put_mat=put_mat,
            verbosity=verbosity)
        self.vec_space = VectorSpaceArrays(weights=inner_product_weights)
        self.vec_array = None

    def set_vec_array(self, vec_array):
        if vec_array.ndim == 1:
            self.vec_array = vec_array.reshape((vec_array.shape[0], 1))
        else:
            self.vec_array = vec_array

    def set_adv_vec_array(self, adv_vec_array):
        if adv_vec_array.ndim == 1:
            self.adv_vec_array = adv_vec_array.reshape(
                (adv_vec_array.shape[0], 1))
        else:
            self.adv_vec_array = adv_vec_array

    def compute_decomp(self, vec_array, adv_vec_array=None):
        """Computes decomposition and returns eigen decomposition matrices.
        
        Args:
            ``vec_array``: 2D array with vectors as columns.
        
        Kwargs:
            ``adv_vec_array``: 2D array with columns corresponding to the
            vectors advanced in time.  If this argument is not provided, it is
            assumed that the vectors come from a sequential time-series, and
            vec_array will be partitioned according to the indices [:-1] and
            [1:].
                
        Returns:
            ``ritz_vals``: 1D array of Ritz values.
            
            ``mode_norms``: 1D array of mode norms.
            
            ``build_coeffs``: 2D array of build coefficients for modes.
        """        
        # Always set vec_array
        self.set_vec_array(vec_array)

        # Only set adv_vec_array if it is not None
        if adv_vec_array is not None:
            # First check shape compatability
            if vec_array.shape != adv_vec_array.shape:
                raise ValueError(('vec_array and adv_vec_array are not the '
                    'same shape.'))
            else:
                self.set_adv_vec_array(adv_vec_array)
        else:
            self.adv_vec_array = None

        # For a sequential dataset, compute correlation mat for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row/column yet.
        # Later we need all but the last element of the last column, so it is
        # faster to compute all of this now.  Only one extra element is
        # computed, since this is a symmetric inner product matrix.  Then
        # slice the expanded correlation matrix accordingly.
        if adv_vec_array is None:
            self.expanded_correlation_mat =\
                N.mat(self.vec_space.compute_symmetric_inner_product_mat(
                self.vec_array))
            self.correlation_mat = self.expanded_correlation_mat[:-1, :-1]
        # For non-sequential data, compute the correlation matrix from the 
        # unadvanced snapshots only.
        else:
            self.correlation_mat = \
                N.mat(self.vec_space.compute_symmetric_inner_product_mat(
                self.vec_array))
        
        # Compute eigendecomposition of correlation matrix
        self.correlation_mat_evals, self.correlation_mat_evecs = \
            _parallel.call_and_bcast(util.eigh, self.correlation_mat, 
            is_positive_definite=True)
        correlation_mat_evals_sqrt = N.mat(N.diag(
            self.correlation_mat_evals**-0.5))
        
        # Compute low-order linear map for sequential snapshot set.  This takes
        # advantage of the fact that for a sequential dataset, the unadvanced
        # and advanced vectors overlap.
        if self.adv_vec_array is None:
            self.low_order_linear_map = correlation_mat_evals_sqrt *\
                self.correlation_mat_evecs.H *\
                self.expanded_correlation_mat[:-1, 1:] *\
                self.correlation_mat_evecs * correlation_mat_evals_sqrt
        # Compute low-order linear map for non-squential snapshot set
        else: 
            self.low_order_linear_map = correlation_mat_evals_sqrt *\
                self.correlation_mat_evecs.H *\
                self.vec_space.compute_inner_product_mat(self.vec_array,
                self.adv_vec_array) * self.correlation_mat_evecs *\
                correlation_mat_evals_sqrt
        
        # Compute eigendecomposition of low-order linear map, finish DMD
        # computation.
        self.compute_eigen_decomp()
        
        return self.ritz_vals, self.mode_norms, self.build_coeffs
        
        
    def compute_modes(self, mode_indices, vec_array=None):
        """Computes modes and returns them.
        
        Args:
            ``mode_indices``: List of mode numbers, 
            ``range(10)`` or ``[3, 0, 5]``.
            
        Kwargs:
            ``vec_array``: 2D array with vectors as columns.
                Can omit if given in :py:meth:`compute_decomp`.

        Returns:
            ``modes``: 2D array with requested modes as columns.
        """
        if self.build_coeffs is None:
            raise util.UndefinedError('build_coeffs is undefined.')
        if vec_array is not None:
            self.set_vec_array(vec_array)
        if self.vec_array is None:
            raise util.UndefinedError('vec_array is undefined.')

        # For sequential data, the user will provide a list vec_handles that
        # whose length is one larger than the number of columns of the 
        # build_coeffs matrix.  This is to be expected, as vec_handles is
        # essentially partitioned into two sets of handles, each of length one
        # less than vec_handles.
        if self.vec_array.shape[1] - self.build_coeffs.shape[1] == 1:
            return self.vec_space.lin_combine(self.vec_array[:, :-1], 
                self.build_coeffs, coeff_mat_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list vec_handles
        # whose length is equal to the number of columns in the build_coeffs
        # matrix.
        elif self.vec_array.shape[1] == self.build_coeffs.shape[1]:
            return self.vec_space.lin_combine(self.vec_array, 
                self.build_coeffs, coeff_mat_col_indices=mode_indices)
        # Otherwise, raise an error, as the number of handles should fit one of
        # the two cases described above.
        else:
            raise ValueError(('Number of cols in vec_array does not match '
                'number of cols in build_coeffs matrix.'))
         

class DMDHandles(DMDBase):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition for large data.

    Args:
        ``inner_product``: Function to compute inner product.
        
    Kwargs:        
        ``put_mat``: Function to put a matrix out of modred.
      	
      	``get_mat``: Function to get a matrix into modred.
               
        ``max_vecs_per_node``: Max number of vectors in memory per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.
               
    Computes Ritz vectors from vecs.
    It uses :py:class:`vectorspace.VectorSpaceHandles` for low level functions.

    Usage::
    
      myDMD = DMDHandles(my_inner_product)
      ritz_values, mode_norms, build_coeffs = myDMD.compute_decomp(vec_handles)
      myDMD.compute_modes(range(50), mode_handles)
    
    """
    def __init__(self, inner_product, 
        get_mat=util.load_array_text, put_mat=util.save_array_text,
        max_vecs_per_node=None, verbosity=1):
        """Constructor"""
        DMDBase.__init__(self, get_mat=get_mat, put_mat=put_mat,
            verbosity=verbosity)
        self.vec_space = VectorSpaceHandles(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.vec_handles = None


    def sanity_check(self, test_vec_handle):
        """Check user-supplied vector handle.
        
        Args:
            ``test_vec_handle``: A vector handle.
        
        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    
    def compute_decomp(self, vec_handles, adv_vec_handles=None):
        """Computes decomposition and returns eigen decomposition matrices.
        
        Args:
            ``vec_handles``: List of handles for the vectors.
        
        Kwargs:
            ``adv_vec_handles``: List of handles for the vectors advanced in
            time.  If this argument is not provided, it is assumed that the
            vectors come from a sequential time-series, and vec_handles will be
            partitioned according to the indices [:-1] and [1:].
       
        Returns:
            ``ritz_vals``: 1D array of Ritz values.
            
            ``mode_norms``: 1D array of mode norms.
            
            ``build_coeffs``: 2D array of build coefficients for modes.
        """
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)
        if self.vec_handles is None:
            raise util.UndefinedError('vec_handles is not given')
        if adv_vec_handles is not None:
            self.adv_vec_handles = util.make_list(adv_vec_handles)
            if len(self.vec_handles) != len(self.adv_vec_handles):
                raise ValueError(('Number of vec_handles and adv_vec_handles'
                    ' is not equal.'))
        else:
            self.adv_vec_handles = None

        # For a sequential dataset, compute correlation mat for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row/column yet.
        # Later we need all but the last element of the last column, so it is
        # faster to compute all of this now.  Only one extra element is
        # computed, since this is a symmetric inner product matrix.  Then
        # slice the expanded correlation matrix accordingly.
        if adv_vec_handles is None:
            self.expanded_correlation_mat =\
                self.vec_space.compute_symmetric_inner_product_mat(
                self.vec_handles)
            self.correlation_mat = self.expanded_correlation_mat[:-1, :-1]
        # For non-sequential data, compute the correlation matrix from the 
        # unadvanced snapshots only.
        else:
            self.correlation_mat = \
                self.vec_space.compute_symmetric_inner_product_mat(
                self.vec_handles)

        # Compute eigendecomposition of correlation matrix
        self.correlation_mat_evals, self.correlation_mat_evecs = \
            _parallel.call_and_bcast(util.eigh, self.correlation_mat, 
            is_positive_definite=True)
        correlation_mat_evals_sqrt = N.mat(N.diag(
            self.correlation_mat_evals**-0.5))
        
        # Compute low-order linear map for sequential snapshot set.  This takes
        # advantage of the fact that for a sequential dataset, the unadvanced
        # and advanced vectors overlap.
        if self.adv_vec_handles is None:
            self.low_order_linear_map = correlation_mat_evals_sqrt *\
                self.correlation_mat_evecs.H *\
                self.expanded_correlation_mat[:-1, 1:] *\
                self.correlation_mat_evecs * correlation_mat_evals_sqrt
        # Compute low-order linear map for non-squential snapshot set
        else: 
            self.low_order_linear_map = correlation_mat_evals_sqrt *\
                self.correlation_mat_evecs.H *\
                self.vec_space.compute_inner_product_mat(self.vec_handles,
                self.adv_vec_handles) * self.correlation_mat_evecs *\
                correlation_mat_evals_sqrt
        
        # Compute eigendecomposition of low-order linear map, finish DMD
        # computation.
        self.compute_eigen_decomp()

        return self.ritz_vals, self.mode_norms, self.build_coeffs
        
        
    def compute_modes(self, mode_indices, mode_handles, vec_handles=None):
        """Computes modes and calls ``put`` on them.
        
        Args:
            ``mode_indices``: List of mode indices, ``range(5)`` or 
            ``[3, 0, 5]``.
            
            ``mode_handles``: List of handles for modes.
            
        Kwargs:
            ``vec_handles``: List of handles for vecs, can omit if given in
            :py:meth:`compute_decomp`.
        """
        if self.build_coeffs is None:
            raise util.UndefinedError('self.build_coeffs is undefined.')
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)

        # For sequential data, the user will provide a list vec_handles that
        # whose length is one larger than the number of columns of the 
        # build_coeffs matrix.  This is to be expected, as vec_handles is
        # essentially partitioned into two sets of handles, each of length one
        # less than vec_handles.
        if len(self.vec_handles) - self.build_coeffs.shape[1] == 1:
            self.vec_space.lin_combine(mode_handles, self.vec_handles[:-1], 
                self.build_coeffs, coeff_mat_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list vec_handles
        # whose length is equal to the number of columns in the build_coeffs
        # matrix.
        elif len(self.vec_handles) == self.build_coeffs.shape[1]:
            self.vec_space.lin_combine(mode_handles, self.vec_handles, 
                self.build_coeffs, coeff_mat_col_indices=mode_indices)
        # Otherwise, raise an error, as the number of handles should fit one of
        # the two cases described above.
        else:
            raise ValueError(('Number of vec_handles does not match number of '
                'columns in build_coeffs matrix.'))
 
