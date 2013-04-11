
import numpy as N
from vectorspace import VectorSpaceMatrices, VectorSpaceHandles
import util
from parallel import parallel_default_instance
_parallel = parallel_default_instance


def compute_DMD_matrices_snaps_method(vecs, mode_indices, adv_vecs=None,
    inner_product_weights=None, return_all=False):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition with data in a
    matrix, using method of snapshots.

    Args:
        ``vecs``: Matrix with vectors as columns.
    
        ``mode_indices``: List of mode numbers, ``range(10)`` or ``[3, 0, 5]``.
    
    Kwargs:
        ``adv_vecs``: Matrix with ``vecs`` advanced in time as columns.
            If not provided, then it is assumed that the vectors are a 
            sequential time-series. Thus ``vecs`` becomes ``vecs[:-1]`` and
            ``adv_vecs`` becomes ``vecs[1:]``.

        ``inner_product_weights``: 1D or Matrix of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``return_all``: Return more objects, see below. Default is false.

    Returns:
        ``modes``: 2D array with requested modes as columns.

        ``ritz_vals``: 1D array of Ritz values.
        
        ``mode_norms``: 1D array of mode norms.
        
        If ``return_all`` is true, also returns:
        
        ``build_coeffs``: 2D array of build coefficients for modes.
    
    This uses the method of snapshots, which is faster than the direct method
    (in :py:func:`compute_DMD_matrices_direct_method`)
    when the ``vecs`` has more rows than columns (more elements in a vector
    than number of vectors). However, it "squares" this matrix and its singular
    values, making it slightly less accurate than the direct method."""
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceMatrices(weights=inner_product_weights)
    vecs = util.make_mat(vecs)
    # Sequential dataset
    if adv_vecs is None:
        # Compute correlation mat for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row and column yet.
        expanded_correlation_mat = \
            vec_space.compute_symmetric_inner_product_mat(vecs)
        correlation_mat = expanded_correlation_mat[:-1, :-1]
    # Non-sequential data
    else:
        adv_vecs = util.make_mat(adv_vecs)
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        # Compute the correlation matrix from the unadvanced snapshots only.
        correlation_mat = N.mat(vec_space.compute_symmetric_inner_product_mat(
            vecs))
    
    correlation_mat_evals, correlation_mat_evecs = util.eigh(correlation_mat, 
        is_positive_definite=True)
    correlation_mat_evals_sqrt = N.mat(N.diag(correlation_mat_evals**-0.5))
 
    # Compute low-order linear map for sequential or non-sequential case.
    # Sequential snapshot set. Takes advantage of the fact that the
    # the unadvanced and advanced vectors are the same except for first
    # and last entries.
    if adv_vecs is None:
        low_order_linear_map = correlation_mat_evals_sqrt * \
            correlation_mat_evecs.H * \
            expanded_correlation_mat[:-1, 1:] * \
            correlation_mat_evecs * correlation_mat_evals_sqrt
    # Non-sequential snapshot set.
    else: 
        low_order_linear_map = correlation_mat_evals_sqrt *\
            correlation_mat_evecs.H *\
            vec_space.compute_inner_product_mat(vecs,
            adv_vecs) * correlation_mat_evecs *\
            correlation_mat_evals_sqrt
    
    # Compute eigendecomposition of low-order linear map.
    ritz_vals, low_order_evecs = N.linalg.eig(low_order_linear_map)
    build_coeffs = correlation_mat_evecs *\
        correlation_mat_evals_sqrt * low_order_evecs *\
        N.diag(N.array(N.array(N.linalg.inv(
        low_order_evecs.H * low_order_evecs) * low_order_evecs.H *\
        correlation_mat_evals_sqrt * correlation_mat_evecs.H * 
        correlation_mat[:, 0]).squeeze(), ndmin=1))
    mode_norms = N.diag(build_coeffs.H * correlation_mat * build_coeffs).real
    if (mode_norms < 0).any():
        print ('Warning: mode norms has negative values. This is often happens '
            'when the rank of the vector matrix is much less than the number '
            'of columns. Try using fewer vectors (fewer columns).')
    # For sequential data, user must provide one more vec than columns of 
    # build_coeffs. 
    if vecs.shape[1] - build_coeffs.shape[0] == 1:
        modes = vec_space.lin_combine(vecs[:, :-1], 
            build_coeffs, coeff_mat_col_indices=mode_indices)
    # For sequential data, user must provide as many vecs as columns of 
    # build_coeffs. 
    elif vecs.shape[1] == build_coeffs.shape[0]:
        modes = vec_space.lin_combine(vecs, 
            build_coeffs, coeff_mat_col_indices=mode_indices)
    else:
        raise ValueError(('Number of cols in vecs does not match '
            'number of rows in build_coeffs matrix.'))
    
    if return_all:
        return modes, ritz_vals, mode_norms, build_coeffs    
    else:
        return modes, ritz_vals, mode_norms





def compute_DMD_matrices_direct_method(vecs, mode_indices, 
    adv_vecs=None, inner_product_weights=None, return_all=False):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition with data in a
    matrix, using a direct method.

    Args:
        ``vecs``: Matrix with vectors as columns.
    
        ``mode_indices``: List of mode numbers, ``range(10)`` or ``[3, 0, 5]``.
    
    Kwargs:
        ``adv_vecs``: Matrix with ``vecs`` advanced in time as columns.
            If not provided, then it is assumed that the vectors are a 
            sequential time-series. Thus ``vecs`` becomes ``vecs[:-1]`` and
            ``adv_vecs`` becomes ``vecs[1:]``.
            
        ``inner_product_weights``: 1D or Matrix of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``return_all``: Return more objects, see below. Default is false.

    Returns:
        ``modes``: Matrix with requested modes as columns.

        ``ritz_vals``: 1D array of Ritz values.
        
        ``mode_norms``: 1D array of mode norms.

        If ``return_all`` is true, also returns:

        ``build_coeffs``: Matrix of build coefficients for modes.
        
    This method does not square the matrix of vectors as in the method of
    snapshots (:py:func:`compute_DMD_matrices_snaps_method`). It's slightly 
    more accurate, but slower when the number of elements in a vector is 
    more than the number of vectors (more rows than columns in ``vecs``).
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceMatrices(weights=inner_product_weights)
    vecs = util.make_mat(vecs)
    if adv_vecs is not None:
        adv_vecs = util.make_mat(adv_vecs)
    
    if inner_product_weights is None:
        vecs_weighted = vecs
        if adv_vecs is not None:
            adv_vecs_weighted = adv_vecs
    elif inner_product_weights.ndim == 1:
        sqrt_weights = N.mat(N.diag(inner_product_weights**0.5))
        vecs_weighted = sqrt_weights * vecs
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights * adv_vecs
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print 'Warning: Cholesky decomposition could be time consuming.'
        sqrt_weights = N.mat(N.linalg.cholesky(inner_product_weights)).H
        vecs_weighted = sqrt_weights * vecs
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights * adv_vecs
    
    # Compute low-order linear map for sequential snapshot set.  This takes
    # advantage of the fact that for a sequential dataset, the unadvanced
    # and advanced vectors overlap.
    if adv_vecs is None:        
        U, sing_vals, correlation_mat_evecs = util.svd(vecs_weighted[:,:-1])
        correlation_mat_evals = sing_vals**2
        correlation_mat = correlation_mat_evecs * \
            N.mat(N.diag(correlation_mat_evals)) * correlation_mat_evecs.H
        last_col = U.H * vecs_weighted[:,-1]
        correlation_mat_evals_sqrt = N.mat(N.diag(sing_vals**-1.0))
        correlation_mat = correlation_mat_evecs * \
            N.mat(N.diag(correlation_mat_evals)) * correlation_mat_evecs.H

        low_order_linear_map = N.mat(N.concatenate(
            (correlation_mat_evals_sqrt * correlation_mat_evecs.H * \
            correlation_mat[:, 1:], last_col), axis=1)) * \
            correlation_mat_evecs * correlation_mat_evals_sqrt
    else: 
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        U, sing_vals, correlation_mat_evecs = util.svd(vecs_weighted)
        correlation_mat_evals_sqrt = N.mat(N.diag(sing_vals**-1.0))
        low_order_linear_map = U.H * adv_vecs_weighted * \
            correlation_mat_evecs * correlation_mat_evals_sqrt   
        correlation_mat_evals = sing_vals**2
        correlation_mat = correlation_mat_evecs * \
            N.mat(N.diag(correlation_mat_evals)) * correlation_mat_evecs.H

    # Compute eigendecomposition of low-order linear map.
    ritz_vals, low_order_evecs = N.linalg.eig(low_order_linear_map)
    build_coeffs = correlation_mat_evecs *\
        correlation_mat_evals_sqrt * low_order_evecs *\
        N.diag(N.array(N.array(N.linalg.inv(
        low_order_evecs.H * low_order_evecs) * low_order_evecs.H *\
        correlation_mat_evals_sqrt * correlation_mat_evecs.H * 
        correlation_mat[:, 0]).squeeze(), ndmin=1))
    mode_norms = N.diag(build_coeffs.H * correlation_mat * build_coeffs).real
    if (mode_norms < 0).any():
        print ('Warning: mode norms has negative values. This is often happens '
            'when the rank of the vector matrix is much less than the number '
            'of columns. Try using fewer vectors (fewer columns).')
    # For sequential data, the user will provide a vecs 
    # whose length is one larger than the number of columns of the 
    # build_coeffs matrix. 
    if vecs.shape[1] - build_coeffs.shape[0] == 1:
        modes = vec_space.lin_combine(vecs[:, :-1], 
            build_coeffs, coeff_mat_col_indices=mode_indices)
    # For a non-sequential dataset, user provides vecs
    # whose length is equal to the number of columns of build_coeffs
    elif vecs.shape[1] == build_coeffs.shape[0]:
        modes = vec_space.lin_combine(vecs, 
            build_coeffs, coeff_mat_col_indices=mode_indices)
    # Raise an error if number of handles isn't one of the two cases above.
    else:
        raise ValueError(('Number of cols in vecs does not match '
            'number of rows in build_coeffs matrix.'))
    
    if return_all:
        return modes, ritz_vals, mode_norms, build_coeffs    
    else:
        return modes, ritz_vals, mode_norms
    

         

class DMDHandles(object):
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
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.ritz_vals = None
        self.build_coeffs = None
        self.mode_norms = None
        self.correlation_mat = None
        self.vec_space = VectorSpaceHandles(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.vec_handles = None
        # Do these need to be member vars?
        self.adv_vec_handles = None
        self.correlation_mat_evals = None
        self.low_order_linear_map = None
        self.correlation_mat_evecs = None
        self.expanded_correlation_mat = None

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

    def _compute_eigen_decomp(self):
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
            ``adv_vec_handles``: List of handles of ``vecs`` advanced in time.
            If not provided, it is assumed that the
            vectors are a sequential time-series. Thus ``vec_handles`` becomes
            ``vec_handless[:-1]`` and ``adv_vec_handles`` 
            becomes ``vec_handles[1:]``.
            
        Returns:
            ``ritz_vals``: 1D array of Ritz values.
            
            ``mode_norms``: 1D array of mode norms.
            
            ``build_coeffs``: Matrix of build coefficients for modes.
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
        self._compute_eigen_decomp()
        if (self.mode_norms < 0).any() and self.verbosity > 0 and \
            _parallel.is_rank_zero():
            print >> output_channel, ('Warning: mode norms has negative '
                'values. This is often happens '
                'when the rank of the vector matrix is much less than the '
                'number of columns. Try using fewer vectors (fewer columns).')
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
        if len(self.vec_handles) - self.build_coeffs.shape[0] == 1:
            self.vec_space.lin_combine(mode_handles, self.vec_handles[:-1], 
                self.build_coeffs, coeff_mat_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list vec_handles
        # whose length is equal to the number of columns in the build_coeffs
        # matrix.
        elif len(self.vec_handles) == self.build_coeffs.shape[0]:
            self.vec_space.lin_combine(mode_handles, self.vec_handles, 
                self.build_coeffs, coeff_mat_col_indices=mode_indices)
        # Otherwise, raise an error, as the number of handles should fit one of
        # the two cases described above.
        else:
            raise ValueError(('Number of vec_handles does not match number of '
                'columns in build_coeffs matrix.'))
 
