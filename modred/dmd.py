from __future__ import print_function
from __future__ import absolute_import
from future.builtins import object

import numpy as np

from .vectorspace import VectorSpaceMatrices, VectorSpaceHandles
from . import util
from .parallel import parallel_default_instance
_parallel = parallel_default_instance


def compute_DMD_matrices_snaps_method(
    vecs, mode_indices, adv_vecs=None, inner_product_weights=None, atol=1e-13,
    rtol=None, max_num_eigvals=None, return_all=False):
    """Computes DMD modes using data stored in matrices, using method of
    snapshots.

    Args:
        ``vecs``: Matrix whose columns are data vectors.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 
    
    Kwargs:
        ``adv_vecs``: Matrix whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``inner_product_weights``: 1D array or matrix of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
        
        ``atol``: Level below which eigenvalues of correlation matrix are 
        truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest 
        eigenvalues of correlation matrix.  Smaller ones are truncated.

        ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
        computed.  This is enforced by truncating the basis onto which the
        approximating linear map is projected.  Computationally, this
        corresponds to truncating the eigendecomposition of the correlation
        matrix. If set to None, no truncation will be performed, and the
        maximum possible number of DMD eigenvalues will be computed.

        ``return_all``: Return more objects, see below. Default is false.

    Returns:
        ``exact_modes``: Matrix whose columns are exact DMD modes.

        ``proj_modes``: Matrix whose columns are projected DMD modes.

        ``eigvals``: 1D array of eigenvalues of approximating low-order linear
        map (DMD eigenvalues).
    
        ``spectral_coeffs``: 1D array of DMD spectral coefficients, based on 
        projection of first data vector.
                        
        If ``return_all`` is true, also returns:
        
        ``R_low_order_eigvecs``: Matrix of right eigenvectors of approximating
        low-order linear map.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of approximating
        low-order linear map.

        ``correlation_mat_eigvals``: 1D array of eigenvalues of 
        correlation matrix.

        ``correlation_mat_eigvecs``: Matrix of eigenvectors of 
        correlation matrix.

        ``correlation_mat``: Correlation matrix; elements are inner products of
        data vectors with each other.

        ``cross_correlation_mat``: Cross-correlation matrix; elements are inner
        products of data vectors with data vectors advanced in time. Going down
        rows, the data vector changes; going across columns the advanced data
        vector changes.
   
    This uses the method of snapshots, which is faster than the direct method
    (see :py:func:`compute_DMD_matrices_direct_method`) when ``vecs`` has more
    rows than columns, i.e., when there are more elements in a vector than
    there are vectors. However, it "squares" this matrix and its singular
    values, making it slightly less accurate than the direct method."""
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceMatrices(weights=inner_product_weights)
    vecs = util.make_mat(vecs)
    # Sequential dataset
    if adv_vecs is None:
        # Compute correlation mat for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row and column 
        # yet.
        expanded_correlation_mat = \
            vec_space.compute_symmetric_inner_product_mat(vecs)
        correlation_mat = expanded_correlation_mat[:-1, :-1]
        cross_correlation_mat = expanded_correlation_mat[:-1, 1:]
    # Non-sequential data
    else:
        adv_vecs = util.make_mat(adv_vecs)
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        # Compute the correlation matrix from the unadvanced snapshots only.
        correlation_mat = np.mat(vec_space.compute_symmetric_inner_product_mat(
            vecs))
        cross_correlation_mat = np.mat(vec_space.compute_inner_product_mat(
            vecs, adv_vecs))
    
    correlation_mat_eigvals, correlation_mat_eigvecs = util.eigh(
        correlation_mat, is_positive_definite=True, atol=atol, rtol=rtol)

    # Truncate if necessary
    if max_num_eigvals is not None and (
        max_num_eigvals < correlation_mat_eigvals.size):
        correlation_mat_eigvals = correlation_mat_eigvals[:max_num_eigvals]
        correlation_mat_eigvecs = correlation_mat_eigvecs[:, :max_num_eigvals]
     
    # Compute low-order linear map for sequential or non-sequential case.
    correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(
        correlation_mat_eigvals ** -0.5))
    low_order_linear_map = (
        correlation_mat_eigvals_sqrt_inv * correlation_mat_eigvecs.H * 
        cross_correlation_mat * correlation_mat_eigvecs * 
        correlation_mat_eigvals_sqrt_inv)

    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
        util.eig_biorthog(low_order_linear_map, scale_choice='left')
    build_coeffs_proj = (
        correlation_mat_eigvecs * correlation_mat_eigvals_sqrt_inv * 
        R_low_order_eigvecs)
    build_coeffs_exact = build_coeffs_proj * np.mat(np.diag(eigvals ** -1.))
    spectral_coeffs = np.abs(np.array(
        L_low_order_eigvecs.H *
        np.mat(np.diag(np.sqrt(correlation_mat_eigvals))) *
        correlation_mat_eigvecs[0, :].T).squeeze())

    # For sequential data, user must provide one more vec than columns of 
    # build_coeffs. 
    if vecs.shape[1] - build_coeffs_exact.shape[0] == 1:
        exact_modes = vec_space.lin_combine(vecs[:, 1:], 
            build_coeffs_exact, coeff_mat_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(vecs[:, :-1], 
            build_coeffs_proj, coeff_mat_col_indices=mode_indices)
    # For non-sequential data, user must provide as many vecs as columns of 
    # build_coeffs. 
    elif vecs.shape[1] == build_coeffs_exact.shape[0]:
        exact_modes = vec_space.lin_combine(
            adv_vecs, build_coeffs_exact, coeff_mat_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs, build_coeffs_proj, coeff_mat_col_indices=mode_indices)
    else:
        raise ValueError(('Number of cols in vecs does not match '
            'number of rows in build_coeffs matrix.'))

    if return_all:
        return (
            exact_modes, proj_modes, eigvals, spectral_coeffs, 
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs, correlation_mat, cross_correlation_mat)
    else:
        return exact_modes, proj_modes, eigvals, spectral_coeffs


def compute_DMD_matrices_direct_method(
    vecs, mode_indices, adv_vecs=None, inner_product_weights=None, atol=1e-13,
    rtol=None, max_num_eigvals=None, return_all=False):
    """Computes DMD modes using data stored in matrices, using direct method. 

    Args:
        ``vecs``: Matrix whose columns are data vectors.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``. 
   
    Kwargs:
        ``adv_vecs``: Matrix whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``inner_product_weights``: 1D array or matrix of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
        
        ``atol``: Level below which eigenvalues of correlation matrix are 
        truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest 
        eigenvalues of correlation matrix.  Smaller ones are truncated.

        ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
        computed.  This is enforced by truncating the basis onto which the
        approximating linear map is projected.  Computationally, this
        corresponds to truncating the eigendecomposition of the correlation
        matrix. If set to None, no truncation will be performed, and the
        maximum possible number of DMD eigenvalues will be computed.

        ``return_all``: Return more objects, see below. Default is false.
 
    Returns:

        ``exact_modes``: Matrix whose columns are exact DMD modes.

        ``proj_modes``: Matrix whose columns are projected DMD modes.

        ``eigvals``: 1D array of eigenvalues of approximating low-order linear
        map (DMD eigenvalues).
        
        ``spectral_coeffs``: 1D array of DMD spectral coefficients, based on 
        projection of first data vector.

        If ``return_all`` is true, also returns:
        
        ``R_low_order_eigvecs``: Matrix of right eigenvectors of approximating
        low-order linear map.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of approximating
        low-order linear map.

        ``correlation_mat_eigvals``: 1D array of eigenvalues of correlation
        matrix, which are the squares of the singular values of the data matrix 

        ``correlation_mat_eigvecs``: Matrix of eigenvectors of correlation
        matrix, which are also the right singular vectors of the data matrix.

    This method does not square the matrix of vectors as in the method of
    snapshots (:py:func:`compute_DMD_matrices_snaps_method`). It's slightly
    more accurate, but slower when the number of elements in a vector is more
    than the number of vectors, i.e.,  when ``vecs`` has more rows than
    columns. 
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
        sqrt_weights = np.mat(np.diag(inner_product_weights**0.5))
        vecs_weighted = sqrt_weights * vecs
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights * adv_vecs
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print('Warning: Cholesky decomposition could be time consuming.')
        sqrt_weights = np.mat(np.linalg.cholesky(inner_product_weights)).H
        vecs_weighted = sqrt_weights * vecs
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights * adv_vecs
    
    # Compute low-order linear map for sequential snapshot set.  This takes
    # advantage of the fact that for a sequential dataset, the unadvanced
    # and advanced vectors overlap.
    if adv_vecs is None:        
        U, sing_vals, correlation_mat_eigvecs = util.svd(
            vecs_weighted[:, :-1], atol=atol, rtol=rtol)
        
        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < sing_vals.size):
            U = U[:, :max_num_eigvals]
            sing_vals = sing_vals[:max_num_eigvals]
            correlation_mat_eigvecs = correlation_mat_eigvecs[
                :, :max_num_eigvals]

        correlation_mat_eigvals = sing_vals ** 2.
        correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(sing_vals ** -1.))
        correlation_mat = (
            correlation_mat_eigvecs * 
            np.mat(np.diag(correlation_mat_eigvals)) * 
            correlation_mat_eigvecs.H)
        last_col = U.H * vecs_weighted[:, -1]
        low_order_linear_map = np.mat(np.concatenate(
            (correlation_mat_eigvals_sqrt_inv * correlation_mat_eigvecs.H * \
            correlation_mat[:, 1:], last_col), axis=1)) * \
            correlation_mat_eigvecs * correlation_mat_eigvals_sqrt_inv
    else: 
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        U, sing_vals, correlation_mat_eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < sing_vals.size):
            U = U[:, :max_num_eigvals]
            sing_vals = sing_vals[:max_num_eigvals]
            correlation_mat_eigvecs = correlation_mat_eigvecs[
                :, :max_num_eigvals]

        correlation_mat_eigvals = sing_vals ** 2
        correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(sing_vals ** -1.))
        low_order_linear_map = (
            U.H * adv_vecs_weighted * 
            correlation_mat_eigvecs * correlation_mat_eigvals_sqrt_inv)
        
    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
        util.eig_biorthog(low_order_linear_map, scale_choice='left')
    build_coeffs_proj = (
        correlation_mat_eigvecs * correlation_mat_eigvals_sqrt_inv *
        R_low_order_eigvecs) 
    build_coeffs_exact = build_coeffs_proj * np.mat(np.diag(eigvals ** -1.))
    spectral_coeffs = np.abs(np.array(
        L_low_order_eigvecs.H *
        np.mat(np.diag(np.sqrt(correlation_mat_eigvals))) * 
        correlation_mat_eigvecs[0, :].T).squeeze())

    # For sequential data, user must provide one more vec than columns of 
    # build_coeffs. 
    if vecs.shape[1] - build_coeffs_exact.shape[0] == 1:
        exact_modes = vec_space.lin_combine(vecs[:, 1:], 
            build_coeffs_exact, coeff_mat_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(vecs[:, :-1], 
            build_coeffs_proj, coeff_mat_col_indices=mode_indices)
    # For sequential data, user must provide as many vecs as columns of 
    # build_coeffs. 
    elif vecs.shape[1] == build_coeffs_exact.shape[0]:
        exact_modes = vec_space.lin_combine(
            adv_vecs, build_coeffs_exact, coeff_mat_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs, build_coeffs_proj, coeff_mat_col_indices=mode_indices)
    else:
        raise ValueError(('Number of cols in vecs does not match '
            'number of rows in build_coeffs matrix.'))
    
    if return_all:
        return (
            exact_modes, proj_modes, eigvals, spectral_coeffs, 
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs)
    else:
        return exact_modes, proj_modes, eigvals, spectral_coeffs


class DMDHandles(object):
    """Dynamic Mode Decomposition implemented for large datasets.

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
               
    Computes DMD modes from vector objects (or handles).  It uses
    :py:class:`vectorspace.VectorSpaceHandles` for low level functions.

    Usage::

      myDMD = DMDHandles(my_inner_product)
      myDMD.compute_decomp(vec_handles)
      myDMD.compute_exact_modes(range(50), mode_handles)

    See also :func:`compute_DMD_matrices_snaps_method`,
    :func:`compute_DMD_matrices_direct_method`, and :mod:`vectors`.
    """
    def __init__(self, inner_product, 
        get_mat=util.load_array_text, put_mat=util.save_array_text,
        max_vecs_per_node=None, verbosity=1):
        """Constructor"""
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.eigvals = None
        self.correlation_mat = None
        self.cross_correlation_mat = None
        self.correlation_mat_eigvals = None
        self.correlation_mat_eigvecs = None
        self.low_order_linear_map = None
        self.L_low_order_eigvecs = None
        self.R_low_order_eigvecs = None
        self.spectral_coeffs = None
        self.proj_coeffs = None
        self.adv_proj_coeffs = None
        self.vec_space = VectorSpaceHandles(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.vec_handles = None
        self.adv_vec_handles = None
        
    def get_decomp(
        self, eigvals_src, R_low_order_eigvecs_src, L_low_order_eigvecs_src,
        correlation_mat_eigvals_src, correlation_mat_eigvecs_src):
        """Gets the decomposition matrices from sources (memory or file).
        
        Args:
            ``eigvals_src``: Source from which to retrieve eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_src``: Source from which to retrieve right
            eigenvectors of approximating low-order linear DMD map.
            
            ``L_low_order_eigvecs_src``: Source from which to retrieve left 
            eigenvectors of approximating low-order linear DMD map.

            ``correlation_mat_eigvals_src``: Source from which to retrieve 
            eigenvalues of correlation matrix.

            ``correlation_mat_eigvecs_src``: Source from which to retrieve 
            eigenvectors of correlation matrix.
        """        
        self.eigvals = np.squeeze(np.array(
            _parallel.call_and_bcast(self.get_mat, eigvals_src)))
        self.R_low_order_eigvecs = _parallel.call_and_bcast(
            self.get_mat, R_low_order_eigvecs_src)
        self.L_low_order_eigvecs = _parallel.call_and_bcast(
            self.get_mat, L_low_order_eigvecs_src)
        self.correlation_mat_eigvals = np.squeeze(np.array(
            _parallel.call_and_bcast(
            self.get_mat, correlation_mat_eigvals_src)))
        self.correlation_mat_eigvecs = _parallel.call_and_bcast(
            self.get_mat, correlation_mat_eigvecs_src)

    def put_decomp(
        self, eigvals_dest, R_low_order_eigvecs_dest, L_low_order_eigvecs_dest,
        correlation_mat_eigvals_dest, correlation_mat_eigvecs_dest):
        """Puts the decomposition matrices in destinations (file or memory).

        Args:
            ``eigvals_dest``: Destination in which to put eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_dest``: Destination in which to put right
            eigenvectors of approximating low-order linear map.
           
            ``L_low_order_eigvecs_dest``: Destination in which to put left 
            eigenvectors of approximating low-order linear map.
            
            ``correlation_mat_eigvals_dest``: Destination in which to put 
            eigenvalues of correlation matrix.

            ``correlation_mat_eigvecs_dest``: Destination in which to put 
            eigenvectors of correlation matrix.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_eigvals(eigvals_dest)
        self.put_R_low_order_eigvecs(R_low_order_eigvecs_dest)
        self.put_L_low_order_eigvecs(L_low_order_eigvecs_dest)
        self.put_correlation_mat_eigvals(correlation_mat_eigvals_dest)
        self.put_correlation_mat_eigvecs(correlation_mat_eigvecs_dest)

    def put_eigvals(self, dest):
        """Puts eigenvalues of approximating low-order-linear map (DMD
        eigenvalues) to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.eigvals, dest)
        _parallel.barrier()

    def put_R_low_order_eigvecs(self, dest):
        """Puts right eigenvectors of approximating low-order linear map to
        ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.R_low_order_eigvecs, dest)
        _parallel.barrier()

    def put_L_low_order_eigvecs(self, dest):
        """Puts left eigenvectors of approximating low-order linear map to 
        ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.L_low_order_eigvecs, dest)
        _parallel.barrier()

    def put_correlation_mat_eigvals(self, dest):
        """Puts eigenvalues of correlation matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat_eigvals, dest)
        _parallel.barrier()

    def put_correlation_mat_eigvecs(self, dest):
        """Puts eigenvectors of correlation matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat_eigvecs, dest)
        _parallel.barrier()

    def put_correlation_mat(self, dest):
        """Puts correlation mat to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, dest)
        _parallel.barrier()

    def put_cross_correlation_mat(self, dest):
        """Puts cross-correlation mat to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.cross_correlation_mat, dest)
        _parallel.barrier()

    def put_spectral_coeffs(self, dest):
        """Puts DMD spectral coefficients to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.spectral_coeffs, dest)
        _parallel.barrier()
    
    def put_proj_coeffs(self, dest, adv_dest):
        """Puts projection coefficients to ``dest``, advanced projection
        coefficients to ``adv_dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.proj_coeffs, dest)
            self.put_mat(self.adv_proj_coeffs, adv_dest)
        _parallel.barrier()

    def sanity_check(self, test_vec_handle):
        """Checks that user-supplied vector handle and vector satisfy 
        requirements.
        
        Args:
            ``test_vec_handle``: A vector handle to test.
        
        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    def compute_eigendecomp(self, atol=1e-13, rtol=None, max_num_eigvals=None):
        """Computes eigendecompositions of correlation matrix and approximating 
        low-order linear map.
       
        Kwargs:
            ``atol``: Level below which eigenvalues of correlation matrix are
            truncated.
            
            ``rtol``: Maximum relative difference between largest and smallest
            eigenvalues of correlation matrix.  Smaller ones are truncated. 

            ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
            computed.  This is enforced by truncating the basis onto which the
            approximating linear map is projected.  Computationally, this
            corresponds to truncating the eigendecomposition of the correlation
            matrix. If set to None, no truncation will be performed, and the
            maximum possible number of DMD eigenvalues will be computed.

        Useful if you already have the correlation matrix and cross-correlation 
        matrix and want to avoid recomputing them.

        Usage::
          
          DMD.correlation_mat = pre_existing_correlation_mat
          DMD.cross_correlation_mat = pre_existing_cross_correlation_mat
          DMD.compute_eigendecomp()
          DMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Another way to use this is to compute a DMD using a truncated basis for
        the projection of the approximating linear map.  Start by either
        computing a full decomposition or by loading pre-computed correlation
        and cross-correlation matrices.  

        Usage::

          # Start with a full decomposition 
          DMD_eigvals, correlation_mat_eigvals = DMD.compute_decomp(
              vec_handles)[0, 3]

          # Do some processing to determine the truncation level, maybe based
          # on the DMD eigenvalues and correlation matrix eigenvalues
          desired_num_eigvals = my_post_processing_func(
              DMD_eigvals, correlation_mat_eigvals)

          # Do a truncated decomposition
          DMD_eigvals_trunc = DMD.compute_eigendecomp(
            max_num_eigvals=desired_num_eigvals)
          
          # Compute modes for truncated decomposition 
          DMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)
        
        Since it doesn't overwrite the correlation and cross-correlation
        matrices, ``compute_eigendecomp`` can be called many times in a row to
        do computations for different truncation levels.  However, the results
        of the decomposition (e.g., ``self.eigvals``) do get overwritten, so
        you may want to call a ``put`` method to save those results somehow.
        """
        # Compute eigendecomposition of correlation matrix
        self.correlation_mat_eigvals, self.correlation_mat_eigvecs = \
            _parallel.call_and_bcast(
            util.eigh, self.correlation_mat, atol=atol, rtol=None,
            is_positive_definite=True)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < self.correlation_mat_eigvals.size):
            self.correlation_mat_eigvals = self.correlation_mat_eigvals[
                :max_num_eigvals]
            self.correlation_mat_eigvecs = self.correlation_mat_eigvecs[
                :, :max_num_eigvals]
                
        # Compute low-order linear map 
        correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(
            self.correlation_mat_eigvals ** -0.5))
        self.low_order_linear_map = (
            correlation_mat_eigvals_sqrt_inv * 
            self.correlation_mat_eigvecs.conj().T * 
            self.cross_correlation_mat * self.correlation_mat_eigvecs * 
            correlation_mat_eigvals_sqrt_inv)
        
        # Compute eigendecomposition of low-order linear map
        self.eigvals, self.R_low_order_eigvecs, self.L_low_order_eigvecs =\
            _parallel.call_and_bcast(
            util.eig_biorthog, self.low_order_linear_map, 
            **{'scale_choice':'left'})

    def compute_decomp(
        self, vec_handles, adv_vec_handles=None, atol=1e-13, rtol=None,
        max_num_eigvals=None):
        """Computes eigendecomposition of low-order linear map approximating
        relationship between vector objects, returning various matrices
        necessary for computing and characterizing DMD modes.
        
        Args:
            ``vec_handles``: List of handles for vector objects.
        
        Kwargs:
            ``adv_vec_handles``: List of handles for vector objects advanced in
            time.  If not provided, it is assumed that the vector objects
            describe a sequential time-series. Thus ``vec_handles`` becomes
            ``vec_handles[:-1]`` and ``adv_vec_handles`` becomes
            ``vec_handles[1:]``.
        
            ``atol``: Level below which DMD eigenvalues are truncated.
     
            ``rtol``: Maximum relative difference between largest and smallest 
            DMD eigenvalues.  Smaller ones are truncated.

            ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
            computed.  This is enforced by truncating the basis onto which the
            approximating linear map is projected.  Computationally, this
            corresponds to truncating the eigendecomposition of the correlation
            matrix. If set to None, no truncation will be performed, and the
            maximum possible number of DMD eigenvalues will be computed.
    
        Returns:
            ``eigvals``: 1D array of eigenvalues of low-order linear map, i.e., 
            the DMD eigenvalues.
            
            ``R_low_order_eigvecs``: Matrix whose columns are right eigenvectors
            of approximating low-order linear map.

            ``L_low_order_eigvecs``: Matrix whose columns are left eigenvectors 
            of approximating low-order linear map.

            ``correlation_mat_eigvals``: 1D array of eigenvalues of 
            correlation matrix.

            ``correlation_mat_eigvecs``: Matrix whose columns are eigenvectors 
            of correlation matrix.
        """
        self.vec_handles = vec_handles
        if adv_vec_handles is not None:
            self.adv_vec_handles = adv_vec_handles
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
            self.cross_correlation_mat = self.expanded_correlation_mat[:-1, 1:]
        # For non-sequential data, compute the correlation matrix from the 
        # unadvanced snapshots only.  Compute the cross correlation matrix
        # involving the unadvanced and advanced snapshots separately.
        else:
            self.correlation_mat = \
                self.vec_space.compute_symmetric_inner_product_mat(
                self.vec_handles)
            self.cross_correlation_mat = \
                self.vec_space.compute_inner_product_mat(
                self.vec_handles, self.adv_vec_handles)

        # Compute eigendecomposition of low-order linear map.
        self.compute_eigendecomp(
            atol=atol, rtol=rtol, max_num_eigvals=max_num_eigvals)

        return (
            self.eigvals, self.R_low_order_eigvecs,
            self.L_low_order_eigvecs, self.correlation_mat_eigvals,
            self.correlation_mat_eigvecs)
        
    def compute_exact_modes(self, mode_indices, mode_handles, 
        adv_vec_handles=None):
        """Computes exact DMD modes and calls ``put`` on them using mode
        handles.
        
        Args:
            ``mode_indices``: List of indices describing which exact modes to
            compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for exact modes to compute.
            
        Kwargs:
            ``vec_handles``: List of handles for vector objects. Optional if 
            when calling :py:meth:`compute_decomp`. 
        """
        # If advanced vec handles are passed in, set the internal attribute,
        if adv_vec_handles is not None:
            self.adv_vec_handles = adv_vec_handles

        # Compute build coefficient matrix
        build_coeffs_exact = (
            self.correlation_mat_eigvecs * 
            np.mat(np.diag(self.correlation_mat_eigvals ** -0.5)) *
            self.R_low_order_eigvecs
            * np.mat(np.diag(self.eigvals ** -1.)))

        # If the internal attribute is set, then compute the modes
        if self.adv_vec_handles is not None:
            self.vec_space.lin_combine(mode_handles, self.adv_vec_handles, 
                build_coeffs_exact, coeff_mat_col_indices=mode_indices)
        # If the internal attribute is not set, then check to see if
        # vec_handles is set.  If so, assume a sequential dataset, in which
        # case adv_vec_handles can be taken from a slice of vec_handles.  
        elif self.vec_handles is not None:
            if len(self.vec_handles) - build_coeffs_exact.shape[0] == 1:
                self.vec_space.lin_combine(mode_handles, self.vec_handles[1:], 
                    build_coeffs_exact, coeff_mat_col_indices=mode_indices)
            else:
                raise(ValueError, ('Number of vec_handles is not correct for a '
                    'sequential dataset.'))
        else:
            raise(ValueError, 'Neither vec_handles nor adv_vec_handles is '
                'defined.')

    def compute_proj_modes(self, mode_indices, mode_handles, vec_handles=None):
        """Computes projected DMD modes and calls ``put`` on them using mode
        handles.
        
        Args:
            ``mode_indices``: List of indices describing which projected modes
            to compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for projected modes to compute.
            
        Kwargs:
            ``vec_handles``: List of handles for vector objects. Optional if 
            when calling :py:meth:`compute_decomp`. 
        """
        if vec_handles is not None:
            self.vec_handles = vec_handles
        
        # Compute build coefficient matrix
        build_coeffs_proj = (
            self.correlation_mat_eigvecs * 
            np.mat(np.diag(self.correlation_mat_eigvals ** -0.5)) *
            self.R_low_order_eigvecs)

        # For sequential data, the user will provide a list vec_handles that
        # whose length is one larger than the number of rows of the 
        # build_coeffs matrix.  This is to be expected, as vec_handles is
        # essentially partitioned into two sets of handles, each of length one
        # less than vec_handles.
        if len(self.vec_handles) - build_coeffs_proj.shape[0] == 1:
            self.vec_space.lin_combine(mode_handles, self.vec_handles[:-1], 
                build_coeffs_proj, coeff_mat_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list vec_handles
        # whose length is equal to the number of rows in the build_coeffs
        # matrix.
        elif len(self.vec_handles) == build_coeffs_proj.shape[0]:
            self.vec_space.lin_combine(mode_handles, self.vec_handles, 
                build_coeffs_proj, coeff_mat_col_indices=mode_indices)
        # Otherwise, raise an error, as the number of handles should fit one of
        # the two cases described above.
        else:
            raise ValueError(('Number of vec_handles does not match number of '
                'columns in build_coeffs_proj matrix.'))

    def compute_spectrum(self):
        """Computes DMD spectral coefficients.  These coefficients come from a
        biorthogonal projection of the first vector object onto the exact DMD
        modes, which is analytically equivalent to doing a least-squares
        projection onto the projected DMD modes.
       
        Returns:
            ``spectral_coeffs``: 1D array of DMD spectral coefficients.
        """
        # TODO: maybe allow for user to choose which column to spectrum from?
        # ie first, last, or mean?  
        self.spectral_coeffs = np.abs(np.array(
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(np.sqrt(self.correlation_mat_eigvals))) * 
            np.mat(self.correlation_mat_eigvecs[0, :]).T).squeeze())
        return self.spectral_coeffs

    # Note that a biorthogonal projection onto the exact DMD modes is the same
    # as a least squares projection onto the projected DMD modes, so there is
    # only one method for computing the projection coefficients.
    def compute_proj_coeffs(self):
        """Computes projection of vector objects onto DMD modes.  Note that a
        biorthogonal projection onto exact DMD modes is analytically equivalent
        to a least-squares projection onto projected DMD modes.
       
        Returns:
            ``proj_coeffs``: Matrix of projection coefficients for vector
            objects, expressed as a linear combination of DMD modes.  Columns
            correspond to vector objects, rows correspond to DMD modes.

            ``adv_proj_coeffs``: Matrix of projection coefficients for vector
            objects advanced in time, expressed as a linear combination of DMD
            modes.  Columns correspond to vector objects, rows correspond to
            DMD modes.
        """
        self.proj_coeffs = ( 
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(np.sqrt(self.correlation_mat_eigvals))) * 
            self.correlation_mat_eigvecs.T)
        self.adv_proj_coeffs = (
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(self.correlation_mat_eigvals ** -0.5)) * 
            self.correlation_mat_eigvecs.T * self.cross_correlation_mat)
        return self.proj_coeffs, self.adv_proj_coeffs






