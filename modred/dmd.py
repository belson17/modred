from __future__ import print_function
from __future__ import absolute_import
from future.builtins import object

import numpy as np

from .vectorspace import VectorSpaceMatrices, VectorSpaceHandles
from . import util
from . import parallel


def compute_DMD_matrices_snaps_method(
    vecs, adv_vecs=None, mode_indices=None, inner_product_weights=None,
    atol=1e-13, rtol=None, max_num_eigvals=None, return_all=False):
    """Computes DMD modes using data stored in matrices, using method of
    snapshots.

    Args:
        ``vecs``: Matrix whose columns are data vectors.

    Kwargs:
        ``adv_vecs``: Matrix whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

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

        ``spectral_coeffs``: 1D array of DMD spectral coefficients, calculated
        as the magnitudes of the projection coefficients of first data vector.
        The projection is onto the span of the DMD modes using the
        (biorthogonal) adjoint DMD modes.  Note that this is the same as a
        least-squares projection onto the span of the DMD modes.

        ``eigvals``: 1D array of eigenvalues of approximating low-order linear
        map (DMD eigenvalues).

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
    if parallel.is_distributed():
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
            exact_modes, proj_modes, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs, correlation_mat, cross_correlation_mat)
    else:
        return exact_modes, proj_modes, spectral_coeffs, eigvals


def compute_DMD_matrices_direct_method(
    vecs, adv_vecs=None, mode_indices=None, inner_product_weights=None,
    atol=1e-13, rtol=None, max_num_eigvals=None, return_all=False):
    """Computes DMD modes using data stored in matrices, using direct method.

    Args:
        ``vecs``: Matrix whose columns are data vectors.

    Kwargs:
        ``adv_vecs``: Matrix whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

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

        ``spectral_coeffs``: 1D array of DMD spectral coefficients, calculated
        as the magnitudes of the projection coefficients of first data vector.
        The projection is onto the span of the DMD modes using the
        (biorthogonal) adjoint DMD modes.  Note that this is the same as a
        least-squares projection onto the span of the DMD modes.

        ``eigvals``: 1D array of eigenvalues of approximating low-order linear
        map (DMD eigenvalues).

        If ``return_all`` is true, also returns:

        ``R_low_order_eigvecs``: Matrix of right eigenvectors of approximating
        low-order linear map.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of approximating
        low-order linear map.

        ``correlation_mat_eigvals``: 1D array of eigenvalues of correlation
        matrix, which are the squares of the singular values of the data
        matrix.

        ``correlation_mat_eigvecs``: Matrix of eigenvectors of correlation
        matrix, which are also the right singular vectors of the data matrix.

    This method does not square the matrix of vectors as in the method of
    snapshots (:py:func:`compute_DMD_matrices_snaps_method`). It's slightly
    more accurate, but slower when the number of elements in a vector is more
    than the number of vectors, i.e.,  when ``vecs`` has more rows than
    columns.
    """
    if parallel.is_distributed():
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
            exact_modes, proj_modes, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs)
    else:
        return exact_modes, proj_modes, spectral_coeffs, eigvals


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
    def __init__(
        self, inner_product, get_mat=util.load_array_text,
        put_mat=util.save_array_text, max_vecs_per_node=None, verbosity=1):
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
            parallel.call_and_bcast(self.get_mat, eigvals_src)))
        self.R_low_order_eigvecs = parallel.call_and_bcast(
            self.get_mat, R_low_order_eigvecs_src)
        self.L_low_order_eigvecs = parallel.call_and_bcast(
            self.get_mat, L_low_order_eigvecs_src)
        self.correlation_mat_eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(
            self.get_mat, correlation_mat_eigvals_src)))
        self.correlation_mat_eigvecs = parallel.call_and_bcast(
            self.get_mat, correlation_mat_eigvecs_src)


    def get_correlation_mat(self, src):
      """Gets the correlation matrix from source (memory or file).

        Args:
            ``src``: Source from which to retrieve correlation matrix.
        """
      self.correlation_mat = parallel.call_and_bcast(self.get_mat, src)


    def get_cross_correlation_mat(self, src):
        """Gets the cross-correlation matrix from source (memory or file).

        Args:
            ``src``: Source from which to retrieve cross-correlation matrix.
        """
        self.cross_correlation_mat = parallel.call_and_bcast(self.get_mat, src)


    def get_spectral_coeffs(self, src):
        """Gets the spectral coefficients from source (memory or file).

        Args:
            ``src``: Source from which to retrieve spectral coefficients.
        """
        self.spectral_coeffs = parallel.call_and_bcast(self.get_mat, src)


    def get_proj_coeffs(self, src, adv_src):
        """Gets the projection coefficients and advanced projection coefficients
        from sources (memory or file).

        Args:
            ``src``: Source from which to retrieve projection coefficients.

            ``adv_src``: Source from which to retrieve advanced projection
            coefficients.
        """
        self.proj_coeffs = parallel.call_and_bcast(self.get_mat, src)
        self.adv_proj_coeffs = parallel.call_and_bcast(self.get_mat, adv_src)


    def put_decomp(
        self, eigvals_dest, R_low_order_eigvecs_dest, L_low_order_eigvecs_dest,
        correlation_mat_eigvals_dest, correlation_mat_eigvecs_dest):
        """Puts the decomposition matrices in destinations (memory or file).

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
        if parallel.is_rank_zero():
            self.put_mat(self.eigvals, dest)
        parallel.barrier()


    def put_R_low_order_eigvecs(self, dest):
        """Puts right eigenvectors of approximating low-order linear map to
        ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.R_low_order_eigvecs, dest)
        parallel.barrier()


    def put_L_low_order_eigvecs(self, dest):
        """Puts left eigenvectors of approximating low-order linear map to
        ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.L_low_order_eigvecs, dest)
        parallel.barrier()


    def put_correlation_mat_eigvals(self, dest):
        """Puts eigenvalues of correlation matrix to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.correlation_mat_eigvals, dest)
        parallel.barrier()


    def put_correlation_mat_eigvecs(self, dest):
        """Puts eigenvectors of correlation matrix to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.correlation_mat_eigvecs, dest)
        parallel.barrier()


    def put_correlation_mat(self, dest):
        """Puts correlation mat to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, dest)
        parallel.barrier()


    def put_cross_correlation_mat(self, dest):
        """Puts cross-correlation mat to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.cross_correlation_mat, dest)
        parallel.barrier()


    def put_spectral_coeffs(self, dest):
        """Puts DMD spectral coefficients to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.spectral_coeffs, dest)
        parallel.barrier()


    def put_proj_coeffs(self, dest, adv_dest):
        """Puts projection coefficients to ``dest``, advanced projection
        coefficients to ``adv_dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.proj_coeffs, dest)
            self.put_mat(self.adv_proj_coeffs, adv_dest)
        parallel.barrier()


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
            parallel.call_and_bcast(
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
            parallel.call_and_bcast(
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

            ``R_low_order_eigvecs``: Matrix whose columns are right
            eigenvectors of approximating low-order linear map.

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
            self.eigvals,
            self.R_low_order_eigvecs,
            self.L_low_order_eigvecs,
            self.correlation_mat_eigvals,
            self.correlation_mat_eigvecs)


    def _compute_build_coeffs_exact(self):
        """Compute build coefficients for exact DMD modes."""
        return (
            self.correlation_mat_eigvecs *
            np.mat(np.diag(self.correlation_mat_eigvals ** -0.5)) *
            self.R_low_order_eigvecs)# *
            #np.mat(np.diag(self.eigvals ** -1.)))


    def _compute_build_coeffs_proj(self):
        """Compute build coefficients for projected DMD modes."""
        return (
            self.correlation_mat_eigvecs *
            np.mat(np.diag(self.correlation_mat_eigvals ** -0.5)) *
            self.R_low_order_eigvecs)


    def compute_exact_modes(
        self, mode_indices, mode_handles, adv_vec_handles=None):
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
        build_coeffs_exact = self._compute_build_coeffs_exact()

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
                raise(
                    ValueError, ('Number of vec_handles is not correct for a '
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
        build_coeffs_proj = self._compute_build_coeffs_proj()

        # For sequential data, the user will provide a list vec_handles that
        # whose length is one larger than the number of rows of the
        # build_coeffs matrix.  This is to be expected, as vec_handles is
        # essentially partitioned into two sets of handles, each of length one
        # less than vec_handles.
        if len(self.vec_handles) - build_coeffs_proj.shape[0] == 1:
            self.vec_space.lin_combine(mode_handles, self.vec_handles[:-1],
                build_coeffs_proj, coeff_mat_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list
        # vec_handles whose length is equal to the number of rows in the
        # build_coeffs matrix.
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
            ``spectral_coeffs``: 1D array of DMD spectral coefficients,
            calculated as the magnitudes of the projection coefficients of first
            data vector.  The projection is onto the span of the DMD modes using
            the (biorthogonal) adjoint DMD modes.  Note that this is the same as
            a least-squares projection onto the span of the DMD modes.

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


def compute_TLSqrDMD_matrices_snaps_method(
    vecs, mode_indices, adv_vecs=None, inner_product_weights=None, atol=1e-13,
    rtol=None, max_num_eigvals=None, return_all=False):
    """Computes Total Least Squares DMD modes using data stored in matrices,
    using method of snapshots.

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

        ``spectral_coeffs``: 1D array of DMD spectral coefficients, based on
        projection of first data vector.

        ``eigvals``: 1D array of eigenvalues of approximating low-order linear
        map (DMD eigenvalues).

        If ``return_all`` is true, also returns:

        ``R_low_order_eigvecs``: Matrix of right eigenvectors of approximating
        low-order linear map.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of approximating
        low-order linear map.

        ``summed_correlation_mats_eigvals``: 1D array of eigenvalues of
        summed correlation matrices.

        ``summed_correlation_mats_eigvecs``: Matrix whose columns are
        eigenvectors of summed correlation matrices.

        ``proj_correlation_mat_eigvals``: 1D array of eigenvalues of
        projected correlation matrix.

        ``proj_correlation_mat_eigvecs``: Matrix whose columns are
        eigenvectors of projected correlation matrix.

        ``correlation_mat``: Correlation matrix; elements are inner products of
        data vectors with each other.

        ``adv_correlation_mat``: Advanced correlation matrix; elements are
        inner products of advanced data vectors with each other.

        ``cross_correlation_mat``: Cross-correlation matrix; elements are inner
        products of data vectors with data vectors advanced in time. Going down
        rows, the data vector changes; going across columns the advanced data
        vector changes.

    This uses the method of snapshots, which is faster than the direct method
    (see :py:func:`compute_TLSqrDMD_matrices_direct_method`) when ``vecs`` has
    more rows than columns, i.e., when there are more elements in a vector than
    there are vectors. However, it "squares" this matrix and its singular
    values, making it slightly less accurate than the direct method.

    Note that max_num_eigvals must be set to a value smaller than the rank of
    the dataset.  In other words, if the projection basis for
    total-least-squares DMD is not truncated, then the algorithm reduces to
    standard DMD.  For over-constrained datasets (number of columns in data
    matrix is larger than the number of rows), this occurs naturally.  For
    under-constrained datasets, (number of vector objects is smaller than size
    of vector objects), this must be done explicitly by the user.  At this
    time, there is no standard method for choosing a truncation level.  One
    approach is to look at the roll-off of the correlation matrix eigenvalues,
    which contains information about the "energy" content of each projection
    basis vectors.
    """
    if parallel.is_distributed():
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
        adv_correlation_mat = expanded_correlation_mat[1:, 1:]
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
        adv_correlation_mat = np.mat(
            vec_space.compute_symmetric_inner_product_mat(
            adv_vecs))

    summed_correlation_mats_eigvals, summed_correlation_mats_eigvecs =\
        util.eigh(
        correlation_mat + adv_correlation_mat, is_positive_definite=True,
        atol=atol, rtol=rtol)

    # Truncate if necessary
    if max_num_eigvals is not None and (
        max_num_eigvals < summed_correlation_mats_eigvals.size):
        summed_correlation_mats_eigvals = summed_correlation_mats_eigvals[
            :max_num_eigvals]
        summed_correlation_mats_eigvecs = summed_correlation_mats_eigvecs[
            :, :max_num_eigvals]

    # Compute eigendecomposition of projected correlation matrix
    proj_correlation_mat = (
        summed_correlation_mats_eigvecs *
        summed_correlation_mats_eigvecs.H *
        correlation_mat *
        summed_correlation_mats_eigvecs *
        summed_correlation_mats_eigvecs.H)
    proj_correlation_mat_eigvals, proj_correlation_mat_eigvecs = util.eigh(
        proj_correlation_mat, atol=atol, rtol=None, is_positive_definite=True)

    # Truncate if necessary
    if max_num_eigvals is not None and (
        max_num_eigvals < proj_correlation_mat_eigvals.size):
        proj_correlation_mat_eigvals = proj_correlation_mat_eigvals[
            :max_num_eigvals]
        proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs[
            :, :max_num_eigvals]

    # Compute low-order linear map
    proj_correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(
        proj_correlation_mat_eigvals ** -0.5))
    low_order_linear_map = (
        proj_correlation_mat_eigvals_sqrt_inv *
        proj_correlation_mat_eigvecs.conj().H *
        summed_correlation_mats_eigvecs *
        summed_correlation_mats_eigvecs.H *
        cross_correlation_mat *
        summed_correlation_mats_eigvecs *
        summed_correlation_mats_eigvecs.H *
        proj_correlation_mat_eigvecs *
        proj_correlation_mat_eigvals_sqrt_inv)

    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
        util.eig_biorthog(low_order_linear_map, scale_choice='left')
    build_coeffs_proj = (summed_correlation_mats_eigvecs *
        summed_correlation_mats_eigvecs.H *
        proj_correlation_mat_eigvecs *
        np.mat(np.diag(proj_correlation_mat_eigvals ** -0.5)) *
        R_low_order_eigvecs)
    build_coeffs_exact = build_coeffs_proj * np.mat(np.diag(eigvals ** -1.))
    spectral_coeffs = np.abs(np.array(
        L_low_order_eigvecs.H *
        np.mat(np.diag(np.sqrt(proj_correlation_mat_eigvals))) *
        proj_correlation_mat_eigvecs[0, :].T)).squeeze()

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
            exact_modes, proj_modes, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs,
            summed_correlation_mats_eigvals, summed_correlation_mats_eigvecs,
            proj_correlation_mat_eigvals, proj_correlation_mat_eigvecs,
            correlation_mat, adv_correlation_mat, cross_correlation_mat)
    else:
        return exact_modes, proj_modes, eigvals, spectral_coeffs


def compute_TLSqrDMD_matrices_direct_method(
    vecs, mode_indices, adv_vecs=None, inner_product_weights=None, atol=1e-13,
    rtol=None, max_num_eigvals=None, return_all=False):
    """Computes Total Least Squares DMD modes using data stored in matrices,
    using direct method.

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

        ``spectral_coeffs``: 1D array of DMD spectral coefficients, based on
        projection of first data vector.

        ``eigvals``: 1D array of eigenvalues of approximating low-order linear
        map (DMD eigenvalues).

        If ``return_all`` is true, also returns:

        ``R_low_order_eigvecs``: Matrix of right eigenvectors of approximating
        low-order linear map.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of approximating
        low-order linear map.

        ``summed_correlation_mats_eigvals``: 1D array of eigenvalues of
        summed correlation matrices.

        ``summed_correlation_mats_eigvecs``: Matrix whose columns are
        eigenvectors of summed correlation matrices.

        ``proj_correlation_mat_eigvals``: 1D array of eigenvalues of
        projected correlation matrix.

        ``proj_correlation_mat_eigvecs``: Matrix whose columns are
        eigenvectors of projected correlation matrix.

    This method does not square the matrix of vectors as in the method of
    snapshots (:py:func:`compute_DMD_matrices_snaps_method`). It's slightly
    more accurate, but slower when the number of elements in a vector is more
    than the number of vectors, i.e.,  when ``vecs`` has more rows than
    columns.

    Note that max_num_eigvals must be set to a value smaller than the rank of
    the dataset.  In other words, if the projection basis for
    total-least-squares DMD is not truncated, then the algorithm reduces to
    standard DMD.  For over-constrained datasets (number of columns in data
    matrix is larger than the number of rows), this occurs naturally.  For
    under-constrained datasets, (number of vector objects is smaller than size
    of vector objects), this must be done explicitly by the user.  At this
    time, there is no standard method for choosing a truncation level.  One
    approach is to look at the roll-off of the correlation matrix eigenvalues,
    which contains information about the "energy" content of each projection
    basis vectors.
    """
    if parallel.is_distributed():
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

    # Compute projections of original data (to de-noise).  First consider the
    # sequential data case.
    if adv_vecs is None:
        stacked_U, stacked_sing_vals, summed_correlation_mats_eigvecs =\
            util.svd(np.vstack((vecs_weighted[:, :-1], vecs_weighted[:, 1:])),
            atol=atol, rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < stacked_sing_vals.size):
            stacked_U = stacked_U[:, :max_num_eigvals]
            stacked_sing_vals = stacked_sing_vals[:max_num_eigvals]
            summed_correlation_mats_eigvecs = summed_correlation_mats_eigvecs[
                :, :max_num_eigvals]

        # Project original data to de-noise
        vecs_proj = (vecs[:, :-1] *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
        adv_vecs_proj = (vecs[:, 1:] *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
    # Non-sequential data case
    else:
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        stacked_U, stacked_sing_vals, summed_correlation_mats_eigvecs =\
            util.svd(np.vstack((vecs_weighted, adv_vecs_weighted)), atol=atol,
            rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < stacked_sing_vals.size):
            stacked_U = stacked_U[:, :max_num_eigvals]
            stacked_sing_vals = stacked_sing_vals[:max_num_eigvals]
            summed_correlation_mats_eigvecs = summed_correlation_mats_eigvecs[
                :, :max_num_eigvals]

        # Project original data to de-noise
        vecs_proj = (vecs *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
        adv_vecs_proj = (adv_vecs *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)

    # Now proceed with DMD of projected data
    summed_correlation_mats_eigvals = stacked_sing_vals ** 2
    (exact_modes, proj_modes, spectral_coeffs, eigvals,
    R_low_order_eigvecs, L_low_order_eigvecs, proj_correlation_mat_eigvals,
    proj_correlation_mat_eigvecs) = compute_DMD_matrices_direct_method(
        vecs_proj, mode_indices, adv_vecs=adv_vecs_proj,
        inner_product_weights=inner_product_weights, atol=atol, rtol=rtol,
        max_num_eigvals=max_num_eigvals, return_all=True)

    if return_all:
        return (
            exact_modes, proj_modes, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs,
            summed_correlation_mats_eigvals,
            summed_correlation_mats_eigvecs,
            proj_correlation_mat_eigvals,
            proj_correlation_mat_eigvecs)
    else:
        return exact_modes, proj_modes, eigvals, spectral_coeffs


class TLSqrDMDHandles(DMDHandles):
    """Total Least Squares Dynamic Mode Decomposition implemented for large
    datasets.

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

    Computes Total-Least-Squares DMD modes from vector objects (or handles).
    It uses :py:class:`vectorspace.VectorSpaceHandles` for low level functions.

    Usage::

      myDMD = TLSqrDMDHandles(my_inner_product)
      myDMD.compute_decomp(vec_handles, max_num_eigvals=100)
      myDMD.compute_exact_modes(range(50), mode_handles)

    The total-least-squares variant of DMD accounts for an asymmetric treatment
    of the non-time-advanced and time-advanced data in standard DMD, in which
    the former is assumed to be perfect information, whereas all noise is
    limited to the latter.  In many cases, since the non-time-advanced and
    advanced data come from the same source, noise could be present in either.

    Note that max_num_eigvals must be set to a value smaller than the rank of
    the dataset.  In other words, if the projection basis for
    total-least-squares DMD is not truncated, then the algorithm reduces to
    standard DMD.  For over-constrained datasets (number of vector objects  is
    larger than the size of each vector object), this occurs naturally.  For
    under-constrained datasets, (number of vector objects is smaller than size
    of vector objects), this must be done explicitly by the user.  At this
    time, there is no standard method for choosing a truncation level.  One
    approach is to look at the roll-off of the correlation matrix eigenvalues,
    which contains information about the "energy" content of each projection
    basis vectors.

    Also, note that :class:`TLSqrDMDHandles` inherits from
    :class:`DMDHandles`, so
    certain methods are available, even though they are not
    implemented/documented here (namely several ``put`` functions).

    See also :func:`compute_TLSqrDMD_matrices_snaps_method`,
    :func:`compute_TLSqrDMD_matrices_direct_method`, and :mod:`vectors`.
    """
    def __init__(
        self, inner_product, get_mat=util.load_array_text,
        put_mat=util.save_array_text, max_vecs_per_node=None, verbosity=1):
        """Constructor"""
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.eigvals = None
        self.correlation_mat = None
        self.cross_correlation_mat = None
        self.adv_correlation_mat = None
        self.summed_correlation_mats_eigvals = None
        self.summed_correlation_mats_eigvecs = None
        self.proj_correlation_mat_eigvals = None
        self.proj_correlation_mat_eigvecs = None
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

          TLSqrDMD.correlation_mat = pre_existing_correlation_mat
          TLSqrDMD.adv_correlation_mat = pre_existing_adv_correlation_mat
          TLSqrDMD.cross_correlation_mat = pre_existing_cross_correlation_mat
          TLSqrDMD.compute_eigendecomp()
          TLSqrDMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Another way to use this is to compute TLSqrDMD using different basis
        truncation levels for the projection of the approximating linear map.
        Start by either computing a full decomposition or by loading
        pre-computed correlation and cross-correlation matrices.

        Usage::

          # Start with a full decomposition
          DMD_eigvals, correlation_mat_eigvals = TLSqrDMD.compute_decomp(
              vec_handles)[0, 3]

          # Do some processing to determine the truncation level, maybe based
          # on the DMD eigenvalues and correlation matrix eigenvalues
          desired_num_eigvals = my_post_processing_func(
              DMD_eigvals, correlation_mat_eigvals)

          # Do a truncated decomposition
          DMD_eigvals_trunc = TLSqrDMD.compute_eigendecomp(
            max_num_eigvals=desired_num_eigvals)

          # Compute modes for truncated decomposition
          TLSqrDMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Since it doesn't overwrite the correlation and cross-correlation
        matrices, ``compute_eigendecomp`` can be called many times in a row to
        do computations for different truncation levels.  However, the results
        of the decomposition (e.g., ``self.eigvals``) do get overwritten, so
        you may want to call a ``put`` method to save those results somehow.

        Note that the truncation level (corresponding to ``max_num_eigvals``)
        must be set to a value smaller than the rank of the dataset, otherwise
        total-least-squares DMD reduces to standard DMD.  This occurs naturally
        for over-constrained datasets, but must be enforced by the user for
        under-constrined datasets.
        """
        # Compute eigendecomposition of stacked correlation matrix
        self.summed_correlation_mats = (
            self.correlation_mat + self.adv_correlation_mat)
        (self.summed_correlation_mats_eigvals,
        self.summed_correlation_mats_eigvecs) = parallel.call_and_bcast(
            util.eigh, self.summed_correlation_mats,
            atol=atol, rtol=None, is_positive_definite=True)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < self.summed_correlation_mats_eigvals.size):
            self.summed_correlation_mats_eigvals =\
                self.summed_correlation_mats_eigvals[:max_num_eigvals]
            self.summed_correlation_mats_eigvecs =\
                self.summed_correlation_mats_eigvecs[:, :max_num_eigvals]

        # Compute eigendecomposition of projected correlation matrix
        self.proj_correlation_mat = (
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H *
            self.correlation_mat *
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H)
        (self.proj_correlation_mat_eigvals,
        self.proj_correlation_mat_eigvecs) = parallel.call_and_bcast(
            util.eigh, self.proj_correlation_mat ,
            atol=atol, rtol=None, is_positive_definite=True)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < self.proj_correlation_mat_eigvals.size):
            self.proj_correlation_mat_eigvals =\
                self.proj_correlation_mat_eigvals[:max_num_eigvals]
            self.proj_correlation_mat_eigvecs =\
                self.proj_correlation_mat_eigvecs[:, :max_num_eigvals]

        # Compute low-order linear map
        proj_correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(
            self.proj_correlation_mat_eigvals ** -0.5))
        self.low_order_linear_map = (
            proj_correlation_mat_eigvals_sqrt_inv *
            self.proj_correlation_mat_eigvecs.conj().H *
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H *
            self.cross_correlation_mat *
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H *
            self.proj_correlation_mat_eigvecs *
            proj_correlation_mat_eigvals_sqrt_inv)

        # Compute eigendecomposition of low-order linear map
        self.eigvals, self.R_low_order_eigvecs, self.L_low_order_eigvecs =\
            parallel.call_and_bcast(
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

            ``R_low_order_eigvecs``: Matrix whose columns are right
            eigenvectors of approximating low-order linear map.

            ``L_low_order_eigvecs``: Matrix whose columns are left eigenvectors
            of approximating low-order linear map.

            ``summed_correlation_mats_eigvals``: 1D array of eigenvalues of
            summed correlation matrices.

            ``summed_correlation_mats_eigvecs``: Matrix whose columns are
            eigenvectors of summed correlation matrices.

            ``proj_correlation_mat_eigvals``: 1D array of eigenvalues of
            projected correlation matrix.

            ``proj_correlation_mat_eigvecs``: Matrix whose columns are
            eigenvectors of projected correlation matrix.

        Note that the truncation level (corresponding to ``max_num_eigvals``)
        must be set to a value smaller than the rank of the dataset, otherwise
        total-least-squares DMD reduces to standard DMD.  This occurs naturally
        for over-constrained datasets, but must be enforced by the user for
        under-constrined datasets.
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
            self.adv_correlation_mat = self.expanded_correlation_mat[1:, 1:]
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
            self.adv_correlation_mat = \
                self.vec_space.compute_symmetric_inner_product_mat(
                self.adv_vec_handles)

        # Compute eigendecomposition of low-order linear map.
        self.compute_eigendecomp(
            atol=atol, rtol=rtol, max_num_eigvals=max_num_eigvals)

        return (
            self.eigvals,
            self.R_low_order_eigvecs,
            self.L_low_order_eigvecs,
            self.summed_correlation_mats_eigvals,
            self.summed_correlation_mats_eigvecs,
            self.proj_correlation_mat_eigvals,
            self.proj_correlation_mat_eigvecs)


    def _compute_build_coeffs_exact(self):
        """Compute build coefficients for exact DMD modes."""
        return (
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H *
            self.proj_correlation_mat_eigvecs *
            np.mat(np.diag(self.proj_correlation_mat_eigvals ** -0.5)) *
            self.R_low_order_eigvecs
            * np.mat(np.diag(self.eigvals ** -1.)))


    def _compute_build_coeffs_proj(self):
        """Compute build coefficients for projected DMD modes."""
        return (
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H *
            self.proj_correlation_mat_eigvecs *
            np.mat(np.diag(self.proj_correlation_mat_eigvals ** -0.5)) *
            self.R_low_order_eigvecs)


    def get_decomp(
        self, eigvals_src, R_low_order_eigvecs_src, L_low_order_eigvecs_src,
        summed_correlation_mats_eigvals_src,
        summed_correlation_mats_eigvecs_src, proj_correlation_mat_eigvals_src,
        proj_correlation_mat_eigvecs_src):
        """Gets the decomposition matrices from sources (memory or file).

        Args:
            ``eigvals_src``: Source from which to retrieve eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_src``: Source from which to retrieve right
            eigenvectors of approximating low-order linear DMD map.

            ``L_low_order_eigvecs_src``: Source from which to retrieve left
            eigenvectors of approximating low-order linear DMD map.

            ``summed_correlation_mats_eigvals_src``: Source from which to
            retrieve eigenvalues of summed correlation matrices.

            ``summed_correlation_mats_eigvecs_src``: Source from which to
            retrieve eigenvectors of summed correlation matrices.

            ``proj_correlation_mat_eigvals_src``: Source from which to
            retrieve eigenvalues of projected correlation matrix.

            ``proj_correlation_mat_eigvecs_src``: Source from which to
            retrieve eigenvectors of projected correlation matrix.
        """
        self.eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(self.get_mat, eigvals_src)))
        self.R_low_order_eigvecs = parallel.call_and_bcast(
            self.get_mat, R_low_order_eigvecs_src)
        self.L_low_order_eigvecs = parallel.call_and_bcast(
            self.get_mat, L_low_order_eigvecs_src)
        self.summed_correlation_mats_eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(
            self.get_mat, summed_correlation_mats_eigvals_src)))
        self.summed_correlation_mats_eigvecs = parallel.call_and_bcast(
            self.get_mat, summed_correlation_mats_eigvecs_src)
        self.proj_correlation_mat_eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(
            self.get_mat, proj_correlation_mat_eigvals_src)))
        self.proj_correlation_mat_eigvecs = parallel.call_and_bcast(
            self.get_mat, proj_correlation_mat_eigvecs_src)


    def put_decomp(
        self, eigvals_dest, R_low_order_eigvecs_dest, L_low_order_eigvecs_dest,
        summed_correlation_mats_eigvals_dest,
        summed_correlation_mats_eigvecs_dest,
        proj_correlation_mat_eigvals_dest, proj_correlation_mat_eigvecs_dest):
        """Puts the decomposition matrices in destinations (file or memory).

        Args:
            ``eigvals_dest``: Destination in which to put eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_dest``: Destination in which to put right
            eigenvectors of approximating low-order linear map.

            ``L_low_order_eigvecs_dest``: Destination in which to put left
            eigenvectors of approximating low-order linear map.

            ``summed_correlation_mats_eigvals_dest``: Destination in which to
            put eigenvalues of summed correlation matrices.

            ``summed_correlation_mats_eigvecs_dest``: Destination in which to
            put eigenvectors of summed correlation matrices.

            ``proj_correlation_mat_eigvals_dest``: Destination in which to put
            eigenvalues of projected correlation matrix.

            ``proj_correlation_mat_eigvecs_dest``: Destination in which to put
            eigenvectors of projected correlation matrix.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_eigvals(eigvals_dest)
        self.put_R_low_order_eigvecs(R_low_order_eigvecs_dest)
        self.put_L_low_order_eigvecs(L_low_order_eigvecs_dest)
        self.put_summed_correlation_mats_eigvals(
            summed_correlation_mats_eigvals_dest)
        self.put_summed_correlation_mats_eigvecs(
            summed_correlation_mats_eigvecs_dest)
        self.put_proj_correlation_mat_eigvals(
            proj_correlation_mat_eigvals_dest)
        self.put_proj_correlation_mat_eigvecs(
            proj_correlation_mat_eigvecs_dest)


    def put_correlation_mat_eigvals(self, dest):
        """This method is not available for total least squares DMD"""
        raise NotImplementedError(
            'This method is not available.  Use '
            'put_summed_correlation_mats_eigvals instead.')


    def put_correlation_mat_eigvecs(self, dest):
        """This method is not available for total least squares DMD"""
        raise NotImplementedError(
            'This method is not available.  Use '
            'put_summed_correlation_mats_eigvecs instead.')


    def put_summed_correlation_mats_eigvals(self, dest):
        """Puts eigenvalues of summed correlation matrices to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.summed_correlation_mats_eigvals, dest)
        parallel.barrier()


    def put_summed_correlation_mats_eigvecs(self, dest):
        """Puts eigenvectors of summed correlation matrices to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.summed_correlation_mats_eigvecs, dest)
        parallel.barrier()


    def put_proj_correlation_mat_eigvals(self, dest):
        """Puts eigenvalues of projected correlation matrix to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.proj_correlation_mat_eigvals, dest)
        parallel.barrier()


    def put_proj_correlation_mat_eigvecs(self, dest):
        """Puts eigenvectors of projected correlation matrix to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.proj_correlation_mat_eigvecs, dest)
        parallel.barrier()


    def put_adv_correlation_mat(self, dest):
        """Puts advanced correlation mat to ``dest``."""
        if parallel.is_rank_zero():
            self.put_mat(self.adv_correlation_mat, dest)
        parallel.barrier()


    def compute_spectrum(self):
        """Computes DMD spectral coefficients.  These coefficients come from a
        biorthogonal projection of the first (de-noised) vector object onto the
        exact DMD modes, which is analytically equivalent to doing a
        least-squares projection onto the projected DMD modes.

        Returns:
            ``spectral_coeffs``: 1D array of DMD spectral coefficients.
        """
        # TODO: maybe allow for user to choose which column to spectrum from?
        # ie first, last, or mean?
        self.spectral_coeffs = np.abs(np.array(
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(np.sqrt(self.proj_correlation_mat_eigvals))) *
            np.mat(self.proj_correlation_mat_eigvecs[0, :]).T)).squeeze()
        return self.spectral_coeffs


    # Note that a biorthogonal projection onto the exact DMD modes is the same
    # as a least squares projection onto the projected DMD modes, so there is
    # only one method for computing the projection coefficients.
    def compute_proj_coeffs(self):
        """Computes projection of (de-noised) vector objects onto DMD modes.
        Note that a biorthogonal projection onto exact DMD modes is
        analytically equivalent to a least-squares projection onto projected
        DMD modes.

        Returns:
            ``proj_coeffs``: Matrix of projection coefficients for
            (de-noised)vector objects, expressed as a linear combination of DMD
            modes.  Columns correspond to vector objects, rows correspond to
            DMD modes.

            ``adv_proj_coeffs``: Matrix of projection coefficients for
            (de-noised) vector objects advanced in time, expressed as a linear
            combination of DMD modes.  Columns correspond to vector objects,
            rows correspond to DMD modes.
        """
        self.proj_coeffs = (
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(np.sqrt(self.proj_correlation_mat_eigvals))) *
            self.proj_correlation_mat_eigvecs.T)
        self.adv_proj_coeffs = (
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(self.proj_correlation_mat_eigvals ** -0.5)) *
            self.proj_correlation_mat_eigvecs.T *
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H *
            self.cross_correlation_mat *
            self.summed_correlation_mats_eigvecs *
            self.summed_correlation_mats_eigvecs.H)

        return self.proj_coeffs, self.adv_proj_coeffs
