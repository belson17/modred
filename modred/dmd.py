from __future__ import print_function
from __future__ import absolute_import
from future.builtins import object
from collections import namedtuple

import numpy as np

from .vectorspace import VectorSpaceArrays, VectorSpaceHandles
from . import util
from . import parallel


def compute_DMD_arrays_snaps_method(
    vecs, adv_vecs=None, mode_indices=None, inner_product_weights=None,
    atol=1e-13, rtol=None, max_num_eigvals=None):
    """Computes DMD modes using data stored in arrays, using method of
    snapshots.

    Args:
        ``vecs``: Array whose columns are data vectors.

    Kwargs:
        ``adv_vecs``: Array whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which eigenvalues of correlation array are
        truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues of correlation array.  Smaller ones are truncated.

        ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
        computed.  This is enforced by truncating the basis onto which the
        approximating linear map is projected.  Computationally, this
        corresponds to truncating the eigendecomposition of the correlation
        array. If set to None, no truncation will be performed, and the
        maximum possible number of DMD eigenvalues will be computed.

    Returns:
        ``res``: Results of DMD computation, stored in a namedtuple with
        the following attributes:

        * ``exact_modes``: Array whose columns are exact DMD modes.

        * ``proj_modes``: Array whose columns are projected DMD modes.

        * ``adjoint_modes``: Array whose columns are adjoint DMD modes.

        * ``spectral_coeffs``: 1D array of DMD spectral coefficients, calculated
          as the magnitudes of the projection coefficients of first data vector.
          The projection is onto the span of the DMD modes using the
          (biorthogonal) adjoint DMD modes.  Note that this is the same as a
          least-squares projection onto the span of the DMD modes.

        * ``eigvals``: 1D array of eigenvalues of approximating low-order linear
          map (DMD eigenvalues).

        * ``R_low_order_eigvecs``: Array of right eigenvectors of approximating
          low-order linear map.

        * ``L_low_order_eigvecs``: Array of left eigenvectors of approximating
          low-order linear map.

        * ``correlation_array_eigvals``: 1D array of eigenvalues of
          correlation array.

        * ``correlation_array_eigvecs``: Array of eigenvectors of
          correlation array.

        * ``correlation_array``: Correlation array; elements are inner products
          of data vectors with each other.

        * ``cross_correlation_array``: Cross-correlation array; elements are
          inner products of data vectors with data vectors advanced in time.
          Going down rows, the data vector changes; going across columns the
          advanced data vector changes.

        Attributes can be accessed using calls like ``res.exact_modes``.  To
        see all available attributes, use ``print(res)``.

    This uses the method of snapshots, which is faster than the direct method
    (see :py:func:`compute_DMD_arrays_direct_method`) when ``vecs`` has more
    rows than columns, i.e., when there are more elements in a vector than
    there are vectors. However, it "squares" this array and its singular
    values, making it slightly less accurate than the direct method.
    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceArrays(weights=inner_product_weights)

    # Sequential dataset
    if adv_vecs is None:
        # Compute correlation array for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row and column
        # yet.
        expanded_correlation_array =\
            vec_space.compute_symm_inner_product_array(vecs)
        correlation_array = expanded_correlation_array[:-1, :-1]
        cross_correlation_array = expanded_correlation_array[:-1, 1:]
    # Non-sequential data
    else:
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))

        # Compute the correlation array from the unadvanced snapshots only.
        correlation_array = vec_space.compute_symm_inner_product_array(
            vecs)
        cross_correlation_array = vec_space.compute_inner_product_array(
            vecs, adv_vecs)

    correlation_array_eigvals, correlation_array_eigvecs = util.eigh(
        correlation_array, is_positive_definite=True, atol=atol, rtol=rtol)

    # Truncate if necessary
    if max_num_eigvals is not None and (
        max_num_eigvals < correlation_array_eigvals.size):
        correlation_array_eigvals = correlation_array_eigvals[:max_num_eigvals]
        correlation_array_eigvecs = correlation_array_eigvecs[
            :, :max_num_eigvals]

    # Compute low-order linear map for sequential or non-sequential case.
    correlation_array_eigvals_sqrt_inv = np.diag(
        correlation_array_eigvals ** -0.5)
    low_order_linear_map = correlation_array_eigvals_sqrt_inv.dot(
        correlation_array_eigvecs.conj().T.dot(
            cross_correlation_array.dot(
                correlation_array_eigvecs.dot(
                    correlation_array_eigvals_sqrt_inv))))

    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
        util.eig_biorthog(low_order_linear_map, scale_choice='left')
    build_coeffs_proj = correlation_array_eigvecs.dot(
        correlation_array_eigvals_sqrt_inv.dot(
            R_low_order_eigvecs))
    build_coeffs_exact = build_coeffs_proj.dot(np.diag(eigvals ** -1.))
    build_coeffs_adjoint = correlation_array_eigvecs.dot(
        correlation_array_eigvals_sqrt_inv.dot(
            L_low_order_eigvecs))
    spectral_coeffs = np.abs(L_low_order_eigvecs.conj().T.dot(
        np.diag(correlation_array_eigvals ** 0.5).dot(
            correlation_array_eigvecs[0, :].conj().T)).squeeze())

    # For sequential data, user must provide one more vec than columns of
    # build_coeffs.
    if vecs.shape[1] - build_coeffs_exact.shape[0] == 1:
        exact_modes = vec_space.lin_combine(
            vecs[:, 1:], build_coeffs_exact,
            coeff_array_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs[:, :-1], build_coeffs_proj,
            coeff_array_col_indices=mode_indices)
        adjoint_modes = vec_space.lin_combine(
            vecs[:, :-1], build_coeffs_adjoint,
            coeff_array_col_indices=mode_indices)
    # For non-sequential data, user must provide as many vecs as columns of
    # build_coeffs.
    elif vecs.shape[1] == build_coeffs_exact.shape[0]:
        exact_modes = vec_space.lin_combine(
            adv_vecs, build_coeffs_exact, coeff_array_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs, build_coeffs_proj, coeff_array_col_indices=mode_indices)
        adjoint_modes = vec_space.lin_combine(
            vecs, build_coeffs_adjoint, coeff_array_col_indices=mode_indices)
    else:
        raise ValueError(
            'Number of cols in vecs does not match number of rows in '
            'build_coeffs array.')

    # Return a namedtuple
    DMD_results = namedtuple(
        'DMD_results', [
            'exact_modes', 'proj_modes', 'adjoint_modes', 'spectral_coeffs',
            'eigvals', 'R_low_order_eigvecs', 'L_low_order_eigvecs',
            'correlation_array_eigvals', 'correlation_array_eigvecs',
            'correlation_array', 'cross_correlation_array'])
    return DMD_results(
        exact_modes=exact_modes, proj_modes=proj_modes,
        adjoint_modes=adjoint_modes, spectral_coeffs=spectral_coeffs,
        eigvals=eigvals, R_low_order_eigvecs=R_low_order_eigvecs,
        L_low_order_eigvecs=L_low_order_eigvecs,
        correlation_array_eigvals=correlation_array_eigvals,
        correlation_array_eigvecs=correlation_array_eigvecs,
        correlation_array=correlation_array,
        cross_correlation_array=cross_correlation_array)


def compute_DMD_arrays_direct_method(
    vecs, adv_vecs=None, mode_indices=None, inner_product_weights=None,
    atol=1e-13, rtol=None, max_num_eigvals=None):
    """Computes DMD modes using data stored in arrays, using direct method.

    Args:
        ``vecs``: Array whose columns are data vectors.

    Kwargs:
        ``adv_vecs``: Array whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which eigenvalues of correlation array are
        truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues of correlation array.  Smaller ones are truncated.

        ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
        computed.  This is enforced by truncating the basis onto which the
        approximating linear map is projected.  Computationally, this
        corresponds to truncating the eigendecomposition of the correlation
        array. If set to None, no truncation will be performed, and the
        maximum possible number of DMD eigenvalues will be computed.

    Returns:
        ``res``: Results of DMD computation, stored in a namedtuple with
        the following attributes:

        * ``exact_modes``: Array whose columns are exact DMD modes.

        * ``proj_modes``: Array whose columns are projected DMD modes.

        * ``adjoint_modes``: Array whose columns are adjoint DMD modes.

        * ``spectral_coeffs``: 1D array of DMD spectral coefficients, calculated
          as the magnitudes of the projection coefficients of first data vector.
          The projection is onto the span of the DMD modes using the
          (biorthogonal) adjoint DMD modes.  Note that this is the same as a
          least-squares projection onto the span of the DMD modes.

        * ``eigvals``: 1D array of eigenvalues of approximating low-order linear
          map (DMD eigenvalues).

        * ``R_low_order_eigvecs``: Array of right eigenvectors of approximating
          low-order linear map.

        * ``L_low_order_eigvecs``: Array of left eigenvectors of approximating
          low-order linear map.

        * ``correlation_array_eigvals``: 1D array of eigenvalues of
          correlation array.

        * ``correlation_array_eigvecs``: Array of eigenvectors of
          correlation array.

        Attributes can be accessed using calls like ``res.exact_modes``.  To
        see all available attributes, use ``print(res)``.

    This method does not square the array of vectors as in the method of
    snapshots (:py:func:`compute_DMD_arrays_snaps_method`). It's slightly
    more accurate, but slower when the number of elements in a vector is more
    than the number of vectors, i.e.,  when ``vecs`` has more rows than
    columns.
    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceArrays(weights=inner_product_weights)

    if inner_product_weights is None:
        vecs_weighted = vecs
        if adv_vecs is not None:
            adv_vecs_weighted = adv_vecs
    elif inner_product_weights.ndim == 1:
        sqrt_weights = np.diag(inner_product_weights ** 0.5)
        vecs_weighted = sqrt_weights.dot(vecs)
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights.dot(adv_vecs)
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print('Warning: Cholesky decomposition could be time consuming.')
        sqrt_weights = np.linalg.cholesky(inner_product_weights).conj().T
        vecs_weighted = sqrt_weights.dot(vecs)
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights.dot(adv_vecs)

    # Compute low-order linear map for sequential snapshot set.  This takes
    # advantage of the fact that for a sequential dataset, the unadvanced
    # and advanced vectors overlap.
    if adv_vecs is None:
        U, sing_vals, correlation_array_eigvecs = util.svd(
            vecs_weighted[:, :-1], atol=atol, rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < sing_vals.size):
            U = U[:, :max_num_eigvals]
            sing_vals = sing_vals[:max_num_eigvals]
            correlation_array_eigvecs = correlation_array_eigvecs[
                :, :max_num_eigvals]

        correlation_array_eigvals = sing_vals ** 2.
        correlation_array_eigvals_sqrt_inv = np.diag(sing_vals ** -1.)
        correlation_array = correlation_array_eigvecs.dot(
            np.diag(correlation_array_eigvals).dot(
                correlation_array_eigvecs.conj().T))
        last_col = U.conj().T.dot(vecs_weighted[:, -1])
        low_order_linear_map = np.concatenate((
            correlation_array_eigvals_sqrt_inv.dot(
                correlation_array_eigvecs.conj().T.dot(
                    correlation_array[:, 1:])),
            util.atleast_2d_col(last_col)), axis=1).dot(
                correlation_array_eigvecs.dot(
                    correlation_array_eigvals_sqrt_inv))
    else:
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        U, sing_vals, correlation_array_eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < sing_vals.size):
            U = U[:, :max_num_eigvals]
            sing_vals = sing_vals[:max_num_eigvals]
            correlation_array_eigvecs = correlation_array_eigvecs[
                :, :max_num_eigvals]

        correlation_array_eigvals = sing_vals ** 2
        correlation_array_eigvals_sqrt_inv = np.diag(sing_vals ** -1.)
        low_order_linear_map = U.conj().T.dot(
            adv_vecs_weighted.dot(
                correlation_array_eigvecs.dot(
                    correlation_array_eigvals_sqrt_inv)))

    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
        util.eig_biorthog(low_order_linear_map, scale_choice='left')
    build_coeffs_proj = correlation_array_eigvecs.dot(
        correlation_array_eigvals_sqrt_inv.dot(
            R_low_order_eigvecs))
    build_coeffs_exact = build_coeffs_proj.dot(np.diag(eigvals ** -1.))
    build_coeffs_adjoint = correlation_array_eigvecs.dot(
        correlation_array_eigvals_sqrt_inv.dot(
            L_low_order_eigvecs))
    spectral_coeffs = np.abs(L_low_order_eigvecs.conj().T.dot(
        np.diag(correlation_array_eigvals ** 0.5).dot(
            correlation_array_eigvecs[0, :].conj().T))).squeeze()

    # For sequential data, user must provide one more vec than columns of
    # build_coeffs.
    if vecs.shape[1] - build_coeffs_exact.shape[0] == 1:
        exact_modes = vec_space.lin_combine(
            vecs[:, 1:], build_coeffs_exact,
            coeff_array_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs[:, :-1], build_coeffs_proj,
            coeff_array_col_indices=mode_indices)
        adjoint_modes = vec_space.lin_combine(
            vecs[:, :-1], build_coeffs_adjoint,
            coeff_array_col_indices=mode_indices)
    # For sequential data, user must provide as many vecs as columns of
    # build_coeffs.
    elif vecs.shape[1] == build_coeffs_exact.shape[0]:
        exact_modes = vec_space.lin_combine(
            adv_vecs, build_coeffs_exact, coeff_array_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs, build_coeffs_proj, coeff_array_col_indices=mode_indices)
        adjoint_modes = vec_space.lin_combine(
            vecs, build_coeffs_adjoint, coeff_array_col_indices=mode_indices)
    else:
        raise ValueError(('Number of cols in vecs does not match '
            'number of rows in build_coeffs array.'))

    # Return a namedtuple
    DMD_results = namedtuple(
        'DMD_results', [
            'exact_modes', 'proj_modes', 'adjoint_modes', 'spectral_coeffs',
            'eigvals', 'R_low_order_eigvecs', 'L_low_order_eigvecs',
            'correlation_array_eigvals', 'correlation_array_eigvecs'])
    return DMD_results(
        exact_modes=exact_modes, proj_modes=proj_modes,
        adjoint_modes=adjoint_modes, spectral_coeffs=spectral_coeffs,
        eigvals=eigvals, R_low_order_eigvecs=R_low_order_eigvecs,
        L_low_order_eigvecs=L_low_order_eigvecs,
        correlation_array_eigvals=correlation_array_eigvals,
        correlation_array_eigvecs=correlation_array_eigvecs)


class DMDHandles(object):
    """Dynamic Mode Decomposition implemented for large datasets.

    Args:
        ``inner_product``: Function that computes inner product of two vector
        objects.

    Kwargs:
        ``put_array``: Function to put a array out of modred, e.g., write it to
        file.

        ``get_array``: Function to get a array into modred, e.g., load it from
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

    See also :func:`compute_DMD_arrays_snaps_method`,
    :func:`compute_DMD_arrays_direct_method`, and :mod:`vectors`.
    """
    def __init__(
        self, inner_product, get_array=util.load_array_text,
        put_array=util.save_array_text, max_vecs_per_node=None, verbosity=1):
        """Constructor"""
        self.get_array = get_array
        self.put_array = put_array
        self.verbosity = verbosity
        self.eigvals = None
        self.correlation_array = None
        self.cross_correlation_array = None
        self.correlation_array_eigvals = None
        self.correlation_array_eigvecs = None
        self.low_order_linear_map = None
        self.L_low_order_eigvecs = None
        self.R_low_order_eigvecs = None
        self.spectral_coeffs = None
        self.proj_coeffs = None
        self.adv_proj_coeffs = None
        self.vec_space = VectorSpaceHandles(
            inner_product=inner_product, max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.vec_handles = None
        self.adv_vec_handles = None


    def get_decomp(
        self, eigvals_src, R_low_order_eigvecs_src, L_low_order_eigvecs_src,
        correlation_array_eigvals_src, correlation_array_eigvecs_src):
        """Gets the decomposition arrays from sources (memory or file).

        Args:
            ``eigvals_src``: Source from which to retrieve eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_src``: Source from which to retrieve right
            eigenvectors of approximating low-order linear DMD map.

            ``L_low_order_eigvecs_src``: Source from which to retrieve left
            eigenvectors of approximating low-order linear DMD map.

            ``correlation_array_eigvals_src``: Source from which to retrieve
            eigenvalues of correlation array.

            ``correlation_array_eigvecs_src``: Source from which to retrieve
            eigenvectors of correlation array.
        """
        self.eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(self.get_array, eigvals_src)))
        self.R_low_order_eigvecs = parallel.call_and_bcast(
            self.get_array, R_low_order_eigvecs_src)
        self.L_low_order_eigvecs = parallel.call_and_bcast(
            self.get_array, L_low_order_eigvecs_src)
        self.correlation_array_eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(
            self.get_array, correlation_array_eigvals_src)))
        self.correlation_array_eigvecs = parallel.call_and_bcast(
            self.get_array, correlation_array_eigvecs_src)


    def get_correlation_array(self, src):
      """Gets the correlation array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve correlation array.
        """
      self.correlation_array = parallel.call_and_bcast(self.get_array, src)


    def get_cross_correlation_array(self, src):
        """Gets the cross-correlation array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve cross-correlation array.
        """
        self.cross_correlation_array = parallel.call_and_bcast(
            self.get_array, src)


    def get_spectral_coeffs(self, src):
        """Gets the spectral coefficients from source (memory or file).

        Args:
            ``src``: Source from which to retrieve spectral coefficients.
        """
        self.spectral_coeffs = parallel.call_and_bcast(self.get_array, src)


    def get_proj_coeffs(self, src, adv_src):
        """Gets the projection coefficients and advanced projection coefficients
        from sources (memory or file).

        Args:
            ``src``: Source from which to retrieve projection coefficients.

            ``adv_src``: Source from which to retrieve advanced projection
            coefficients.
        """
        self.proj_coeffs = parallel.call_and_bcast(self.get_array, src)
        self.adv_proj_coeffs = parallel.call_and_bcast(self.get_array, adv_src)


    def put_decomp(
        self, eigvals_dest, R_low_order_eigvecs_dest, L_low_order_eigvecs_dest,
        correlation_array_eigvals_dest, correlation_array_eigvecs_dest):
        """Puts the decomposition arrays in destinations (memory or file).

        Args:
            ``eigvals_dest``: Destination in which to put eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_dest``: Destination in which to put right
            eigenvectors of approximating low-order linear map.

            ``L_low_order_eigvecs_dest``: Destination in which to put left
            eigenvectors of approximating low-order linear map.

            ``correlation_array_eigvals_dest``: Destination in which to put
            eigenvalues of correlation array.

            ``correlation_array_eigvecs_dest``: Destination in which to put
            eigenvectors of correlation array.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_eigvals(eigvals_dest)
        self.put_R_low_order_eigvecs(R_low_order_eigvecs_dest)
        self.put_L_low_order_eigvecs(L_low_order_eigvecs_dest)
        self.put_correlation_array_eigvals(correlation_array_eigvals_dest)
        self.put_correlation_array_eigvecs(correlation_array_eigvecs_dest)


    def put_eigvals(self, dest):
        """Puts eigenvalues of approximating low-order-linear map (DMD
        eigenvalues) to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.eigvals, dest)
        parallel.barrier()


    def put_R_low_order_eigvecs(self, dest):
        """Puts right eigenvectors of approximating low-order linear map to
        ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.R_low_order_eigvecs, dest)
        parallel.barrier()


    def put_L_low_order_eigvecs(self, dest):
        """Puts left eigenvectors of approximating low-order linear map to
        ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.L_low_order_eigvecs, dest)
        parallel.barrier()


    def put_correlation_array_eigvals(self, dest):
        """Puts eigenvalues of correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.correlation_array_eigvals, dest)
        parallel.barrier()


    def put_correlation_array_eigvecs(self, dest):
        """Puts eigenvectors of correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.correlation_array_eigvecs, dest)
        parallel.barrier()


    def put_correlation_array(self, dest):
        """Puts correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.correlation_array, dest)
        parallel.barrier()


    def put_cross_correlation_array(self, dest):
        """Puts cross-correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.cross_correlation_array, dest)
        parallel.barrier()


    def put_spectral_coeffs(self, dest):
        """Puts DMD spectral coefficients to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.spectral_coeffs, dest)
        parallel.barrier()


    def put_proj_coeffs(self, dest, adv_dest):
        """Puts projection coefficients to ``dest``, advanced projection
        coefficients to ``adv_dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.proj_coeffs, dest)
            self.put_array(self.adv_proj_coeffs, adv_dest)
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
        """Computes eigendecompositions of correlation array and approximating
        low-order linear map.

        Kwargs:
            ``atol``: Level below which eigenvalues of correlation array are
            truncated.

            ``rtol``: Maximum relative difference between largest and smallest
            eigenvalues of correlation array.  Smaller ones are truncated.

            ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
            computed.  This is enforced by truncating the basis onto which the
            approximating linear map is projected.  Computationally, this
            corresponds to truncating the eigendecomposition of the correlation
            array. If set to None, no truncation will be performed, and the
            maximum possible number of DMD eigenvalues will be computed.

        Useful if you already have the correlation array and cross-correlation
        array and want to avoid recomputing them.

        Usage::

          DMD.correlation_array = pre_existing_correlation_array
          DMD.cross_correlation_array = pre_existing_cross_correlation_array
          DMD.compute_eigendecomp()
          DMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Another way to use this is to compute a DMD using a truncated basis for
        the projection of the approximating linear map.  Start by either
        computing a full decomposition or by loading pre-computed correlation
        and cross-correlation arrays.

        Usage::

          # Start with a full decomposition
          DMD_eigvals, correlation_array_eigvals = DMD.compute_decomp(
              vec_handles)[0, 3]

          # Do some processing to determine the truncation level, maybe based
          # on the DMD eigenvalues and correlation array eigenvalues
          desired_num_eigvals = my_post_processing_func(
              DMD_eigvals, correlation_array_eigvals)

          # Do a truncated decomposition
          DMD_eigvals_trunc = DMD.compute_eigendecomp(
            max_num_eigvals=desired_num_eigvals)

          # Compute modes for truncated decomposition
          DMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Since it doesn't overwrite the correlation and cross-correlation
        arrays, ``compute_eigendecomp`` can be called many times in a row to
        do computations for different truncation levels.  However, the results
        of the decomposition (e.g., ``self.eigvals``) do get overwritten, so
        you may want to call a ``put`` method to save those results somehow.
        """
        # Compute eigendecomposition of correlation array
        self.correlation_array_eigvals, self.correlation_array_eigvecs =\
            parallel.call_and_bcast(
            util.eigh, self.correlation_array, atol=atol, rtol=None,
            is_positive_definite=True)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < self.correlation_array_eigvals.size):
            self.correlation_array_eigvals = self.correlation_array_eigvals[
                :max_num_eigvals]
            self.correlation_array_eigvecs = self.correlation_array_eigvecs[
                :, :max_num_eigvals]

        # Compute low-order linear map
        correlation_array_eigvals_sqrt_inv = np.diag(
            self.correlation_array_eigvals ** -0.5)
        self.low_order_linear_map = correlation_array_eigvals_sqrt_inv.dot(
            self.correlation_array_eigvecs.conj().T.dot(
                self.cross_correlation_array.dot(
                    self.correlation_array_eigvecs.dot(
                        correlation_array_eigvals_sqrt_inv))))

        # Compute eigendecomposition of low-order linear map
        self.eigvals, self.R_low_order_eigvecs, self.L_low_order_eigvecs =\
            parallel.call_and_bcast(
                util.eig_biorthog, self.low_order_linear_map,
                **{'scale_choice':'left'})


    def compute_decomp(
        self, vec_handles, adv_vec_handles=None, atol=1e-13, rtol=None,
        max_num_eigvals=None):
        """Computes eigendecomposition of low-order linear map approximating
        relationship between vector objects, returning various arrays
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
            array. If set to None, no truncation will be performed, and the
            maximum possible number of DMD eigenvalues will be computed.

        Returns:
            ``eigvals``: 1D array of eigenvalues of low-order linear map, i.e.,
            the DMD eigenvalues.

            ``R_low_order_eigvecs``: Array whose columns are right
            eigenvectors of approximating low-order linear map.

            ``L_low_order_eigvecs``: Array whose columns are left eigenvectors
            of approximating low-order linear map.

            ``correlation_array_eigvals``: 1D array of eigenvalues of
            correlation array.

            ``correlation_array_eigvecs``: Array whose columns are eigenvectors
            of correlation array.
        """
        self.vec_handles = vec_handles
        if adv_vec_handles is not None:
            self.adv_vec_handles = adv_vec_handles
            if len(self.vec_handles) != len(self.adv_vec_handles):
                raise ValueError(('Number of vec_handles and adv_vec_handles'
                    ' is not equal.'))

        # For a sequential dataset, compute correlation array for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row/column yet.
        # Later we need all but the last element of the last column, so it is
        # faster to compute all of this now.  Only one extra element is
        # computed, since this is a symmetric inner product array.  Then
        # slice the expanded correlation array accordingly.
        if adv_vec_handles is None:
            self.expanded_correlation_array =\
                self.vec_space.compute_symm_inner_product_array(
                self.vec_handles)
            self.correlation_array = self.expanded_correlation_array[:-1, :-1]
            self.cross_correlation_array = self.expanded_correlation_array[
                :-1, 1:]
        # For non-sequential data, compute the correlation array from the
        # unadvanced snapshots only.  Compute the cross correlation array
        # involving the unadvanced and advanced snapshots separately.
        else:
            self.correlation_array =\
                self.vec_space.compute_symm_inner_product_array(
                    self.vec_handles)
            self.cross_correlation_array =\
                self.vec_space.compute_inner_product_array(
                self.vec_handles, self.adv_vec_handles)

        # Compute eigendecomposition of low-order linear map.
        self.compute_eigendecomp(
            atol=atol, rtol=rtol, max_num_eigvals=max_num_eigvals)

        # Return values
        return (
            self.eigvals,
            self.R_low_order_eigvecs,
            self.L_low_order_eigvecs,
            self.correlation_array_eigvals,
            self.correlation_array_eigvecs)


    def _compute_build_coeffs_exact(self):
        """Compute build coefficients for exact DMD modes."""
        return self.correlation_array_eigvecs.dot(
            np.diag(self.correlation_array_eigvals ** -0.5).dot(
                self.R_low_order_eigvecs.dot(
                    np.diag(self.eigvals ** -1.))))


    def _compute_build_coeffs_proj(self):
        """Compute build coefficients for projected DMD modes."""
        return self.correlation_array_eigvecs.dot(
            np.diag(self.correlation_array_eigvals ** -0.5).dot(
                self.R_low_order_eigvecs))


    def _compute_build_coeffs_adjoint(self):
        """Compute build coefficients for adjoint DMD modes."""
        return self.correlation_array_eigvecs.dot(
            np.diag(self.correlation_array_eigvals ** -0.5).dot(
                self.L_low_order_eigvecs))


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

        # Compute build coefficient array
        build_coeffs_exact = self._compute_build_coeffs_exact()

        # If the internal attribute is set, then compute the modes
        if self.adv_vec_handles is not None:
            self.vec_space.lin_combine(
                mode_handles, self.adv_vec_handles, build_coeffs_exact,
                coeff_array_col_indices=mode_indices)
        # If the internal attribute is not set, then check to see if
        # vec_handles is set.  If so, assume a sequential dataset, in which
        # case adv_vec_handles can be taken from a slice of vec_handles.
        elif self.vec_handles is not None:
            if len(self.vec_handles) - build_coeffs_exact.shape[0] == 1:
                self.vec_space.lin_combine(
                    mode_handles, self.vec_handles[1:], build_coeffs_exact,
                    coeff_array_col_indices=mode_indices)
            else:
                raise(
                    ValueError, (
                    'Number of vec_handles is not correct for a sequential '
                    'dataset.'))
        else:
            raise(
                ValueError,
                'Neither vec_handles nor adv_vec_handles is defined.')


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

        # Compute build coefficient array
        build_coeffs_proj = self._compute_build_coeffs_proj()

        # For sequential data, the user will provide a list vec_handles that
        # whose length is one larger than the number of rows of the
        # build_coeffs array.  This is to be expected, as vec_handles is
        # essentially partitioned into two sets of handles, each of length one
        # less than vec_handles.
        if len(self.vec_handles) - build_coeffs_proj.shape[0] == 1:
            self.vec_space.lin_combine(
                mode_handles, self.vec_handles[:-1], build_coeffs_proj,
                coeff_array_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list
        # vec_handles whose length is equal to the number of rows in the
        # build_coeffs array.
        elif len(self.vec_handles) == build_coeffs_proj.shape[0]:
            self.vec_space.lin_combine(
                mode_handles, self.vec_handles, build_coeffs_proj,
                coeff_array_col_indices=mode_indices)
        # Otherwise, raise an error, as the number of handles should fit one of
        # the two cases described above.
        else:
            raise ValueError((
                'Number of vec_handles does not match number of columns in '
                'build_coeffs_proj array.'))


    def compute_adjoint_modes(
        self, mode_indices, mode_handles, vec_handles=None):
        """Computes adjoint DMD modes and calls ``put`` on them using mode
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

        # Compute build coefficient array
        build_coeffs_adjoint = self._compute_build_coeffs_adjoint()

        # For sequential data, the user will provide a list vec_handles that
        # whose length is one larger than the number of rows of the
        # build_coeffs array.  This is to be expected, as vec_handles is
        # essentially partitioned into two sets of handles, each of length one
        # less than vec_handles.
        if len(self.vec_handles) - build_coeffs_adjoint.shape[0] == 1:
            self.vec_space.lin_combine(
                mode_handles, self.vec_handles[:-1], build_coeffs_adjoint,
                coeff_array_col_indices=mode_indices)
        # For a non-sequential dataset, the user will provide a list
        # vec_handles whose length is equal to the number of rows in the
        # build_coeffs array.
        elif len(self.vec_handles) == build_coeffs_adjoint.shape[0]:
            self.vec_space.lin_combine(
                mode_handles, self.vec_handles, build_coeffs_adjoint,
                coeff_array_col_indices=mode_indices)
        # Otherwise, raise an error, as the number of handles should fit one of
        # the two cases described above.
        else:
            raise ValueError((
                'Number of vec_handles does not match number of columns in '
                'build_coeffs_proj array.'))


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
        self.spectral_coeffs = np.abs(
            self.L_low_order_eigvecs.conj().T.dot(
                np.diag(np.sqrt(self.correlation_array_eigvals)).dot(
                    self.correlation_array_eigvecs[0, :]).conj().T)).squeeze()
        return self.spectral_coeffs


    # Note that a biorthogonal projection onto the exact DMD modes is the same
    # as a least squares projection onto the projected DMD modes, so there is
    # only one method for computing the projection coefficients.
    def compute_proj_coeffs(self):
        """Computes projection of vector objects onto DMD modes.  Note that a
        biorthogonal projection onto exact DMD modes is analytically equivalent
        to a least-squares projection onto projected DMD modes.

        Returns:
            ``proj_coeffs``: Array of projection coefficients for vector
            objects, expressed as a linear combination of DMD modes.  Columns
            correspond to vector objects, rows correspond to DMD modes.

            ``adv_proj_coeffs``: Array of projection coefficients for vector
            objects advanced in time, expressed as a linear combination of DMD
            modes.  Columns correspond to vector objects, rows correspond to
            DMD modes.
        """
        self.proj_coeffs = self.L_low_order_eigvecs.conj().T.dot(
            np.diag(np.sqrt(self.correlation_array_eigvals)).dot(
                self.correlation_array_eigvecs.conj().T))
        self.adv_proj_coeffs = self.L_low_order_eigvecs.conj().T.dot(
            np.diag(self.correlation_array_eigvals ** -0.5).dot(
                self.correlation_array_eigvecs.conj().T.dot(
                    self.cross_correlation_array)))
        return self.proj_coeffs, self.adv_proj_coeffs


def compute_TLSqrDMD_arrays_snaps_method(
    vecs, adv_vecs=None, mode_indices=None, inner_product_weights=None,
    atol=1e-13, rtol=None, max_num_eigvals=None):
    """Computes Total Least Squares DMD modes using data stored in arrays,
    using method of snapshots.

    Args:
        ``vecs``: Array whose columns are data vectors.

    Kwargs:
        ``adv_vecs``: Array whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which eigenvalues of correlation array are
        truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues of correlation array.  Smaller ones are truncated.

        ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
        computed.  This is enforced by truncating the basis onto which the
        approximating linear map is projected.  Computationally, this
        corresponds to truncating the eigendecomposition of the correlation
        array. If set to None, no truncation will be performed, and the
        maximum possible number of DMD eigenvalues will be computed.

    Returns:
        ``res``: Results of DMD computation, stored in a namedtuple with
        the following attributes:

        * ``exact_modes``: Array whose columns are exact DMD modes.

        * ``proj_modes``: Array whose columns are projected DMD modes.

        * ``adjoint_modes``: Array whose columns are adjoint DMD modes.

        * ``spectral_coeffs``: 1D array of DMD spectral coefficients, calculated
          as the magnitudes of the projection coefficients of first (de-noised)
          data vector.  The projection is onto the span of the DMD modes using
          the (biorthogonal) adjoint DMD modes.  Note that this is the same as a
          least-squares projection onto the span of the DMD modes.

        * ``eigvals``: 1D array of eigenvalues of approximating low-order linear
          map (DMD eigenvalues).

        * ``R_low_order_eigvecs``: Array of right eigenvectors of approximating
          low-order linear map.

        * ``L_low_order_eigvecs``: Array of left eigenvectors of approximating
          low-order linear map.

        * ``sum_correlation_array_eigvals``: 1D array of eigenvalues of
          sum correlation array.

        * ``sum_correlation_array_eigvecs``: Array whose columns are
          eigenvectors of sum correlation array.

        * ``proj_correlation_array_eigvals``: 1D array of eigenvalues of
          projected correlation array.

        * ``proj_correlation_array_eigvecs``: Array whose columns are
          eigenvectors of projected correlation array.

        * ``correlation_array``: Correlation array; elements are inner products
          of data vectors with each other.

        * ``adv_correlation_array``: Advanced correlation array; elements are
          inner products of advanced data vectors with each other.

        * ``cross_correlation_array``: Cross-correlation array; elements are
          inner products of data vectors with data vectors advanced in
          time. Going down rows, the data vector changes; going across columns
          the advanced data vector changes.

        Attributes can be accessed using calls like ``res.exact_modes``.  To
        see all available attributes, use ``print(res)``.

    This uses the method of snapshots, which is faster than the direct method
    (see :py:func:`compute_TLSqrDMD_arrays_direct_method`) when ``vecs`` has
    more rows than columns, i.e., when there are more elements in a vector than
    there are vectors. However, it "squares" this array and its singular
    values, making it slightly less accurate than the direct method.

    Note that max_num_eigvals must be set to a value smaller than the rank of
    the dataset.  In other words, if the projection basis for
    total-least-squares DMD is not truncated, then the algorithm reduces to
    standard DMD.  For over-constrained datasets (number of columns in data
    array is larger than the number of rows), this occurs naturally.  For
    under-constrained datasets, (number of vector objects is smaller than size
    of vector objects), this must be done explicitly by the user.  At this
    time, there is no standard method for choosing a truncation level.  One
    approach is to look at the roll-off of the correlation array eigenvalues,
    which contains information about the "energy" content of each projection
    basis vectors.

    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceArrays(weights=inner_product_weights)

    # Sequential dataset
    if adv_vecs is None:
        # Compute correlation array for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row and column
        # yet.
        expanded_correlation_array = vec_space.compute_symm_inner_product_array(
            vecs)
        correlation_array = expanded_correlation_array[:-1, :-1]
        cross_correlation_array = expanded_correlation_array[:-1, 1:]
        adv_correlation_array = expanded_correlation_array[1:, 1:]
    # Non-sequential data
    else:
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        # Compute the correlation array from the unadvanced snapshots only.
        correlation_array = vec_space.compute_symm_inner_product_array(vecs)
        cross_correlation_array = vec_space.compute_inner_product_array(
            vecs, adv_vecs)
        adv_correlation_array = vec_space.compute_symm_inner_product_array(
            adv_vecs)

    sum_correlation_array_eigvals, sum_correlation_array_eigvecs =\
        util.eigh(
            correlation_array + adv_correlation_array,
            is_positive_definite=True,
            atol=atol, rtol=rtol)

    # Truncate if necessary
    if max_num_eigvals is not None and (
        max_num_eigvals < sum_correlation_array_eigvals.size):
        sum_correlation_array_eigvals = sum_correlation_array_eigvals[
            :max_num_eigvals]
        sum_correlation_array_eigvecs = sum_correlation_array_eigvecs[
            :, :max_num_eigvals]

    # Compute eigendecomposition of projected correlation array
    proj_correlation_array = sum_correlation_array_eigvecs.dot(
        sum_correlation_array_eigvecs.conj().T.dot(
            correlation_array.dot(
                sum_correlation_array_eigvecs.dot(
                    sum_correlation_array_eigvecs.conj().T))))
    proj_correlation_array_eigvals, proj_correlation_array_eigvecs = util.eigh(
        proj_correlation_array, atol=atol, rtol=None, is_positive_definite=True)

    # Truncate if necessary
    if max_num_eigvals is not None and (
        max_num_eigvals < proj_correlation_array_eigvals.size):
        proj_correlation_array_eigvals = proj_correlation_array_eigvals[
            :max_num_eigvals]
        proj_correlation_array_eigvecs = proj_correlation_array_eigvecs[
            :, :max_num_eigvals]

    # Compute low-order linear map
    proj_correlation_array_eigvals_sqrt_inv = np.diag(
        proj_correlation_array_eigvals ** -0.5)
    low_order_linear_map = proj_correlation_array_eigvals_sqrt_inv.dot(
        proj_correlation_array_eigvecs.conj().T.dot(
            sum_correlation_array_eigvecs.dot(
                sum_correlation_array_eigvecs.conj().T.dot(
                    cross_correlation_array.dot(
                        sum_correlation_array_eigvecs.dot(
                            sum_correlation_array_eigvecs.conj().T.dot(
                                proj_correlation_array_eigvecs.dot(
                                    proj_correlation_array_eigvals_sqrt_inv
                                ))))))))

    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs = util.eig_biorthog(
        low_order_linear_map, scale_choice='left')
    build_coeffs_proj = sum_correlation_array_eigvecs.dot(
        sum_correlation_array_eigvecs.conj().T.dot(
            proj_correlation_array_eigvecs.dot(
                np.diag(proj_correlation_array_eigvals ** -0.5).dot(
                    R_low_order_eigvecs))))
    build_coeffs_exact = build_coeffs_proj.dot(np.diag(eigvals ** -1.))
    build_coeffs_adjoint = sum_correlation_array_eigvecs.dot(
        sum_correlation_array_eigvecs.conj().T.dot(
            proj_correlation_array_eigvecs.dot(
                np.diag(proj_correlation_array_eigvals ** -0.5).dot(
                    L_low_order_eigvecs))))
    spectral_coeffs = np.abs(
        L_low_order_eigvecs.conj().T.dot(
            np.diag(np.sqrt(proj_correlation_array_eigvals)).dot(
                proj_correlation_array_eigvecs[0, :].T))).squeeze()

    # For sequential data, user must provide one more vec than columns of
    # build_coeffs.
    if vecs.shape[1] - build_coeffs_exact.shape[0] == 1:
        exact_modes = vec_space.lin_combine(
            vecs[:, 1:], build_coeffs_exact,
            coeff_array_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs[:, :-1], build_coeffs_proj,
            coeff_array_col_indices=mode_indices)
        adjoint_modes = vec_space.lin_combine(
            vecs[:, :-1], build_coeffs_adjoint,
            coeff_array_col_indices=mode_indices)
    # For non-sequential data, user must provide as many vecs as columns of
    # build_coeffs.
    elif vecs.shape[1] == build_coeffs_exact.shape[0]:
        exact_modes = vec_space.lin_combine(
            adv_vecs, build_coeffs_exact, coeff_array_col_indices=mode_indices)
        proj_modes = vec_space.lin_combine(
            vecs, build_coeffs_proj, coeff_array_col_indices=mode_indices)
        adjoint_modes = vec_space.lin_combine(
            vecs, build_coeffs_adjoint, coeff_array_col_indices=mode_indices)
    else:
        raise ValueError(('Number of cols in vecs does not match '
            'number of rows in build_coeffs array.'))

    # Return a namedtuple
    TLSqrDMD_results = namedtuple(
        'TLSqrDMD_results', [
            'exact_modes', 'proj_modes', 'adjoint_modes', 'spectral_coeffs',
            'eigvals', 'R_low_order_eigvecs', 'L_low_order_eigvecs',
            'sum_correlation_array_eigvals', 'sum_correlation_array_eigvecs',
            'proj_correlation_array_eigvals', 'proj_correlation_array_eigvecs',
            'correlation_array', 'adv_correlation_array',
            'cross_correlation_array'])
    return TLSqrDMD_results(
        exact_modes=exact_modes, proj_modes=proj_modes,
        adjoint_modes=adjoint_modes, spectral_coeffs=spectral_coeffs,
        eigvals=eigvals, R_low_order_eigvecs=R_low_order_eigvecs,
        L_low_order_eigvecs=L_low_order_eigvecs,
        sum_correlation_array_eigvals=sum_correlation_array_eigvals,
        sum_correlation_array_eigvecs=sum_correlation_array_eigvecs,
        proj_correlation_array_eigvals=proj_correlation_array_eigvals,
        proj_correlation_array_eigvecs=proj_correlation_array_eigvecs,
        correlation_array=correlation_array,
        adv_correlation_array=adv_correlation_array,
        cross_correlation_array=cross_correlation_array)


def compute_TLSqrDMD_arrays_direct_method(
    vecs, adv_vecs=None, mode_indices=None, inner_product_weights=None,
    atol=1e-13, rtol=None, max_num_eigvals=None):
    """Computes Total Least Squares DMD modes using data stored in arrays,
    using direct method.

    Args:
        ``vecs``: Array whose columns are data vectors.

    Kwargs:
        ``adv_vecs``: Array whose columns are data vectors advanced in time.
        If not provided, then it is assumed that the vectors describe a
        sequential time-series. Thus ``vecs`` becomes ``vecs[:, :-1]`` and
        ``adv_vecs`` becomes ``vecs[:, 1:]``.

        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which eigenvalues of correlation array are
        truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues of correlation array.  Smaller ones are truncated.

        ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
        computed.  This is enforced by truncating the basis onto which the
        approximating linear map is projected.  Computationally, this
        corresponds to truncating the eigendecomposition of the correlation
        array. If set to None, no truncation will be performed, and the
        maximum possible number of DMD eigenvalues will be computed.

    Returns:
        ``res``: Results of DMD computation, stored in a namedtuple with
        the following attributes:

        * ``exact_modes``: Array whose columns are exact DMD modes.

        * ``proj_modes``: Array whose columns are projected DMD modes.

        * ``adjoint_modes``: Array whose columns are adjoint DMD modes.

        * ``spectral_coeffs``: 1D array of DMD spectral coefficients, calculated
          as the magnitudes of the projection coefficients of first (de-noised)
          data vector.  The projection is onto the span of the DMD modes using
          the (biorthogonal) adjoint DMD modes.  Note that this is the same as a
          least-squares projection onto the span of the DMD modes.

        * ``eigvals``: 1D array of eigenvalues of approximating low-order linear
          map (DMD eigenvalues).

        * ``R_low_order_eigvecs``: Array of right eigenvectors of approximating
          low-order linear map.

        * ``L_low_order_eigvecs``: Array of left eigenvectors of approximating
          low-order linear map.

        * ``sum_correlation_array_eigvals``: 1D array of eigenvalues of
          sum correlation array.

        * ``sum_correlation_array_eigvecs``: Array whose columns are
          eigenvectors of sum correlation array.

        * ``proj_correlation_array_eigvals``: 1D array of eigenvalues of
          projected correlation array.

        * ``proj_correlation_array_eigvecs``: Array whose columns are
          eigenvectors of projected correlation array.

        Attributes can be accessed using calls like ``res.exact_modes``.  To
        see all available attributes, use ``print(res)``.

    This method does not square the array of vectors as in the method of
    snapshots (:py:func:`compute_DMD_arrays_snaps_method`). It's slightly
    more accurate, but slower when the number of elements in a vector is more
    than the number of vectors, i.e.,  when ``vecs`` has more rows than
    columns.

    Note that max_num_eigvals must be set to a value smaller than the rank of
    the dataset.  In other words, if the projection basis for
    total-least-squares DMD is not truncated, then the algorithm reduces to
    standard DMD.  For over-constrained datasets (number of columns in data
    array is larger than the number of rows), this occurs naturally.  For
    under-constrained datasets, (number of vector objects is smaller than size
    of vector objects), this must be done explicitly by the user.  At this
    time, there is no standard method for choosing a truncation level.  One
    approach is to look at the roll-off of the correlation array eigenvalues,
    which contains information about the "energy" content of each projection
    basis vectors.

    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceArrays(weights=inner_product_weights)

    if inner_product_weights is None:
        vecs_weighted = vecs
        if adv_vecs is not None:
            adv_vecs_weighted = adv_vecs
    elif inner_product_weights.ndim == 1:
        sqrt_weights = np.diag(inner_product_weights ** 0.5)
        vecs_weighted = sqrt_weights.dot(vecs)
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights.dot(adv_vecs)
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print('Warning: Cholesky decomposition could be time consuming.')
        sqrt_weights = np.linalg.cholesky(inner_product_weights).conj().T
        vecs_weighted = sqrt_weights.dot(vecs)
        if adv_vecs is not None:
            adv_vecs_weighted = sqrt_weights.dot(adv_vecs)

    # Compute projections of original data (to de-noise).  First consider the
    # sequential data case.
    if adv_vecs is None:
        stacked_U, stacked_sing_vals, sum_correlation_array_eigvecs =util.svd(
            np.vstack((vecs_weighted[:, :-1], vecs_weighted[:, 1:])),
            atol=atol, rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < stacked_sing_vals.size):
            stacked_U = stacked_U[:, :max_num_eigvals]
            stacked_sing_vals = stacked_sing_vals[:max_num_eigvals]
            sum_correlation_array_eigvecs = sum_correlation_array_eigvecs[
                :, :max_num_eigvals]

        # Project original data to de-noise
        vecs_proj = vecs[:, :-1].dot(
            sum_correlation_array_eigvecs.dot(
                sum_correlation_array_eigvecs.conj().T))
        adv_vecs_proj = vecs[:, 1:].dot(
            sum_correlation_array_eigvecs.dot(
                sum_correlation_array_eigvecs.conj().T))
    # Non-sequential data case
    else:
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        stacked_U, stacked_sing_vals, sum_correlation_array_eigvecs = util.svd(
            np.vstack((vecs_weighted, adv_vecs_weighted)),
            atol=atol, rtol=rtol)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < stacked_sing_vals.size):
            stacked_U = stacked_U[:, :max_num_eigvals]
            stacked_sing_vals = stacked_sing_vals[:max_num_eigvals]
            sum_correlation_array_eigvecs = sum_correlation_array_eigvecs[
                :, :max_num_eigvals]

        # Project original data to de-noise
        vecs_proj = vecs.dot(
            sum_correlation_array_eigvecs.dot(
                sum_correlation_array_eigvecs.conj().T))
        adv_vecs_proj = adv_vecs.dot(
            sum_correlation_array_eigvecs.dot(
                sum_correlation_array_eigvecs.conj().T))

    # Now proceed with DMD of projected data
    sum_correlation_array_eigvals = stacked_sing_vals ** 2
    DMD_res = compute_DMD_arrays_direct_method(
        vecs_proj, adv_vecs=adv_vecs_proj, mode_indices=mode_indices,
        inner_product_weights=inner_product_weights, atol=atol, rtol=rtol,
        max_num_eigvals=max_num_eigvals)

    # Return a namedtuple
    TLSqrDMD_results = namedtuple(
        'TLSqrDMD_results', [
            'exact_modes', 'proj_modes', 'adjoint_modes', 'spectral_coeffs',
            'eigvals', 'R_low_order_eigvecs', 'L_low_order_eigvecs',
            'sum_correlation_array_eigvals', 'sum_correlation_array_eigvecs',
            'proj_correlation_array_eigvals', 'proj_correlation_array_eigvecs'])
    return TLSqrDMD_results(
        exact_modes=DMD_res.exact_modes, proj_modes=DMD_res.proj_modes,
        adjoint_modes=DMD_res.adjoint_modes,
        spectral_coeffs=DMD_res.spectral_coeffs,
        eigvals=DMD_res.eigvals,
        R_low_order_eigvecs=DMD_res.R_low_order_eigvecs,
        L_low_order_eigvecs=DMD_res.L_low_order_eigvecs,
        sum_correlation_array_eigvals=sum_correlation_array_eigvals,
        sum_correlation_array_eigvecs=sum_correlation_array_eigvecs,
        proj_correlation_array_eigvals=DMD_res.correlation_array_eigvals,
        proj_correlation_array_eigvecs=DMD_res.correlation_array_eigvecs)


class TLSqrDMDHandles(DMDHandles):
    """Total Least Squares Dynamic Mode Decomposition implemented for large
    datasets.

    Args:
        ``inner_product``: Function that computes inner product of two vector
        objects.

    Kwargs:
        ``put_array``: Function to put a array out of modred, e.g., write it to
        file.

      	``get_array``: Function to get a array into modred, e.g., load it from
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
    approach is to look at the roll-off of the correlation array eigenvalues,
    which contains information about the "energy" content of each projection
    basis vectors.

    Also, note that :class:`TLSqrDMDHandles` inherits from
    :class:`DMDHandles`, so
    certain methods are available, even though they are not
    implemented/documented here (namely several ``put`` functions).

    See also :func:`compute_TLSqrDMD_arrays_snaps_method`,
    :func:`compute_TLSqrDMD_arrays_direct_method`, and :mod:`vectors`.
    """
    def __init__(
        self, inner_product, get_array=util.load_array_text,
        put_array=util.save_array_text, max_vecs_per_node=None, verbosity=1):
        """Constructor"""
        self.get_array = get_array
        self.put_array = put_array
        self.verbosity = verbosity
        self.eigvals = None
        self.correlation_array = None
        self.cross_correlation_array = None
        self.adv_correlation_array = None
        self.sum_correlation_array = None
        self.sum_correlation_array_eigvals = None
        self.sum_correlation_array_eigvecs = None
        self.proj_correlation_array = None
        self.proj_correlation_array_eigvals = None
        self.proj_correlation_array_eigvecs = None
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
        """Computes eigendecompositions of correlation array and approximating
        low-order linear map.

        Kwargs:
            ``atol``: Level below which eigenvalues of correlation array are
            truncated.

            ``rtol``: Maximum relative difference between largest and smallest
            eigenvalues of correlation array.  Smaller ones are truncated.

            ``max_num_eigvals``: Maximum number of DMD eigenvalues that will be
            computed.  This is enforced by truncating the basis onto which the
            approximating linear map is projected.  Computationally, this
            corresponds to truncating the eigendecomposition of the correlation
            array. If set to None, no truncation will be performed, and the
            maximum possible number of DMD eigenvalues will be computed.

        Useful if you already have the correlation array and cross-correlation
        array and want to avoid recomputing them.

        Usage::

          TLSqrDMD.correlation_array = pre_existing_correlation_array
          TLSqrDMD.adv_correlation_array = pre_existing_adv_correlation_array
          TLSqrDMD.cross_correlation_array =
              pre_existing_cross_correlation_array
          TLSqrDMD.compute_eigendecomp()
          TLSqrDMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Another way to use this is to compute TLSqrDMD using different basis
        truncation levels for the projection of the approximating linear map.
        Start by either computing a full decomposition or by loading
        pre-computed correlation and cross-correlation arrays.

        Usage::

          # Start with a full decomposition
          DMD_eigvals, correlation_array_eigvals = TLSqrDMD.compute_decomp(
              vec_handles)[0, 3]

          # Do some processing to determine the truncation level, maybe based
          # on the DMD eigenvalues and correlation array eigenvalues
          desired_num_eigvals = my_post_processing_func(
              DMD_eigvals, correlation_array_eigvals)

          # Do a truncated decomposition
          DMD_eigvals_trunc = TLSqrDMD.compute_eigendecomp(
            max_num_eigvals=desired_num_eigvals)

          # Compute modes for truncated decomposition
          TLSqrDMD.compute_exact_modes(
              mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)

        Since it doesn't overwrite the correlation and cross-correlation
        arrays, ``compute_eigendecomp`` can be called many times in a row to
        do computations for different truncation levels.  However, the results
        of the decomposition (e.g., ``self.eigvals``) do get overwritten, so
        you may want to call a ``put`` method to save those results somehow.

        Note that the truncation level (corresponding to ``max_num_eigvals``)
        must be set to a value smaller than the rank of the dataset, otherwise
        total-least-squares DMD reduces to standard DMD.  This occurs naturally
        for over-constrained datasets, but must be enforced by the user for
        under-constrined datasets.
        """
        # Compute eigendecomposition of stacked correlation array
        self.sum_correlation_array = (
            self.correlation_array + self.adv_correlation_array)
        (self.sum_correlation_array_eigvals,
        self.sum_correlation_array_eigvecs) = parallel.call_and_bcast(
            util.eigh, self.sum_correlation_array,
            atol=atol, rtol=None, is_positive_definite=True)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < self.sum_correlation_array_eigvals.size):
            self.sum_correlation_array_eigvals =\
                self.sum_correlation_array_eigvals[:max_num_eigvals]
            self.sum_correlation_array_eigvecs =\
                self.sum_correlation_array_eigvecs[:, :max_num_eigvals]

        # Compute eigendecomposition of projected correlation array
        self.proj_correlation_array = self.sum_correlation_array_eigvecs.dot(
            self.sum_correlation_array_eigvecs.conj().T.dot(
                self.correlation_array.dot(
                    self.sum_correlation_array_eigvecs.dot(
                        self.sum_correlation_array_eigvecs.conj().T))))
        (self.proj_correlation_array_eigvals,
        self.proj_correlation_array_eigvecs) = parallel.call_and_bcast(
            util.eigh, self.proj_correlation_array,
            atol=atol, rtol=None, is_positive_definite=True)

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < self.proj_correlation_array_eigvals.size):
            self.proj_correlation_array_eigvals =\
                self.proj_correlation_array_eigvals[:max_num_eigvals]
            self.proj_correlation_array_eigvecs =\
                self.proj_correlation_array_eigvecs[:, :max_num_eigvals]

        # Compute low-order linear map
        proj_correlation_array_eigvals_sqrt_inv = np.diag(
            self.proj_correlation_array_eigvals ** -0.5)
        self.low_order_linear_map = proj_correlation_array_eigvals_sqrt_inv.dot(
            self.proj_correlation_array_eigvecs.conj().T.dot(
                self.sum_correlation_array_eigvecs.dot(
                    self.sum_correlation_array_eigvecs.conj().T.dot(
                        self.cross_correlation_array.dot(
                            self.sum_correlation_array_eigvecs.dot(
                                self.sum_correlation_array_eigvecs.conj().T.dot(
                                    self.proj_correlation_array_eigvecs.dot(
                                        proj_correlation_array_eigvals_sqrt_inv
                                    ))))))))

        # Compute eigendecomposition of low-order linear map
        self.eigvals, self.R_low_order_eigvecs, self.L_low_order_eigvecs =\
            parallel.call_and_bcast(
            util.eig_biorthog, self.low_order_linear_map,
            **{'scale_choice':'left'})


    def compute_decomp(
        self, vec_handles, adv_vec_handles=None, atol=1e-13, rtol=None,
        max_num_eigvals=None):
        """Computes eigendecomposition of low-order linear map approximating
        relationship between vector objects, returning various arrays
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
            array. If set to None, no truncation will be performed, and the
            maximum possible number of DMD eigenvalues will be computed.

        Returns:
            ``eigvals``: 1D array of eigenvalues of low-order linear map, i.e.,
            the DMD eigenvalues.

            ``R_low_order_eigvecs``: Array whose columns are right
            eigenvectors of approximating low-order linear map.

            ``L_low_order_eigvecs``: Array whose columns are left eigenvectors
            of approximating low-order linear map.

            ``sum_correlation_array_eigvals``: 1D array of eigenvalues of
            sum correlation array.

            ``sum_correlation_array_eigvecs``: Array whose columns are
            eigenvectors of sum correlation array.

            ``proj_correlation_array_eigvals``: 1D array of eigenvalues of
            projected correlation array.

            ``proj_correlation_array_eigvecs``: Array whose columns are
            eigenvectors of projected correlation array.

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

        # For a sequential dataset, compute correlation array for all vectors.
        # This is more efficient because only one call is made to the inner
        # product routine, even though we don't need the last row/column yet.
        # Later we need all but the last element of the last column, so it is
        # faster to compute all of this now.  Only one extra element is
        # computed, since this is a symmetric inner product array.  Then
        # slice the expanded correlation array accordingly.
        if adv_vec_handles is None:
            self.expanded_correlation_array =\
                self.vec_space.compute_symm_inner_product_array(
                    self.vec_handles)
            self.correlation_array = self.expanded_correlation_array[:-1, :-1]
            self.cross_correlation_array = self.expanded_correlation_array[
                :-1, 1:]
            self.adv_correlation_array = self.expanded_correlation_array[1:, 1:]
        # For non-sequential data, compute the correlation array from the
        # unadvanced snapshots only.  Compute the cross correlation array
        # involving the unadvanced and advanced snapshots separately.
        else:
            self.correlation_array =\
                self.vec_space.compute_symm_inner_product_array(
                    self.vec_handles)
            self.cross_correlation_array =\
                self.vec_space.compute_inner_product_array(
                self.vec_handles, self.adv_vec_handles)
            self.adv_correlation_array =\
                self.vec_space.compute_symm_inner_product_array(
                    self.adv_vec_handles)

        # Compute eigendecomposition of low-order linear map.
        self.compute_eigendecomp(
            atol=atol, rtol=rtol, max_num_eigvals=max_num_eigvals)

        # Return values
        return (
            self.eigvals,
            self.R_low_order_eigvecs,
            self.L_low_order_eigvecs,
            self.sum_correlation_array_eigvals,
            self.sum_correlation_array_eigvecs,
            self.proj_correlation_array_eigvals,
            self.proj_correlation_array_eigvecs)


    def _compute_build_coeffs_exact(self):
        """Compute build coefficients for exact DMD modes."""
        return self.sum_correlation_array_eigvecs.dot(
            self.sum_correlation_array_eigvecs.conj().T.dot(
                self.proj_correlation_array_eigvecs.dot(
                    np.diag(self.proj_correlation_array_eigvals ** -0.5)).dot(
                        self.R_low_order_eigvecs.dot(
                            np.diag(self.eigvals ** -1.)))))


    def _compute_build_coeffs_proj(self):
        """Compute build coefficients for projected DMD modes."""
        return self.sum_correlation_array_eigvecs.dot(
            self.sum_correlation_array_eigvecs.conj().T.dot(
                self.proj_correlation_array_eigvecs.dot(
                    np.diag(self.proj_correlation_array_eigvals ** -0.5).dot(
                        self.R_low_order_eigvecs))))


    def _compute_build_coeffs_adjoint(self):
        """Compute build coefficients for adjoint DMD modes."""
        return self.sum_correlation_array_eigvecs.dot(
            self.sum_correlation_array_eigvecs.conj().T.dot(
                self.proj_correlation_array_eigvecs.dot(
                    np.diag(self.proj_correlation_array_eigvals ** -0.5).dot(
                        self.L_low_order_eigvecs))))


    def get_decomp(
        self, eigvals_src, R_low_order_eigvecs_src, L_low_order_eigvecs_src,
        sum_correlation_array_eigvals_src,
        sum_correlation_array_eigvecs_src, proj_correlation_array_eigvals_src,
        proj_correlation_array_eigvecs_src):
        """Gets the decomposition arrays from sources (memory or file).

        Args:
            ``eigvals_src``: Source from which to retrieve eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_src``: Source from which to retrieve right
            eigenvectors of approximating low-order linear DMD map.

            ``L_low_order_eigvecs_src``: Source from which to retrieve left
            eigenvectors of approximating low-order linear DMD map.

            ``sum_correlation_array_eigvals_src``: Source from which to
            retrieve eigenvalues of sum correlation array.

            ``sum_correlation_array_eigvecs_src``: Source from which to
            retrieve eigenvectors of sum correlation array.

            ``proj_correlation_array_eigvals_src``: Source from which to
            retrieve eigenvalues of projected correlation array.

            ``proj_correlation_array_eigvecs_src``: Source from which to
            retrieve eigenvectors of projected correlation array.
        """
        self.eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(self.get_array, eigvals_src)))
        self.R_low_order_eigvecs = parallel.call_and_bcast(
            self.get_array, R_low_order_eigvecs_src)
        self.L_low_order_eigvecs = parallel.call_and_bcast(
            self.get_array, L_low_order_eigvecs_src)
        self.sum_correlation_array_eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(
            self.get_array, sum_correlation_array_eigvals_src)))
        self.sum_correlation_array_eigvecs = parallel.call_and_bcast(
            self.get_array, sum_correlation_array_eigvecs_src)
        self.proj_correlation_array_eigvals = np.squeeze(np.array(
            parallel.call_and_bcast(
            self.get_array, proj_correlation_array_eigvals_src)))
        self.proj_correlation_array_eigvecs = parallel.call_and_bcast(
            self.get_array, proj_correlation_array_eigvecs_src)


    def get_adv_correlation_array(self, src):
        """Gets the advanced correlation array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve advanced correlation array.
        """
        self.adv_correlation_array = parallel.call_and_bcast(
            self.get_array, src)


    def get_sum_correlation_array(self, src):
        """Gets the sum correlation array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve sum correlation array.
        """
        self.sum_correlation_array = parallel.call_and_bcast(
            self.get_array, src)


    def get_proj_correlation_array(self, src):
        """Gets the projected correlation array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve projected correlation
            array.
        """
        self.proj_correlation_array = parallel.call_and_bcast(
            self.get_array, src)


    def put_decomp(
        self, eigvals_dest, R_low_order_eigvecs_dest, L_low_order_eigvecs_dest,
        sum_correlation_array_eigvals_dest,
        sum_correlation_array_eigvecs_dest,
        proj_correlation_array_eigvals_dest,
        proj_correlation_array_eigvecs_dest):
        """Puts the decomposition arrays in destinations (file or memory).

        Args:
            ``eigvals_dest``: Destination in which to put eigenvalues of
            approximating low-order linear map (DMD eigenvalues).

            ``R_low_order_eigvecs_dest``: Destination in which to put right
            eigenvectors of approximating low-order linear map.

            ``L_low_order_eigvecs_dest``: Destination in which to put left
            eigenvectors of approximating low-order linear map.

            ``sum_correlation_array_eigvals_dest``: Destination in which to
            put eigenvalues of sum correlation array.

            ``sum_correlation_array_eigvecs_dest``: Destination in which to
            put eigenvectors of sum correlation array.

            ``proj_correlation_array_eigvals_dest``: Destination in which to put
            eigenvalues of projected correlation array.

            ``proj_correlation_array_eigvecs_dest``: Destination in which to put
            eigenvectors of projected correlation array.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_eigvals(eigvals_dest)
        self.put_R_low_order_eigvecs(R_low_order_eigvecs_dest)
        self.put_L_low_order_eigvecs(L_low_order_eigvecs_dest)
        self.put_sum_correlation_array_eigvals(
            sum_correlation_array_eigvals_dest)
        self.put_sum_correlation_array_eigvecs(
            sum_correlation_array_eigvecs_dest)
        self.put_proj_correlation_array_eigvals(
            proj_correlation_array_eigvals_dest)
        self.put_proj_correlation_array_eigvecs(
            proj_correlation_array_eigvecs_dest)


    def put_correlation_array_eigvals(self, dest):
        """This method is not available for total least squares DMD"""
        raise NotImplementedError(
            'This method is not available.  Use '
            'put_sum_correlation_array_eigvals instead.')


    def put_correlation_array_eigvecs(self, dest):
        """This method is not available for total least squares DMD"""
        raise NotImplementedError(
            'This method is not available.  Use '
            'put_sum_correlation_array_eigvecs instead.')


    def put_sum_correlation_array_eigvals(self, dest):
        """Puts eigenvalues of sum correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.sum_correlation_array_eigvals, dest)
        parallel.barrier()


    def put_sum_correlation_array_eigvecs(self, dest):
        """Puts eigenvectors of sum correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.sum_correlation_array_eigvecs, dest)
        parallel.barrier()


    def put_proj_correlation_array_eigvals(self, dest):
        """Puts eigenvalues of projected correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.proj_correlation_array_eigvals, dest)
        parallel.barrier()


    def put_proj_correlation_array_eigvecs(self, dest):
        """Puts eigenvectors of projected correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.proj_correlation_array_eigvecs, dest)
        parallel.barrier()


    def put_adv_correlation_array(self, dest):
        """Puts advanced correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.adv_correlation_array, dest)
        parallel.barrier()


    def put_sum_correlation_array(self, dest):
        """Puts sum correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.sum_correlation_array, dest)
        parallel.barrier()


    def put_proj_correlation_array(self, dest):
        """Puts projected correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.proj_correlation_array, dest)
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
        self.spectral_coeffs = np.abs(self.L_low_order_eigvecs.conj().T.dot(
            np.diag(self.proj_correlation_array_eigvals ** 0.5).dot(
                self.proj_correlation_array_eigvecs[0, :]).conj().T)).squeeze()
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
            ``proj_coeffs``: Array of projection coefficients for
            (de-noised)vector objects, expressed as a linear combination of DMD
            modes.  Columns correspond to vector objects, rows correspond to
            DMD modes.

            ``adv_proj_coeffs``: Array of projection coefficients for
            (de-noised) vector objects advanced in time, expressed as a linear
            combination of DMD modes.  Columns correspond to vector objects,
            rows correspond to DMD modes.
        """
        self.proj_coeffs = self.L_low_order_eigvecs.conj().T.dot(
            np.diag(self.proj_correlation_array_eigvals ** 0.5).dot(
                self.proj_correlation_array_eigvecs.conj().T))
        self.adv_proj_coeffs = self.L_low_order_eigvecs.conj().T.dot(
            np.diag(self.proj_correlation_array_eigvals ** -0.5).dot(
                self.proj_correlation_array_eigvecs.conj().T.dot(
                    self.sum_correlation_array_eigvecs.dot(
                        self.sum_correlation_array_eigvecs.conj().T.dot(
                            self.cross_correlation_array.dot(
                                self.sum_correlation_array_eigvecs.dot(
                                    self.sum_correlation_array_eigvecs.conj().T
                                )))))))
        return self.proj_coeffs, self.adv_proj_coeffs
