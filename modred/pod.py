from __future__ import print_function
from __future__ import absolute_import
from future.builtins import object
from collections import namedtuple

import numpy as np

from .vectorspace import VectorSpaceArrays, VectorSpaceHandles
from . import util
from . import parallel


def compute_POD_arrays_snaps_method(
    vecs, mode_indices=None, inner_product_weights=None, atol=1e-13, rtol=None):
    """Computes POD modes using data stored in an array, using the method of
    snapshots.

    Args:
        ``vecs``: Array whose columns are data vectors (:math:`X`).

    Kwargs:
        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which eigenvalues of correlation array are
        truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues of correlation array.  Smaller ones are truncated.

    Returns:
        ``res``: Results of POD computation, stored in a namedtuple with
        the following attributes:

        * ``eigvals``: 1D array of eigenvalues of correlation array
          (:math:`E`).

        * ``modes``: Array whose columns are POD modes.

        * ``proj_coeffs``: Array of projection coefficients for vector objects,
          expressed as a linear combination of POD modes.  Columns correspond to
          vector objects, rows correspond to POD modes.

        * ``eigvecs``: Array wholse columns are eigenvectors of correlation
          array (:math:`U`).

        * ``correlation_array``: Correlation array (:math:`X^* W X`).

        Attributes can be accessed using calls like ``res.modes``.  To see all
        available attributes, use ``print(res)``.

    The algorithm is

    1. Solve eigenvalue problem :math:`X^* W X U = U E`
    2. Coefficient array :math:`T = U E^{-1/2}`
    3. Modes are :math:`X T`

    where :math:`X`, :math:`W`, :math:`X^* W X`, and :math:`T` correspond to
    ``vecs``, ``inner_product_weights``, ``correlation_array``, and
    ``build_coeffs``, respectively.

    Since this method "squares" the vector array and thus its singular values,
    it is slightly less accurate than taking the SVD of :math:`X` directly,
    as in :py:func:`compute_POD_arrays_direct_method`.
    However, this method is faster when :math:`X` has more rows than columns,
    i.e. there are more elements in each vector than there are vectors.

    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')

    # Force data to be arrays (not matrices)
    vecs = np.array(vecs)

    # Set up vector space (for inner products)
    vec_space = VectorSpaceArrays(weights=inner_product_weights)

    # Compute decomp
    correlation_array = vec_space.compute_symm_inner_product_array(vecs)
    eigvals, eigvecs = util.eigh(
        correlation_array, atol=atol, rtol=rtol, is_positive_definite=True)

    # Compute modes
    build_coeffs = eigvecs.dot(np.diag(eigvals ** -0.5))
    modes = vec_space.lin_combine(
        vecs, build_coeffs, coeff_array_col_indices=mode_indices)

    # Compute projection coefficients
    proj_coeffs = np.diag(eigvals ** 0.5).dot(eigvecs.conj().T)

    # Return a namedtuple
    POD_results = namedtuple(
        'POD_results',
        ['eigvals', 'modes', 'proj_coeffs', 'eigvecs', 'correlation_array'])
    return POD_results(
        eigvals=eigvals, modes=modes, proj_coeffs=proj_coeffs, eigvecs=eigvecs,
        correlation_array=correlation_array)


def compute_POD_arrays_direct_method(
    vecs, mode_indices=None, inner_product_weights=None, atol=1e-13, rtol=None):
    """Computes POD modes using data stored in an array, using direct method.

    Args:
        ``vecs``: Array whose columns are data vectors (:math:`X`).

    Kwargs:
        ``mode_indices``: List of indices describing which modes to compute.
        Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no mode indices are
        specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which eigenvalues of correlation array are
        truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues of correlation array.  Smaller ones are truncated.

        ``return_all``: Return more objects; see below. Default is false.

    Returns:
        ``res``: Results of POD computation, stored in a namedtuple with the
        following attributes:

        * ``eigvals``: 1D array of eigenvalues of correlation array
          (:math:`E`).

        * ``modes``: Array whose columns are POD modes.

        * ``proj_coeffs``: Array of projection coefficients for vector objects,
          expressed as a linear combination of POD modes.  Columns correspond to
          vector objects, rows correspond to POD modes.

        * ``eigvecs``: Array wholse columns are eigenvectors of correlation
          array (:math:`U`).

        Attributes can be accessed using calls like ``res.modes``.
        To see all available attributes, use ``print(res)``.

    The algorithm is

    1. SVD :math:`U S V^* = W^{1/2} X`
    2. Modes are :math:`W^{-1/2} U`

    where :math:`X`, :math:`W`, :math:`S`, :math:`V`, correspond to
    ``vecs``, ``inner_product_weights``, ``eigvals**0.5``,
    and ``eigvecs``, respectively.

    Since this method does not square the vectors and singular values, it is
    more accurate than taking the eigendecomposition of the correlation array
    :math:`X^* W X`, as in the method of snapshots
    (:py:func:`compute_POD_arrays_snaps_method`).  However, this method is
    slower when :math:`X` has more rows than columns, i.e. there are fewer
    vectors than elements in each vector.

    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')

    # Force data to be arrays (not matrices)
    vecs = np.array(vecs)

    # If no inner product weights, compute SVD directly
    if inner_product_weights is None:
        modes, sing_vals, eigvecs = util.svd(vecs, atol=atol, rtol=rtol)
        if mode_indices is None:
            mode_indices = range(sing_vals.size)
        modes = modes[:, mode_indices]
    # For 1D inner product weights, compute square root and weight vecs
    # accordingly
    elif inner_product_weights.ndim == 1:
        sqrt_weights = inner_product_weights ** 0.5
        vecs_weighted = np.diag(sqrt_weights).dot(vecs)
        modes_weighted, sing_vals, eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)
        if mode_indices is None:
            mode_indices = range(sing_vals.size)
        modes = np.diag(sqrt_weights ** -1.).dot(
            modes_weighted[:, mode_indices])
    # For 2D inner product weights, compute Cholesky factorization and weight
    # vecs accordingly.
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print('Warning: Cholesky decomposition could be time consuming.')
        sqrt_weights = np.linalg.cholesky(inner_product_weights).conj().T
        vecs_weighted = sqrt_weights.dot(vecs)
        modes_weighted, sing_vals, eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)
        if mode_indices is None:
            mode_indices = range(sing_vals.size)
        #modes = np.linalg.solve(sqrt_weights, modes_weighted[:, mode_indices])
        inv_sqrt_weights = np.linalg.inv(sqrt_weights)
        modes = inv_sqrt_weights.dot(modes_weighted[:, mode_indices])

    # Compute projection coefficients
    eigvals = sing_vals ** 2.
    proj_coeffs = np.diag(eigvals ** 0.5).dot(eigvecs.conj().T)

    # Return a namedtuple
    POD_results = namedtuple(
        'POD_results',
        ['eigvals', 'modes', 'proj_coeffs', 'eigvecs'])
    return POD_results(
        eigvals=eigvals, modes=modes, proj_coeffs=proj_coeffs, eigvecs=eigvecs)


class PODHandles(object):
    """Proper Orthogonal Decomposition implemented for large datasets.

    Args:
        ``inner_product``: Function that computes inner product of two vector
        objects.

    Kwargs:
        ``put_array``: Function to put an array out of modred, e.g., write it to
        file.

      	``get_array``: Function to get an array into modred, e.g., load it from
        file.

        ``max_vecs_per_node``: Maximum number of vectors that can be stored in
        memory, per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.

    Computes POD modes from vector objects (or handles).  Uses
    :py:class:`vectorspace.VectorSpaceHandles` for low level functions.

    Usage::

      myPOD = POD(my_inner_product)
      myPOD.compute_decomp(vec_handles)
      myPOD.compute_modes(range(10), modes)

    See also :func:`compute_POD_arrays_snaps_method`,
    :func:`compute_POD_arrays_direct_method`, and :mod:`vectors`.
    """
    def __init__(
        self, inner_product, get_array=util.load_array_text,
        put_array=util.save_array_text, max_vecs_per_node=None, verbosity=1):
        self.get_array = get_array
        self.put_array = put_array
        self.verbosity = verbosity
        self.eigvecs = None
        self.eigvals = None

        self.vec_space = VectorSpaceHandles(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.vec_handles = None
        self.correlation_array = None


    def get_decomp(self, eigvals_src, eigvecs_src):
        """Gets the decomposition arrays from sources (memory or file).

        Args:
            ``eigvals_src``: Source from which to retrieve eigenvalues of
            correlation array.

            ``eigvecs_src``: Source from which to retrieve eigenvectors of
            correlation array.
        """
        self.eigvals = np.squeeze(np.array(parallel.call_and_bcast(
            self.get_array, eigvals_src)))
        self.eigvecs = parallel.call_and_bcast(self.get_array, eigvecs_src)


    def get_correlation_array(self, src):
        """Gets the correlation array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve correlation array.
        """
        self.correlation_array = parallel.call_and_bcast(self.get_array, src)


    def get_proj_coeffs(self, src):
        """Gets projection coefficients from source (memory or file).

        Args:
            ``src``: Source from which to retrieve projection coefficients.
        """
        self.proj_coeffs = parallel.call_and_bcast(self.get_array, src)


    def put_decomp(self, eigvals_dest, eigvecs_dest):
        """Puts the decomposition arrays in destinations (memory or file).

        Args:
            ``eigvals_dest``: Destination in which to put eigenvalues of
            correlation array.

            ``eigvecs_dest``: Destination in which to put the eigenvectors of
            correlation array.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_eigvecs(eigvecs_dest)
        self.put_eigvals(eigvals_dest)


    def put_eigvals(self, dest):
        """Puts eigenvalues of correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.eigvals, dest)
        parallel.barrier()


    def put_eigvecs(self, dest):
        """Puts eigenvectors of correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.eigvecs, dest)
        parallel.barrier()


    def put_correlation_array(self, dest):
        """Puts correlation array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.correlation_array, dest)
        parallel.barrier()


    def put_proj_coeffs(self, dest):
        """Puts projection coefficients to ``dest``"""
        if parallel.is_rank_zero():
            self.put_array(self.proj_coeffs, dest)
        parallel.barrier()


    def sanity_check(self, test_vec_handle):
        """Checks that user-supplied vector handle and vector satisfy
        requirements.

        Args:
            ``test_vec_handle``: A vector handle to test.

        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)


    def compute_eigendecomp(self, atol=1e-13, rtol=None):
        """Computes eigendecomposition of correlation array.

        Kwargs:
            ``atol``: Level below which eigenvalues of correlation array are
            truncated.

            ``rtol``: Maximum relative difference between largest and smallest
            eigenvalues of correlation array.  Smaller ones are truncated.

        Useful if you already have the correlation array and to want to avoid
        recomputing it.

        Usage::

          POD.correlation_array = pre_existing_correlation_array
          POD.compute_eigendecomp()
          POD.compute_modes(range(10), mode_handles, vec_handles=vec_handles)
        """
        self.eigvals, self.eigvecs = parallel.call_and_bcast(
            util.eigh, self.correlation_array, atol=atol, rtol=rtol,
            is_positive_definite=True)


    def compute_decomp(self, vec_handles, atol=1e-13, rtol=None):
        """Computes correlation array :math:`X^*WX` and its eigendecomposition.

        Args:
            ``vec_handles``: List of handles for vector objects.

        Kwargs:
            ``atol``: Level below which eigenvalues of correlation array are
            truncated.

            ``rtol``: Maximum relative difference between largest and smallest
            eigenvalues of correlation array.  Smaller ones are truncated.

        Returns:
            ``eigvals``: 1D array of eigenvalues of correlation array.

            ``eigvecs``: Array whose columns are eigenvectors of correlation
            array.
        """
        self.vec_handles = vec_handles
        self.correlation_array =\
            self.vec_space.compute_symm_inner_product_array(
                self.vec_handles)
        self.compute_eigendecomp(atol=atol, rtol=rtol)
        return self.eigvals, self.eigvecs


    def compute_modes(self, mode_indices, mode_handles, vec_handles=None):
        """Computes POD modes and calls ``put`` on them using mode handles.

        Args:
            ``mode_indices``: List of indices describing which modes to
            compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for modes to compute.

        Kwargs:
            ``vec_handles``: List of handles for vector objects. Optional if
            when calling :py:meth:`compute_decomp`.
        """
        if vec_handles is not None:
            self.vec_handles = util.make_iterable(vec_handles)
        build_coeffs = np.dot(self.eigvecs, np.diag(self.eigvals**-0.5))
        self.vec_space.lin_combine(
            mode_handles, self.vec_handles, build_coeffs,
            coeff_array_col_indices=mode_indices)


    def compute_proj_coeffs(self):
        """Computes orthogonal projection of vector objects onto POD modes.

        Returns:
            ``proj_coeffs``: Array of projection coefficients for vector
            objects, expressed as a linear combination of POD modes.  Columns
            correspond to vector objects, rows correspond to POD modes.
        """
        self.proj_coeffs = np.diag(self.eigvals ** 0.5).dot(
            self.eigvecs.conj().T)
        return self.proj_coeffs
