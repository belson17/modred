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
    rtol=None, return_all=False):
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

        ``inner_product_weights``: 1D array or matrix of inner product weights.
          It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
        
        ``atol``: Level below which DMD eigenvalues are truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest 
          DMD eigenvalues.  Smaller ones are truncated.

        ``return_all``: Return more objects, see below. Default is false.

    Returns:
        ``exact_modes``: Matrix with requested exact DMD modes as columns.

        ``proj_modes``: Matrix with requested projected DMD modes as columns.

        ``spectral_coeffs``: 1D array of DMD spectral coefficients, based on a
          projection of the first data vector.

        ``eigvals``: 1D array of DMD eigenvalues.
                
        If ``return_all`` is true, also returns:
        
        ``R_low_order_eigvecs``: Matrix of right eigenvectors of the low-order
          linear DMD operator.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of the low-order
          linear DMD operator.

        ``correlation_mat_eigvals``: 1D array of eigenvalues of the 
          correlation matrix.

        ``correlation_mat_eigvecs``: Matrix of eigenvectors of the 
          correlation matrix.

        ``correlation_mat``: Matrix whose elements are inner products of each
          data vector with each other data vector.

        ``cross_correlation_mat``: Matrix whos elements are inner products of 
          each data vector with each advanced data vector.  Going down rows,
          the data vector changes; going across columns the advanced data
          vector changes.
   
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
    correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(
        correlation_mat_eigvals ** -0.5))
 
    # Compute low-order linear map for sequential or non-sequential case.
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
    spectral_coeffs = np.array(
        L_low_order_eigvecs.H *
        np.mat(np.diag(np.sqrt(correlation_mat_eigvals))) *
        correlation_mat_eigvecs[0, :].T).squeeze()

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
        return exact_modes, proj_modes, eigvals, spectral_coeffs


def compute_DMD_matrices_direct_method(
    vecs, mode_indices, adv_vecs=None, inner_product_weights=None, atol=1e-13,
    rtol=None, return_all=False):
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
            
        ``inner_product_weights``: 1D array or matrix of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which DMD eigenvalues are truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest 
            DMD eigenvalues.  Smaller ones are truncated.

        ``return_all``: Return more objects, see below. Default is false.

    Returns:
        ``exact_modes``: Matrix with requested exact DMD modes as columns.

        ``proj_modes``: Matrix with requested projected DMD modes as columns.

        ``spectral_coeffs``: 1D array of DMD spectral coefficients, based on a
            projection of the first data vector.

        ``eigvals``: 1D array of DMD eigenvalues.
        
        If ``return_all`` is true, also returns:
        
        ``R_low_order_eigvecs``: Matrix of right eigenvectors of the low-order
            linear DMD operator.

        ``L_low_order_eigvecs``: Matrix of left eigenvectors of the low-order
            linear DMD operator.

        ``correlation_mat_eigvals``: 1D array of eigenvalues of the 
            correlation matrix.

        ``correlation_mat_eigvecs``: Matrix of eigenvectors of the 
            correlation matrix.

        ``correlation_mat``: Matrix whose elements are inner products of each
            data vector with each other data vector.

        ``cross_correlation_mat``: Matrix whos elements are inner products of 
            each data vector with each advanced data vector.  Going down rows,
            the data vector changes; going across columns the advanced data
            vector changes.
        
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

        # If returning full set of data, compute cross-correlation mat
        # explicitly
        if return_all:
            cross_correlation_mat = np.concatenate(
                (correlation_mat[:, 1:], 
                vecs_weighted[:, :-1].H * vecs_weighted[:, -1]), axis=1)
    else: 
        if vecs.shape != adv_vecs.shape:
            raise ValueError(('vecs and adv_vecs are not the same shape.'))
        U, sing_vals, correlation_mat_eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)
        correlation_mat_eigvals = sing_vals ** 2
        correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(sing_vals ** -1.))
        low_order_linear_map = (
            U.H * adv_vecs_weighted * 
            correlation_mat_eigvecs * correlation_mat_eigvals_sqrt_inv)

        # If returning full set of data, compute correlation mat and cross-
        # correlation mat explicitly
        if return_all:
            correlation_mat = (
                correlation_mat_eigvecs * 
                np.mat(np.diag(correlation_mat_eigvals)) * 
                correlation_mat_eigvecs.H)
            cross_correlation_mat = vecs_weighted.H * adv_vecs_weighted

    # Compute eigendecomposition of low-order linear map.
    eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
        util.eig_biorthog(low_order_linear_map, scale_choice='left')
    build_coeffs_proj = (
        correlation_mat_eigvecs * correlation_mat_eigvals_sqrt_inv *
        R_low_order_eigvecs) 
    build_coeffs_exact = build_coeffs_proj * np.mat(np.diag(eigvals ** -1.))
    spectral_coeffs = np.array(
        L_low_order_eigvecs.H *
        np.mat(np.diag(np.sqrt(correlation_mat_eigvals))) * 
        correlation_mat_eigvecs[0, :].T).squeeze()

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
            correlation_mat_eigvecs, correlation_mat, cross_correlation_mat)
    else:
        return exact_modes, proj_modes, eigvals, spectral_coeffs


class DMDHandles(object):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition for large data.

    Args:
        ``inner_product``: Function to compute inner product.
        
    Kwargs:        
        ``put_mat``: Function to put a matrix out of modred.
      	
      	``get_mat``: Function to get a matrix into modred.
               
        ``max_vecs_per_node``: Max number of vectors in memory per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.
               
    Computes DMD modes from vecs.
    It uses :py:class:`vectorspace.VectorSpaceHandles` for low level functions.

    Usage::
    
      myDMD = DMDHandles(my_inner_product)
      myDMD.compute_decomp(vec_handles)
      myDMD.compute_modes(range(50), mode_handles)
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
        self, eigvals_src, R_low_order_eigvecs_src, 
        L_low_order_eigvecs_src, correlation_mat_eigvals_src,
        correlation_mat_eigvecs_src):
        """Retrieves the decomposition matrices from sources (memory or file).
        
        Args:
            ``eigvals_src``: Source from which to retrieve the DMD
              eigenvalues.

            ``R_low_order_eigvecs_src``: Source from which to retrieve the right
              eigenvectors of the low-order linear DMD map.
            
            ``L_low_order_eigvecs_src``: Source from wich to retrieve the left 
              eigenvectors of the low-order linear DMD map.

            ``correlation_mat_eigvals_src``: Source from which to retrieve the 
              eigenvalues of the correlation matrix.

            ``correlation_mat_eigvecs_src``: Source from which to retrieve the 
              eigenvectors of the correlation matrix.
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
        """Puts the decomposition matrices to destinations (file or memory).

        Args:
            ``eigvals_dest``: Destination to which to put the DMD
              eigenvalues.

            ``R_low_order_eigvecs_dest``: Destination to which to put the right
              eigenvectors of the low-order linear DMD map.
           
            ``L_low_order_eigvecs_dest``: Destination to which to put the left 
              eigenvectors of the low-order linear DMD map.
            
            ``correlation_mat_eigvals_dest``: Destination to which to put the 
              eigenvalues of the correlation matrix.

            ``correlation_mat_eigvecs_dest``: Destination to which to put the 
              eigenvectors of the correlation matrix.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_eigvals(eigvals_dest)
        self.put_R_low_order_eigvecs(R_low_order_eigvecs_dest)
        self.put_L_low_order_eigvecs(L_low_order_eigvecs_dest)
        self.put_correlation_mat_eigvals(correlation_mat_eigvals_dest)
        self.put_correlation_mat_eigvecs(correlation_mat_eigvecs_dest)

    def put_eigvals(self, dest):
        """Puts the DMD eigenvalues to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.eigvals, dest)
        _parallel.barrier()

    def put_R_low_order_eigvecs(self, dest):
        """Puts the right eigenvectors of the low-order linear DMD map to 
        ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.R_low_order_eigvecs, dest)
        _parallel.barrier()

    def put_L_low_order_eigvecs(self, dest):
        """Puts the left eigenvectors of the low-order linear DMD map to 
        ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.L_low_order_eigvecs, dest)
        _parallel.barrier()

    def put_correlation_mat_eigvals(self, dest):
        """Puts the eigenvalues of the correlation matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat_eigvals, dest)
        _parallel.barrier()

    def put_correlation_mat_eigvecs(self, dest):
        """Puts the eigenvectors of the correlation matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat_eigvecs, dest)
        _parallel.barrier()

    def put_correlation_mat(self, dest):
        """Puts the correlation mat to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, dest)
        _parallel.barrier()

    def put_cross_correlation_mat(self, dest):
        """Puts the correlation mat to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.cross_correlation_mat, dest)
        _parallel.barrier()

    def put_spectral_coeffs(self, dest):
        """Puts the spectral coefficients to ``dest``."""
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
        """Check user-supplied vector handle.
        
        Args:
            ``test_vec_handle``: A vector handle.
        
        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    def compute_eigendecomp(self, atol=1e-13, rtol=None):
        """Computes eigendecomposition of correlation matrix and low-order 
        linear DMD map.
       
        Kwargs:
            ``atol``: Level below which correlation matrix eigenvalues are
              truncated.
            
            ``rtol``: Maximum relative difference between largest and smallest
              correlation matrix eigenvalues.  Smaller ones are truncated. 

        Useful if already have correlation mat and cross-correlation mat and
        don't want to recompute them.
        Usage::
          
          DMD.correlation_mat = pre_existing_correlation_mat
          DMD.cross_correlation_mat = pre_existing_cross_correlation_mat
          DMD.compute_eigendecomp()
          DMD.compute_exact_modes(
            mode_idx_list, mode_handles, adv_vec_handles=adv_vec_handles)
        """
        # Compute eigendecomposition of correlation matrix
        self.correlation_mat_eigvals, self.correlation_mat_eigvecs = \
            _parallel.call_and_bcast(
            util.eigh, self.correlation_mat, atol=atol, rtol=None,
            is_positive_definite=True)
        correlation_mat_eigvals_sqrt_inv = np.mat(np.diag(
            self.correlation_mat_eigvals ** -0.5))
        
        # Compute low-order linear map 
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
        self, vec_handles, adv_vec_handles=None, atol=1e-13, rtol=None):
        """Computes decomposition and returns eigen decomposition matrices.
        
        Args:
            ``vec_handles``: List of handles for the data vectors.
        
        Kwargs:
            ``adv_vec_handles``: List of handles of ``vecs`` advanced in time.
            If not provided, it is assumed that the
            vectors are a sequential time-series. Thus ``vec_handles`` becomes
            ``vec_handles[:-1]`` and ``adv_vec_handles`` becomes 
            ``vec_handles[1:]``.
        
        ``atol``: Level below which DMD eigenvalues are truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest 
            DMD eigenvalues.  Smaller ones are truncated.
    
        Returns:
            ``eigvals``: 1D array of DMD eigenvalues.
            
            ``R_low_order_eigvecs``: Matrix of right eigenvectors of the
              low-order linear DMD operator.

            ``L_low_order_eigvecs``: Matrix of left eigenvectors of the
              low-order linear DMD operator.

            ``correlation_mat_eigvals``: 1D array of eigenvalues of the
              correlation matrix.

            ``correlation_mat_eigvecs``: Matrix of eigenvectors of the
              correlation matrix.
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
        self.compute_eigendecomp(atol=atol, rtol=rtol)

        return (
            self.eigvals, self.R_low_order_eigvecs,
            self.L_low_order_eigvecs, self.correlation_mat_eigvals,
            self.correlation_mat_eigvecs)
        
    def compute_exact_modes(self, mode_indices, mode_handles, 
        adv_vec_handles=None):
        """Computes exact DMD modes and calls ``put`` on them.
        
        Args:
            ``mode_indices``: List of mode indices, ``range(5)`` or 
            ``[3, 0, 5]``.
            
            ``mode_handles``: List of handles for modes.
            
        Kwargs:
            ``vec_handles``: List of handles for vecs, can omit if given in
            :py:meth:`compute_decomp`.
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
            if len(self.vec_handles) - self.build_coeffs_exact.shape[0] == 1:
                self.vec_space.lin_combine(mode_handles, self.vec_handles[1:], 
                    build_coeffs_exact, coeff_mat_col_indices=mode_indices)
            else:
                raise(ValueError, ('Number of vec_handles is not correct for a '
                    'sequential dataset.'))
        else:
            raise(ValueError, 'Neither vec_handles nor adv_vec_handles is '
                'defined.')

    def compute_proj_modes(self, mode_indices, mode_handles, vec_handles=None):
        """Computes projected DMD modes and calls ``put`` on them.
        
        Args:
            ``mode_indices``: List of mode indices, ``range(5)`` or 
            ``[3, 0, 5]``.
            
            ``mode_handles``: List of handles for modes.
            
        Kwargs:
            ``vec_handles``: List of handles for vecs, can omit if given in
            :py:meth:`compute_decomp`.
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
        """Computes DMD spectral coefficients.
        These coefficients come from projecting the first data vector onto the
        exact DMD modes, which is analytically equivalent to doing a least
        squares projection onto the projected DMD modes.
       
        Returns:
            ``spectral_coeffs``: 1D array of spectral coefficients.

        """
        # TODO: maybe allow for user to choose which column to spectrum from?
        # ie first, last, or mean?  
        self.spectral_coeffs = np.array(
            self.L_low_order_eigvecs.H *
            np.mat(np.diag(np.sqrt(self.correlation_mat_eigvals))) * 
            np.mat(self.correlation_mat_eigvecs[0, :]).T).squeeze()
        return self.spectral_coeffs

    # Note that a biorthogonal projection onto the exact DMD modes is the same
    # as a least squares projection onto the projected DMD modes, so there is
    # only one method for computing the projection coefficients.
    def compute_proj_coeffs(self):
        """Computes projection of data vectors onto DMD modes.  
        Note that a biorthogonal projection onto the exact DMD modes is
        analytically equivalent to a least squares projection onto the
        projected DMD modes.
       
        Returns:
            ``proj_coeffs``: Matrix of projection coefficients for the vectors.

            ``adv_proj_coeffs``: Matrix of projection coefficients for the
            vectors advanced in time.

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






