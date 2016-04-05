from __future__ import print_function
from __future__ import absolute_import
from future.builtins import object

import numpy as np
from .vectorspace import VectorSpaceMatrices, VectorSpaceHandles
from . import util
from .parallel import parallel_default_instance
_parallel = parallel_default_instance

def compute_POD_matrices_snaps_method(
    vecs, mode_indices, inner_product_weights=None, atol=1e-13, rtol=None,
    return_all=False):
    """Computes POD modes with data in a matrix using the method of snapshots.
    
    Args:
        ``vecs``: Matrix of data vectors stacked as columns.
        
        ``mode_indices``: List of mode indices to compute.
            Examples are ``range(10)`` or ``[3, 0, 6, 8]``.

    Kwargs:
        ``inner_product_weights``: 1D array or matrix of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

`       ``atol``: Level below which POD eigenvalues are truncated.
 
        ``rtol``: Maximum relative difference between largest and smallest POD
            eigenvalues.  Smaller ones are truncated.

        ``return_all``: Return more objects, see below. Default is false.
        
    Returns:
        ``modes``: Matrix with requested modes as columns.
        
        ``eigvals``: 1D array of eigenvalues.
        
        If ``return_all`` is true, also returns:
        
        ``eigvecs``: Matrix of eigenvectors of correlation matrix.
        
        ``correlation_mat``: Matrix of inner products of all vecs in ``vecs``.
                
    The algorithm is
    
    1. Solve eigenvalue problem :math:`X^* W X U = U E`
    2. Coefficient matrix :math:`T = U E^{-1/2}`
    3. Modes are :math:`X T`
    
    where :math:`X`, :math:`W`, :math:`X^* W X`, and :math:`T` correspond to 
    ``vecs``, ``inner_product_weights``, ``correlation_mat``, 
    and ``build_coeff_mat``, respectively.
       
    Since this method "squares" the vector array and thus its singular values,
    it is slightly less accurate than taking the SVD of :math:`X` directly,
    as in :py:func:`compute_POD_arrays_direct_method`. 
    However, this method is faster when :math:`X` has more rows than columns, 
    i.e. there are more elements in each vector than there are vectors.
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceMatrices(weights=inner_product_weights)
    # compute decomp
    vecs = util.make_mat(vecs)
    correlation_mat = \
        vec_space.compute_symmetric_inner_product_mat(vecs)
    eigvals, eigvecs = util.eigh(
        correlation_mat, atol=atol, rtol=rtol, is_positive_definite=True)
    # compute modes
    build_coeff_mat = eigvecs * np.mat(np.diag(eigvals**-0.5))
    modes = vec_space.lin_combine(vecs,
        build_coeff_mat, coeff_mat_col_indices=mode_indices)
    if return_all:
        return modes, eigvals, eigvecs, correlation_mat
    else:
        return modes, eigvals

def compute_POD_matrices_direct_method(
    vecs, mode_indices, inner_product_weights=None, atol=1e-13, rtol=None, 
    return_all=False):
    """Computes POD modes with data in a matrix using the direct method.
    
    Args:
        ``vecs``: Matrix of data vectors stacked as columns.
        
        ``mode_indices``: List of mode indices to compute.
            Examples are ``range(10)`` or ``[3, 0, 6, 8]``.

    Kwargs:
        ``inner_product_weights``: 1D array or matrix of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
       
        ``atol``: Level below which POD eigenvalues are truncated.

        ``rtol``: Maximum relative difference between largest and smallest
            POD eigenvalues.  Smaller ones are truncated.

        ``return_all``: Return more objects, see below. Default is false.
    
    Returns:
        ``modes``: Matrix with requested modes as columns.
        
        ``eigvals``: 1D array of eigenvalues. 
            These are the eigenvalues of the correlation matrix 
            (:math:`X^* W X`), and are also the squares of the singular values 
            of :math:`X`. 
        
        If ``return_all`` is true, also returns:
               
        ``eigvecs``: Matrix of eigenvectors.
            These are the eigenvectors of correlation matrix (:math:`X^* W X`),
            and are also the right singular vectors of :math:`X`.
                
    The algorithm is
    
    1. SVD :math:`U E V^* = W^{1/2} X`
    2. Modes are :math:`W^{-1/2} U`
    
    where :math:`X`, :math:`W`, :math:`E`, :math:`V`, correspond to 
    ``vecs``, ``inner_product_weights``, ``eigvals**0.5``, 
    and ``eigvecs``, respectively.
       
    Since this method does not square the vectors and singular values,
    it is more accurate than taking the eigen decomposition of :math:`X^* W X`,
    as in the method of snapshots (:py:func:`compute_POD_arrays_direct_method`). 
    However, this method is slower when :math:`X` has more rows than columns, 
    i.e. there are fewer vectors than elements in each vector.
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vecs = util.make_mat(vecs)
    if inner_product_weights is None:
        modes, sing_vals, eigvecs = util.svd(vecs, atol=atol, rtol=rtol)
        modes = modes[:, mode_indices]
    
    elif inner_product_weights.ndim == 1:
        sqrt_weights = inner_product_weights**0.5
        vecs_weighted = np.mat(np.diag(sqrt_weights)) * vecs
        modes_weighted, sing_vals, eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)
        modes = np.mat(
            np.diag(sqrt_weights**-1.0)) * modes_weighted[:,mode_indices]
            
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print('Warning: Cholesky decomposition could be time consuming.')
        sqrt_weights = np.linalg.cholesky(inner_product_weights).H
        vecs_weighted = sqrt_weights * vecs
        modes_weighted, sing_vals, eigvecs = util.svd(
            vecs_weighted, atol=atol, rtol=rtol)
        modes = np.linalg.solve(sqrt_weights, modes_weighted[:, mode_indices])
        #inv_sqrt_weights = np.linalg.inv(sqrt_weights)
        #modes = inv_sqrt_weights.dot(modes_weighted[:, mode_indices])
    
    eigvals = sing_vals**2
    
    if return_all:
        return modes, eigvals, eigvecs
    else:
        return modes, eigvals


class PODHandles(object):
    """Proper Orthogonal Decomposition.

    Args:
        ``inner_product``: Function to find inner product of two vector objects.

    Kwargs:
        ``put_mat``: Function to put a matrix out of modred.

        ``get_mat``: Function to get a matrix into modred.

        ``verbosity``: 0 prints almost nothing, 1 prints progress and warnings.

        ``max_vec_handles_per_node``: Max number of vectors in memory per node.

    Computes orthonormal POD modes from vec_handles.
    It uses :py:class:`vectorspace.VectorSpaceHandles` for low level functions.

    Usage::

      myPOD = POD(my_inner_product)
      myPOD.compute_decomp(vec_handles)
      myPOD.compute_modes(range(10), modes)

    See also :func:`compute_POD_arrays_snaps_meth`, 
    :func:`compute_POD_arrays_direct_meth`, and :mod:`vectors`.
    """
    def __init__(self, inner_product,
        get_mat=util.load_array_text, put_mat=util.save_array_text,
        max_vecs_per_node=None, verbosity=1):
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.eigvecs = None
        self.eigvals = None

        self.vec_space = VectorSpaceHandles(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.vec_handles = None
        self.correlation_mat = None

    def get_decomp(self, eigvecs_source, eigvals_source):
        """Gets the decomposition matrices from sources (memory or file)."""
        self.eigvecs = _parallel.call_and_bcast(self.get_mat,
            eigvecs_source)
        self.eigvals = np.squeeze(np.array(_parallel.call_and_bcast(
            self.get_mat, eigvals_source)))

    def put_decomp(self, eigvecs_dest, eigvals_dest):
        """Put the decomposition matrices to file or memory."""
        # Don't check if rank is zero because the following methods do.
        self.put_eigvecs(eigvecs_dest)
        self.put_eigvals(eigvals_dest)

    def put_eigvecs(self, dest):
        """Put eigenvectors to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.eigvecs, dest)
        _parallel.barrier()

    def put_eigvals(self, dest):
        """Put eigenvalues to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.eigvals, dest)
        _parallel.barrier()

    def put_correlation_mat(self, dest):
        """Put correlation matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, dest)
        _parallel.barrier()


    def sanity_check(self, test_vec_handle):
        """Check user-supplied vector handle.

        Args:
            ``test_vec_handle``: A vector handle.

        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    
    def compute_eigendecomp(self, atol=1e-13, rtol=None):
        """Computes eigendecomp of correlation matrix.
       
        Kwargs:
            ``atol``: Level below which POD eigenvalues are truncated.
 
            ``rtol``: Maximum relative difference between largest and smallest 
                POD eigenvalues.  Smaller ones are truncated.

        Useful if already have correlation mat and don't want to recompute it.
        Usage::
        
          POD.correlation_mat = pre_existing_correlation_mat
          POD.compute_eigendecomp()
          POD.compute_modes(range(10), modes, vec_handles=vec_handles)
        
        """
        self.eigvals, self.eigvecs = _parallel.call_and_bcast(
            util.eigh, self.correlation_mat, atol=atol, rtol=rtol, 
            is_positive_definite=True)
        
    def compute_decomp(self, vec_handles, atol=1e-13, rtol=None):
        """Computes correlation matrix, :math:`X^*WX` and its eigen decomp.

        Args:
            ``vec_handles``: List of vector handles.

        Kwargs:
            ``atol``: Level below which POD eigenvalues are truncated.
 
            ``rtol``: Maximum relative difference between largest and smallest 
                POD eigenvalues.  Smaller ones are truncated.

        Returns:
            ``eigenvec_handles``: Matrix with eigenvectors as columns.

            ``eigvals``: 1D array of eigenvalues.
        """
        self.vec_handles = vec_handles
        self.correlation_mat = self.vec_space.\
            compute_symmetric_inner_product_mat(self.vec_handles)
        #self.correlation_mat = self.vec_space.\
        #    compute_inner_product_mat(self.vec_handles, self.vec_handles)
        #self.eigvals, self.eigvecs = _parallel.call_and_bcast(
        #    util.eigh, self.correlation_mat, is_positive_definite=True)
        self.compute_eigendecomp(atol=atol, rtol=rtol)
        return self.eigvecs, self.eigvals

    def compute_modes(self, mode_indices, modes, vec_handles=None):
        """Computes the modes and calls ``put`` on the mode handles.

        Args:
            ``mode_indices``: List of mode numbers, e.g. ``range(10)`` or 
            ``[3, 0, 5]``.

            ``modes``: List of handles for modes.

        Kwargs:
            ``vec_handles``: List of handles for vectors. Optional if given
            when calling ``compute_decomp``.
        """
        if vec_handles is not None:
            self.vec_handles = util.make_iterable(vec_handles)
        build_coeff_mat = np.dot(self.eigvecs, np.diag(self.eigvals**-0.5))
        self.vec_space.lin_combine(modes, self.vec_handles, build_coeff_mat, 
            coeff_mat_col_indices=mode_indices)


    def compute_proj_coeffs(self):
        """Computes projection of data vectors onto POD modes.  
       
        Returns:
            ``proj_coeffs``: Matrix of projection coefficients for the vectors.

        """
        self.proj_coeffs = np.mat(np.diag(self.eigvals ** 0.5)) * self.eigvecs.H
        return self.proj_coeffs        


