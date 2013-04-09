"""POD class"""
import numpy as N

from vectorspace import *
import util
from parallel import parallel_default_instance
_parallel = parallel_default_instance

def compute_POD_arrays_snaps_method(vec_array, mode_indices, 
    inner_product_weights=None, return_all=False):
    """Computes the POD modes from arrays using the method of snapshots.
    
    Args:
        ``vec_array``: 2D array of vectors stacked as columns.
        
        ``mode_indices``: List of mode indices to compute.
            Examples are ``range(10)`` or ``[3, 0, 6, 8]``.

    Kwargs:
        ``inner_product_weights``: 1D or 2D array of inner product weights.
            It corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
        
        ``return_all``: Changes what is returned, see below. Default is false.
        
    Returns:
        ``modes``: 2D array with requested modes as columns.
        
        ``eigen_vals``: 1D array of eigen values.
        
        If ``return_all`` is true, also returns:
        
        ``eigen_vecs``: 2D array of eigen vectors of correlation matrix.
        
        ``correlation_mat``: Matrix of inner products of all vecs in ``vec_array``.
                
    The algorithm is
    
    1. Eigendecomp :math:`U E = X^* W X`.
    2. Coefficient matrix :math:`T = U E^{-1/2}`.
    3. Modes are :math:`T X`.
    
    where :math:`X`, :math:`W`, :math:`X^* W X`, and :math:`T` correspond to 
    ``vec_array``, ``inner_product_weights``, ``correlation_mat``, 
    and ``build_coeff_mat``, respectively.
       
    Since this method "squares" the vectors and thus the singular values,
    it is slightly less accurate than taking the SVD of :math:`X` directly,
    as in :py:func:`compute_POD_arrays_direct_method`. 
    However, this method is faster when :math:`X` has more rows than columns, 
    i.e. there are fewer vectors than elements in each vector.
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceArrays(weights=inner_product_weights)
    # compute decomp
    vec_array = util.make_2D_array(vec_array)
    correlation_mat = \
        vec_space.compute_symmetric_inner_product_mat(vec_array)
    eigen_vals, eigen_vecs = util.eigh(correlation_mat, 
        is_positive_definite=True)
    # compute modes
    build_coeff_mat = N.dot(eigen_vecs, N.diag(eigen_vals**-0.5))
    modes = vec_space.lin_combine(vec_array,
        build_coeff_mat, coeff_mat_col_indices=mode_indices)
    if return_all:
        return modes, eigen_vals, eigen_vecs, correlation_mat
    else:
        return modes, eigen_vals

def compute_POD_arrays_direct_method(vec_array, mode_indices,
    inner_product_weights=None, return_all=False):
    """Computes the POD modes from arrays using the method of snapshots.
    
    Args:
        ``vec_array``: 2D array of vectors stacked as columns.
        
        ``mode_indices``: List of mode indices to compute.
            Examples are ``range(10)`` or ``[3, 0, 6, 8]``.

    Kwargs:
        ``inner_product_weights``: 1D or 2D array of inner product weights.
            The inner product of two vectors is :math:`v_1^* W v_2` where 
            :math:`W` corresponds to ``inner_product_weights``.
        
        ``return_all``: Changes what is returned, see below. Default is false.
    
    Returns:
        ``modes``: 2D array with requested modes as columns.
        
        ``eigen_vals``: 1D array of eigenvalues. 
            These are the eigenvalues of the correlation matrix (:math:`X^* W X`), 
            and are also the squares of the singular values of :math:`X`. 
        
        If ``return_all`` is true, also returns:
               
        ``eigen_vecs``: 2D array of eigenvectors.
            These are the eigenvectors of correlation matrix (:math:`X^* W X`),
            and are also the right singular vectors of :math:`X`.
                
    The algorithm is
    
    1. SVD :math:`U E V = W^{1/2} X`.
    2. Modes are :math:`W^{-1/2} U`.
    
    where :math:`X`, :math:`W`, :math:`E`, :math:`V`, correspond to 
    ``vec_array``, ``inner_product_weights``, ``eigen_vals**0.5``, 
    and ``eigen_vecs``, respectively.
       
    Since this method does not square the vectors and singular values,
    it is more accurate than taking the eigen decomposition of :math:`X^* W X`,
    as in the method of snapshots (:py:func:`compute_POD_arrays_direct_method`). 
    However, this method is slower when :math:`X` has more rows than columns, 
    i.e. there are fewer vectors than elements in each vector.
    """
    if _parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_array = util.make_2D_array(vec_array)
    if inner_product_weights is None:
        modes, E, eigen_vecs = util.svd(N.array(vec_array))
        eigen_vals = E**2
        modes = modes[:,mode_indices]
    
    elif inner_product_weights.ndim == 2:
        if inner_product_weights.shape[0] > 500:
            print 'Warning: Cholesky decomposition could be time consuming.'
        sqrt_weights = N.linalg.cholesky(inner_product_weights)
        vec_array_weighted = sqrt_weights.dot(N.array(vec_array))
        modes_weighted, E, eigen_vecs = util.svd(vec_array_weighted)
        eigen_vals = E**2
        inv_sqrt_weights = N.linalg.inv(sqrt_weights)
        modes = N.linalg.solve(sqrt_weights, modes_weighted[:,mode_indices])
        #modes = inv_sqrt_weights.dot(modes_weighted[:, mode_indices])
    
    elif inner_product_weights.ndim == 1:
        sqrt_weights = inner_product_weights**0.5
        vec_array_weighted = N.swapaxes(N.swapaxes(
            N.array(vec_array), 0, 1) * sqrt_weights, 0, 1)
        modes_weighted, E, eigen_vecs = util.svd(vec_array_weighted)
        eigen_vals = E**2
        modes = N.swapaxes(N.swapaxes(
            N.array(modes_weighted[:,mode_indices]),0,1) * 
            sqrt_weights**-1.0, 0, 1)
    
    if return_all:
        return modes, eigen_vals, eigen_vecs
    else:
        return modes, eigen_vals
    





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

    See also :mod:`vectors`.
    """
    def __init__(self, inner_product,
        get_mat=util.load_array_text, put_mat=util.save_array_text,
        max_vecs_per_node=None, verbosity=1):
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.eigen_vecs = None
        self.eigen_vals = None

        self.vec_space = VectorSpaceHandles(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.vec_handles = None
        self.correlation_mat = None

    def get_decomp(self, eigen_vecs_source, eigen_vals_source):
        """Gets the decomposition matrices from sources (memory or file)."""
        self.eigen_vecs = _parallel.call_and_bcast(self.get_mat,
            eigen_vecs_source)
        self.eigen_vals = N.squeeze(N.array(_parallel.call_and_bcast(
            self.get_mat, eigen_vals_source)))

    def put_decomp(self, eigen_vecs_dest, eigen_vals_dest):
        """Put the decomposition matrices to file or memory."""
        # Don't check if rank is zero because the following methods do.
        self.put_eigen_vecs(eigen_vecs_dest)
        self.put_eigen_vals(eigen_vals_dest)

    def put_eigen_vecs(self, dest):
        """Put eigenvectors to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.eigen_vecs, dest)
        _parallel.barrier()

    def put_eigen_vals(self, dest):
        """Put eigenvalues to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.eigen_vals, dest)
        _parallel.barrier()

    def put_correlation_mat(self, dest):
        """Put correlation matrix to ``dest``."""
        if _parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, dest)
        _parallel.barrier()

    def _compute_build_coeff_mat(self):
        """Compute transformation matrix (:math:`T`) from vectors to modes.
        Helper for ``compute_modes`` and ``compute_modes_and_return``."""
        if self.eigen_vecs is None:
            raise util.UndefinedError('Must define self.eigen_vecs')
        if self.eigen_vals is None:
            raise util.UndefinedError('Must define self.eigen_vals')
        build_coeff_mat = N.dot(self.eigen_vecs, N.diag(self.eigen_vals**-0.5))
        return build_coeff_mat


    def sanity_check(self, test_vec_handle):
        """Check user-supplied vector handle.

        Args:
            ``test_vec_handle``: A vector handle.

        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)


    def compute_decomp(self, vec_handles):
        """Computes correlation mat (:math:`X^*X`), and its eigen decomp.

        Args:
            ``vec_handles``: List of vector handles (:math:`X`).

        Returns:
            ``eigen_vec_handles``: Matrix with eigen vectors as columns.

            ``eigen_vals``: 1D array of eigen values.
        """
        self.vec_handles = vec_handles
        self.correlation_mat = self.vec_space.\
            compute_symmetric_inner_product_mat(self.vec_handles)
        #self.correlation_mat = self.vec_space.\
        #    compute_inner_product_mat(self.vec_handles, self.vec_handles)
        self.eigen_vals, self.eigen_vecs = _parallel.call_and_bcast(
            util.eigh, self.correlation_mat, is_positive_definite=True)
        return self.eigen_vecs, self.eigen_vals

    def compute_modes(self, mode_indices, modes, vec_handles=None):
        """Computes the modes and calls ``put`` on the mode handles.

        Args:
            ``mode_indices``: List of mode indices to compute.
                Examples are ``range(10)`` or ``[3, 0, 6, 8]``.

            ``modes``: List of handles for modes.

        Kwargs:
            ``vec_handles``: List of handles for vectors. Optional if given
            when calling ``compute_decomp``.
        """
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)
        build_coeff_mat = self._compute_build_coeff_mat()
        self.vec_space.lin_combine(modes, self.vec_handles, build_coeff_mat, 
            coeff_mat_col_indices=mode_indices)
