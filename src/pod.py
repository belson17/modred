"""POD class"""
import numpy as N

from vectorspace import *
import util
from parallel import parallel_default_instance
_parallel = parallel_default_instance

class PODBase(object):
    """Base class, instantiate py:class:`PODArrays` or py:class:`PODHandles`."""
    def __init__(self, get_mat=util.load_array_text, 
        put_mat=util.save_array_text, verbosity=1):
        """Constructor """
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbosity = verbosity
        self.eigen_vecs = None
        self.eigen_vals = None
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

    def compute_eigen_decomp(self):
        """Compute eigen decomp of ``correlation_mat``."""
        self.eigen_vals, self.eigen_vecs = _parallel.call_and_bcast(
            util.eigh, self.correlation_mat, is_positive_definite=True)

    def _compute_build_coeff_mat(self):
        """Compute transformation matrix (:math:`T`) from vectors to modes.
        Helper for ``compute_modes`` and ``compute_modes_and_return``."""
        if self.eigen_vecs is None:
            raise util.UndefinedError('Must define self.eigen_vecs')
        if self.eigen_vals is None:
            raise util.UndefinedError('Must define self.eigen_vals')
        build_coeff_mat = N.dot(self.eigen_vecs, N.diag(self.eigen_vals**-0.5))
        return build_coeff_mat
        

class PODArrays(PODBase):
    """Compute POD modes for small data.
    
    Kwargs:
    ``inner_product_weights``: 1D or 2D array of weights.
        The product ``X* inner_product_weights X`` will be computed.
    
    ``put_mat``: Function to put a matrix out of modred.

    ``get_mat``: Function to get a matrix into modred.

    ``verbosity``: 0 prints almost nothing, 1 prints progress and warnings.

    Computes orthonormal POD modes from vectors. Only use this in serial.
    It uses :py:class:`vectorspace.VectorSpaceArrays` for low level 
    functions.

    Usage::

      myPOD = PODArrays()
      eig_vecs, eig_vals = myPOD.compute_decomp(vecs)
      modes = myPOD.compute_modes(range(10))

    See also :mod:`vectors`.
    """
    def __init__(self, inner_product_weights=None, get_mat=util.load_array_text, 
        put_mat=util.save_array_text, verbosity=1):
        """Constructor"""
        if _parallel.is_distributed():
            raise RuntimeError('Cannot be used in parallel.')
        PODBase.__init__(self, get_mat=get_mat, put_mat=put_mat, 
            verbosity=verbosity)
        self.vec_space = VectorSpaceArrays(weights=inner_product_weights)
        self.vec_array = None
    
    def set_vec_array(self, vec_array):
        if vec_array.ndim == 1:
            self.vec_array = vec_array.reshape((vec_array.shape[0], 1))
        else:
            self.vec_array = vec_array
            
    def compute_decomp(self, vec_array):
        """Computes correlation matrix (:math:`X^*X`) and its eigen decomp.

        Args:
            ``vec_array``: 2D array of vectors stacked as columns (:math:`X`).

        Returns:
            ``eigen_vecs``: Matrix with eigen vectors as columns.

            ``eigen_vals``: 1D array of eigen values.
        """
        self.set_vec_array(vec_array)
        self.correlation_mat = \
            self.vec_space.compute_symmetric_inner_product_mat(self.vec_array)
        self.compute_eigen_decomp()
        return self.eigen_vecs, self.eigen_vals
    
    
    def compute_modes(self, mode_indices, vec_array=None):
        """Computes the modes and returns them.

        Args:
            ``mode_indices``: List of mode indices to compute.
                Examples are ``range(10)`` or ``[3, 0, 6, 8]``.

        Kwargs:
            ``vec_array``: 2D array with vectors as columns.  
                Optional if given when calling ``compute_decomp``.

        Returns:
            ``modes``: 2D array with requested modes as columns.
        """
        if vec_array is not None:
            self.set_vec_array(vec_array)
        if self.vec_array is None:
            raise UndefinedError('vec_array undefined')
        build_coeff_mat = self._compute_build_coeff_mat()
        return self.vec_space.lin_combine(self.vec_array,
            build_coeff_mat, coeff_mat_col_indices=mode_indices)
        

class PODHandles(PODBase):
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
        PODBase.__init__(self, get_mat=get_mat, put_mat=put_mat, 
            verbosity=verbosity)
        self.vec_space = VectorSpaceHandles(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.vec_handles = None
        

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
        self.compute_eigen_decomp()
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
