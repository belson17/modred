""" Vectors, handles, and inner products.

We recommend using these functions and classes when possible.
Otherwise, you can write your own vector class and/or vector handle,
see documentation :ref:`sec_details`.
"""

import cPickle
import numpy as N
import util

class VecHandle(object):
    """Recommended base class for vector handles (not required)."""
    cached_base_vec_handle = None
    cached_base_vec = None

    def __init__(self, base_vec_handle=None, scale=None):
        self.__base_vec_handle = base_vec_handle
        self.scale = scale
    
    def get(self):
        """Get a vector, using the private ``_get`` function.  If available,
        the base vector will be subtracted from the specified vector.  Then, if
        a scale factor is specified, the base-subtracted vector will be scaled.
        The scaled, base-subtracted vector is then returned."""
        vec = self._get()
        if self.__base_vec_handle is None:
            return self.__scale_vec(vec)
        if self.__base_vec_handle == VecHandle.cached_base_vec_handle:
            base_vec = VecHandle.cached_base_vec
        else:
            base_vec = self.__base_vec_handle.get()
            VecHandle.cached_base_vec_handle = self.__base_vec_handle
            VecHandle.cached_base_vec = base_vec
        return self.__scale_vec(vec - base_vec)
        
    def put(self, vec):
        """Put a vector to file or memory using the ``_put`` function.""" 
        return self._put(vec)
    def _get(self):
        """Subclass must overwrite, retrieves a vector."""
        raise NotImplementedError("must be implemented by subclasses")
    def _put(self, vec):
        """Subclass must overwrite, puts a vector."""
        raise NotImplementedError("must be implemented by subclasses")
    def __scale_vec(self, vec):
        """Scales the vector by a scalar."""
        if self.scale is not None:
            return vec*self.scale
        return vec
    


class InMemoryVecHandle(VecHandle):
    """Gets and puts vectors in memory."""
    def __init__(self, vec=None, base_vec_handle=None, scale=None):
        VecHandle.__init__(self, base_vec_handle, scale)
        self.vec = vec
    def _get(self):
        """Returns the vector."""
        return self.vec
    def _put(self, vec):
        """Stores the vector, ``vec``."""
        self.vec = vec
    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return util.smart_eq(self.vec, other.vec)
        
        
class ArrayTextVecHandle(VecHandle):
    """Gets and puts array vector objects to text files."""
    def __init__(self, vec_path, base_vec_handle=None, scale=None):
        VecHandle.__init__(self, base_vec_handle, scale)
        self.vec_path = vec_path
    def _get(self):
        """Loads vector from path."""
        return util.load_array_text(self.vec_path)
    def _put(self, vec):
        """Saves vector to path."""
        util.save_array_text(vec, self.vec_path)
    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.vec_path == other.vec_path
        

class PickleVecHandle(VecHandle):
    """Gets and puts any vector object to pickle files."""
    def __init__(self, vec_path, base_vec_handle=None, scale=None):
        VecHandle.__init__(self, base_vec_handle, scale)
        self.vec_path = vec_path
    def _get(self):
        """Loads vector from path."""
        return cPickle.load(open(self.vec_path, 'rb'))
    def _put(self, vec):
        """Saves vector to path."""
        cPickle.dump(vec, open(self.vec_path, 'wb'))
    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.vec_path == other.vec_path
        
        
def inner_product_array_uniform(vec1, vec2):
    """Takes inner product of numpy arrays without weighting."""
    return N.vdot(vec1, vec2)


class InnerProductTrapz(object):
    """Inner product of n-dim arrays on grid using the trapezoidal rule.
    
    Args:
        ``grids``: 1D arrays of grid points, in the order of the dims.
            x_grid: 1D array of grid points in x-dimension;
            y_grid: 1D array of grid points in y-dimension; ...
    
    Usage::
      
      nx = 10
      ny = 11
      v1 = N.random.random((nx,ny))
      v2 = N.random.random((nx,ny))
      x_grid = N.arange(0, N.pi, nx)
      y_grid = N.arange(-1, 1, ny)
      my_trapz = InnerProductTrapz(x_grid, y_grid)
      IP = my_trapz(v1, v2)
    
    """
    def __init__(self, *grids):
        if len(grids) == 0:
            raise ValueError('Must supply at least one 1D grid array')
        self.grids = grids
    def __call__(self, vec1, vec2):
        return self.inner_product(vec1, vec2)
    def inner_product(self, vec1, vec2):
        """Takes the inner product. Also can call instances of the class."""
        IP = vec1 * vec2
        for grid in reversed(self.grids):
            if not isinstance(grid, N.ndarray):
                raise TypeError('Each grid must be a numpy array, not a '
                    '%s'%str(type(grid)))
            IP = N.trapz(IP, x=grid)
        return IP


class Vector(object):
    """Recommended base class for vector objects (not required)."""
    def __init__(self):
        """Must overwrite"""
        raise NotImplementedError('constructor must be implemented by subclass')
    def __add__(self, other):
        raise NotImplementedError('addition must be implemented by subclasses')
    def __mul__(self, scalar):
        raise NotImplementedError('multiplication must be implemented by '
            'subclasses')
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __lmul__(self, scalar):
        return self.__mul__(scalar)
    def __sub__(self, other):
        return self + other*-1    

