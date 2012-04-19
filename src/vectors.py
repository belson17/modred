""" Vectors, handles, and inner products.

You should check if your case is provided.
If it is, use the corresponding functions or class.
If not, then you'll need to write your own vector class and/or vector handle.
You can use these as examples of how to do so.
"""

import cPickle
import sys
import numpy as N
import util

class VecHandle(object):
    """Recommended base class for vector handles (not required)"""
    cached_base_vec_handle = None
    cached_base_vec = None

    def __init__(self, base_vec_handle=None, scale=None):
        self.__base_vec_handle = base_vec_handle
        self.scale = scale
    
    def get(self):
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
        return self._put(vec)
    def _get(self):
        raise NotImplementedError("must be implemented by subclasses")
    def _put(self):
        raise NotImplementedError("must be implemented by subclasses")
    def __scale_vec(self, vec):
        if self.scale is not None:
            return vec*self.scale
        return vec


class InMemoryVecHandle(VecHandle):
    """Gets and puts vectors in memory"""
    def __init__(self, vec=None, base_vec_handle=None, scale=None):
        VecHandle.__init__(self, base_vec_handle, scale)
        self.vec = vec
    def _get(self):
        return self.vec
    def _put(self, vec):
        self.vec = vec
    def __eq__(self, other):
        if other is not None:
            return (self.vec == other.vec).all()
        return False
        
class ArrayTextVecHandle(VecHandle):
    """Gets and puts array vector objects to text files"""
    def __init__(self, vec_path, base_vec_handle=None, scale=None):
        VecHandle.__init__(self, base_vec_handle, scale)
        self.vec_path = vec_path
    def _get(self):
        return util.load_array_text(self.vec_path)
    def _put(self, vec):
        util.save_array_text(vec, self.vec_path)
    def __eq__(self, other):
        if other is not None:
            return (self.vec_path == other.vec_path).all()
        return False
        

class PickleVecHandle(VecHandle):
    """Gets and puts any vector object to pickle files"""
    def __init__(self, vec_path, base_vec_handle=None, scale=None):
        VecHandle.__init__(self, base_vec_handle, scale)
        self.vec_path = vec_path
    def _get(self):
        return cPickle.load(open(self.vec_path, 'rb'))
    def _put(self, vec):
        cPickle.dump(vec, open(self.vec_path, 'wb'))
    def __eq__(self, other):
        if other is not None:
            return (self.vec_path == other.vec_path).all()
        return False
        
        
def inner_product_array_uniform(self, vec1, vec2):
    return N.vdot(vec1, vec2)     


class InnerProductTrapz(object):
    """Vec object is an n-dimensional array on non-uniform grid.
    
    Args:
        As many 1D arrays of grid points as there are dimensions, in the
        order of the dimensions.
            x_grid: 1D array of grid points in x-dimension;
            y_grid: 1D array of grid points in y-dimension; ...
    
    The inner products are taken with trapezoidal rule.
    """
    def __init__(self, *grids):
        if len(grids) == 0:
            raise ValueError('Must supply at least one 1D grid array')
        self.grids = grids
    def __call__(self, v1, v2):
        return self.inner_product(v1, v2)
    def inner_product(self, v1, v2):
        IP = v1 * v2
        for grid in reversed(self.grids):
            IP = N.trapz(IP, x=grid)
        return IP


class Vector(object):
    """Recommended base class for vector objects (not required)."""
    def __add__(self, other):
        raise NotImplementedError('addition must be implemented by subclasses')
    def __mul(self, scalar):
        raise NotImplementedError('multiplication must be implemented by '
            'subclasses')
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    def __lmul__(self, scalar):
        return self.__mul__(scalar)
    def __sub__(self, other):
        return self + other*-1    

"""
def run_example():
    b = CustomVector("base")
    snapshots = [CustomVector("vec%d" % i, base_vector=b) for i in range(5)]
    b2 = CustomVector("base2")
    snapshots2 = [CustomVector("vec%d" % i, base_vector=b2) for i in range(5)]
    for s in snapshots:
        print s.get()
    for s in snapshots2:
        print s.get()
    
if __name__ == "__main__":
    run_example()
"""


"""
class Wrapper(Base):
    #Wraps an existing vector class.

    Args:
        constructor: The method that creates an instance of your vector class.
    
    If you have your own vector class that has member functions:
    1. ``my_vec.get_vec(vec_source)`` that gets a vector from ``vec_source`` 
        and puts the data into ``my_vec`` instance, and returns nothing.
    2. ``my_vec.put_vec(vec_dest)`` that puts ``my_vec`` in ``vec_dest`` and
        returns nothing
    3. ``my_vec.inner_product(other_vec)`` that returns the inner product of
       ``my_vec`` and ``other_vec``,
    
    then this class serves as a general wrapper.
    
    def __init__(self, constructor):
        self.constructor = constructor
        
    def get_vec(vec_source):
        vec = constructor()
        vec.get_vec(vec_source)
        return vec
    
    def put_vec(vec, vec_dest):
        vec.put_vec(vec_dest)
    
    def inner_product(vec1, vec2):
        return vec1.inner_product(vec2)
        
"""
