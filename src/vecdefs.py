""" Commonly used vector functions.

We provide  
``get_vec``, ``put_vec``, and ``inner_product`` functions for some more common
cases and a group of classes that use these functions.

You should check if your case is provided.
If it is, use the corresponding functions or class.
If not, then you'll need to write your own vector definition class or module,
which is really not complicated.
You can use these as examples of how to do so.
"""

import cPickle
import numpy as N
import util


class Base(object):
    """Supplies the common methods.
    
    Kwargs:
        base_vec: Subtracted from each vector when ``get_vec`` is called.
    """
    def __init__(self, base_vec_source=None):
        if base_vec_source is not None:
            self.base_vec = self.get_vec(base_vec_source)
        else:
            self.base_vec = None
    
    def _subtract_base_vec(self, vec):
        """Subtracts base vector from vec, if it is not none"""
        if self.base_vec is not None:
            return vec - base_vec
        else:
            return vec

    def __eq__(self, other): 
        """Check equal"""
        return util.get_data_members(self) == util.get_data_members(other)
    
    
class ArrayText(Base):
    """Vec object is an array, loads/saves to text files.
    
    This class is meant to be a base class, where the derived class
    also contains an inner product.
    """
    def __init__(self, base_vec_source=None):
        Base.__init__(self, base_vec_source=base_vec_source)
    def get_vec(self, vec_path):
        """Loads 1D or 2D array from text file"""
        return self._subtract_base_vec(util.load_mat_text(vec_path))
    def put_vec(self, vec, vec_path):
        """Saves 1D or 2D arrays to text file"""
        util.save_mat_text(vec, vec_path)
    def inner_product(self, v1, v2):
        raise RuntimeError('Base class, has no inner_product')
    
    
class ArrayPickle(Base):
    """Vec object is an array, loads/saves to pickle files.
    
    This class is meant to be a base class, where the derived class
    also contains an inner product.
    """
    def __init__(self, base_vec_source=None):
        Base.__init__(self, base_vec_source=base_vec_source)
    def get_vec(self, vec_path):
        """Loads 1D or 2D array from text file"""
        return self._subtract_base_vec(cPickle.load(open(vec_path, 'rb')))
    def put_vec(self, vec, vec_path):
        """Saves 1D or 2D arrays to text file"""
        cPickle.dump(vec, open(vec_path, 'wb'))
    def inner_product(self, v1, v2):
        raise RuntimeError('Base class, has no inner_product')      

class ArrayInMemory(Base):
    """Vec object is an array, puts/gets arrays in memory.
    
    This class is meant to be a base class, where the derived class
    also contains an inner product.
    """
    def __init__(self, base_vec_source=None):
        Base.__init__(self, base_vec_source=base_vec_source)
    def get_vec(self, vec):
        """Returns vec in memory"""
        return self._subtract_base_vec(vec)
    def put_vec(self, vec, dummy_dest):
        """Returns vec in memory, ignores destination argument"""
        return vec


class ArrayInMemoryUniform(ArrayInMemory):
    def inner_product(self, vec1, vec2):
        return N.vdot(vec1, vec2)  
        
class ArrayTextUniform(ArrayText):
    """Vec object is a 1D or 2D array, loads/saves in text.
       
    The inner products are taken assuming a uniform sampling or grid.
    
      my_POD = POD(ArrayTextUniform(base_vec_source='base_vec.txt'))
      sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vectors)
      modes = my_POD.compute_modes(range(20), ['mode%02d.txt'%i for i in range(20)])
    
    """
    def inner_product(self, vec1, vec2):
        return N.vdot(vec1, vec2)     


class ArrayPickleUniform(ArrayPickle):
    """Vec object is a 1D or 2D array, loads/saves in pickle format.
       
    The inner products are taken assuming a uniform sampling or grid.
    
      my_POD = POD(ArrayTextUniform(base_vec_source='base_vec.pkl'))
      sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vectors)
      modes = my_POD.compute_modes(range(20), ['mode%02d.pkl'%i for i in range(20)])
    
    """
    def inner_product(self, vec1, vec2):
        return N.vdot(vec1, vec2)     




class ArrayTextNonUniform2D(ArrayText):
    """Vec object is a 2D array on non-uniform grid, loads/saves in text.
    
    Args:
        x_grid: 1D array of grid points in x-dimension
        
        y_grid: 1D array of grid points in y-dimension
    
    The inner products are taken with trapezoidal rule.
    
      # Define a list of vectors and 1D grid vectors ``x`` and ``y``.
      my_POD = POD(ArrayTextNonUniform2D(x, y))
      sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vectors)
      modes = my_POD.compute_modes(range(20), ['mode%02d.pkl'%i for i in range(20)])
    
    """
    def __init__(self, x_grid, y_grid, base_vec_source=None):
        Base.__init__(self, base_vec_source=base_vec_source)
        self.x_grid = x_grid
        self.y_grid = y_grid
       
    def inner_product(self, vec1, vec2):
        return N.trapz(N.trapz(vec1 * vec2, x=self.y_grid), x=self.x_grid) 
    


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
