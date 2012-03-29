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

##### Common functions that *only* work when vectors are arrays. ######
# Mix and match these as needed.
def get_vec_text(vec_path):
    """Loads 1D and 2D arrays from text file"""
    return util.load_mat_text(vec_path)
def put_vec_text(vec, vec_path):
    """Saves 1D and 2D arrays to text file"""
    util.save_mat_text(vec, vec_path)
def inner_product_uniform(vec1, vec2):
    """Only for arrays"""
    return util.inner_product(vec1, vec2)



##### Common functions that work when vectors are *any* object. ######
# Mix and match these as needed.
def get_vec_pickle(vec_path):
    """Loads object from pickle file"""
    return cPickle.load(open(vec_path, 'rb'))
def put_vec_pickle(vec, vec_path):
    """Saves object from pickle file"""
    cPickle.dump(vec, open(vec_path, 'wb'))
def get_vec_in_memory(vec):
    """Returns object"""
    return vec
def put_vec_in_memory(vec, dummy_dest):
    """Return object (ignores second argument)"""
    return vec


class Base(object):
    """Supplies the common methods.
    
    Kwargs:
        base_vec: Subtracted from each vector when ``get_vec`` is called.
    """
    def __init__(self, base_vec=None):
        self.base_vec = base_vec
    def __eq__(self, other): 
        """Check equal"""
        return util.get_data_members(self) == util.get_data_members(other)
    
class ArrayText(Base):
    """Vec object is an array, loads/saves to text files.
    
    Usage::
    
      my_POD = POD(ArrayText())
      sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vectors)
      modes = my_POD.compute_modes_and_return(range(10))
      
    """
    def __init__(self, base_vec=None):
        Base.__init__(self, base_vec)
        self.put_vec = put_vec_text
        self.inner_product = inner_product_uniform
    def get_vec(self, vec_source):
        vec = get_vec_text(vec_source)
        if self.base_vec is not None:
            vec = vec - base_vec
        return vec
    
class ArrayPickle(Base):
    """Vec object is an array, loads/saves to pickle files.
    
    Usage::
    
      my_POD = POD(ArrayPickle())
      my_POD.compute_decomp(vectors, 'sing_vecs.txt', 'sing_vals.txt')
      my_POD.compute_modes(range(10), ['mode%02d.pkl'%i for i in range(10)])
        
    """
    def __init__(self, base_vec=None):
        Base.__init__(self, base_vec)
        self.put_vec = put_vec_pickle
        self.inner_product = inner_product_uniform
    def get_vec(self, vec_source):
        if self.base_vec is not None:
            return get_vec_pickle(vec_source) - base_vec
        return get_vec_pickle(vec_source)        

class ArrayInMemory(Base):
    """Vec object is an array, returns vecs in memory.
    
    Usage::
    
      # Define a list of vectors.
      my_POD = POD(ArrayInMemory(base_vec))
      sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vectors)
      modes = my_POD.compute_modes_and_return(range(10))      
    """
    def __init__(self, base_vec=None):
        Base.__init__(self, base_vec)
        self.inner_product = inner_product_uniform
        self.put_vec = put_vec_in_memory
    def get_vec(self, vec_source):
        if self.base_vec is not None:
            return get_vec_in_memory(vec_source) - base_vec
        return get_vec_in_memory(vec_source)
    def __eq__(self, other): 
        """Check equal"""
        return (other.get_vec==self.get_vec and other.put_vec==self.put_vec and
            other.inner_product==self.inner_product)



class ArrayTextNonUniform2D(Base):
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
    def __init__(self, x_grid, y_grid, base_vec=None):
        Base.__init__(self, base_vec)
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.put_vec = put_vec_text
        
    def get_vec(self, vec_source):
        if self.base_vec is not None:
            return get_vec_text(vec_source) - base_vec
        return get_vec_text(vec_source)

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
