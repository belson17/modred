""" Commonly used vector functions.

We provide  
``get_vec``, ``put_vec``, and ``inner_product`` functions for some more common
cases.
We also provide a group of classes that use these functions.

You should check if your case is provided.
If it is, great! Just use the corresponding functions or class.
If not, then you'll need to write your own vector definition class or module,
which is really not complicated.
"""

import cPickle
import numpy as N

import util
import parallel
parallel = parallel.default_instance

# Common get, put, and inner product functions to mix and match
def get_vec_text(vec_path):
    """Only for arrays"""
    return util.load_mat_text(vec_path)
def put_vec_text(vec, vec_path):
    """Only for arrays"""
    util.save_mat_text(vec, vec_path)

def get_vec_pickle(vec_path):
    """For any object"""
    return cPickle.load(open(vec_path, 'rb'))
def put_vec_pickle(vec, vec_path):
    """For any object"""
    cPickle.dump(vec, open(vec_path, 'wb'))

def get_vec_in_memory(vec):
    """For any object."""
    return vec
def put_vec_in_memory(vec, dummy_dest):
    """For any object. Makes ``compute_modes`` return a list of modes. 
    
    Destination is ``dummy_dest`` and isn't used"""
    return vec

def inner_product_uniform(vec1, vec2):
    """Only for arrays"""
    return util.inner_product(vec1, vec2)


class BaseEqual(object):
    """Supplies the equality special method only.
    
    It is useful for tests when asserting that vec_def instances are equal.
    """
    def __eq__(self, other): 
        """Check equal"""
        return (other.get_vec==self.get_vec and other.put_vec==self.put_vec and
            other.inner_product==self.inner_product)

    
class VecDefsArrayText(BaseEqual):
    """Saves to text, vec object is an array.
    
    Usage::
      my_POD = POD(VecDefsArrayText())
      my_POD.compute_decomp(vec_paths)
      my_POD.compute_modes(10, mode_paths)
      
    """
    def __init__(self):
        self.get_vec = get_vec_text
        self.put_vec = put_vec_text
        self.inner_product = inner_product_uniform


class VecDefsArrayPickle(BaseEqual):
    """Saves to pickle files, vec object is an array.
    
    Usage::
      my_POD = POD(VecDefsArrayPickle())
      my_POD.compute_decomp(vec_paths)
      my_POD.compute_modes(10, mode_paths)
        
    """
    def __init__(self):
        self.get_vec = get_vec_pickle
        self.put_vec = put_vec_pickle
        self.inner_product = inner_product_uniform
        

class VecDefsArrayInMemory(BaseEqual):
    """Returns vectors in memory, vec object is an array.
    
    Can only be used in serial.
    
    Usage::
    
      # Define a list of vectors.
      my_POD = POD(VecDefsArrayInMemory())
      my_POD.compute_decomp(vectors)
      modes = my_POD.compute_modes(10, 'return')
      
    """
    def __init__(self):
        """Check if serial, only works in serial"""
        #if parallel.is_distributed():
        #    raise RuntimeError('This vec def class cannot be used in parallel')
        self.inner_product = inner_product_uniform
        self.get_vec = get_vec_in_memory
        self.put_vec = put_vec_in_memory


class VecDefsNonUniform2D(BaseEqual):
    """For vecs that are 2D arrays on non-uniform grid. Saves in text.
    
    Args:
        x_grid: 1D array of grid points in x-dimension
        
        y_grid: 1D array of grid points in y-dimension
    
    The inner products are taken with trapezoidal rule.
    """
    def __init__(self, x_grid, y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.get_vec = get_vec_text
        self.put_vec = put_vec_text

    def inner_product(self, vec1, vec2):
        return N.trapz(N.trapz(vec1 * vec2, x=self.y_grid), x=self.x_grid) 
    


"""
class VecDefsWrapper(BaseEqual):
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
    
    



