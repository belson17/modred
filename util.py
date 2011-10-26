#  A group of useful functions that don't belong to anything in particular

import os
import subprocess as SP
import numpy as N
import inspect 
import copy

class UndefinedError(Exception): pass
    
def save_mat_text(mat, filename, delimiter=' '):
    """Writes a 1D or 2D array or matrix to a text file
    
    delimeter separates the elements
    Complex data is saved in the following format (as floats)::
    
      real00 imag00 real01 imag01 ...
      real10 imag10 real11 imag11 ...
      ...
  
    It can easily be read in Matlab (provided .m files?). Brandt has written
    these matlab functions.
    """
    # Must cast mat into an array, makes it memory C-contiguous.
    mat_save = N.array(mat)
    
    # If one-dimensional arry, then make a vector of many rows, 1 column
    if mat_save.ndim == 1:
        mat_save = mat_save.reshape((-1,1))
    elif mat_save.ndim > 2:
        raise RuntimeError('Cannot save a matrix with >2 dimensions')

    N.savetxt(filename, mat_save.view(float), delimiter=delimiter)
    
    
def load_mat_text(filename, delimiter=' ', is_complex=False):
    """ Reads a matrix written by write_mat_text, returns an *array*
    
    If the data saved is complex, then is_complex must be set to True.
    If this is not done, the array returned will be real with 2x the 
    correct number of columns.
    """
    # Check the version of numpy, requires version >= 1.6 for ndmin option
    numpy_version = int(N.version.version[2])
    if numpy_version < 6:
        print 'Warning: load_mat_text requires numpy version >= 1.6 '+\
            'but you are running version %d'%numpy_version
    
    if is_complex:
        dtype = complex
    else:
        dtype = float
    mat = N.loadtxt(filename, delimiter=delimiter, ndmin=2)
    if is_complex and mat.shape[1]%2 != 0:
        raise ValueError(('Cannot load complex data, file %s '%filename)+\
            'has an odd number of columns. Maybe it has real data.')
            
    # Cast as an array, copies to make it C-contiguous memory
    return N.array(mat.view(dtype))


def inner_product(field1, field2):
    """ A default inner product for n-dimensional numpy arrays """
    return (field1*field2.conj()).sum()

    
def svd(mat, tol = 1e-13):
    """An SVD that better meets our needs.
    
    Returns U,E,V where U.E.V* = mat. It truncates the matrices such that
    there are no ~0 singular values. U and V are numpy.matrix's, E is
    a 1D numpy.array.
    """
    
    import copy
    mat_copied = N.mat(copy.deepcopy(mat))
    
    U, E, V_comp_conj = N.linalg.svd(mat_copied, full_matrices=0)
    V = N.mat(V_comp_conj).H
    U = N.mat(U)
    
    # Only return sing vals above the tolerance
    num_nonzeros = (abs(E) > tol).sum()
    if num_nonzeros > 0:
        U=U[:,:num_nonzeros]
        V=V[:,:num_nonzeros]
        E=E[:num_nonzeros]
    
    return U,E,V


def get_file_list(directory, file_extension=None):
    """Returns list of files in directory with file_extension"""
    files = os.listdir(directory)
    if file_extension is not None:
        if len(file_extension) == 0:
            print 'Warning: gave an empty file extension'
        filtered_files = []
        for f in files:
            if f[-len(fileExtension):] == file_extension:
                filtered_files.append(f)
        return filtered_files
    else:
        return files
        

def get_data_members(obj):
    """ Returns a dictionary containing data members of an object"""
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr


def sum_arrays(arr1,arr2):
    """Used for allreduce command, may not be necessary"""
    return arr1+arr2

    
def sum_lists(list1,list2):
    """Sum the elements of each list, return a new list.
    
    This function is used in MPI reduce commands, but could be used
    elsewhere too"""
    assert len(list1)==len(list2)
    return [list1[i]+list2[i] for i in xrange(len(list1))]


