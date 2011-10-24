#  A group of useful functions that don't belong to anything in particular

import os
import subprocess as SP
import numpy as N
import inspect 
import copy


class UndefinedError(Exception): pass
    
def save_mat_text(A,filename,delimiter=' '):
    """Writes a 1D or 2D array or matrix to a text file
    
    delimeter separates the elements
    Complex data is saved in the following format (as floats)::
    
      real00 imag00 real01 imag01 ...
      real10 imag10 real11 imag11 ...
      ...
  
    It can easily be read in Matlab (provided .m files?).
    """
    """
    import csv
    if len(N.shape(A))>2:
        raise RuntimeError('Can only write matrices with 1 or 2 dimensions') 
    AMat = N.mat(copy.deepcopy(A))
    numRows,numCols = N.shape(AMat) #must be 2D since it is a matrix
    writer = csv.writer(open(filename,'w'),delimiter=delimiter)
       
    for rowNum in xrange(numRows):
        row=[str(AMat[rowNum,colNum]) for colNum in range(numCols)]
        writer.writerow(row)
    """
    # Must cast A into an array, makes it memory C-contiguous.
    N.savetxt(filename, N.array(A).view(float), delimiter=delimiter)
    
    
def load_mat_text(filename,delimiter=' ', is_complex=False):
    """ Reads a matrix written by write_mat_text, returns an *array*
    
    If the data saved is complex, then is_complex must be set to True.
    If this is not done, the array returned will be real with 2x the 
    correct number of columns.
    """
    """
    #print 'loading*file'
    import csv
    f = open(filename,'r')
    matReader = csv.reader(f,delimiter=delimiter)
    A=[]
    if isComplex:
        dtype = complex
    else:
        dtype = float
    for i,line in enumerate(matReader):
        A.append(N.array([dtype(j) for j in line]))
    return N.array(A)
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
    A = N.loadtxt(filename, delimiter=delimiter, ndmin=2).view(dtype)
    # Cast as an array, copies to make it C-contiguous memory
    return N.array(A)


def inner_product(snap1,snap2):
    """ A default inner product for n-dimensional numpy arrays """
    return (snap1*snap2.conj()).sum()

    
def svd(A):
    """An svd that better meets our needs.
    
    Returns U,E,V where U.E.V*=A. It truncates the matrices such that
    there are no ~0 singular values. U and V are numpy.matrix's, E is
    a 1D numpy.array.
    """
    singValTol=1e-13
    
    import copy
    AMat = N.mat(copy.deepcopy(A))
    
    U,E,VCompConj=N.linalg.svd(AMat,full_matrices=0)
    V=N.mat(VCompConj).H
    U=N.mat(U)
    
    #Take care of case where sing vals are ~0
    indexZeroSingVal=N.nonzero(abs(E)<singValTol)
    if len(indexZeroSingVal[0])>0:
        U=U[:,:indexZeroSingVal[0][0]]
        V=V[:,:indexZeroSingVal[0][0]]
        E=E[:indexZeroSingVal[0][0]]
    
    return U,E,V

def getFileList(dir,fileExtension=''):
    """Returns list of files in dir with file extension fileExtension"""
    fileList = os.listdir(dir)
    if len(fileExtension)>=1:
        filteredFileList = []
        for f in fileList:
            if f[-len(fileExtension):] == fileExtension:
                filteredFileList.append(f)
        return filteredFileList
    else:
        return fileList
        

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


