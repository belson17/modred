#  A group of useful functions that don't belong to anything in particular

import os
import subprocess as SP
import multiprocessing
from multiprocessing import Pool
import numpy as N
import inspect 
import copy
import operator
import pickle #cPickle?

class UndefinedError(Exception): pass

def getNumProcs():
    """Returns number of processors available (on a node)"""
    return multiprocessing.cpu_count()

def find_numRows_numCols_per_chunk(memoryLimit):
    """
    Returns optimal number of rows and columns for inner prod mat
    
    Maximizes the number of rows while making numCols big enough to
    make use of the shared memory. The argument memoryLimit is the
    maximum number of elements (fields) that can be in memory on a 
    node at once.
    This function might belong in fieldoperations module
    """
    # A sub chunk is the unit of "work" to be done by each processor
    # in the shared memory setting. For efficiency, we give pool jobs
    # that have # tasks = some multiple of the number of procs/node.
    # numSubChunks is the number of these work units.
    numSubChunks = memoryLimit / getNumProcs()
    
    # If can read more than 2 subChunks, where a sub chunk has
    # procs/node fields, then distribute
    # reading with one sub chunk in col, rest of the sub chunks in rows.
    # That is, maximize the number of rows per chunk.
    # iterate:
    #  rows
    #    cols
    if numSubChunks >= 2:
        numColsPerChunk = getNumProcs() 
        numRowsPerChunk = memoryLimit - numColsPerChunk
        
    # If not, still maximize the number of rows per chunk, leftovers for col
    elif numSubChunks == 1:
        numRowsPerChunk = getNumProcs()
        numColsPerChunk = memoryLimit - numRowsPerChunk
        
    # If can't get even numProcsPerNode fields in memory at once, then
    # default to slowest option, will not make full use of shared memory
    # NOTE: I'm not sure this is the fastest way for this case
    else:
        numColsPerChunk = 1
        numRowsPerChunk = memoryLimit - numColsPerChunk
    return numRowsPerChunk, numColsPerChunk


def addSlash(s):
    if len(s) > 0:
        if s[-1] != '/':
            s+='/'
        return s
    else:
        raise RuntimeError('empty string, cant add directory slash')

def save_mat_text(A,filename,delimiter=' '):
    """Writes a 1D or 2D array or matrix to a text file
    
    delimeter seperates the elements.
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
    
    
def load_mat_text(filename,delimiter=' ',isComplex=False):
    """ Reads a matrix written by write_mat_text, plain text, returns ARRAY"""
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


def inner_product(snap1,snap2):
    """ A default inner product for n-dimensional numpy arrays """
    return N.sum(snap1*snap2.conj()) 

    
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

def getFileListSubprocess(dir, fileExtension=''):
    """Gets the files in dir with 'ls' and subprocess"""
    rawList = SP.Popen(['ls '+dir+'*'+fileExtension],stdout=SP.PIPE,shell=True).communicate()[0]
    fileList = ['']
    for chari,char in enumerate(rawList):
        if char != '\n':
            fileList[-1]+=char
        elif chari != len(rawList)-1:
            fileList.append('')
    
    return [file[len(dir):] for file in fileList]


def get_data_members(obj):
    """ Returns a dictionary containing data members of an object"""
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr
    
    
def sum_lists(list1,list2):
    """Sum the elements of each list, return a new list.
    
    This function is used in MPI reduce commands, but could be used
    elsewhere too"""
    assert len(list1)==len(list2)
    return [list1[i]+list2[i] for i in xrange(len(list1))]



def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])
    
# Simple function for testing
def add(x,y): return x+y
def my_random(arg):
    return N.random.random(arg)
def inner_product_wrapper(args):
    assert(len(args)==2)
    return inner_product(*args)
    
def my_inner_product(a,b):
    ip = 0
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            ip += a[r,c]*b[r,c]
    return ip





