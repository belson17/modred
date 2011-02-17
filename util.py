#  A group of useful functions that don't belong to anything in particular

import subprocess as SP
import numpy as N

def write_mat_text(A,filename,delimiter=','):
    """
    Writes a matrix to file, 1D or 2D, in plain text with commas separating the 
    values.
    """
    s = N.shape(A)
    fid = open(filename,'w')
    if len(s) == 1:
        for r in range(s[0]):
            fid.write(str(A[r])+'\n')
    elif len(s) ==2:
        for r in range(s[0]):
            for c in range(s[1]-1):
                fid.write(str(A[r,c])+delimiter+' ')
                fid.write(str(A[r,-1])+'\n')
    else:
        raise RuntimeError('Can only save a 1D or 2D array, you gave a '+\
            str(len(s))+' dimensional array')    
    fid.close() 

def read_mat_text(filename,delimeter=','):
    """ Reads a matrix written by write_mat_text, plain text"""
    import csv
    f = open(filename,'r')
    matReader = csv.reader(f,delimiter=',')
    #read the entire file first to get dimensions.
    for i,line in enumerate(matReader):
        pass
  
    #rewind to beginning of file and read again
    f.seek(0)
    A = N.zeros((i+1,len(line)))
    for i,line in enumerate(matReader):
        A[i,:] =  [float(j) for j in line]
    return A

def find_file_type(filename):
    l = len(filename)
    n=-1
    while abs(n)<l and filename[n]!='.':
        n-=1
    fileExtension = filename[n+1:]
    return fileExtension

def get_file_list(dir,fileExtension=None):
    """ Finds all files in the given directory that have the given file extension"""
    filesRaw = SP.Popen(['ls',dir],stdout=SP.PIPE).communicate()[0]
    #files separated by endlines
    filename= ''
    fileList=[]
    #print 'filesRaw is ',filesRaw
    for c in filesRaw:
        if c!='\n':
            filename+=c
        else: #completed file name
            if fileExtension is not None and filename[-len(fileExtension):] == fileExtension:
                fileList.append(filename)
            elif fileExtension is None:
                fileList.append(filename)
            filename=''
    return fileList
  
class MPI(object):
    """Simple container for information about how many processors there are.
    It ensures no failure in case mpi4py is not installed or running serial."""
    def __init__(self,numCPUs=None):
        try:
            import mpi4py
            self.comm = mpi4py.COMM_WORLD
            if numCPUs is None or numCPUs > self.comm.Get_size(): 
                self.numCPUs = self.comm.Get_size()
            else: #use fewer CPUs than are available
                self.numCPUs = numCPUs      
            self.rank = self.comm.Get_rank()
        except ImportError:
            self.numCPUs=1
            self.rank=0
            self.comm = None



        
  
