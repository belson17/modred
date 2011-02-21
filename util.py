#  A group of useful functions that don't belong to anything in particular

import subprocess as SP
import numpy as N

class UndefinedError(Exception):
    pass

def save_mat_text(A,filename,delimiter=','):
    """
    Writes a matrix to file, 1D or 2D, in text with delimeter and a space
    seperating the elements.
    """
    import csv
    import copy
    if len(N.shape(A))>2:
        raise RuntimeError('Can only write matrices with 1 or 2 dimensions') 
    AMat = N.mat(copy.deepcopy(A))
    numRows,numCols = N.shape(AMat) #must be 2D since it is a matrix
    writer = csv.writer(open(filename,'w'),delimiter=delimiter)
       
    for rowNum in range(numRows):
        row=[]
        for colNum in range(numCols):
            row.append(str(AMat[rowNum,colNum]))
        writer.writerow(row)
    
def load_mat_text(filename,delimiter=','):
    """ Reads a matrix written by write_mat_text, plain text"""
    import csv
    f = open(filename,'r')
    matReader = csv.reader(f,delimiter=delimiter)
    #read the entire file first to get dimensions.
    numLines = 0
    for line in matReader:
        if numLines ==0:
            lineLength = len(line)
        numLines+=1
    if numLines == 0:
        raise RuntimeError('File is empty! '+filename)
    #rewind to beginning of file and read again
    f.seek(0)
    A = N.zeros((numLines,lineLength))
    for i,line in enumerate(matReader):
        A[i,:] =  N.array([float(j) for j in line])
    return A
    
def inner_product(snap1,snap2):
    """ A default inner product for n-dimensional numpy arrays """
    return N.sum(snap1*snap2)
  
  
class MPIError(Exception):
    """For MPI related errors"""
    pass
  
class MPI(object):
    """Simple container for information about how many processors there are.
    It ensures no failure in case mpi4py is not installed or running serial."""
    def __init__(self,numProcs=None):
        try:
            from mpi4py import MPI as MPI_mod
            self.comm = MPI_mod.COMM_WORLD
            if (numProcs is None) or (numProcs > self.comm.Get_size()) or \
            (numProcs<=0): 
                self._numProcs = self.comm.Get_size()
            else: #use fewer CPUs than are available
                self._numProcs = numProcs      
            self._rank = self.comm.Get_rank()
            self.parallel=True
        except ImportError:
            self._numProcs=1
            self._rank=0
            self.comm = None
            self.parallel=False
            
    def sync(self):
        """Forces all processors to synchronize
        
        Method computes simple formula based on ranks of each proc, then
        asserts that results make sense and each proc reported back. This
        forces all processors to wait for others to "catch up"
        It is self-testing and for now does not need a unittest."""
        if self.parallel:
            data = (self._rank+1)**2
            data = self.comm.gather(data, root=0)
            if self._rank == 0:
                for i in range(self._numProcs):
                    assert data[i] == (i+1)**2
            else:
                assert data is None

    def find_proc_assignments(self,taskList):
      """ Returns a 2D list of tasks, [rank][taskIndex], evenly
      breaking up the tasks in the taskList. 
      
      It returns a list that has numProcs+1 entries. 
      Proc n is responsible for taskProcAssignments[n][...]
      where the 2nd dimension of the 2D list contains the tasks (whatever
      they were in the original taskList).
      """
      
      taskProcAssignments= []
      prevMaxTaskIndex = 0
      import copy
      taskListUse = copy.deepcopy(taskList)
      numTasks = len(taskList)
      for procNum in range(self._numProcs):
          numRemainingTasks = len(taskListUse)
          numRemainingProcs = self._numProcs - procNum
          numTasksPerProc = int(N.ceil(numRemainingTasks/
            (1.*numRemainingProcs)))
          newMaxTaskIndex = min(numTasksPerProc,numRemainingTasks)
          taskProcAssignments.append(taskListUse[:newMaxTaskIndex])
          for removeElement in taskListUse[:newMaxTaskIndex]:
              taskListUse.remove(removeElement)
          prevMaxTaskIndex = newMaxTaskIndex
      #currently do not support 0 tasks for a proc
      for assignment in taskProcAssignments:
          if len(assignment)==0:
              print taskProcAssignments
              raise MPIError('At least one processor has no tasks'+\
                ', currently this is unsupported, lower num of procs')
      return taskProcAssignments
    
    
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





"""
def find_file_type(filename):
    l = len(filename)
    n=-1
    while abs(n)<l and filename[n]!='.':
        n-=1
    fileExtension = filename[n+1:]
    return fileExtension

def get_file_list(dir,fileExtension=None):
    # Finds all files in the given directory that have file extension
    filesRaw = SP.Popen(['ls',dir],stdout=SP.PIPE).communicate()[0]
    #files separated by endlines
    filename= ''
    fileList=[]
    #print 'filesRaw is ',filesRaw
    for c in filesRaw:
        if c!='\n':
            filename+=c
        else: #completed file name
            if fileExtension is not None and \
              filename[-len(fileExtension):] == fileExtension:
                fileList.append(filename)
            elif fileExtension is None:
                fileList.append(filename)
            filename=''
    return fileList
"""
