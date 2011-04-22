#  A group of useful functions that don't belong to anything in particular

import subprocess as SP
import numpy as N
import inspect 
import copy

class UndefinedError(Exception): pass
    
class MPIError(Exception):
    """For MPI related errors"""
    pass

def save_mat_text(A,filename,delimiter=' '):
    """
    Writes a matrix to file, 1D or 2D, in text with delimeter and a space
    seperating the elements.
    """
    import csv
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
    
def load_mat_text(filename,delimiter=' ',isComplex=False):
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
    if isComplex:
        A = N.zeros((numLines,lineLength),dtype=complex)
        for i,line in enumerate(matReader):
            A[i,:] =  N.array([complex(j) for j in line])
    else:
        A = N.zeros((numLines,lineLength))
        for i,line in enumerate(matReader):
            A[i,:] =  N.array([float(j) for j in line])
    return A

def inner_product(snap1,snap2):
    """ A default inner product for n-dimensional numpy arrays """
    return N.sum(snap1*snap2.conj())
  
  
class MPI(object):
    """Simple container for information about how many processors there are.
    It ensures no failure in case mpi4py is not installed or running serial."""
    def __init__(self,verbose=False):
        self.verbose = verbose
        try:
            # Must call MPI module MPI_mod to avoid naming confusion with
            # the MPI class
            from mpi4py import MPI as MPI_mod
            self.comm = MPI_mod.COMM_WORLD
            
            # Must use custom_comm for reduce commands! This is
            # more scalable, see reductions.py for more details
            from reductions import Intracomm
            self.custom_comm = Intracomm(self.comm)
            
            # To adjust number of procs, use submission script/mpiexec
            self._numProcs = self.comm.Get_size()          
            self._rank = self.comm.Get_rank()
            if self._numProcs > 1:
                self.parallel = True
            else:
                self.parallel = False
        except ImportError:
            self._numProcs=1
            self._rank=0
            self.comm = None
            self.parallel=False
    
    
    def sync(self):
        """Forces all processors to synchronize.
        
        Method computes simple formula based on ranks of each proc, then
        asserts that results make sense and each proc reported back. This
        forces all processors to wait for others to "catch up"
        It is self-testing and for now does not need a unittest."""
        if self.parallel:
            self.comm.Barrier()
    
    def isRankZero(self):
        """Returns True if rank is zero, false if not, useful for prints"""
        if self._rank == 0:
            return True
        else:
            return False
    def isParallel(self):
        """Returns true if in parallel (requires mpi4py and >1 processor)"""
        return self.parallel
        
    def getRank(self):
        """Returns the rank of this processor"""
        return self._rank
    
    def getNumProcs(self):
        """Returns the number of processors"""
        return self._numProcs

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
        for assignment in taskProcAssignments:
            if len(assignment)==0 and self.isRankZero() and self.verbose:
                print 'Warning - at least one processor has no tasks'
        return taskProcAssignments
    
    # CURRENTLY THIS FUNCTION DOESNT WORK
    def evaluate_and_bcast(self,outputs, function, arguments=[], keywords={}):
        """
        Evaluates function with inputs and broadcasts outputs to procs
        
        outputs must be a list
        function must be a callable function given the arguments and keywords
        arguments is a list containing required arguments to "function"
        keywords is a dictionary containing optional keywords and values
        for "function"
        function is called with outputs = function(*arguments,**keywords)
        For more information, see http://docs.python.org/tutorial/controlflow.html
        section on keyword arguments, 4.7.2
        
        The result is then broadcast to all processors if in parallel.
        """
        raise RuntimeError('function isnt fully implemented')
        print 'outputs are ',outputs
        print 'function is',function
        print 'arguments are',arguments
        print 'keywords are',keywords
        if self.isRankZero():
            print function(*arguments,**keywords)
            outputList = function(*arguments,**keywords)
            if not isinstance(outputList,tuple):
                outputList = (outputList)
            if len(outputList) != len(outputs):
                raise ValueError('Length of outputs differ')
                
            for i in range(len(outputs)):
                temp = outputs[i]
                temp = outputList[i]

            print 'outputList is',outputList
            print 'outputs is',outputs
        """    
        else:
            for outputNum in range(len(outputs)):
                outputs[outputNum] = None
        if self.isParallel():
            for outputNum in range(len(outputs)):
                outputs[outputNum] = self.comm.bcast(outputs[outputNum], root=0)
        """
        print 'Done broadcasting'
        
        
    def __eq__(self, other):
        a = (self._numProcs == other.getNumProcs() and \
        self._rank == other.getRank() and self.parallel == other.isParallel())
        #print self._numProcs == other.getNumProcs() ,\
        #self._rank == other.getRank() ,self.parallel == other.isParallel()
        return a
    def __ne__(self,other):
        return not (self.__eq__(other))
    def __add__(self,other):
        print 'Adding MPI objects doesnt make sense, returning original'
        return self
    
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
    list3=[]
    for i in xrange(len(list1)):
        list3.append(list1[i]+list2[i])
    return list3


