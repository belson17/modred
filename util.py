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
    """
    Contains both distributed and shared memory information and methods.
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        # Distributed memory (may not be available if mpi4py not installed)
        try:
            # Must call MPI module MPI_mod to avoid naming confusion with
            # the MPI class
            from mpi4py import MPI as MPI_mod
            
            self.comm = MPI_mod.COMM_WORLD
            
            # Must use custom_comm for reduce commands! This is
            # more scalable, see reductions.py for more details
            from reductions import Intracomm
            self.custom_comm = Intracomm(self.comm)
            
            # To adjust number of nodes, use submission script/mpiexec
            self._numMPITasks = self.comm.Get_size()  
            self._rank = self.comm.Get_rank()
            if self._numMPITasks > 1:
                self.parallel = True
            else:
                self.parallel = False
        except ImportError:
            self._numMPITasks=1
            self._rank=0
            self.comm = None
            self.parallel=False
        # Shared memory, available since python 2.6
        
        self._numProcsPerNode = multiprocessing.cpu_count()
        
        # number of nodes is number of MPI tasks because submissions should always
        # have number of MPI tasks per node (sometimes loosely referred to as
        # "procs per node")  = 1.
        if self.parallel:
            nodeID = self.findNodeID()
            nodeIDList = self.comm.allgather(nodeID)
            nodeIDListWithoutDuplicates = []
            for ID in nodeIDList:
                if not (ID in nodeIDListWithoutDuplicates):
                    nodeIDListWithoutDuplicates.append(ID)
            self._numNodes = len(nodeIDListWithoutDuplicates)
        else:
            self._numNodes = 1
        
        if self._numNodes != self._numMPITasks:
            raise ValueError('The number of nodes does not match the '+\
                'number of MPI tasks. The number of nodes (from hostnames) '+\
                'is '+str(self._numNodes)+' but the number of MPI tasks is '+\
                str(self._numMPITasks)+' (from mpiexec -n <MPI tasks> '+\
                'or a submission script processors per node variable not being 1). '+\
                'You probably need to change your submission script or mpiexec')     
                
    
    def findNodeID(self):
        """
        Finds a unique ID number for each node.
        
        Taken from email with mpi4py list"""
        hostname = os.uname()[1]
        
        # xor_hash is an integer that corresponds to a unique node
        # xor_hash = 0
        #for char in hostname:
        #    xor_hash = xor_hash ^ ord(char)
        #return xor_hash
        return hash(hostname)
        
        
    def printRankZero(self,msgs):
      """
      Prints the elements of the list given from rank=0 only.
      
      Could be improved to better mimic if isRankZero(): print a,b,...
      """
      if not isintance(msgs,list):
          msgs = [msgs]
      if self.isRankZero():
          for msg in msgs:
              print msg
    
    def sync(self):
        """Forces all processors to synchronize.
        
        Method computes simple formula based on ranks of each proc, then
        asserts that results make sense and each proc reported back. This
        forces all processors to wait for others to "catch up"
        """
        if self.parallel:
            self.comm.Barrier()
    
    def isRankZero(self):
        """Returns True if rank is zero, false if not, useful for prints"""
        return self._rank == 0
            
            
    def isParallel(self):
        """Returns true if in parallel (requires mpi4py and >1 processor)"""
        return self.parallel
        
    #def getRank(self):
    #    """Returns the rank of this processor"""
    #    return self._rank
    
    def getNodeNum(self):
        """ Returns the node number, same as the rank since 1 mpi task/node"""
        return self._rank
    
    def getNumNodes(self):
        return self._numNodes
    
    def getNumProcsPerNode(self):
        return self._numProcsPerNode

    def find_assignments(self, taskList, taskWeights=None):
        """ 
        Returns a 2D list [rank][taskIndex] distributing the tasks.
        
        It returns a list that has numNodes entries. 
        Node n is responsible for taskProcAssignments[n][:]
        where the 2nd dimension of the 2D list contains the tasks (whatever
        they were in the original taskList).
        """        
        taskProcAssignments= []
        _taskList = copy.deepcopy(taskList)

        # If no weights are given, assume each task has uniform weight
        if taskWeights is None:
            _taskWeights = N.ones(len(taskList))
        else:
            _taskWeights = N.array(taskWeights)

        numTasks = sum(_taskWeights)

        # Function that searches for closes match to val in valList
        def find_closest_val(val, valArray):
            closestVal = min(abs(valArray - val))
            for testInd, testVal in enumerate(valList):
                if testVal == closestVal:
                    ind = testInd
            return closestVal, ind

        for nodeNum in range(self._numNodes):
            # Number of remaining tasks (scaled by weights)
            numRemainingTasks = sum(_taskWeights) 

            # Number of processors whose jobs have not yet been assigned
            numRemainingNodes = self._numNodes - nodeNum

            # Distribute weighted task list evenly across processors
            numTasksPerNode = 1. * numRemainingTasks / numRemainingNodes

            # If task list is not empty, compute assignments
            if _taskWeights.size != 0:
                # Index of task list element such that sum(_taskList[:ind]) 
                # comes closest to numTasksPerProc
                newMaxTaskIndex = N.abs(N.cumsum(_taskWeights) -\
                    numTasksPerNode).argmin()

                # Add all tasks up to and including newMaxTaskIndex to the
                # assignment list
                taskProcAssignments.append(_taskList[:newMaxTaskIndex + 1])

                # Remove assigned tasks, weights from list
                del _taskList[:newMaxTaskIndex + 1]
                _taskWeights = _taskWeights[newMaxTaskIndex + 1:]
            else:
                taskProcAssignments.append([])

            # Warning if some processors have no tasks
            if self.verbose and self.isRankZero():
                printedPreviously = False
                for r, assignment in enumerate(taskProcAssignments):
                    if len(assignment) == 0 and not printedPreviously:
                        print ('Warning - %d out of %d nodes have no ' +\
                            'tasks') % (self._numNodes - r, self._numNodes)
                        printedPreviously = True

        return taskProcAssignments
        
    def bcast_pickle(self, obj, root = 0, fileName = None):
        """
        Saves a pickle file for broadcast rather than use self.comm.bcast
        
        Since multiprocessing breaks the bcast function, new implementation
        saves a pickle file and loads it on each node. Inefficient but
        fool-proof. Should be called from ALL nodes. obj only needs to 
        be non-None for rank=0, and whatever obj is on root=0, it is 
        returned from bcast_pickle.
        
        Usage:
        if mpi.isRankZero():
            obj = N.random.random(4)
        else:
            obj = None
        obj = mpi.bcast_pickle(obj)
        """
        if fileName is None:
            fileName = 'bcast.pickle'
        
        if self.getNodeNum() == root:
            pickleFile = open(fileName,'wb')
            pickle.dump(obj,pickleFile)
            pickleFile.close()
            
        self.sync()
        pickleFile = open(fileName,'rb')
        obj = pickle.load(pickleFile)
        pickleFile.close()
        self.sync()
        if self.isRankZero():
            SP.call(['rm', fileName])
        return obj
        
    def gather_pickle(self, obj, root=0, fileName = None):
        """
        Saves pickle files for gathering rather than use self.comm.gather
        
        Since multiprocessing breaks the gather function, new implementation
        saves pickle files and loads all on the root node. Inefficient but
        fool-proof. Should be called from ALL nodes. obj needs to 
        be defined on all nodes. If fileName is used, must contain
        a %03d.
        
        Usage:
        obj = N.random.random(4)
        objList = mpi.gather_pickle(obj)
        # objList is only non-None on root node (default 0)
        """
        if fileName is None:
            fileName = 'gather_node_%03d.pickle'
            
        pickleFile = open(fileName % self.getNodeNum(), 'wb')
        pickle.dump(obj,pickleFile)
        pickleFile.close()
        
        self.sync()
        if self.getNodeNum() == root:
            objList = []
            for nodeNum in range(self.getNumNodes()):
                pickleFile = open(fileName % nodeNum,'rb')
                objList.append(pickle.load(pickleFile))
                pickleFile.close()
                SP.call(['rm',fileName % nodeNum]) 
            return objList
        else:
            return None
    
    def allgather_pickle(self, obj, fileName = None):
        """
        Saves pickle files for gathering rather than use self.comm.allgather
        
        Simply calls gather_pickle and bcast_pickle. See the documentation
        of these for more details.
        If fileName is used, must contain a %03d.
        
        Usage:
        obj = N.random.random(4)
        objList = mpi.allgather_pickle(obj)
        # objList is the same list of arrays on all nodes
        """
        if fileName is None:
            fileName = 'allgather_node_%03d.pickle'
        # gather obj's into objList, only non-None on rank 0
        objList = self.gather_pickle(obj)
        # bcast objList from rank 0 to all other ranks
        return self.bcast_pickle(objList)
    

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
        a = (self._numNodes == other.getNumNodes() and \
        self._rank == other.getNodeNum() and self.parallel == other.isParallel())
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


def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])
    
# Simple function for testing
def add(x,y): return x+y

# Create an instance of MPI class that is used everywhere, "singleton"
MPIInstance = MPI(verbose=True)





