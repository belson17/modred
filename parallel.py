
import os
import numpy as N
import copy

class ParallelError(Exception):
    """For MPI related errors"""
    pass
    
class Parallel(object):
    """Simple container for information about how many processors there are.
    It ensures no failure in case mpi4py is not installed or running serial.
    
    Almost always one should use the given instance of this class, parallel.parallelInstance!
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        try:
            # Must call MPI module MPI_mod to avoid naming confusion with
            # the MPI class
            from mpi4py import MPI as MPI_mod
            self.comm = MPI_mod.COMM_WORLD
            
            self._nodeID = self.findNodeID()
            nodeIDs = self.comm.allgather(self._nodeID)
            nodeIDsWithoutDuplicates = []
            for ID in nodeIDs:
                if not (ID in nodeIDsWithoutDuplicates):
                    nodeIDsWithoutDuplicates.append(ID)
                
            self._numNodes = len(nodeIDsWithoutDuplicates)
            
            # Must use custom_comm for reduce commands! This is
            # more scalable, see reductions.py for more details
            from reductions import Intracomm
            self.custom_comm = Intracomm(self.comm)
            
            # To adjust number of procs, use submission script/mpiexec
            self._numMPIWorkers = self.comm.Get_size()          
            self._rank = self.comm.Get_rank()
            if self._numMPIWorkers > 1:
                self.distributed = True
            else:
                self.distributed = False
        except ImportError:
            self._numNodes = 1
            self._numMPIWorkers = 1
            self._rank = 0
            self.comm = None
            self.distributed = False
    
    def findNodeID(self):
        """
        Finds a unique ID number for each node. Taken from mpi4py emails.
        """
        hostname = os.uname()[1]
        return hash(hostname)
    
    def getNumNodes(self):
        """Return the number of nodes"""
        return self._numNodes
    
    def printRankZero(self,msgs):
        """
        Prints the elements of the list given from rank=0 only.
        
        Could be improved to better mimic if isRankZero(): print a,b,...
        """
        # If not a list, convert to list
        if not isinstance(msgs, list):
            msgs = [msgs]
            
        if self.isRankZero():
            for msg in msgs:
                print msg
    
    def sync(self):
        """Forces all processors to synchronize.
        
        Method computes simple formula based on ranks of each proc, then
        asserts that results make sense and each proc reported back. This
        forces all processors to wait for others to "catch up"
        It is self-testing and for now does not need a unittest."""
        if self.distributed:
            self.comm.Barrier()
    
    def isRankZero(self):
        """Returns True if rank is zero, false if not, useful for prints"""
        if self._rank == 0:
            return True
        else:
            return False
            
    def isDistributed(self):
        """Returns true if in parallel (requires mpi4py and >1 processor)"""
        return self.distributed
        
    def getRank(self):
        """Returns the rank of this processor"""
        return self._rank
    
    def getNumMPIWorkers(self):
        """Returns the number of MPI workers, currently same as numProcs"""
        return self._numMPIWorkers
    
    def getNumProcs(self):
        """Returns the number of processors"""
        return self.getNumMPIWorkers()

    
    def find_assignments(self, taskList, taskWeights=None):
        """ Returns a 2D list of tasks, [rank][taskIndex], 
        
        Evenly distributing the tasks in taskList, allowing for uneven
        task weights. 
        It returns a list that has numMPITasks entries. 
        Proc n is responsible for task MPITaskAssignments[n][...]
        where the 2nd dimension of the 2D list contains the tasks (whatever
        they were in the original taskList).
        """
        taskAssignments= []
        
        # If no weights are given, assume each task has uniform weight
        if taskWeights is None:
            taskWeights = N.ones(len(taskList))
        else:
            taskWeights = N.array(taskWeights)
        
        firstUnassignedIndex = 0

        for workerNum in range(self._numMPIWorkers):
            # amount of work to do, float (scaled by weights)
            workRemaining = sum(taskWeights[firstUnassignedIndex:]) 

            # Number of MPI workers whose jobs have not yet been assigned
            numRemainingWorkers = self._numMPIWorkers - workerNum

            # Distribute work load evenly across workers
            workPerWorker = 1. * workRemaining / numRemainingWorkers

            # If task list is not empty, compute assignments
            if taskWeights[firstUnassignedIndex:].size != 0:
                # Index of taskList element which has sum(taskList[:ind]) 
                # closest to workPerWorker
                newMaxTaskIndex = N.abs(N.cumsum(taskWeights[firstUnassignedIndex:]) -\
                    workPerWorker).argmin() + firstUnassignedIndex
                # Append all tasks up to and including newMaxTaskIndex
                taskAssignments.append(taskList[firstUnassignedIndex:\
                    newMaxTaskIndex+1])
                firstUnassignedIndex = newMaxTaskIndex+1
            else:
                taskAssignments.append([])
                
        return taskAssignments
        
        

    def checkEmptyTasks(self, taskAssignments):
        """Convenience function that checks if empty worker assignments"""
        emptyTasks = False
        for r, assignment in enumerate(taskMPITasksAssignments):
            if len(assignment) == 0 and not emptyTasks:
                if self.verbose and self.isRankZero():
                    print ('Warning: %d out of %d processors have no ' +\
                        'tasks') % (self._numMPITasks - r, self._numMPITasks)
                emptyTasks = True
        return emptyTasks


    def evaluate_and_bcast(self,outputs, function, arguments=[], keywords={}):
        """
        Evaluates function with inputs and broadcasts outputs to procs
        
        CURRENTLY THIS FUNCTION DOESNT WORK
    
        outputs 
          must be a list

        function
          must be a callable function given the arguments and keywords

        arguments
          a list containing required arguments to *function*

        keywords 
          a dictionary containing optional keywords and values for *function*

        function is called with outputs = function(\*arguments,\*\*keywords)

        For more information, see http://docs.python.org/tutorial/controlflow.html section on keyword arguments, 4.7.2
        
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
        a = (self._numMPIWorkers == other.getNumMPIWorkers() and \
        self._rank == other.getRank() and \
        self.distributed == other.isDistributed())
        #print self._numProcs == other.getNumProcs() ,\
        #self._rank == other.getRank() ,self.parallel == other.isParallel()
        return a
    def __ne__(self,other):
        return not (self.__eq__(other))
    def __add__(self,other):
        print 'Adding MPI objects doesnt make sense, returning original'
        return self
        
# Create an instance of the Parallel class that is used everywhere, "singleton"
parallelInstance = Parallel()
        
