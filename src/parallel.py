"""Parallel class and functions for distributed memory"""
import os
import numpy as N

class ParallelError(Exception):
    """Parallel related errors"""
    pass
    
class Parallel(object):
    """For parallelization with mpi4py.
    
    It ensures no failure in case mpi4py is not installed or when running in
    serial.    
    It is best to use the given instance, parallel.parallel_default_instance.
    """
    # TODO: Could be extended for shared memory.
    def __init__(self):
        """Constructor, tries to import mpi4py and reductions."""
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            
            self._node_ID = self.find_node_ID()
            node_IDs = self.comm.allgather(self._node_ID)
            node_IDs_no_duplicates = []
            for ID in node_IDs:
                if not (ID in node_IDs_no_duplicates):
                    node_IDs_no_duplicates.append(ID)
                
            self._num_nodes = len(node_IDs_no_duplicates)
            
            # Must use custom_comm for reduce commands! This is
            # more scalable, see reductions.py for more details
            from reductions import Intracomm
            self.custom_comm = Intracomm(self.comm)
            
            # To adjust number of procs, use submission script/mpiexec
            self._num_MPI_workers = self.comm.Get_size()          
            self._rank = self.comm.Get_rank()
            if self._num_MPI_workers > 1:
                self.distributed = True
            else:
                self.distributed = False
        except ImportError:
            self._num_nodes = 1
            self._num_MPI_workers = 1
            self._rank = 0
            self.comm = None
            self.distributed = False
    
    @staticmethod
    def find_node_ID():
        """Finds a unique ID number for each node."""
        hostname = os.uname()[1]
        return hash(hostname)
    
    def get_num_nodes(self):
        """Return the number of nodes."""
        return self._num_nodes
    
    def print_from_rank_zero(self, msgs):
        """Prints the string ``msgs`` from rank=0 only."""
        if self.is_rank_zero():
            print msg
    
    def barrier(self):
        """Forces all processors to synchronize. Wrapper for Barrier()."""
        if self.distributed:
            self.comm.Barrier()
    
    def is_rank_zero(self):
        """Returns True if rank is zero, False if not."""
        return self._rank == 0
            
    def call_from_rank_zero(self, func, *args, **kwargs):
        """Calls function from rank zero, does not call ``barrier()``.
        
        Args:
            ``func``: Function to call.
            
            ``*args``: All required arguments to ``func``.
            
            ``**kwargs``: All keyword (optional) arguments to ``func``.
        
        Usage::
        
          parallel.call_from_rank_zero(lambda x: x+1, 1)
        
        """
        if self.is_rank_zero():
            out = func(*args, **kwargs)
        else:
            out = None
        return out
        
    def is_distributed(self):
        """Returns True if >1 processor and mpi4py imported properly."""
        return self.distributed
        
    def get_rank(self):
        """Returns the rank of this processor."""
        return self._rank
    
    def get_num_MPI_workers(self):
        """Returns number of MPI workers, currently same as ``num_procs``."""
        return self._num_MPI_workers
    
    def get_num_procs(self):
        """Returns the number of processors."""
        return self.get_num_MPI_workers()

    
    def find_assignments(self, tasks, task_weights=None):
        """Evenly distributes the tasks by task weights among all MPI workers.
        
        Args:
            ``tasks``: List of "tasks".
            Tasks can be any object that corresponds to
            a task that needs to be completed, for example an index.
    
        Kwargs:
            ``task_weights``: List of weights for each task. 
            These are used to equally distribute the work load among MPI 
            workers.
       
        Returns:
            ``task_assignments``: 2D list of tasks, indices [rank][task_index].
            Each MPI worker is responsible for tasks ``task_assignments[rank]``
        """
        task_assignments = []
        
        # If no weights are given, assume each task has uniform weight
        if task_weights is None:
            task_weights = N.ones(len(tasks))
        else:
            task_weights = N.array(task_weights)
        
        first_unassigned_index = 0

        for worker_num in range(self._num_MPI_workers):
            # amount of work to do, float (scaled by weights)
            work_remaining = sum(task_weights[first_unassigned_index:]) 

            # Number of MPI workers whose jobs have not yet been assigned
            num_remaining_workers = self._num_MPI_workers - worker_num

            # Distribute work load evenly across workers
            work_per_worker = (1. * work_remaining) / num_remaining_workers

            # If task list is not empty, compute assignments
            if task_weights[first_unassigned_index:].size != 0:
                # Index of tasks element which has sum(tasks[:ind]) 
                # closest to work_per_worker
                new_max_task_index = N.abs(N.cumsum(
                    task_weights[first_unassigned_index:]) -\
                    work_per_worker).argmin() + first_unassigned_index
                # Append all tasks up to and including new_max_task_index
                task_assignments.append(tasks[first_unassigned_index:\
                    new_max_task_index+1])
                first_unassigned_index = new_max_task_index+1
            else:
                task_assignments.append([])
                
        return task_assignments
        
        

    def check_for_empty_tasks(self, task_assignments):
        """Convenience function that checks if empty worker assignments.
        
        Args:
            ``task_assignments``: List of task assignments.
        
        Returns:
            ``empty_tasks``: ``True`` if no tasks, else ``False``.
        """
        empty_tasks = False
        for assignment in task_assignments:
            if len(assignment) == 0 and not empty_tasks:
                empty_tasks = True
        return empty_tasks

    
    def call_and_bcast(self, func, *args, **kwargs):
        """Calls function on rank 0 and broadcasts outputs to all workers.
               
        Args:
            ``func``: A callable that takes ``*args`` and ``**kwargs``
    
            ``*args``: Required arguments for ``func``.
    
            ``**kwargs``: Optional keyword args for ``func``.
        
        Usage::
          
          # Adds one to the rank, but only evaluated on rank 0, so
          # ``outputs==1`` on all workers.
          outputs = parallel.call_and_bcast(lambda x: x+1, parallel.get_rank())
            
        """
        if self.is_rank_zero():
            outputs = func(*args, **kwargs)
        else:
            outputs = None
        if self.is_distributed():
            outputs = self.comm.bcast(outputs, root=0)
        return outputs
        
    def __eq__(self, other):
        equal = (self._num_MPI_workers == other.get_num_MPI_workers() and \
        self._rank == other.get_rank() and \
        self.distributed == other.is_distributed())
        #print self._numProcs == other.getNumProcs() ,\
        #self._rank == other.getRank() ,self.parallel == other.isParallel()
        return equal
    def __ne__(self, other):
        return not (self.__eq__(other))
        
        
# Default instance to be used everywhere, "singleton"
parallel_default_instance = Parallel()     
