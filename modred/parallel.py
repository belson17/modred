"""Parallel class and functions for distributed memory"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future.builtins import range
from future.builtins import object
import socket

import numpy as np


# Check to see if MPI is available by importing MPI-related modules
try:
    from mpi4py import MPI
    from .reductions import Intracomm
    _MPI_avail = True
except ImportError:
    _MPI_avail = False

# Determine host name
_hostname = socket.gethostname()
_node_ID = hash(_hostname)

# If MPI is available, gather MPI data
if _MPI_avail:

    # Determine number of nodes
    comm = MPI.COMM_WORLD
    _num_nodes = len(set(comm.allgather(_node_ID)))

    # Must use custom_comm for reduce commands! This is
    # more scalable, see reductions.py for more details
    custom_comm = Intracomm(comm)

    # To adjust number of procs, use submission script/mpiexec
    _num_MPI_workers = comm.Get_size()
    _rank = comm.Get_rank()
    if _num_MPI_workers > 1:
        _is_distributed = True
    else:
        _is_distributed = False
else:
    _num_nodes = 1
    _num_MPI_workers = 1
    _rank = 0
    _is_distributed = False
    comm = None
    custom_comm = None


def get_hostname():
    """Returns hostname for this node."""
    return _hostname


def get_node_ID():
    """Returns unique ID number for this node."""
    return _node_ID


def get_num_nodes():
    """Returns number of nodes."""
    return _num_nodes


def get_num_MPI_workers():
    """Returns number of MPI workers (currently same as number of
    processors)."""
    return _num_MPI_workers


def get_rank():
    """Returns rank of this processor/MPI worker."""
    return _rank


def get_num_procs():
    """Returns number of processors (currently same as number of MPI
    workers)."""
    return get_num_MPI_workers()


def is_distributed():
    """Returns True if there is more than one processor/MPI worker and mpi4py
    was imported properly."""
    return _is_distributed


def is_rank_zero():
    """Returns True if rank is zero, False if not."""
    return _rank == 0


def barrier():
    """Wrapper for Barrier(); forces all processors/MPI workers to
    synchronize."""
    if _is_distributed:
        comm.Barrier()


def print_from_rank_zero(msgs):
    """Prints ``msgs`` from rank zero processor/MPI worker only."""
    if is_rank_zero():
        print(msg)


def call_from_rank_zero(func, *args, **kwargs):
    """Calls function from rank zero processor/MPI worker, does not call
    ``barrier()``.

    Args:
        ``func``: Function to call.

        ``*args``: Required arguments for ``func``.

        ``**kwargs``: Keyword args for ``func``.

    Usage::

      parallel.call_from_rank_zero(lambda x: x+1, 1)

    """
    if is_rank_zero():
        out = func(*args, **kwargs)
    else:
        out = None
    return out


def bcast(vals):
    """Broadcasts values from rank zero processor/MPI worker to all others.

    Args:
        ``vals``: Values to broadcast from rank zero processor/MPI worker.

    Returns:
        ``outputs``: Broadcasted values
    """
    if is_rank_zero():
        outputs = vals
    else:
        outputs = None
    if _is_distributed:
        outputs = comm.bcast(outputs, root=0)
    return outputs


def call_and_bcast(func, *args, **kwargs):
    """Calls function on rank zero processor/MPI worker and broadcasts
    outputs to all others.

    Args:
        ``func``: A callable that takes ``*args`` and ``**kwargs``

        ``*args``: Required arguments for ``func``.

        ``**kwargs``: Keyword args for ``func``.

    Usage::

      # Adds one to the rank, but only evaluated on rank 0, so
      # ``outputs==1`` on all processors/MPI workers.
      outputs = parallel.call_and_bcast(lambda x: x+1, parallel.get_rank())

    """
    if is_rank_zero():
        outputs = func(*args, **kwargs)
    else:
        outputs = None
    if _is_distributed:
        outputs = comm.bcast(outputs, root=0)
    return outputs


def find_assignments(tasks, task_weights=None):
    """Evenly distributes tasks among all processors/MPI workers using task
    weights.

    Args:
        ``tasks``: List of tasks.  A "task" can be any object that
        corresponds to a set of operations that needs to be completed. For
        example ``tasks`` could be a list of indices, telling each
        processor/MPI worker which indices of an array to operate on.

    Kwargs:
        ``task_weights``: List of weights for each task.  These are used to
        equally distribute the workload among processors/MPI workers, in
        case some tasks are more expensive than others.

    Returns:
        ``task_assignments``: 2D list of tasks, with indices corresponding
        to [rank][task_index].  Each processor/MPI worker is responsible
        for ``task_assignments[rank]``
    """
    task_assignments = []

    # If no weights are given, assume each task has uniform weight
    if task_weights is None:
        task_weights = np.ones(len(tasks))
    else:
        task_weights = np.array(task_weights)

    first_unassigned_index = 0

    for worker_num in range(_num_MPI_workers):
        # amount of work to do, float (scaled by weights)
        work_remaining = sum(task_weights[first_unassigned_index:])

        # Number of MPI workers whose jobs have not yet been assigned
        num_remaining_workers = _num_MPI_workers - worker_num

        # Distribute work load evenly across workers
        work_per_worker = (1. * work_remaining) / num_remaining_workers

        # If task list is not empty, compute assignments
        if task_weights[first_unassigned_index:].size != 0:
            # Index of tasks element which has sum(tasks[:ind])
            # closest to work_per_worker
            new_max_task_index = np.abs(np.cumsum(
                task_weights[first_unassigned_index:]) -\
                work_per_worker).argmin() + first_unassigned_index
            # Append all tasks up to and including new_max_task_index
            task_assignments.append(tasks[first_unassigned_index:\
                new_max_task_index+1])
            first_unassigned_index = new_max_task_index+1
        else:
            task_assignments.append([])

    return task_assignments


def check_for_empty_tasks(task_assignments):
    """Convenience function that checks for empty processor/MPI worker
    assignments.

    Args:
        ``task_assignments``: List of task assignments.

    Returns:
        ``empty_tasks``: ``True`` if any processor/MPI worker has no tasks,
        otherwise ``False``.
    """
    empty_tasks = False
    for assignment in task_assignments:
        if len(assignment) == 0 and not empty_tasks:
            empty_tasks = True
    return empty_tasks
