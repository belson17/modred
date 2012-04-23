#!/usr/bin/env python

import unittest
import copy
import numpy as N
import os

import helper
helper.add_to_path('src')
    
import parallel as parallel_mod
parallel = parallel_mod.parallel_default_instance

try: 
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    distributed = MPI.COMM_WORLD.Get_size() > 1
except ImportError:
    print 'Warning: without mpi4py module, only serial behavior is tested'
    distributed = False
    rank = 0


class TestParallel(unittest.TestCase):

    def setUp(self):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.num_MPI_workers = comm.Get_size()
            self.rank = comm.Get_rank()
        except ImportError:
            self.num_MPI_workers = 1
            self.rank = 0
        self.my_parallel = parallel_mod.Parallel()
        
        
    def tearDown(self):
        if distributed:
            MPI.COMM_WORLD.Barrier()
        
        
    def test_sync(self):
        """
        Test that can properly synchronize processors when in parallel
        """
        pass
        # not sure how to test this
        
        
    def test_init(self):
        """Test that the MPI object uses arguments correctly.
        """
        self.assertEqual(self.my_parallel._num_MPI_workers, self.num_MPI_workers)
        self.assertEqual(self.my_parallel._rank, self.rank)
             
             
    def test_find_assignments(self):
        """Tests that the correct processor assignments are determined
        
        Given a list of tasks, it tests that the correct assignment list is 
        returned. Rather than requiring the testutil.py script to be run 
        with many different numbers of procs the behavior of this function
        is mimicked by manually setting num_MPI_workers
        """
        # Assume each item in task list has equal weight
        tasks = ['1', '2', '4', '3', '6', '7', '5']
        copy_task_list = copy.deepcopy(tasks)
        self.my_parallel._num_MPI_workers = 5
        correct_assignments = [['1'], ['2'], ['4', '3'], ['6'], ['7', '5']]
        self.assertEqual(self.my_parallel.find_assignments(tasks), 
            correct_assignments)
        # Check that the original list is not modified.
        self.assertEqual(tasks, copy_task_list)
        
        tasks = [3, 4, 1, 5]
        self.my_parallel._num_MPI_workers = 2
        correct_assignments = [[3, 4], [1, 5]]
        self.assertEqual(self.my_parallel.find_assignments(tasks),
            correct_assignments)
       
        # Allow for uneven weighting of items in task list
        tasks = ['1', '2', '4', '3', '6', '7', '5']
        task_weights = [1, 3, 2, 3, 3, 2, 1]
        self.my_parallel._num_MPI_workers = 5
        correct_assignments = [['1','2'], ['4'], ['3'], ['6'], ['7', '5']]
        self.assertEqual(self.my_parallel.find_assignments(tasks, 
            task_weights=task_weights), correct_assignments)
        
        # At first, each proc tries to take a task weight load of 2.  This is
        # closer to 0 than it is to 5, but the first assignment should be [3],
        # not []
        tasks = [3, 4, 2, 6, 1]
        task_weights = [5, 0.25, 1.75, 0.5, 0.5]
        self.my_parallel._num_MPI_workers = 4
        correct_assignments = [[3], [4], [2], [6, 1]]
        self.assertEqual(self.my_parallel.find_assignments(tasks, 
            task_weights=task_weights), correct_assignments)
       
        # Due to the highly uneven task weighting, the first proc will take up
        # the first 3 tasks, leaving none for the last processor
        tasks = ['a', 4, (2, 1), 4.3]
        task_weights = [.1, .1, .1, .7]
        copy_task_weights = copy.deepcopy(task_weights)
        self.my_parallel._num_MPI_workers = 3
        correct_assignments = [['a', 4, (2, 1)], [4.3], []]
        self.assertEqual(self.my_parallel.find_assignments(tasks, 
            task_weights=task_weights), correct_assignments)
        self.assertEqual(task_weights, copy_task_weights)


    @unittest.skip('Currently function isnt completed or used')
    def test_evaluate_and_bcast(self):
        """Test that can evaluate a function and broadcast to all procs"""
        def myAdd(a,b):
            return a,b
        class ThisClass(object):
            def __init__(self):
                self.a=0
                self.b=0
        
        myClass = ThisClass()
        d = (myClass.a,myClass.b)
        self.my_parallel.evaluate_and_bcast(d, myAdd, arguments=[1,2])
        print myClass.a,myClass.b
        self.assertEqual((1,2),(myClass.a,myClass.b))


if __name__=='__main__':
    unittest.main()

