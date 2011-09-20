
import os
import unittest
import copy
import numpy as N
import parallel as parallel_mod


parallel = parallel_mod.parallelInstance

if parallel.isRankZero():
    print 'To test fully, remember to do both:'
    print '    1) python testpod.py'
    print '    2) mpiexec -n <# procs> python testpod.py\n'

class TestParallel(unittest.TestCase):

    def setUp(self):
        try:
            from mpi4py import MPI
            self.comm=MPI.COMM_WORLD
            self.numMPIWorkers = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        except ImportError:
            self.numProcs=1
            self.rank=0
        self.myParallel=parallel_mod.Parallel()
        #if not os.path.isdir('testfiles'):
        #    SP.call(['mkdir','testfiles'])
    
    def test_sync(self):
        """
        Test that can properly synchronize processors when in parallel
        """
        #not sure how to test this
        
        
    def test_init(self):
        """Test that the MPI object uses arguments correctly.
        """
        self.assertEqual(self.myParallel._numMPIWorkers,self.numMPIWorkers)
        self.assertEqual(self.myParallel._rank,self.rank)
                        
    def test_find_assignments(self):
        """Tests that the correct processor assignments are determined
        
        Given a list of tasks, it tests
        that the correct assignment list is returned. Rather than requiring
        the testutil.py script to be run with many different numbers of procs,
        the behavior of this function is mimicked by manually setting numProcs.
        This should not be done by a user!
        """
        # Assume each item in task list has equal weight
        taskList = ['1', '2', '4', '3', '6', '7', '5']
        copyTaskList = copy.deepcopy(taskList)
        self.myParallel._numMPIWorkers = 5
        correctAssignments = [['1'], ['2'], ['4', '3'], ['6'], ['7', '5']]
        self.assertEqual(self.myParallel.find_assignments(taskList), 
            correctAssignments)
        # Check that the original list is not modified.
        self.assertEqual(taskList, copyTaskList)
         
        taskList = [3, 4, 1, 5]
        self.myParallel._numMPIWorkers = 2
        correctAssignments=[[3, 4], [1, 5]]
        self.assertEqual(self.myParallel.find_assignments(taskList),
            correctAssignments)
       
        # Allow for uneven weighting of items in task list
        taskList = ['1', '2', '4', '3', '6', '7', '5']
        taskWeights = [1, 3, 2, 3, 3, 2, 1]
        self.myParallel._numMPIWorkers = 5
        correctAssignments = [['1','2'], ['4'], ['3'], ['6'], ['7', '5']]
        self.assertEqual(self.myParallel.find_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
        
        # At first, each proc tries to take a task weight load of 2.  This is
        # closer to 0 than it is to 5, but the first assignment should be [3],
        # not []
        taskList = [3, 4, 2, 6, 1]
        taskWeights = [5, 0.25, 1.75, 0.5, 0.5]
        self.myParallel._numMPIWorkers = 4
        correctAssignments = [[3], [4], [2], [6, 1]]
        self.assertEqual(self.myParallel.find_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
       
        # Due to the highly uneven task weighting, the first proc will take up
        # the first 3 tasks, leaving none for the last processor
        taskList = ['a', 4, (2, 1), 4.3]
        taskWeights = [.1, .1, .1, .7]
        copyTaskWeights = copy.deepcopy(taskWeights)
        self.myParallel._numMPIWorkers = 3
        correctAssignments = [['a', 4, (2, 1)], [4.3], []]
        self.assertEqual(self.myParallel.find_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
        self.assertEqual(taskWeights, copyTaskWeights)

    @unittest.skip('Currently function isnt completed or used')
    def test_evaluate_and_bcast(self):
        """Test that can evaluate a function and broadcast to all procs"""
        def myAdd(a,b):
            return a,b
        class ThisClass(object):
            def __init__(self):
                self.a=0
                self.b=0
        
        myClass=ThisClass()
        d = (myClass.a,myClass.b)
        self.myParallel.evaluate_and_bcast(d,myAdd,arguments=[1,2])
        print myClass.a,myClass.b
        self.assertEqual((1,2),(myClass.a,myClass.b))

if __name__=='__main__':
    unittest.main(verbosity=2)

