#!/usr/local/bin/env python

import numpy as N
import util
import unittest
import subprocess as SP
import sys
import os
try: 
    from mpi4py import MPI
    parallel = MPI.COMM_WORLD.Get_size() > 1
except ImportError:
    print 'WARNING - no mpi4py module, only default serial behavior tested'
    parallel = False
    
print 'To fully test, must do both:'
print ' 1) python testutil.py'
print ' 2) mpiexec -n <# procs> python testutil.py'

class TestUtil(unittest.TestCase):
    """Tests all of the functions in util.py
    
    To test all parallel features, use "mpiexec -n 2 python testutil.py"
    Some parallel features are tested even when running in serial.
    If you run this test with mpiexec, mpi4py MUST be installed or you will
    receive meaningless errors. 
    """
    
    def setUp(self):
        try:
            from mpi4py import MPI
            self.comm=MPI.COMM_WORLD
            self.numProcs = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        except ImportError:
            self.numProcs=1
            self.rank=0
        self.myMPI=util.MPI(verbose=False)
        if not os.path.isdir('testfiles'):
            SP.call(['mkdir','testfiles'])
        
    @unittest.skipIf(parallel,'Only save/load matrices in serial')
    def test_load_save_mat_text(self):
        """Test that can read/write text matrices"""
        tol = 8
        maxNumRows = 20
        maxNumCols = 8
        matPath = 'testMatrix.txt'
        delimiters = [',',' ',';']
        for delimiter in delimiters:
            for numRows in range(1,maxNumRows):
                for numCols in range(1,maxNumCols):
                    mat=N.random.random((numRows,numCols))
                    util.save_mat_text(mat,matPath,delimiter=delimiter)
                    matRead = util.load_mat_text(matPath,delimiter=delimiter)
                    N.testing.assert_array_almost_equal(mat,matRead,
                      decimal=tol)
        SP.call(['rm',matPath])
        
    def test_MPI_sync(self):
        """
        Test that can properly synchronize processors when in parallel
        """
        #not sure how to test this
        
        
    def test_MPI_init(self):
        """Test that the MPI object uses arguments correctly.
        """
        
        self.assertEqual(self.myMPI._numProcs,self.numProcs)
        self.assertEqual(self.myMPI._rank,self.rank)
                        
    def test_MPI_find_proc_assignments(self):
        """Tests that the correct processor assignments are determined
        
        Given a list of tasks, it tests
        that the correct assignment list is returned. Rather than requiring
        the testutil.py script to be run with many different numbers of procs,
        the behavior of this function is mimicked by manually setting numProcs.
        This should not be done by a user!
        """
    
        # Assume each item in task list has equal weight
        taskList = ['1', '2', '4', '3', '6', '7', '5']
        self.myMPI._numProcs = 5
        correctAssignments = [['1'], ['2'], ['4', '3'], ['6'], ['7', '5']]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList), 
            correctAssignments)
         
        taskList = [3, 4, 1, 5]
        self.myMPI._numProcs = 2
        correctAssignments=[[3, 4], [1, 5]]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList),
            correctAssignments)
       
        # Allow for uneven weighting of items in task list
        taskList = ['1', '2', '4', '3', '6', '7', '5']
        taskWeights = [1, 3, 2, 3, 3, 2, 1]
        self.myMPI._numProcs = 5
        correctAssignments = [['1','2'], ['4'], ['3'], ['6'], ['7', '5']]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
        
        # At first, each proc tries to take a task weight load of 2.  This is
        # closer to 0 than it is to 5, but the first assignment should be [3],
        # not []
        taskList = [3, 4, 2, 6, 1]
        taskWeights = [5, 0.25, 1.75, 0.5, 0.5]
        self.myMPI._numProcs = 4
        correctAssignments = [[3], [4], [2], [6, 1]]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
       
        # Due to the highly uneven task weighting, the first proc will take up
        # the first 3 tasks, leaving none for the last processor
        taskList = ['a', 4, (2, 1), 4.3]
        taskWeights = [.1, .1, .1, .7]
        self.myMPI._numProcs = 3
        correctAssignments = [['a', 4, (2, 1)], [4.3], []]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
       

    @unittest.skip('Currently function isnt completed or used')
    def test_MPI_evaluate_and_bcast(self):
        """Test that can evaluate a function and broadcast to all procs"""
        
        def myAdd(a,b):
            return a,b
        class ThisClass(object):
            def __init__(self):
                self.a=0
                self.b=0
        
        myClass=ThisClass()
        d = (myClass.a,myClass.b)
        self.myMPI.evaluate_and_bcast(d,myAdd,arguments=[1,2])
        print myClass.a,myClass.b
        self.assertEqual((1,2),(myClass.a,myClass.b))
               
        
    def test_svd(self):
        numInternalList = [10,50]
        numRowsList = [3,5,40]
        numColsList = [1,9,70]
        for numRows in numRowsList:
            for numCols in numColsList:
                for numInternal in numInternalList:
                    leftMat = N.mat(N.random.random((numRows,numInternal)))
                    rightMat = N.mat(N.random.random((numInternal,numCols)))
                    A = leftMat*rightMat
                    [LSingVecs,singVals,RSingVecs]=util.svd(A)
                    
                    U,E,Vstar=N.linalg.svd(A,full_matrices=0)
                    V = N.mat(Vstar).H
                    if numInternal < numRows or numInternal <numCols:
                        U=U[:,:numInternal]
                        V=V[:,:numInternal]
                        E=E[:numInternal]
          
                    N.testing.assert_array_almost_equal(LSingVecs,U)
                    N.testing.assert_array_almost_equal(singVals,E)
                    N.testing.assert_array_almost_equal(RSingVecs,V)
    
    
if __name__=='__main__':
    unittest.main(verbosity=2)


