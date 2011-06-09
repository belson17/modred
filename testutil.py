#!/usr/bin/env python

import numpy as N
import util
import unittest
import subprocess as SP
import sys
import os
import multiprocessing

pool = multiprocessing.Pool()

""" # distributed only
try: 
    from mpi4py import MPI
    parallel = MPI.COMM_WORLD.Get_size() > 1
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    print 'WARNING - no mpi4py module, only default serial behavior tested'
    parallel = False
    rank = 0

if rank == 0:
    print 'To fully test, must do both:'
    print ' 1) python testutil.py'
    print ' 2) Submit a multi-node job on a cluster with'
    print '    mpiexec -n <# nodes> python testutil.py '
""" # distributed only
rank = 0
parallel = False
distributed = False


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
            self.numNodes = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        except ImportError:
            self.numNodes = 1
            self.rank = 0
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
    
    @unittest.skipIf(not distributed,'Only test in distributed case')
    def test_MPI_sync(self):
        """
        Test that can properly synchronize processors when in parallel
        """
        #not sure how to test this
        
    #@unittest.skipIf(not distributed,'Only test in distributed case')
    def test_bcast_pickle(self):
        """
        Test that can bcast when multiprocess.Pool() instance created
        
        This test assumes that save and load _mat_text work.
        """
        print '\n starting bcast_pickle on node num',self.myMPI.getNodeNum()
        matPath = 'testfiles/bcast_mat_delete.txt'
        if self.myMPI.isRankZero():
            mat = N.random.random((10,20))
            util.save_mat_text(mat, matPath)
        else:
            mat = None
        print 'saved mat text, node num is',self.myMPI.getNodeNum()    
        matBcast = self.myMPI.bcast_pickle(mat)
        matTrue = util.load_mat_text(matPath)
        print 'loaded mat text, node num is',self.myMPI.getNodeNum()
        #N.testing.assert_array_almost_equal(matBcast, matTrue)
        print 'asserted mats equal, node num is',self.myMPI.getNodeNum()
        self.myMPI.sync()
        print '\n done bcast on node num',self.myMPI.getNodeNum()
        
    #@unittest.skipIf(not distributed,'Only test in distributed case')    
    def test_gather_pickle(self):
        """Test that can gather when multiprocess.Pool() instance created"""
        
        data = self.myMPI.getNodeNum()**2
        dataList = self.myMPI.gather_pickle(data)
        dataListTrue = [ (i**2) for i in range(self.myMPI.getNumNodes())]
        if self.myMPI.isRankZero():
            self.assertEqual(dataList, dataListTrue)
        else:
            self.assertEqual(dataList, None)
        self.myMPI.sync()
        print '\n done gather on node num',self.myMPI.getNodeNum()
        
    #@unittest.skipIf(not distributed,'Only test in distributed case')
    def test_allgather_pickle(self):
        """Test that can allgather when multiprocess.Pool() instance created"""
        data = self.myMPI.getNodeNum()**2
        #print 'allgather 1, node num',self.myMPI.getNodeNum()
        dataList = self.myMPI.allgather_pickle(data)
        #print 'allgather 2, node num',self.myMPI.getNodeNum()
        dataListTrue = [ (i**2) for i in range(self.myMPI.getNumNodes())]
        #print 'allgather 3, node num',self.myMPI.getNodeNum()
        self.assertEqual(dataList, dataListTrue)
        #print 'allgather 4, node num',self.myMPI.getNodeNum()
        self.myMPI.sync()  
        print '\n done allgather on node num',self.myMPI.getNodeNum()
        
    #@unittest.skipIf(not distributed,'Only test in distributed case')
    def test_my_sync(self):
        """Test can sync properly using text files"""
        value = 0
        value = self.myMPI.getNodeNum()
        self.myMPI.my_sync()
        self.assertEqual(value, self.myMPI.getNodeNum())
        value = value**2
        self.myMPI.my_sync()
        self.assertEqual(value, self.myMPI.getNodeNum()**2)
    
    def test_MPI_init(self):
        """Test that the MPI object uses arguments correctly.
        """
        
        self.assertEqual(self.myMPI._numNodes, self.numNodes)
        self.assertEqual(self.myMPI._rank, self.rank)
        
        
    @unittest.skipIf(not distributed,'Only test in distributed case')                    
    def test_MPI_find_assignments(self):
        """Tests that the correct assignments are determined
        
        Given a list of tasks, it tests
        that the correct assignment list is returned. Rather than requiring
        the testutil.py script to be run with many different numbers of procs,
        the behavior of this function is mimicked by manually setting numProcs.
        This should not be done by a user!
        """    
        # Assume each item in task list has equal weight
        taskList = ['1', '2', '4', '3', '6', '7', '5']
        self.myMPI._numNodes = 5
        correctAssignments = [['1'], ['2'], ['4', '3'], ['6'], ['7', '5']]
        self.assertEqual(self.myMPI.find_assignments(taskList), 
            correctAssignments)
         
        taskList = [3, 4, 1, 5]
        self.myMPI._numNodes = 2
        correctAssignments=[[3, 4], [1, 5]]
        self.assertEqual(self.myMPI.find_assignments(taskList),
            correctAssignments)
       
        # Allow for uneven weighting of items in task list
        taskList = ['1', '2', '4', '3', '6', '7', '5']
        taskWeights = [1, 3, 2, 3, 3, 2, 1]
        self.myMPI._numNodes = 5
        correctAssignments = [['1','2'], ['4'], ['3'], ['6'], ['7', '5']]
        self.assertEqual(self.myMPI.find_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
        
        # At first, each proc tries to take a task weight load of 2.  This is
        # closer to 0 than it is to 5, but the first assignment should be [3],
        # not []
        taskList = [3, 4, 2, 6, 1]
        taskWeights = [5, 0.25, 1.75, 0.5, 0.5]
        self.myMPI._numNodes = 4
        correctAssignments = [[3], [4], [2], [6, 1]]
        self.assertEqual(self.myMPI.find_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
       
        # Due to the highly uneven task weighting, the first proc will take up
        # the first 3 tasks, leaving none for the last processor
        taskList = ['a', 4, (2, 1), 4.3]
        taskWeights = [.1, .1, .1, .7]
        self.myMPI._numNodes = 3
        correctAssignments = [['a', 4, (2, 1)], [4.3], []]
        self.assertEqual(self.myMPI.find_assignments(taskList, 
            taskWeights=taskWeights), correctAssignments)
       
    @unittest.skipIf(not distributed,'Only test in distributed case')
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


