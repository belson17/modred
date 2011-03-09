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
        self.myMPI=util.MPI(numProcs=self.numProcs)
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
        
        #Test that it is possible to use fewer CPUs than available
        if parallel and self.numProcs>1:
            mpiChangeCPUs = util.MPI(numProcs=self.numProcs-1)
            self.assertEqual(mpiChangeCPUs._numProcs,self.numProcs-1)
        
        #Test that non-sensible values of CPUs are defaulted to num available.
        mpiZeroCPUs = util.MPI(numProcs=0)
        self.assertEqual(mpiZeroCPUs._numProcs,self.numProcs)
        self.assertRaises(util.MPIError,util.MPI,(self.numProcs+1))        
        
        
    def test_MPI_find_proc_assignments(self):
        """Tests that the correct processor assignments are determined
        
        Given a list of tasks, it tests
        that the correct assignment list is returned. Rather than requiring
        the testutil.py script to be run with many different numbers of procs,
        the behavior of this function is mimicked by manually setting numProcs.
        This should not be done by a user!
        """
        
        taskList = ['1','2','4','3','6','7','5']
        numTasks = len(taskList)
        numProcs = 5
        self.myMPI._numProcs=numProcs
        correctAssignments=[['1','2'],['4','3'],['6'],['7'],['5']]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList),
          correctAssignments)
          
        taskList = [3,4,1,5]
        numTasks = len(taskList)
        numProcs = 2
        self.myMPI._numProcs=numProcs
        correctAssignments=[[3,4],[1,5]]
        self.assertEqual(self.myMPI.find_proc_assignments(taskList),
          correctAssignments)
        
        #more tests?

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


