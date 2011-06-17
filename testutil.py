#!/usr/bin/env python

import numpy as N
import util
import unittest
import subprocess as SP
import sys
import os
import multiprocessing

pool = multiprocessing.Pool()

class TestUtil(unittest.TestCase):
    """Tests all of the functions in util.py
    
    To test all parallel features, use "mpiexec -n 2 python testutil.py"
    Some parallel features are tested even when running in serial.
    If you run this test with mpiexec, mpi4py MUST be installed or you will
    receive meaningless errors. 
    """
    
    def setUp(self):        
        if not os.path.isdir('testfiles'):
            SP.call(['mkdir','testfiles'])
        
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
    unittest.main(verbosity=1)


