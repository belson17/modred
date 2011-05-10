#!/usr/bin/env python

import subprocess as SP
import os
import numpy as N
from bpod import BPOD
import unittest
import util
import copy

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    parallel = comm.Get_size() >=2
    rank = comm.Get_rank()
    numProcs = comm.Get_size()
except ImportError:
    parallel = False
    rank = 0
    numProcs = 1

print 'To test fully, remember to do both:'
print '    1) python testbpod.py'
print '    2) mpiexec -n <# procs> python testbpod.py\n'

class TestBPOD(unittest.TestCase):
    """ Test all the BPOD class methods """
    
    def setUp(self):
        if not os.path.isdir('files_modaldecomp_test'):        
            SP.call(['mkdir','files_modaldecomp_test'])
        self.modeNumList =[2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.numDirectSnaps = 40
        self.numAdjointSnaps = 45
        self.numStates = 100
        self.indexFrom = 2
        self.bpod = BPOD(save_mat=util.save_mat_text, load_field=util.\
            load_mat_text, inner_product=util.inner_product, save_field=util.\
            save_mat_text, verbose=False)
        self.generate_data_set()
   
        # Default data members for constructor test
        self.defaultMPI = util.MPI()
        self.defaultDataMembers = {'load_field': None, 'save_field': None, 
            'save_mat': util.save_mat_text, 'load_mat': util.load_mat_text, 
            'inner_product': None, 'maxFieldsPerNode': 2, 'maxFieldsPerProc': 2,
            'mpi': self.defaultMPI, 'numNodes': 1, 'directSnapPaths': None, 
            'adjointSnapPaths': None, 'LSingVecs': None, 'singVals': None,
            'RSingVecs': None, 'hankelMat': None, 'verbose': False}

    def tearDown(self):
        self.bpod.mpi.sync()
        if self.bpod.mpi.isRankZero():
            SP.call(['rm -rf files_modaldecomp_test/*'], shell=True)
        self.bpod.mpi.sync()
    
    def generate_data_set(self):
        #create data set (saved to file)
        self.directSnapPath = 'files_modaldecomp_test/direct_snap_%03d.txt'
        self.adjointSnapPath = 'files_modaldecomp_test/adjoint_snap_%03d.txt'

        self.directSnapPaths=[]
        self.adjointSnapPaths=[]
        
        if self.bpod.mpi.isRankZero():
            self.directSnapMat = N.mat(N.random.random((self.numStates,self.\
                numDirectSnaps)))
            self.adjointSnapMat = N.mat(N.random.random((self.numStates,self.\
                numAdjointSnaps))) 
            
            for directSnapIndex in range(self.numDirectSnaps):
                util.save_mat_text(self.directSnapMat[:,directSnapIndex],self.\
                    directSnapPath%directSnapIndex)
                self.directSnapPaths.append(self.directSnapPath%directSnapIndex)
            for adjointSnapIndex in range(self.numAdjointSnaps):
                util.save_mat_text(self.adjointSnapMat[:,adjointSnapIndex],
                  self.adjointSnapPath%adjointSnapIndex)
                self.adjointSnapPaths.append(self.adjointSnapPath%\
                    adjointSnapIndex)
        else:
            self.directSnapPaths=None
            self.adjointSnapPaths=None
            self.directSnapMat = None
            self.adjointSnapMat = None
        if self.bpod.mpi.isParallel():
            self.directSnapPaths = self.bpod.mpi.comm.bcast(
              self.directSnapPaths,root=0)
            self.adjointSnapPaths = self.bpod.mpi.comm.bcast(
              self.adjointSnapPaths,root=0)
            self.directSnapMat = self.bpod.mpi.comm.bcast(
              self.directSnapMat,root=0)
            self.adjointSnapMat = self.bpod.mpi.comm.bcast(
              self.adjointSnapMat,root=0)
         
        self.hankelMatTrue=self.adjointSnapMat.T*self.directSnapMat
        
        #Do the SVD on all procs.
        self.LSingVecsTrue,self.singValsTrue,self.RSingVecsTrue=\
          util.svd(self.hankelMatTrue)
        
        self.directModeMat = self.directSnapMat * N.mat(self.RSingVecsTrue)*\
          N.mat(N.diag(self.singValsTrue**(-0.5)))

        self.adjointModeMat = self.adjointSnapMat*N.mat(self.LSingVecsTrue)*\
          N.mat(N.diag(self.singValsTrue**(-0.5)))
        
        #self.bpod.directSnapPaths=self.directSnapPaths
        #self.bpod.adjointSnapPaths=self.adjointSnapPaths
        
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        dataMembersOriginal = util.get_data_members(BPOD(verbose=False))
        self.assertEqual(dataMembersOriginal, self.defaultDataMembers)

        def my_load(fname): pass
        myBPOD = BPOD(load_field=my_load, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['load_field'] = my_load
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        
        def my_save(data,fname): pass 
        myBPOD = BPOD(save_field=my_save, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['save_field'] = my_save
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        self.assertEqual(myBPOD.save_field,my_save)
        
        myBPOD = BPOD(save_mat=my_save, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['save_mat'] = my_save
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        self.assertEqual(myBPOD.save_mat,my_save)
        
        def my_ip(f1,f2): pass
        myBPOD = BPOD(inner_product=my_ip, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['inner_product'] = my_ip
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        self.assertEqual(myBPOD.inner_product,my_ip)
                
        directSnapPaths=['a','b']
        myBPOD = BPOD(directSnapPaths = directSnapPaths, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['directSnapPaths'] = directSnapPaths
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        self.assertEqual(myBPOD.directSnapPaths,directSnapPaths)
        
        adjointSnapPaths=['a','c']
        myBPOD = BPOD(adjointSnapPaths = adjointSnapPaths, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['adjointSnapPaths'] = adjointSnapPaths
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        self.assertEqual(myBPOD.adjointSnapPaths,adjointSnapPaths)
        
        LSingVecs=N.mat(N.random.random((2,3)))
        myBPOD = BPOD(LSingVecs = LSingVecs, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['LSingVecs'] = LSingVecs
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        N.testing.assert_array_almost_equal(myBPOD.LSingVecs,LSingVecs)
        
        RSingVecs = N.mat(N.random.random((3,2)))
        myBPOD = BPOD(RSingVecs = RSingVecs, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['RSingVecs'] = RSingVecs
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        N.testing.assert_array_almost_equal(myBPOD.RSingVecs,RSingVecs)
        
        singVals = N.mat(N.random.random((2,2)))
        myBPOD = BPOD(singVals = singVals, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['singVals'] = singVals
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        N.testing.assert_array_almost_equal(myBPOD.singVals,singVals)
        
        hankelMat = N.mat(N.random.random((4,4)))
        myBPOD = BPOD(hankelMat = hankelMat, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['hankelMat'] = hankelMat
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        N.testing.assert_array_almost_equal(myBPOD.hankelMat,hankelMat)
                
        maxFieldsPerNode = 500
        myBPOD = BPOD(maxFieldsPerNode=maxFieldsPerNode, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['maxFieldsPerNode'] = maxFieldsPerNode
        dataMembers['maxFieldsPerProc'] = maxFieldsPerNode * myBPOD.numNodes /\
            myBPOD.mpi.getNumProcs() / myBPOD.numNodes
        self.assertEqual(util.get_data_members(myBPOD), dataMembers)
        
        self.assertRaises(util.MPIError, BPOD, numNodes=numProcs+1, verbose=\
            False)
        
    def test_compute_decomp(self):
        """
        Test that can take snapshots, compute the Hankel and SVD matrices
        
        With previously generated random snapshots, compute the Hankel
        matrix, then take the SVD. The computed matrices are saved, then
        loaded and compared to the true matrices. 
        """
        tol = 8
        directSnapPath = 'files_modaldecomp_test/direct_snap_%03d.txt'
        adjointSnapPath = 'files_modaldecomp_test/adjoint_snap_%03d.txt'
        LSingVecsPath ='files_modaldecomp_test/lsingvecs.txt'
        RSingVecsPath ='files_modaldecomp_test/rsingvecs.txt'
        singValsPath ='files_modaldecomp_test/singvals.txt'
        hankelMatPath='files_modaldecomp_test/hankel.txt'
        
        self.bpod.compute_decomp(RSingVecsPath=RSingVecsPath,
          LSingVecsPath=LSingVecsPath,singValsPath=singValsPath,
          hankelMatPath=hankelMatPath,directSnapPaths=self.directSnapPaths,
          adjointSnapPaths=self.adjointSnapPaths)
                
        if self.bpod.mpi.isRankZero():
            LSingVecsLoaded = util.load_mat_text(LSingVecsPath)
            RSingVecsLoaded = util.load_mat_text(RSingVecsPath)
            singValsLoaded = N.squeeze(N.array(
              util.load_mat_text(singValsPath)))
            hankelMatLoaded = util.load_mat_text(hankelMatPath)
        else:
            LSingVecsLoaded=None
            RSingVecsLoaded=None
            singValsLoaded=None
            hankelMatLoaded=None

        if self.bpod.mpi.isParallel():
            LSingVecsLoaded=self.bpod.mpi.comm.bcast(LSingVecsLoaded,root=0)
            RSingVecsLoaded=self.bpod.mpi.comm.bcast(RSingVecsLoaded,root=0)
            singValsLoaded=self.bpod.mpi.comm.bcast(singValsLoaded,root=0)
            hankelMatLoaded=self.bpod.mpi.comm.bcast(hankelMatLoaded,root=0)
        
        N.testing.assert_array_almost_equal(self.bpod.hankelMat,
          self.hankelMatTrue,decimal=tol)
        N.testing.assert_array_almost_equal(self.bpod.LSingVecs,
          self.LSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(self.bpod.RSingVecs,
          self.RSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(self.bpod.singVals,
          self.singValsTrue,decimal=tol)
          
        N.testing.assert_array_almost_equal(hankelMatLoaded,
          self.hankelMatTrue,decimal=tol)
        N.testing.assert_array_almost_equal(LSingVecsLoaded,
          self.LSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(RSingVecsLoaded,
          self.RSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(singValsLoaded,
          self.singValsTrue,decimal=tol)
        

    def test_compute_modes(self):
        """
        Test computing modes in serial and parallel. 
        
        This method uses the existing random data set saved to disk. It tests
        that BPOD can generate the modes, save them, and load them, then
        compares them to the known solution.
        """

        directModePath ='files_modaldecomp_test/direct_mode_%03d.txt'
        adjointModePath ='files_modaldecomp_test/adjoint_mode_%03d.txt'
        
        # starts with the  CORRECT decomposition.
        
        self.bpod.RSingVecs=self.RSingVecsTrue
        self.bpod.LSingVecs=self.LSingVecsTrue
        self.bpod.singVals=self.singValsTrue
        
        self.bpod.compute_direct_modes(self.modeNumList,directModePath,
          indexFrom=self.indexFrom,directSnapPaths=self.directSnapPaths)
          
        self.bpod.compute_adjoint_modes(self.modeNumList,adjointModePath,
          indexFrom=self.indexFrom,adjointSnapPaths=self.adjointSnapPaths)
          
        for modeNum in self.modeNumList:
            if self.bpod.mpi.isRankZero:
                directMode = util.load_mat_text(directModePath%modeNum)
                adjointMode=util.load_mat_text(adjointModePath%modeNum)
            else:
                directMode = None
                adjointMode = None
            if self.bpod.mpi.parallel:
                directMode = self.bpod.mpi.comm.bcast(directMode,root=0)
                adjointMode = self.bpod.mpi.comm.bcast(adjointMode,root=0)
            N.testing.assert_array_almost_equal(directMode,self.directModeMat[:,
                modeNum-self.indexFrom])
            N.testing.assert_array_almost_equal(adjointMode,self.\
                adjointModeMat[:,modeNum-self.indexFrom])
        
        if self.bpod.mpi.isRankZero():
            for modeNum1 in self.modeNumList:
                directMode = util.load_mat_text(
                  directModePath%modeNum1)
                for modeNum2 in self.modeNumList:
                    adjointMode = util.load_mat_text(
                      adjointModePath%modeNum2)
                    innerProduct = self.bpod.inner_product(
                      directMode,adjointMode)
                    if modeNum1 != modeNum2:
                        self.assertAlmostEqual(innerProduct,0.)
                    else:
                        self.assertAlmostEqual(innerProduct,1.)
      
if __name__=='__main__':
    unittest.main(verbosity=2)
    
        
    


