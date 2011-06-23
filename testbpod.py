#!/usr/bin/env python
import subprocess as SP
import os
import numpy as N
from bpod import BPOD
from fieldoperations import FieldOperations
import unittest
import util
import copy


class TestBPOD(unittest.TestCase):
    """ Test all the BPOD class methods """
    def setUp(self):
        self.maxDiff = 2000
        if not os.path.isdir('files_modaldecomp_test'):        
            SP.call(['mkdir','files_modaldecomp_test'])
        self.modeNumList =[2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.numDirectSnaps = 40
        self.numAdjointSnaps = 45
        self.numStates = 100
        self.indexFrom = 2
        self.bpod = BPOD(load_field=util.load_mat_text, save_field=util.\
            save_mat_text, inner_product=util.\
            inner_product, verbose=False)
        self.generate_data_set()
   

    def tearDown(self):
        SP.call(['rm -rf files_modaldecomp_test/*'], shell=True)
    
    def generate_data_set(self):
        # create data set (saved to file)
        self.directSnapPath = 'files_modaldecomp_test/direct_snap_%03d.txt'
        self.adjointSnapPath = 'files_modaldecomp_test/adjoint_snap_%03d.txt'

        self.directSnapPaths=[]
        self.adjointSnapPaths=[]
        
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

         
        self.hankelMatTrue = self.adjointSnapMat.T * self.directSnapMat
        
        #Do the SVD
        self.LSingVecsTrue, self.singValsTrue, self.RSingVecsTrue = util.svd(
            self.hankelMatTrue)
        self.directModeMat = self.directSnapMat * N.mat(self.RSingVecsTrue) *\
            N.mat(N.diag(self.singValsTrue ** -0.5))
        self.adjointModeMat = self.adjointSnapMat * N.mat(self.LSingVecsTrue) *\
            N.mat(N.diag(self.singValsTrue ** -0.5))
        
        #self.bpod.directSnapPaths=self.directSnapPaths
        #self.bpod.adjointSnapPaths=self.adjointSnapPaths
        
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Default data members for constructor test
        def getDataMembersDefault():
            return {'save_mat': util.save_mat_text,
                'load_mat': util.load_mat_text, 
                'verbose': False,
                'fieldOperations': FieldOperations(load_field=None, save_field=\
                None, inner_product=None, maxFields=2, verbose=False)}
        
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        self.assertEqual(util.get_data_members(BPOD(verbose=False)), \
            getDataMembersDefault())
        
        def my_load(fname): pass
        myBPOD = BPOD(load_field=my_load, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].load_field = my_load
        self.assertEqual(util.get_data_members(myBPOD), dataMembersModified)
        
        def my_save(data,fname): pass 
        myBPOD = BPOD(save_field=my_save, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].save_field = my_save
        self.assertEqual(util.get_data_members(myBPOD), dataMembersModified)
        
        myBPOD = BPOD(save_mat=my_save, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['save_mat'] = my_save
        self.assertEqual(util.get_data_members(myBPOD), dataMembersModified)
                
        def my_ip(f1,f2): pass
        myBPOD = BPOD(inner_product=my_ip, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].inner_product = my_ip
        self.assertEqual(util.get_data_members(myBPOD), dataMembersModified)
                                
        maxFields = 500
        myBPOD = BPOD(maxFields=maxFields, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].maxFields = maxFields
        self.assertEqual(util.get_data_members(myBPOD), dataMembersModified)
        
        
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
        LSingVecsPath = 'files_modaldecomp_test/lsingvecs.txt'
        RSingVecsPath = 'files_modaldecomp_test/rsingvecs.txt'
        singValsPath = 'files_modaldecomp_test/singvals.txt'
        hankelMatPath = 'files_modaldecomp_test/hankel.txt'
        
        for sharedMemLoad in [True, False]:
            for sharedMemInnerProduct in [True, False]:
                self.bpod.compute_decomp(directSnapPaths=self.directSnapPaths, 
                    adjointSnapPaths=self.adjointSnapPaths, sharedMemLoad=\
                    sharedMemLoad, sharedMemInnerProduct=sharedMemInnerProduct)
                
                self.bpod.save_hankel_mat(hankelMatPath)
                self.bpod.save_decomp(LSingVecsPath, singValsPath, 
                    RSingVecsPath)
                LSingVecsLoaded = util.load_mat_text(LSingVecsPath)
                RSingVecsLoaded = util.load_mat_text(RSingVecsPath)
                singValsLoaded = N.squeeze(N.array(util.load_mat_text(
                    singValsPath)))
                hankelMatLoaded = util.load_mat_text(hankelMatPath)
                
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

        directModePath = 'files_modaldecomp_test/direct_mode_%03d.txt'
        adjointModePath = 'files_modaldecomp_test/adjoint_mode_%03d.txt'
        
        # starts with the CORRECT decomposition.
        self.bpod.RSingVecs = self.RSingVecsTrue
        self.bpod.LSingVecs = self.LSingVecsTrue
        self.bpod.singVals = self.singValsTrue
        
        for sharedMemLoad in [True, False]:
            for sharedMemSave in [True, False]:
                self.bpod.compute_direct_modes(self.modeNumList, directModePath,
                    indexFrom=self.indexFrom, directSnapPaths=self.\
                    directSnapPaths, sharedMemLoad=sharedMemLoad, 
                    sharedMemSave=sharedMemSave)
                  
                self.bpod.compute_adjoint_modes(self.modeNumList, 
                    adjointModePath, indexFrom=self.indexFrom, 
                    adjointSnapPaths=self.adjointSnapPaths, sharedMemLoad=\
                    sharedMemLoad, sharedMemSave=sharedMemSave)
                  
                for modeNum in self.modeNumList:
                    directMode = util.load_mat_text(directModePath % modeNum)
                    adjointMode = util.load_mat_text(adjointModePath % modeNum)
                    N.testing.assert_array_almost_equal(directMode, self.\
                        directModeMat[:, modeNum-self.indexFrom])
                    N.testing.assert_array_almost_equal(adjointMode,self.\
                        adjointModeMat[:,modeNum-self.indexFrom])
                
                for modeNum1 in self.modeNumList:
                    directMode = util.load_mat_text(directModePath%modeNum1)
                    for modeNum2 in self.modeNumList:
                        adjointMode = util.load_mat_text(adjointModePath %\
                            modeNum2)
                        innerProduct = self.bpod.fieldOperations.inner_product(
                            directMode, adjointMode)
                        if modeNum1 != modeNum2:
                            self.assertAlmostEqual(innerProduct,0.)
                        else:
                            self.assertAlmostEqual(innerProduct,1.)
      
if __name__=='__main__':
    unittest.main(verbosity=2)
    
        
    


