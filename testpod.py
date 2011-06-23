#!/usr/bin/env python
import subprocess as SP
import os
import numpy as N
from pod import POD
from fieldoperations import FieldOperations
import unittest
import util
import copy


class TestPOD(unittest.TestCase):
    """ Test all the POD class methods """
    
    def setUp(self):
        if not os.path.isdir('files_modaldecomp_test'):        
            SP.call(['mkdir','files_modaldecomp_test'])
        self.modeNumList =[2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.numSnaps = 40
        self.numStates = 100
        self.indexFrom = 2
        self.pod = POD(load_field=util.load_mat_text, save_field=\
            util.save_mat_text, save_mat=util.save_mat_text, inner_product=\
            util.inner_product, verbose=False)
        self.generate_data_set()  
      

    def tearDown(self):
        SP.call(['rm -rf files_modaldecomp_test/*'], shell=True)

    def generate_data_set(self):
        # create data set (saved to file)
        self.snapPath = 'files_modaldecomp_test/snap_%03d.txt'
        self.snapPaths = []
        
        self.snapMat = N.mat(N.random.random((self.numStates,self.\
            numSnaps)))
        for snapIndex in range(self.numSnaps):
            util.save_mat_text(self.snapMat[:, snapIndex], self.snapPath %\
                snapIndex)
            self.snapPaths.append(self.snapPath % snapIndex)
         
        self.correlationMatTrue = self.snapMat.T * self.snapMat
        
        self.singVecsTrue, self.singValsTrue, dummy = util.svd(self.\
            correlationMatTrue)        
        self.modeMat = self.snapMat * N.mat(self.singVecsTrue) * N.mat(N.diag(
            self.singValsTrue ** -0.5))

     
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        def getDataMembersDefault():
            return {'save_mat': util.save_mat_text,
                'load_mat': util.load_mat_text, 
                'verbose': False,
                'fieldOperations': FieldOperations(load_field=None, save_field=None,
                inner_product=None, maxFields=\
                2, verbose=False)}
            
        #for k,v in dataMembersDefault.iteritems():
        #    print k,v,util.get_data_members(POD(verbose=False))[k]
        #    print v==util.get_data_members(POD(verbose=False))[k]
        self.assertEqual(util.get_data_members(POD(verbose=False)), \
            getDataMembersDefault())

        def my_load(fname): pass
        myPOD = POD(load_field=my_load, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].load_field = my_load
        self.assertEqual(util.get_data_members(myPOD), dataMembersModified)
        
        def my_save(data,fname): pass 
        myPOD = POD(save_field=my_save, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].save_field = my_save
        self.assertEqual(util.get_data_members(myPOD), dataMembersModified)
        
        myPOD = POD(save_mat=my_save, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['save_mat'] = my_save
        self.assertEqual(util.get_data_members(myPOD), dataMembersModified)
                              
        def my_ip(f1,f2): pass
        myPOD = POD(inner_product=my_ip, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].inner_product = my_ip
        self.assertEqual(util.get_data_members(myPOD), dataMembersModified)

        maxFields = 500
        myPOD = POD(maxFields=maxFields, verbose=False)
        dataMembersModified = getDataMembersDefault()
        dataMembersModified['fieldOperations'].maxFields = maxFields
        self.assertEqual(util.get_data_members(myPOD), dataMembersModified)
        
    
        
    def test_compute_decomp(self):
        """
        Test that can take snapshots, compute the correlation and SVD matrices
        
        With previously generated random snapshots, compute the correlation 
        matrix, then take the SVD. The computed matrices are saved, then
        loaded and compared to the true matrices. 
        """
        tol = 8
        snapPath = 'files_modaldecomp_test/snap_%03d.txt'
        singVecsPath = 'files_modaldecomp_test/singvecs.txt'
        singValsPath = 'files_modaldecomp_test/singvals.txt'
        correlationMatPath = 'files_modaldecomp_test/correlation.txt'
        
        for sharedMemLoad in [True, False]:
            for sharedMemInnerProduct in [True, False]:
                self.pod.compute_decomp(snapPaths=self.snapPaths, 
                    sharedMemLoad=sharedMemLoad, sharedMemInnerProduct=\
                    sharedMemInnerProduct)
                self.pod.save_correlation_mat(correlationMatPath)
                self.pod.save_decomp(singVecsPath, singValsPath)
                
                singVecsLoaded = util.load_mat_text(singVecsPath)
                singValsLoaded = N.squeeze(N.array(util.load_mat_text(
                    singValsPath)))
                correlationMatLoaded = util.load_mat_text(correlationMatPath)
                
                N.testing.assert_array_almost_equal(self.pod.correlationMat, 
                    self.correlationMatTrue, decimal=tol)
                N.testing.assert_array_almost_equal(self.pod.singVecs, self.\
                    singVecsTrue, decimal=tol)
                N.testing.assert_array_almost_equal(self.pod.singVals, self.\
                    singValsTrue, decimal=tol)
                  
                N.testing.assert_array_almost_equal(correlationMatLoaded, self.\
                    correlationMatTrue, decimal=tol)
                N.testing.assert_array_almost_equal(singVecsLoaded, self.\
                    singVecsTrue, decimal=tol)
                N.testing.assert_array_almost_equal(singValsLoaded, self.\
                    singValsTrue, decimal=tol)
        

    def test_compute_modes(self):
        """
        Test computing modes. 
        
        This method uses the existing random data set saved to disk. It tests
        that POD can generate the modes, save them, and load them, then
        compares them to the known solution.
        """

        modePath = 'files_modaldecomp_test/mode_%03d.txt'
        
        # starts with the CORRECT decomposition.
        self.pod.singVecs = self.singVecsTrue
        self.pod.singVals = self.singValsTrue
        
        for sharedMemLoad in [True, False]:
            for sharedMemSave in [True, False]:
                self.pod.compute_modes(self.modeNumList, modePath, indexFrom=\
                    self.indexFrom, snapPaths=self.snapPaths, sharedMemLoad=\
                    sharedMemLoad, sharedMemSave=sharedMemSave)
                  
                for modeNum in self.modeNumList:
                    mode = util.load_mat_text(modePath % modeNum)
                    N.testing.assert_array_almost_equal(mode, self.modeMat[:, 
                        modeNum - self.indexFrom])
                
                for modeNum1 in self.modeNumList:
                    mode1 = util.load_mat_text(modePath % modeNum1)
                    for modeNum2 in self.modeNumList:
                        mode2 = util.load_mat_text(modePath % modeNum2)
                        innerProduct = self.pod.fieldOperations.inner_product(
                            mode1, mode2)
                        if modeNum1 != modeNum2:
                            self.assertAlmostEqual(innerProduct, 0.)
                        else:
                            self.assertAlmostEqual(innerProduct, 1.)


if __name__=='__main__':
    unittest.main(verbosity=2)


