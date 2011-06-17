#!/usr/bin/env python
import numpy as N
from dmd import DMD
from pod import POD
import unittest
import util
import subprocess as SP
import os
import copy

class TestDMD(unittest.TestCase):
    """ Test all the DMD class methods 
    
    Since most of the computations of DMD are done by POD methods
    currently, there should be less to test here"""
    
    def setUp(self):
        if not os.path.isdir('files_modaldecomp_test'):        
            SP.call(['mkdir','files_modaldecomp_test'])
        self.numSnaps = 6 # number of snapshots to generate
        self.numStates = 7 # dimension of state vector
        self.indexFrom = 2
        self.dmd = DMD(verbose=False, load_field=util.load_mat_text, 
            save_field=util.save_mat_text, save_mat=util.save_mat_text, 
            inner_product=util.inner_product)
        self.generate_data_set()
   
        # Default data members for constructor test
        self.defaultDataMembers = {'save_mat': util.save_mat_text,
            'maxFields': 2, 
            'pod': None, 'verbose': False}
            #'snapPaths': None, 'buildCoeff': None,


    def generate_data_set(self):
        # create data set (saved to file)
        self.snapPath = 'files_modaldecomp_test/dmd_snap_%03d.txt'
        self.trueModePath = 'files_modaldecomp_test/dmd_truemode_%03d.txt'
        self.snapPathList = []
       
        # Generate modes if we are on the first processor
            # A random matrix of data (#cols = #snapshots)
        self.snapMat = N.mat(N.random.random((self.numStates, self.\
            numSnaps)))
        
        for snapNum in range(self.numSnaps):
            util.save_mat_text(self.snapMat[:,snapNum], self.snapPath %\
                snapNum)
            self.snapPathList.append(self.snapPath % snapNum) 

        # Do direct DMD decomposition on all processors
        U, Sigma, W = util.svd(self.snapMat[:,:-1])
        SigmaMat = N.mat(N.diag(Sigma))
        self.ritzValsTrue, evecs = N.linalg.eig(U.H * self.snapMat[:,1:] * \
            W * (SigmaMat ** -1))
        evecs = N.mat(evecs)
        ritzVecs = U * evecs
        scaling, dummy1, dummy2, dummy3 = N.linalg.lstsq(ritzVecs, self.\
            snapMat[:,0])
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        self.ritzVecsTrue = ritzVecs * scaling
        self.buildCoeffTrue = W * (SigmaMat ** -1) * evecs * scaling
        self.modeNormsTrue = N.zeros(self.ritzVecsTrue.shape[1])
        for i in xrange(self.ritzVecsTrue.shape[1]):
            self.modeNormsTrue[i] = self.dmd.inner_product(N.array(self.\
                ritzVecsTrue[:,i]), N.array(self.ritzVecsTrue[:,i])).real

        # Generate modes if we are on the first processor
        for i in xrange(self.ritzVecsTrue.shape[1]):
            util.save_mat_text(self.ritzVecsTrue[:,i], self.trueModePath%\
                (i+1))

    def tearDown(self):
        SP.call(['rm -rf files_modaldecomp_test/*'],shell=True)

    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        dataMembersOriginal = util.get_data_members(DMD(verbose=False))
        self.assertEqual(dataMembersOriginal, self.defaultDataMembers)

        def my_load(fname): pass
        myDMD = DMD(load_field=my_load, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['load_field'] = my_load
        self.assertEqual(util.get_data_members(myDMD), dataMembers)
        
        def my_save(data, fname): pass 
        myDMD = DMD(save_field=my_save, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['fieldOperations'].save_field = my_save
        self.assertEqual(util.get_data_members(myDMD), dataMembers)
 
        myDMD = DMD(save_mat=my_save, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['save_mat'] = my_save
        self.assertEqual(util.get_data_members(myDMD), dataMembers)
 
        def my_ip(f1,f2): pass
        myDMD = DMD(inner_product=my_ip, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['inner_product'] = my_ip
        self.assertEqual(util.get_data_members(myDMD), dataMembers)
 
        maxFields = 500
        myDMD = DMD(maxFields=maxFields, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['maxFields'] = maxFields
        self.assertEqual(util.get_data_members(myDMD), dataMembers)
 
        snapPathList=['a', 'b']
        myDMD = DMD(snapPaths=snapPathList, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['snapPaths'] = snapPathList
        self.assertEqual(util.get_data_members(myDMD), dataMembers)

        buildCoeff = N.mat(N.random.random((2,2)))
        myDMD = DMD(buildCoeff=buildCoeff, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['buildCoeff'] = buildCoeff
        self.assertEqual(util.get_data_members(myDMD), dataMembers)
        
        podObj = POD(verbose=False)
        myDMD = DMD(pod=podObj, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['pod'] = podObj
        self.assertEqual(util.get_data_members(myDMD), dataMembers)


    def test_compute_decomp(self):
        """ 
        #Mostly need to tests the part unique to this function. The 
        #lower level functions are tested independently.
        pass
        """
        # Depending on snapshots generated, test will fail if tol = 8, so use 7
        tol = 7 

        # Run decomposition and save matrices to file
        ritzValsPath = 'files_modaldecomp_test/dmd_ritzvals.txt'
        modeNormsPath = 'files_modaldecomp_test/dmd_modeenergies.txt'
        buildCoeffPath = 'files_modaldecomp_test/dmd_buildcoeff.txt'
        self.dmd.compute_decomp(snapPaths=self.snapPathList, ritzValsPath=\
            ritzValsPath, modeNormsPath=modeNormsPath, buildCoeffPath=\
            buildCoeffPath)
       
        # Test that matrices were correctly computed
        N.testing.assert_array_almost_equal(self.dmd.ritzVals, self.\
            ritzValsTrue, decimal=tol)
        N.testing.assert_array_almost_equal(self.dmd.buildCoeff, self.\
            buildCoeffTrue, decimal=tol)
        N.testing.assert_array_almost_equal(self.dmd.modeNorms, self.\
            modeNormsTrue, decimal=tol)

        # Test that matrices were correctly stored
        ritzValsLoaded = N.array(util.load_mat_text(ritzValsPath,isComplex=\
            True)).squeeze()
        buildCoeffLoaded = util.load_mat_text(buildCoeffPath,isComplex=True)
        modeNormsLoaded = N.array(util.load_mat_text(modeNormsPath).\
            squeeze())

        N.testing.assert_array_almost_equal(ritzValsLoaded, self.\
            ritzValsTrue, decimal=tol)
        N.testing.assert_array_almost_equal(buildCoeffLoaded, self.\
            buildCoeffTrue, decimal=tol)
        N.testing.assert_array_almost_equal(modeNormsLoaded, self.\
            modeNormsTrue, decimal=tol)

    def test_compute_modes(self):
        """
        Test building of modes, reconstruction formula.

        """
        tol = 8

        modePath ='files_modaldecomp_test/dmd_mode_%03d.txt'
        self.dmd.buildCoeff = self.buildCoeffTrue
        modeNumList = list(N.array(range(self.numSnaps-1))+self.indexFrom)
        self.dmd.compute_modes(modeNumList, modePath, indexFrom=self.indexFrom, 
            snapPaths=self.snapPathList)
       
        # Load all snapshots into matrix
        modeMat = N.mat(N.zeros((self.numStates, self.numSnaps-1)), dtype=\
            complex)
        for i in range(self.numSnaps-1):
            modeMat[:,i] = util.load_mat_text(modePath % (i+self.indexFrom),
                isComplex=True)
        N.testing.assert_array_almost_equal(modeMat,self.ritzVecsTrue, decimal=\
            tol)

        vandermondeMat = N.fliplr(N.vander(self.ritzValsTrue,self.numSnaps-1))
        N.testing.assert_array_almost_equal(self.snapMat[:,:-1], self.\
            ritzVecsTrue * vandermondeMat, decimal=tol)

        util.save_mat_text(vandermondeMat, 'files_modaldecomp_test/' +\
            'dmd_vandermonde.txt')

if __name__=='__main__':
    unittest.main(verbosity=2)


