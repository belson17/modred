#!/usr/bin/env python

import numpy as N
import dmd as D
import pod as P
import unittest
import util
import subprocess as SP
import os

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    parallel = comm.Get_size() >=2
    rank = comm.Get_rank()
except ImportError:
    parallel = False
    rank = 0

print 'To test fully, remember to do both:'
print '    1) python testdmd.py'
print '    2) mpiexec -n <# procs> python testdmd.py\n'

class TestDMD(unittest.TestCase):
    """ Test all the DMD class methods 
    
    Since most of the computations of DMD are done by POD methods
    currently, there should be less to test here"""
    
    def setUp(self):
        if not os.path.isdir('modaldecomp_testfiles'):        
            SP.call(['mkdir','modaldecomp_testfiles'])
        self.dmd = D.DMD()
        self.numSnaps = 6 # number of snapshots to generate
        self.numStates = 7 # dimension of state vector
        self.dmd.save_mat=util.save_mat_text
        self.dmd.load_field=util.load_mat_text
        self.dmd.inner_product=util.inner_product
        self.dmd.save_field = util.save_mat_text
        self.indexFrom = 2
        self.dmd.indexFrom=self.indexFrom
        self.generate_data_set()

    def generate_data_set(self):
        #create data set (saved to file)
        self.snapPath ='modaldecomp_testfiles/dmd_snap_%03d.txt'
        self.trueModePath = 'modaldecomp_testfiles/dmd_truemode_%03d.txt'
        self.snapPathList = []
       
        # Generate modes if we are on the first processor
        if self.dmd.mpi.isRankZero():
            # A random matrix of data (#cols = #snapshots)
            self.snapMat = N.mat(N.random.random((self.numStates, self.\
                numSnaps)))
            
            for snapNum in range(self.numSnaps):
                util.save_mat_text(self.snapMat[:,snapNum], self.snapPath %\
                    snapNum)
                self.snapPathList.append(self.snapPath % snapNum) 
        else:
            self.snapPathList=None
            self.snapMat = None
        if self.dmd.mpi.isParallel():
            self.snapPathList = self.dmd.mpi.comm.bcast(self.snapPathList, 
                root=0)
            self.snapMat = self.dmd.mpi.comm.bcast(self.snapMat, root=0)

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
        if self.dmd.mpi.isRankZero():
            for i in xrange(self.ritzVecsTrue.shape[1]):
                util.save_mat_text(self.ritzVecsTrue[:,i], self.trueModePath%\
                    (i+1))

    def tearDown(self):
        self.dmd.mpi.sync()
        if self.dmd.mpi.isRankZero():
            SP.call(['rm -rf modaldecomp_testfiles/*'],shell=True)
        self.dmd.mpi.sync()

    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Test that default optional arguments are correct
        myDMD = D.DMD()
        self.assertEqual(myDMD.load_field,None)
        self.assertEqual(myDMD.save_field,None)
        self.assertEqual(myDMD.save_mat,util.save_mat_text)
        self.assertEqual(myDMD.inner_product,None)
        self.assertEqual(myDMD.maxFieldsPerNode,2)
        self.assertEqual(myDMD.snapPaths,None)
        self.assertEqual(myDMD.buildCoeff,None)
        self.assertEqual(myDMD.pod,None)

        # Test that constructor assignments are correct
        def my_load(fname): 
            return 0
        myDMD = D.DMD(load_field=my_load)
        self.assertEqual(myDMD.load_field,my_load)
        
        def my_save(data,fname):
            pass 
        myDMD = D.DMD(save_field=my_save)
        self.assertEqual(myDMD.save_field,my_save)
        
        myDMD = D.DMD(save_mat=my_save)
        self.assertEqual(myDMD.save_mat,my_save)
        
        def my_ip(f1,f2): 
            return 0
        myDMD = D.DMD(inner_product=my_ip)
        self.assertEqual(myDMD.inner_product,my_ip)
        
        maxSnaps = 500
        myDMD = D.DMD(maxFieldsPerNode=maxSnaps)
        self.assertEqual(myDMD.maxFieldsPerNode,maxSnaps)
        
        snapPathList=['a','b']
        myDMD = D.DMD(snapPaths=snapPathList)
        self.assertEqual(myDMD.snapPaths,snapPathList)

        buildCoeff = N.mat(N.random.random((2,2)))
        myDMD = D.DMD(buildCoeff=buildCoeff)
        N.testing.assert_array_almost_equal(myDMD.buildCoeff,buildCoeff)
        
        podObj = P.POD()
        myDMD = D.DMD(pod=podObj)
        N.testing.assert_equal(myDMD.pod, podObj)

    def test_compute_decomp(self):
        """ 
        #Mostly need to tests the part unique to this function. The 
        #lower level functions are tested independently.
        pass
        """
        # Depending on snapshots generated, test will fail if tol = 8, so use 7
        tol = 7 

        # Run decomposition and save matrices to file
        ritzValsPath = 'modaldecomp_testfiles/dmd_ritzvals.txt'
        modeNormsPath = 'modaldecomp_testfiles/dmd_modeenergies.txt'
        buildCoeffPath = 'modaldecomp_testfiles/dmd_buildcoeff.txt'
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
        if self.dmd.mpi.isRankZero():
            ritzValsLoaded = N.array(util.load_mat_text(ritzValsPath,isComplex=\
                True)).squeeze()
            buildCoeffLoaded = util.load_mat_text(buildCoeffPath,isComplex=True)
            modeNormsLoaded = N.array(util.load_mat_text(modeNormsPath).\
                squeeze())
        else:   
            ritzValsLoaded = None
            buildCoeffLoaded = None
            modeNormsLoaded = None

        if self.dmd.mpi.isParallel():
            ritzValsLoaded = self.dmd.mpi.comm.bcast(ritzValsLoaded,root=0)
            buildCoeffLoaded = self.dmd.mpi.comm.bcast(buildCoeffLoaded,root=0)
            modeNormsLoaded = self.dmd.mpi.comm.bcast(modeNormsLoaded,
                root=0)

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

        modePath ='modaldecomp_testfiles/dmd_mode_%03d.txt'
        self.dmd.buildCoeff = self.buildCoeffTrue
        modeNumList = list(N.array(range(self.numSnaps-1))+self.indexFrom)
        self.dmd.compute_modes(modeNumList, modePath, indexFrom=self.indexFrom, 
            snapPaths=self.snapPathList)
       
        # Load all snapshots into matrix
        if self.dmd.mpi.isRankZero():
            modeMat = N.mat(N.zeros((self.numStates, self.numSnaps-1)), dtype=\
                complex)
            for i in range(self.numSnaps-1):
                modeMat[:,i] = util.load_mat_text(modePath % (i+self.indexFrom),
                    isComplex=True)
        else:
            modeMat = None
        if self.dmd.mpi.isParallel():
            modeMat = self.dmd.mpi.comm.bcast(modeMat,root=0)
        N.testing.assert_array_almost_equal(modeMat,self.ritzVecsTrue, decimal=\
            tol)

        vandermondeMat = N.fliplr(N.vander(self.ritzValsTrue,self.numSnaps-1))
        N.testing.assert_array_almost_equal(self.snapMat[:,:-1], self.\
            ritzVecsTrue * vandermondeMat, decimal=tol)

        util.save_mat_text(vandermondeMat,'modaldecomp_testfiles/dmd_vandermonde.txt')
if __name__=='__main__':
    unittest.main(verbosity=2)




