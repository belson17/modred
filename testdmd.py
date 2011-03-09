#!/usr/local/bin/env python

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

if parallel:
    if rank==0:
        print 'Remember to test in serial also with command:'
        print 'python testdmd.py'
else:
    print 'Remember to test in parallel also with command:'
    print 'mpiexec -n <numProcs> python testdmd.py' 

class TestDMD(unittest.TestCase):
    """ Test all the DMD class methods 
    
    Since most of the computations of DMD are done by POD methods
    currently, there should be less to test here"""
    
    def setUp(self):
        if not os.path.isdir('testfiles'):        
            SP.call(['mkdir','testfiles'])
        self.dmd = D.DMD()
        self.numSnaps = 6 # number of snapshots to generate
        self.numStates = 7 # dimension of state vector
        self.dmd.save_mat=util.save_mat_text
        self.dmd.load_snap=util.load_mat_text
        self.dmd.inner_product=util.inner_product
        self.dmd.save_mode = util.save_mat_text
        self.indexFrom = 2
        self.dmd.indexFrom=self.indexFrom
        self.generate_data_set()

    def generate_data_set(self):
        #create data set (saved to file)
        self.snapPath ='testfiles/dmd_snap_%03d.txt'
        self.trueModePath = 'testfiles/dmd_truemode_%03d.txt'
        self.snapPathList = []
       
        # Generate modes if we are on the first processor
        if self.dmd.mpi._rank==0:
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
        if self.dmd.mpi.parallel:
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
        self.modeEnergiesTrue = N.zeros(self.ritzVecsTrue.shape[1])
        for i in xrange(self.ritzVecsTrue.shape[1]):
            self.modeEnergiesTrue[i] = self.dmd.inner_product(N.array(self.\
                ritzVecsTrue[:,i]), N.array(self.ritzVecsTrue[:,i])).real

        # Generate modes if we are on the first processor
        if self.dmd.mpi._rank==0:
            for i in xrange(self.ritzVecsTrue.shape[1]):
                util.save_mat_text(self.ritzVecsTrue[:,i], self.trueModePath%\
                    (i+1))

    def tearDown(self):
        self.dmd.mpi.sync()
        if self.dmd.mpi._rank == 0:
            SP.call(['rm -rf testfiles/*'],shell=True)
        self.dmd.mpi.sync()
   
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Test that default optional arguments are correct
        myDMD = D.DMD()
        self.assertEqual(myDMD.load_snap,None)
        self.assertEqual(myDMD.save_mode,None)
        self.assertEqual(myDMD.save_mat,util.save_mat_text)
        self.assertEqual(myDMD.inner_product,None)
        self.assertEqual(myDMD.maxSnapsInMem,100)
        self.assertEqual(myDMD.snapPaths,None)
        self.assertEqual(myDMD.buildCoeff,None)
        self.assertEqual(myDMD.ritzVals,None)
        self.assertEqual(myDMD.pod,None)

        # Test that constructor assignments are correct
        def my_load(fname): 
            return 0
        myDMD = D.DMD(load_snap=my_load)
        self.assertEqual(myDMD.load_snap,my_load)
        
        def my_save(data,fname):
            pass 
        myDMD = D.DMD(save_mode=my_save)
        self.assertEqual(myDMD.save_mode,my_save)
        
        myDMD = D.DMD(save_mat=my_save)
        self.assertEqual(myDMD.save_mat,my_save)
        
        def my_ip(f1,f2): 
            return 0
        myDMD = D.DMD(inner_product=my_ip)
        self.assertEqual(myDMD.inner_product,my_ip)
        
        maxSnaps = 500
        myDMD = D.DMD(maxSnapsInMem=maxSnaps)
        self.assertEqual(myDMD.maxSnapsInMem,maxSnaps)
        
        snapPathList=['a','b']
        myDMD = D.DMD(snapPaths=snapPathList)
        self.assertEqual(myDMD.snapPaths,snapPathList)

        buildCoeff = N.mat(N.random.random((2,2)))
        myDMD = D.DMD(buildCoeff=buildCoeff)
        N.testing.assert_array_almost_equal(myDMD.buildCoeff,buildCoeff)
        
        ritzVals = N.mat(N.random.random((2,2)))
        myDMD = D.DMD(ritzVals=ritzVals)
        N.testing.assert_array_almost_equal(myDMD.ritzVals,ritzVals)
        
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
        ritzValsPath = 'testfiles/dmd_ritzvals.txt'
        modeEnergiesPath = 'testfiles/dmd_modeenergies.txt'
        buildCoeffPath = 'testfiles/dmd_buildcoeff.txt'
        self.dmd.compute_decomp(snapPaths=self.snapPathList, ritzValsPath=\
            ritzValsPath, modeEnergiesPath=modeEnergiesPath, buildCoeffPath=\
            buildCoeffPath)
       
        # Test that matrices were correctly computed
        N.testing.assert_array_almost_equal(self.dmd.ritzVals, self.\
            ritzValsTrue, decimal=tol)
        N.testing.assert_array_almost_equal(self.dmd.buildCoeff, self.\
            buildCoeffTrue, decimal=tol)
        N.testing.assert_array_almost_equal(self.dmd.modeEnergies, self.\
            modeEnergiesTrue, decimal=tol)

        # Test that matrices were correctly stored
        if self.dmd.mpi._rank==0:
            ritzValsLoaded = N.array(util.load_mat_text(ritzValsPath,isComplex=\
                True)).squeeze()
            buildCoeffLoaded = util.load_mat_text(buildCoeffPath,isComplex=True)
            modeEnergiesLoaded = N.array(util.load_mat_text(modeEnergiesPath).\
                squeeze())
        else:   
            ritzValsLoaded = None
            buildCoeffLoaded = None
            modeEnergiesLoaded = None

        if self.dmd.mpi._numProcs>1:
            ritzValsLoaded = self.dmd.mpi.comm.bcast(ritzValsLoaded,root=0)
            buildCoeffLoaded = self.dmd.mpi.comm.bcast(buildCoeffLoaded,root=0)
            modeEnergiesLoaded = self.dmd.mpi.comm.bcast(modeEnergiesLoaded,
                root=0)

        N.testing.assert_array_almost_equal(ritzValsLoaded, self.\
            ritzValsTrue, decimal=tol)
        N.testing.assert_array_almost_equal(buildCoeffLoaded, self.\
            buildCoeffTrue, decimal=tol)
        N.testing.assert_array_almost_equal(modeEnergiesLoaded, self.\
            modeEnergiesTrue, decimal=tol)

    def test_compute_modes(self):
        """
        Test building of modes, reconstruction formula.

        """
        tol = 8

        modePath ='testfiles/dmd_mode_%03d.txt'
        self.dmd.buildCoeff = self.buildCoeffTrue
        modeNumList = list(N.array(range(self.numSnaps-1))+self.indexFrom)
        self.dmd.compute_modes(modeNumList, modePath, indexFrom=self.indexFrom, 
            snapPaths=self.snapPathList)
       
        # Load all snapshots into matrix
        if self.dmd.mpi._rank==0:
            modeMat = N.mat(N.zeros((self.numStates, self.numSnaps-1)), dtype=\
                complex)
            for i in range(self.numSnaps-1):
                modeMat[:,i] = util.load_mat_text(modePath % (i+self.indexFrom),
                    isComplex=True)
        else:
            modeMat = None
        if self.dmd.mpi._numProcs>1:
            modeMat = self.dmd.mpi.comm.bcast(modeMat,root=0)
        N.testing.assert_array_almost_equal(modeMat,self.ritzVecsTrue, decimal=\
            tol)

        vandermondeMat = N.fliplr(N.vander(self.ritzValsTrue,self.numSnaps-1))
        N.testing.assert_array_almost_equal(self.snapMat[:,:-1], self.\
            ritzVecsTrue * vandermondeMat, decimal=tol)

        util.save_mat_text(vandermondeMat,'testfiles/dmd_vandermonde.txt')
if __name__=='__main__':
    unittest.main()




