#!/usr/bin/env python

import subprocess as SP
import era
import util
import numpy as N
import unittest
import copy

class testERA(unittest.TestCase):
    
    def setUp(self):
        self.testDirectory = 'ERA_test_files/'
        self.IOPaths = [self.testDirectory+'input1_impulse.txt', \
          self.testDirectory+'input2_impulse.txt']
        outputSignals = util.load_mat_text(self.IOPaths[0],delimiter=' ')
        self.dt = 25.
        self.numSnaps,self.numOutputs = N.shape(outputSignals[:,1:])
        self.numInputs = len(self.IOPaths)
        
        self.IOSignals = N.zeros((self.numOutputs,self.numInputs,self.numSnaps))
        self.time = outputSignals[:,0]
        self.IOSignals[:,0,:] = outputSignals[:,1:].T
        for inputIndex, IOPath in enumerate(self.IOPaths[1:]):
            outputSignals = util.load_mat_text(IOPath,delimiter=' ')
            self.IOSignals[:,inputIndex+1,:] = outputSignals[:,1:].T
        
        self.hankelMatKnown = \
          N.mat(util.load_mat_text(self.testDirectory+'hankelMatKnown.txt'))
        self.hankelMat2Known = \
          N.mat(util.load_mat_text(self.testDirectory+'hankelMat2Known.txt'))
        self.LSingVecsKnown = \
          N.mat(util.load_mat_text(self.testDirectory+'LSingVecsKnown.txt'))
        self.singValsKnown =  \
          N.squeeze(util.load_mat_text(self.testDirectory+'singValsKnown.txt'))
        self.RSingVecsKnown = \
          N.mat(util.load_mat_text(self.testDirectory+'RSingVecsKnown.txt'))
                
        self.APathKnown = self.testDirectory+'AKnown.txt'
        self.BPathKnown = self.testDirectory+'BKnown.txt'
        self.CPathKnown = self.testDirectory+'CKnown.txt'
        self.numStates = 50
        self.AKnown = util.load_mat_text(self.APathKnown)
        self.BKnown = util.load_mat_text(self.BPathKnown)
        self.CKnown = util.load_mat_text(self.CPathKnown)
        
    def tearDown(self):
        """Deletes all of the matrices created by the tests"""
        SP.call(['rm -f '+self.testDirectory+'*Computed*'],shell=True)

    def test_init(self):
        """Tests the constructor"""
        IOSignals = N.random.random((4,5,6))
        myERA = era.ERA(IOSignals = IOSignals)
        N.testing.assert_array_almost_equal(myERA.IOSignals,IOSignals)
        
        # Should test others
        
    def test_set_impulse_outputs(self):
        """Test setting the impulse data"""
        numInputsList = [1,3,2]
        numOutputsList = [1,5,3]
        numSnapsList = [1, 100, 21]
        for numOutputs in numOutputsList:
            for numInputs in numInputsList:
                for numSnaps in numSnapsList:
                    IOSignals = N.random.random((numOutputs,numInputs,numSnaps))%100
                    myERA = era.ERA()
                    myERA.set_impulse_outputs(IOSignals)
                    self.assertEqual(myERA.numOutputs,numOutputs)
                    self.assertEqual(myERA.numInputs,numInputs)
                    self.assertEqual(myERA.numSnaps,numSnaps)
                    N.testing.assert_array_almost_equal(myERA.IOSignals, IOSignals)
    
    def test_load_impulse_outputs(self):
        """Test that can load in a list of impulse output signals from file"""
        myERA = era.ERA()
        myERA.load_impulse_outputs(self.IOPaths)
        N.testing.assert_array_almost_equal(myERA.IOSignals,self.IOSignals)
    
    def test__compute_hankel(self):
        """Test that with given signals, compute correct hankel matrix"""
        myERA = era.ERA()
        myERA.IOSignals = self.IOSignals
        myERA._compute_hankel()
        N.testing.assert_array_almost_equal(myERA.hankelMat,self.hankelMatKnown)
        N.testing.assert_array_almost_equal(myERA.hankelMat2,self.hankelMat2Known)
        
    def test_compute_decomp(self):
        myERA = era.ERA()
        myERA.IOSignals = self.IOSignals
        hankelMatPath = self.testDirectory+'hankelMatComputed.txt'
        hankelMat2Path = self.testDirectory+'hankelMat2Computed.txt'
        LSingVecsPath = self.testDirectory+'LSingVecsComputed.txt'
        singValsPath = self.testDirectory+'singValsComputed.txt'
        RSingVecsPath = self.testDirectory+'RSingVecsComputed.txt'
        myERA.compute_decomp(hankelMatPath=hankelMatPath,
          hankelMat2Path = hankelMat2Path, LSingVecsPath=LSingVecsPath,
          singValsPath = singValsPath, RSingVecsPath=RSingVecsPath)
        
        N.testing.assert_array_almost_equal(myERA.hankelMat,self.hankelMatKnown)
        N.testing.assert_array_almost_equal(myERA.hankelMat2,self.hankelMat2Known)
        N.testing.assert_array_almost_equal(myERA.LSingVecs,self.LSingVecsKnown)
        N.testing.assert_array_almost_equal(N.squeeze(myERA.singVals),
          N.squeeze(self.singValsKnown))
        N.testing.assert_array_almost_equal(myERA.RSingVecs,self.RSingVecsKnown)
        
        # Load in saved decomp matrices, check they are the same
        hankelMatLoaded = util.load_mat_text(hankelMatPath)
        hankelMat2Loaded = myERA.load_mat(hankelMat2Path)
        LSingVecsLoaded = myERA.load_mat(LSingVecsPath)
        RSingVecsLoaded = myERA.load_mat(RSingVecsPath)
        singValsLoaded = myERA.load_mat(singValsPath)

        N.testing.assert_array_almost_equal(hankelMatLoaded,self.hankelMatKnown)
        N.testing.assert_array_almost_equal(hankelMat2Loaded,self.hankelMat2Known)
        N.testing.assert_array_almost_equal(LSingVecsLoaded,self.LSingVecsKnown)
        N.testing.assert_array_almost_equal(N.squeeze(singValsLoaded), \
          N.squeeze(self.singValsKnown))
        N.testing.assert_array_almost_equal(RSingVecsLoaded,self.RSingVecsKnown)
    
    def test_compute_ROM(self):
        """Test forming the ROM matrices from decomp matrices"""
        myERA = era.ERA(numStates = self.numStates)
        myERA.numInputs = 2
        myERA.numOutputs = 2
        myERA.hankelMat = self.hankelMatKnown
        myERA.hankelMat2 = self.hankelMat2Known
        myERA.LSingVecs = self.LSingVecsKnown
        myERA.RSingVecs = self.RSingVecsKnown
        myERA.singVals = self.singValsKnown
        
        APathComputed = self.testDirectory+'AComputed.txt'
        BPathComputed = self.testDirectory+'BComputed.txt'
        CPathComputed = self.testDirectory+'CComputed.txt'
        # Gives an error if there is no time step specified or read from file
        self.assertRaises(util.UndefinedError,myERA.compute_ROM,\
          None,None,APathComputed,BPathComputed,\
          CPathComputed)
        
        myERA.dt = self.dt
        myERA.compute_ROM(APath=APathComputed,BPath=BPathComputed,
          CPath=CPathComputed)
        
        N.testing.assert_array_almost_equal(myERA.A,self.AKnown)
        N.testing.assert_array_almost_equal(myERA.B,self.BKnown)
        N.testing.assert_array_almost_equal(myERA.C,self.CKnown)
        
        ALoaded = myERA.load_mat(APathComputed)
        BLoaded = myERA.load_mat(BPathComputed)
        CLoaded = myERA.load_mat(CPathComputed)
        
        N.testing.assert_array_almost_equal(ALoaded,self.AKnown)
        N.testing.assert_array_almost_equal(BLoaded,self.BKnown)
        N.testing.assert_array_almost_equal(CLoaded,self.CKnown)

    def test_save_load_decomp(self):
        """Test that properly saves decomp matrices"""
        myERA = era.ERA(numStates = self.numStates)
        myERA.hankelMat = copy.deepcopy(self.hankelMatKnown)
        myERA.hankelMat2 = copy.deepcopy(self.hankelMat2Known)
        myERA.LSingVecs = copy.deepcopy(self.LSingVecsKnown)
        myERA.singVals = copy.deepcopy(self.singValsKnown)
        myERA.RSingVecs = copy.deepcopy(self.RSingVecsKnown)
        
        hankelMatPath = self.testDirectory+'hankelMatComputed.txt'
        hankelMat2Path = self.testDirectory+'hankelMat2Computed.txt'
        LSingVecsPath = self.testDirectory+'LSingVecsComputed.txt'
        singValsPath = self.testDirectory+'singValsComputed.txt'
        RSingVecsPath = self.testDirectory+'RSingVecsComputed.txt'
        
        myERA.save_decomp(hankelMatPath=hankelMatPath,
          hankelMat2Path = hankelMat2Path, LSingVecsPath=LSingVecsPath,
          singValsPath = singValsPath, RSingVecsPath=RSingVecsPath)
        
        myERA.load_decomp(hankelMatPath=hankelMatPath,
          hankelMat2Path = hankelMat2Path, LSingVecsPath=LSingVecsPath,
          singValsPath = singValsPath, RSingVecsPath=RSingVecsPath,
          numInputs = self.numInputs, numOutputs=self.numOutputs)
        
        N.testing.assert_array_almost_equal(myERA.hankelMat,self.hankelMatKnown)
        N.testing.assert_array_almost_equal(myERA.hankelMat2,self.hankelMat2Known)
        N.testing.assert_array_almost_equal(myERA.LSingVecs,self.LSingVecsKnown)
        N.testing.assert_array_almost_equal(N.squeeze(myERA.singVals),
          N.squeeze(self.singValsKnown))
        N.testing.assert_array_almost_equal(myERA.RSingVecs,self.RSingVecsKnown)
        
        myERANoSVD = era.ERA()
        myERANoSVD.load_decomp(hankelMatPath=hankelMatPath,
          hankelMat2Path = hankelMat2Path,numInputs = self.numInputs, 
          numOutputs=self.numOutputs)
        
        N.testing.assert_array_almost_equal(myERANoSVD.hankelMat,self.hankelMatKnown)
        N.testing.assert_array_almost_equal(myERANoSVD.hankelMat2,self.hankelMat2Known)
        N.testing.assert_array_almost_equal(myERANoSVD.LSingVecs,self.LSingVecsKnown)
        N.testing.assert_array_almost_equal(N.squeeze(myERANoSVD.singVals),
          N.squeeze(self.singValsKnown))
        N.testing.assert_array_almost_equal(myERANoSVD.RSingVecs,self.RSingVecsKnown)     
        
        
        
    def test_save_ROM(self):
        """Test can save ROM matrices A,B,C correctly"""
        myERA = era.ERA()
        myERA.A = copy.deepcopy(self.AKnown)
        myERA.B = copy.deepcopy(self.BKnown)
        myERA.C = copy.deepcopy(self.CKnown)
        APathComputed = self.testDirectory+'AComputed.txt'
        BPathComputed = self.testDirectory+'BComputed.txt'
        CPathComputed = self.testDirectory+'CComputed.txt'
        myERA.save_ROM(APathComputed,BPathComputed,CPathComputed)
        
        ALoaded = myERA.load_mat(APathComputed)
        BLoaded = myERA.load_mat(BPathComputed)
        CLoaded = myERA.load_mat(CPathComputed)
        
        N.testing.assert_array_almost_equal(ALoaded,self.AKnown)
        N.testing.assert_array_almost_equal(BLoaded,self.BKnown)
        N.testing.assert_array_almost_equal(CLoaded,self.CKnown)

if __name__ =='__main__':
    unittest.main(verbosity=2)
        
        
    


