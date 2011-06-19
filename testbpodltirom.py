#!/usr/bin/env python
import subprocess as SP
import bpodltirom as BPR
import unittest
import numpy as N
import util

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        raise RuntimeError('Not implemented in parallel! Use "python testbpodltirom.py" ')
except ImportError:
    pass

class TestBPODROM(unittest.TestCase):
    """
    Tests that can find the correct A,B,C matrices from modes
    """
    
    def setUp(self):
        self.myBPODROM = BPR.BPODROM(save_mat=util.save_mat_text, load_field=\
            util.load_mat_text,inner_product=util.inner_product, save_field=\
            util.save_mat_text)
        self.directModePath = 'files_bpodltirom_test/direct_mode_%03d.txt'
        self.adjointModePath = 'files_bpodltirom_test/adjoint_mode_%03d.txt'
        self.directDerivModePath='files_bpodltirom_test/direct_deriv_mode_' +\
            '%03d.txt'
        self.inputPath = 'files_bpodltirom_test/input_%03d.txt'
        self.outputPath='files_bpodltirom_test/output_%03d.txt'
        
        self.numDirectModes = 10
        self.numAdjointModes = 8
        self.numROMModes = 7
        self.numStates = 10
        self.numInputs = 2
        self.numOutputs = 2
        
        self.generate_data_set(self.numDirectModes, self.numAdjointModes, self.\
            numROMModes, self.numStates, self.numInputs, self.numOutputs)

    def tearDown(self):
        SP.call(['rm -rf files_bpodltirom_test/*'], shell=True)
        
    def test_init(self):
        """ """
        pass
    
    def generate_data_set(self,numDirectModes,numAdjointModes,numROMModes,
      numStates,numInputs,numOutputs):
        """
        Generates random data, saves to file, and computes corect A,B,C.
        """
        self.directModePaths=[]
        self.directDerivModePaths=[]
        self.adjointModePaths=[]
        self.inputPaths=[]
        self.outputPaths=[]
        
        self.directModeMat = N.mat(
              N.random.random((numStates,numDirectModes)))
        self.directDerivModeMat = N.mat(
              N.random.random((numStates,numDirectModes)))
              
        self.adjointModeMat = N.mat(
              N.random.random((numStates,numAdjointModes))) 
        self.inputMat = N.mat(
              N.random.random((numStates,numInputs))) 
        self.outputMat = N.mat(
              N.random.random((numStates,numOutputs))) 
        
        for directModeNum in range(numDirectModes):
            util.save_mat_text(self.directModeMat[:,directModeNum],
              self.directModePath%directModeNum)
            util.save_mat_text(self.directDerivModeMat[:,directModeNum],
              self.directDerivModePath%directModeNum)
            self.directModePaths.append(self.directModePath%directModeNum)
            self.directDerivModePaths.append(self.directDerivModePath %\
                directModeNum)
            
        for adjointModeNum in range(self.numAdjointModes):
            util.save_mat_text(self.adjointModeMat[:,adjointModeNum],
              self.adjointModePath%adjointModeNum)
            self.adjointModePaths.append(self.adjointModePath%adjointModeNum)
        
        for inputNum in xrange(numInputs):
            self.inputPaths.append(self.inputPath%inputNum)
            util.save_mat_text(self.inputMat[:,inputNum],self.inputPaths[
                inputNum])
        for outputNum in xrange(numInputs):
            self.outputPaths.append(self.outputPath%outputNum)
            util.save_mat_text(self.outputMat[:,outputNum],self.outputPaths[
                outputNum])            
        
        self.ATrue = (self.adjointModeMat.T*self.directDerivModeMat)[
            :numROMModes,:numROMModes]
        self.BTrue = (self.adjointModeMat.T*self.inputMat)[:numROMModes,:]
        self.CTrue = (self.outputMat.T*self.directModeMat)[:,:numROMModes]
        
    
    def test_form_A(self):
        """
        Test that, given modes, can find correct A matrix
        """
        APath ='files_bpodltirom_test/A.txt'
        self.myBPODROM.form_A(APath, directDerivModePaths=self.\
            directDerivModePaths, adjointModePaths=self.adjointModePaths,
            numModes=self.numROMModes)
        N.testing.assert_array_almost_equal(self.ATrue,util.load_mat_text(
            APath))

    def test_form_B(self):
        """
        Test that, given modes, can find correct B matrix
        """
        BPath ='files_bpodltirom_test/B.txt'
        self.myBPODROM.form_B(BPath,self.inputPaths, adjointModePaths=self.\
            adjointModePaths, numModes=self.numROMModes)
        N.testing.assert_array_almost_equal(self.BTrue, util.load_mat_text(
            BPath))

    def test_form_C(self):
        """
        Test that, given modes, can find correct C matrix
        """
        CPath ='files_bpodltirom_test/C.txt'
        self.myBPODROM.form_C(CPath,self.outputPaths,
        directModePaths=self.directModePaths,
        numModes=self.numROMModes)
        
        N.testing.assert_array_almost_equal(self.CTrue,util.load_mat_text(
            CPath))

if __name__ == '__main__':
    unittest.main(verbosity=2)
