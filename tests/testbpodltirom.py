#!/usr/bin/env python

import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

import bpodltirom as BPR
import util
import vectors as V

@unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
class TestBPODROM(unittest.TestCase):
    """
    Tests that can find the correct A, B, and C matrices from modes
    """   
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
            
        self.test_dir ='DELETE_ME_test_files_bpodltirom'
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

        self.direct_mode_path = join(self.test_dir, 'direct_mode_%03d.txt')
        self.adjoint_mode_path = join(self.test_dir, 'adjoint_mode_%03d.txt')
        self.direct_deriv_mode_path =join(self.test_dir,
            'direct_deriv_mode_%03d.txt')
        self.input_vec_path = join(self.test_dir, 'input_vec_%03d.txt')
        self.output_vec_path = join(self.test_dir, 'output_vec_%03d.txt')
        
        self.myBPODROM = BPR.BPODROM(inner_product=N.vdot, verbose=False)
            
        self.num_direct_modes = 10
        self.num_adjoint_modes = 8
        self.num_ROM_modes = 7
        self.num_states = 10
        self.num_inputs = 2
        self.num_outputs = 3
        
        self.generate_data_set(self.num_direct_modes, self.num_adjoint_modes,
            self.num_ROM_modes, self.num_states, self.num_inputs, 
            self.num_outputs)
        
    def tearDown(self):
        rmtree(self.test_dir, ignore_errors=True)
        
    def test_init(self):
        """ """
        pass
        
    def generate_data_set(self, num_direct_modes, num_adjoint_modes,
        num_ROM_modes, num_states, num_inputs, num_outputs):
        """
        Generates random data, saves to file, and computes corect A,B,C.
        """
        self.direct_mode_handles = [V.ArrayTextHandle(self.direct_mode_path%i)
            for i in range(self.num_direct_modes)]
        self.direct_deriv_mode_handles = \
            [V.ArrayTextHandle(self.direct_deriv_mode_path%i) 
                for i in range(self.num_direct_modes)]
        self.adjoint_mode_handles = [V.ArrayTextHandle(self.adjoint_mode_path%i)
            for i in range(self.num_adjoint_modes)]
        
        self.input_vec_handles = [V.ArrayTextHandle(self.input_vec_path%i)
            for i in range(self.num_inputs)]
        self.output_vec_handles = [V.ArrayTextHandle(self.output_vec_path%i)
            for i in range(self.num_outputs)]
        
        self.direct_mode_mat = N.random.random((num_states, num_direct_modes))
        self.direct_deriv_mode_mat = \
            N.random.random((num_states, num_direct_modes))
              
        self.adjoint_mode_mat = N.random.random((num_states, num_adjoint_modes))
        self.input_mat = N.random.random((num_states, num_inputs))
        self.output_mat = N.random.random((num_states, num_outputs))
        
        for i,handle in enumerate(self.direct_mode_handles):
            handle.put(self.direct_mode_mat[:,i])
        for i,handle in enumerate(self.direct_deriv_mode_handles):
            handle.put(self.direct_deriv_mode_mat[:,i])
        for i,handle in enumerate(self.adjoint_mode_handles):
            handle.put(self.adjoint_mode_mat[:,i])
        for i,handle in enumerate(self.input_vec_handles):
            handle.put(self.input_mat[:,i])
        for i,handle in enumerate(self.output_vec_handles):
            handle.put(self.output_mat[:,i])     
        
        self.A_true = N.dot(self.adjoint_mode_mat.T, self.direct_deriv_mode_mat)[
            :num_ROM_modes,:num_ROM_modes]
        self.B_true = N.dot(self.adjoint_mode_mat.T, self.input_mat)[:num_ROM_modes,:]
        self.C_true = N.dot(self.output_mat.T, self.direct_mode_mat)[:,:num_ROM_modes]
        
        
    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')    
    def test_compute_A(self):
        """Test that, given modes, can find correct A matrix."""
        A_path = join(self.test_dir, 'A.txt')
        self.myBPODROM.compute_A(A_path, self.direct_deriv_mode_handles,
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        A_returned = self.myBPODROM.compute_A_and_return(
            self.direct_deriv_mode_handles,
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        N.testing.assert_allclose(self.A_true, util.load_array_text(A_path))
        N.testing.assert_allclose(self.A_true, A_returned)



    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
    def test_compute_B(self):
        """
        Test that, given modes, can find correct B matrix
        """
        B_path = join(self.test_dir, 'B.txt')
        self.myBPODROM.compute_B(B_path, self.input_vec_handles,
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        B_returned = self.myBPODROM.compute_B_and_return(
            self.input_vec_handles,
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        N.testing.assert_allclose(self.B_true, util.load_array_text(B_path))
        N.testing.assert_allclose(self.B_true, B_returned)


    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
    def test_compute_C(self):
        """
        Test that, given modes, can find correct C matrix
        """
        C_path = join(self.test_dir, 'C.txt')
        self.myBPODROM.compute_C(C_path, self.output_vec_handles,
            self.direct_mode_handles, num_modes=self.num_ROM_modes)
        C_returned = self.myBPODROM.compute_C_and_return(self.output_vec_handles,
            self.direct_mode_handles, num_modes=self.num_ROM_modes)
        N.testing.assert_allclose(self.C_true, util.load_array_text(C_path))
        N.testing.assert_allclose(self.C_true, C_returned)



if __name__ == '__main__':
    unittest.main()
    
