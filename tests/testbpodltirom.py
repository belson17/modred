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
        self.direct_deriv_mode_path =join(self.test_dir, 'direct_deriv_mode_%03d.txt')
        self.input_vec_path = join(self.test_dir, 'input_vec_%03d.txt')
        self.output_vec_path = join(self.test_dir, 'output_vec_%03d.txt')
        
        self.myBPODROM = BPR.BPODROM(put_mat=util.save_mat_text, get_vec=\
            util.load_mat_text,inner_product=util.inner_product, put_vec=\
            util.save_mat_text, verbose=False)
            
        self.num_direct_modes = 10
        self.num_adjoint_modes = 8
        self.num_ROM_modes = 7
        self.num_states = 10
        self.num_inputs = 2
        self.num_outputs = 3
        self.dt = 0
        
        self.generate_data_set(self.num_direct_modes, self.num_adjoint_modes, \
            self.num_ROM_modes, self.num_states, self.num_inputs, self.num_outputs)
        
    def tearDown(self):
        rmtree(self.test_dir, ignore_errors=True)
        
        
    def test_init(self):
        """ """
        pass
        
    
    def generate_data_set(self,num_direct_modes,num_adjoint_modes,num_ROM_modes,
        num_states,num_inputs,num_outputs):
        """
        Generates random data, saves to file, and computes corect A,B,C.
        """
        self.direct_mode_paths=[]
        self.direct_deriv_mode_paths=[]
        self.adjoint_mode_paths=[]
        self.input_vec_paths=[]
        self.output_vec_paths=[]
        
        self.direct_mode_mat = N.mat(
              N.random.random((num_states, num_direct_modes)))
        self.direct_deriv_mode_mat = N.mat(
              N.random.random((num_states, num_direct_modes)))
              
        self.adjoint_mode_mat = N.mat(
              N.random.random((num_states, num_adjoint_modes))) 
        self.input_mat = N.mat(
              N.random.random((num_states, num_inputs))) 
        self.output_mat = N.mat(
              N.random.random((num_outputs, num_states))) 
        
        for direct_mode_num in range(num_direct_modes):
            util.save_mat_text(self.direct_mode_mat[:,direct_mode_num],
              self.direct_mode_path%direct_mode_num)
            util.save_mat_text(self.direct_deriv_mode_mat[:,direct_mode_num],
              self.direct_deriv_mode_path%direct_mode_num)
            self.direct_mode_paths.append(self.direct_mode_path%direct_mode_num)
            self.direct_deriv_mode_paths.append(self.direct_deriv_mode_path %\
                direct_mode_num)
            
        for adjoint_mode_num in range(self.num_adjoint_modes):
            util.save_mat_text(self.adjoint_mode_mat[:,adjoint_mode_num],
              self.adjoint_mode_path%adjoint_mode_num)
            self.adjoint_mode_paths.append(self.adjoint_mode_path%adjoint_mode_num)
        
        for input_num in xrange(num_inputs):
            self.input_vec_paths.append(self.input_vec_path%input_num)
            util.save_mat_text(self.input_mat[:,input_num],self.input_vec_paths[
                input_num])
        for output_num in xrange(num_outputs):
            self.output_vec_paths.append(self.output_vec_path%output_num)
            # TODO: Sort out why this has to be a transpose, something to do with IPs
            # and matrix sizes.
            util.save_mat_text(self.output_mat[output_num].T,self.output_vec_paths[
                output_num])            
        
        self.A_true = (self.adjoint_mode_mat.T*self.direct_deriv_mode_mat)[
            :num_ROM_modes,:num_ROM_modes]
        self.B_true = (self.adjoint_mode_mat.T*self.input_mat)[:num_ROM_modes,:]
        self.C_true = (self.output_mat*self.direct_mode_mat)[:,:num_ROM_modes]
        
        
    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')    
    def test_form_A(self):
        """Test that, given modes, can find correct A matrix."""
        A_path = join(self.test_dir, 'A.txt')
        self.myBPODROM.form_A(A_path, self.direct_deriv_mode_paths,
            self.adjoint_mode_paths, self.dt, num_modes=self.num_ROM_modes)
        N.testing.assert_allclose(self.A_true, util.load_mat_text(A_path))


    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
    def test_form_B(self):
        """
        Test that, given modes, can find correct B matrix
        """
        B_path = join(self.test_dir, 'B.txt')
        self.myBPODROM.form_B(B_path,self.input_vec_paths,\
            self.adjoint_mode_paths, self.dt, num_modes=self.num_ROM_modes)
        N.testing.assert_allclose(self.B_true, \
            util.load_mat_text(B_path))


    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
    def test_form_C(self):
        """
        Test that, given modes, can find correct C matrix
        """
        C_path = join(self.test_dir, 'C.txt')
        self.myBPODROM.form_C(C_path, self.output_vec_paths,
            self.direct_mode_paths, num_modes=self.num_ROM_modes)
        
        N.testing.assert_allclose(self.C_true, util.load_mat_text(C_path))


if __name__ == '__main__':
    unittest.main()
    
