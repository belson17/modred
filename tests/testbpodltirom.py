#!/usr/bin/env python

import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.parallel_default_instance

import bpodltirom as BPR
import util
import vectors as V

#@unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
class TestBPODROM(unittest.TestCase):
    """
    Tests that can find the correct A, B, and C matrices from modes
    """   
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
            
        self.test_dir ='DELETE_ME_test_files_bpodltirom'
        if parallel.is_rank_zero() and not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        parallel.barrier()

        self.direct_mode_path = join(self.test_dir, 'direct_mode_%03d.txt')
        self.adjoint_mode_path = join(self.test_dir, 'adjoint_mode_%03d.txt')
        self.A_times_direct_mode_path =join(self.test_dir,
            'direct_deriv_mode_%03d.txt')
        self.B_vec_path = join(self.test_dir, 'B_vec_%03d.txt')
        self.C_vec_path = join(self.test_dir, 'C_vec_%03d.txt')
        
        self.myBPODROM = BPR.BPODROM(N.vdot, verbosity=0)
            
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
        parallel.barrier()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.barrier()
        
    def test_init(self):
        """ """
        pass
        
    def generate_data_set(self, num_direct_modes, num_adjoint_modes,
        num_ROM_modes, num_states, num_inputs, num_outputs):
        """
        Generates random data, saves to file, and computes corect A,B,C.
        """
        self.direct_mode_handles = [V.ArrayTextVecHandle(self.direct_mode_path%i)
            for i in range(self.num_direct_modes)]
        self.A_times_direct_mode_handles = \
            [V.ArrayTextVecHandle(self.A_times_direct_mode_path%i) 
                for i in range(self.num_direct_modes)]
        self.adjoint_mode_handles = [V.ArrayTextVecHandle(self.adjoint_mode_path%i)
            for i in range(self.num_adjoint_modes)]
        
        self.B_vec_handles = [V.ArrayTextVecHandle(self.B_vec_path%i)
            for i in range(self.num_inputs)]
        self.C_vec_handles = [V.ArrayTextVecHandle(self.C_vec_path%i)
            for i in range(self.num_outputs)]
        if parallel.is_rank_zero():
            self.direct_mode_array = N.random.random((num_states, num_direct_modes))
            self.A_times_direct_mode_array = \
                N.random.random((num_states, num_direct_modes))      
            self.adjoint_mode_array = N.random.random((num_states, num_adjoint_modes))
            self.B_array = N.random.random((num_states, num_inputs))
            self.C_array = N.random.random((num_states, num_outputs))
        else:
            self.direct_mode_array = None
            self.A_times_direct_mode_array = None      
            self.adjoint_mode_array = None
            self.B_array = None
            self.C_array = None
        if parallel.is_distributed():
            self.direct_mode_array = parallel.comm.bcast(self.direct_mode_array, root=0)
            self.A_times_direct_mode_array = parallel.comm.bcast(self.A_times_direct_mode_array, root=0)
            self.adjoint_mode_array = parallel.comm.bcast(self.adjoint_mode_array, root=0)
            self.B_array = parallel.comm.bcast(self.B_array, root=0)
            self.C_array = parallel.comm.bcast(self.C_array, root=0)

        self.direct_modes = [self.direct_mode_array[:,i].squeeze()
            for i in range(num_direct_modes)]
        self.A_times_direct_modes = [self.A_times_direct_mode_array[:,i].squeeze() 
            for i in range(num_direct_modes)]
        self.adjoint_modes = [self.adjoint_mode_array[:,i].squeeze()
            for i in range(num_adjoint_modes)]
        self.B_vecs = [self.B_array[:,i].squeeze()
            for i in range(self.num_inputs)]
        self.C_vecs = [self.C_array[:,i].squeeze()
            for i in range(self.num_outputs)]

        if parallel.is_rank_zero():
            for i,handle in enumerate(self.direct_mode_handles):
                handle.put(self.direct_modes[i])
            for i,handle in enumerate(self.A_times_direct_mode_handles):
                handle.put(self.A_times_direct_modes[i])
            for i,handle in enumerate(self.adjoint_mode_handles):
                handle.put(self.adjoint_modes[i])
            for i,handle in enumerate(self.B_vec_handles):
                handle.put(self.B_array[:,i].squeeze())
            for i,handle in enumerate(self.C_vec_handles):
                handle.put(self.C_array[:,i].squeeze())
        parallel.barrier()
        
        self.A_true = N.dot(self.adjoint_mode_array.T, self.A_times_direct_mode_array)[
            :num_ROM_modes,:num_ROM_modes]
        self.B_true = N.dot(self.adjoint_mode_array.T, self.B_array)[:num_ROM_modes,:]
        self.C_true = N.dot(self.C_array.T, self.direct_mode_array)[:,:num_ROM_modes]
        
    def test_derivs(self):
        """Test can take derivs"""
        dt = 0.1
        true_derivs = []
        num_vecs = len(self.direct_mode_handles)
        for i in range(num_vecs):
            true_derivs.append((self.A_times_direct_mode_handles[i].get() - 
                self.direct_mode_handles[i].get()).squeeze()/dt)
        deriv_handles = [V.ArrayTextVecHandle(join(self.test_dir, 'deriv_test%d'%i))
            for i in range(num_vecs)]
        BPR.compute_derivs(self.direct_mode_handles, 
            self.A_times_direct_mode_handles, deriv_handles, dt)
        derivs_loaded = [v.get() for v in deriv_handles]
        derivs_returned = BPR.compute_derivs_in_memory(
            self.direct_modes, self.A_times_direct_modes, dt)
        derivs_loaded = map(N.squeeze, derivs_loaded)
        map(N.testing.assert_allclose, derivs_loaded, true_derivs)
        map(N.testing.assert_allclose, derivs_returned, true_derivs)
    
    
    def test_compute_A(self):
        """Test that, given modes, can find correct A matrix."""
        A_path = join(self.test_dir, 'A.txt')
        A_returned_in_mem = self.myBPODROM.compute_A_in_memory(
            self.A_times_direct_modes,
            self.adjoint_modes, num_modes=self.num_ROM_modes)
        A_returned = self.myBPODROM.compute_A(self.A_times_direct_mode_handles,
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        self.myBPODROM.put_A(A_path)
        A_returned_model, dum1, dum2 = self.myBPODROM.compute_model(
            self.A_times_direct_mode_handles, self.B_vec_handles,
            self.C_vec_handles, self.direct_mode_handles, 
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        A_returned_model_in_mem, dum1, dum2 = \
            self.myBPODROM.compute_model_in_memory(
            self.A_times_direct_modes, self.B_vecs,
            self.C_vecs, self.direct_modes, 
            self.adjoint_modes, num_modes=self.num_ROM_modes)
        parallel.barrier()
        N.testing.assert_allclose(util.load_array_text(A_path), self.A_true)
        N.testing.assert_allclose(A_returned, self.A_true)
        N.testing.assert_allclose(A_returned_in_mem, self.A_true)
        N.testing.assert_allclose(A_returned_model, self.A_true)
        N.testing.assert_allclose(A_returned_model_in_mem, self.A_true)



    def test_compute_B(self):
        """
        Test that, given modes, can find correct B matrix
        """
        B_path = join(self.test_dir, 'B.txt')
        B_returned_in_mem = self.myBPODROM.compute_B_in_memory(self.B_vecs,
            self.adjoint_modes, num_modes=self.num_ROM_modes)
        B_returned = self.myBPODROM.compute_B(self.B_vec_handles,
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        self.myBPODROM.put_B(B_path)
        dum1, B_returned_model, dum2 = self.myBPODROM.compute_model(
            self.A_times_direct_mode_handles, self.B_vec_handles,
            self.C_vec_handles, self.direct_mode_handles, 
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        dum1, B_returned_model_in_mem, dum2 = \
            self.myBPODROM.compute_model_in_memory(
            self.A_times_direct_modes, self.B_vecs,
            self.C_vecs, self.direct_modes, 
            self.adjoint_modes, num_modes=self.num_ROM_modes)
        parallel.barrier()
        N.testing.assert_allclose(util.load_array_text(B_path), self.B_true)
        N.testing.assert_allclose(B_returned, self.B_true)
        N.testing.assert_allclose(B_returned_in_mem, self.B_true)
        N.testing.assert_allclose(B_returned_model, self.B_true)
        N.testing.assert_allclose(B_returned_model_in_mem, self.B_true)


    def test_compute_C(self):
        """
        Test that, given modes, can find correct C matrix
        """
        C_path = join(self.test_dir, 'C.txt')
        C_returned_in_mem = self.myBPODROM.compute_C_in_memory(self.C_vecs,
            self.direct_modes, num_modes=self.num_ROM_modes)
        C_returned = self.myBPODROM.compute_C(self.C_vec_handles,
            self.direct_mode_handles, num_modes=self.num_ROM_modes)
        self.myBPODROM.put_C(C_path)
        dum1, dum2, C_returned_model = self.myBPODROM.compute_model(
            self.A_times_direct_mode_handles, self.B_vec_handles,
            self.C_vec_handles, self.direct_mode_handles, 
            self.adjoint_mode_handles, num_modes=self.num_ROM_modes)
        dum1, dum2, C_returned_model_in_mem = \
            self.myBPODROM.compute_model_in_memory(
            self.A_times_direct_modes, self.B_vecs,
            self.C_vecs, self.direct_modes, 
            self.adjoint_modes, num_modes=self.num_ROM_modes)
        parallel.barrier()
        N.testing.assert_allclose(util.load_array_text(C_path), self.C_true)
        N.testing.assert_allclose(C_returned, self.C_true)
        N.testing.assert_allclose(C_returned_in_mem, self.C_true)
        N.testing.assert_allclose(C_returned_model, self.C_true)
        N.testing.assert_allclose(C_returned_model_in_mem, self.C_true)



if __name__ == '__main__':
    unittest.main()
    
