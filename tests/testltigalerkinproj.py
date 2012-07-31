#!/usr/bin/env python
"""Test ltigalerkinproj module"""

import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path(join(join(os.path.abspath(__file__), '..', 'src')))
import parallel as parallel_mod
parallel = parallel_mod.parallel_default_instance

import ltigalerkinproj as LGP
import util
import vectors as V


#@unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
class TestLTIGalerkinProjection(unittest.TestCase):
    """Tests that can find the correct A, B, and C matrices from modes."""   
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
            
        self.test_dir ='DELETE_ME_test_files_bpodltirom'
        if parallel.is_rank_zero() and not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        parallel.barrier()

        self.direct_mode_path = join(self.test_dir, 'direct_mode_%02d.txt')
        self.adjoint_mode_path = join(self.test_dir, 'adjoint_mode_%02d.txt')
        self.A_on_direct_mode_path = join(self.test_dir,
            'A_on_mode_%02d.txt')
        self.B_on_basis_path = join(self.test_dir, 'B_on_basis_%02d.txt')
        self.C_on_direct_mode_path = join(self.test_dir, 'C_on_mode_%02d.txt')
        
        self.num_direct_modes = 10
        self.num_adjoint_modes = 8
        self.model_dim = 7
        self.num_states = 11
        self.num_inputs = 3
        self.num_outputs = 2
        
        self.generate_data_set(self.num_direct_modes, self.num_adjoint_modes,
            self.model_dim, self.num_states, self.num_inputs, 
            self.num_outputs)
        
        self.LTI_proj = LGP.LTIGalerkinProjection(N.vdot, 
            self.direct_mode_handles,
            self.adjoint_mode_handles, are_modes_orthonormal=True, verbosity=0)
            
        self.LTI_proj_in_memory = LGP.LTIGalerkinProjection(N.vdot, 
            self.direct_modes, self.adjoint_modes, are_modes_orthonormal=True, 
            verbosity=0)
        
        
    def tearDown(self):
        parallel.barrier()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.barrier()
        
    def test_init(self):
        """ """
        pass
        
    def generate_data_set(self, num_direct_modes, num_adjoint_modes,
        model_dim, num_states, num_inputs, num_outputs):
        """Generates random data, saves, and computes true reduced A,B,C."""
        self.direct_mode_handles = [
            V.ArrayTextVecHandle(self.direct_mode_path%i)
            for i in range(self.num_direct_modes)]
        self.adjoint_mode_handles = [
            V.ArrayTextVecHandle(self.adjoint_mode_path%i)
            for i in range(self.num_adjoint_modes)]
        self.A_on_direct_mode_handles = \
            [V.ArrayTextVecHandle(self.A_on_direct_mode_path%i) 
                for i in range(self.num_direct_modes)]
        self.B_on_basis_handles = [V.ArrayTextVecHandle(self.B_on_basis_path%i)
            for i in range(self.num_inputs)]
        self.C_on_direct_mode_handles = [
            V.ArrayTextVecHandle(self.C_on_direct_mode_path%i)
            for i in range(self.num_direct_modes)]
        if parallel.is_rank_zero():
            self.direct_mode_array = N.random.random((num_states, 
                num_direct_modes))
            self.adjoint_mode_array = N.random.random((num_states, 
                num_adjoint_modes))
            self.A_array = N.random.random((num_states, num_states))
            self.B_array = N.random.random((num_states, num_inputs))
            self.C_array = N.random.random((num_outputs, num_states))
        else:
            self.direct_mode_array = None
            self.adjoint_mode_array = None
            self.A_array = None
            self.B_array = None
            self.C_array = None
        if parallel.is_distributed():
            self.direct_mode_array = parallel.comm.bcast(self.direct_mode_array, 
                root=0)
            self.adjoint_mode_array = parallel.comm.bcast(
                self.adjoint_mode_array, root=0)
            self.A_array = parallel.comm.bcast(self.A_array, root=0)
            self.B_array = parallel.comm.bcast(self.B_array, root=0)
            self.C_array = parallel.comm.bcast(self.C_array, root=0)
            
        self.direct_modes = [self.direct_mode_array[:, i].squeeze()
            for i in range(num_direct_modes)]
        self.adjoint_modes = [self.adjoint_mode_array[:, i].squeeze()
            for i in range(num_adjoint_modes)]
        self.A_on_direct_modes = [N.dot(self.A_array, direct_mode).squeeze() 
            for direct_mode in self.direct_modes]
        self.B_on_basis = [self.B_array[:, i].squeeze()
            for i in range(self.num_inputs)]
        self.C_on_direct_modes = [N.array(
            N.dot(self.C_array, direct_mode).squeeze(), ndmin=1)
            for direct_mode in self.direct_modes]

        if parallel.is_rank_zero():
            for handle,vec in zip(self.direct_mode_handles, self.direct_modes):
                handle.put(vec)
            for handle,vec in zip(self.adjoint_mode_handles, self.adjoint_modes):
                handle.put(vec)
            for handle,vec in zip(self.A_on_direct_mode_handles,
                self.A_on_direct_modes):
                handle.put(vec)
            for handle,vec in zip(self.B_on_basis_handles, self.B_on_basis):
                handle.put(vec)
            for handle,vec in zip(self.C_on_direct_mode_handles,
                self.C_on_direct_modes):
                handle.put(vec)
        parallel.barrier()
        
        self.A_true = N.dot(
            self.adjoint_mode_array.T, 
            N.dot(self.A_array, self.direct_mode_array)
            )[:model_dim,:model_dim]
        self.B_true = N.dot(self.adjoint_mode_array.T, 
            self.B_array)[:model_dim, :]
        self.C_true = N.dot(self.C_array, 
            self.direct_mode_array)[:,:model_dim]
        self.proj_mat = N.linalg.inv(N.dot(self.adjoint_mode_array.T,
            self.direct_mode_array)[:self.model_dim, :self.model_dim])
        self.A_true_nonorth = N.dot(self.proj_mat, self.A_true)
        self.B_true_nonorth = N.dot(self.proj_mat, self.B_true)
        
        
    #@unittest.skip('testing others')
    def test_derivs(self):
        """Test can take derivs"""
        dt = 0.1
        true_derivs = []
        num_vecs = len(self.direct_mode_handles)
        for i in range(num_vecs):
            true_derivs.append((self.A_on_direct_mode_handles[i].get() - 
                self.direct_mode_handles[i].get()).squeeze()/dt)
        deriv_handles = [V.ArrayTextVecHandle(join(self.test_dir, 
            'deriv_test%d'%i))
            for i in range(num_vecs)]
        LGP.compute_derivs(self.direct_mode_handles, 
            self.A_on_direct_mode_handles, deriv_handles, dt)
        derivs_loaded = [v.get() for v in deriv_handles]
        derivs_returned = LGP.compute_derivs_in_memory(
            self.direct_modes, self.A_on_direct_modes, dt)
        derivs_loaded = map(N.squeeze, derivs_loaded)
        map(N.testing.assert_allclose, derivs_loaded, true_derivs)
        map(N.testing.assert_allclose, derivs_returned, true_derivs)
    
    #@unittest.skip('testing others')
    def test_reduce_A(self):
        """Reduction of A matrix for Matrix, LookUp operators and in_memory."""
        A_reduced_path = join(self.test_dir, 'A.txt')
        
        # Matrix multiplication operator A with vecs  
        A = LGP.MatrixOperator(self.A_array) 
        A_returned_in_mem = self.LTI_proj_in_memory.reduce_A_in_memory(A,
            model_dim=self.model_dim)
        N.testing.assert_allclose(A_returned_in_mem, self.A_true)
        
        # Precomputed operations A object with vecs
        A = LGP.LookUpOperator(self.direct_modes,
            self.A_on_direct_modes)
        A_returned = self.LTI_proj_in_memory.reduce_A_in_memory(A, 
            model_dim=self.model_dim)
        N.testing.assert_allclose(A_returned_in_mem, self.A_true)
        
        # Precomputed operations A object with vec handles
        A = LGP.LookUpOperator(self.direct_mode_handles,
            self.A_on_direct_mode_handles)
        A_returned = self.LTI_proj.reduce_A(A, model_dim=self.model_dim)
        N.testing.assert_allclose(A_returned, self.A_true)
        
        # Precomputed operations A object with vec handles, non-orthonormal modes
        LTI_proj = LGP.LTIGalerkinProjection(N.vdot, self.direct_mode_handles,
            self.adjoint_mode_handles, are_modes_orthonormal=False, verbosity=0)
        A = LGP.LookUpOperator(self.direct_mode_handles,
            self.A_on_direct_mode_handles)
        A_returned = LTI_proj.reduce_A(A, model_dim=self.model_dim)
        N.testing.assert_allclose(LTI_proj._proj_mat, self.proj_mat)
        N.testing.assert_allclose(A_returned, self.A_true_nonorth)
        
        
        
    #@unittest.skip('testing others')
    def test_reduce_B(self):
        """Given modes, test reduced B matrix"""
        B_reduced_path = join(self.test_dir, 'B.txt')
        
        # Matrix multiplication operator B with vecs  
        B = LGP.MatrixOperator(self.B_array) 
        B_returned = self.LTI_proj_in_memory.reduce_B_in_memory(B,
            self.num_inputs, model_dim=self.model_dim)
        N.testing.assert_allclose(B_returned, self.B_true)
        
        # Precomputed operations B object with vecs
        B = LGP.LookUpOperator(LGP.standard_basis(self.num_inputs),
            self.B_on_basis)
        B_returned = self.LTI_proj_in_memory.reduce_B_in_memory(B, self.num_inputs,
            model_dim=self.model_dim)
        N.testing.assert_allclose(B_returned, self.B_true)
        
        # Precomputed operations B object with vec handles
        B = LGP.LookUpOperator(LGP.standard_basis(self.num_inputs),
            self.B_on_basis_handles)
        B_returned = self.LTI_proj.reduce_B(B, self.num_inputs, 
            model_dim=self.model_dim)
        N.testing.assert_allclose(B_returned, self.B_true)
        
        # Precomputed operations B object with vec handles, non-orthonormal        
        B = LGP.LookUpOperator(LGP.standard_basis(self.num_inputs),
            self.B_on_basis_handles)
        LTI_proj = LGP.LTIGalerkinProjection(N.vdot,
            self.direct_mode_handles, self.adjoint_mode_handles,
            are_modes_orthonormal=False, verbosity=0)
        B_returned = LTI_proj.reduce_B(B, self.num_inputs, 
            model_dim=self.model_dim)
        N.testing.assert_allclose(B_returned, self.B_true_nonorth)
        
        
        
    #@unittest.skip('testing others')
    def test_reduce_C(self):
        """Test that, given modes, can find correct C matrix"""
        # Matrix multiplication operator C with vecs  
        C = LGP.MatrixOperator(self.C_array) 
        C_returned = self.LTI_proj_in_memory.reduce_C(C,
            model_dim=self.model_dim)
        N.testing.assert_allclose(C_returned, self.C_true)
        
        # Precomputed operations C object with vec handles
        C = LGP.LookUpOperator(self.direct_mode_handles,
            self.C_on_direct_modes)
        C_returned = self.LTI_proj.reduce_C(C,
            model_dim=self.model_dim)
        N.testing.assert_allclose(C_returned, self.C_true)
        
        # Precomputed operations C object with vecs
        C = LGP.LookUpOperator(self.direct_modes,
            self.C_on_direct_modes)
        C_returned = self.LTI_proj_in_memory.reduce_C_in_memory(C,
            model_dim=self.model_dim)
        N.testing.assert_allclose(C_returned, self.C_true)
            

    def test_compute_model(self):
        """Test that reduce_A, reduce_B, reduce_C give same as compute_model."""
        A = LGP.MatrixOperator(self.A_array) 
        A_true = self.LTI_proj_in_memory.reduce_A_in_memory(A, 
            model_dim=self.model_dim)
        B = LGP.MatrixOperator(self.B_array) 
        B_true = self.LTI_proj_in_memory.reduce_B_in_memory(B, self.num_inputs)
        C = LGP.MatrixOperator(self.C_array) 
        C_true = self.LTI_proj_in_memory.reduce_C_in_memory(C)
            
        new_LTI_proj = LGP.LTIGalerkinProjection(N.vdot, self.direct_modes,
            self.adjoint_modes, are_modes_orthonormal=True, verbosity=0)
        A_test, B_test, C_test = new_LTI_proj.compute_model_in_memory(A, B, C,
            self.num_inputs, model_dim=self.model_dim)
        N.testing.assert_equal(A_test, A_true)
        N.testing.assert_equal(B_test, B_true)
        N.testing.assert_equal(C_test, C_true)
        
    def test_adjoint_mode_optional(self):
        """Test that adjoint modes default to direct modes"""
        A = LGP.MatrixOperator(self.A_array) 
        B = LGP.MatrixOperator(self.B_array) 
        C = LGP.MatrixOperator(self.C_array) 

        no_adjoints_LTI_proj = LGP.LTIGalerkinProjection(N.vdot, self.direct_modes,
            are_modes_orthonormal=True, verbosity=0)
        adjoints_LTI_proj = LGP.LTIGalerkinProjection(N.vdot, self.direct_modes,
            adjoint_modes=self.direct_modes, are_modes_orthonormal=True, verbosity=0)
        A_no, B_no, C_no = no_adjoints_LTI_proj.compute_model_in_memory(A, B, C,
            self.num_inputs, model_dim=self.model_dim)
        A_with, B_with, C_with = adjoints_LTI_proj.compute_model_in_memory(A, B, C,
            self.num_inputs, model_dim=self.model_dim)
        N.testing.assert_equal(A_no, A_with)
        N.testing.assert_equal(B_no, B_with)
        N.testing.assert_equal(C_no, C_with)


    
    def test_put_reduced_mats(self):
        """Test putting reduced mats"""
        A_reduced_path = join(self.test_dir, 'A.txt')
        B_reduced_path = join(self.test_dir, 'B.txt')
        C_reduced_path = join(self.test_dir, 'C.txt')
        self.LTI_proj.A_reduced = N.copy(self.A_array)
        self.LTI_proj.B_reduced = N.copy(self.B_array)
        self.LTI_proj.C_reduced = N.copy(self.C_array)
        self.LTI_proj.put_model(A_reduced_path, B_reduced_path, C_reduced_path)
        N.testing.assert_equal(util.load_array_text(A_reduced_path), self.A_array)
        N.testing.assert_equal(util.load_array_text(B_reduced_path), self.B_array)
        N.testing.assert_equal(util.load_array_text(C_reduced_path), self.C_array)
        
       
if __name__ == '__main__':
    unittest.main()
    
