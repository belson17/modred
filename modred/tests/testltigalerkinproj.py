#!/usr/bin/env python
"""Test ltigalerkinproj module"""
from __future__ import division
from future.builtins import zip
from future.builtins import map
from future.builtins import range
import unittest
import os
from os.path import join
from shutil import rmtree

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
import modred.ltigalerkinproj as LGP
from modred import util
import modred.vectors as V


class TestLTIGalerkinProjectionBase(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir ='DELETE_ME_test_files_ltigalerkinproj'
        if _parallel.is_rank_zero() and not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        _parallel.barrier()
    
    def tearDown(self):
        _parallel.barrier()
        _parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        _parallel.barrier()
    
    def test_put_reduced_mats(self):
        """Test putting reduced mats"""        
        A_reduced_path = join(self.test_dir, 'A.txt')
        B_reduced_path = join(self.test_dir, 'B.txt')
        C_reduced_path = join(self.test_dir, 'C.txt')
        A = _parallel.call_and_bcast(np.random.random, ((10,10)))
        B = _parallel.call_and_bcast(np.random.random, ((1,10)))
        C = _parallel.call_and_bcast(np.random.random, ((10,2)))
        LTI_proj = LGP.LTIGalerkinProjectionBase()
        LTI_proj.A_reduced = A.copy()
        LTI_proj.B_reduced = B.copy()
        LTI_proj.C_reduced = C.copy()
        LTI_proj.put_model(A_reduced_path, B_reduced_path, C_reduced_path)
        np.testing.assert_equal(util.load_array_text(A_reduced_path), A)
        np.testing.assert_equal(util.load_array_text(B_reduced_path), B)
        np.testing.assert_equal(util.load_array_text(C_reduced_path), C)


@unittest.skipIf(_parallel.is_distributed(), 'Serial only')
class TestLTIGalerkinProjectionMatrices(unittest.TestCase):
    """Tests that can find the correct A, B, and C matrices."""   
    def setUp(self):
        self.num_basis_vecs = 10
        self.num_adjoint_basis_vecs = 10
        self.num_states = 11
        self.num_inputs = 3
        self.num_outputs = 2
        
        self.generate_data_set(self.num_basis_vecs, self.num_adjoint_basis_vecs,
            self.num_states, self.num_inputs, self.num_outputs)
        
        self.LTI_proj = LGP.LTIGalerkinProjectionMatrices(self.basis_vecs, 
            self.adjoint_basis_vecs, is_basis_orthonormal=True)
        
    def tearDown(self):
        pass
        
    def test_init(self):
        """ """
        pass
        
    def generate_data_set(self, num_basis_vecs, num_adjoint_basis_vecs,
        num_states, num_inputs, num_outputs):
        """Generates random data, saves, and computes true reduced A,B,C."""
        self.basis_vecs = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_basis_vecs))
        self.adjoint_basis_vecs = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_adjoint_basis_vecs))
        self.A_array = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_states))
        self.B_array = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_inputs))
        self.C_array = _parallel.call_and_bcast(np.random.random, 
            (num_outputs, num_states))
            
        self.A_on_basis_vecs = np.dot(self.A_array, self.basis_vecs)
        self.B_on_standard_basis_array = self.B_array
        self.C_on_basis_vecs = self.C_array.dot(self.basis_vecs).squeeze()

        _parallel.barrier()
        
        self.A_true = np.dot(self.adjoint_basis_vecs.T, 
            np.dot(self.A_array, self.basis_vecs))
        self.B_true = np.dot(self.adjoint_basis_vecs.T, self.B_array)
        self.C_true = np.dot(self.C_array, self.basis_vecs)
        self.proj_mat = np.linalg.inv(np.dot(self.adjoint_basis_vecs.T,
            self.basis_vecs))
        self.A_true_nonorth = np.dot(self.proj_mat, self.A_true)
        self.B_true_nonorth = np.dot(self.proj_mat, self.B_true)
        
    #@unittest.skip('testing others')
    def test_reduce_A(self):
        """Reduction of A matrix for Matrix, LookUp operators and in_memory."""
        A_returned = self.LTI_proj.reduce_A(self.A_on_basis_vecs)
        np.testing.assert_allclose(A_returned, self.A_true)
        
        # Precomputed operations A object with vec handles, non-orthonormal
        # modes
        LTI_proj = LGP.LTIGalerkinProjectionMatrices(self.basis_vecs,
            self.adjoint_basis_vecs, is_basis_orthonormal=False)
        A_returned = LTI_proj.reduce_A(self.A_on_basis_vecs)
        np.testing.assert_allclose(LTI_proj._proj_mat, self.proj_mat)
        np.testing.assert_allclose(A_returned, self.A_true_nonorth)
        
    #@unittest.skip('testing others')
    def test_reduce_B(self):
        """Given modes, test reduced B matrix"""
        B_returned = self.LTI_proj.reduce_B(self.B_on_standard_basis_array)
        np.testing.assert_allclose(B_returned, self.B_true)
        
        LTI_proj = LGP.LTIGalerkinProjectionMatrices(self.basis_vecs, 
            self.adjoint_basis_vecs, is_basis_orthonormal=False)
        B_returned = LTI_proj.reduce_B(self.B_on_standard_basis_array)
        np.testing.assert_allclose(B_returned, self.B_true_nonorth)
        
    #@unittest.skip('testing others')
    def test_reduce_C(self):
        """Test that, given modes, can find correct C matrix"""
        C_returned = self.LTI_proj.reduce_C(self.C_on_basis_vecs)
        np.testing.assert_allclose(C_returned, self.C_true)
    
    def test_compute_model(self):
        A,B,C = self.LTI_proj.compute_model(self.A_on_basis_vecs,
            self.B_on_standard_basis_array, self.C_on_basis_vecs)
        # np. test, just check it runs. Results are checked in other tests.
        
    def test_adjoint_basis_vec_optional(self):
        """Test that adjoint modes default to direct modes"""
        no_adjoints_LTI_proj = LGP.LTIGalerkinProjectionMatrices(
            self.basis_vecs, is_basis_orthonormal=True)
        np.testing.assert_equal(no_adjoints_LTI_proj.adjoint_basis_vecs, 
            self.basis_vecs)

        
#@unittest.skipIf(_parallel.is_distributed(), 'Only test in serial')
#@unittest.skip('others')
class TestLTIGalerkinProjectionHandles(unittest.TestCase):
    """Tests that can find the correct A, B, and C matrices from modes."""   
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
            
        self.test_dir ='DELETE_ME_test_files_ltigalerkinproj'
        if _parallel.is_rank_zero() and not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        _parallel.barrier()

        self.basis_vec_path = join(self.test_dir, 'basis_vec_%02d.txt')
        self.adjoint_basis_vec_path = join(
            self.test_dir, 'adjoint_basis_vec_%02d.txt')
        self.A_on_basis_vec_path = join(self.test_dir, 'A_on_mode_%02d.txt')
        self.B_on_basis_path = join(self.test_dir, 'B_on_basis_%02d.txt')
        self.C_on_basis_vec_path = join(self.test_dir, 'C_on_mode_%02d.txt')
        
        self.num_basis_vecs = 10
        self.num_adjoint_basis_vecs = 10
        self.num_states = 11
        self.num_inputs = 3
        self.num_outputs = 2
        
        self.generate_data_set(self.num_basis_vecs, self.num_adjoint_basis_vecs,
            self.num_states, self.num_inputs, self.num_outputs)
        
        self.LTI_proj = LGP.LTIGalerkinProjectionHandles(
            np.vdot, self.basis_vec_handles, self.adjoint_basis_vec_handles,
            is_basis_orthonormal=True, verbosity=0)
        
    def tearDown(self):
        _parallel.barrier()
        if _parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        _parallel.barrier()
        
    def test_init(self):
        """ """
        pass
        
    def generate_data_set(self, num_basis_vecs, num_adjoint_basis_vecs,
        num_states, num_inputs, num_outputs):
        """Generates random data, saves, and computes true reduced A,B,C."""
        self.basis_vec_handles = [
            V.VecHandleArrayText(self.basis_vec_path%i)
            for i in range(self.num_basis_vecs)]
        self.adjoint_basis_vec_handles = [
            V.VecHandleArrayText(self.adjoint_basis_vec_path%i)
            for i in range(self.num_adjoint_basis_vecs)]
        self.A_on_basis_vec_handles = \
            [V.VecHandleArrayText(self.A_on_basis_vec_path%i) 
                for i in range(self.num_basis_vecs)]
        self.B_on_standard_basis_handles = [
            V.VecHandleArrayText(self.B_on_basis_path%i)
            for i in range(self.num_inputs)]
        self.C_on_basis_vec_handles = [
            V.VecHandleArrayText(self.C_on_basis_vec_path%i)
            for i in range(self.num_basis_vecs)]
            
        self.basis_vec_array = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_basis_vecs))
        self.adjoint_basis_vec_array = _parallel.call_and_bcast(
            np.random.random, (num_states, num_adjoint_basis_vecs))
        self.A_array = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_states))
        self.B_array = _parallel.call_and_bcast(np.random.random, 
            (num_states, num_inputs))
        self.C_array = _parallel.call_and_bcast(np.random.random, 
            (num_outputs, num_states))
            
        self.basis_vecs = [self.basis_vec_array[:, i].squeeze()
            for i in range(num_basis_vecs)]
        self.adjoint_basis_vecs = [self.adjoint_basis_vec_array[:, i].squeeze()
            for i in range(num_adjoint_basis_vecs)]
        self.A_on_basis_vecs = [np.dot(self.A_array, basis_vec).squeeze() 
            for basis_vec in self.basis_vecs]
        self.B_on_basis = [self.B_array[:, i].squeeze()
            for i in range(self.num_inputs)]
        self.C_on_basis_vecs = [np.array(
            np.dot(self.C_array, basis_vec).squeeze(), ndmin=1)
            for basis_vec in self.basis_vecs]

        if _parallel.is_rank_zero():
            for handle,vec in zip(self.basis_vec_handles, self.basis_vecs):
                handle.put(vec)
            for handle,vec in zip(
                self.adjoint_basis_vec_handles, self.adjoint_basis_vecs):
                handle.put(vec)
            for handle,vec in zip(self.A_on_basis_vec_handles,
                self.A_on_basis_vecs):
                handle.put(vec)
            for handle,vec in zip(
                self.B_on_standard_basis_handles, self.B_on_basis):
                handle.put(vec)
            for handle,vec in zip(self.C_on_basis_vec_handles,
                self.C_on_basis_vecs):
                handle.put(vec)
        _parallel.barrier()
        
        self.A_true = np.dot(self.adjoint_basis_vec_array.T, 
            np.dot(self.A_array, self.basis_vec_array))
        self.B_true = np.dot(self.adjoint_basis_vec_array.T, self.B_array)
        self.C_true = np.dot(self.C_array, self.basis_vec_array)
        self.proj_mat = np.linalg.inv(np.dot(self.adjoint_basis_vec_array.T,
            self.basis_vec_array))
        self.A_true_nonorth = np.dot(self.proj_mat, self.A_true)
        self.B_true_nonorth = np.dot(self.proj_mat, self.B_true)
        
    #@unittest.skip('testing others')
    def test_derivs(self):
        """Test can take derivs"""
        dt = 0.1
        true_derivs = []
        num_vecs = len(self.basis_vec_handles)
        for i in range(num_vecs):
            true_derivs.append((self.A_on_basis_vec_handles[i].get() - 
                self.basis_vec_handles[i].get()).squeeze()/dt)
        deriv_handles = [V.VecHandleArrayText(join(self.test_dir, 
            'deriv_test%d'%i))
            for i in range(num_vecs)]
        LGP.compute_derivs_handles(self.basis_vec_handles, 
            self.A_on_basis_vec_handles, deriv_handles, dt)
        derivs_loaded = [v.get() for v in deriv_handles]
        derivs_loaded = list(map(np.squeeze, derivs_loaded))
        list(map(np.testing.assert_allclose, derivs_loaded, true_derivs))
        
    #@unittest.skip('testing others')
    def test_reduce_A(self):
        """Reduction of A matrix for Matrix, LookUp operators and in_memory."""
        # Precomputed operations A object with vec handles
        A_returned = self.LTI_proj.reduce_A(self.A_on_basis_vec_handles)
        np.testing.assert_allclose(A_returned, self.A_true)
        
        # Precomputed operations A object with vec handles, non-orthonormal
        # modes
        LTI_proj = LGP.LTIGalerkinProjectionHandles(
            np.vdot, self.basis_vec_handles, self.adjoint_basis_vec_handles,
            is_basis_orthonormal=False, verbosity=0)
        A_returned = LTI_proj.reduce_A(self.A_on_basis_vec_handles)
        np.testing.assert_allclose(LTI_proj._proj_mat, self.proj_mat)
        np.testing.assert_allclose(A_returned, self.A_true_nonorth)
        
    #@unittest.skip('testing others')
    def test_reduce_B(self):
        """Given modes, test reduced B matrix, orthogonal and non-orthogonal."""
        B_returned = self.LTI_proj.reduce_B(self.B_on_standard_basis_handles)
        np.testing.assert_allclose(B_returned, self.B_true)
        
        LTI_proj = LGP.LTIGalerkinProjectionHandles(np.vdot,
            self.basis_vec_handles, self.adjoint_basis_vec_handles,
            is_basis_orthonormal=False, verbosity=0)
        B_returned = LTI_proj.reduce_B(self.B_on_standard_basis_handles)
        np.testing.assert_allclose(B_returned, self.B_true_nonorth)
        
    #@unittest.skip('testing others')
    def test_reduce_C(self):
        """Test that, given modes, can find correct C matrix"""
        C_returned = self.LTI_proj.reduce_C(self.C_on_basis_vecs)
        np.testing.assert_allclose(C_returned, self.C_true)
        
    def test_adjoint_basis_vec_optional(self):
        """Test that adjoint modes default to direct modes"""
        no_adjoints_LTI_proj = LGP.LTIGalerkinProjectionHandles(np.vdot, 
            self.basis_vec_handles, is_basis_orthonormal=True, verbosity=0)
        np.testing.assert_equal(no_adjoints_LTI_proj.adjoint_basis_vec_handles, 
            self.basis_vec_handles)

       
if __name__ == '__main__':
    unittest.main()
    
