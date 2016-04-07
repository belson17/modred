#!/usr/bin/env python
"""Test the bpod module"""
from __future__ import division
from future.builtins import zip
from future.builtins import range
import unittest
import copy
import os
from os.path import join
from shutil import rmtree

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
from modred.bpod import *
from modred.vectorspace import *
from modred import util
from modred import vectors as V


@unittest.skipIf(_parallel.is_distributed(),'Serial only.')
@unittest.skip('Testing others')
class TestBPODMatrices(unittest.TestCase):
    def setUp(self):
        self.mode_indices = [2, 4, 6, 3]
        self.num_direct_vecs = 10
        self.num_adjoint_vecs = 12
        self.num_states = 15
        num_inputs = 1
        num_outputs = 1
        self.A = np.mat(np.random.random((self.num_states, self.num_states)))
        self.B = np.mat(np.random.random((self.num_states, self.num_inputs)))
        self.C = np.mat(np.random.random((self.num_outputs, self.num_states)))
        self.direct_vecs = []
        A_powers = np.identity(A.shape[0])
        for t in range(self.num_direct_vecs):
            A_powers = A_powers.dot(A)
            self.direct_vecs.append(A_powers.dot(B))
        self.direct_vecs = np.array(self.direct_vecs).squeeze().T
        #self.direct_vecs = _parallel.call_and_bcast(np.random.random,
        #    (self.num_states, self.num_direct_vec_handles))
        #self.adjoint_vecs = _parallel.call_and_bcast(np.random.random,
        #    (self.num_states, self.num_adjoint_vec_handles)) 
        
    def test_compute_modes(self):
        """Test computing modes in serial and parallel."""
        tol = 1e-6
        ws = np.identity(self.num_states)
        ws[0,0] = 2
        ws[1,0] = 1.1
        ws[0,1] = 1.1
        weights_list = [None, np.random.random(self.num_states), ws]
        weights_mats = [
            np.mat(np.identity(self.num_states)), 
            np.mat(np.diag(weights_list[1])),
            np.mat(ws)]
        for weights, weights_mat in zip(weights_list, weights_mats):
            A_adjoint = np.linalg.inv(weights_mat) * self.A.H * weights_mat
            C_adjoint = np.linalg.inv(weights_mat) * self.C.H
            A_adjoint_powers = np.identity(A_adjoint.shape[0])
            adjoint_vecs = []
            for t in range(self.num_adjoint_vecs):
                A_adjoint_powers = A_adjoint_powers.dot(A_adjoint)
                adjoint_vecs.append(A_adjoint_powers.dot(C_adjoint))
            adjoint_vecs = np.array(adjoint_vecs).squeeze().T
        
            IP = VectorSpaceMatrices(weights=weights).compute_inner_product_mat
            Hankel_mat_true = IP(adjoint_vecs, self.direct_vecs)
            
            L_sing_vecs_true, sing_vals_true, R_sing_vecs_true = \
                _parallel.call_and_bcast(util.svd, Hankel_mat_true)
            direct_modes_array_true = self.direct_vecs.dot(
                R_sing_vecs_true).dot(np.diag(sing_vals_true**-0.5))
            adjoint_modes_array_true = adjoint_vecs.dot(
                L_sing_vecs_true).dot(np.diag(sing_vals_true**-0.5))
            direct_modes_array, adjoint_modes_array, sing_vals, \
                L_sing_vecs, R_sing_vecs, Hankel_mat = \
                compute_BPOD_matrices(self.direct_vecs, 
                adjoint_vecs, self.mode_indices, self.mode_indices,
                inner_product_weights=weights, return_all=True)

            np.testing.assert_allclose(Hankel_mat, Hankel_mat_true)
            np.testing.assert_allclose(sing_vals, sing_vals_true)
            np.testing.assert_allclose(L_sing_vecs, L_sing_vecs_true)
            np.testing.assert_allclose(R_sing_vecs, R_sing_vecs_true)
            np.testing.assert_allclose(
                direct_modes_array, 
                direct_modes_array_true[:,self.mode_indices], 
                rtol=tol, atol=tol)
            np.testing.assert_allclose(
                adjoint_modes_array, 
                adjoint_modes_array_true[:,self.mode_indices], 
                rtol=tol, atol=tol)


class TestBPODHandles(unittest.TestCase):
    """Test the BPOD class methods """
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
    
        self.test_dir = 'DELETE_ME_test_files_bpod'
        if not os.path.isdir(self.test_dir):
            _parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        
        self.mode_nums = [2, 3, 0]
        self.num_direct_vecs = 10
        self.num_adjoint_vecs = 12
        self.num_inputs = 1
        self.num_outputs = 1
        self.num_states = 20
        
        #A = np.mat(np.random.random((self.num_states, self.num_states)))
        A = np.mat(_parallel.call_and_bcast(util.drss, self.num_states,1,1)[0])
        B = np.mat(_parallel.call_and_bcast(np.random.random, 
            (self.num_states, self.num_inputs)))
        C = np.mat(_parallel.call_and_bcast(np.random.random, 
            (self.num_outputs, self.num_states)))
        self.direct_vecs = [B]
        A_powers = np.identity(A.shape[0])
        for t in range(self.num_direct_vecs-1):
            A_powers = A_powers.dot(A)
            self.direct_vecs.append(A_powers.dot(B))
        self.direct_vec_array = np.array(self.direct_vecs).squeeze().T

        A_adjoint = A.H
        C_adjoint = C.H
        A_adjoint_powers = np.identity(A_adjoint.shape[0])
        self.adjoint_vecs = [C_adjoint]
        for t in range(self.num_adjoint_vecs-1):
            A_adjoint_powers = A_adjoint_powers.dot(A_adjoint)
            self.adjoint_vecs.append(A_adjoint_powers.dot(C_adjoint))
        self.adjoint_vec_array = np.array(self.adjoint_vecs).squeeze().T

        self.direct_vec_path = join(self.test_dir, 'direct_vec_%03d.txt')
        self.adjoint_vec_path = join(self.test_dir, 'adjoint_vec_%03d.txt')

        self.direct_vec_handles = [V.VecHandleArrayText(self.direct_vec_path%i) 
            for i in range(self.num_direct_vecs)]
        self.adjoint_vec_handles = [
            V.VecHandleArrayText(self.adjoint_vec_path%i) 
            for i in range(self.num_adjoint_vecs)]
        
        if _parallel.is_rank_zero():
            for i, handle in enumerate(self.direct_vec_handles):
                handle.put(self.direct_vecs[i])
            for i, handle in enumerate(self.adjoint_vec_handles):
                handle.put(self.adjoint_vecs[i])
               
        self.Hankel_mat_true = np.dot(self.adjoint_vec_array.T, 
            self.direct_vec_array)
    
        self.L_sing_vecs_true, self.sing_vals_true, self.R_sing_vecs_true = \
            _parallel.call_and_bcast(util.svd, self.Hankel_mat_true, atol=1e-10)
        
        self.direct_mode_array = self.direct_vec_array * \
            np.mat(self.R_sing_vecs_true) * \
            np.mat(np.diag(self.sing_vals_true ** -0.5))
        self.adjoint_mode_array = self.adjoint_vec_array * \
            np.mat(self.L_sing_vecs_true) *\
            np.mat(np.diag(self.sing_vals_true ** -0.5))

        self.my_BPOD = BPODHandles(np.vdot, verbosity=0)
        _parallel.barrier()

    def tearDown(self):
        _parallel.barrier()
        _parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        _parallel.barrier()
        
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        
        def my_load(fname): pass
        def my_save(data, fname): pass 
        def my_IP(vec1, vec2): pass
        
        data_members_default = {'put_mat': util.save_array_text, 'get_mat':
             util.load_array_text,
            'verbosity': 0, 'L_sing_vecs': None, 'R_sing_vecs': None,
            'sing_vals': None, 'direct_vec_handles': None,
            'adjoint_vec_handles': None, 
            'direct_vec_handles': None, 'adjoint_vec_handles': None,
            'Hankel_mat': None,
            'vec_space': VectorSpaceHandles(inner_product=my_IP, verbosity=0)}
        
        # Get default data member values
        #self.maxDiff = None
        for k,v in util.get_data_members(
            BPODHandles(my_IP, verbosity=0)).items():
            self.assertEqual(v, data_members_default[k])
        
        my_BPOD = BPODHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])
       
        my_BPOD = BPODHandles(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])
 
        my_BPOD = BPODHandles(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])
        
        max_vecs_per_node = 500
        my_BPOD = BPODHandles(
            my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * _parallel.get_num_nodes() / _parallel.\
            get_num_procs()
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])
       
    def test_puts_gets(self):
        """Test that put/get work in base class."""
        test_dir = 'DELETE_ME_test_files_bpod'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(test_dir) and _parallel.is_rank_zero():        
            os.mkdir(test_dir)
        num_vecs = 10
        num_states = 30
        Hankel_mat_true = _parallel.call_and_bcast(
            np.random.random, ((num_vecs, num_vecs)))
        L_sing_vecs_true, sing_vals_true, R_sing_vecs_true = \
            _parallel.call_and_bcast(util.svd, Hankel_mat_true)

        my_BPOD = BPODHandles(None, verbosity=0)
        my_BPOD.Hankel_mat = Hankel_mat_true
        my_BPOD.sing_vals = sing_vals_true
        my_BPOD.L_sing_vecs = L_sing_vecs_true
        my_BPOD.R_sing_vecs = R_sing_vecs_true

        L_sing_vecs_path = join(test_dir, 'L_sing_vecs.txt')
        R_sing_vecs_path = join(test_dir, 'R_sing_vecs.txt')
        sing_vals_path = join(test_dir, 'sing_vals.txt')
        Hankel_mat_path = join(test_dir, 'Hankel_mat.txt')
        my_BPOD.put_decomp(sing_vals_path, L_sing_vecs_path, R_sing_vecs_path)
        my_BPOD.put_Hankel_mat(Hankel_mat_path)
        _parallel.barrier()

        BPOD_load = BPODHandles(None, verbosity=0)
        
        BPOD_load.get_decomp(
            sing_vals_path, L_sing_vecs_path, R_sing_vecs_path)
        Hankel_mat_loaded = _parallel.call_and_bcast(
            util.load_array_text, Hankel_mat_path)

        np.testing.assert_allclose(Hankel_mat_loaded, Hankel_mat_true)
        np.testing.assert_allclose(BPOD_load.L_sing_vecs, L_sing_vecs_true)
        np.testing.assert_allclose(BPOD_load.R_sing_vecs, R_sing_vecs_true)
        np.testing.assert_allclose(BPOD_load.sing_vals, sing_vals_true)

    #@unittest.skip('testing others')
    def test_compute_decomp(self):
        """Test that can take vecs, compute the Hankel and SVD matrices. """
        tol = 1e-6
        
        sing_vals_return, L_sing_vecs_return, R_sing_vecs_return = \
            self.my_BPOD.compute_decomp(self.direct_vec_handles, 
                self.adjoint_vec_handles)
        
        np.testing.assert_allclose(self.my_BPOD.Hankel_mat,
            self.Hankel_mat_true, rtol=tol)

        num_sing_vals = len(self.sing_vals_true)
        np.testing.assert_allclose(self.my_BPOD.L_sing_vecs[:,:num_sing_vals],
            self.L_sing_vecs_true, rtol=tol, atol=tol)
        
        np.testing.assert_allclose(self.my_BPOD.R_sing_vecs[:,:num_sing_vals],
            self.R_sing_vecs_true, rtol=tol, atol=tol)
        np.testing.assert_allclose(self.my_BPOD.sing_vals[:num_sing_vals],
            self.sing_vals_true, rtol=tol, atol=tol)
        
        np.testing.assert_allclose(L_sing_vecs_return[:,:num_sing_vals],
            self.L_sing_vecs_true, rtol=tol, atol=tol)
        np.testing.assert_allclose(R_sing_vecs_return[:,:num_sing_vals],
            self.R_sing_vecs_true, rtol=tol, atol=tol)
        np.testing.assert_allclose(sing_vals_return[:num_sing_vals],
            self.sing_vals_true, rtol=tol, atol=tol)

    #@unittest.skip('testing others')
    def test_compute_modes(self):
        """Test computing modes in serial and parallel."""
        atol = 1e-6
        direct_mode_path = join(self.test_dir, 'direct_mode_%03d.txt')
        adjoint_mode_path = join(self.test_dir, 'adjoint_mode_%03d.txt')
        
        # starts with the correct decomposition.
        self.my_BPOD.R_sing_vecs = self.R_sing_vecs_true
        self.my_BPOD.L_sing_vecs = self.L_sing_vecs_true
        self.my_BPOD.sing_vals = self.sing_vals_true
        
        direct_mode_handles = [V.VecHandleArrayText(direct_mode_path%i) 
            for i in self.mode_nums]
        adjoint_mode_handles = [V.VecHandleArrayText(adjoint_mode_path%i)
            for i in self.mode_nums]

        self.my_BPOD.compute_direct_modes(self.mode_nums, direct_mode_handles,
            direct_vec_handles=self.direct_vec_handles)
        self.my_BPOD.compute_adjoint_modes(self.mode_nums, 
            adjoint_mode_handles,
            adjoint_vec_handles=self.adjoint_vec_handles)

        for mode_index, mode_handle in enumerate(direct_mode_handles):
            mode = mode_handle.get()
            np.testing.assert_allclose(mode, 
                self.direct_mode_array[:,self.mode_nums[mode_index]], atol=atol)
            
        for mode_index, mode_handle in enumerate(adjoint_mode_handles):
            mode = mode_handle.get()
            np.testing.assert_allclose(
                mode, 
                self.adjoint_mode_array[:,self.mode_nums[mode_index]], 
                atol=atol)
            
        for direct_mode_index, direct_handle in \
            enumerate(direct_mode_handles):
            direct_mode = direct_handle.get()
            for adjoint_mode_index, adjoint_handle in \
                enumerate(adjoint_mode_handles):
                adjoint_mode = adjoint_handle.get()
                IP = self.my_BPOD.vec_space.inner_product(direct_mode, 
                    adjoint_mode)
                if self.mode_nums[direct_mode_index] != \
                    self.mode_nums[adjoint_mode_index]:
                    self.assertAlmostEqual(IP, 0., places=6)
                else:
                    self.assertAlmostEqual(IP, 1., places=6)
 
    def test_compute_proj_coeffs(self):
        # Tests fail if tolerance is too tight, likely due to random nature of
        # data.  Maximum error (elementwise) seems to come out ~1e-11.
        rtol = 1e-8
        atol = 1e-10  

        # Compute true projection coefficients by simply projecting directly
        # onto the modes.
        proj_coeffs_true = (
            self.adjoint_mode_array.H * self.direct_vec_array)

        # Initialize the POD object with the known correct decomposition
        # matrices, to avoid errors in computing those matrices.
        self.my_BPOD.R_sing_vecs = self.R_sing_vecs_true
        self.my_BPOD.L_sing_vecs = self.L_sing_vecs_true
        self.my_BPOD.sing_vals = self.sing_vals_true

        # Compute projection coefficients
        proj_coeffs = self.my_BPOD.compute_proj_coeffs()

        # Test values
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)

    def test_compute_adj_proj_coeffs(self):
        # Tests fail if tolerance is too tight, likely due to random nature of
        # data.  Maximum error (elementwise) seems to come out ~1e-11.
        rtol = 1e-8
        atol = 1e-10  

        # Compute true projection coefficients by simply projecting directly
        # onto the modes.
        adj_proj_coeffs_true = (
            self.direct_mode_array.H * self.adjoint_vec_array)

        # Initialize the POD object with the known correct decomposition
        # matrices, to avoid errors in computing those matrices.
        self.my_BPOD.R_sing_vecs = self.R_sing_vecs_true
        self.my_BPOD.L_sing_vecs = self.L_sing_vecs_true
        self.my_BPOD.sing_vals = self.sing_vals_true

        # Compute projection coefficients
        adj_proj_coeffs = self.my_BPOD.compute_adjoint_proj_coeffs()

        # Test values
        np.testing.assert_allclose(
            adj_proj_coeffs, adj_proj_coeffs_true, rtol=rtol, atol=atol)
   

if __name__ == '__main__':
    unittest.main()

