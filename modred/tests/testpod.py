#!/usr/bin/env python
"""Test POD module"""

import unittest
import os
import numpy as N
from os.path import join
from shutil import rmtree
import copy

import helper
helper.add_to_path(join(join(os.path.dirname(os.path.abspath(__file__)), 
    '..', 'src')))
import parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance

from pod import *
from vectorspace import *
import vectors as V
import util



@unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
class TestPODArraysFunctions(unittest.TestCase):
    def setUp(self):
        self.mode_indices = [2, 4, 6, 3]
        self.num_vecs = 10
        self.num_states = 30                     

    def test_compute_modes(self):
        ws = N.identity(self.num_states)
        tol = 1e-6
        weights_full = N.mat(N.random.random((self.num_states, self.num_states)))
        weights_full = N.triu(weights_full) + N.triu(weights_full, 1).H
        weights_full = weights_full*weights_full
        weights_diag = N.random.random(self.num_states)
        weights_list = [None, weights_diag, weights_full]
        vec_array = N.random.random((self.num_states, self.num_vecs))
        for weights in weights_list:
            IP = VectorSpaceMatrices(weights=weights).compute_inner_product_mat
            correlation_mat_true = IP(vec_array, vec_array)
            eigen_vals_true, eigen_vecs_true = util.eigh(correlation_mat_true)
            build_coeff_mat_true = eigen_vecs_true * N.mat(N.diag(
                eigen_vals_true**-0.5))
            modes_true = vec_array.dot(build_coeff_mat_true)
            
            modes, eigen_vals, eigen_vecs, correlation_mat = \
                compute_POD_matrices_snaps_method(vec_array, self.mode_indices, 
                inner_product_weights=weights, return_all=True)
            
            N.testing.assert_allclose(eigen_vals, eigen_vals_true, rtol=tol)
            N.testing.assert_allclose(eigen_vecs, eigen_vecs_true)
            N.testing.assert_allclose(correlation_mat, correlation_mat_true)
            N.testing.assert_allclose(modes, modes_true[:,self.mode_indices])
                        
            modes, eigen_vals, eigen_vecs = \
                compute_POD_matrices_direct_method(vec_array, self.mode_indices, 
                inner_product_weights=weights, return_all=True)
            
            N.testing.assert_allclose(eigen_vals, eigen_vals_true)
            N.testing.assert_allclose(N.abs(eigen_vecs), N.abs(eigen_vecs_true))
            N.testing.assert_allclose(N.abs(modes), N.abs(modes_true[:,self.mode_indices]))
            
            


class TestPODHandles(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'DELETE_ME_test_files_pod'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(self.test_dir) and _parallel.is_rank_zero():        
            os.mkdir(self.test_dir)
        self.mode_indices = [2, 4, 3, 6]
        self.num_vecs = 10
        self.num_states = 30
        self.vec_array = _parallel.call_and_bcast(N.random.random, 
            (self.num_states, self.num_vecs))
        self.correlation_mat_true = self.vec_array.conj().transpose().dot(
            self.vec_array)
    
        self.eigen_vals_true, self.eigen_vecs_true = \
            _parallel.call_and_bcast(util.eigh, self.correlation_mat_true)

        self.mode_array = N.dot(self.vec_array, N.dot(self.eigen_vecs_true,
            N.diag(self.eigen_vals_true**-0.5)))
        self.vec_path = join(self.test_dir, 'vec_%03d.txt')
        self.vec_handles = [V.VecHandleArrayText(self.vec_path%i)
            for i in range(self.num_vecs)]
        for vec_index, handle in enumerate(self.vec_handles):
            handle.put(self.vec_array[:, vec_index])
        
        self.my_POD = PODHandles(N.vdot, verbosity=0)
        _parallel.barrier()


    def tearDown(self):
        _parallel.barrier()
        _parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        _parallel.barrier()
                          
    
    def test_put_gets(self):
        test_dir = 'DELETE_ME_test_files_pod'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(test_dir):        
            _parallel.call_from_rank_zero(os.mkdir, test_dir)
        num_vecs = 10
        num_states = 30
        correlation_mat_true = _parallel.call_and_bcast(
            N.random.random, ((num_vecs, num_vecs)))
        eigen_vals_true = _parallel.call_and_bcast(
            N.random.random, num_vecs)
        eigen_vecs_true = _parallel.call_and_bcast(
            N.random.random, ((num_states, num_vecs)))
        my_POD = PODHandles(None, verbosity=0)
        my_POD.correlation_mat = correlation_mat_true
        my_POD.eigen_vals = eigen_vals_true
        my_POD.eigen_vecs = eigen_vecs_true
        _parallel.barrier()
        
        eigen_vecs_path = join(test_dir, 'eigen_vecs.txt')
        eigen_vals_path = join(test_dir, 'eigen_vals.txt')
        correlation_mat_path = join(test_dir, 'correlation.txt')
        my_POD.put_decomp(eigen_vecs_path, eigen_vals_path)
        my_POD.put_correlation_mat(correlation_mat_path)
        
        POD_load = PODHandles(None, verbosity=0)
        POD_load.get_decomp(eigen_vecs_path, eigen_vals_path)
        correlation_mat_loaded = util.load_array_text(correlation_mat_path)

        N.testing.assert_allclose(correlation_mat_loaded, 
            correlation_mat_true)
        N.testing.assert_allclose(POD_load.eigen_vecs, eigen_vecs_true)
        N.testing.assert_allclose(POD_load.eigen_vals, eigen_vals_true)

     
     
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        def my_load(): pass
        def my_save(): pass
        def my_IP(): pass
        
        data_members_default = {'put_mat': util.save_array_text, 'get_mat': 
            util.load_array_text,
            'verbosity': 0, 'eigen_vecs': None, 'eigen_vals': None,
            'correlation_mat': None, 'vec_handles': None, 'vecs': None,
            'vec_space': VectorSpaceHandles(inner_product=my_IP, verbosity=0)}
        for k,v in util.get_data_members(PODHandles(my_IP, verbosity=0)).iteritems():
            self.assertEqual(v, data_members_default[k])
        
        my_POD = PODHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_POD).iteritems():
            self.assertEqual(v, data_members_modified[k])
        
        my_POD = PODHandles(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_POD).iteritems():
            self.assertEqual(v, data_members_modified[k])
            
        my_POD = PODHandles(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_POD).iteritems():
            self.assertEqual(v, data_members_modified[k])
            
        max_vecs_per_node = 500
        my_POD = PODHandles(my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * _parallel.get_num_nodes() / \
            _parallel.get_num_procs()
        for k,v in util.get_data_members(my_POD).iteritems():
            self.assertEqual(v, data_members_modified[k])
            
        
    def test_compute_decomp(self):
        """Test computation of the correlation mat and SVD matrices."""
        tol = 1e-6
        eigen_vecs_returned, eigen_vals_returned = \
            self.my_POD.compute_decomp(self.vec_handles)
                        
        N.testing.assert_allclose(self.my_POD.correlation_mat, 
            self.correlation_mat_true, rtol=tol)
        N.testing.assert_allclose(self.my_POD.eigen_vecs, 
            self.eigen_vecs_true, rtol=tol)
        N.testing.assert_allclose(self.my_POD.eigen_vals, 
            self.eigen_vals_true, rtol=tol)
          
        N.testing.assert_allclose(eigen_vecs_returned, 
            self.eigen_vecs_true, rtol=tol)
        N.testing.assert_allclose(eigen_vals_returned, 
            self.eigen_vals_true, rtol=tol)

        

    def test_compute_modes(self):
        mode_path = join(self.test_dir, 'mode_%03d.txt')
        mode_handles = [V.VecHandleArrayText(mode_path%i) 
            for i in self.mode_indices]
        # starts with the CORRECT decomposition.
        self.my_POD.eigen_vecs = self.eigen_vecs_true
        self.my_POD.eigen_vals = self.eigen_vals_true
        
        self.my_POD.compute_modes(self.mode_indices, mode_handles, 
            vec_handles=self.vec_handles)
                
        for mode_index, mode_handle in enumerate(mode_handles):
            mode = mode_handle.get()
            N.testing.assert_allclose(mode.squeeze(), 
                self.mode_array[:,self.mode_indices[mode_index]])
        
        for mode_index1, handle1 in enumerate(mode_handles):
            mode1 = handle1.get()
            for mode_index2, handle2 in enumerate(mode_handles):
                mode2 = handle2.get()
                IP = self.my_POD.vec_space.inner_product(mode1, mode2)
                if self.mode_indices[mode_index1] != \
                    self.mode_indices[mode_index2]:
                    self.assertAlmostEqual(IP, 0.)
                else:
                    self.assertAlmostEqual(IP, 1.)


if __name__ == '__main__':
    unittest.main()
