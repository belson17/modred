#!/usr/bin/env python
"""Test dmd module"""

import copy
import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path(join(join(os.path.dirname(os.path.abspath(__file__)), 
    '..', 'src')))
import parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance

from dmd import *
from vectorspace import *
import vectors as V
import util


class TestDMDBase(unittest.TestCase):
    def test_gets_puts(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(test_dir) and _parallel.is_rank_zero():
            os.mkdir(test_dir)
        ritz_vals = _parallel.call_and_bcast(N.random.random, 5)
        mode_norms = _parallel.call_and_bcast(N.random.random, 5)
        build_coeffs = _parallel.call_and_bcast(N.random.random, (10,10))
        
        my_DMD = DMDBase()
        my_DMD.ritz_vals = ritz_vals
        my_DMD.mode_norms = mode_norms
        my_DMD.build_coeffs = build_coeffs
        
        ritz_vals_path = join(test_dir, 'dmd_ritz_vals.txt')
        mode_norms_path = join(test_dir, 'dmd_mode_energies.txt')
        build_coeffs_path = join(test_dir, 'dmd_build_coeffs.txt')
        
        my_DMD.put_decomp(ritz_vals_path, mode_norms_path, 
            build_coeffs_path)
        _parallel.barrier()

        DMD_load = DMDBase()
        DMD_load.get_decomp(
            ritz_vals_path, mode_norms_path, build_coeffs_path)

        N.testing.assert_allclose(DMD_load.ritz_vals, ritz_vals)
        N.testing.assert_allclose(DMD_load.build_coeffs, build_coeffs)
        N.testing.assert_allclose(DMD_load.mode_norms, mode_norms)



@unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
class TestDMDArrays(unittest.TestCase):

    def setUp(self):
        # Generate vecs if we are on the first processor
        # A random matrix of data (#cols = #vecs)
        self.num_vecs = 6
        self.num_states = 12
        self.my_DMD = DMDArrays()


    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        def my_load(): pass
        def my_save(): pass
        if _parallel.is_distributed():
            self.assertRaises(RuntimeError, DMDArrays)

        def my_load(fname): pass
        def my_save(data, fname): pass
        weights = _parallel.call_and_bcast(N.random.random ,5)
        
        data_members_default = {'put_mat': util.save_array_text, 'get_mat':
             util.load_array_text,
            'verbosity': 0, 'ritz_vals': None, 
            'correlation_mat': None, 'build_coeffs': None,
            'mode_norms': None, 'vec_array': None,
            'vec_space': VectorSpaceArrays()}
        
        # Get default data member values
        for k,v in util.get_data_members(DMDArrays(verbosity=0)).iteritems():
            self.assertEqual(v, data_members_default[k])
        
        my_DMD = DMDArrays(inner_product_weights=weights, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceArrays(weights)
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertTrue(util.smart_eq(v, data_members_modified[k]))
       
        my_DMD = DMDArrays(get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertEqual(v, data_members_modified[k])
 
        my_DMD = DMDArrays(put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertEqual(v, data_members_modified[k])
    
    
    def test_compute_decomp(self):
        tol = 1e-6
        ws = N.identity(self.num_states)
        ws[0,0] = 2
        ws[1,0] = 1.1
        ws[0,1] = 1.1
        weights_list = [None, N.arange(self.num_states), ws]
        for weights in weights_list:
            IP = VectorSpaceArrays(weights=weights).compute_inner_product_mat
            vec_array = _parallel.call_and_bcast(N.random.random, 
                ((self.num_states, self.num_vecs)))
            correlation_mat = IP(vec_array[:, :-1], vec_array[:, :-1])
            W, Sigma, dummy = util.svd(correlation_mat) # dummy = W.
            U = vec_array[:,:-1].dot(W).dot(N.diag(Sigma**-0.5))
            ritz_vals, eig_vecs = N.linalg.eig(IP(
                U, vec_array[:,1:]).dot(W).dot(N.diag(Sigma**-0.5)))
            eig_vecs = N.mat(eig_vecs)
            ritz_vecs = U.dot(eig_vecs)
            scaling = N.linalg.lstsq(ritz_vecs, vec_array[:, 0])[0]
            scaling = N.mat(N.diag(N.array(scaling).squeeze()))
            ritz_vecs = ritz_vecs.dot(scaling)
            build_coeffs = W.dot(N.diag(Sigma**-0.5)).dot(eig_vecs).dot(
                scaling)
            _parallel.barrier()
            mode_norms = N.diag(IP(ritz_vecs, ritz_vecs)).real

            my_DMD = DMDArrays(inner_product_weights=weights, verbosity=0)
            ritz_vals_returned, mode_norms_returned, build_coeffs_returned = \
                my_DMD.compute_decomp(vec_array)
            
            N.testing.assert_allclose(my_DMD.ritz_vals, ritz_vals, rtol=tol)
            N.testing.assert_allclose(my_DMD.build_coeffs,
                build_coeffs, rtol=tol)
            N.testing.assert_allclose(my_DMD.mode_norms, 
                mode_norms, rtol=tol)
    
            N.testing.assert_allclose(ritz_vals_returned, 
                ritz_vals, rtol=tol)
            N.testing.assert_allclose(build_coeffs_returned, 
                build_coeffs, rtol=tol)
            N.testing.assert_allclose(mode_norms_returned, 
                mode_norms, rtol=tol)
        
        
#@unittest.skip('others')
class TestDMDHandles(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(self.test_dir) and _parallel.is_rank_zero():
            os.mkdir(self.test_dir)
        
        self.num_vecs = 6
        self.num_states = 12
        self.my_DMD = DMDHandles(N.vdot, verbosity=0)

        self.vec_path = join(self.test_dir, 'dmd_vec_%03d.pkl')
        self.mode_path = join(self.test_dir, 'dmd_truemode_%03d.pkl')
        self.vec_handles = [V.PickleVecHandle(self.vec_path%i) 
            for i in range(self.num_vecs)]
        _parallel.barrier()
        
    
    def tearDown(self):
        _parallel.barrier()
        if _parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        _parallel.barrier()


    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        def my_load(fname): pass
        def my_save(data, fname): pass
        def my_IP(vec1, vec2): pass
        
        data_members_default = {'put_mat': util.save_array_text, 'get_mat':
             util.load_array_text,
            'verbosity': 0, 'ritz_vals': None, 'build_coeffs': None,
            'correlation_mat': None, 'mode_norms': None, 'vec_handles': None,
            'vec_space': VectorSpaceHandles(my_IP, verbosity=0)}
        
        # Get default data member values
        for k,v in util.get_data_members(
            DMDHandles(my_IP, verbosity=0)).iteritems():
            self.assertEqual(v, data_members_default[k])
        
        my_DMD = DMDHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertEqual(v, data_members_modified[k])
       
        my_DMD = DMDHandles(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertEqual(v, data_members_modified[k])
 
        my_DMD = DMDHandles(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertEqual(v, data_members_modified[k])
        
        max_vecs_per_node = 500
        my_DMD = DMDHandles(my_IP, max_vecs_per_node=max_vecs_per_node,
            verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * _parallel.get_num_nodes() / \
            _parallel.get_num_procs()
        for k,v in util.get_data_members(my_DMD).iteritems():
            self.assertEqual(v, data_members_modified[k])
       

    def test_compute_decomp(self):
        # Depending on vecs generated, test will fail if tol = 8, so use 7
        tol = 1e-7 
        IP = util.InnerProductBlock(N.vdot)

        vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        vecs = [vec_array[:, i] for i in range(self.num_vecs)]
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(N.array(vec_array[:, vec_index]).squeeze())
        correlation_mat = IP(vecs[:-1], vecs[:-1])
        
        W, Sigma, dummy = util.svd(correlation_mat) # dummy = W.
        U = vec_array[:, :-1].dot(W).dot(N.diag(Sigma**-0.5))
        U_list = [U[:,i] for i in range(U.shape[1])]
        ritz_vals, eig_vecs = N.linalg.eig(IP(
            U_list, vecs[1:]).dot(W).dot(N.diag(Sigma**-0.5)))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U.dot(eig_vecs)
        scaling = N.linalg.lstsq(ritz_vecs, vec_array[:, 0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        ritz_vecs = ritz_vecs.dot(scaling)
        ritz_vecs_list = [N.array(ritz_vecs[:,i]).squeeze() 
            for i in range(ritz_vecs.shape[1])]
        build_coeffs = W.dot(N.diag(Sigma**-0.5)).dot(eig_vecs).dot(scaling)
        _parallel.barrier()
        mode_norms = N.diag(IP(ritz_vecs_list, ritz_vecs_list)).real
        ritz_vals_returned, mode_norms_returned, build_coeffs_returned = \
            self.my_DMD.compute_decomp(self.vec_handles)
        
        # Test that matrices were correctly computed
        N.testing.assert_allclose(self.my_DMD.ritz_vals, 
            ritz_vals, rtol=tol)
        N.testing.assert_allclose(self.my_DMD.build_coeffs, 
            build_coeffs, rtol=tol)
        N.testing.assert_allclose(self.my_DMD.mode_norms, 
            mode_norms, rtol=tol)

        N.testing.assert_allclose(ritz_vals_returned, 
            ritz_vals, rtol=tol)
        N.testing.assert_allclose(build_coeffs_returned, 
            build_coeffs, rtol=tol)
        N.testing.assert_allclose(mode_norms_returned, 
            mode_norms, rtol=tol)


    def test_compute_modes(self):
        """Test building of modes, reconstruction formula."""
        tol = 1e-7
        IP = util.InnerProductBlock(N.vdot)

        vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        vecs = [vec_array[:, i] for i in range(self.num_vecs)]
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(N.array(vec_array[:, vec_index]).squeeze())
        correlation_mat = IP(vecs[:-1], vecs[:-1])
        
        W, Sigma, dummy = util.svd(correlation_mat) # dummy = W.
        U = vec_array[:, :-1].dot(W).dot(N.diag(Sigma**-0.5))
        U_list = [U[:,i] for i in range(U.shape[1])]
        ritz_vals, eig_vecs = N.linalg.eig(IP(
            U_list, vecs[1:]).dot(W).dot(N.diag(Sigma**-0.5)))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U.dot(eig_vecs)
        scaling = N.linalg.lstsq(ritz_vecs, vec_array[:, 0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        ritz_vecs = ritz_vecs.dot(scaling)
        ritz_vecs_list = [N.array(ritz_vecs[:,i]).squeeze() 
            for i in range(ritz_vecs.shape[1])]
        build_coeffs = W.dot(N.diag(Sigma**-0.5)).dot(eig_vecs).dot(scaling)
        _parallel.barrier()        
        mode_norms = N.diag(IP(ritz_vecs_list, ritz_vecs_list)).real
        ritz_vals_returned, mode_norms_returned, build_coeffs_returned = \
            self.my_DMD.compute_decomp(self.vec_handles)
            
        mode_path = join(self.test_dir, 'dmd_mode_%03d.pkl')
                
        self.my_DMD.build_coeffs = build_coeffs
        mode_indices = range(self.num_vecs-1)
        self.my_DMD.compute_modes(mode_indices, 
            [V.PickleVecHandle(mode_path%i) for i in mode_indices],
            vec_handles=self.vec_handles)
        
        # Load all modes into matrix
        mode_array = N.array(N.zeros((self.num_states, self.num_vecs-1)), 
            dtype=complex)
        for i in range(self.num_vecs-1):
            mode_array[:, i] = V.PickleVecHandle(
                mode_path % i).get()
        N.testing.assert_allclose(mode_array, ritz_vecs, rtol=tol)
        
        vandermonde_mat = N.fliplr(N.vander(ritz_vals, self.num_vecs-1))
        N.testing.assert_allclose(vec_array[:, :-1],
            ritz_vecs * vandermonde_mat, rtol=tol)


if __name__ == '__main__':
    unittest.main()

