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



@unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
class TestDMDArraysFunctions(unittest.TestCase):

    def setUp(self):
        # Generate vecs if we are on the first processor
        # A random matrix of data (#cols = #vecs)
        self.num_vecs = 6
        self.num_states = 12

    def _helper_compute_DMD_from_data(self, vecs, adv_vecs,
        inner_product):
        correlation_mat = inner_product(vecs, vecs)
        W, Sigma, dummy = util.svd(correlation_mat) # dummy = W.
        U = vecs.dot(W).dot(N.diag(Sigma**-0.5))
        ritz_vals, eig_vecs = N.linalg.eig(inner_product(
            U, adv_vecs).dot(W).dot(N.diag(Sigma**-0.5)))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U.dot(eig_vecs)
        scaling = N.linalg.lstsq(ritz_vecs, vecs[:, 0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        ritz_vecs = ritz_vecs.dot(scaling)
        build_coeffs = W.dot(N.diag(Sigma**-0.5)).dot(eig_vecs).dot(scaling)
        mode_norms = N.diag(inner_product(ritz_vecs, ritz_vecs)).real
        return ritz_vals, ritz_vecs, build_coeffs, mode_norms


    def test_all(self):
        tol = 1e-8
        mode_indices = [2, 0, 3]
        weights_full = N.mat(N.random.random((self.num_states, self.num_states)))
        weights_full = N.triu(weights_full) + N.triu(weights_full, 1).H
        weights_full = weights_full*weights_full
        weights_diag = N.random.random(self.num_states)
        weights_list = [None, weights_diag, weights_full]
        for weights in weights_list:
            IP = VectorSpaceMatrices(weights=weights).compute_inner_product_mat
            vecs = N.random.random((self.num_states, self.num_vecs))

            # Compute true DMD ritz vecs from data for a sequential dataset
            ritz_vals_true, ritz_vecs_true, build_coeffs_true, mode_norms_true=\
                self._helper_compute_DMD_from_data(vecs[:, :-1], 
                    vecs[:, 1:], IP)

            # Compute modes and compare to true ritz vecs           
            modes, ritz_vals, mode_norms, build_coeffs = \
                compute_DMD_matrices_snaps_method(vecs, mode_indices, 
                inner_product_weights=weights, return_all=True)
            N.testing.assert_allclose(mode_norms, mode_norms_true, rtol=tol)
            N.testing.assert_allclose(ritz_vals, ritz_vals_true, rtol=tol)
            N.testing.assert_allclose(build_coeffs, build_coeffs_true, rtol=tol)
            N.testing.assert_allclose(modes, ritz_vecs_true[:, mode_indices], 
                rtol=tol)
            
            modes, ritz_vals, mode_norms, build_coeffs = \
                compute_DMD_matrices_direct_method(vecs, mode_indices, 
                inner_product_weights=weights, return_all=True)
            N.testing.assert_allclose(mode_norms, mode_norms_true, rtol=tol)
            N.testing.assert_allclose(ritz_vals, ritz_vals_true, rtol=tol)
            N.testing.assert_allclose(build_coeffs, build_coeffs_true, rtol=tol)
            N.testing.assert_allclose(modes, ritz_vecs_true[:, mode_indices], 
                rtol=tol)
            
           
            # Generate data for a non-sequential dataset
            adv_vecs = N.random.random((self.num_states, self.num_vecs))

            # Compute true DMD ritz vecs from data for a non-sequential dataset
            ritz_vals_true, ritz_vecs_true, build_coeffs_true, mode_norms_true =\
                self._helper_compute_DMD_from_data(vecs, adv_vecs, IP)
            # Compare computed modes to truth
            modes, ritz_vals, mode_norms, build_coeffs = \
                compute_DMD_matrices_snaps_method(vecs, mode_indices, 
                adv_vecs=adv_vecs, inner_product_weights=weights, 
                return_all=True)
            N.testing.assert_allclose(modes, ritz_vecs_true[:, mode_indices], 
                rtol=tol)
            N.testing.assert_allclose(ritz_vals, ritz_vals_true, rtol=tol)
            N.testing.assert_allclose(build_coeffs, build_coeffs_true, rtol=tol)
            N.testing.assert_allclose(mode_norms, mode_norms_true, rtol=tol)
            
            modes, ritz_vals, mode_norms, build_coeffs = \
                compute_DMD_matrices_direct_method(
                vecs, mode_indices, adv_vecs=adv_vecs, 
                inner_product_weights=weights, return_all=True)
            N.testing.assert_allclose(ritz_vals, ritz_vals_true, rtol=tol)
            N.testing.assert_allclose(modes, ritz_vecs_true[:, mode_indices], 
                rtol=tol)
            N.testing.assert_allclose(build_coeffs, build_coeffs_true, rtol=tol)
            N.testing.assert_allclose(mode_norms, mode_norms_true, rtol=tol)


    

           
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
        self.adv_vec_path = join(self.test_dir, 'dmd_adv_vec_%03d.pkl')
        self.mode_path = join(self.test_dir, 'dmd_truemode_%03d.pkl')
        self.vec_handles = [V.VecHandlePickle(self.vec_path%i) 
            for i in range(self.num_vecs)]
        self.adv_vec_handles = [V.VecHandlePickle(self.adv_vec_path%i) for i in 
            range(self.num_vecs)]
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
            'adv_vec_handles': None, 'expanded_correlation_mat': None,
            'correlation_mat_evals': None, 'low_order_linear_map': None,
            'correlation_mat_evecs': None,
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


    def test_gets_puts(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(test_dir) and _parallel.is_rank_zero():
            os.mkdir(test_dir)
        ritz_vals = _parallel.call_and_bcast(N.random.random, 5)
        mode_norms = _parallel.call_and_bcast(N.random.random, 5)
        build_coeffs = _parallel.call_and_bcast(N.random.random, (10,10))
        
        my_DMD = DMDHandles(None, verbosity=0)
        my_DMD.ritz_vals = ritz_vals
        my_DMD.mode_norms = mode_norms
        my_DMD.build_coeffs = build_coeffs
        
        ritz_vals_path = join(test_dir, 'dmd_ritz_vals.txt')
        mode_norms_path = join(test_dir, 'dmd_mode_energies.txt')
        build_coeffs_path = join(test_dir, 'dmd_build_coeffs.txt')
        
        my_DMD.put_decomp(ritz_vals_path, mode_norms_path, 
            build_coeffs_path)
        _parallel.barrier()

        DMD_load = DMDHandles(None, verbosity=0)
        DMD_load.get_decomp(
            ritz_vals_path, mode_norms_path, build_coeffs_path)

        N.testing.assert_allclose(DMD_load.ritz_vals, ritz_vals)
        N.testing.assert_allclose(DMD_load.build_coeffs, build_coeffs)
        N.testing.assert_allclose(DMD_load.mode_norms, mode_norms)


    def _helper_compute_DMD_from_data(self, vec_array, adv_vec_array,
        inner_product):
        # Create lists of vecs, advanced vecs for inner product function
        vecs = [vec_array[:, i] for i in range(vec_array.shape[1])]
        adv_vecs = [adv_vec_array[:, i] for i in range(adv_vec_array.shape[1])]

        # Compute DMD
        correlation_mat = inner_product(vecs, vecs)
        W, Sigma, dummy = util.svd(correlation_mat) # dummy = W.
        U = vec_array.dot(W).dot(N.diag(Sigma**-0.5))
        U_list = [U[:,i] for i in range(U.shape[1])]
        ritz_vals, eig_vecs = N.linalg.eig(inner_product(
            U_list, adv_vecs).dot(W).dot(N.diag(Sigma**-0.5)))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U.dot(eig_vecs)
        scaling = N.linalg.lstsq(ritz_vecs, vec_array[:, 0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        ritz_vecs = ritz_vecs.dot(scaling)
        build_coeffs = W.dot(N.diag(Sigma**-0.5)).dot(eig_vecs).dot(scaling)
        ritz_vecs_list = [N.array(ritz_vecs[:,i]).squeeze() 
            for i in range(ritz_vecs.shape[1])]
        mode_norms = N.diag(inner_product(ritz_vecs_list, ritz_vecs_list)).real

        return ritz_vals, ritz_vecs, build_coeffs, mode_norms

    
    def _helper_check_decomp(self, ritz_vals, build_coeffs, mode_norms, 
        vec_handles, adv_vec_handles=None):
        # Set tolerance.  
        tol = 1e-10

        # Compute DMD using modred
        ritz_vals_returned, mode_norms_returned, build_coeffs_returned = \
            self.my_DMD.compute_decomp(vec_handles, 
            adv_vec_handles=adv_vec_handles)

        # Test that matrices were correctly computed
        N.testing.assert_allclose(self.my_DMD.ritz_vals, 
            ritz_vals, rtol=tol)
        N.testing.assert_allclose(self.my_DMD.build_coeffs, 
            build_coeffs, rtol=tol)
        N.testing.assert_allclose(self.my_DMD.mode_norms, 
            mode_norms, rtol=tol)

        # Test that matrices were correctly returned
        N.testing.assert_allclose(ritz_vals_returned, 
            ritz_vals, rtol=tol)
        N.testing.assert_allclose(build_coeffs_returned, 
            build_coeffs, rtol=tol)
        N.testing.assert_allclose(mode_norms_returned, 
            mode_norms, rtol=tol)


    def _helper_check_modes(self, ritz_vecs, build_coeffs, vec_handles):
        # Set tolerance. 
        tol = 1e-10

        # Load build coefficients into empty DMD object.  (This way the mode
        # building test is guaranteed to use the same coefficients.) 
        self.my_DMD.build_coeffs = build_coeffs
        
        # Compute modes and save to file
        mode_path = join(self.test_dir, 'dmd_mode_%03d.pkl')
        mode_indices = range(build_coeffs.shape[1])
        self.my_DMD.compute_modes(mode_indices, 
            [V.VecHandlePickle(mode_path%i) for i in mode_indices],
            vec_handles=vec_handles)
        
        # Load all modes into matrix, compare to ritz vecs from direct
        # computation
        mode_array = N.array(N.zeros((self.num_states, build_coeffs.shape[1])), 
            dtype=complex)
        for i in range(build_coeffs.shape[1]):
            mode_array[:, i] = V.VecHandlePickle(mode_path % i).get()
        N.testing.assert_allclose(mode_array, ritz_vecs, rtol=tol)


    def test_compute_decomp(self):
        """Test DMD decomposition (ritz values, build coefficients, mode
        norms)"""    
        # Define an array of vectors, with corresponding handles
        vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(N.array(vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data, for a sequential dataset
        ritz_vals, ritz_vecs, build_coeffs, mode_norms =\
            self._helper_compute_DMD_from_data(vec_array[:, :-1],
            vec_array[:, 1:], util.InnerProductBlock(N.vdot))

        # Check modred against direct computation, for a sequential dataset
        _parallel.barrier()
        self._helper_check_decomp(ritz_vals, build_coeffs, mode_norms,
            self.vec_handles)

        # Now create more data, to check a non-sequential dataset
        adv_vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(N.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data, for a non-sequential dataset
        ritz_vals, ritz_vecs, build_coeffs, mode_norms =\
            self._helper_compute_DMD_from_data(vec_array, 
                adv_vec_array, util.InnerProductBlock(N.vdot))

        # Check modred against direct computation, for a non-sequential dataset
        _parallel.barrier()
        self._helper_check_decomp(ritz_vals, build_coeffs, mode_norms,
            self.vec_handles, adv_vec_handles=self.adv_vec_handles)

        # Check that if mismatched sets of handles are passed in, an error is
        # raised.
        self.assertRaises(ValueError, self.my_DMD.compute_decomp,
            self.vec_handles, self.adv_vec_handles[:-1])


    def test_compute_modes(self):
        """Test building of modes."""
        vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(N.array(vec_array[:, vec_index]).squeeze())
        
        # Compute DMD directly from data, for a sequential dataset
        ritz_vals, ritz_vecs, build_coeffs, mode_norms =\
            self._helper_compute_DMD_from_data(vec_array[:, :-1],
            vec_array[:, 1:], util.InnerProductBlock(N.vdot))
 
        # Check direct computation against modred   
        _parallel.barrier()
        self._helper_check_modes(ritz_vecs, build_coeffs, self.vec_handles)

        # Generate data for a non-sequential dataset
        adv_vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(N.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data, for a non-sequential dataset
        ritz_vals, ritz_vecs, build_coeffs, mode_norms =\
            self._helper_compute_DMD_from_data(vec_array, 
                adv_vec_array, util.InnerProductBlock(N.vdot))
 
        # Check direct computation against modred   
        _parallel.barrier()
        self._helper_check_modes(ritz_vecs, build_coeffs, self.vec_handles)



if __name__ == '__main__':
    unittest.main()

