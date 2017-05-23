#!/usr/bin/env python
"""Test dmd module"""
from __future__ import division
from future.builtins import range
import copy
import unittest
import os
from os.path import join
from shutil import rmtree

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
from modred.dmd import *
from modred.vectorspace import *
import modred.vectors as V
from modred import util


@unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
class TestDMDArraysFunctions(unittest.TestCase):
    def setUp(self):
        # Generate vecs if we are on the first processor
        # A random matrix of data (#cols = #vecs)
        self.num_vecs = 10
        self.num_states = 20

    def _helper_compute_DMD_from_data(
        self, vecs, inner_product, adv_vecs=None, max_num_eigvals=None):
        if adv_vecs is None:
            adv_vecs = vecs[:, 1:]
            vecs = vecs[:, :-1]
        correlation_mat = inner_product(vecs, vecs)
        cross_correlation_mat = inner_product(vecs, adv_vecs)
        V, Sigma, dummy = util.svd(correlation_mat)     # dummy = V.T 
        U = vecs.dot(V).dot(np.diag(Sigma ** -0.5))

        # Truncate if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < Sigma.size):
            V = V[:, :max_num_eigvals]
            Sigma = Sigma[:max_num_eigvals]
            U = U[:, :max_num_eigvals]

        A_tilde = inner_product(
            U, adv_vecs).dot(V).dot(np.diag(Sigma ** -0.5))
        eigvals, W, Z = util.eig_biorthog(
            A_tilde, scale_choice='left')
        build_coeffs_proj = V.dot(np.diag(Sigma ** -0.5)).dot(W)
        build_coeffs_exact = (
            V.dot(np.diag(Sigma ** -0.5)).dot(W).dot(np.diag(eigvals ** -1.)))
        modes_proj = vecs.dot(build_coeffs_proj)
        modes_exact = adv_vecs.dot(build_coeffs_exact)
        adj_modes = U.dot(Z)
        spectral_coeffs = np.abs(np.array(
            inner_product(adj_modes, np.mat(vecs[:, 0]).T)).squeeze())
        return (
            modes_exact, modes_proj, spectral_coeffs, eigvals,
            W, Z, Sigma, V, correlation_mat, cross_correlation_mat)
        
    def _helper_test_mat_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), len(test_vals.shape))
        for shape_idx in range(len(true_vals.shape)):
            self.assertEqual(
                true_vals.shape[shape_idx], test_vals.shape[shape_idx])

        # Check values column by columns.  To allow for matrices or arrays,
        # turn columns into arrays and squeeze them (forcing 1D arrays).  This
        # avoids failures due to trivial shape mismatches.
        for col_idx in range(true_vals.shape[1]):
            true_col = np.array(true_vals[:, col_idx]).squeeze()
            test_col = np.array(test_vals[:, col_idx]).squeeze()
            self.assertTrue(
                np.allclose(true_col, test_col, rtol=rtol, atol=atol) 
                or
                np.allclose(-true_col, test_col, rtol=rtol, atol=atol))

    def _helper_check_decomp(
        self, method_type, vecs, mode_indices, inner_product, 
        inner_product_weights, rtol, atol, adv_vecs=None, max_num_eigvals=None):

        # Compute reference values for testing DMD computation
        (modes_exact_true, modes_proj_true, spectral_coeffs_true, 
            eigvals_true, R_low_order_eigvecs_true, L_low_order_eigvecs_true,
            correlation_mat_eigvals_true, correlation_mat_eigvecs_true,
            correlation_mat_true, cross_correlation_mat_true) = (
            self._helper_compute_DMD_from_data(
            vecs, inner_product, adv_vecs=adv_vecs,
            max_num_eigvals=max_num_eigvals))
 
        # Compute DMD using modred method of choice
        if method_type == 'snaps':
            (modes_exact, modes_proj, eigvals, spectral_coeffs, 
                R_low_order_eigvecs, L_low_order_eigvecs,
                correlation_mat_eigvals, correlation_mat_eigvecs, 
                correlation_mat, cross_correlation_mat) = (
                compute_DMD_matrices_snaps_method(
                vecs, mode_indices, adv_vecs=adv_vecs,
                inner_product_weights=inner_product_weights,
                max_num_eigvals=max_num_eigvals, return_all=True))
        elif method_type == 'direct':
            (modes_exact, modes_proj, eigvals, spectral_coeffs, 
                R_low_order_eigvecs, L_low_order_eigvecs,
                correlation_mat_eigvals, correlation_mat_eigvecs) = (
                compute_DMD_matrices_direct_method(
                vecs, mode_indices, adv_vecs=adv_vecs,
                inner_product_weights=inner_product_weights,
                max_num_eigvals=max_num_eigvals, return_all=True))
        else:
            raise ValueError('Invalid DMD matrix method.')

        # Compare values to reference values, allowing for sign differences in
        # some cases.  For the low-order eigenvectors, check that the elements
        # differ at most by a sign, as the eigenvectors may vary by sign even
        # element-wise.  This is due to the fact that the low-order linear maps
        # may have sign differences, as they depend on the correlation matrix
        # eigenvectors, which themselves may have column-wise sign differences.
        self._helper_test_mat_to_sign(
            modes_exact, modes_exact_true[:, mode_indices], rtol=rtol, 
            atol=atol)
        self._helper_test_mat_to_sign(
            modes_proj, modes_proj_true[:, mode_indices], rtol=rtol, 
            atol=atol)
        self._helper_test_mat_to_sign(
            np.mat(spectral_coeffs), np.mat(spectral_coeffs_true), 
            rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            eigvals, eigvals_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            np.abs(R_low_order_eigvecs / R_low_order_eigvecs_true), 
            np.ones(R_low_order_eigvecs.shape), 
            rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            np.abs(L_low_order_eigvecs / L_low_order_eigvecs_true), 
            np.ones(L_low_order_eigvecs.shape), 
            rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            correlation_mat_eigvals, correlation_mat_eigvals_true, 
            rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            correlation_mat_eigvecs, correlation_mat_eigvecs_true, 
            rtol=rtol, atol=atol)
        if method_type == 'snaps':
            np.testing.assert_allclose(
                correlation_mat, correlation_mat_true, rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                cross_correlation_mat, cross_correlation_mat_true, 
                rtol=rtol, atol=atol)

    def test_all(self):
        rtol = 1e-8
        atol = 1e-15
        mode_indices = [2, 0, 3]

        # Generate weight matrices for inner products, which should all be
        # positive semidefinite.
        weights_full = np.mat(
            np.random.random((self.num_states, self.num_states)))
        weights_full = 0.5 * (weights_full + weights_full.T)
        weights_full = weights_full + self.num_states * np.eye(self.num_states)
        weights_diag = np.random.random(self.num_states)
        weights_list = [None, weights_diag, weights_full]
        for weights in weights_list:
            IP = VectorSpaceMatrices(weights=weights).compute_inner_product_mat
            vecs = np.random.random((self.num_states, self.num_vecs))

            # Test DMD for a sequential dataset, method of snapshots
            self._helper_check_decomp(
                'snaps', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=None) 
            
            # Check that truncation works
            max_num_eigvals = int(np.round(self.num_vecs / 2))
            self._helper_check_decomp(
                'snaps', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=None, max_num_eigvals=max_num_eigvals) 

            # Test DMD for a sequential dataset, direct method
            self._helper_check_decomp(
                'direct', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=None) 
           
            # Check that truncation works
            self._helper_check_decomp(
                'direct', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=None, max_num_eigvals=max_num_eigvals) 
 
            # Generate data for a non-sequential dataset
            adv_vecs = np.random.random((self.num_states, self.num_vecs))

            # Test DMD for a non-sequential dataset, method of snapshots
            self._helper_check_decomp(
                'snaps', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=adv_vecs) 
            
            # Check that truncation works
            self._helper_check_decomp(
                'snaps', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=adv_vecs, max_num_eigvals=max_num_eigvals) 

            # Test DMD for a non-sequential dataset, direct method
            self._helper_check_decomp(
                'direct', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=adv_vecs) 

            # Check that truncation works
            self._helper_check_decomp(
                'direct', vecs, mode_indices, IP, weights, rtol, atol,
                adv_vecs=adv_vecs, max_num_eigvals=max_num_eigvals) 


#@unittest.skip('others')
class TestDMDHandles(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(self.test_dir) and _parallel.is_rank_zero():
            os.mkdir(self.test_dir)
        
        self.num_vecs = 10
        self.num_states = 20
        self.my_DMD = DMDHandles(np.vdot, verbosity=0)

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
    
    #@unittest.skip('Testing something else.')
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        def my_load(fname): pass
        def my_save(data, fname): pass
        def my_IP(vec1, vec2): pass
       
        data_members_default = {
            'put_mat': util.save_array_text, 'get_mat': util.load_array_text,
            'verbosity': 0, 'eigvals': None, 'correlation_mat': None,
            'cross_correlation_mat': None, 'correlation_mat_eigvals': None,
            'correlation_mat_eigvecs': None, 'low_order_linear_map': None,
            'L_low_order_eigvecs': None, 'R_low_order_eigvecs': None,
            'spectral_coeffs': None, 'proj_coeffs': None, 'adv_proj_coeffs':
            None, 'vec_handles': None, 'adv_vec_handles': None, 'vec_space':
            VectorSpaceHandles(my_IP, verbosity=0)}
        
        # Get default data member values
        for k,v in util.get_data_members(
            DMDHandles(my_IP, verbosity=0)).items():
            self.assertEqual(v, data_members_default[k])
        
        my_DMD = DMDHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])
       
        my_DMD = DMDHandles(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])
 
        my_DMD = DMDHandles(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_DMD).items():
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
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])

    #@unittest.skip('Testing something else.')
    def test_puts_gets(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(test_dir) and _parallel.is_rank_zero():
            os.mkdir(test_dir)
        eigvals = _parallel.call_and_bcast(np.random.random, 5)
        R_low_order_eigvecs = _parallel.call_and_bcast(
            np.random.random, (10,10))
        L_low_order_eigvecs = _parallel.call_and_bcast(
            np.random.random, (10,10))
        correlation_mat_eigvals = _parallel.call_and_bcast(np.random.random, 5)
        correlation_mat_eigvecs = _parallel.call_and_bcast(
            np.random.random, (10,10))
        correlation_mat = _parallel.call_and_bcast(np.random.random, (10,10))
        cross_correlation_mat = _parallel.call_and_bcast(
            np.random.random, (10,10))
        spectral_coeffs = _parallel.call_and_bcast(np.random.random, 5)
        proj_coeffs = _parallel.call_and_bcast(np.random.random, 5)
        adv_proj_coeffs = _parallel.call_and_bcast(np.random.random, 5)

        my_DMD = DMDHandles(None, verbosity=0)
        my_DMD.eigvals = eigvals
        my_DMD.R_low_order_eigvecs = R_low_order_eigvecs 
        my_DMD.L_low_order_eigvecs = L_low_order_eigvecs 
        my_DMD.correlation_mat_eigvals = correlation_mat_eigvals
        my_DMD.correlation_mat_eigvecs = correlation_mat_eigvecs
        my_DMD.correlation_mat = correlation_mat
        my_DMD.cross_correlation_mat = cross_correlation_mat
        my_DMD.spectral_coeffs = spectral_coeffs
        my_DMD.proj_coeffs = proj_coeffs
        my_DMD.adv_proj_coeffs = adv_proj_coeffs
        
        eigvals_path = join(test_dir, 'dmd_eigvals.txt')
        R_low_order_eigvecs_path = join(test_dir, 'dmd_R_low_order_eigvecs.txt')
        L_low_order_eigvecs_path = join(test_dir, 'dmd_L_low_order_eigvecs.txt')
        correlation_mat_eigvals_path = join(
            test_dir, 'dmd_corr_mat_eigvals.txt')
        correlation_mat_eigvecs_path = join(
            test_dir, 'dmd_corr_mat_eigvecs.txt')
        correlation_mat_path = join(test_dir, 'dmd_corr_mat.txt')
        cross_correlation_mat_path = join(test_dir, 'dmd_cross_corr_mat.txt')
        spectral_coeffs_path = join(test_dir, 'dmd_spectral_coeffs.txt')
        proj_coeffs_path = join(test_dir, 'dmd_proj_coeffs.txt')
        adv_proj_coeffs_path = join(test_dir, 'dmd_adv_proj_coeffs.txt')

        my_DMD.put_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            correlation_mat_eigvals_path , correlation_mat_eigvecs_path) 
        my_DMD.put_correlation_mat(correlation_mat_path)
        my_DMD.put_cross_correlation_mat(cross_correlation_mat_path)
        my_DMD.put_spectral_coeffs(spectral_coeffs_path)
        my_DMD.put_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)
        _parallel.barrier()

        DMD_load = DMDHandles(None, verbosity=0)
        DMD_load.get_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            correlation_mat_eigvals_path, correlation_mat_eigvecs_path)
        correlation_mat_loaded = util.load_array_text(correlation_mat_path)
        cross_correlation_mat_loaded = util.load_array_text(
            cross_correlation_mat_path)
        spectral_coeffs_loaded = np.squeeze(np.array(
            util.load_array_text(spectral_coeffs_path)))
        proj_coeffs_loaded = np.squeeze(np.array(
            util.load_array_text(proj_coeffs_path)))
        adv_proj_coeffs_loaded = np.squeeze(np.array(
            util.load_array_text(adv_proj_coeffs_path)))

        np.testing.assert_allclose(DMD_load.eigvals, eigvals)
        np.testing.assert_allclose(
            DMD_load.R_low_order_eigvecs, R_low_order_eigvecs)
        np.testing.assert_allclose(
            DMD_load.L_low_order_eigvecs, L_low_order_eigvecs)
        np.testing.assert_allclose(
            DMD_load.correlation_mat_eigvals, correlation_mat_eigvals)
        np.testing.assert_allclose(
            DMD_load.correlation_mat_eigvecs, correlation_mat_eigvecs)
        np.testing.assert_allclose(correlation_mat_loaded, correlation_mat)
        np.testing.assert_allclose(
            cross_correlation_mat_loaded, cross_correlation_mat)
        np.testing.assert_allclose(spectral_coeffs_loaded, spectral_coeffs)
        np.testing.assert_allclose(proj_coeffs_loaded, proj_coeffs)
        np.testing.assert_allclose(adv_proj_coeffs_loaded, adv_proj_coeffs)

    def _helper_compute_DMD_from_data(
        self, vec_array, inner_product, adv_vec_array=None,
        max_num_eigvals=None):
        # Generate adv_vec_array for the case of a sequential dataset
        if adv_vec_array is None:
            adv_vec_array = vec_array[:, 1:]
            vec_array = vec_array[:, :-1]

        # Create lists of vecs, advanced vecs for inner product function
        vecs = [vec_array[:, i] for i in range(vec_array.shape[1])]
        adv_vecs = [adv_vec_array[:, i] for i in range(adv_vec_array.shape[1])]

        # Compute SVD of data vectors
        correlation_mat = inner_product(vecs, vecs)
        correlation_mat_eigvals, correlation_mat_eigvecs = util.eigh(
            correlation_mat) 
        U = vec_array.dot(np.array(correlation_mat_eigvecs)).dot(
            np.diag(correlation_mat_eigvals ** -0.5))
        U_list = [U[:, i] for i in range(U.shape[1])]

        # Truncate SVD if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < correlation_mat_eigvals.size):
            correlation_mat_eigvals = correlation_mat_eigvals[:max_num_eigvals]
            correlation_mat_eigvecs = correlation_mat_eigvecs[
                :, :max_num_eigvals]
            U = U[:, :max_num_eigvals]
            U_list = U_list[:max_num_eigvals]

        # Compute eigendecomposition of low order linear operator
        A_tilde = inner_product(U_list, adv_vecs).dot(
            np.array(correlation_mat_eigvecs)).dot(
            np.diag(correlation_mat_eigvals ** -0.5))
        eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
            util.eig_biorthog(A_tilde, scale_choice='left')
        R_low_order_eigvecs = np.mat(R_low_order_eigvecs)
        L_low_order_eigvecs = np.mat(L_low_order_eigvecs)

        # Compute build coefficients
        build_coeffs_proj = (
            correlation_mat_eigvecs.dot(
            np.diag(correlation_mat_eigvals ** -0.5)).dot(R_low_order_eigvecs))
        build_coeffs_exact = (
            correlation_mat_eigvecs.dot(
            np.diag(correlation_mat_eigvals ** -0.5)).dot(
            R_low_order_eigvecs).dot(
            np.diag(eigvals ** -1.)))
 
        # Compute modes
        modes_proj = vec_array.dot(build_coeffs_proj)
        modes_exact = adv_vec_array.dot(build_coeffs_exact)
        adj_modes = U.dot(L_low_order_eigvecs)
        adj_modes_list = [
            np.array(adj_modes[:, i]) for i in range(adj_modes.shape[1])]

        # Compute spectrum 
        spectral_coeffs = np.abs(np.array(
            inner_product(adj_modes_list, vecs[0:1])).squeeze())

        return (
            modes_exact, modes_proj, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs, adj_modes)

    def _helper_test_1D_array_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), 1)
        self.assertEqual(len(test_vals.shape), 1)
        self.assertEqual(true_vals.size, test_vals.size)

        # Check values entry by entry.  
        for idx in range(true_vals.size):
            true_val = true_vals[idx]
            test_val = test_vals[idx]
            self.assertTrue(
                np.allclose(true_val, test_val, rtol=rtol, atol=atol) 
                or
                np.allclose(-true_val, test_val, rtol=rtol, atol=atol))
    
    def _helper_test_mat_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), len(test_vals.shape))
        for shape_idx in range(len(true_vals.shape)):
            self.assertEqual(
                true_vals.shape[shape_idx], test_vals.shape[shape_idx])

        # Check values column by columns.  To allow for matrices or arrays,
        # turn columns into arrays and squeeze them (forcing 1D arrays).  This
        # avoids failures due to trivial shape mismatches.
        for col_idx in range(true_vals.shape[1]):
            true_col = np.array(true_vals[:, col_idx]).squeeze()
            test_col = np.array(test_vals[:, col_idx]).squeeze()
            self.assertTrue(
                np.allclose(true_col, test_col, rtol=rtol, atol=atol) 
                or
                np.allclose(-true_col, test_col, rtol=rtol, atol=atol))

    def _helper_check_decomp(
        self, vec_array,  vec_handles, adv_vec_array=None,
        adv_vec_handles=None, max_num_eigvals=None):
        # Set tolerance.  
        tol = 1e-10

        # Compute reference DMD values
        (eigvals_true, R_low_order_eigvecs_true, L_low_order_eigvecs_true,
            correlation_mat_eigvals_true, correlation_mat_eigvecs_true) = (
            self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array, max_num_eigvals=max_num_eigvals))[3:-1]

        # Compute DMD using modred
        (eigvals_returned,  R_low_order_eigvecs_returned,
            L_low_order_eigvecs_returned, correlation_mat_eigvals_returned,
            correlation_mat_eigvecs_returned) = self.my_DMD.compute_decomp(
            vec_handles, adv_vec_handles=adv_vec_handles, 
            max_num_eigvals=max_num_eigvals)

        # Test that matrices were correctly computed.  For build coeffs, check
        # column by column, as it is ok to be off by a negative sign.
        np.testing.assert_allclose(
            self.my_DMD.eigvals, eigvals_true, rtol=tol)
        self._helper_test_mat_to_sign(
            self.my_DMD.R_low_order_eigvecs, R_low_order_eigvecs_true, rtol=tol)
        self._helper_test_mat_to_sign(
            self.my_DMD.L_low_order_eigvecs, L_low_order_eigvecs_true, rtol=tol)
        np.testing.assert_allclose(
            self.my_DMD.correlation_mat_eigvals, correlation_mat_eigvals_true, 
            rtol=tol)
        self._helper_test_mat_to_sign(
            self.my_DMD.correlation_mat_eigvecs, correlation_mat_eigvecs_true, 
            rtol=tol)

        # Test that matrices were correctly returned
        np.testing.assert_allclose(
            eigvals_returned, eigvals_true, rtol=tol)
        self._helper_test_mat_to_sign(
            R_low_order_eigvecs_returned, R_low_order_eigvecs_true, rtol=tol)
        self._helper_test_mat_to_sign(
            L_low_order_eigvecs_returned, L_low_order_eigvecs_true, rtol=tol)
        np.testing.assert_allclose(
            correlation_mat_eigvals_returned, correlation_mat_eigvals_true, 
            rtol=tol)
        self._helper_test_mat_to_sign(
            correlation_mat_eigvecs_returned, correlation_mat_eigvecs_true, 
            rtol=tol)

    def _helper_check_modes(self, modes_true, mode_path_list):
        # Set tolerance. 
        tol = 1e-10

        # Load all modes into matrix, compare to modes from direct computation
        modes_computed = np.zeros(modes_true.shape, dtype=complex)
        for i, path in enumerate(mode_path_list):
            modes_computed[:, i] = V.VecHandlePickle(path).get()
        np.testing.assert_allclose(modes_true, modes_computed, rtol=tol)

    #@unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test DMD decomposition (eigenvalues, build coefficients)"""    
        # Define an array of vectors, with corresponding handles
        vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())

        # Check modred against direct computation, for a sequential dataset
        _parallel.barrier()
        self._helper_check_decomp(vec_array, self.vec_handles)

        # Make sure truncation works
        max_num_eigvals = int(np.round(self.num_vecs / 2))
        self._helper_check_decomp(vec_array, self.vec_handles, 
            max_num_eigvals=max_num_eigvals)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Check modred against direct computation, for a non-sequential dataset
        _parallel.barrier()
        self._helper_check_decomp(
            vec_array, self.vec_handles, adv_vec_array=adv_vec_array,
            adv_vec_handles=self.adv_vec_handles)

        # Make sure truncation works
        self._helper_check_decomp(
            vec_array, self.vec_handles, adv_vec_array=adv_vec_array,
            adv_vec_handles=self.adv_vec_handles,
            max_num_eigvals=max_num_eigvals)
        
        # Check that if mismatched sets of handles are passed in, an error is
        # raised.
        self.assertRaises(ValueError, self.my_DMD.compute_decomp,
            self.vec_handles, self.adv_vec_handles[:-1])


    #@unittest.skip('Testing something else.')
    def test_compute_modes(self):
        """Test building of modes."""
        # Generate path names for saving modes to disk
        mode_path = join(self.test_dir, 'dmd_mode_%03d.pkl')
       
        ### SEQUENTIAL DATASET ###
        # Generate data 
        seq_vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(seq_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data
        (modes_exact, modes_proj, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs) = self._helper_compute_DMD_from_data(
            seq_vec_array, util.InnerProductBlock(np.vdot))[:-1]

        # Set the build_coeffs attribute of an empty DMD object each time, so
        # that the modred computation uses the same coefficients as the direct
        # computation.
        _parallel.barrier()
        self.my_DMD.eigvals = eigvals
        self.my_DMD.R_low_order_eigvecs = R_low_order_eigvecs
        self.my_DMD.correlation_mat_eigvals = correlation_mat_eigvals
        self.my_DMD.correlation_mat_eigvecs = correlation_mat_eigvecs

        # Generate mode paths for saving modes to disk
        seq_mode_path_list = [
            mode_path % i for i in range(eigvals.size)]
        seq_mode_indices = range(len(seq_mode_path_list))

        # Compute modes by passing in handles
        self.my_DMD.compute_exact_modes(seq_mode_indices, 
            [V.VecHandlePickle(path) for path in seq_mode_path_list],
            adv_vec_handles=self.vec_handles[1:])
        self._helper_check_modes(modes_exact, seq_mode_path_list)
        self.my_DMD.compute_proj_modes(seq_mode_indices, 
            [V.VecHandlePickle(path) for path in seq_mode_path_list],
            vec_handles=self.vec_handles)
        self._helper_check_modes(modes_proj, seq_mode_path_list)

        # Compute modes without passing in handles, so first set full
        # sequential dataset as vec_handles.
        self.my_DMD.vec_handles = self.vec_handles
        self.my_DMD.compute_exact_modes(seq_mode_indices, 
            [V.VecHandlePickle(path) for path in seq_mode_path_list])
        self._helper_check_modes(modes_exact, seq_mode_path_list)
        self.my_DMD.compute_proj_modes(seq_mode_indices, 
            [V.VecHandlePickle(path) for path in seq_mode_path_list])
        self._helper_check_modes(modes_proj, seq_mode_path_list)

        # For exact modes, also compute by setting adv_vec_handles
        self.my_DMD.vec_handles = None
        self.my_DMD.adv_vec_handles = self.vec_handles[1:]
        self.my_DMD.compute_exact_modes(seq_mode_indices, 
            [V.VecHandlePickle(path) for path in seq_mode_path_list])
        self._helper_check_modes(modes_exact, seq_mode_path_list)

        ### NONSEQUENTIAL DATA ###
        # Generate data 
        vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        adv_vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, (handle, adv_handle) in enumerate(
                zip(self.vec_handles, self.adv_vec_handles)):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())
                adv_handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data
        (modes_exact, modes_proj, spectral_coeffs, eigvals,
            R_low_order_eigvecs, L_low_order_eigvecs, correlation_mat_eigvals,
            correlation_mat_eigvecs) = self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array)[:-1]

        # Set the build_coeffs attribute of an empty DMD object each time, so
        # that the modred computation uses the same coefficients as the direct
        # computation.
        _parallel.barrier()
        self.my_DMD.eigvals = eigvals
        self.my_DMD.R_low_order_eigvecs = R_low_order_eigvecs
        self.my_DMD.correlation_mat_eigvals = correlation_mat_eigvals
        self.my_DMD.correlation_mat_eigvecs = correlation_mat_eigvecs

        # Generate mode paths for saving modes to disk
        mode_path_list = [
            mode_path % i for i in range(eigvals.size)]
        mode_indices = range(len(mode_path_list))

        # Compute modes by passing in handles
        self.my_DMD.compute_exact_modes(mode_indices, 
            [V.VecHandlePickle(path) for path in mode_path_list],
            adv_vec_handles=self.adv_vec_handles)
        self._helper_check_modes(modes_exact, mode_path_list)
        self.my_DMD.compute_proj_modes(mode_indices, 
            [V.VecHandlePickle(path) for path in mode_path_list],
            vec_handles=self.vec_handles)
        self._helper_check_modes(modes_proj, mode_path_list)

        # Compute modes without passing in handles, so first set full
        # sequential dataset as vec_handles.
        self.my_DMD.vec_handles = self.vec_handles
        self.my_DMD.adv_vec_handles = self.adv_vec_handles
        self.my_DMD.compute_exact_modes(mode_indices, 
            [V.VecHandlePickle(path) for path in mode_path_list])
        self._helper_check_modes(modes_exact, mode_path_list)
        self.my_DMD.compute_proj_modes(mode_indices, 
            [V.VecHandlePickle(path) for path in mode_path_list])
        self._helper_check_modes(modes_proj, mode_path_list)

    #@unittest.skip('Testing something else.')
    def test_compute_spectrum(self):
        """Test DMD spectrum""" 
        rtol = 1e-10
        atol = 1e-16

        # Define an array of vectors, with corresponding handles
        vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())

        # Check that spectral coefficients computed using adjoints match those
        # computed using a pseudoinverse.  Perform the decomp only once using
        # the DMD object, so that the spectral coefficients are computed from
        # the same data, but in two different ways.
        _parallel.barrier()
        self.my_DMD.compute_decomp(self.vec_handles) 
        spectral_coeffs_computed = self.my_DMD.compute_spectrum()
        spectral_coeffs_true = self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot))[2]
        self._helper_test_1D_array_to_sign(
            spectral_coeffs_true, spectral_coeffs_computed, rtol=rtol, 
            atol=atol)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Check that spectral coefficients computed using adjoints match those
        # computed using a pseudoinverse
        _parallel.barrier()
        self.my_DMD.compute_decomp(
            self.vec_handles, adv_vec_handles=self.adv_vec_handles) 
        spectral_coeffs_computed = self.my_DMD.compute_spectrum()
        spectral_coeffs_true = self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array)[2]
        self._helper_test_1D_array_to_sign(
            spectral_coeffs_true, spectral_coeffs_computed, rtol=rtol, 
            atol=atol)
    
    #@unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        rtol = 1e-10
        atol = 1e-13    # Sometimes fails if tol too high

        # Define an array of vectors, with corresponding handles
        vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())

        # Check the spectral coefficient values.  Sometimes the values in the
        # projections are not just scaled by -1 column-wise, but element-wise.
        # So just test that the projection coefficients differ by sign at most,
        # element-wise.
        _parallel.barrier()
        self.my_DMD.compute_decomp(self.vec_handles) 
        proj_coeffs, adv_proj_coeffs = self.my_DMD.compute_proj_coeffs()
        adj_modes = self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot))[-1]
        proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array[:, :-1])
        adv_proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array[:, 1:])
        np.testing.assert_allclose(
            np.abs(proj_coeffs / proj_coeffs_true), np.ones(proj_coeffs.shape),
            rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            np.abs(adv_proj_coeffs / adv_proj_coeffs_true),
            np.ones(adv_proj_coeffs.shape), rtol=rtol, atol=atol)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Check the spectral coefficient values.  Sometimes the values in the
        # projections are not just scaled by -1 column-wise, but element-wise.
        # So just test that the projection coefficients differ by sign at most,
        # element-wise.
        _parallel.barrier()
        self.my_DMD.compute_decomp(
            self.vec_handles, adv_vec_handles=self.adv_vec_handles) 
        adj_modes = self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array)[-1]
        proj_coeffs, adv_proj_coeffs= self.my_DMD.compute_proj_coeffs()
        proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array)
        adv_proj_coeffs_true = np.dot(adj_modes.conj().T, adv_vec_array)
        np.testing.assert_allclose(
            np.abs(proj_coeffs / proj_coeffs_true), np.ones(proj_coeffs.shape),
            rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            np.abs(adv_proj_coeffs / adv_proj_coeffs_true),
            np.ones(adv_proj_coeffs.shape), rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()

