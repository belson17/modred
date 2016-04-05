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
        self.num_vecs = 6
        self.num_states = 12


    def _helper_compute_DMD_from_data(
        self, vecs, adv_vecs, inner_product):
        correlation_mat = inner_product(vecs, vecs)
        V, Sigma, dummy = util.svd(correlation_mat) # dummy = W.
        U = vecs.dot(V).dot(np.diag(Sigma ** -0.5))
        A_tilde = inner_product(
            U, adv_vecs).dot(V).dot(np.diag(Sigma ** -0.5))
        eigvals, W, Z = util.eig_biorthog(
            A_tilde, scale_choice='left')
        build_coeffs_proj = V.dot(np.diag(Sigma ** -0.5)).dot(W)
        build_coeffs_exact = (
            V.dot(np.diag(Sigma ** -0.5)).dot(W).dot(np.diag(eigvals ** -1.)))
        modes_proj = vecs.dot(build_coeffs_proj)
        modes_exact = adv_vecs.dot(build_coeffs_exact)
        spectral_coeffs = np.linalg.lstsq(modes_proj, vecs[:, 0])[0]
        return (
            modes_exact, modes_proj, eigvals, spectral_coeffs, 
            build_coeffs_exact, build_coeffs_proj) 


    def _helper_test_mat_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), len(test_vals.shape))
        for shape_idx in xrange(len(true_vals.shape)):
            self.assertEqual(
                true_vals.shape[shape_idx], test_vals.shape[shape_idx])

        # Check values column by columns.  To allow for matrices or arrays,
        # turn columns into arrays and squeeze them (forcing 1D arrays).  This
        # avoids failures due to trivial shape mismatches.
        for col_idx in xrange(true_vals.shape[1]):
            true_col = np.array(true_vals[:, col_idx]).squeeze()
            test_col = np.array(test_vals[:, col_idx]).squeeze()
            self.assertTrue(
                np.allclose(true_col, test_col, rtol=rtol, atol=atol) 
                or
                np.allclose(-true_col, test_col, rtol=rtol, atol=atol))


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

            # Test DMD for a sequential dataset
            (modes_exact_true, modes_proj_true, eigvals_true,
                spectral_coeffs_true, build_coeffs_exact_true,
                build_coeffs_proj_true) = (
                self._helper_compute_DMD_from_data(
                vecs[:, :-1], vecs[:, 1:], IP))

            # Test method of snapshots, allowing for sign difference in modes, 
            # build coeffs, and spectral coeffs
            (modes_exact, modes_proj, eigvals, spectral_coeffs, 
                build_coeffs_exact, build_coeffs_proj) = (
                compute_DMD_matrices_snaps_method(vecs, mode_indices, 
                inner_product_weights=weights, return_all=True))
            np.testing.assert_allclose(
                eigvals, eigvals_true, rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                np.mat(spectral_coeffs), np.mat(spectral_coeffs_true), 
                rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                modes_exact, modes_exact_true[:, mode_indices], rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_proj, modes_proj_true[:, mode_indices], rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_exact, build_coeffs_exact_true, rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_proj, build_coeffs_proj_true, rtol=rtol, atol=atol)
            
            # Test direct method, allowing for sign difference in modes, build
            # coeffs, and spectral coeffs
            (modes_exact, modes_proj, eigvals, spectral_coeffs, 
                build_coeffs_exact, build_coeffs_proj) = (
                compute_DMD_matrices_direct_method(vecs, mode_indices, 
                inner_product_weights=weights, return_all=True))
            np.testing.assert_allclose(eigvals, eigvals_true, rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                np.mat(spectral_coeffs), np.mat(spectral_coeffs_true), 
                rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_proj, build_coeffs_proj_true, rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_exact, build_coeffs_exact_true, rtol=rtol,
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_proj, modes_proj_true[:, mode_indices], rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_exact, modes_exact_true[:, mode_indices], rtol=rtol, 
                atol=atol)
                       
            # Generate data for a non-sequential dataset
            adv_vecs = np.random.random((self.num_states, self.num_vecs))

            # Compute true DMD modes from data for a non-sequential dataset
            (modes_exact_true, modes_proj_true, eigvals_true, 
                spectral_coeffs_true, build_coeffs_exact_true,
                build_coeffs_proj_true) = self._helper_compute_DMD_from_data(
                vecs, adv_vecs, IP)
            
            # Compare computed modes to truth
            (modes_exact, modes_proj, eigvals, spectral_coeffs, 
                build_coeffs_exact, build_coeffs_proj) = (
                compute_DMD_matrices_snaps_method(vecs, mode_indices, 
                adv_vecs=adv_vecs, inner_product_weights=weights, 
                return_all=True))
            np.testing.assert_allclose(eigvals, eigvals_true, rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                np.mat(spectral_coeffs), np.mat(spectral_coeffs_true), 
                rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_proj, build_coeffs_proj_true, rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_exact, build_coeffs_exact_true, rtol=rtol,
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_proj, modes_proj_true[:, mode_indices], rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_exact, modes_exact_true[:, mode_indices], rtol=rtol, 
                atol=atol)

            (modes_exact, modes_proj, eigvals, spectral_coeffs, 
                build_coeffs_exact, build_coeffs_proj) = (
                compute_DMD_matrices_direct_method(vecs, mode_indices, 
                adv_vecs=adv_vecs, inner_product_weights=weights, 
                return_all=True))
            np.testing.assert_allclose(eigvals, eigvals_true, rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                np.mat(spectral_coeffs), np.mat(spectral_coeffs_true), 
                rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_proj, build_coeffs_proj_true, rtol=rtol, atol=atol)
            self._helper_test_mat_to_sign(
                build_coeffs_exact, build_coeffs_exact_true, rtol=rtol,
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_proj, modes_proj_true[:, mode_indices], rtol=rtol, 
                atol=atol)
            self._helper_test_mat_to_sign(
                modes_exact, modes_exact_true[:, mode_indices], rtol=rtol, 
                atol=atol)

           
#@unittest.skip('others')
class TestDMDHandles(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(self.test_dir) and _parallel.is_rank_zero():
            os.mkdir(self.test_dir)
        
        # TODO:  randomize size?
        self.num_vecs = 6
        self.num_states = 12
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
       
        # TODO: adjust this list
        data_members_default = {
            'put_mat': util.save_array_text, 'get_mat': util.load_array_text,
            'verbosity': 0, 'eigvals': None, 'build_coeffs_exact': None,
            'build_coeffs_proj': None, 'correlation_mat': None,
            'cross_correlation_mat': None, 'correlation_mat_eigvals': None,
            'correlation_mat_eigvecs': None, 'low_order_linear_map': None,
            'eigvals': None, 'L_eigvals': None, 'L_low_order_eigvecs': None,
            'R_low_order_eigvecs': None, 'spectral_coeffs': None,
            'proj_coeffs': None, 'adv_proj_coeffs': None, 'vec_handles': None,
            'adv_vec_handles': None, 'vec_space':
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
    def test_gets_puts(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(test_dir) and _parallel.is_rank_zero():
            os.mkdir(test_dir)
        eigvals = _parallel.call_and_bcast(np.random.random, 5)
        build_coeffs_exact = _parallel.call_and_bcast(np.random.random, (10,10))
        build_coeffs_proj = _parallel.call_and_bcast(np.random.random, (10,10))
        
        my_DMD = DMDHandles(None, verbosity=0)
        my_DMD.eigvals = eigvals
        my_DMD.build_coeffs_exact = build_coeffs_exact
        my_DMD.build_coeffs_proj = build_coeffs_proj
        
        eigvals_path = join(test_dir, 'dmd_eigvals.txt')
        build_coeffs_exact_path = join(test_dir, 'dmd_build_coeffs_exact.txt')
        build_coeffs_proj_path = join(test_dir, 'dmd_build_coeffs_proj.txt')
        
        my_DMD.put_decomp(eigvals_path, build_coeffs_exact_path, 
            build_coeffs_proj_path)
        _parallel.barrier()

        DMD_load = DMDHandles(None, verbosity=0)
        DMD_load.get_decomp(
            eigvals_path, build_coeffs_exact_path, build_coeffs_proj_path)

        np.testing.assert_allclose(DMD_load.eigvals, eigvals)
        np.testing.assert_allclose(
            DMD_load.build_coeffs_exact, build_coeffs_exact)
        np.testing.assert_allclose(DMD_load.build_coeffs_proj,
            build_coeffs_proj)


    def _helper_compute_DMD_from_data(self, vec_array, adv_vec_array,
        inner_product):
        # Create lists of vecs, advanced vecs for inner product function
        vecs = [vec_array[:, i] for i in range(vec_array.shape[1])]
        adv_vecs = [adv_vec_array[:, i] for i in range(adv_vec_array.shape[1])]

        # Compute SVD of data vectors
        correlation_mat = inner_product(vecs, vecs)
        correlation_mat_eigvals, correlation_mat_eigvecs = util.eigh(
            correlation_mat) 
        U = vec_array.dot(np.array(correlation_mat_eigvecs)).dot(
            np.diag(correlation_mat_eigvals ** -0.5))
        U_list = [U[:,i] for i in range(U.shape[1])]

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

        # Compute DMD modes
        eigvecs = vec_array.dot(build_coeffs_proj)
        eigvecs_list = [np.array(eigvecs[:,i]).squeeze() 
            for i in range(eigvecs.shape[1])]

        # Compute spectrum using pesudoinverse, which is analytically
        # equivalent to the adjoint approach used in the DMD class.  This
        # should work so long as the eigenvector matrices are full rank, which
        # is the case as long as there are no Jordan blocks.  For random data,
        # this should be ok.
        spectral_coeffs = np.linalg.lstsq(eigvecs, vec_array[:, 0])[0]

        return (
            eigvals, modes_exact, modes_proj, build_coeffs_exact, 
            build_coeffs_proj, spectral_coeffs, adj_modes)

    
    def _helper_test_mat_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), len(test_vals.shape))
        for shape_idx in xrange(len(true_vals.shape)):
            self.assertEqual(
                true_vals.shape[shape_idx], test_vals.shape[shape_idx])

        # Check values column by columns.  To allow for matrices or arrays,
        # turn columns into arrays and squeeze them (forcing 1D arrays).  This
        # avoids failures due to trivial shape mismatches.
        for col_idx in xrange(true_vals.shape[1]):
            true_col = np.array(true_vals[:, col_idx]).squeeze()
            test_col = np.array(test_vals[:, col_idx]).squeeze()
            self.assertTrue(
                np.allclose(true_col, test_col, rtol=rtol, atol=atol) 
                or
                np.allclose(-true_col, test_col, rtol=rtol, atol=atol))


    def _helper_check_decomp(self, eigvals, build_coeffs_exact, 
        build_coeffs_proj, vec_handles, adv_vec_handles=None):
        # Set tolerance.  
        tol = 1e-10

        # Compute DMD using modred
        (eigvals_returned, build_coeffs_exact_returned, 
            build_coeffs_proj_returned) = self.my_DMD.compute_decomp(
            vec_handles, adv_vec_handles=adv_vec_handles)

        # Test that matrices were correctly computed.  For build coeffs, check
        # column by column, as it is ok to be off by a negative sign.
        np.testing.assert_allclose(
            self.my_DMD.eigvals, eigvals, rtol=tol)
        self._helper_test_mat_to_sign(
            self.my_DMD.build_coeffs_exact, build_coeffs_exact, rtol=tol)
        self._helper_test_mat_to_sign(
            self.my_DMD.build_coeffs_proj, build_coeffs_proj, rtol=tol)

        # Test that matrices were correctly returned
        np.testing.assert_allclose(
            eigvals_returned, eigvals, rtol=tol)
        self._helper_test_mat_to_sign(
            build_coeffs_exact_returned, build_coeffs_exact, rtol=tol)
        self._helper_test_mat_to_sign(
            build_coeffs_proj_returned, build_coeffs_proj, rtol=tol)


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

        # Compute DMD directly from data, for a sequential dataset
        (eigvals, modes_exact, modes_proj, build_coeffs_exact, 
            build_coeffs_proj) = self._helper_compute_DMD_from_data(
            vec_array[:, :-1], vec_array[:, 1:], 
            util.InnerProductBlock(np.vdot))[:5]

        # Check modred against direct computation, for a sequential dataset
        _parallel.barrier()
        self._helper_check_decomp(
            eigvals, build_coeffs_exact, build_coeffs_proj, self.vec_handles)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = _parallel.call_and_bcast(np.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data, for a non-sequential dataset
        (eigvals, modes_exact, modes_proj, build_coeffs_exact, 
            build_coeffs_proj) = self._helper_compute_DMD_from_data(
            vec_array, adv_vec_array, util.InnerProductBlock(np.vdot))[:5]

        # Check modred against direct computation, for a non-sequential dataset
        _parallel.barrier()
        self._helper_check_decomp(
            eigvals, build_coeffs_exact, build_coeffs_proj, self.vec_handles,
            adv_vec_handles=self.adv_vec_handles)

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
        modes_exact, modes_proj, build_coeffs_exact, build_coeffs_proj = (
            self._helper_compute_DMD_from_data(
            seq_vec_array[:, :-1], seq_vec_array[:, 1:], 
            util.InnerProductBlock(np.vdot)))[1:5]

        # Set the build_coeffs attribute of an empty DMD object each time, so
        # that the modred computation uses the same coefficients as the direct
        # computation.
        _parallel.barrier()
        self.my_DMD.build_coeffs_exact = build_coeffs_exact
        self.my_DMD.build_coeffs_proj = build_coeffs_proj

        # Generate mode paths for saving modes to disk
        seq_mode_path_list = [
            mode_path % i for i in range(build_coeffs_exact.shape[1])]
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
        modes_exact, modes_proj, build_coeffs_exact, build_coeffs_proj = (
            self._helper_compute_DMD_from_data(
            vec_array, adv_vec_array, util.InnerProductBlock(np.vdot))[1:5])

        # Set the build_coeffs attribute of an empty DMD object each time, so
        # that the modred computation uses the same coefficients as the direct
        # computation.
        _parallel.barrier()
        self.my_DMD.build_coeffs_exact = build_coeffs_exact
        self.my_DMD.build_coeffs_proj = build_coeffs_proj

        # Generate mode paths for saving modes to disk
        mode_path_list = [
            mode_path % i for i in range(build_coeffs_exact.shape[1])]
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
        spectral_coeffs = np.array(
            np.linalg.pinv(self.my_DMD.R_low_order_eigvecs) * 
            np.mat(np.diag(np.sqrt(self.my_DMD.correlation_mat_eigvals))) * 
            np.mat(self.my_DMD.correlation_mat_eigvecs[0, :]).T).squeeze()
        np.testing.assert_allclose(
            spectral_coeffs_computed, spectral_coeffs, rtol=rtol, atol=atol)

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
        spectral_coeffs = np.array(
            np.linalg.pinv(self.my_DMD.R_low_order_eigvecs) * 
            np.mat(np.diag(np.sqrt(self.my_DMD.correlation_mat_eigvals))) * 
            np.mat(self.my_DMD.correlation_mat_eigvecs[0, :]).T).squeeze()
        spectral_coeffs_computed = self.my_DMD.compute_spectrum()
        np.testing.assert_allclose(
            spectral_coeffs_computed, spectral_coeffs, rtol=rtol, atol=atol)

    
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

        # TODO: Change comment, check that matches pseudoinverse?
        # Check that spectral coefficients computed using adjoints match those
        # computed using a pseudoinverse
        _parallel.barrier()
        self.my_DMD.compute_decomp(self.vec_handles) 
        proj_coeffs, adv_proj_coeffs = self.my_DMD.compute_proj_coeffs()
        adj_modes = (
            np.mat(vec_array[:, :-1]) *
            self.my_DMD.correlation_mat_eigvecs *
            np.mat(np.diag(self.my_DMD.correlation_mat_eigvals ** -0.5)) *
            self.my_DMD.L_low_order_eigvecs)
        proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array[:, :-1])
        adv_proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array[:, 1:])
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            adv_proj_coeffs, adv_proj_coeffs_true, rtol=rtol, atol=atol)

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
        adj_modes = (
            np.mat(vec_array) *
            self.my_DMD.correlation_mat_eigvecs *
            np.mat(np.diag(self.my_DMD.correlation_mat_eigvals ** -0.5)) *
            self.my_DMD.L_low_order_eigvecs)
        proj_coeffs, adv_proj_coeffs= self.my_DMD.compute_proj_coeffs()
        proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array)
        adv_proj_coeffs_true = np.dot(adj_modes.conj().T, adv_vec_array)
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            adv_proj_coeffs, adv_proj_coeffs_true, rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()

