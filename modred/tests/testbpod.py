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

import modred.parallel as parallel
from modred.bpod import *
from modred.vectorspace import *
from modred import util
from modred import vectors as V


#@unittest.skip('Testing something else.')
@unittest.skipIf(parallel.is_distributed(), 'Serial only.')
class TestBPODMatrices(unittest.TestCase):
    def setUp(self):
        self.num_states = 10

        # Later, test various SISO, MIMO configurations
        self.num_inputs = np.random.randint(1, high=self.num_states + 1)
        self.num_outputs = np.random.randint(1, high=self.num_states + 1)

        # Generate random, stable A matrix
        self.A = util.drss(self.num_states, 1, 1)[0]

        # Generate random B and C matrix
        self.B = np.mat(np.random.random((self.num_states, self.num_inputs)))
        self.C = np.mat(np.random.random((self.num_outputs, self.num_states)))


        # Compute direct snapshots
        self.num_steps = self.num_states * 2
        self.direct_vecs_mat = np.mat(np.zeros(
            (self.num_states, self.num_steps * self.num_inputs)))
        A_powers = np.mat(np.identity(self.num_states))
        for idx in xrange(self.num_steps):
            self.direct_vecs_mat[
                :, idx * self.num_inputs:(idx + 1) * self.num_inputs] =\
                A_powers * self.B
            A_powers = A_powers * self.A


    def test_all(self):
        # Set test tolerances.  Separate, more relaxed tolerances are required
        # for testing the BPOD modes, since that test requires "squaring" the
        # gramians and thus involves more ill-conditioned matrices.
        rtol = 1e-8
        atol = 1e-10
        rtol_grams = 1e-8
        atol_grams = 1e-8

        # Generate weights to test different inner products.  Keep most of the
        # weights close to one, to avoid overly weighting certain states over
        # others.
        ws = np.identity(self.num_states)
        ws[0,0] = 1.1
        ws[1,0] = 0.1
        ws[0,1] = 0.1
        weights_list = [None, 0.1 * np.random.random(self.num_states) + 1., ws]
        weights_mats = [
            np.mat(np.identity(self.num_states)),
            np.mat(np.diag(weights_list[1])),
            np.mat(ws)]

        # Loop through different weightings and check BPOD computation for each
        for weights, weights_mat in zip(weights_list, weights_mats):
            # Define inner product based on weights
            IP = VectorSpaceMatrices(weights=weights).compute_inner_product_mat

            # Compute adjoint snapshots
            A_adjoint = np.linalg.inv(weights_mat) * self.A.H * weights_mat
            C_adjoint = np.linalg.inv(weights_mat) * self.C.H
            adjoint_vecs_mat = np.mat(np.zeros(
                (self.num_states, self.num_steps * self.num_outputs)))
            A_adjoint_powers = np.mat(np.identity(self.num_states))
            for idx in xrange(self.num_steps):
                adjoint_vecs_mat[
                    :, (idx * self.num_outputs):(idx + 1) * self.num_outputs] =\
                    A_adjoint_powers * C_adjoint
                A_adjoint_powers = A_adjoint_powers * A_adjoint

            # Compute BPOD using modred.  Use tolerance to avoid spurious Hankel
            # singular values, which cause the tests to fail.
            (direct_modes_mat, adjoint_modes_mat, sing_vals,
            L_sing_vecs, R_sing_vecs, Hankel_mat) =  compute_BPOD_matrices(
                self.direct_vecs_mat, adjoint_vecs_mat,
                num_inputs=self.num_inputs, num_outputs=self.num_outputs,
                inner_product_weights=weights, rtol=1e-10, atol=1e-12,
                return_all=True)

            # Check Hankel mat values.  These are computed fast internally by
            # only computing the first column and last row of chunks.  Here,
            # simply take all the inner products.
            Hankel_mat_slow = IP(adjoint_vecs_mat, self.direct_vecs_mat)
            np.testing.assert_allclose(
                Hankel_mat,
                Hankel_mat_slow,
                rtol=rtol, atol=atol)

            # Check properties of SVD of Hankel matrix: singular vectors should
            # be orthogonal, should be able to reconstruct Hankel matrix.
            np.testing.assert_allclose(
                L_sing_vecs.T * L_sing_vecs, np.identity(sing_vals.size),
                rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                R_sing_vecs.T * R_sing_vecs, np.identity(sing_vals.size),
                rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                L_sing_vecs * np.mat(np.diag(sing_vals)) * R_sing_vecs.T,
                Hankel_mat,
                rtol=rtol, atol=atol)

            # Check that the modes diagonalize the gramians.  This test requires
            # looser tolerances than the other tests, likely due to the
            # "squaring" of the matrices in computing the gramians.
            np.testing.assert_allclose((
                IP(adjoint_modes_mat, self.direct_vecs_mat) *
                IP(self.direct_vecs_mat, adjoint_modes_mat)),
                np.diag(sing_vals),
                rtol=rtol_grams, atol=atol_grams)
            np.testing.assert_allclose((
                IP(direct_modes_mat, adjoint_vecs_mat) *
                IP(adjoint_vecs_mat, direct_modes_mat)),
                np.diag(sing_vals),
                rtol=rtol_grams, atol=atol_grams)

            # For debugging.  Check each mode individually and print some
            # information if the test fails.
            '''
            for mode_idx in xrange(sing_vals.size):
                try:

                    np.testing.assert_allclose(
                        self.direct_vecs_mat *
                        IP(self.direct_vecs_mat,
                           adjoint_modes_mat[:, mode_idx]),
                        sing_vals[mode_idx] * direct_modes_mat[:, mode_idx],
                        rtol=rtol_grams, atol=atol_grams)
                    np.testing.assert_allclose(
                        adjoint_vecs_mat *
                        IP(adjoint_vecs_mat, direct_modes_mat[:, mode_idx]),
                        sing_vals[mode_idx] * adjoint_modes_mat[:, mode_idx],
                        rtol=rtol_grams, atol=atol_grams)
                except:
                    dir_LHS = self.direct_vecs_mat * IP(
                        self.direct_vecs_mat, adjoint_modes_mat[:, mode_idx])
                    dir_RHS = (
                        sing_vals[mode_idx] * direct_modes_mat[:, mode_idx])
                    dir_diff = dir_LHS - dir_RHS
                    adj_LHS = adjoint_vecs_mat * IP(
                        adjoint_vecs_mat, direct_modes_mat[:, mode_idx])
                    adj_RHS = (
                        sing_vals[mode_idx] * adjoint_modes_mat[:, mode_idx])
                    adj_diff = adj_LHS - adj_RHS
                    print '\n\nFailed modes test at mode idx %d' % mode_idx
                    print 'Weights', weights
                    print 'HSVs', sing_vals
                    print (
                        '    Curr HSV val %0.3E,    ratio %0.3E' % (
                            sing_vals[mode_idx],
                            sing_vals[mode_idx] / sing_vals.max()))
                    print (
                        '    HSV max %0.3E,    min %.3E,    ratio %0.3E' % (
                            sing_vals.max(),
                            sing_vals.min(),
                            sing_vals.min() / sing_vals.max()))
                    print (
                        '    Direct error abs %0.3E,    rel %0.3E' % (
                            np.abs(dir_diff).max(),
                            np.abs(dir_diff / dir_RHS).max()))
                    print (
                        '    Adjoint error abs %0.3E,    rel %0.3E' % (
                        np.abs(adj_diff).max(),
                        np.abs(adj_diff / dir_RHS).max()))
            '''

            # Check that if mode indices are passed in, the correct modes are
            # returned.
            mode_indices = np.random.randint(
                0, high=sing_vals.size, size=(sing_vals.size // 2))
            (direct_modes_mat_sliced, adjoint_modes_mat_sliced) =\
                compute_BPOD_matrices(
                    self.direct_vecs_mat, adjoint_vecs_mat,
                    direct_mode_indices=mode_indices,
                    adjoint_mode_indices=mode_indices,
                    num_inputs=self.num_inputs, num_outputs=self.num_outputs,
                    inner_product_weights=weights,
                    rtol=1e-12, atol=1e-12, return_all=True)[:2]
            np.testing.assert_allclose(
                direct_modes_mat_sliced, direct_modes_mat[:, mode_indices],
                rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                adjoint_modes_mat_sliced, adjoint_modes_mat[:, mode_indices],
                rtol=rtol, atol=atol)


@unittest.skip('Testing something else.')
class TestBPODHandles(unittest.TestCase):
    """Test the BPOD class methods """
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')

        self.test_dir = 'BPOD_files'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)

        self.mode_nums = [2, 3, 0]
        self.num_direct_vecs = 10
        self.num_adjoint_vecs = 12
        self.num_inputs = 1
        self.num_outputs = 1
        self.num_states = 20

        #A = np.mat(np.random.random((self.num_states, self.num_states)))
        A = np.mat(parallel.call_and_bcast(util.drss, self.num_states,1,1)[0])
        B = np.mat(parallel.call_and_bcast(np.random.random,
            (self.num_states, self.num_inputs)))
        C = np.mat(parallel.call_and_bcast(np.random.random,
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

        if parallel.is_rank_zero():
            for i, handle in enumerate(self.direct_vec_handles):
                handle.put(self.direct_vecs[i])
            for i, handle in enumerate(self.adjoint_vec_handles):
                handle.put(self.adjoint_vecs[i])

        self.Hankel_mat_true = np.dot(self.adjoint_vec_array.T,
            self.direct_vec_array)

        self.L_sing_vecs_true, self.sing_vals_true, self.R_sing_vecs_true = \
            parallel.call_and_bcast(util.svd, self.Hankel_mat_true, atol=1e-10)

        self.direct_mode_array = self.direct_vec_array * \
            np.mat(self.R_sing_vecs_true) * \
            np.mat(np.diag(self.sing_vals_true ** -0.5))
        self.adjoint_mode_array = self.adjoint_vec_array * \
            np.mat(self.L_sing_vecs_true) *\
            np.mat(np.diag(self.sing_vals_true ** -0.5))

        self.my_BPOD = BPODHandles(np.vdot, verbosity=0)
        parallel.barrier()


    def tearDown(self):
        parallel.barrier()
        parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        parallel.barrier()


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
            max_vecs_per_node * parallel.get_num_nodes() / parallel.\
            get_num_procs()
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])


    def test_puts_gets(self):
        """Test that put/get work in base class."""
        test_dir = 'BPOD_files'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(test_dir) and parallel.is_rank_zero():
            os.mkdir(test_dir)
        num_vecs = 10
        num_states = 30
        Hankel_mat_true = parallel.call_and_bcast(
            np.random.random, ((num_vecs, num_vecs)))
        L_sing_vecs_true, sing_vals_true, R_sing_vecs_true = \
            parallel.call_and_bcast(util.svd, Hankel_mat_true)

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
        parallel.barrier()

        BPOD_load = BPODHandles(None, verbosity=0)

        BPOD_load.get_decomp(
            sing_vals_path, L_sing_vecs_path, R_sing_vecs_path)
        Hankel_mat_loaded = parallel.call_and_bcast(
            util.load_array_text, Hankel_mat_path)

        np.testing.assert_allclose(Hankel_mat_loaded, Hankel_mat_true)
        np.testing.assert_allclose(BPOD_load.L_sing_vecs, L_sing_vecs_true)
        np.testing.assert_allclose(BPOD_load.R_sing_vecs, R_sing_vecs_true)
        np.testing.assert_allclose(BPOD_load.sing_vals, sing_vals_true)


    #@unittest.skip('testing others')
    def test_compute_decomp(self):
        """Test that can take vecs, compute the Hankel and SVD matrices. """
        tol = 1e-5
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
        atol = 1e-5
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
                    self.assertAlmostEqual(IP, 0., places=5)
                else:
                    self.assertAlmostEqual(IP, 1., places=5)


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
        rtol = 1e-7
        atol = 1e-8

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
