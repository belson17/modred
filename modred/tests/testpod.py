#!/usr/bin/env python
"""Test POD module"""
from __future__ import division
from future.builtins import range
import unittest
import os
from os.path import join
from shutil import rmtree
import copy

import numpy as np

import modred.parallel as parallel
from modred.pod import *
from modred.vectorspace import *
import modred.vectors as V
from modred import util


#@unittest.skip('Testing something else.')
@unittest.skipIf(parallel.is_distributed(), 'Serial only.')
class TestPODArraysFunctions(unittest.TestCase):
    def setUp(self):
        self.num_states = 30
        self.num_vecs = 10


    def test_compute_modes(self):
        rtol = 1e-10
        atol = 1e-12

        # Generate weights to test different inner products.
        ws = np.identity(self.num_states)
        ws[0, 0] = 2.
        ws[2, 1] = 0.3
        ws[1, 2] = 0.3
        weights_list = [None, np.random.random(self.num_states), ws]

        # Generate random snapshot data
        vec_mat = np.mat(np.random.random((self.num_states, self.num_vecs)))

        # Loop through different inner product weights
        for weights in weights_list:
            IP = VectorSpaceMatrices(weights=weights).compute_inner_product_mat

            # Compute POD using method of snapshots
            modes_snaps, eigvals_snaps, eigvecs_snaps, correlation_mat_snaps =\
            compute_POD_matrices_snaps_method(
                vec_mat, inner_product_weights=weights, return_all=True)

            # Test correlation mat values
            np.testing.assert_allclose(
                IP(vec_mat, vec_mat), correlation_mat_snaps,
                rtol=rtol, atol=atol)

            # Check POD eigenvalues and eigenvectors
            np.testing.assert_allclose(
                correlation_mat_snaps * eigvecs_snaps,
                eigvecs_snaps * np.mat(np.diag(eigvals_snaps)),
                rtol=rtol, atol=atol)

            # Check POD modes
            np.testing.assert_allclose(
                vec_mat * IP(vec_mat, modes_snaps),
                modes_snaps * np.mat(np.diag(eigvals_snaps)),
                rtol=rtol, atol=atol)

            # Check that if mode indices are passed in, the correct
            # modes are returned.
            mode_indices_snaps = np.unique(np.random.randint(
                0, high=eigvals_snaps.size, size=(eigvals_snaps.size // 2)))
            modes_sliced_snaps = compute_POD_matrices_snaps_method(
                vec_mat, mode_indices=mode_indices_snaps,
                inner_product_weights=weights, return_all=True)[0]
            np.testing.assert_allclose(
                modes_sliced_snaps, modes_snaps[:, mode_indices_snaps],
                rtol=rtol, atol=atol)

            # Compute POD using direct method
            modes_direct, eigvals_direct, eigvecs_direct =\
            compute_POD_matrices_direct_method(
                vec_mat, inner_product_weights=weights, return_all=True)

            # Check POD eigenvalues and eigenvectors
            np.testing.assert_allclose(
                IP(vec_mat, vec_mat) * eigvecs_direct,
                eigvecs_direct * np.mat(np.diag(eigvals_direct)),
                rtol=rtol, atol=atol)

            # Check POD modes
            np.testing.assert_allclose(
                vec_mat * IP(vec_mat, modes_direct),
                modes_direct * np.mat(np.diag(eigvals_direct)),
                rtol=rtol, atol=atol)

            # Check that if mode indices are passed in, the correct
            # modes are returned.
            mode_indices_direct = np.unique(np.random.randint(
                0, high=eigvals_direct.size, size=(eigvals_direct.size // 2)))
            modes_sliced_direct = compute_POD_matrices_direct_method(
                vec_mat, mode_indices=mode_indices_direct,
                inner_product_weights=weights, return_all=True)[0]
            np.testing.assert_allclose(
                modes_sliced_direct, modes_direct[:, mode_indices_direct],
                rtol=rtol, atol=atol)


@unittest.skip('Testing something else.')
class TestPODHandles(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'POD_files'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():
            os.mkdir(self.test_dir)
        self.mode_indices = [2, 4, 3, 6]
        self.num_vecs = 10
        self.num_states = 30
        self.vec_array = parallel.call_and_bcast(np.random.random,
            (self.num_states, self.num_vecs))
        self.correlation_mat_true = self.vec_array.conj().transpose().dot(
            self.vec_array)

        self.eigvals_true, self.eigvecs_true = \
            parallel.call_and_bcast(util.eigh, self.correlation_mat_true)

        self.mode_array = np.dot(self.vec_array, np.dot(self.eigvecs_true,
            np.diag(self.eigvals_true**-0.5)))
        self.vec_path = join(self.test_dir, 'vec_%03d.txt')
        self.vec_handles = [V.VecHandleArrayText(self.vec_path%i)
            for i in range(self.num_vecs)]
        for vec_index, handle in enumerate(self.vec_handles):
            handle.put(self.vec_array[:, vec_index])

        self.my_POD = PODHandles(np.vdot, verbosity=0)
        parallel.barrier()


    def tearDown(self):
        parallel.barrier()
        parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        parallel.barrier()


    @unittest.skip('Testing something else.')
    def test_puts_gets(self):
        test_dir = 'DELETE_ME_test_files_pod'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(test_dir):
            parallel.call_from_rank_zero(os.mkdir, test_dir)
        num_vecs = 10
        num_states = 30
        correlation_mat_true = parallel.call_and_bcast(
            np.random.random, ((num_vecs, num_vecs)))
        eigvals_true = parallel.call_and_bcast(
            np.random.random, num_vecs)
        eigvecs_true = parallel.call_and_bcast(
            np.random.random, ((num_states, num_vecs)))

        my_POD = PODHandles(None, verbosity=0)
        my_POD.correlation_mat = correlation_mat_true
        my_POD.eigvals = eigvals_true
        my_POD.eigvecs = eigvecs_true

        eigvecs_path = join(test_dir, 'eigvecs.txt')
        eigvals_path = join(test_dir, 'eigvals.txt')
        correlation_mat_path = join(test_dir, 'correlation.txt')
        my_POD.put_decomp(eigvals_path, eigvecs_path)
        my_POD.put_correlation_mat(correlation_mat_path)
        parallel.barrier()

        POD_load = PODHandles(None, verbosity=0)
        POD_load.get_decomp(eigvals_path, eigvecs_path)
        correlation_mat_loaded = util.load_array_text(correlation_mat_path)

        np.testing.assert_allclose(correlation_mat_loaded,
            correlation_mat_true)
        np.testing.assert_allclose(POD_load.eigvals, eigvals_true)
        np.testing.assert_allclose(POD_load.eigvecs, eigvecs_true)


    @unittest.skip('Testing something else.')
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        def my_load(): pass
        def my_save(): pass
        def my_IP(): pass

        data_members_default = {'put_mat': util.save_array_text, 'get_mat':
            util.load_array_text,
            'verbosity': 0, 'eigvecs': None, 'eigvals': None,
            'correlation_mat': None, 'vec_handles': None, 'vecs': None,
            'vec_space': VectorSpaceHandles(inner_product=my_IP, verbosity=0)}
        for k,v in util.get_data_members(
            PODHandles(my_IP, verbosity=0)).items():
            self.assertEqual(v, data_members_default[k])

        my_POD = PODHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_POD).items():
            self.assertEqual(v, data_members_modified[k])

        my_POD = PODHandles(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_POD).items():
            self.assertEqual(v, data_members_modified[k])

        my_POD = PODHandles(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_POD).items():
            self.assertEqual(v, data_members_modified[k])

        max_vecs_per_node = 500
        my_POD = PODHandles(
            my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * parallel.get_num_nodes() / \
            parallel.get_num_procs()
        for k,v in util.get_data_members(my_POD).items():
            self.assertEqual(v, data_members_modified[k])


    @unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test computation of the correlation mat and SVD matrices."""
        tol = 1e-6
        eigvals_returned, eigvecs_returned = \
            self.my_POD.compute_decomp(self.vec_handles)

        np.testing.assert_allclose(self.my_POD.correlation_mat,
            self.correlation_mat_true, rtol=tol)
        np.testing.assert_allclose(self.my_POD.eigvecs,
            self.eigvecs_true, rtol=tol)
        np.testing.assert_allclose(self.my_POD.eigvals,
            self.eigvals_true, rtol=tol)

        np.testing.assert_allclose(eigvecs_returned,
            self.eigvecs_true, rtol=tol)
        np.testing.assert_allclose(eigvals_returned,
            self.eigvals_true, rtol=tol)


    @unittest.skip('Testing something else.')
    def test_compute_modes(self):
        mode_path = join(self.test_dir, 'mode_%03d.txt')
        mode_handles = [V.VecHandleArrayText(mode_path%i)
            for i in self.mode_indices]
        # starts with the CORRECT decomposition.
        self.my_POD.eigvecs = self.eigvecs_true
        self.my_POD.eigvals = self.eigvals_true

        self.my_POD.compute_modes(self.mode_indices, mode_handles,
            vec_handles=self.vec_handles)

        for mode_index, mode_handle in enumerate(mode_handles):
            mode = mode_handle.get()
            np.testing.assert_allclose(
                mode, self.mode_array[:,self.mode_indices[mode_index]])

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


    @unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        rtol = 1e-10
        atol = 1e-13

        # Compute true projection coefficients by simply projecting directly
        # onto the modes.
        proj_coeffs_true = np.dot(self.mode_array.conj().T, self.vec_array)

        # Initialize the POD object with the known correct decomposition
        # matrices, to avoid errors in computing those matrices.
        self.my_POD.eigvecs = self.eigvecs_true
        self.my_POD.eigvals = self.eigvals_true

        # Compute projection coefficients
        proj_coeffs = self.my_POD.compute_proj_coeffs()

        # Test values
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
