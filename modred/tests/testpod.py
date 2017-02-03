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


@unittest.skip('Testing something else.')
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


#@unittest.skip('Testing something else.')
class TestPODHandles(unittest.TestCase):
    def setUp(self):
        # Specify output ocations
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'POD_files'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        self.vec_path = join(self.test_dir, 'vec_%03d.txt')
        self.mode_path = join(self.test_dir, 'mode_%03d.txt')

        # Specify data dimensions
        self.num_states = 30
        self.num_vecs = 10

        # Generate random data and write to disk using handles
        self.vec_mat = np.mat(parallel.call_and_bcast(
            np.random.random, (self.num_states, self.num_vecs)))
        self.vec_handles = [
            V.VecHandleArrayText(self.vec_path % i)
            for i in range(self.num_vecs)]
        for idx, hdl in enumerate(self.vec_handles):
            hdl.put(self.vec_mat[:, idx])

        parallel.barrier()


    def tearDown(self):
        parallel.barrier()
        parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        parallel.barrier()


    @unittest.skip('Testing something else.')
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        def my_load(): pass
        def my_save(): pass
        def my_IP(): pass

        data_members_default = {
            'put_mat': util.save_array_text, 'get_mat':util.load_array_text,
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
        data_members_modified['vec_space'].max_vecs_per_node = max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = (
            max_vecs_per_node *
            parallel.get_num_nodes() /
            parallel.get_num_procs())
        for k,v in util.get_data_members(my_POD).items():
            self.assertEqual(v, data_members_modified[k])


    @unittest.skip('Testing something else.')
    def test_puts_gets(self):
        # Generate some random data
        correlation_mat_true = parallel.call_and_bcast(
            np.random.random, ((self.num_vecs, self.num_vecs)))
        eigvals_true = parallel.call_and_bcast(
            np.random.random, self.num_vecs)
        eigvecs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_states, self.num_vecs)))

        # Create a POD object and store the data in it
        POD_save = PODHandles(None, verbosity=0)
        POD_save.correlation_mat = correlation_mat_true
        POD_save.eigvals = eigvals_true
        POD_save.eigvecs = eigvecs_true

        # Write the data to disk
        eigvecs_path = join(self.test_dir, 'eigvecs.txt')
        eigvals_path = join(self.test_dir, 'eigvals.txt')
        correlation_mat_path = join(self.test_dir, 'correlation.txt')
        POD_save.put_decomp(eigvals_path, eigvecs_path)
        POD_save.put_correlation_mat(correlation_mat_path)
        parallel.barrier()

        # Create a new POD object and use it to load the data
        POD_load = PODHandles(None, verbosity=0)
        POD_load.get_correlation_mat(correlation_mat_path)
        POD_load.get_decomp(eigvals_path, eigvecs_path)

        # Check that the loaded data is correct
        np.testing.assert_equal(POD_load.correlation_mat, correlation_mat_true)
        np.testing.assert_equal(POD_load.eigvals, eigvals_true)
        np.testing.assert_equal(POD_load.eigvecs, eigvecs_true)


    @unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test computation of the correlation mat and SVD matrices."""
        rtol = 1e-10
        atol = 1e-12

        # Compute POD using modred
        POD = PODHandles(np.vdot, verbosity=0)
        eigvals, eigvecs = POD.compute_decomp(self.vec_handles)

        # Check POD eigenvectors and eigenvalues
        np.testing.assert_allclose(
            self.vec_mat.T * self.vec_mat * eigvecs,
            eigvecs * np.mat(np.diag(eigvals)), rtol=rtol, atol=atol)

        # Check that returned values match internal values
        np.testing.assert_equal(eigvals, POD.eigvals)


    #@unittest.skip('Testing something else.')
    def test_compute_modes(self):
        rtol = 1e-10
        atol = 1e-12

        # Compute POD using modred.  (The properties defining a POD mode require
        # manipulations involving the correct decomposition, so we cannot
        # isolate the mode computation from the decomposition step.)
        POD = PODHandles(np.vdot, verbosity=0)
        POD.compute_decomp(self.vec_handles)

        # Select a subset of modes to compute.  Compute at least half
        # the modes, and up to all of them.  Make sure to use unique
        # values.  (This may reduce the number of modes computed.)
        num_modes = parallel.call_and_bcast(
            np.random.randint,
            POD.eigvals.size // 2, POD.eigvals.size + 1)
        mode_idxs = np.unique(parallel.call_and_bcast(
            np.random.randint,
            0, POD.eigvals.size, num_modes))

        # Create handles for the modes
        mode_handles = [
            V.VecHandleArrayText(self.mode_path % i) for i in mode_idxs]

        # Compute modes
        POD.compute_modes(mode_idxs, mode_handles, vec_handles=self.vec_handles)

        # Test modes
        np.testing.assert_allclose(
            POD.vec_space.compute_inner_product_mat(
                mode_handles, self.vec_handles) *
            POD.vec_space.compute_inner_product_mat(
                self.vec_handles, mode_handles),
            np.diag(POD.eigvals[mode_idxs]),
            rtol=rtol, atol=atol)


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
