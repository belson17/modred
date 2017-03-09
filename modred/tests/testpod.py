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

        # Test both method of snapshots and direct method
        for method in ['snaps', 'direct']:

            # Loop through different inner product weights
            for weights in weights_list:
                IP = VectorSpaceMatrices(
                    weights=weights).compute_inner_product_mat

                # Choose a random subset of modes to compute, for testing mode
                # indices argument
                mode_indices = np.unique(np.random.randint(
                    0, high=np.linalg.matrix_rank(vec_mat),
                    size=np.linalg.matrix_rank(vec_mat) // 2))

                # Compute POD using appropriate method.  Also compute a subset
                # of modes, for later testing mode indices argument.
                if method == 'snaps':
                    modes, eigvals, eigvecs, correlation_mat =\
                    compute_POD_matrices_snaps_method(
                        vec_mat, inner_product_weights=weights, return_all=True)
                    modes_sliced = compute_POD_matrices_snaps_method(
                        vec_mat, mode_indices=mode_indices,
                        inner_product_weights=weights, return_all=True)[0]

                    # For method of snapshots, test correlation mat values
                    np.testing.assert_allclose(
                        IP(vec_mat, vec_mat), correlation_mat,
                        rtol=rtol, atol=atol)

                elif method == 'direct':
                    modes, eigvals, eigvecs =\
                    compute_POD_matrices_direct_method(
                        vec_mat, inner_product_weights=weights, return_all=True)
                    modes_sliced = compute_POD_matrices_direct_method(
                        vec_mat, mode_indices=mode_indices,
                        inner_product_weights=weights, return_all=True)[0]

                else:
                    raise ValueError('Invalid POD method.')

                # Check POD eigenvalues and eigenvectors
                np.testing.assert_allclose(
                    IP(vec_mat, vec_mat) * eigvecs,
                    eigvecs * np.mat(np.diag(eigvals)),
                    rtol=rtol, atol=atol)

                # Check POD modes
                np.testing.assert_allclose(
                    vec_mat * IP(vec_mat, modes),
                    modes * np.mat(np.diag(eigvals)),
                    rtol=rtol, atol=atol)

                # Check that if mode indices are passed in, the correct
                # modes are returned.
                np.testing.assert_allclose(
                    modes_sliced, modes[:, mode_indices],
                    rtol=rtol, atol=atol)


#@unittest.skip('Testing something else.')
class TestPODHandles(unittest.TestCase):
    def setUp(self):
        # Specify output locations
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


    #@unittest.skip('Testing something else.')
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


    #@unittest.skip('Testing something else.')
    def test_puts_gets(self):
        # Generate some random data
        correlation_mat_true = parallel.call_and_bcast(
            np.random.random, ((self.num_vecs, self.num_vecs)))
        eigvals_true = parallel.call_and_bcast(
            np.random.random, self.num_vecs)
        eigvecs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_states, self.num_vecs)))
        proj_coeffs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_vecs, self.num_vecs)))

        # Create a POD object and store the data in it
        POD_save = PODHandles(None, verbosity=0)
        POD_save.correlation_mat = correlation_mat_true
        POD_save.eigvals = eigvals_true
        POD_save.eigvecs = eigvecs_true
        POD_save.proj_coeffs = proj_coeffs_true

        # Write the data to disk
        eigvecs_path = join(self.test_dir, 'eigvecs.txt')
        eigvals_path = join(self.test_dir, 'eigvals.txt')
        correlation_mat_path = join(self.test_dir, 'correlation.txt')
        proj_coeffs_path = join(self.test_dir, 'proj_coeffs.txt')
        POD_save.put_decomp(eigvals_path, eigvecs_path)
        POD_save.put_correlation_mat(correlation_mat_path)
        POD_save.put_proj_coeffs(proj_coeffs_path)
        parallel.barrier()

        # Create a new POD object and use it to load the data
        POD_load = PODHandles(None, verbosity=0)
        POD_load.get_decomp(eigvals_path, eigvecs_path)
        POD_load.get_correlation_mat(correlation_mat_path)
        POD_load.get_proj_coeffs(proj_coeffs_path)

        # Check that the loaded data is correct
        np.testing.assert_equal(POD_load.eigvals, eigvals_true)
        np.testing.assert_equal(POD_load.eigvecs, eigvecs_true)
        np.testing.assert_equal(POD_load.correlation_mat, correlation_mat_true)
        np.testing.assert_equal(POD_load.proj_coeffs, proj_coeffs_true)

    #@unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test computation of the correlation mat and SVD matrices."""
        rtol = 1e-10
        atol = 1e-12

        # Compute POD using modred
        POD = PODHandles(np.vdot, verbosity=0)
        eigvals, eigvecs = POD.compute_decomp(self.vec_handles)

        # Test correlation mats values by simply recomputing them.  Here simply
        # take all inner products, rather than assuming a symmetric inner
        # product.
        np.testing.assert_allclose(
            POD.correlation_mat,
            POD.vec_space.compute_inner_product_mat(
                self.vec_handles, self.vec_handles),
            rtol=rtol, atol=atol)

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


    #@unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        rtol = 1e-10
        atol = 1e-12

        # Compute POD using modred.  (The properties defining a projection onto
        # POD modes require manipulations involving the correct decomposition
        # and modes, so we cannot isolate the mode computation from those
        # computations.)
        POD = PODHandles(np.vdot, verbosity=0)
        POD.compute_decomp(self.vec_handles)
        mode_idxs = range(POD.eigvals.size)
        mode_handles = [
            V.VecHandleArrayText(self.mode_path % i) for i in mode_idxs]
        POD.compute_modes(mode_idxs, mode_handles, vec_handles=self.vec_handles)

        # Compute true projection coefficients by computing the inner products
        # between modes and snapshots.
        proj_coeffs_true = POD.vec_space.compute_inner_product_mat(
            mode_handles, self.vec_handles)

        # Compute projection coefficients using POD object, which avoids
        # actually manipulating handles and computing their inner products,
        # instead using elements of the decomposition for a more efficient
        # computations.
        proj_coeffs = POD.compute_proj_coeffs()

        # Test values
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
