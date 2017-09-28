#!/usr/bin/env python
"""Test POD module"""
import unittest
import os
from os.path import join
from shutil import rmtree
import copy

import numpy as np

import modred.parallel as parallel
from modred.pod import *
from modred.vectorspace import *
from modred.vectors import VecHandlePickle
from modred import util
from modred.py2to3 import range


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
        weights_1D = np.random.random(self.num_states)
        weights_2D = np.identity(self.num_states, dtype=np.complex)
        weights_2D[0, 0] = 2.
        weights_2D[2, 1] = 0.3j
        weights_2D[1, 2] = weights_2D[2, 1].conj()

        # Generate random snapshot data
        vecs_array = (
            np.random.random((self.num_states, self.num_vecs)) +
            1j * np.random.random((self.num_states, self.num_vecs)))

        # Test both method of snapshots and direct method
        for method in ['snaps', 'direct']:

            # Loop through different inner product weights
            for weights in [None, weights_1D, weights_2D]:
                IP = VectorSpaceArrays(
                    weights=weights).compute_inner_product_array

                # Choose a random subset of modes to compute, for testing mode
                # indices argument
                mode_indices = np.unique(np.random.randint(
                    0, high=np.linalg.matrix_rank(vecs_array),
                    size=np.linalg.matrix_rank(vecs_array) // 2))

                # Compute POD using appropriate method.  Also compute a subset
                # of modes, for later testing mode indices argument.
                if method == 'snaps':
                    POD_res = compute_POD_arrays_snaps_method(
                        vecs_array, inner_product_weights=weights)
                    POD_res_sliced = compute_POD_arrays_snaps_method(
                        vecs_array, mode_indices=mode_indices,
                        inner_product_weights=weights)

                    # For method of snapshots, test correlation array values
                    np.testing.assert_allclose(
                        IP(vecs_array, vecs_array), POD_res.correlation_array,
                        rtol=rtol, atol=atol)

                elif method == 'direct':
                    POD_res = compute_POD_arrays_direct_method(
                        vecs_array, inner_product_weights=weights)
                    POD_res_sliced = compute_POD_arrays_direct_method(
                        vecs_array, mode_indices=mode_indices,
                        inner_product_weights=weights)

                else:
                    raise ValueError('Invalid POD method.')

                # Check POD eigenvalues and eigenvectors
                np.testing.assert_allclose(
                    IP(vecs_array, vecs_array).dot(POD_res.eigvecs),
                    POD_res.eigvecs.dot(np.diag(POD_res.eigvals)),
                    rtol=rtol, atol=atol)

                # Check POD modes
                np.testing.assert_allclose(
                    vecs_array.dot(IP(vecs_array, POD_res.modes)),
                    POD_res.modes.dot(np.diag(POD_res.eigvals)),
                    rtol=rtol, atol=atol)

                # Check projection coefficients
                np.testing.assert_allclose(
                    POD_res.proj_coeffs, IP(POD_res.modes, vecs_array),
                    rtol=rtol, atol=atol)

                # Check that if mode indices are passed in, the correct
                # modes are returned.
                np.testing.assert_allclose(
                    POD_res_sliced.modes, POD_res.modes[:, mode_indices],
                    rtol=rtol, atol=atol)


#@unittest.skip('Testing something else.')
class TestPODHandles(unittest.TestCase):
    def setUp(self):
        # Specify output locations
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'files_POD_DELETE_ME'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        self.vec_path = join(self.test_dir, 'vec_%03d.pkl')
        self.mode_path = join(self.test_dir, 'mode_%03d.pkl')

        # Specify data dimensions
        self.num_states = 30
        self.num_vecs = 10

        # Generate random data and write to disk using handles
        self.vecs_array = (
            parallel.call_and_bcast(
                np.random.random, (self.num_states, self.num_vecs)) +
            1j * parallel.call_and_bcast(
                np.random.random, (self.num_states, self.num_vecs)))
        self.vec_handles = [
            VecHandlePickle(self.vec_path % i) for i in range(self.num_vecs)]
        for idx, hdl in enumerate(self.vec_handles):
            hdl.put(self.vecs_array[:, idx])

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
            'put_array': util.save_array_text, 'get_array':util.load_array_text,
            'verbosity': 0, 'eigvecs': None, 'eigvals': None,
            'correlation_array': None, 'vec_handles': None, 'vecs': None,
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

        my_POD = PODHandles(my_IP, get_array=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_array'] = my_load
        for k,v in util.get_data_members(my_POD).items():
            self.assertEqual(v, data_members_modified[k])

        my_POD = PODHandles(my_IP, put_array=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_array'] = my_save
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
        correlation_array_true = parallel.call_and_bcast(
            np.random.random, ((self.num_vecs, self.num_vecs)))
        eigvals_true = parallel.call_and_bcast(
            np.random.random, self.num_vecs)
        eigvecs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_states, self.num_vecs)))
        proj_coeffs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_vecs, self.num_vecs)))

        # Create a POD object and store the data in it
        POD_save = PODHandles(None, verbosity=0)
        POD_save.correlation_array = correlation_array_true
        POD_save.eigvals = eigvals_true
        POD_save.eigvecs = eigvecs_true
        POD_save.proj_coeffs = proj_coeffs_true

        # Write the data to disk
        eigvecs_path = join(self.test_dir, 'eigvecs.txt')
        eigvals_path = join(self.test_dir, 'eigvals.txt')
        correlation_array_path = join(self.test_dir, 'correlation.txt')
        proj_coeffs_path = join(self.test_dir, 'proj_coeffs.txt')
        POD_save.put_decomp(eigvals_path, eigvecs_path)
        POD_save.put_correlation_array(correlation_array_path)
        POD_save.put_proj_coeffs(proj_coeffs_path)
        parallel.barrier()

        # Create a new POD object and use it to load the data
        POD_load = PODHandles(None, verbosity=0)
        POD_load.get_decomp(eigvals_path, eigvecs_path)
        POD_load.get_correlation_array(correlation_array_path)
        POD_load.get_proj_coeffs(proj_coeffs_path)

        # Check that the loaded data is correct
        np.testing.assert_equal(POD_load.eigvals, eigvals_true)
        np.testing.assert_equal(POD_load.eigvecs, eigvecs_true)
        np.testing.assert_equal(
            POD_load.correlation_array, correlation_array_true)
        np.testing.assert_equal(POD_load.proj_coeffs, proj_coeffs_true)


    #@unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test computation of the correlation array and SVD arrays."""
        rtol = 1e-10
        atol = 1e-12

        # Compute POD using modred
        POD = PODHandles(np.vdot, verbosity=0)
        eigvals, eigvecs = POD.compute_decomp(self.vec_handles)

        # Test correlation array values by simply recomputing them.  Here simply
        # take all inner products, rather than assuming a symmetric inner
        # product.
        np.testing.assert_allclose(
            POD.correlation_array,
            POD.vec_space.compute_inner_product_array(
                self.vec_handles, self.vec_handles),
            rtol=rtol, atol=atol)

        # Check POD eigenvectors and eigenvalues
        np.testing.assert_allclose(
            self.vecs_array.conj().T.dot(self.vecs_array.dot(eigvecs)),
            eigvecs.dot(np.diag(eigvals)), rtol=rtol, atol=atol)

        # Check that returned values match internal values
        np.testing.assert_equal(eigvals, POD.eigvals)
        np.testing.assert_equal(eigvecs, POD.eigvecs)


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
        mode_handles = [VecHandlePickle(self.mode_path % i) for i in mode_idxs]

        # Compute modes
        POD.compute_modes(mode_idxs, mode_handles, vec_handles=self.vec_handles)

        # Test modes
        np.testing.assert_allclose(
            POD.vec_space.compute_inner_product_array(
                mode_handles, self.vec_handles).dot(
                    POD.vec_space.compute_inner_product_array(
                        self.vec_handles, mode_handles)),
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
        mode_handles = [VecHandlePickle(self.mode_path % i) for i in mode_idxs]
        POD.compute_modes(mode_idxs, mode_handles, vec_handles=self.vec_handles)

        # Compute true projection coefficients by computing the inner products
        # between modes and snapshots.
        proj_coeffs_true = POD.vec_space.compute_inner_product_array(
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
