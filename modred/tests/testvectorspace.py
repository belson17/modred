#!/usr/bin/env python
"""Test vectorspace module"""
from __future__ import division
from future.builtins import range
import os
from os.path import join
from shutil import rmtree
import copy
import random
import unittest

import numpy as np

import modred.parallel as parallel
from modred.vectorspace import *
import modred.vectors as vcs
import modred.util


#@unittest.skip('Testing other things')
@unittest.skipIf(parallel.is_distributed(), 'Serial only')
class TestVectorSpaceArrays(unittest.TestCase):
    """ Tests of the VectorSpaceArrays class """
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_equals(self):
        """ Test equality operator """
        vec_space1 = VectorSpaceArrays(weights=np.array([1., 1., 1.]))
        vec_space2 = VectorSpaceArrays(weights=[1., 1., 1.])
        vec_space3 = VectorSpaceArrays(weights=[2., 4., 6.])
        vec_space4 = VectorSpaceArrays(weights=np.diag(np.ones(3) * 2.))
        self.assertEquals(vec_space1, vec_space2)
        self.assertNotEqual(vec_space1, vec_space3)
        self.assertNotEqual(vec_space1, vec_space4)


    def test_lin_combine(self):
        """ Test that linear combinations are correctly computed """
        # Set test tolerances
        rtol = 1e-10
        atol = 1e-12

        # Generate data
        num_states = 100
        num_vecs = 30
        num_modes = 10
        vecs_array = (
            np.random.random((num_states, num_vecs)) +
            1j * np.random.random((num_states, num_vecs)))
        coeffs_array = (
            np.random.random((num_vecs, num_modes)) +
            1j * np.random.random((num_vecs, num_modes)))
        modes_array_true = vecs_array.dot(coeffs_array)

        # Do computation using a vector space object
        mode_indices = np.random.randint(0, high=num_modes, size=num_modes // 2)
        vec_space = VectorSpaceArrays()
        modes_array = vec_space.lin_combine(
            vecs_array, coeffs_array, coeff_array_col_indices=mode_indices)
        np.testing.assert_allclose(
            modes_array, modes_array_true[:, mode_indices],
            rtol=rtol, atol=atol)


    def test_inner_product_arrays(self):
        """ Test that inner product arrays are correctly computed """
        # Set test tolerances
        rtol = 1e-10
        atol = 1e-12

        # Generate data
        num_states = 10
        num_rows = 2
        num_cols = 3
        row_array = (
            np.random.random((num_states, num_rows)) +
            1j * np.random.random((num_states, num_rows)))
        col_array = (
            np.random.random((num_states, num_cols)) +
            1j * np.random.random((num_states, num_cols)))

        # Test different weights
        weights_1d = np.random.random(num_states)
        weights_2d = np.random.random((num_states, num_states))
        weights_2d = 0.5 * (weights_2d + weights_2d.T)
        for weights in [None, weights_1d, weights_2d]:
            # Reshape weights
            if weights is None:
                weights_array = np.eye(num_states)
            elif weights.ndim == 1:
                weights_array = np.diag(weights)
            else:
                weights_array = weights

            # Correct inner product values
            ip_array_true = row_array.conj().T.dot(weights_array.dot(col_array))
            ip_array_symm_true = row_array.conj().T.dot(
                weights_array.dot(row_array))

            # Compute inner products using vec space object
            vec_space = VectorSpaceArrays(weights=weights)
            ip_array = vec_space.compute_inner_product_array(
                row_array, col_array)
            ip_array_symm = vec_space.compute_symm_inner_product_array(
                row_array)
            np.testing.assert_allclose(
                ip_array, ip_array_true, rtol=rtol, atol=atol)


#@unittest.skip('Testing other things')
class TestVectorSpaceHandles(unittest.TestCase):
    """ Tests of the VectorSpaceHandles class """
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'files_vectorspace_DELETE_ME'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)

        self.max_vecs_per_proc = 10
        self.total_num_vecs_in_mem = (
            parallel.get_num_procs() * self.max_vecs_per_proc)

        self.vec_space = VectorSpaceHandles(inner_product=np.vdot, verbosity=0)
        self.vec_space.max_vecs_per_proc = self.max_vecs_per_proc

        # Default data members; set verbosity to 0 even though default is 1
        # so messages won't print during tests
        self.default_data_members = {
            'inner_product': np.vdot, 'max_vecs_per_node': 10000,
            'max_vecs_per_proc': (
                10000 * parallel.get_num_nodes() // parallel.get_num_procs()),
            'verbosity': 0, 'print_interval': 10, 'prev_print_time': 0.}
        parallel.barrier()


    def tearDown(self):
        parallel.barrier()
        parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        parallel.barrier()


    #@unittest.skip('Testing other things')
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly."""
        data_members_original = util.get_data_members(
            VectorSpaceHandles(inner_product=np.vdot, verbosity=0))
        self.assertEqual(data_members_original, self.default_data_members)

        max_vecs_per_node = 500
        vec_space = VectorSpaceHandles(
            inner_product=np.vdot, max_vecs_per_node=max_vecs_per_node,
            verbosity=0)
        data_members = copy.deepcopy(data_members_original)
        data_members['max_vecs_per_node'] = max_vecs_per_node
        data_members['max_vecs_per_proc'] = (
            max_vecs_per_node *
            parallel.get_num_nodes() // parallel.get_num_procs())
        self.assertEqual(util.get_data_members(vec_space), data_members)


    #@unittest.skip('Testing other things')
    def test_sanity_check(self):
        """Tests correctly checks user-supplied objects and functions."""
        # Setup
        nx = 40
        ny = 15
        test_array = np.random.random((nx, ny))
        vec_space = VectorSpaceHandles(inner_product=np.vdot, verbosity=0)
        in_mem_handle = vcs.VecHandleInMemory(test_array)
        vec_space.sanity_check(in_mem_handle)

        # Define some weird vectors that alter their internal data when adding
        # or multiplying (which they shouldn't do).
        class SanityMultVec(vcs.Vector):
            def __init__(self, arr):
                self.arr = arr

            def __add__(self, obj):
                f_return = copy.deepcopy(self)
                f_return.arr += obj.arr
                return f_return

            def __mul__(self, a):
                self.arr *= a
                return self

        class SanityAddVec(vcs.Vector):
            def __init__(self, arr):
                self.arr = arr

            def __add__(self, obj):
                self.arr += obj.arr
                return self

            def __mul__(self, a):
                f_return = copy.deepcopy(self)
                f_return.arr *= a
                return f_return

        # Define an inner product function for Sanity vec handles
        def good_custom_ip(vec1, vec2):
            return np.vdot(vec1.arr, vec2.arr)

        # Define a bad inner product for regular arrays
        def bad_array_ip(vec1, vec2):
            return np.vdot(vec1, vec2 ** 2.)

        # Make sure that sanity check passes for vectors with properly defined
        # vector operations.  Do so by simply calling the function.  If it
        # passes, then no error will be raised.
        vec_space.inner_product = np.vdot
        vec_space.sanity_check(vcs.VecHandleInMemory(test_array))

        # Make sure that sanity check fails if inner product values are not
        # correct.
        vec_space.inner_product = bad_array_ip
        self.assertRaises(
            ValueError,
            vec_space.sanity_check, vcs.VecHandleInMemory(test_array))

        # Make sure that sanity check fails if vectors alter their
        # internal data when doing vector space operations.
        vec_space.inner_product = good_custom_ip
        sanity_mult_vec = SanityMultVec(test_array)
        self.assertRaises(
            ValueError,
            vec_space.sanity_check, vcs.VecHandleInMemory(sanity_mult_vec))
        sanity_add_vec = SanityAddVec(test_array)
        self.assertRaises(
            ValueError,
            vec_space.sanity_check, vcs.VecHandleInMemory(sanity_add_vec))


    def generate_vecs_modes(
        self, num_states, num_vecs, num_modes, squeeze=False):
        """Generates random vecs and finds the modes.

        Returns:
            vec_array: array in which each column is a vec (in order)
            mode_indices: unordered list of integers representing mode indices,
                each entry is unique. Mode indices are picked randomly.
            coeff_array: array of shape num_vecs x num_modes, random
                entries
            mode_array: array of modes, each column is a mode.
                The index of the array column equals the mode index.
        """
        mode_indices = list(range(num_modes))
        random.shuffle(mode_indices)
        coeff_array = (
            np.random.random((num_vecs, num_modes)) +
            1j * np.random.random((num_vecs, num_modes)))
        vec_array = (
            np.random.random((num_states, num_vecs)) +
            1j * np.random.random((num_states, num_vecs)))
        mode_array = vec_array.dot(coeff_array)
        if squeeze:
            build_coeff_array = coeff_array.squeeze()
        return vec_array, mode_indices, coeff_array, mode_array


    #@unittest.skip('Testing other things')
    def test_lin_combine(self):
        # Set test tolerances
        rtol = 1e-10
        atol = 1e-12

        # Setup
        mode_path = join(self.test_dir, 'mode_%03d.pkl')
        vec_path = join(self.test_dir, 'vec_%03d.pkl')

        # Test cases where number of modes:
        #   less, equal, more than num_states
        #   less, equal, more than num_vecs
        #   less, equal, more than total_num_vecs_in_mem
        num_states = 20
        num_vecs_list = [1, 15, 40]
        num_modes_list = [
            1, 8, 10, 20, 25, 45,
            int(np.ceil(self.total_num_vecs_in_mem / 2.)),
            self.total_num_vecs_in_mem, self.total_num_vecs_in_mem * 2]

        # Check for correct computations
        for num_vecs in num_vecs_list:
            for num_modes in num_modes_list:
                for squeeze in [True, False]:

                    # Generate data and then broadcast to all procs
                    vec_handles = [
                        vcs.VecHandlePickle(vec_path % i)
                        for i in range(num_vecs)]
                    vec_array, mode_indices, coeff_array, true_modes =\
                        parallel.call_and_bcast(
                            self.generate_vecs_modes, num_states, num_vecs,
                            num_modes, squeeze=squeeze)
                    if parallel.is_rank_zero():
                        for vec_index, vec_handle in enumerate(vec_handles):
                            vec_handle.put(vec_array[:, vec_index])
                    parallel.barrier()
                    mode_handles = [
                        vcs.VecHandlePickle(mode_path % mode_num)
                        for mode_num in mode_indices]

                    # Test the case that only one mode is desired,
                    # in which case user might pass in an int
                    if len(mode_indices) == 1:
                        mode_indices = mode_indices[0]
                        mode_handles = mode_handles[0]

                    # Saves modes to files
                    self.vec_space.lin_combine(
                        mode_handles, vec_handles, coeff_array,
                        mode_indices)

                    # Change mode indices back to list to make iterable for
                    # testing modes one by one.
                    if not isinstance(mode_indices, list):
                        mode_indices = [mode_indices]
                    parallel.barrier()

                    for mode_index in mode_indices:
                        computed_mode = vcs.VecHandlePickle(
                            mode_path % mode_index).get()
                        np.testing.assert_allclose(
                            computed_mode, true_modes[:, mode_index],
                            rtol=rtol, atol=atol)
                    parallel.barrier()

                parallel.barrier()

            parallel.barrier()

        # Test that errors are caught for mismatched dimensions
        mode_handles = [
            vcs.VecHandlePickle(mode_path % i) for i in range(10)]
        vec_handles = [
            vcs.VecHandlePickle(vec_path % i) for i in range(15)]
        coeffs_array_too_short = np.zeros(
            (len(vec_handles) - 1, len(mode_handles)))
        coeffs_array_too_fat = np.zeros(
            (len(vec_handles), len(mode_handles) + 1))
        index_list_too_long = range(len(mode_handles) + 1)
        self.assertRaises(
            ValueError, self.vec_space.lin_combine, mode_handles, vec_handles,
            coeffs_array_too_short)
        self.assertRaises(
            ValueError, self.vec_space.lin_combine, mode_handles, vec_handles,
            coeffs_array_too_fat)


    #@unittest.skip('Testing other things')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only')
    def test_compute_inner_product_array_types(self):
        num_row_vecs = 4
        num_col_vecs = 6
        num_states = 7

        row_vec_path = join(self.test_dir, 'row_vec_%03d.pkl')
        col_vec_path = join(self.test_dir, 'col_vec_%03d.pkl')

        # Check complex and real data
        for is_complex in [True, False]:

            # Generate data
            row_vec_array = np.random.random((num_states, num_row_vecs))
            col_vec_array = np.random.random((num_states, num_col_vecs))
            if is_complex:
                row_vec_array = row_vec_array * (
                    1j * np.random.random((num_states, num_row_vecs)))
                col_vec_array = col_vec_array * (
                    1j * np.random.random((num_states, num_col_vecs)))

            # Generate handles and save to file
            row_vec_paths = [row_vec_path % i for i in range(num_row_vecs)]
            col_vec_paths = [col_vec_path % i for i in range(num_col_vecs)]
            row_vec_handles = [
                vcs.VecHandlePickle(path) for path in row_vec_paths]
            col_vec_handles = [
                vcs.VecHandlePickle(path) for path in col_vec_paths]
            for idx, handle in enumerate(row_vec_handles):
                handle.put(row_vec_array[:, idx])
            for idx, handle in enumerate(col_vec_handles):
                handle.put(col_vec_array[:, idx])

            # Compute inner product array and check type
            inner_product_array = self.vec_space.compute_inner_product_array(
                row_vec_handles, col_vec_handles)
            symm_inner_product_array =\
                self.vec_space.compute_symm_inner_product_array(
                    row_vec_handles)
            self.assertEqual(inner_product_array.dtype, row_vec_array.dtype)
            self.assertEqual(
                symm_inner_product_array.dtype, row_vec_array.dtype)


    #@unittest.skip('Testing other things')
    def test_compute_inner_product_arrays(self):
        """Test computation of array of inner products."""
        rtol = 1e-10
        atol = 1e-12

        num_row_vecs_list = [
            1,
            int(round(self.total_num_vecs_in_mem / 2.)),
            self.total_num_vecs_in_mem,
            self.total_num_vecs_in_mem * 2,
            parallel.get_num_procs() + 1]
        num_col_vecs_list = num_row_vecs_list
        num_states = 6

        row_vec_path = join(self.test_dir, 'row_vec_%03d.pkl')
        col_vec_path = join(self.test_dir, 'col_vec_%03d.pkl')

        for num_row_vecs in num_row_vecs_list:
            for num_col_vecs in num_col_vecs_list:

                # Generate vecs
                parallel.barrier()
                row_vec_array = (
                    parallel.call_and_bcast(
                        np.random.random, (num_states, num_row_vecs))
                    + 1j * parallel.call_and_bcast(
                        np.random.random, (num_states, num_row_vecs)))
                col_vec_array = (
                    parallel.call_and_bcast(
                        np.random.random, (num_states, num_col_vecs))
                    + 1j * parallel.call_and_bcast(
                        np.random.random, (num_states, num_col_vecs)))
                row_vec_handles = [
                    vcs.VecHandlePickle(row_vec_path % i)
                    for i in range(num_row_vecs)]
                col_vec_handles = [
                    vcs.VecHandlePickle(col_vec_path % i)
                    for i in range(num_col_vecs)]

                # Save vecs
                if parallel.is_rank_zero():
                    for i, h in enumerate(row_vec_handles):
                        h.put(row_vec_array[:, i])
                    for i, h in enumerate(col_vec_handles):
                        h.put(col_vec_array[:, i])
                parallel.barrier()

                # If number of rows/cols is 1, check case of passing a handle
                if len(row_vec_handles) == 1:
                    row_vec_handles = row_vec_handles[0]
                if len(col_vec_handles) == 1:
                    col_vec_handles = col_vec_handles[0]

                # Test ip computation.
                product_true = np.dot(row_vec_array.conj().T, col_vec_array)
                product_computed = self.vec_space.compute_inner_product_array(
                    row_vec_handles, col_vec_handles)
                np.testing.assert_allclose(
                    product_computed, product_true, rtol=rtol, atol=atol)

                # Test symm ip computation
                product_true = np.dot(row_vec_array.conj().T, row_vec_array)
                product_computed =\
                    self.vec_space.compute_symm_inner_product_array(
                        row_vec_handles)
                np.testing.assert_allclose(
                    product_computed, product_true, rtol=rtol, atol=atol)


if __name__=='__main__':
    unittest.main()
