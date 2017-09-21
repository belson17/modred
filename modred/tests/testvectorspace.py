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
import modred.vectors as V
import modred.util


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
        nx = 40
        ny = 15
        test_array = np.random.random((nx, ny))

        vec_space = VectorSpaceHandles(inner_product=np.vdot, verbosity=0)
        in_mem_handle = V.VecHandleInMemory(test_array)
        vec_space.sanity_check(in_mem_handle)

        # Define some weird vectors that alter their internal data when adding
        # or multiplying (which they shouldn't do).
        class SanityMultVec(V.Vector):
            def __init__(self, arr):
                self.arr = arr

            def __add__(self, obj):
                f_return = copy.deepcopy(self)
                f_return.arr += obj.arr
                return f_return

            def __mul__(self, a):
                self.arr *= a
                return self

        class SanityAddVec(V.Vector):
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
        def good_custom_IP(vec1, vec2):
            return np.vdot(vec1.arr, vec2.arr)

        # Define a bad inner product for regular arrays
        def bad_array_IP(vec1, vec2):
            return np.vdot(vec1, vec2 ** 2.)

        # Make sure that sanity check passes for vectors with properly defined
        # vector operations.  Do so by simply calling the function.  If it
        # passes, then no error will be raised.
        vec_space.inner_product = np.vdot
        vec_space.sanity_check(V.VecHandleInMemory(test_array))

        # Make sure that sanity check fails if inner product values are not
        # correct.
        vec_space.inner_product = bad_array_IP
        self.assertRaises(
            ValueError,
            vec_space.sanity_check, V.VecHandleInMemory(test_array))

        # Make sure that sanity check fails if vectors alter their
        # internal data when doing vector space operations.
        vec_space.inner_product = good_custom_IP
        sanity_mult_vec = SanityMultVec(test_array)
        self.assertRaises(
            ValueError,
            vec_space.sanity_check, V.VecHandleInMemory(sanity_mult_vec))
        sanity_add_vec = SanityAddVec(test_array)
        self.assertRaises(
            ValueError,
            vec_space.sanity_check, V.VecHandleInMemory(sanity_add_vec))


    def generate_vecs_modes(self, num_states, num_vecs, num_modes):
        """Generates random vecs and finds the modes.

        Returns:
            vec_array: array in which each column is a vec (in order)
            mode_indices: unordered list of integers representing mode indices,
                each entry is unique. Mode indices are picked randomly.
            build_coeff_array: array of shape num_vecs x num_modes, random
                entries
            mode_array: array of modes, each column is a mode.
                The index of the array column equals the mode index.
        """
        mode_indices = list(range(num_modes))
        random.shuffle(mode_indices)
        build_coeff_array = np.mat(np.random.random((num_vecs, num_modes)))
        vec_array = np.random.random((num_states, num_vecs))
        mode_array = vec_array.dot(build_coeff_array)
        return vec_array, mode_indices, build_coeff_array, mode_array


    #@unittest.skip('Testing other things')
    def test_lin_combine(self):
        rtol = 1e-10
        atol = 1e-12

        num_vecs_list = [1, 15, 40]
        num_states = 20
        # Test cases where number of modes:
        #   less, equal, more than num_states
        #   less, equal, more than num_vecs
        #   less, equal, more than total_num_vecs_in_mem
        num_modes_list = [
            1, 8, 10, 20, 25, 45,
            int(np.ceil(self.total_num_vecs_in_mem / 2.)),
            self.total_num_vecs_in_mem, self.total_num_vecs_in_mem * 2]
        mode_path = join(self.test_dir, 'mode_%03d.txt')
        vec_path = join(self.test_dir, 'vec_%03d.txt')

        for num_vecs in num_vecs_list:
            for num_modes in num_modes_list:
                # Generate data and then broadcast to all procs
                vec_handles = [
                    V.VecHandleArrayText(vec_path % i) for i in range(num_vecs)]
                vec_array, mode_indices, build_coeff_mat, true_modes =\
                    parallel.call_and_bcast(
                        self.generate_vecs_modes, num_states, num_vecs,
                        num_modes)
                if parallel.is_rank_zero():
                    for vec_index, vec_handle in enumerate(vec_handles):
                        vec_handle.put(vec_array[:, vec_index])
                parallel.barrier()
                mode_handles = [
                    V.VecHandleArrayText(mode_path % mode_num)
                    for mode_num in mode_indices]

                # If there are more vecs than mat has rows
                build_coeff_array_too_small = np.zeros(
                    (build_coeff_mat.shape[0] - 1, build_coeff_mat.shape[1]))
                self.assertRaises(
                    ValueError,
                    self.vec_space.lin_combine, mode_handles, vec_handles,
                    build_coeff_array_too_small, mode_indices)

                # Test the case that only one mode is desired,
                # in which case user might pass in an int
                if len(mode_indices) == 1:
                    mode_indices = mode_indices[0]
                    mode_handles = mode_handles[0]

                # Saves modes to files
                self.vec_space.lin_combine(
                    mode_handles, vec_handles, build_coeff_mat, mode_indices)

                # Change back to list so is iterable
                if not isinstance(mode_indices, list):
                    mode_indices = [mode_indices]
                parallel.barrier()
                for mode_index in mode_indices:
                    computed_mode = V.VecHandleArrayText(
                        mode_path % mode_index).get()
                    np.testing.assert_allclose(
                        computed_mode, true_modes[:,mode_index],
                        rtol=rtol, atol=atol)
                parallel.barrier()
        parallel.barrier()


    #@unittest.skip('Testing other things')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only')
    def test_compute_inner_product_mat_types(self):
        class ArrayTextComplexHandle(V.VecHandleArrayText):
            def get(self):
                return (1 + 1j) * V.VecHandleArrayText.get(self)

        num_row_vecs = 4
        num_col_vecs = 6
        num_states = 7

        row_vec_path = join(self.test_dir, 'row_vec_%03d.txt')
        col_vec_path = join(self.test_dir, 'col_vec_%03d.txt')

        # Generate vecs and save to file
        row_vec_array = np.mat(np.random.random(
            (num_states, num_row_vecs)))
        col_vec_array = np.mat(np.random.random(
            (num_states, num_col_vecs)))
        row_vec_paths = []
        col_vec_paths = []
        for vec_index in range(num_row_vecs):
            path = row_vec_path % vec_index
            util.save_array_text(row_vec_array[:, vec_index], path)
            row_vec_paths.append(path)
        for vec_index in range(num_col_vecs):
            path = col_vec_path % vec_index
            util.save_array_text(col_vec_array[:, vec_index], path)
            col_vec_paths.append(path)

        # Compute inner product matrix and check type
        for handle, dtype in [
            (V.VecHandleArrayText, float),
            (ArrayTextComplexHandle, complex)]:
            row_vec_handles = [handle(path) for path in row_vec_paths]
            col_vec_handles = [handle(path) for path in col_vec_paths]
            inner_product_mat = self.vec_space.compute_inner_product_array(
                row_vec_handles, col_vec_handles)
            symm_inner_product_mat =\
                self.vec_space.compute_symmetric_inner_product_array(
                    row_vec_handles)
            self.assertEqual(inner_product_mat.dtype, dtype)
            self.assertEqual(symm_inner_product_mat.dtype, dtype)


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
                    V.VecHandlePickle(row_vec_path % i)
                    for i in range(num_row_vecs)]
                col_vec_handles = [
                    V.VecHandlePickle(col_vec_path % i)
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

                # Test IP computation.
                product_true = np.dot(row_vec_array.conj().T, col_vec_array)
                product_computed = self.vec_space.compute_inner_product_array(
                    row_vec_handles, col_vec_handles)
                np.testing.assert_allclose(
                    product_computed, product_true, rtol=rtol, atol=atol)

                # Test symm IP computation
                product_true = np.dot(row_vec_array.conj().T, row_vec_array)
                product_computed =\
                    self.vec_space.compute_symmetric_inner_product_array(
                        row_vec_handles)
                np.testing.assert_allclose(
                    product_computed, product_true, rtol=rtol, atol=atol)


if __name__=='__main__':
    unittest.main()
