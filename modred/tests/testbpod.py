#!/usr/bin/env python
"""Test the bpod module"""
import unittest
import os
from os.path import join
from shutil import rmtree
import copy

import numpy as np

from modred import bpod, parallel, util
from modred.py2to3 import range
from modred.vectorspace import VectorSpaceArrays, VectorSpaceHandles
from modred.vectors import VecHandlePickle


def get_system_arrays(num_states, num_inputs, num_outputs):
    eig_vals = (
        (0.05 * np.random.random(num_states) + 0.8) +
        1j * (0.05 * np.random.random(num_states) + 0.8))
    eig_vecs = (
        (2. * np.random.random((num_states, num_states)) - 1.) +
        1j * (2. * np.random.random((num_states, num_states)) - 1.))
    A = np.linalg.inv(eig_vecs).dot(np.diag(eig_vals).dot(eig_vecs))
    B = (
        (2. * np.random.random((num_states, num_inputs)) - 1.) +
        1j * (2. * np.random.random((num_states, num_inputs)) - 1.))
    C = (
        (2. * np.random.random((num_outputs, num_states)) - 1.) +
        1j * (2. * np.random.random((num_outputs, num_states)) - 1.))
    return A, B, C


def get_direct_impulse_response_array(A, B, num_steps):
    num_states, num_inputs = B.shape
    direct_vecs = np.zeros((num_states, num_steps * num_inputs), dtype=A.dtype)
    A_powers = np.identity(num_states)
    for idx in range(num_steps):
        direct_vecs[:, idx * num_inputs:(idx + 1) * num_inputs] =\
            A_powers.dot(B)
        A_powers = A_powers.dot(A)
    return direct_vecs


def get_adjoint_impulse_response_array(A, C, num_steps, weights_array):
    num_outputs, num_states = C.shape
    A_adjoint = np.linalg.inv(weights_array).dot(A.conj().T.dot(weights_array))
    C_adjoint = np.linalg.inv(weights_array).dot(C.conj().T)
    adjoint_vecs = np.zeros((num_states, num_steps * num_outputs), dtype=A.dtype)
    A_adjoint_powers = np.identity(num_states)
    for idx in range(num_steps):
        adjoint_vecs[:, (idx * num_outputs):(idx + 1) * num_outputs] =\
            A_adjoint_powers.dot(C_adjoint)
        A_adjoint_powers = A_adjoint_powers.dot(A_adjoint)
    return adjoint_vecs


#@unittest.skip('Testing something else.')
@unittest.skipIf(parallel.is_distributed(), 'Serial only.')
class TestBPODArrays(unittest.TestCase):
    def setUp(self):
        self.num_states = 10
        self.num_steps = self.num_states + 1


    def test_all(self):
        # Set test tolerances.  Separate, more relaxed tolerances are required
        # for testing the BPOD modes, since that test requires "squaring" the
        # gramians and thus involves more ill-conditioned arrays.
        rtol = 1e-8
        atol = 1e-10
        rtol_sqr = 1e-8
        atol_sqr = 1e-8

        # Set tolerances for SVD step of BPOD.  This is necessary to avoid
        # dealing with very uncontrollable/unobservable states, which can cause
        # the tests to fail.
        rtol_svd = 1e-6
        atol_svd = 1e-12

        # Generate weights to test different inner products.  Keep most of the
        # weights close to one, to avoid overly weighting certain states over
        # others.  This can dramatically affect the rate at which the tests
        # pass.
        weights_1D = np.random.random(self.num_states)
        weights_2D = np.identity(self.num_states, dtype=np.complex)
        weights_2D[0, 0] = 2.
        weights_2D[2, 1] = 0.3j
        weights_2D[1, 2] = weights_2D[2, 1].conj()
        weights_list = [None, weights_1D, weights_2D]
        weights_array_list = [
            np.identity(self.num_states), np.diag(weights_1D), weights_2D]

        # Check different system sizes.  Make sure to test a single input/output
        # in addition to multiple inputs/outputs.  Also allow for the number of
        # inputs/outputs to exceed the number of states.
        for num_inputs in [1, np.random.randint(2, high=self.num_states + 2)]:

            for num_outputs in [
                1, np.random.randint(2, high=self.num_states + 2)]:

                # Get state space system
                A, B, C = get_system_arrays(
                    self.num_states, num_inputs, num_outputs)

                # Compute direct impulse response
                direct_vecs_array = get_direct_impulse_response_array(
                    A, B, self.num_steps)

                # Loop through different inner product weights
                for weights, weights_array in zip(
                    weights_list, weights_array_list):

                    # Define inner product based on weights
                    IP = VectorSpaceArrays(
                        weights=weights).compute_inner_product_array

                    # Compute adjoint impulse response
                    adjoint_vecs_array = get_adjoint_impulse_response_array(
                        A, C, self.num_steps, weights_array)

                    # Compute BPOD using modred.  Use absolute tolerance to
                    # avoid Hankel singular values that approach numerical
                    # precision.  Use relative tolerance to avoid Hankel
                    # singular values which may correspond to very
                    # uncontrollable/unobservable states.  It is ok to use a
                    # more relaxed tolerance here than in the actual test/assert
                    # statements, as here we are saying it is ok to ignore
                    # highly uncontrollable/unobservable states, rather than
                    # allowing loose tolerances in the comparison of two
                    # numbers.  Furthermore, it is likely that in actual use,
                    # users would want to ignore relatively small Hankel
                    # singular values anyway, as that is the point of doing a
                    # balancing transformation.
                    BPOD_res = bpod.compute_BPOD_arrays(
                        direct_vecs_array, adjoint_vecs_array,
                        num_inputs=num_inputs, num_outputs=num_outputs,
                        inner_product_weights=weights,
                        rtol=rtol_svd, atol=atol_svd)

                    # Check Hankel array values.  These are computed fast
                    # internally by only computing the first column and last row
                    # of chunks.  Here, simply take all the inner products.
                    Hankel_array_slow = IP(
                        adjoint_vecs_array, direct_vecs_array)
                    np.testing.assert_allclose(
                        BPOD_res.Hankel_array, Hankel_array_slow,
                        rtol=rtol, atol=atol)

                    # Check properties of SVD of Hankel array.  Since the SVD
                    # may be truncated, instead of checking orthogonality and
                    # reconstruction of the Hankel array, check that the left
                    # and right singular vectors satisfy eigendecomposition
                    # properties with respect to the Hankel array.  Since this
                    # involves "squaring" the Hankel array, it requires more
                    # relaxed test tolerances.
                    np.testing.assert_allclose(
                        BPOD_res.Hankel_array.dot(
                            BPOD_res.Hankel_array.conj().T.dot(
                                BPOD_res.L_sing_vecs)),
                        BPOD_res.L_sing_vecs.dot(
                            np.diag(BPOD_res.sing_vals ** 2.)),
                        rtol=rtol_sqr, atol=atol_sqr)
                    np.testing.assert_allclose(
                        BPOD_res.Hankel_array.conj().T.dot(
                            BPOD_res.Hankel_array.dot(
                                BPOD_res.R_sing_vecs)),
                        BPOD_res.R_sing_vecs.dot(
                            np.diag(BPOD_res.sing_vals ** 2.)),
                        rtol=rtol_sqr, atol=atol_sqr)

                    # Check that the modes diagonalize the gramians.  This test
                    # requires looser tolerances than the other tests, likely
                    # due to the "squaring" of the arrays in computing the
                    # gramians.
                    np.testing.assert_allclose(
                        IP(BPOD_res.adjoint_modes, direct_vecs_array).dot(
                            IP(direct_vecs_array, BPOD_res.adjoint_modes)),
                        np.diag(BPOD_res.sing_vals),
                        rtol=rtol_sqr, atol=atol_sqr)
                    np.testing.assert_allclose(
                        IP(BPOD_res.direct_modes, adjoint_vecs_array).dot(
                            IP(adjoint_vecs_array, BPOD_res.direct_modes)),
                        np.diag(BPOD_res.sing_vals),
                        rtol=rtol_sqr, atol=atol_sqr)

                    # Check the value of the projection coefficients against a
                    # projection onto the adjoint and direct modes,
                    # respectively.
                    np.testing.assert_allclose(
                        BPOD_res.direct_proj_coeffs,
                        IP(BPOD_res.adjoint_modes, direct_vecs_array),
                        rtol=rtol, atol=atol)
                    np.testing.assert_allclose(
                        BPOD_res.adjoint_proj_coeffs,
                        IP(BPOD_res.direct_modes, adjoint_vecs_array),
                        rtol=rtol, atol=atol)

                    # Check that if mode indices are passed in, the correct
                    # modes are returned.  Test both an explicit selection of
                    # mode indices and a None argument.
                    mode_indices_trunc = np.unique(np.random.randint(
                        0, high=BPOD_res.sing_vals.size,
                        size=(BPOD_res.sing_vals.size // 2)))
                    for mode_indices_arg, mode_indices_vals in zip(
                        [None, mode_indices_trunc],
                        [range(BPOD_res.sing_vals.size), mode_indices_trunc]):
                        BPOD_res_sliced = bpod.compute_BPOD_arrays(
                            direct_vecs_array, adjoint_vecs_array,
                            direct_mode_indices=mode_indices_arg,
                            adjoint_mode_indices=mode_indices_arg,
                            num_inputs=num_inputs, num_outputs=num_outputs,
                            inner_product_weights=weights,
                            rtol=rtol_svd, atol=atol_svd)
                        np.testing.assert_allclose(
                            BPOD_res_sliced.direct_modes,
                            BPOD_res.direct_modes[:, mode_indices_vals],
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            BPOD_res_sliced.adjoint_modes,
                            BPOD_res.adjoint_modes[:, mode_indices_vals],
                            rtol=rtol, atol=atol)


#@unittest.skip('Testing something else.')
class TestBPODHandles(unittest.TestCase):
    """Test the BPOD class methods """
    def setUp(self):
        # Specify output locations
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'files_BPOD_DELETE_ME'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        self.direct_vec_path = join(self.test_dir, 'direct_vec_%03d.pkl')
        self.adjoint_vec_path = join(self.test_dir, 'adjoint_vec_%03d.pkl')
        self.direct_mode_path = join(self.test_dir, 'direct_mode_%03d.pkl')
        self.adjoint_mode_path = join(self.test_dir, 'adjoint_mode_%03d.pkl')

        # Specify system dimensions.  Test single inputs/outputs as well as
        # multiple inputs/outputs.  Also allow for more inputs/outputs than
        # states.
        self.num_states = 10
        self.num_inputs_list = [
            1,
            parallel.call_and_bcast(np.random.randint, 2, self.num_states + 2)]
        self.num_outputs_list = [
            1,
            parallel.call_and_bcast(np.random.randint, 2, self.num_states + 2)]

        # Specify how long to run impulse responses
        self.num_steps = self.num_states + 1

        parallel.barrier()


    def tearDown(self):
        parallel.barrier()
        parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        parallel.barrier()


    #@unittest.skip('Testing something else.')
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""

        def my_load(fname): pass
        def my_save(data, fname): pass
        def my_IP(vec1, vec2): pass

        data_members_default = {'put_array': util.save_array_text, 'get_array':
             util.load_array_text,
            'verbosity': 0, 'L_sing_vecs': None, 'R_sing_vecs': None,
            'sing_vals': None, 'direct_vec_handles': None,
            'adjoint_vec_handles': None,
            'direct_vec_handles': None, 'adjoint_vec_handles': None,
            'Hankel_array': None,
            'vec_space': VectorSpaceHandles(inner_product=my_IP, verbosity=0)}

        # Get default data member values
        for k,v in util.get_data_members(
            bpod.BPODHandles(my_IP, verbosity=0)).items():
            self.assertEqual(v, data_members_default[k])

        my_BPOD = bpod.BPODHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])

        my_BPOD = bpod.BPODHandles(my_IP, get_array=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_array'] = my_load
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])

        my_BPOD = bpod.BPODHandles(my_IP, put_array=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_array'] = my_save
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])

        max_vecs_per_node = 500
        my_BPOD = bpod.BPODHandles(
            my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * parallel.get_num_nodes() / parallel.\
            get_num_procs()
        for k,v in util.get_data_members(my_BPOD).items():
            self.assertEqual(v, data_members_modified[k])


    #@unittest.skip('Testing something else.')
    def test_puts_gets(self):
        """Test that put/get work in base class."""
        # Generate some random data
        Hankel_array_true = parallel.call_and_bcast(
            np.random.random, ((self.num_states, self.num_states)))
        L_sing_vecs_true, sing_vals_true, R_sing_vecs_true = \
            parallel.call_and_bcast(util.svd, Hankel_array_true)
        direct_proj_coeffs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_steps, self.num_steps)))
        adj_proj_coeffs_true = parallel.call_and_bcast(
            np.random.random, ((self.num_steps, self.num_steps)))

        # Store the data in a BPOD object
        BPOD_save = bpod.BPODHandles(None, verbosity=0)
        BPOD_save.Hankel_array = Hankel_array_true
        BPOD_save.sing_vals = sing_vals_true
        BPOD_save.L_sing_vecs = L_sing_vecs_true
        BPOD_save.R_sing_vecs = R_sing_vecs_true
        BPOD_save.direct_proj_coeffs = direct_proj_coeffs_true
        BPOD_save.adjoint_proj_coeffs = adj_proj_coeffs_true

        # Use the BPOD object to save the data to disk
        sing_vals_path = join(self.test_dir, 'sing_vals.txt')
        L_sing_vecs_path = join(self.test_dir, 'L_sing_vecs.txt')
        R_sing_vecs_path = join(self.test_dir, 'R_sing_vecs.txt')
        Hankel_array_path = join(self.test_dir, 'Hankel_array.txt')
        direct_proj_coeffs_path = join(self.test_dir, 'direct_proj_coeffs.txt')
        adj_proj_coeffs_path = join(self.test_dir, 'adj_proj_coeffs.txt')
        BPOD_save.put_decomp(sing_vals_path, L_sing_vecs_path, R_sing_vecs_path)
        BPOD_save.put_Hankel_array(Hankel_array_path)
        BPOD_save.put_direct_proj_coeffs(direct_proj_coeffs_path)
        BPOD_save.put_adjoint_proj_coeffs(adj_proj_coeffs_path)

        # Create a BPOD object and use it to load the data from disk
        BPOD_load = bpod.BPODHandles(None, verbosity=0)
        BPOD_load.get_decomp(sing_vals_path, L_sing_vecs_path, R_sing_vecs_path)
        BPOD_load.get_Hankel_array(Hankel_array_path)
        BPOD_load.get_direct_proj_coeffs(direct_proj_coeffs_path)
        BPOD_load.get_adjoint_proj_coeffs(adj_proj_coeffs_path)

        # Compare loaded data or original data
        np.testing.assert_equal(BPOD_load.sing_vals, sing_vals_true)
        np.testing.assert_equal(BPOD_load.L_sing_vecs, L_sing_vecs_true)
        np.testing.assert_equal(BPOD_load.R_sing_vecs, R_sing_vecs_true)
        np.testing.assert_equal(BPOD_load.Hankel_array, Hankel_array_true)
        np.testing.assert_equal(
            BPOD_load.direct_proj_coeffs, direct_proj_coeffs_true)
        np.testing.assert_equal(
            BPOD_load.adjoint_proj_coeffs, adj_proj_coeffs_true)


    # Compute impulse responses and generate corresponding handles
    def _helper_get_impulse_response_handles(self, num_inputs, num_outputs):
        # Get state space system
        A, B, C = parallel.call_and_bcast(
            get_system_arrays, self.num_states, num_inputs, num_outputs)

        # Run impulse responses
        direct_vec_array = parallel.call_and_bcast(
            get_direct_impulse_response_array, A, B, self.num_steps)
        adjoint_vec_array = parallel.call_and_bcast(
            get_adjoint_impulse_response_array, A, C, self.num_steps,
            np.identity(self.num_states))

        # Save data to disk
        direct_vec_handles = [
            VecHandlePickle(self.direct_vec_path % i)
            for i in range(direct_vec_array.shape[1])]
        adjoint_vec_handles = [
            VecHandlePickle(self.adjoint_vec_path % i)
            for i in range(adjoint_vec_array.shape[1])]
        if parallel.is_rank_zero():
            for idx, handle in enumerate(direct_vec_handles):
                handle.put(direct_vec_array[:, idx])
            for idx, handle in enumerate(adjoint_vec_handles):
                handle.put(adjoint_vec_array[:, idx])

        parallel.barrier()
        return direct_vec_handles, adjoint_vec_handles


    #@unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test that can take vecs, compute the Hankel and SVD arrays. """
        # Set test tolerances.  Separate, more relaxed tolerances may be
        # required for testing the SVD arrays, since that test requires
        # "squaring" the Hankel array and thus involves more ill-conditioned
        # arrays.
        rtol = 1e-8
        atol = 1e-10
        rtol_sqr = 1e-8
        atol_sqr = 1e-8

        # Test a single input/output as well as multiple inputs/outputs.  Allow
        # for more inputs/outputs than states.  (This is determined in setUp()).
        for num_inputs in self.num_inputs_list:
            for num_outputs in self.num_outputs_list:

                # Get impulse response data
                direct_vec_handles, adjoint_vec_handles =\
                self._helper_get_impulse_response_handles(
                    num_inputs, num_outputs)

                # Compute BPOD using modred.
                BPOD = bpod.BPODHandles(np.vdot, verbosity=0)
                sing_vals, L_sing_vecs, R_sing_vecs = BPOD.compute_decomp(
                    direct_vec_handles, adjoint_vec_handles,
                    num_inputs=num_inputs, num_outputs=num_outputs)

                # Check Hankel array values.  These are computed fast
                # internally by only computing the first column and last row
                # of chunks.  Here, simply take all the inner products.
                Hankel_array_slow = BPOD.vec_space.compute_inner_product_array(
                    adjoint_vec_handles, direct_vec_handles)
                np.testing.assert_allclose(
                    BPOD.Hankel_array, Hankel_array_slow, rtol=rtol, atol=atol)

                # Check properties of SVD of Hankel array.  Since the SVD
                # may be truncated, instead of checking orthogonality and
                # reconstruction of the Hankel array, check that the left
                # and right singular vectors satisfy eigendecomposition
                # properties with respect to the Hankel array.  Since this
                # involves "squaring" the Hankel array, it may require more
                # relaxed test tolerances.
                np.testing.assert_allclose(
                    BPOD.Hankel_array.dot(
                        BPOD.Hankel_array.conj().T.dot(
                            BPOD.L_sing_vecs)),
                    BPOD.L_sing_vecs.dot(np.diag(BPOD.sing_vals ** 2.)),
                    rtol=rtol_sqr, atol=atol_sqr)
                np.testing.assert_allclose(
                    BPOD.Hankel_array.conj().T.dot(
                        BPOD.Hankel_array.dot(
                            BPOD.R_sing_vecs)),
                    BPOD.R_sing_vecs.dot(np.diag(BPOD.sing_vals ** 2.)),
                    rtol=rtol_sqr, atol=atol_sqr)

                # Check that returned values match internal values
                np.testing.assert_equal(sing_vals, BPOD.sing_vals)
                np.testing.assert_equal(L_sing_vecs, BPOD.L_sing_vecs)
                np.testing.assert_equal(R_sing_vecs, BPOD.R_sing_vecs)


    #@unittest.skip('Testing something else.')
    def test_compute_modes(self):
        """Test computing modes in serial and parallel."""
        # Set test tolerances.  More relaxed tolerances are required for testing
        # the BPOD modes, since that test requires "squaring" the gramians and
        # thus involves more ill-conditioned arrays.
        rtol_sqr = 1e-8
        atol_sqr = 1e-8

        # Test a single input/output as well as multiple inputs/outputs.  Allow
        # for more inputs/outputs than states.  (This is determined in setUp()).
        for num_inputs in self.num_inputs_list:
            for num_outputs in self.num_outputs_list:

                # Get impulse response data
                direct_vec_handles, adjoint_vec_handles =\
                    self._helper_get_impulse_response_handles(
                        num_inputs, num_outputs)

                # Create BPOD object and perform decomposition.  (The properties
                # defining a BPOD mode require manipulations involving the
                # correct decomposition, so we cannot isolate the mode
                # computation from the decomposition step.)  Use relative
                # tolerance to avoid Hankel singular values which may correspond
                # to very uncontrollable/unobservable states.  It is ok to use a
                # more relaxed tolerance here than in the actual test/assert
                # statements, as here we are saying it is ok to ignore highly
                # uncontrollable/unobservable states, rather than allowing loose
                # tolerances in the comparison of two numbers.  Furthermore, it
                # is likely that in actual use, users would want to ignore
                # relatively small Hankel singular values anyway, as that is the
                # point of doing a balancing transformation.
                BPOD = bpod.BPODHandles(np.vdot, verbosity=0)
                BPOD.compute_decomp(
                    direct_vec_handles, adjoint_vec_handles,
                    num_inputs=num_inputs, num_outputs=num_outputs,
                    rtol=1e-6, atol=1e-12)

                # Select a subset of modes to compute.  Compute at least half
                # the modes, and up to all of them.  Make sure to use unique
                # values.  (This may reduce the number of modes computed.)
                num_modes = parallel.call_and_bcast(
                    np.random.randint,
                    BPOD.sing_vals.size // 2, BPOD.sing_vals.size + 1)
                mode_idxs = np.unique(parallel.call_and_bcast(
                    np.random.randint,
                    0, BPOD.sing_vals.size, num_modes))

                # Create handles for the modes
                direct_mode_handles = [
                    VecHandlePickle(self.direct_mode_path % i)
                    for i in mode_idxs]
                adjoint_mode_handles = [
                    VecHandlePickle(self.adjoint_mode_path % i)
                    for i in mode_idxs]

                # Compute modes
                BPOD.compute_direct_modes(
                    mode_idxs, direct_mode_handles,
                    direct_vec_handles=direct_vec_handles)
                BPOD.compute_adjoint_modes(
                    mode_idxs, adjoint_mode_handles,
                    adjoint_vec_handles=adjoint_vec_handles)

                # Test modes against empirical gramians
                np.testing.assert_allclose(
                    BPOD.vec_space.compute_inner_product_array(
                        adjoint_mode_handles, direct_vec_handles).dot(
                            BPOD.vec_space.compute_inner_product_array(
                                direct_vec_handles, adjoint_mode_handles)),
                    np.diag(BPOD.sing_vals[mode_idxs]),
                    rtol=rtol_sqr, atol=atol_sqr)
                np.testing.assert_allclose(
                    BPOD.vec_space.compute_inner_product_array(
                        direct_mode_handles, adjoint_vec_handles).dot(
                            BPOD.vec_space.compute_inner_product_array(
                                adjoint_vec_handles, direct_mode_handles)),
                    np.diag(BPOD.sing_vals[mode_idxs]),
                    rtol=rtol_sqr, atol=atol_sqr)


    #@unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        # Set test tolerances.  Use a slightly more relaxed absolute tolerance
        # here because the projection test uses modes that may correspond to
        # smaller Hankel singular values (i.e., less controllable/unobservable
        # states).  Those mode pairs are not as close to biorthogonal, so a more
        # relaxed tolerance is required.
        rtol = 1e-8
        atol = 1e-8

        # Test a single input/output as well as multiple inputs/outputs.  Allow
        # for more inputs/outputs than states.  (This is determined in setUp()).
        for num_inputs in self.num_inputs_list:
            for num_outputs in self.num_outputs_list:

                # Get impulse response data
                direct_vec_handles, adjoint_vec_handles =\
                self._helper_get_impulse_response_handles(
                    num_inputs, num_outputs)

                # Create BPOD object and compute decomposition, modes.  (The
                # properties defining a projection onto BPOD modes require
                # manipulations involving the correct decomposition and modes,
                # so we cannot isolate the projection step from those
                # computations.)  Use relative tolerance to avoid Hankel
                # singular values which may correspond to very
                # uncontrollable/unobservable states.  It is ok to use a
                # more relaxed tolerance here than in the actual test/assert
                # statements, as here we are saying it is ok to ignore
                # highly uncontrollable/unobservable states, rather than
                # allowing loose tolerances in the comparison of two
                # numbers.  Furthermore, it is likely that in actual use,
                # users would want to ignore relatively small Hankel
                # singular values anyway, as that is the point of doing a
                # balancing transformation.
                BPOD = bpod.BPODHandles(np.vdot, verbosity=0)
                BPOD.compute_decomp(
                    direct_vec_handles, adjoint_vec_handles,
                    num_inputs=num_inputs, num_outputs=num_outputs,
                    rtol=1e-6, atol=1e-12)
                mode_idxs = range(BPOD.sing_vals.size)
                direct_mode_handles = [
                    VecHandlePickle(self.direct_mode_path % i)
                    for i in mode_idxs]
                adjoint_mode_handles = [
                    VecHandlePickle(self.adjoint_mode_path % i)
                    for i in mode_idxs]
                BPOD.compute_direct_modes(
                    mode_idxs, direct_mode_handles,
                    direct_vec_handles=direct_vec_handles)
                BPOD.compute_adjoint_modes(
                    mode_idxs, adjoint_mode_handles,
                    adjoint_vec_handles=adjoint_vec_handles)

                # Compute true projection coefficients by computing the inner
                # products between modes and snapshots.
                direct_proj_coeffs_true =\
                BPOD.vec_space.compute_inner_product_array(
                    adjoint_mode_handles, direct_vec_handles)
                adjoint_proj_coeffs_true =\
                BPOD.vec_space.compute_inner_product_array(
                    direct_mode_handles, adjoint_vec_handles)

                # Compute projection coefficients using BPOD object, which
                # avoids actually manipulating handles and computing inner
                # products, instead using elements of the decomposition for a
                # more efficient computation.
                direct_proj_coeffs = BPOD.compute_direct_proj_coeffs()
                adjoint_proj_coeffs = BPOD.compute_adjoint_proj_coeffs()

                # Test values
                np.testing.assert_allclose(
                    direct_proj_coeffs, direct_proj_coeffs_true,
                    rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    adjoint_proj_coeffs, adjoint_proj_coeffs_true,
                    rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
