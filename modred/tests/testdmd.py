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

import modred.parallel as parallel
from modred.dmd import *
from modred.vectorspace import *
from modred.vectors import VecHandlePickle
from modred import util
from modred import pod


#@unittest.skip('Testing something else.')
@unittest.skipIf(parallel.is_distributed(), 'Serial only.')
class TestDMDArraysFunctions(unittest.TestCase):
    def setUp(self):
        self.num_states = 30
        self.num_vecs = 10


    def test_all(self):
        rtol = 1e-10
        atol = 1e-12

        # Generate weights to test different inner products.
        ws = np.identity(self.num_states)
        ws[0, 0] = 2.
        ws[2, 1] = 0.3
        ws[1, 2] = 0.3
        weights_list = [None, np.random.random(self.num_states), ws]

        # Generate random snapshot data
        vecs_mat = np.mat(np.random.random((self.num_states, self.num_vecs)))
        adv_vecs_mat = np.mat(np.random.random(
            (self.num_states, self.num_vecs)))

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [vecs_mat, vecs_mat],
            [None, adv_vecs_mat],
            [vecs_mat[:, :-1], vecs_mat],
            [vecs_mat[:, 1:], adv_vecs_mat]):

            # Test both method of snapshots and direct method
            for method in ['snaps', 'direct']:

                # Consider different inner product weights
                for weights in weights_list:
                    IP = VectorSpaceMatrices(
                        weights=weights).compute_inner_product_mat

                    # Test that results hold for truncated or untruncated DMD
                    # (i.e., whether or not the underlying POD basis is
                    # truncated).
                    for max_num_eigvals in [None, self.num_vecs // 2]:

                        # Choose subset of modes to compute, for testing mode
                        # indices argument
                        if max_num_eigvals is None:
                            mode_indices = np.unique(np.random.randint(
                                0, high=np.linalg.matrix_rank(vecs_vals),
                                size=np.linalg.matrix_rank(vecs_vals) // 2))
                        else:
                            mode_indices = np.unique(np.random.randint(
                                0, high=max_num_eigvals,
                                size=max_num_eigvals // 2))

                        # Compute DMD using appropriate method.  Also compute
                        # POD modes of vecs using same method, and a random
                        # subset of the DMD modes, in preparation for later
                        # tests.
                        if method == 'snaps':
                            DMD_res = compute_DMD_matrices_snaps_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals)
                            DMD_res_sliced = compute_DMD_matrices_snaps_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                mode_indices=mode_indices,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals)

                            # For method of snapshots, test correlation mats
                            # values by simply recomputing them.
                            np.testing.assert_allclose(
                                IP(vecs_vals, vecs_vals),
                                DMD_res.correlation_mat,
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                IP(vecs_vals, adv_vecs_vals),
                                DMD_res.cross_correlation_mat,
                                rtol=rtol, atol=atol)

                        elif method == 'direct':
                            DMD_res = compute_DMD_matrices_direct_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals)
                            DMD_res_sliced = compute_DMD_matrices_direct_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                mode_indices=mode_indices,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals)

                        else:
                            raise ValueError('Invalid DMD method.')

                        # Test correlation mat eigenvalues and eigenvectors.
                        np.testing.assert_allclose(
                            IP(vecs_vals, vecs_vals) *\
                            DMD_res.correlation_mat_eigvecs,
                            DMD_res.correlation_mat_eigvecs * np.mat(np.diag(
                                DMD_res.correlation_mat_eigvals)),
                            rtol=rtol, atol=atol)

                        # Compute the approximating linear operator relating the
                        # vecs to the adv_vecs.  To do this, use the
                        # eigendecomposition of the correlation mat.
                        vecs_POD_build_coeffs = (
                            DMD_res.correlation_mat_eigvecs *
                            np.mat(np.diag(
                                DMD_res.correlation_mat_eigvals ** -0.5)))
                        vecs_POD_modes = vecs_vals * vecs_POD_build_coeffs
                        approx_linear_op = (
                            adv_vecs_vals * DMD_res.correlation_mat_eigvecs *
                            np.mat(np.diag(
                                DMD_res.correlation_mat_eigvals ** -0.5)) *
                            vecs_POD_modes.H)
                        low_order_linear_op = IP(
                            vecs_POD_modes,
                            IP(approx_linear_op.H, vecs_POD_modes))

                        # Test the left and right eigenvectors of the low-order
                        # (projected) approximating linear operator.
                        np.testing.assert_allclose(
                            low_order_linear_op * DMD_res.R_low_order_eigvecs,
                            DMD_res.R_low_order_eigvecs * np.mat(np.diag(
                                DMD_res.eigvals)),
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            DMD_res.L_low_order_eigvecs.H * low_order_linear_op,
                            np.mat(np.diag(DMD_res.eigvals)) *\
                            DMD_res.L_low_order_eigvecs.H,
                            rtol=rtol, atol=atol)

                        # Test the exact modes, which are eigenvectors of the
                        # approximating linear operator.
                        np.testing.assert_allclose(
                            IP(approx_linear_op.H, DMD_res.exact_modes),
                            DMD_res.exact_modes * np.mat(np.diag(
                                DMD_res.eigvals)),
                            rtol=rtol, atol=atol)

                        # Test the projected modes, which are eigenvectors of
                        # the approximating linear operator projected onto the
                        # POD modes of the vecs.
                        np.testing.assert_allclose(
                            vecs_POD_modes * IP(
                                vecs_POD_modes,
                                IP(approx_linear_op.H, DMD_res.proj_modes)),
                            DMD_res.proj_modes * np.mat(np.diag(
                                DMD_res.eigvals)),
                            rtol=rtol, atol=atol)

                        # Test the adjoint modes, which are left eigenvectors of
                        # the approximating linear operator.
                        np.testing.assert_allclose(
                            IP(approx_linear_op, DMD_res.adjoint_modes),
                            DMD_res.adjoint_modes * np.mat(np.diag(
                                DMD_res.eigvals.conj().T)),
                            rtol=rtol, atol=atol)

                        # Test spectral coefficients against an explicit
                        # projection using the adjoint DMD modes.
                        np.testing.assert_allclose(
                            np.array(np.abs(IP(
                                DMD_res.adjoint_modes,
                                vecs_vals[:, 0]))).squeeze(),
                            DMD_res.spectral_coeffs,
                            rtol=rtol, atol=atol)

                        # Test that use of mode indices argument returns correct
                        # subset of modes
                        np.testing.assert_allclose(
                            DMD_res_sliced.exact_modes,
                            DMD_res.exact_modes[:, mode_indices],
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            DMD_res_sliced.proj_modes,
                            DMD_res.proj_modes[:, mode_indices],
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            DMD_res_sliced.adjoint_modes,
                            DMD_res.adjoint_modes[:, mode_indices],
                            rtol=rtol, atol=atol)


#@unittest.skip('Testing something else.')
class TestDMDHandles(unittest.TestCase):
    def setUp(self):
        # Specify output locations
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DMD_files'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        self.vec_path = join(self.test_dir, 'dmd_vec_%03d.pkl')
        self.adv_vec_path = join(self.test_dir, 'dmd_adv_vec_%03d.pkl')
        self.exact_mode_path = join(self.test_dir, 'dmd_exactmode_%03d.pkl')
        self.proj_mode_path = join(self.test_dir, 'dmd_projmode_%03d.pkl')
        self.adjoint_mode_path = join(self.test_dir, 'dmd_adjmode_%03d.pkl')

        # Specify data dimensions
        self.num_states = 30
        self.num_vecs = 10

        # Generate random data and write to disk using handles
        self.vecs_array = parallel.call_and_bcast(
            np.random.random, (self.num_states, self.num_vecs))
        self.adv_vecs_array = parallel.call_and_bcast(
            np.random.random, (self.num_states, self.num_vecs))
        self.vec_handles = [
            VecHandlePickle(self.vec_path % i) for i in range(self.num_vecs)]
        self.adv_vec_handles = [
            VecHandlePickle(self.adv_vec_path % i)
            for i in range(self.num_vecs)]
        for idx, (hdl, adv_hdl) in enumerate(
            zip(self.vec_handles, self.adv_vec_handles)):
            hdl.put(self.vecs_array[:, idx])
            adv_hdl.put(self.adv_vecs_array[:, idx])

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
        my_DMD = DMDHandles(
            my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = (
            max_vecs_per_node *
            parallel.get_num_nodes() /
            parallel.get_num_procs())
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])


    #@unittest.skip('Testing something else.')
    def test_puts_gets(self):
        """Test get and put functions"""
        # Generate some random data
        eigvals = parallel.call_and_bcast(np.random.random, 5)
        R_low_order_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        L_low_order_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        correlation_mat_eigvals = parallel.call_and_bcast(np.random.random, 5)
        correlation_mat_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        correlation_mat = parallel.call_and_bcast(np.random.random, (10,10))
        cross_correlation_mat = parallel.call_and_bcast(
            np.random.random, (10,10))
        spectral_coeffs = parallel.call_and_bcast(np.random.random, 5)
        proj_coeffs = parallel.call_and_bcast(np.random.random, (5, 5))
        adv_proj_coeffs = parallel.call_and_bcast(np.random.random, (5, 5))

        # Create a DMD object and store the data in it
        DMD_save = DMDHandles(None, verbosity=0)
        DMD_save.eigvals = eigvals
        DMD_save.R_low_order_eigvecs = R_low_order_eigvecs
        DMD_save.L_low_order_eigvecs = L_low_order_eigvecs
        DMD_save.correlation_mat_eigvals = correlation_mat_eigvals
        DMD_save.correlation_mat_eigvecs = correlation_mat_eigvecs
        DMD_save.correlation_mat = correlation_mat
        DMD_save.cross_correlation_mat = cross_correlation_mat
        DMD_save.spectral_coeffs = spectral_coeffs
        DMD_save.proj_coeffs = proj_coeffs
        DMD_save.adv_proj_coeffs = adv_proj_coeffs

        # Write the data to disk
        eigvals_path = join(self.test_dir, 'dmd_eigvals.txt')
        R_low_order_eigvecs_path = join(
            self.test_dir, 'dmd_R_low_order_eigvecs.txt')
        L_low_order_eigvecs_path = join(
            self.test_dir, 'dmd_L_low_order_eigvecs.txt')
        correlation_mat_eigvals_path = join(
            self.test_dir, 'dmd_corr_mat_eigvals.txt')
        correlation_mat_eigvecs_path = join(
            self.test_dir, 'dmd_corr_mat_eigvecs.txt')
        correlation_mat_path = join(self.test_dir, 'dmd_corr_mat.txt')
        cross_correlation_mat_path = join(
            self.test_dir, 'dmd_cross_corr_mat.txt')
        spectral_coeffs_path = join(self.test_dir, 'dmd_spectral_coeffs.txt')
        proj_coeffs_path = join(self.test_dir, 'dmd_proj_coeffs.txt')
        adv_proj_coeffs_path = join(self.test_dir, 'dmd_adv_proj_coeffs.txt')
        DMD_save.put_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            correlation_mat_eigvals_path , correlation_mat_eigvecs_path)
        DMD_save.put_correlation_mat(correlation_mat_path)
        DMD_save.put_cross_correlation_mat(cross_correlation_mat_path)
        DMD_save.put_spectral_coeffs(spectral_coeffs_path)
        DMD_save.put_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)
        parallel.barrier()

        # Create a new DMD object and use it to load data
        DMD_load = DMDHandles(None, verbosity=0)
        DMD_load.get_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            correlation_mat_eigvals_path, correlation_mat_eigvecs_path)
        DMD_load.get_correlation_mat(correlation_mat_path)
        DMD_load.get_cross_correlation_mat(cross_correlation_mat_path)
        DMD_load.get_spectral_coeffs(spectral_coeffs_path)
        DMD_load.get_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)

        # Check that the loaded data is correct
        np.testing.assert_equal(DMD_load.eigvals, eigvals)
        np.testing.assert_equal(
            DMD_load.L_low_order_eigvecs, L_low_order_eigvecs)
        np.testing.assert_equal(
            DMD_load.R_low_order_eigvecs, R_low_order_eigvecs)
        np.testing.assert_equal(
            DMD_load.correlation_mat_eigvals, correlation_mat_eigvals)
        np.testing.assert_equal(
            DMD_load.correlation_mat_eigvecs, correlation_mat_eigvecs)
        np.testing.assert_equal(DMD_load.correlation_mat, correlation_mat)
        np.testing.assert_equal(
            DMD_load.cross_correlation_mat, cross_correlation_mat)
        np.testing.assert_equal(
            np.array(DMD_load.spectral_coeffs).squeeze(), spectral_coeffs)
        np.testing.assert_equal(DMD_load.proj_coeffs, proj_coeffs)
        np.testing.assert_equal(DMD_load.adv_proj_coeffs, adv_proj_coeffs)


    #@unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test DMD decomposition"""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated DMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute DMD using modred
                DMD = DMDHandles(np.vdot, verbosity=0)
                (eigvals, R_low_order_eigvecs, L_low_order_eigvecs,
                correlation_mat_eigvals, correlation_mat_eigvecs) =\
                DMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Test correlation mats values by simply recomputing them.  Here
                # compute the full inner product matrix, rather than assuming it
                # is symmetric.
                np.testing.assert_allclose(
                    DMD.vec_space.compute_inner_product_mat(
                        vecs_vals, vecs_vals),
                    DMD.correlation_mat,
                    rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    DMD.vec_space.compute_inner_product_mat(
                        vecs_vals, adv_vecs_vals),
                    DMD.cross_correlation_mat,
                    rtol=rtol, atol=atol)

                # Test correlation mat eigenvalues and eigenvectors.
                np.testing.assert_allclose(
                    DMD.correlation_mat * correlation_mat_eigvecs,
                    correlation_mat_eigvecs * np.mat(np.diag(
                        correlation_mat_eigvals)),
                    rtol=rtol, atol=atol)

                # Compute the projection of the approximating linear operator
                # relating the vecs to the adv_vecs.  To do this, compute the
                # POD modes of the vecs using the eigendecomposition of the
                # correlation mat.
                POD_build_coeffs = (
                    correlation_mat_eigvecs *
                    np.mat(np.diag(correlation_mat_eigvals ** -0.5)))
                POD_mode_path = join(self.test_dir, 'pod_mode_%03d.txt')
                POD_mode_handles = [
                    VecHandlePickle(POD_mode_path % i)
                    for i in xrange(correlation_mat_eigvals.size)]
                DMD.vec_space.lin_combine(
                    POD_mode_handles, vecs_vals, POD_build_coeffs)
                low_order_linear_op = (
                    DMD.vec_space.compute_inner_product_mat(
                        POD_mode_handles, adv_vecs_vals) *
                    correlation_mat_eigvecs *
                    np.mat(np.diag(correlation_mat_eigvals ** -0.5)))

                # Test the left and right eigenvectors of the low-order
                # (projected) approximating linear operator.
                np.testing.assert_allclose(
                    low_order_linear_op * R_low_order_eigvecs,
                    R_low_order_eigvecs * np.mat(np.diag(eigvals)),
                    rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    L_low_order_eigvecs.H * low_order_linear_op,
                    np.mat(np.diag(eigvals)) * L_low_order_eigvecs.H,
                    rtol=rtol, atol=atol)

                # Check that returned values match internal values
                np.testing.assert_equal(eigvals, DMD.eigvals)
                np.testing.assert_equal(
                    R_low_order_eigvecs, DMD.R_low_order_eigvecs)
                np.testing.assert_equal(
                    L_low_order_eigvecs, DMD.L_low_order_eigvecs)
                np.testing.assert_equal(
                    correlation_mat_eigvals, DMD.correlation_mat_eigvals)
                np.testing.assert_equal(
                    correlation_mat_eigvecs, DMD.correlation_mat_eigvecs)

        # Check that if mismatched sets of handles are passed in, an error is
        # raised.
        DMD = DMDHandles(np.vdot, verbosity=0)
        self.assertRaises(
            ValueError, DMD.compute_decomp, self.vec_handles,
            self.adv_vec_handles[:-1])


    #@unittest.skip('Testing something else.')
    def test_compute_modes(self):
        """Test building of modes."""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated DMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute DMD using modred.  (The properties defining a DMD mode
                # require manipulations involving the correct decomposition, so
                # we cannot isolate the mode computation from the decomposition
                # step.
                DMD = DMDHandles(np.vdot, verbosity=0)
                DMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Compute the projection of the approximating linear operator
                # relating the projected vecs to the projected adv_vecs.  To do
                # this, compute the POD modes of the projected vecs using the
                # eigendecomposition of the projected correlation mat.
                POD_build_coeffs = (
                    DMD.correlation_mat_eigvecs *
                    np.mat(np.diag(DMD.correlation_mat_eigvals ** -0.5)))
                POD_mode_path = join(self.test_dir, 'pod_mode_%03d.txt')
                POD_mode_handles = [
                    VecHandlePickle(POD_mode_path % i)
                    for i in xrange(DMD.correlation_mat_eigvals.size)]
                DMD.vec_space.lin_combine(
                    POD_mode_handles, vecs_vals, POD_build_coeffs)

                # Select a subset of modes to compute.  Compute at least half
                # the modes, and up to all of them.  Make sure to use unique
                # values.  (This may reduce the number of modes computed.)
                num_modes = parallel.call_and_bcast(
                    np.random.randint,
                    DMD.eigvals.size // 2, DMD.eigvals.size + 1)
                mode_idxs = np.unique(parallel.call_and_bcast(
                    np.random.randint,
                    0, DMD.eigvals.size, num_modes))

                # Create handles for the modes
                DMD_exact_mode_handles = [
                    VecHandlePickle(self.exact_mode_path % i)
                    for i in mode_idxs]
                DMD_proj_mode_handles = [
                    VecHandlePickle(self.proj_mode_path % i)
                    for i in mode_idxs]
                DMD_adjoint_mode_handles = [
                    VecHandlePickle(self.adjoint_mode_path % i)
                    for i in mode_idxs]

                # Compute modes
                DMD.compute_exact_modes(mode_idxs, DMD_exact_mode_handles)
                DMD.compute_proj_modes(mode_idxs, DMD_proj_mode_handles)
                DMD.compute_adjoint_modes(mode_idxs, DMD_adjoint_mode_handles)

                # Test that exact modes are eigenvectors of the approximating
                # linear operator by checking A \Phi = \Phi \Lambda.  Do this
                # using handles, i.e. check mode by mode.  Note that since
                # np.vdot takes the conjugate of its second argument, whereas
                # modred assumes a conjugate is taken on the first inner product
                # argument, the inner product matrix in the LHS computation must
                # be conjugated.
                LHS_path = join(self.test_dir, 'LHS_%03d.pkl')
                LHS_handles = [
                    VecHandlePickle(LHS_path % i) for i in mode_idxs]
                RHS_path = join(self.test_dir, 'RHS_%03d.pkl')
                RHS_handles = [
                    VecHandlePickle(RHS_path % i) for i in mode_idxs]
                DMD.vec_space.lin_combine(
                    LHS_handles,
                    adv_vecs_vals,
                    DMD.correlation_mat_eigvecs *
                    np.mat(np.diag(DMD.correlation_mat_eigvals ** -0.5)) *
                    DMD.vec_space.compute_inner_product_mat(
                        POD_mode_handles, DMD_exact_mode_handles))
                DMD.vec_space.lin_combine(
                    RHS_handles,
                    DMD_exact_mode_handles,
                    np.mat(np.diag(DMD.eigvals[mode_idxs])))
                for LHS, RHS in zip(LHS_handles, RHS_handles):
                    np.testing.assert_allclose(
                        LHS.get(), RHS.get(), rtol=rtol, atol=atol)

                # Test that projected modes are eigenvectors of the projection
                # of the approximating linear operator by checking
                # U U^* A \Phi = \Phi \Lambda.  As above, check this using
                # handles, and be careful about the order of arguments when
                # taking inner products.
                LHS_path = join(self.test_dir, 'LHS_%03d.pkl')
                LHS_handles = [
                    VecHandlePickle(LHS_path % i) for i in mode_idxs]
                RHS_path = join(self.test_dir, 'RHS_%03d.pkl')
                RHS_handles = [
                    VecHandlePickle(RHS_path % i) for i in mode_idxs]
                DMD.vec_space.lin_combine(
                    LHS_handles,
                    POD_mode_handles,
                    DMD.vec_space.compute_inner_product_mat(
                        POD_mode_handles, adv_vecs_vals) *
                    DMD.correlation_mat_eigvecs *
                    np.mat(np.diag(DMD.correlation_mat_eigvals ** -0.5)) *
                    DMD.vec_space.compute_inner_product_mat(
                        POD_mode_handles, DMD_proj_mode_handles))
                DMD.vec_space.lin_combine(
                    RHS_handles,
                    DMD_proj_mode_handles,
                    np.mat(np.diag(DMD.eigvals[mode_idxs])))
                for LHS, RHS in zip(LHS_handles, RHS_handles):
                    np.testing.assert_allclose(
                        LHS.get(), RHS.get(), rtol=rtol, atol=atol)

                # Test that adjoint modes are eigenvectors of the conjugate
                # transpose of approximating linear operator by checking
                # A^* \Phi = \Phi \Lambda^*.  Do this using handles, i.e. check
                # mode by mode.  Note that since np.vdot takes the conjugate of
                # its second argument, whereas modred assumes a conjugate is
                # taken on the first inner product argument, the inner product
                # matrix in the LHS computation must be conjugated.
                LHS_path = join(self.test_dir, 'LHS_%03d.pkl')
                LHS_handles = [
                    VecHandlePickle(LHS_path % i) for i in mode_idxs]
                RHS_path = join(self.test_dir, 'RHS_%03d.pkl')
                RHS_handles = [
                    VecHandlePickle(RHS_path % i) for i in mode_idxs]
                DMD.vec_space.lin_combine(
                    LHS_handles,
                    POD_mode_handles,
                    np.mat(np.diag(DMD.correlation_mat_eigvals ** -0.5)) *
                    DMD.correlation_mat_eigvecs.T *
                    DMD.vec_space.compute_inner_product_mat(
                        adv_vecs_vals, DMD_adjoint_mode_handles))
                DMD.vec_space.lin_combine(
                    RHS_handles,
                    DMD_adjoint_mode_handles,
                    np.mat(np.diag(DMD.eigvals[mode_idxs])).H)
                for LHS, RHS in zip(LHS_handles, RHS_handles):
                    np.testing.assert_allclose(
                        LHS.get(), RHS.get(), rtol=rtol, atol=atol)


    #@unittest.skip('Testing something else.')
    def test_compute_spectrum(self):
        """Test DMD spectrum"""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated DMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute DMD using modred.  (The DMD spectral coefficients are
                # defined by a projection onto DMD modes.  As such, testing them
                # requires manipulations involving the correct decomposition and
                # modes, so we cannot isolate the spectral coefficient
                # computation from those computations.)
                DMD = DMDHandles(np.vdot, verbosity=0)
                DMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Test by checking a least-squares projection onto the projected
                # modes, which is analytically equivalent to a biorthogonal
                # projection onto the exact modes.  The latter is implemented
                # (using various identities) in modred.  Here, test using the
                # former approach, as it doesn't require adjoint modes.
                mode_idxs = range(DMD.eigvals.size)
                proj_mode_handles = [
                    V.VecHandlePickle(self.proj_mode_path % i)
                    for i in mode_idxs]
                DMD.compute_proj_modes(mode_idxs, proj_mode_handles)
                spectral_coeffs_true = np.array(np.abs(
                    np.linalg.inv(
                        DMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    DMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, vecs_vals[0]))).squeeze()
                spectral_coeffs = DMD.compute_spectrum()
                np.testing.assert_allclose(
                    spectral_coeffs, spectral_coeffs_true, rtol=rtol, atol=atol)


    #@unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        """Test projection coefficients"""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated DMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute DMD using modred.  (Testing the DMD projection
                # coefficients requires the correct DMD decomposition and modes,
                # so we cannot isolate the projection coefficient computation
                # from those computations.)
                DMD = DMDHandles(np.vdot, verbosity=0)
                DMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Test by checking a least-squares projection onto the projected
                # modes, which is analytically equivalent to a biorthogonal
                # projection onto the exact modes.  The latter is implemented
                # (using various identities) in modred.  Here, test using the
                # former approach, as it doesn't require adjoint modes.
                mode_idxs = range(DMD.eigvals.size)
                proj_mode_handles = [
                    V.VecHandlePickle(self.proj_mode_path % i)
                    for i in mode_idxs]
                DMD.compute_proj_modes(mode_idxs, proj_mode_handles)
                proj_coeffs_true = (
                    np.linalg.inv(
                        DMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    DMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, vecs_vals))
                adv_proj_coeffs_true = (
                    np.linalg.inv(
                        DMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    DMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, adv_vecs_vals))
                proj_coeffs, adv_proj_coeffs = DMD.compute_proj_coeffs()
                np.testing.assert_allclose(
                    proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    adv_proj_coeffs, adv_proj_coeffs_true, rtol=rtol, atol=atol)


#@unittest.skip('Testing something else.')
@unittest.skipIf(parallel.is_distributed(), 'Serial only.')
class TestTLSqrDMDArraysFunctions(unittest.TestCase):
    def setUp(self):
        # Generate vecs if we are on the first processor
        # A random matrix of data (#cols = #vecs)
        self.num_vecs = 30
        self.num_states = 10
        self.max_num_eigvals = int(np.round(self.num_states / 2))


    def test_all(self):
        rtol = 1e-10
        atol = 1e-12

        # Generate weights to test different inner products.
        ws = np.identity(self.num_states)
        ws[0, 0] = 2.
        ws[2, 1] = 0.3
        ws[1, 2] = 0.3
        weights_list = [None, np.random.random(self.num_states), ws]

        # Generate random snapshot data
        vecs_mat = np.mat(np.random.random((self.num_states, self.num_vecs)))
        adv_vecs_mat = np.mat(np.random.random(
            (self.num_states, self.num_vecs)))

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to
        # a sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [vecs_mat, vecs_mat],
            [None, adv_vecs_mat],
            [vecs_mat[:, :-1], vecs_mat],
            [vecs_mat[:, 1:], adv_vecs_mat]):

            # Stack the data matrices, for doing total-least squares
            stacked_vecs_mat = np.mat(np.vstack((vecs_vals, adv_vecs_vals)))

            # Test both method of snapshots and direct method
            for method in ['snaps', 'direct']:

                # Consider different inner product weights
                for weights in weights_list:
                    IP = VectorSpaceMatrices(
                        weights=weights).compute_inner_product_mat
                    symmetric_IP = VectorSpaceMatrices(
                        weights=weights).compute_symmetric_inner_product_mat

                    # Define inner product for stacked vectors
                    if weights is None:
                        stacked_weights = None
                    elif len(weights.shape) == 1:
                        stacked_weights = np.hstack((weights, weights))
                    elif len(weights.shape) == 2:
                        stacked_weights = np.vstack((
                            np.hstack((weights, 0. * weights)),
                            np.hstack((0. * weights, weights))))
                    else:
                        raise ValueError('Invalid inner product weights.')
                    stacked_IP = VectorSpaceMatrices(
                        weights=stacked_weights).compute_inner_product_mat

                    # Test that results hold for truncated or untruncated DMD
                    # (i.e., whether or not the underlying POD basis is
                    # truncated).
                    for max_num_eigvals in [None, self.num_states // 2]:

                        # Choose subset of modes to compute, for testing mode
                        # indices argument
                        if max_num_eigvals is None:
                            mode_indices = np.unique(np.random.randint(
                                0, high=np.linalg.matrix_rank(vecs_vals),
                                size=np.linalg.matrix_rank(vecs_vals) // 2))
                        else:
                            mode_indices = np.unique(np.random.randint(
                                0, high=max_num_eigvals,
                                size=max_num_eigvals // 2))

                        # Compute TLSqrDMD using appropriate method.  Then
                        # compute transformation of raw data for TLSqrDMD using
                        # POD of the stacked vecs, and transform the data.  Also
                        # compute a random subset of the TLSqrDMD modes, in
                        # preparation for later tests.
                        if method == 'snaps':
                            DMD_res = compute_TLSqrDMD_matrices_snaps_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals)
                            DMD_res_sliced =\
                                compute_TLSqrDMD_matrices_snaps_method(
                                    vecs_arg, adv_vecs=adv_vecs_arg,
                                    mode_indices=mode_indices,
                                    inner_product_weights=weights,
                                    max_num_eigvals=max_num_eigvals)

                            # For method of snapshots, test correlation mats
                            # values by simply recomputing them.
                            np.testing.assert_allclose(
                                IP(vecs_vals, vecs_vals),
                                DMD_res.correlation_mat,
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                IP(vecs_vals, adv_vecs_vals),
                                DMD_res.cross_correlation_mat,
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                IP(adv_vecs_vals, adv_vecs_vals),
                                DMD_res.adv_correlation_mat,
                                rtol=rtol, atol=atol)

                        elif method == 'direct':
                            DMD_res = compute_TLSqrDMD_matrices_direct_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals)
                            DMD_res_sliced =\
                                compute_TLSqrDMD_matrices_direct_method(
                                    vecs_arg, adv_vecs=adv_vecs_arg,
                                    mode_indices=mode_indices,
                                    inner_product_weights=weights,
                                    max_num_eigvals=max_num_eigvals)

                        else:
                            raise ValueError('Invalid DMD method.')

                        # Test summed correlation mat eigenvalues and
                        # eigenvectors
                        np.testing.assert_allclose((
                            IP(vecs_vals, vecs_vals) +
                            IP(adv_vecs_vals, adv_vecs_vals)) *\
                            DMD_res.sum_correlation_mat_eigvecs,
                            DMD_res.sum_correlation_mat_eigvecs *\
                            np.mat(np.diag(
                                DMD_res.sum_correlation_mat_eigvals)),
                            rtol=rtol, atol=atol)

                        # Test projected correlation mat eigenvalues and
                        # eigenvectors
                        proj_vecs_vals = (
                            vecs_vals *
                            DMD_res.sum_correlation_mat_eigvecs *
                            DMD_res.sum_correlation_mat_eigvecs.H)
                        proj_adv_vecs_vals = (
                            adv_vecs_vals *
                            DMD_res.sum_correlation_mat_eigvecs *
                            DMD_res.sum_correlation_mat_eigvecs.H)
                        np.testing.assert_allclose(
                            IP(proj_vecs_vals, proj_vecs_vals) *\
                            DMD_res.proj_correlation_mat_eigvecs,
                            DMD_res.proj_correlation_mat_eigvecs *\
                            np.mat(np.diag(
                                DMD_res.proj_correlation_mat_eigvals)),
                            rtol=rtol, atol=atol)

                        # Compute the approximating linear operator relating the
                        # projected vecs to the projected adv_vecs.  To do this,
                        # compute the POD of the projected vecs using the
                        # eigendecomposition of the projected correlation mat.
                        proj_vecs_POD_build_coeffs = (
                            DMD_res.proj_correlation_mat_eigvecs *
                            np.mat(np.diag(
                                DMD_res.proj_correlation_mat_eigvals ** -0.5)))
                        proj_vecs_POD_modes = (
                            proj_vecs_vals * proj_vecs_POD_build_coeffs)
                        approx_linear_op = (
                            proj_adv_vecs_vals *
                            DMD_res.proj_correlation_mat_eigvecs *
                            np.mat(np.diag(
                                DMD_res.proj_correlation_mat_eigvals ** -0.5)) *
                            proj_vecs_POD_modes.H)
                        low_order_linear_op = IP(
                            proj_vecs_POD_modes,
                            IP(approx_linear_op.H, proj_vecs_POD_modes))

                        # Test the left and right eigenvectors of the low-order
                        # (projected) approximating linear operator.
                        np.testing.assert_allclose(
                            low_order_linear_op * DMD_res.R_low_order_eigvecs,
                            DMD_res.R_low_order_eigvecs * np.mat(np.diag(
                                DMD_res.eigvals)),
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            DMD_res.L_low_order_eigvecs.H * low_order_linear_op,
                            np.mat(np.diag(DMD_res.eigvals)) *\
                            DMD_res.L_low_order_eigvecs.H,
                            rtol=rtol, atol=atol)

                        # Test the exact modes, which are eigenvectors of the
                        # approximating linear operator.
                        np.testing.assert_allclose(
                            IP(approx_linear_op.H, DMD_res.exact_modes),
                            DMD_res.exact_modes * np.mat(np.diag(
                                DMD_res.eigvals)),
                            rtol=rtol, atol=atol)

                        # Test the projected modes, which are eigenvectors of
                        # the approximating linear operator projected onto the
                        # POD modes of the vecs.
                        np.testing.assert_allclose(
                            proj_vecs_POD_modes * IP(
                                proj_vecs_POD_modes,
                                IP(approx_linear_op.H, DMD_res.proj_modes)),
                            DMD_res.proj_modes * np.mat(np.diag(
                                DMD_res.eigvals)),
                            rtol=rtol, atol=atol)

                        # Test the adjoint modes, which are left eigenvectors of
                        # the approximating linear operator.
                        np.testing.assert_allclose(
                            IP(approx_linear_op, DMD_res.adjoint_modes),
                            DMD_res.adjoint_modes * np.mat(np.diag(
                                DMD_res.eigvals.conj().T)),
                            rtol=rtol, atol=atol)

                        # Test spectral coefficients against an explicit
                        # projection using the adjoint DMD modes.
                        np.testing.assert_allclose(
                            np.array(np.abs(IP(
                                DMD_res.adjoint_modes,
                                proj_vecs_vals[:, 0]))).squeeze(),
                            DMD_res.spectral_coeffs,
                            rtol=rtol, atol=atol)

                        # Test that use of mode indices argument returns correct
                        # subset of modes
                        np.testing.assert_allclose(
                            DMD_res_sliced.exact_modes,
                            DMD_res.exact_modes[:, mode_indices],
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            DMD_res_sliced.proj_modes,
                            DMD_res.proj_modes[:, mode_indices],
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            DMD_res_sliced.adjoint_modes,
                            DMD_res.adjoint_modes[:, mode_indices],
                            rtol=rtol, atol=atol)


#@unittest.skip('Testing something else.')
class TestTLSqrDMDHandles(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'TLSqrDMD_files'
        if not os.path.isdir(self.test_dir):
            parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        self.vec_path = join(self.test_dir, 'tlsqrdmd_vec_%03d.pkl')
        self.adv_vec_path = join(self.test_dir, 'tlsqrdmd_adv_vec_%03d.pkl')
        self.exact_mode_path = join(
            self.test_dir, 'tlsqrdmd_exactmode_%03d.pkl')
        self.proj_mode_path = join(self.test_dir, 'tlsqrdmd_projmode_%03d.pkl')
        self.adjoint_mode_path = join(
            self.test_dir, 'tlsqrdmd_adjmode_%03d.pkl')

        # Specify data dimensions
        self.num_states = 30
        self.num_vecs = 10

        # Generate random data and write to disk using handles
        self.vecs_array = parallel.call_and_bcast(
            np.random.random, (self.num_states, self.num_vecs))
        self.adv_vecs_array = parallel.call_and_bcast(
            np.random.random, (self.num_states, self.num_vecs))
        self.vec_handles = [
            VecHandlePickle(self.vec_path % i) for i in range(self.num_vecs)]
        self.adv_vec_handles = [
            VecHandlePickle(self.adv_vec_path % i)
            for i in range(self.num_vecs)]
        for idx, (hdl, adv_hdl) in enumerate(
            zip(self.vec_handles, self.adv_vec_handles)):
            hdl.put(self.vecs_array[:, idx])
            adv_hdl.put(self.adv_vecs_array[:, idx])

        # Path for saving projected (de-noised) vecs and advanced vecs to disk
        # later.
        self.proj_vec_path = join(
            self.test_dir, 'tlsqrdmd_proj_vec_%03d.pkl')
        self.proj_adv_vec_path = join(
            self.test_dir, 'tlsqrdmd_proj_adv_vec_%03d.pkl')

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
        def my_load(fname): pass
        def my_save(data, fname): pass
        def my_IP(vec1, vec2): pass

        data_members_default = {
            'put_mat': util.save_array_text, 'get_mat': util.load_array_text,
            'verbosity': 0, 'eigvals': None, 'correlation_mat': None,
            'cross_correlation_mat': None, 'adv_correlation_mat': None,
            'sum_correlation_mat': None, 'proj_correlation_mat': None,
            'sum_correlation_mat_eigvals': None,
            'sum_correlation_mat_eigvecs': None,
            'proj_correlation_mat_eigvals': None,
            'proj_correlation_mat_eigvecs': None, 'low_order_linear_map': None,
            'L_low_order_eigvecs': None, 'R_low_order_eigvecs': None,
            'spectral_coeffs': None, 'proj_coeffs': None, 'adv_proj_coeffs':
            None, 'vec_handles': None, 'adv_vec_handles': None, 'vec_space':
            VectorSpaceHandles(my_IP, verbosity=0)}

        # Get default data member values
        for k,v in util.get_data_members(
            TLSqrDMDHandles(my_IP, verbosity=0)).items():
            self.assertEqual(v, data_members_default[k])

        my_DMD = TLSqrDMDHandles(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpaceHandles(
            inner_product=my_IP, verbosity=0)
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])

        my_DMD = TLSqrDMDHandles(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])

        my_DMD = TLSqrDMDHandles(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])

        max_vecs_per_node = 500
        my_DMD = TLSqrDMDHandles(my_IP, max_vecs_per_node=max_vecs_per_node,
            verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * parallel.get_num_nodes() / \
            parallel.get_num_procs()
        for k,v in util.get_data_members(my_DMD).items():
            self.assertEqual(v, data_members_modified[k])


    #@unittest.skip('Testing something else.')
    def test_puts_gets(self):
        """Test get and put functions"""
        # Generate some random data
        eigvals = parallel.call_and_bcast(np.random.random, 5)
        R_low_order_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        L_low_order_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        sum_correlation_mat_eigvals = parallel.call_and_bcast(
            np.random.random, 5)
        sum_correlation_mat_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        proj_correlation_mat_eigvals = parallel.call_and_bcast(
            np.random.random, 5)
        proj_correlation_mat_eigvecs = parallel.call_and_bcast(
            np.random.random, (10,10))
        correlation_mat = parallel.call_and_bcast(np.random.random, (10,10))
        cross_correlation_mat = parallel.call_and_bcast(
            np.random.random, (10,10))
        adv_correlation_mat = parallel.call_and_bcast(
            np.random.random, (10,10))
        sum_correlation_mat = parallel.call_and_bcast(
            np.random.random, (10,10))
        proj_correlation_mat = parallel.call_and_bcast(
            np.random.random, (10,10))
        spectral_coeffs = parallel.call_and_bcast(np.random.random, 5)
        proj_coeffs = parallel.call_and_bcast(np.random.random, (5,5))
        adv_proj_coeffs = parallel.call_and_bcast(np.random.random, (5,5))

        # Create a DMD object and store the data in it
        TLSqrDMD_save = TLSqrDMDHandles(None, verbosity=0)
        TLSqrDMD_save.eigvals = eigvals
        TLSqrDMD_save.R_low_order_eigvecs = R_low_order_eigvecs
        TLSqrDMD_save.L_low_order_eigvecs = L_low_order_eigvecs
        TLSqrDMD_save.sum_correlation_mat_eigvals =\
            sum_correlation_mat_eigvals
        TLSqrDMD_save.sum_correlation_mat_eigvecs =\
            sum_correlation_mat_eigvecs
        TLSqrDMD_save.proj_correlation_mat_eigvals =\
            proj_correlation_mat_eigvals
        TLSqrDMD_save.proj_correlation_mat_eigvecs =\
            proj_correlation_mat_eigvecs
        TLSqrDMD_save.correlation_mat = correlation_mat
        TLSqrDMD_save.cross_correlation_mat = cross_correlation_mat
        TLSqrDMD_save.adv_correlation_mat = adv_correlation_mat
        TLSqrDMD_save.sum_correlation_mat = sum_correlation_mat
        TLSqrDMD_save.proj_correlation_mat = proj_correlation_mat
        TLSqrDMD_save.spectral_coeffs = spectral_coeffs
        TLSqrDMD_save.proj_coeffs = proj_coeffs
        TLSqrDMD_save.adv_proj_coeffs = adv_proj_coeffs

        # Write the data to disk
        eigvals_path = join(self.test_dir, 'tlsqrdmd_eigvals.txt')
        R_low_order_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_R_low_order_eigvecs.txt')
        L_low_order_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_L_low_order_eigvecs.txt')
        sum_correlation_mat_eigvals_path = join(
            self.test_dir, 'tlsqrdmd_sum_corr_mat_eigvals.txt')
        sum_correlation_mat_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_sum_corr_mat_eigvecs.txt')
        proj_correlation_mat_eigvals_path = join(
            self.test_dir, 'tlsqrdmd_proj_corr_mat_eigvals.txt')
        proj_correlation_mat_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_proj_corr_mat_eigvecs.txt')
        correlation_mat_path = join(self.test_dir, 'tlsqrdmd_corr_mat.txt')
        cross_correlation_mat_path = join(
            self.test_dir, 'tlsqrdmd_cross_corr_mat.txt')
        adv_correlation_mat_path = join(
            self.test_dir, 'tlsqrdmd_adv_corr_mat.txt')
        sum_correlation_mat_path = join(
            self.test_dir, 'tlsqrdmd_sum_corr_mat.txt')
        proj_correlation_mat_path = join(
            self.test_dir, 'tlsqrdmd_proj_corr_mat.txt')
        spectral_coeffs_path = join(
            self.test_dir, 'tlsqrdmd_spectral_coeffs.txt')
        proj_coeffs_path = join(self.test_dir, 'tlsqrdmd_proj_coeffs.txt')
        adv_proj_coeffs_path = join(
            self.test_dir, 'tlsqrdmd_adv_proj_coeffs.txt')
        TLSqrDMD_save.put_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            sum_correlation_mat_eigvals_path,
            sum_correlation_mat_eigvecs_path,
            proj_correlation_mat_eigvals_path ,
            proj_correlation_mat_eigvecs_path)
        TLSqrDMD_save.put_correlation_mat(correlation_mat_path)
        TLSqrDMD_save.put_cross_correlation_mat(cross_correlation_mat_path)
        TLSqrDMD_save.put_adv_correlation_mat(adv_correlation_mat_path)
        TLSqrDMD_save.put_sum_correlation_mat(sum_correlation_mat_path)
        TLSqrDMD_save.put_proj_correlation_mat(proj_correlation_mat_path)
        TLSqrDMD_save.put_spectral_coeffs(spectral_coeffs_path)
        TLSqrDMD_save.put_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)
        parallel.barrier()

        # Create a new TLSqrDMD object and use it to load data
        TLSqrDMD_load = TLSqrDMDHandles(None, verbosity=0)
        TLSqrDMD_load.get_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            sum_correlation_mat_eigvals_path,
            sum_correlation_mat_eigvecs_path,
            proj_correlation_mat_eigvals_path,
            proj_correlation_mat_eigvecs_path)
        TLSqrDMD_load.get_correlation_mat(correlation_mat_path)
        TLSqrDMD_load.get_cross_correlation_mat(cross_correlation_mat_path)
        TLSqrDMD_load.get_adv_correlation_mat(adv_correlation_mat_path)
        TLSqrDMD_load.get_sum_correlation_mat(sum_correlation_mat_path)
        TLSqrDMD_load.get_proj_correlation_mat(proj_correlation_mat_path)
        TLSqrDMD_load.get_spectral_coeffs(spectral_coeffs_path)
        TLSqrDMD_load.get_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)

        # Check that the loaded data is correct
        np.testing.assert_allclose(TLSqrDMD_load.eigvals, eigvals)
        np.testing.assert_allclose(
            TLSqrDMD_load.R_low_order_eigvecs, R_low_order_eigvecs)
        np.testing.assert_allclose(
            TLSqrDMD_load.L_low_order_eigvecs, L_low_order_eigvecs)
        np.testing.assert_allclose(
            TLSqrDMD_load.sum_correlation_mat_eigvals,
            sum_correlation_mat_eigvals)
        np.testing.assert_allclose(
            TLSqrDMD_load.sum_correlation_mat_eigvecs,
            sum_correlation_mat_eigvecs)
        np.testing.assert_allclose(
            TLSqrDMD_load.proj_correlation_mat_eigvals,
            proj_correlation_mat_eigvals)
        np.testing.assert_allclose(
            TLSqrDMD_load.proj_correlation_mat_eigvecs,
            proj_correlation_mat_eigvecs)
        np.testing.assert_allclose(
            TLSqrDMD_load.correlation_mat, correlation_mat)
        np.testing.assert_allclose(
            TLSqrDMD_load.cross_correlation_mat, cross_correlation_mat)
        np.testing.assert_allclose(
            TLSqrDMD_load.adv_correlation_mat, adv_correlation_mat)
        np.testing.assert_allclose(
            TLSqrDMD_load.sum_correlation_mat, sum_correlation_mat)
        np.testing.assert_allclose(
            TLSqrDMD_load.proj_correlation_mat, proj_correlation_mat)
        np.testing.assert_allclose(
            np.array(TLSqrDMD_load.spectral_coeffs).squeeze(), spectral_coeffs)
        np.testing.assert_allclose(
            TLSqrDMD_load.proj_coeffs, proj_coeffs)
        np.testing.assert_allclose(
            TLSqrDMD_load.adv_proj_coeffs, adv_proj_coeffs)


    #@unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test TLSqrDMD decomposition"""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated TLSqrDMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute DMD using modred
                TLSqrDMD = TLSqrDMDHandles(np.vdot, verbosity=0)
                (eigvals, R_low_order_eigvecs, L_low_order_eigvecs,
                sum_correlation_mat_eigvals,
                sum_correlation_mat_eigvecs,
                proj_correlation_mat_eigvals, proj_correlation_mat_eigvecs) =\
                    TLSqrDMD.compute_decomp(
                        vecs_arg, adv_vec_handles=adv_vecs_arg,
                        max_num_eigvals=max_num_eigvals)

                # Test correlation mats values by simply recomputing them.  Here
                # compute the full inner product matrix, rather than assuming it
                # is symmetric.
                np.testing.assert_allclose(
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        vecs_vals, vecs_vals),
                    TLSqrDMD.correlation_mat,
                    rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        vecs_vals, adv_vecs_vals),
                    TLSqrDMD.cross_correlation_mat,
                    rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        adv_vecs_vals, adv_vecs_vals),
                    TLSqrDMD.adv_correlation_mat,
                    rtol=rtol, atol=atol)

                # Test sum correlation mat values by adding together correlation
                # mats.
                np.testing.assert_allclose(
                    TLSqrDMD.correlation_mat +
                    TLSqrDMD.adv_correlation_mat,
                    TLSqrDMD.sum_correlation_mat,
                    rtol=rtol, atol=atol)

                # Test sum correlation mat eigenvalues and eigenvectors.
                np.testing.assert_allclose((
                    TLSqrDMD.sum_correlation_mat *
                    sum_correlation_mat_eigvecs),
                    sum_correlation_mat_eigvecs * np.mat(np.diag(
                        sum_correlation_mat_eigvals)),
                    rtol=rtol, atol=atol)

                # Test projected correlation mat values by projecting the raw
                # data, saving them to disk using handles, and then
                # computing the correlation matrix of the projected vectors.
                proj_mat = (
                    TLSqrDMD.sum_correlation_mat_eigvecs *
                    TLSqrDMD.sum_correlation_mat_eigvecs.H)
                proj_vec_path = join(
                    self.test_dir, 'tlsqrdmd_proj_vec_%03d.pkl')
                proj_vecs_handles = [
                    VecHandlePickle(proj_vec_path % i)
                    for i in range(len(vecs_vals))]
                TLSqrDMD.vec_space.lin_combine(
                    proj_vecs_handles, vecs_vals, proj_mat)
                np.testing.assert_allclose(
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_vecs_handles, proj_vecs_handles),
                    TLSqrDMD.proj_correlation_mat,
                    rtol=rtol, atol=atol)

                # Test projected correlation mat eigenvalues and eigenvectors.
                np.testing.assert_allclose((
                    TLSqrDMD.proj_correlation_mat *
                    proj_correlation_mat_eigvecs),
                    proj_correlation_mat_eigvecs * np.mat(np.diag(
                        proj_correlation_mat_eigvals)),
                    rtol=rtol, atol=atol)

                # Compute the projection of the approximating linear operator
                # relating the projected vecs to the projected adv_vecs.  To do
                # this, compute the POD modes of the projected vecs using the
                # eigendecomposition of the projected correlation mat.
                proj_POD_build_coeffs = (
                    proj_correlation_mat_eigvecs *
                    np.mat(np.diag(proj_correlation_mat_eigvals ** -0.5)))
                proj_POD_mode_path = join(
                    self.test_dir, 'proj_pod_mode_%03d.txt')
                proj_POD_mode_handles = [
                    VecHandlePickle(proj_POD_mode_path % i)
                    for i in xrange(proj_correlation_mat_eigvals.size)]
                TLSqrDMD.vec_space.lin_combine(
                    proj_POD_mode_handles, vecs_vals, proj_POD_build_coeffs)
                low_order_linear_op = (
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_POD_mode_handles, adv_vecs_vals) *
                    proj_correlation_mat_eigvecs *
                    np.mat(np.diag(proj_correlation_mat_eigvals ** -0.5)))

                # Test the left and right eigenvectors of the low-order
                # (projected) approximating linear operator.
                np.testing.assert_allclose(
                    low_order_linear_op * R_low_order_eigvecs,
                    R_low_order_eigvecs * np.mat(np.diag(eigvals)),
                    rtol=rtol, atol=atol)
                np.testing.assert_allclose(
                    L_low_order_eigvecs.H * low_order_linear_op,
                    np.mat(np.diag(eigvals)) * L_low_order_eigvecs.H,
                    rtol=rtol, atol=atol)

                # Check that returned values match internal values
                np.testing.assert_equal(eigvals, TLSqrDMD.eigvals)
                np.testing.assert_equal(
                    R_low_order_eigvecs, TLSqrDMD.R_low_order_eigvecs)
                np.testing.assert_equal(
                    L_low_order_eigvecs, TLSqrDMD.L_low_order_eigvecs)
                np.testing.assert_equal(
                    sum_correlation_mat_eigvals,
                    TLSqrDMD.sum_correlation_mat_eigvals)
                np.testing.assert_equal(
                    sum_correlation_mat_eigvecs,
                    TLSqrDMD.sum_correlation_mat_eigvecs)
                np.testing.assert_equal(
                    proj_correlation_mat_eigvals,
                    TLSqrDMD.proj_correlation_mat_eigvals)
                np.testing.assert_equal(
                    proj_correlation_mat_eigvecs,
                    TLSqrDMD.proj_correlation_mat_eigvecs)

        # Check that if mismatched sets of handles are passed in, an error is
        # raised.
        TLSqrDMD = TLSqrDMDHandles(np.vdot, verbosity=0)
        self.assertRaises(
            ValueError, TLSqrDMD.compute_decomp, self.vec_handles,
            self.adv_vec_handles[:-1])


    #@unittest.skip('Testing something else.')
    def test_compute_modes(self):
        """Test building of modes."""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated TLSqrDMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute TLSqrDMD using modred.  (The properties defining a
                # TLSqrDMD mode require manipulations involving the correct
                # decomposition, so we cannot isolate the mode computation from
                # the decomposition step.)
                TLSqrDMD = TLSqrDMDHandles(np.vdot, verbosity=0)
                TLSqrDMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Compute the projection of the approximating linear operator
                # relating the projected vecs to the projected adv_vecs.  To do
                # this, compute the POD modes of the projected vecs using the
                # eigendecomposition of the projected correlation mat.
                proj_POD_build_coeffs = (
                    TLSqrDMD.proj_correlation_mat_eigvecs *
                    np.mat(np.diag(
                        TLSqrDMD.proj_correlation_mat_eigvals ** -0.5)))
                proj_POD_mode_path = join(
                    self.test_dir, 'proj_pod_mode_%03d.txt')
                proj_POD_mode_handles = [
                    VecHandlePickle(proj_POD_mode_path % i)
                    for i in xrange(TLSqrDMD.proj_correlation_mat_eigvals.size)]
                TLSqrDMD.vec_space.lin_combine(
                    proj_POD_mode_handles, vecs_vals, proj_POD_build_coeffs)

                # Select a subset of modes to compute.  Compute at least half
                # the modes, and up to all of them.  Make sure to use unique
                # values.  (This may reduce the number of modes computed.)
                num_modes = parallel.call_and_bcast(
                    np.random.randint,
                    TLSqrDMD.eigvals.size // 2, TLSqrDMD.eigvals.size + 1)
                mode_idxs = np.unique(parallel.call_and_bcast(
                    np.random.randint,
                    0, TLSqrDMD.eigvals.size, num_modes))

                # Create handles for the modes
                TLSqrDMD_exact_mode_handles = [
                    VecHandlePickle(self.exact_mode_path % i)
                    for i in mode_idxs]
                TLSqrDMD_proj_mode_handles = [
                    VecHandlePickle(self.proj_mode_path % i)
                    for i in mode_idxs]
                TLSqrDMD_adjoint_mode_handles = [
                    VecHandlePickle(self.adjoint_mode_path % i)
                    for i in mode_idxs]

                # Compute modes
                TLSqrDMD.compute_exact_modes(
                    mode_idxs, TLSqrDMD_exact_mode_handles)
                TLSqrDMD.compute_proj_modes(
                    mode_idxs, TLSqrDMD_proj_mode_handles)
                TLSqrDMD.compute_adjoint_modes(
                    mode_idxs, TLSqrDMD_adjoint_mode_handles)

                # Test that exact modes are eigenvectors of the approximating
                # linear operator by checking A \Phi = \Phi \Lambda.  Do this
                # using handles, i.e. check mode by mode.  Note that since
                # np.vdot takes the conjugate of its second argument, whereas
                # modred assumes a conjugate is taken on the first inner product
                # argument, the inner product matrix in the LHS computation must
                # be conjugated.
                LHS_path = join(self.test_dir, 'LHS_%03d.pkl')
                LHS_handles = [
                    VecHandlePickle(LHS_path % i) for i in mode_idxs]
                RHS_path = join(self.test_dir, 'RHS_%03d.pkl')
                RHS_handles = [
                    VecHandlePickle(RHS_path % i) for i in mode_idxs]
                TLSqrDMD.vec_space.lin_combine(
                    LHS_handles,
                    adv_vecs_vals,
                    TLSqrDMD.proj_correlation_mat_eigvecs *
                    np.mat(np.diag(
                        TLSqrDMD.proj_correlation_mat_eigvals ** -0.5)) *
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_POD_mode_handles, TLSqrDMD_exact_mode_handles))
                TLSqrDMD.vec_space.lin_combine(
                    RHS_handles,
                    TLSqrDMD_exact_mode_handles,
                    np.mat(np.diag(TLSqrDMD.eigvals[mode_idxs])))
                for LHS, RHS in zip(LHS_handles, RHS_handles):
                    np.testing.assert_allclose(
                        LHS.get(), RHS.get(), rtol=rtol, atol=atol)

                # Test that projected modes are eigenvectors of the projection
                # of the approximating linear operator by checking
                # U U^* A \Phi = \Phi \Lambda.  As above, check this using
                # handles, and be careful about the order of arguments when
                # taking inner products.
                LHS_path = join(self.test_dir, 'LHS_%03d.pkl')
                LHS_handles = [
                    VecHandlePickle(LHS_path % i) for i in mode_idxs]
                RHS_path = join(self.test_dir, 'RHS_%03d.pkl')
                RHS_handles = [
                    VecHandlePickle(RHS_path % i) for i in mode_idxs]
                TLSqrDMD.vec_space.lin_combine(
                    LHS_handles,
                    proj_POD_mode_handles,
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_POD_mode_handles, adv_vecs_vals) *
                    TLSqrDMD.proj_correlation_mat_eigvecs *
                    np.mat(np.diag(
                        TLSqrDMD.proj_correlation_mat_eigvals ** -0.5)) *
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_POD_mode_handles, TLSqrDMD_proj_mode_handles))
                TLSqrDMD.vec_space.lin_combine(
                    RHS_handles,
                    TLSqrDMD_proj_mode_handles,
                    np.mat(np.diag(TLSqrDMD.eigvals[mode_idxs])))
                for LHS, RHS in zip(LHS_handles, RHS_handles):
                    np.testing.assert_allclose(
                        LHS.get(), RHS.get(), rtol=rtol, atol=atol)

                # Test that adjoint modes are eigenvectors of the conjugate
                # transpose of approximating linear operator by checking
                # A^* \Phi = \Phi \Lambda^*.  Do this using handles, i.e. check
                # mode by mode.  Note that since np.vdot takes the conjugate of
                # its second argument, whereas modred assumes a conjugate is
                # taken on the first inner product argument, the inner product
                # matrix in the LHS computation must be conjugated.
                LHS_path = join(self.test_dir, 'LHS_%03d.pkl')
                LHS_handles = [
                    VecHandlePickle(LHS_path % i) for i in mode_idxs]
                RHS_path = join(self.test_dir, 'RHS_%03d.pkl')
                RHS_handles = [
                    VecHandlePickle(RHS_path % i) for i in mode_idxs]
                TLSqrDMD.vec_space.lin_combine(
                    LHS_handles,
                    proj_POD_mode_handles,
                    np.mat(np.diag(
                        TLSqrDMD.proj_correlation_mat_eigvals ** -0.5)) *
                    TLSqrDMD.proj_correlation_mat_eigvecs.T *
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        adv_vecs_vals, TLSqrDMD_adjoint_mode_handles))
                TLSqrDMD.vec_space.lin_combine(
                    RHS_handles,
                    TLSqrDMD_adjoint_mode_handles,
                    np.mat(np.diag(TLSqrDMD.eigvals[mode_idxs])).H)
                for LHS, RHS in zip(LHS_handles, RHS_handles):
                    np.testing.assert_allclose(
                        LHS.get(), RHS.get(), rtol=rtol, atol=atol)


    #@unittest.skip('Testing something else.')
    def test_compute_spectrum(self):
        """Test TLSqrDMD spectrum"""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated DMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute TLSqrDMD using modred.  (The TLSqrDMD spectral
                # coefficients are defined by a projection onto TLSqrDMD modes.
                # As such, testing them requires manipulations involving the
                # correct decomposition and modes, so we cannot isolate the
                # spectral coefficient computation from those computations.)
                TLSqrDMD = TLSqrDMDHandles(np.vdot, verbosity=0)
                TLSqrDMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Compute the projection of the vecs.
                proj_mat = (
                    TLSqrDMD.sum_correlation_mat_eigvecs *
                    TLSqrDMD.sum_correlation_mat_eigvecs.H)
                proj_vecs_handles = [
                    VecHandlePickle(self.proj_vec_path % i)
                    for i in range(len(vecs_vals))]
                TLSqrDMD.vec_space.lin_combine(
                    proj_vecs_handles, vecs_vals, proj_mat)

                # Test by checking a least-squares projection (of the projected
                # vecs) onto the projected modes, which is analytically
                # equivalent to a biorthogonal projection onto the exact modes.
                # The latter is implemented (using various identities) in
                # modred.  Here, test using the former approach, as it doesn't
                # require adjoint modes.
                mode_idxs = range(TLSqrDMD.eigvals.size)
                proj_mode_handles = [
                    V.VecHandlePickle(self.proj_mode_path % i)
                    for i in mode_idxs]
                TLSqrDMD.compute_proj_modes(mode_idxs, proj_mode_handles)
                spectral_coeffs_true = np.array(np.abs(
                    np.linalg.inv(
                        TLSqrDMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, proj_vecs_handles[0]))).squeeze()
                spectral_coeffs = TLSqrDMD.compute_spectrum()
                np.testing.assert_allclose(
                    spectral_coeffs, spectral_coeffs_true, rtol=rtol, atol=atol)


    #@unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        """Test projection coefficients"""
        rtol = 1e-10
        atol = 1e-12

        # Consider sequential time series as well as non-sequential.  In the
        # below for loop, the first elements of each zipped list correspond to a
        # sequential time series.  The second elements correspond to a
        # non-sequential time series.
        for vecs_arg, adv_vecs_arg, vecs_vals, adv_vecs_vals in zip(
            [self.vec_handles, self.vec_handles],
            [None, self.adv_vec_handles],
            [self.vec_handles[:-1], self.vec_handles],
            [self.vec_handles[1:], self.adv_vec_handles]):

            # Test that results hold for truncated or untruncated DMD
            # (i.e., whether or not the underlying POD basis is
            # truncated).
            for max_num_eigvals in [None, self.num_vecs // 2]:

                # Compute TLSqrDMD using modred.  (Testing the TLSqrDMD
                # projection coefficients requires the correct TLSqrDMD
                # decomposition and modes, so we cannot isolate the projection
                # coefficient computation from those computations.)
                TLSqrDMD = TLSqrDMDHandles(np.vdot, verbosity=0)
                TLSqrDMD.compute_decomp(
                    vecs_arg, adv_vec_handles=adv_vecs_arg,
                    max_num_eigvals=max_num_eigvals)

                # Compute the projection of the vecs and advanced vecs.
                proj_mat = (
                    TLSqrDMD.sum_correlation_mat_eigvecs *
                    TLSqrDMD.sum_correlation_mat_eigvecs.H)
                proj_vecs_handles = [
                    VecHandlePickle(self.proj_vec_path % i)
                    for i in range(len(vecs_vals))]
                proj_adv_vecs_handles = [
                    VecHandlePickle(self.proj_adv_vec_path % i)
                    for i in range(len(vecs_vals))]
                TLSqrDMD.vec_space.lin_combine(
                    proj_vecs_handles, vecs_vals, proj_mat)
                TLSqrDMD.vec_space.lin_combine(
                    proj_adv_vecs_handles, adv_vecs_vals, proj_mat)

                # Test by checking a least-squares projection (of the projected
                # vecs) onto the projected modes, which is analytically
                # equivalent to a biorthogonal projection onto the exact modes.
                # The latter is implemented (using various identities) in
                # modred.  Here, test using the former approach, as it doesn't
                # require adjoint modes.
                mode_idxs = range(TLSqrDMD.eigvals.size)
                proj_mode_handles = [
                    V.VecHandlePickle(self.proj_mode_path % i)
                    for i in mode_idxs]
                TLSqrDMD.compute_proj_modes(mode_idxs, proj_mode_handles)
                proj_coeffs_true = (
                    np.linalg.inv(
                        TLSqrDMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, proj_vecs_handles))
                adv_proj_coeffs_true = (
                    np.linalg.inv(
                        TLSqrDMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    TLSqrDMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, proj_adv_vecs_handles))
                proj_coeffs, adv_proj_coeffs = TLSqrDMD.compute_proj_coeffs()
                np.testing.assert_allclose(
                    proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
