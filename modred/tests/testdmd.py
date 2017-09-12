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


@unittest.skip('Testing something else.')
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
                            (exact_modes, proj_modes, spectral_coeffs, eigvals,
                            R_low_order_eigvecs, L_low_order_eigvecs,
                            correlation_mat_eigvals, correlation_mat_eigvecs,
                            correlation_mat, cross_correlation_mat) =\
                            compute_DMD_matrices_snaps_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals,
                                return_all=True)
                            exact_modes_sliced, proj_modes_sliced =\
                            compute_DMD_matrices_snaps_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                mode_indices=mode_indices,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals,
                                return_all=True)[:2]

                            # For method of snapshots, test correlation mats
                            # values by simply recomputing them.
                            np.testing.assert_allclose(
                                IP(vecs_vals, vecs_vals), correlation_mat,
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                IP(vecs_vals, adv_vecs_vals),
                                cross_correlation_mat,
                                rtol=rtol, atol=atol)

                        elif method == 'direct':
                            (exact_modes, proj_modes, spectral_coeffs, eigvals,
                            R_low_order_eigvecs, L_low_order_eigvecs,
                            correlation_mat_eigvals, correlation_mat_eigvecs) =\
                            compute_DMD_matrices_direct_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals,
                                return_all=True)
                            (vecs_POD_modes, vecs_POD_eigvals,
                            vecs_POD_eigvecs) =\
                            pod.compute_POD_matrices_direct_method(
                                vecs_vals, inner_product_weights=weights,
                                return_all=True)
                            exact_modes_sliced, proj_modes_sliced =\
                            compute_DMD_matrices_direct_method(
                                vecs_arg, adv_vecs=adv_vecs_arg,
                                mode_indices=mode_indices,
                                inner_product_weights=weights,
                                max_num_eigvals=max_num_eigvals,
                                return_all=True)[:2]

                        else:
                            raise ValueError('Invalid DMD method.')

                        # Test correlation mat eigenvalues and eigenvectors.
                        np.testing.assert_allclose(
                            IP(vecs_vals, vecs_vals) * correlation_mat_eigvecs,
                            correlation_mat_eigvecs * np.mat(np.diag(
                                correlation_mat_eigvals)),
                            rtol=rtol, atol=atol)

                        # Compute the approximating linear operator relating the
                        # vecs to the adv_vecs.  To do this, use the
                        # eigendecomposition of the correlation mat.
                        vecs_POD_build_coeffs = (
                            correlation_mat_eigvecs *
                            np.mat(np.diag(correlation_mat_eigvals ** -0.5)))
                        vecs_POD_modes = vecs_vals * vecs_POD_build_coeffs
                        approx_linear_op = (
                            adv_vecs_vals * correlation_mat_eigvecs *
                            np.mat(np.diag(correlation_mat_eigvals ** -0.5)) *
                            vecs_POD_modes.H)
                        low_order_linear_op = IP(
                            vecs_POD_modes,
                            IP(approx_linear_op.H, vecs_POD_modes))

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

                        # Test the exact modes, which are eigenvectors of the
                        # approximating linear operator.
                        np.testing.assert_allclose(
                            IP(approx_linear_op.H, exact_modes),
                            exact_modes * np.mat(np.diag(eigvals)),
                            rtol=rtol, atol=atol)

                        # Test the projected modes, which are eigenvectors of
                        # the approximating linear operator projected onto the
                        # POD modes of the vecs.
                        np.testing.assert_allclose(
                            vecs_POD_modes * IP(
                                vecs_POD_modes,
                                IP(approx_linear_op.H, proj_modes)),
                            proj_modes * np.mat(np.diag(eigvals)),
                            rtol=rtol, atol=atol)

                        # Test spectral coefficients against an explicit
                        # projection using the adjoint DMD modes.
                        adjoint_modes = vecs_POD_modes * L_low_order_eigvecs
                        np.testing.assert_allclose(
                            np.array(np.abs(
                                IP(adjoint_modes, vecs_vals[:, 0]))).squeeze(),
                            spectral_coeffs,
                            rtol=rtol, atol=atol)

                        # Test that use of mode indices argument returns correct
                        # subset of modes
                        np.testing.assert_allclose(
                            exact_modes_sliced, exact_modes[:, mode_indices],
                            rtol=rtol, atol=atol)
                        np.testing.assert_allclose(
                            proj_modes_sliced, proj_modes[:, mode_indices],
                            rtol=rtol, atol=atol)


@unittest.skip('Testing something else.')
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
                # relating the vecs to the adv_vecs.  Do this using the POD of
                # the vecs rather than using the outputs of the DMD object
                # above, as this helps maintain the independence of the
                # reference data for the tests.  This also makes sure that the
                # inner product weights are correctly accounted for in computing
                # the pseudo-inverse of the vecs.  (Make sure to truncate the
                # POD basis as necessary.)
                POD = pod.PODHandles(DMD.vec_space.inner_product, verbosity=0)
                POD.compute_decomp(vecs_vals)
                POD.eigvals = POD.eigvals[:eigvals.size]
                POD.eigvecs = POD.eigvecs[:, :eigvals.size]
                POD_mode_path = join(self.test_dir, 'pod_mode_%03d.txt')
                POD_mode_handles = [
                    VecHandlePickle(POD_mode_path % i)
                    for i in xrange(eigvals.size)]
                POD.compute_modes(range(eigvals.size), POD_mode_handles)
                low_order_linear_op = (
                    DMD.vec_space.compute_inner_product_mat(
                        POD_mode_handles, adv_vecs_vals) *
                    POD.eigvecs *
                    np.mat(np.diag(POD.eigvals ** -0.5)))

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

                # Compute the approximating linear operator relating the vecs to
                # the adv_vecs.  Do this using the POD of the vecs rather than
                # using the outputs of the DMD object above, as this helps
                # maintain the independence of the reference data for the tests.
                # This also makes sure that the inner product weights are
                # correctly accounted for in computing the pseudo-inverse of the
                # vecs.  (Make sure to truncate the POD basis as necessary.)
                POD = pod.PODHandles(DMD.vec_space.inner_product, verbosity=0)
                POD.compute_decomp(vecs_vals)
                if max_num_eigvals is not None:
                    POD.eigvals = POD.eigvals[:max_num_eigvals]
                    POD.eigvecs = POD.eigvecs[:, :max_num_eigvals]
                POD_mode_path = join(self.test_dir, 'pod_mode_%03d.pkl')
                POD_mode_handles = [
                    VecHandlePickle(POD_mode_path % i)
                    for i in xrange(POD.eigvals.size)]
                POD.compute_modes(range(POD.eigvals.size), POD_mode_handles)

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

                # Compute modes
                DMD.compute_exact_modes(mode_idxs, DMD_exact_mode_handles)
                DMD.compute_proj_modes(mode_idxs, DMD_proj_mode_handles)

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
                    POD.eigvecs *
                    np.mat(np.diag(POD.eigvals ** -0.5)) *
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
                    POD.eigvecs *
                    np.mat(np.diag(POD.eigvals ** -0.5)) *
                    DMD.vec_space.compute_inner_product_mat(
                        POD_mode_handles, DMD_proj_mode_handles))
                DMD.vec_space.lin_combine(
                    RHS_handles,
                    DMD_proj_mode_handles,
                    np.mat(np.diag(DMD.eigvals[mode_idxs])))
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
                # coefficients the correct DMD decomposition and modes, so we
                # cannot isolate the projection coefficient computation from
                # those computations.)
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
                proj_coeffs_true = np.array(
                    np.linalg.inv(
                        DMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    DMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, vecs_vals))
                adv_proj_coeffs_true = np.array(
                    np.linalg.inv(
                        DMD.vec_space.compute_symmetric_inner_product_mat(
                            proj_mode_handles)) *
                    DMD.vec_space.compute_inner_product_mat(
                        proj_mode_handles, adv_vecs_vals))
                proj_coeffs, adv_proj_coeffs = DMD.compute_proj_coeffs()
                np.testing.assert_allclose(
                    proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)


@unittest.skip('Testing something else.')
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
                            (exact_modes, proj_modes, eigvals, spectral_coeffs,
                            R_low_order_eigvecs, L_low_order_eigvecs,
                            summed_correlation_mats_eigvals,
                            summed_correlation_mats_eigvecs,
                            proj_correlation_mat_eigvals,
                            proj_correlation_mat_eigvecs,
                            correlation_mat, adv_correlation_mat,
                            cross_correlation_mat) =\
                                compute_TLSqrDMD_matrices_snaps_method(
                                    vecs_arg, adv_vecs=adv_vecs_arg,
                                    inner_product_weights=weights,
                                    max_num_eigvals=max_num_eigvals,
                                    return_all=True)
                            exact_modes_sliced, proj_modes_sliced =\
                                compute_TLSqrDMD_matrices_snaps_method(
                                    vecs_arg, adv_vecs=adv_vecs_arg,
                                    inner_product_weights=weights,
                                    max_num_eigvals=max_num_eigvals,
                                    return_all=True)[:2]

                            # For method of snapshots, test correlation mats
                            # values by simply recomputing them.
                            np.testing.assert_allclose(
                                IP(vecs_vals, vecs_vals), correlation_mat,
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                IP(vecs_vals, adv_vecs_vals),
                                cross_correlation_mat,
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                IP(adv_vecs_vals, adv_vecs_vals),
                                adv_correlation_mat,
                                rtol=rtol, atol=atol)

                        elif method == 'direct':
                            (exact_modes, proj_modes, eigvals, spectral_coeffs,
                            R_low_order_eigvecs, L_low_order_eigvecs,
                            summed_correlation_mats_eigvals,
                            summed_correlation_mats_eigvecs,
                            proj_correlation_mat_eigvals,
                            proj_correlation_mat_eigvecs) =\
                                compute_TLSqrDMD_matrices_direct_method(
                                    vecs_arg, adv_vecs=adv_vecs_arg,
                                    inner_product_weights=weights,
                                    max_num_eigvals=max_num_eigvals,
                                    return_all=True)
                            exact_modes_sliced, proj_modes_sliced =\
                                compute_TLSqrDMD_matrices_direct_method(
                                    vecs_arg, adv_vecs=adv_vecs_arg,
                                    inner_product_weights=weights,
                                    max_num_eigvals=max_num_eigvals,
                                    return_all=True)[:2]

                        else:
                            raise ValueError('Invalid DMD method.')

                        # Test summed correlation mat eigenvalues and
                        # eigenvectors
                        np.testing.assert_allclose((
                            IP(vecs_vals, vecs_vals) +
                            IP(adv_vecs_vals, adv_vecs_vals)) *
                            summed_correlation_mats_eigvecs,
                            summed_correlation_mats_eigvecs * np.mat(np.diag(
                                summed_correlation_mats_eigvals)),
                            rtol=rtol, atol=atol)

                        # Test projected correlation mat eigenvalues and
                        # eigenvectors
                        proj_vecs_vals = (
                            vecs_vals *
                            summed_correlation_mats_eigvecs *
                            summed_correlation_mats_eigvecs.H)
                        proj_adv_vecs_vals = (
                            adv_vecs_vals *
                            summed_correlation_mats_eigvecs *
                            summed_correlation_mats_eigvecs.H)
                        np.testing.assert_allclose(
                            IP(proj_vecs_vals, proj_vecs_vals) *
                            proj_correlation_mat_eigvecs,
                            proj_correlation_mat_eigvecs * np.mat(np.diag(
                                proj_correlation_mat_eigvals)),
                            rtol=rtol, atol=atol)

                        # Compute the approximating linear operator relating the
                        # projected vecs to the projected adv_vecs.  To do this,
                        # compute the POD of the projected vecs using the
                        # eigendecomposition of the projected correlation mat.
                        proj_vecs_POD_build_coeffs = (
                            proj_correlation_mat_eigvecs *
                            np.mat(np.diag(
                                proj_correlation_mat_eigvals ** -0.5)))
                        proj_vecs_POD_modes = (
                            proj_vecs_vals * proj_vecs_POD_build_coeffs)
                        approx_linear_op = (
                            proj_adv_vecs_vals * proj_correlation_mat_eigvecs *
                            np.mat(np.diag(
                                proj_correlation_mat_eigvals ** -0.5)) *
                            proj_vecs_POD_modes.H)
                        low_order_linear_op = IP(
                            proj_vecs_POD_modes,
                            IP(approx_linear_op.H, proj_vecs_POD_modes))

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

                        # Test the exact modes, which are eigenvectors of the
                        # approximating linear operator.
                        np.testing.assert_allclose(
                            IP(approx_linear_op.H, exact_modes),
                            exact_modes * np.mat(np.diag(eigvals)),
                            rtol=rtol, atol=atol)

                        # Test the projected modes, which are eigenvectors of
                        # the approximating linear operator projected onto the
                        # POD modes of the vecs.
                        np.testing.assert_allclose(
                            proj_vecs_POD_modes * IP(
                                proj_vecs_POD_modes,
                                IP(approx_linear_op.H, proj_modes)),
                            proj_modes * np.mat(np.diag(eigvals)),
                            rtol=rtol, atol=atol)

                        # Test spectral coefficients against an explicit
                        # projection using the adjoint DMD modes.
                        adjoint_modes = (
                            proj_vecs_POD_modes * L_low_order_eigvecs)
                        np.testing.assert_allclose(
                            np.array(np.abs(IP(
                                adjoint_modes,
                                proj_vecs_vals[:, 0]))).squeeze(),
                            spectral_coeffs,
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


        # Specify data dimensions
        self.num_states = 30
        self.num_vecs = 10

        # Specify truncation level
        self.max_num_eigvals = int(np.round(self.num_states / 2))

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


    @unittest.skip('Testing something else.')
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
            'summed_correlation_mats_eigvals': None,
            'summed_correlation_mats_eigvecs': None,
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
        summed_correlation_mats_eigvals = parallel.call_and_bcast(
            np.random.random, 5)
        summed_correlation_mats_eigvecs = parallel.call_and_bcast(
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
        spectral_coeffs = parallel.call_and_bcast(np.random.random, 5)
        proj_coeffs = parallel.call_and_bcast(np.random.random, (5,5))
        adv_proj_coeffs = parallel.call_and_bcast(np.random.random, (5,5))

        # Create a DMD object and store the data in it
        TLSqrDMD_save = TLSqrDMDHandles(None, verbosity=0)
        TLSqrDMD_save.eigvals = eigvals
        TLSqrDMD_save.R_low_order_eigvecs = R_low_order_eigvecs
        TLSqrDMD_save.L_low_order_eigvecs = L_low_order_eigvecs
        TLSqrDMD_save.summed_correlation_mats_eigvals =\
            summed_correlation_mats_eigvals
        TLSqrDMD_save.summed_correlation_mats_eigvecs =\
            summed_correlation_mats_eigvecs
        TLSqrDMD_save.proj_correlation_mat_eigvals =\
            proj_correlation_mat_eigvals
        TLSqrDMD_save.proj_correlation_mat_eigvecs =\
            proj_correlation_mat_eigvecs
        TLSqrDMD_save.correlation_mat = correlation_mat
        TLSqrDMD_save.cross_correlation_mat = cross_correlation_mat
        TLSqrDMD_save.adv_correlation_mat = adv_correlation_mat
        TLSqrDMD_save.spectral_coeffs = spectral_coeffs
        TLSqrDMD_save.proj_coeffs = proj_coeffs
        TLSqrDMD_save.adv_proj_coeffs = adv_proj_coeffs

        # Write the data to disk
        eigvals_path = join(self.test_dir, 'tlsqrdmd_eigvals.txt')
        R_low_order_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_R_low_order_eigvecs.txt')
        L_low_order_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_L_low_order_eigvecs.txt')
        summed_correlation_mats_eigvals_path = join(
            self.test_dir, 'tlsqrdmd_summed_corr_mats_eigvals.txt')
        summed_correlation_mats_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_summed_corr_mats_eigvecs.txt')
        proj_correlation_mat_eigvals_path = join(
            self.test_dir, 'tlsqrdmd_proj_corr_mat_eigvals.txt')
        proj_correlation_mat_eigvecs_path = join(
            self.test_dir, 'tlsqrdmd_proj_corr_mat_eigvecs.txt')
        correlation_mat_path = join(self.test_dir, 'tlsqrdmd_corr_mat.txt')
        cross_correlation_mat_path = join(
            self.test_dir, 'tlsqrdmd_cross_corr_mat.txt')
        adv_correlation_mat_path = join(
            self.test_dir, 'tlsqrdmd_adv_corr_mat.txt')
        spectral_coeffs_path = join(
            self.test_dir, 'tlsqrdmd_spectral_coeffs.txt')
        proj_coeffs_path = join(self.test_dir, 'tlsqrdmd_proj_coeffs.txt')
        adv_proj_coeffs_path = join(
            self.test_dir, 'tlsqrdmd_adv_proj_coeffs.txt')
        TLSqrDMD_save.put_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            summed_correlation_mats_eigvals_path,
            summed_correlation_mats_eigvecs_path,
            proj_correlation_mat_eigvals_path ,
            proj_correlation_mat_eigvecs_path)
        TLSqrDMD_save.put_correlation_mat(correlation_mat_path)
        TLSqrDMD_save.put_cross_correlation_mat(cross_correlation_mat_path)
        TLSqrDMD_save.put_adv_correlation_mat(adv_correlation_mat_path)
        TLSqrDMD_save.put_spectral_coeffs(spectral_coeffs_path)
        TLSqrDMD_save.put_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)
        parallel.barrier()

        # Create a new TLSqrDMD object and use it to load data
        TLSqrDMD_load = TLSqrDMDHandles(None, verbosity=0)
        TLSqrDMD_load.get_decomp(
            eigvals_path, R_low_order_eigvecs_path, L_low_order_eigvecs_path,
            summed_correlation_mats_eigvals_path,
            summed_correlation_mats_eigvecs_path,
            proj_correlation_mat_eigvals_path,
            proj_correlation_mat_eigvecs_path)
        TLSqrDMD_load.get_correlation_mat(correlation_mat_path)
        TLSqrDMD_load.get_cross_correlation_mat(cross_correlation_mat_path)
        TLSqrDMD_load.get_adv_correlation_mat(adv_correlation_mat_path)
        TLSqrDMD_load.get_spectral_coeffs(spectral_coeffs_path)
        TLSqrDMD_load.get_proj_coeffs(proj_coeffs_path, adv_proj_coeffs_path)

        # Check that the loaded data is correct
        np.testing.assert_allclose(TLSqrDMD_load.eigvals, eigvals)
        np.testing.assert_allclose(
            TLSqrDMD_load.R_low_order_eigvecs, R_low_order_eigvecs)
        np.testing.assert_allclose(
            TLSqrDMD_load.L_low_order_eigvecs, L_low_order_eigvecs)
        np.testing.assert_allclose(
            TLSqrDMD_load.summed_correlation_mats_eigvals,
            summed_correlation_mats_eigvals)
        np.testing.assert_allclose(
            TLSqrDMD_load.summed_correlation_mats_eigvecs,
            summed_correlation_mats_eigvecs)
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
            np.array(TLSqrDMD_load.spectral_coeffs).squeeze(), spectral_coeffs)
        np.testing.assert_allclose(
            TLSqrDMD_load.proj_coeffs, proj_coeffs)
        np.testing.assert_allclose(
            TLSqrDMD_load.adv_proj_coeffs, adv_proj_coeffs)


    def _helper_compute_DMD_from_data(
        self, vec_array, inner_product, adv_vec_array=None,
        max_num_eigvals=None):
        # Generate adv_vec_array for the case of a sequential dataset
        if adv_vec_array is None:
            adv_vec_array = vec_array[:, 1:]
            vec_array = vec_array[:, :-1]

        # Stack arrays for total-least-squares DMD
        stacked_vec_array = np.vstack((vec_array, adv_vec_array))

        # Create lists of vecs, advanced vecs for inner product function
        vecs = [vec_array[:, i] for i in range(vec_array.shape[1])]
        adv_vecs = [adv_vec_array[:, i] for i in range(adv_vec_array.shape[1])]
        stacked_vecs = [
            stacked_vec_array[:, i] for i in range(stacked_vec_array.shape[1])]

        # Compute SVD of stacked data vectors
        summed_correlation_mats = inner_product(vecs, vecs) + inner_product(
            adv_vecs, adv_vecs)
        summed_correlation_mats_eigvals, summed_correlation_mats_eigvecs =\
            util.eigh(summed_correlation_mats)
        cross_correlation_mat = inner_product(vecs, adv_vecs)
        stacked_U = vec_array.dot(
            np.array(summed_correlation_mats_eigvecs)).dot(
            np.diag(summed_correlation_mats_eigvals ** -0.5))
        stacked_U_list = [stacked_U[:, i] for i in range(stacked_U.shape[1])]

        # Truncate stacked SVD if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < summed_correlation_mats_eigvals.size):
            summed_correlation_mats_eigvals = summed_correlation_mats_eigvals[
                :max_num_eigvals]
            summed_correlation_mats_eigvecs = summed_correlation_mats_eigvecs[
                :, :max_num_eigvals]
            stacked_U = stacked_U[:, :max_num_eigvals]
            stacked_U_list = stacked_U_list[:max_num_eigvals]

        # Project data matrices
        vec_array_proj = np.array(
            vec_array *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
        adv_vec_array_proj = np.array(
            adv_vec_array *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
        vecs_proj = [
            vec_array_proj[:, i] for i in range(vec_array_proj.shape[1])]
        adv_vecs_proj = [
            adv_vec_array_proj[:, i]
            for i in range(adv_vec_array_proj.shape[1])]

        # SVD of projected snapshots
        proj_correlation_mat = inner_product(vecs_proj, vecs_proj)
        proj_correlation_mat_eigvals, proj_correlation_mat_eigvecs =\
            util.eigh(proj_correlation_mat)
        proj_U = vec_array.dot(
            np.array(proj_correlation_mat_eigvecs)).dot(
            np.diag(proj_correlation_mat_eigvals ** -0.5))
        proj_U_list = [proj_U[:, i] for i in range(proj_U.shape[1])]

        # Truncate stacked SVD if necessary
        if max_num_eigvals is not None and (
            max_num_eigvals < proj_correlation_mat_eigvals.size):
            proj_correlation_mats_eigvals = proj_correlation_mats_eigvals[
                :max_num_eigvals]
            proj_correlation_mats_eigvecs = proj_correlation_mats_eigvecs[
                :, :max_num_eigvals]
            proj_U = proj_U[:, :max_num_eigvals]
            proj_U_list = proj_U_list[:max_num_eigvals]

        # Compute eigendecomposition of low order linear operator
        A_tilde = inner_product(proj_U_list, adv_vecs_proj).dot(
            np.array(proj_correlation_mat_eigvecs)).dot(
            np.diag(proj_correlation_mat_eigvals ** -0.5))
        eigvals, R_low_order_eigvecs, L_low_order_eigvecs =\
            util.eig_biorthog(A_tilde, scale_choice='left')
        R_low_order_eigvecs = np.mat(R_low_order_eigvecs)
        L_low_order_eigvecs = np.mat(L_low_order_eigvecs)

        # Compute build coefficients
        build_coeffs_proj = (
            summed_correlation_mats_eigvecs.dot(
            summed_correlation_mats_eigvecs.T.dot(
            proj_correlation_mat_eigvecs.dot(
            np.diag(proj_correlation_mat_eigvals ** -0.5)).dot(
            R_low_order_eigvecs))))
        build_coeffs_exact = (
            summed_correlation_mats_eigvecs.dot(
            summed_correlation_mats_eigvecs.T.dot(
            proj_correlation_mat_eigvecs.dot(
            np.diag(proj_correlation_mat_eigvals ** -0.5)).dot(
            R_low_order_eigvecs).dot(
            np.diag(eigvals ** -1.)))))

        # Compute modes
        modes_proj = vec_array_proj.dot(build_coeffs_proj)
        modes_exact = adv_vec_array_proj.dot(build_coeffs_exact)
        adj_modes = proj_U.dot(L_low_order_eigvecs)
        adj_modes_list = [
            np.array(adj_modes[:, i]) for i in range(adj_modes.shape[1])]

        return (
            modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
            L_low_order_eigvecs, summed_correlation_mats_eigvals,
            summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
            proj_correlation_mat_eigvecs, cross_correlation_mat, adj_modes)


    def _helper_test_1D_array_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), 1)
        self.assertEqual(len(test_vals.shape), 1)
        self.assertEqual(true_vals.size, test_vals.size)

        # Check values entry by entry.
        for idx in range(true_vals.size):
            true_val = true_vals[idx]
            test_val = test_vals[idx]
            self.assertTrue(
                np.allclose(true_val, test_val, rtol=rtol, atol=atol)
                or
                np.allclose(-true_val, test_val, rtol=rtol, atol=atol))


    def _helper_test_mat_to_sign(
        self, true_vals, test_vals, rtol=1e-12, atol=1e-16):
        # Check that shapes are the same
        self.assertEqual(len(true_vals.shape), len(test_vals.shape))
        for shape_idx in range(len(true_vals.shape)):
            self.assertEqual(
                true_vals.shape[shape_idx], test_vals.shape[shape_idx])

        # Check values column by columns.  To allow for matrices or arrays,
        # turn columns into arrays and squeeze them (forcing 1D arrays).  This
        # avoids failures due to trivial shape mismatches.
        for col_idx in range(true_vals.shape[1]):
            true_col = np.array(true_vals[:, col_idx]).squeeze()
            test_col = np.array(test_vals[:, col_idx]).squeeze()
            self.assertTrue(
                np.allclose(true_col, test_col, rtol=rtol, atol=atol)
                or
                np.allclose(-true_col, test_col, rtol=rtol, atol=atol))


    def _helper_check_decomp(
        self, vec_array,  vec_handles, adv_vec_array=None,
        adv_vec_handles=None, max_num_eigvals=None):
        # Set tolerance.
        rtol = 1e-10
        atol = 1e-12

        # Compute reference DMD values
        (eigvals_true, R_low_order_eigvecs_true, L_low_order_eigvecs_true,
            summed_correlation_mats_eigvals_true,
            summed_correlation_mats_eigvecs_true,
            proj_correlation_mat_eigvals_true,
            proj_correlation_mat_eigvecs_true) = (
            self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array,
            max_num_eigvals=max_num_eigvals))[2:-2]

        # Compute DMD using modred
        (eigvals_returned,  R_low_order_eigvecs_returned,
            L_low_order_eigvecs_returned,
            summed_correlation_mats_eigvals_returned,
            summed_correlation_mats_eigvecs_returned,
            proj_correlation_mat_eigvals_returned,
            proj_correlation_mat_eigvecs_returned
            ) = self.my_DMD.compute_decomp(
            vec_handles, adv_vec_handles=adv_vec_handles,
            max_num_eigvals=max_num_eigvals)

        # Test that matrices were correctly computed.  For build coeffs, check
        # column by column, as it is ok to be off by a negative sign.
        np.testing.assert_allclose(
            self.my_DMD.eigvals, eigvals_true, rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            self.my_DMD.R_low_order_eigvecs, R_low_order_eigvecs_true,
            rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            self.my_DMD.L_low_order_eigvecs, L_low_order_eigvecs_true,
            rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            self.my_DMD.summed_correlation_mats_eigvals,
            summed_correlation_mats_eigvals_true, rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            self.my_DMD.summed_correlation_mats_eigvecs,
            summed_correlation_mats_eigvecs_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            self.my_DMD.proj_correlation_mat_eigvals,
            proj_correlation_mat_eigvals_true, rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            self.my_DMD.proj_correlation_mat_eigvecs,
            proj_correlation_mat_eigvecs_true, rtol=rtol, atol=atol)

        # Test that matrices were correctly returned
        np.testing.assert_allclose(
            eigvals_returned, eigvals_true, rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            R_low_order_eigvecs_returned, R_low_order_eigvecs_true, rtol=rtol,
            atol=atol)
        self._helper_test_mat_to_sign(
            L_low_order_eigvecs_returned, L_low_order_eigvecs_true, rtol=rtol,
            atol=atol)
        np.testing.assert_allclose(
            summed_correlation_mats_eigvals_returned,
            summed_correlation_mats_eigvals_true, rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            summed_correlation_mats_eigvecs_returned,
            summed_correlation_mats_eigvecs_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            proj_correlation_mat_eigvals_returned,
            proj_correlation_mat_eigvals_true, rtol=rtol, atol=atol)
        self._helper_test_mat_to_sign(
            proj_correlation_mat_eigvecs_returned,
            proj_correlation_mat_eigvecs_true, rtol=rtol, atol=atol)


    def _helper_check_modes(self, modes_true, mode_path_list):
        # Set tolerance.
        rtol = 1e-10
        atol = 1e-12

        # Load all modes into matrix, compare to modes from direct computation
        modes_computed = np.zeros(modes_true.shape, dtype=complex)
        for i, path in enumerate(mode_path_list):
            modes_computed[:, i] = VecHandlePickle(path).get()
        np.testing.assert_allclose(
            modes_true, modes_computed, rtol=rtol, atol=atol)


    @unittest.skip('Testing something else.')
    def test_compute_decomp(self):
        """Test DMD decomposition"""
        # Define an array of vectors, with corresponding handles
        vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())

        # Check modred against direct computation, for a sequential dataset
        # (always need to truncate for TLSDMD).
        parallel.barrier()
        self._helper_check_decomp(vec_array, self.vec_handles,
            max_num_eigvals=self.max_num_eigvals)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Check modred against direct computation, for a non-sequential dataset
        # (always need to truncate for TLSDMD).
        parallel.barrier()
        self._helper_check_decomp(
            vec_array, self.vec_handles, adv_vec_array=adv_vec_array,
            adv_vec_handles=self.adv_vec_handles,
            max_num_eigvals=self.max_num_eigvals)

        # Check that if mismatched sets of handles are passed in, an error is
        # raised.
        self.assertRaises(ValueError, self.my_DMD.compute_decomp,
            self.vec_handles, self.adv_vec_handles[:-1])


    @unittest.skip('Testing something else.')
    def test_compute_modes(self):
        """Test building of modes."""
        # Generate path names for saving modes to disk
        mode_path = join(self.test_dir, 'dmd_mode_%03d.pkl')

        ### SEQUENTIAL DATASET ###
        # Generate data
        seq_vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(seq_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data (must truncate for TLSDMD)
        (modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
        L_low_order_eigvecs, summed_correlation_mats_eigvals,
        summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
        proj_correlation_mat_eigvecs) = self._helper_compute_DMD_from_data(
            seq_vec_array, util.InnerProductBlock(np.vdot),
            max_num_eigvals=self.max_num_eigvals)[:-2]

        # Set the build_coeffs attribute of an empty DMD object each time, so
        # that the modred computation uses the same coefficients as the direct
        # computation.
        parallel.barrier()
        self.my_DMD.eigvals = eigvals
        self.my_DMD.R_low_order_eigvecs = R_low_order_eigvecs
        self.my_DMD.summed_correlation_mats_eigvals =\
            summed_correlation_mats_eigvals
        self.my_DMD.summed_correlation_mats_eigvecs =\
            summed_correlation_mats_eigvecs
        self.my_DMD.proj_correlation_mat_eigvals = proj_correlation_mat_eigvals
        self.my_DMD.proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs

        # Generate mode paths for saving modes to disk
        seq_mode_path_list = [
            mode_path % i for i in range(eigvals.size)]
        seq_mode_indices = range(len(seq_mode_path_list))

        # Compute modes by passing in handles
        self.my_DMD.compute_exact_modes(seq_mode_indices,
            [VecHandlePickle(path) for path in seq_mode_path_list],
            adv_vec_handles=self.vec_handles[1:])
        self._helper_check_modes(modes_exact, seq_mode_path_list)
        self.my_DMD.compute_proj_modes(seq_mode_indices,
            [VecHandlePickle(path) for path in seq_mode_path_list],
            vec_handles=self.vec_handles)
        self._helper_check_modes(modes_proj, seq_mode_path_list)

        # Compute modes without passing in handles, so first set full
        # sequential dataset as vec_handles.
        self.my_DMD.vec_handles = self.vec_handles
        self.my_DMD.compute_exact_modes(seq_mode_indices,
            [VecHandlePickle(path) for path in seq_mode_path_list])
        self._helper_check_modes(modes_exact, seq_mode_path_list)
        self.my_DMD.compute_proj_modes(seq_mode_indices,
            [VecHandlePickle(path) for path in seq_mode_path_list])
        self._helper_check_modes(modes_proj, seq_mode_path_list)

        # For exact modes, also compute by setting adv_vec_handles
        self.my_DMD.vec_handles = None
        self.my_DMD.adv_vec_handles = self.vec_handles[1:]
        self.my_DMD.compute_exact_modes(seq_mode_indices,
            [VecHandlePickle(path) for path in seq_mode_path_list])
        self._helper_check_modes(modes_exact, seq_mode_path_list)

        ### NONSEQUENTIAL DATA ###
        # Generate data
        vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        adv_vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, (handle, adv_handle) in enumerate(
                zip(self.vec_handles, self.adv_vec_handles)):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())
                adv_handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD directly from data (must truncate for TLSDMD)
        (modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
        L_low_order_eigvecs, summed_correlation_mats_eigvals,
        summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
        proj_correlation_mat_eigvecs ) = self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array,
            max_num_eigvals=self.max_num_eigvals)[:-2]

        # Set the build_coeffs attribute of an empty DMD object each time, so
        # that the modred computation uses the same coefficients as the direct
        # computation.
        parallel.barrier()
        self.my_DMD.eigvals = eigvals
        self.my_DMD.R_low_order_eigvecs = R_low_order_eigvecs
        self.my_DMD.summed_correlation_mats_eigvals =\
            summed_correlation_mats_eigvals
        self.my_DMD.summed_correlation_mats_eigvecs =\
            summed_correlation_mats_eigvecs
        self.my_DMD.proj_correlation_mat_eigvals = proj_correlation_mat_eigvals
        self.my_DMD.proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs

        # Generate mode paths for saving modes to disk
        mode_path_list = [
            mode_path % i for i in range(eigvals.size)]
        mode_indices = range(len(mode_path_list))

        # Compute modes by passing in handles
        self.my_DMD.compute_exact_modes(mode_indices,
            [VecHandlePickle(path) for path in mode_path_list],
            adv_vec_handles=self.adv_vec_handles)
        self._helper_check_modes(modes_exact, mode_path_list)
        self.my_DMD.compute_proj_modes(mode_indices,
            [VecHandlePickle(path) for path in mode_path_list],
            vec_handles=self.vec_handles)
        self._helper_check_modes(modes_proj, mode_path_list)

        # Compute modes without passing in handles, so first set full
        # sequential dataset as vec_handles.
        self.my_DMD.vec_handles = self.vec_handles
        self.my_DMD.adv_vec_handles = self.adv_vec_handles
        self.my_DMD.compute_exact_modes(mode_indices,
            [VecHandlePickle(path) for path in mode_path_list])
        self._helper_check_modes(modes_exact, mode_path_list)
        self.my_DMD.compute_proj_modes(mode_indices,
            [VecHandlePickle(path) for path in mode_path_list])
        self._helper_check_modes(modes_proj, mode_path_list)


    @unittest.skip('Testing something else.')
    def test_compute_spectrum(self):
        """Test DMD spectrum"""
        rtol = 1e-10
        atol = 1e-12

        # Define an array of vectors, with corresponding handles
        vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())

        # Compute DMD manually and then set the data in a DMDHandles object.
        # This way, we test only the task of computing the spectral
        # coefficients, and not also the task of computing the decomposition.
        (modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
        L_low_order_eigvecs, summed_correlation_mats_eigvals,
        summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
        proj_correlation_mat_eigvecs, cross_correlation_mat, adj_modes) =\
            self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            max_num_eigvals=self.max_num_eigvals)
        self.my_DMD.L_low_order_eigvecs = L_low_order_eigvecs
        self.my_DMD.proj_correlation_mat_eigvals = proj_correlation_mat_eigvals
        self.my_DMD.proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs

        # Check that spectral coefficients computed using adjoints match those
        # computed using a direct projection onto the adjoint modes
        parallel.barrier()
        spectral_coeffs = self.my_DMD.compute_spectrum()
        vec_array_proj = np.array(
            vec_array[:, :-1] *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
        spectral_coeffs_true = np.abs(np.array(
            np.dot(adj_modes.conj().T, vec_array_proj[:, 0]))).squeeze()
        np.testing.assert_allclose(
            spectral_coeffs, spectral_coeffs_true, rtol=rtol, atol=atol)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD manually and then set the data in a DMDHandles object.
        # This way, we test only the task of computing the spectral
        # coefficients, and not also the task of computing the decomposition.
        (modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
        L_low_order_eigvecs, summed_correlation_mats_eigvals,
        summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
        proj_correlation_mat_eigvecs, cross_correlation_mat, adj_modes) =\
            self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array,
            max_num_eigvals=self.max_num_eigvals)
        self.my_DMD.L_low_order_eigvecs = L_low_order_eigvecs
        self.my_DMD.proj_correlation_mat_eigvals = proj_correlation_mat_eigvals
        self.my_DMD.proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs

        # Check spectral coefficients using a direct projection onto the
        # adjoint modes.  (Must always truncate for TLSDMD.)
        parallel.barrier()
        spectral_coeffs = self.my_DMD.compute_spectrum()
        vec_array_proj = np.array(
            vec_array *
            summed_correlation_mats_eigvecs *
            summed_correlation_mats_eigvecs.H)
        spectral_coeffs_true = np.abs(np.array(
            np.dot(adj_modes.conj().T, vec_array_proj[:, 0]))).squeeze()
        np.testing.assert_allclose(
            spectral_coeffs, spectral_coeffs_true, rtol=rtol, atol=atol)


    @unittest.skip('Testing something else.')
    def test_compute_proj_coeffs(self):
        """Test projection coefficients"""
        rtol = 1e-10
        atol = 1e-12

        # Define an array of vectors, with corresponding handles
        vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(np.array(vec_array[:, vec_index]).squeeze())

        # Compute DMD manually and then set the data in a TLSqrDMDHandles
        # object.  This way, we test only the task of computing the projection
        # coefficients, and not also the task of computing the decomposition.
        (modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
        L_low_order_eigvecs, summed_correlation_mats_eigvals,
        summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
        proj_correlation_mat_eigvecs, cross_correlation_mat, adj_modes) =\
            self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            max_num_eigvals=self.max_num_eigvals)
        self.my_DMD.L_low_order_eigvecs = L_low_order_eigvecs
        self.my_DMD.proj_correlation_mat_eigvals = proj_correlation_mat_eigvals
        self.my_DMD.proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs
        self.my_DMD.summed_correlation_mats_eigvecs =\
            summed_correlation_mats_eigvecs
        self.my_DMD.cross_correlation_mat = cross_correlation_mat

        # Check the spectral coefficient values.  Compare the formula
        # implemented in modred to a direct projection onto the adjoint modes.
        parallel.barrier()
        proj_coeffs, adv_proj_coeffs = self.my_DMD.compute_proj_coeffs()
        vec_array_proj = np.array(
            vec_array[:, :-1] *
            summed_correlation_mats_eigvecs * summed_correlation_mats_eigvecs.H)
        adv_vec_array_proj = np.array(
            vec_array[:, 1:] *
            summed_correlation_mats_eigvecs * summed_correlation_mats_eigvecs.H)
        proj_coeffs_true = np.dot(
            adj_modes.conj().T, vec_array_proj)
        adv_proj_coeffs_true = np.dot(
            adj_modes.conj().T, adv_vec_array_proj)
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            adv_proj_coeffs, adv_proj_coeffs_true, rtol=rtol, atol=atol)

        # Create more data, to check a non-sequential dataset
        adv_vec_array = parallel.call_and_bcast(np.random.random,
            ((self.num_states, self.num_vecs)))
        if parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.adv_vec_handles):
                handle.put(np.array(adv_vec_array[:, vec_index]).squeeze())

        # Compute DMD manually and then set the data in a TLSqrDMDHandles
        # object.  This way, we test only the task of computing the projection
        # coefficients, and not also the task of computing the decomposition.
        (modes_exact, modes_proj, eigvals, R_low_order_eigvecs,
        L_low_order_eigvecs, summed_correlation_mats_eigvals,
        summed_correlation_mats_eigvecs, proj_correlation_mat_eigvals,
        proj_correlation_mat_eigvecs, cross_correlation_mat, adj_modes) =\
            self._helper_compute_DMD_from_data(
            vec_array, util.InnerProductBlock(np.vdot),
            adv_vec_array=adv_vec_array, max_num_eigvals=self.max_num_eigvals)
        self.my_DMD.L_low_order_eigvecs = L_low_order_eigvecs
        self.my_DMD.proj_correlation_mat_eigvals = proj_correlation_mat_eigvals
        self.my_DMD.proj_correlation_mat_eigvecs = proj_correlation_mat_eigvecs
        self.my_DMD.summed_correlation_mats_eigvecs =\
            summed_correlation_mats_eigvecs
        self.my_DMD.cross_correlation_mat = cross_correlation_mat

        # Check the spectral coefficient values.  Compare the formula
        # implemented in modred to a direct projection onto the adjoint modes.
        parallel.barrier()
        proj_coeffs, adv_proj_coeffs= self.my_DMD.compute_proj_coeffs()
        vec_array_proj = np.array(
            vec_array *
            summed_correlation_mats_eigvecs * summed_correlation_mats_eigvecs.H)
        adv_vec_array_proj = np.array(
            adv_vec_array *
            summed_correlation_mats_eigvecs * summed_correlation_mats_eigvecs.H)
        proj_coeffs_true = np.dot(adj_modes.conj().T, vec_array_proj)
        adv_proj_coeffs_true = np.dot(adj_modes.conj().T, adv_vec_array_proj)
        np.testing.assert_allclose(
            proj_coeffs, proj_coeffs_true, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            adv_proj_coeffs, adv_proj_coeffs_true,rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
