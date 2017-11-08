#!/usr/bin/env python
"""Test util module"""
import unittest
from shutil import rmtree   # For deleting directories and their contents
import os
from os.path import join

import numpy as np

from modred import util, parallel
from modred.py2to3 import range


class TestUtil(unittest.TestCase):
    """Tests all of the functions in util.py

    To test all parallel features, use "mpiexec -n 2 python testutil.py"
    """
    def setUp(self):
        self.test_dir = 'files_util_DELETE_ME'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if parallel.is_rank_zero():
            if not os.path.isdir(self.test_dir):
                os.mkdir(self.test_dir)


    def tearDown(self):
        parallel.barrier()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.barrier()


    #@unittest.skip('Testing something else.')
    def test_atleast_2d(self):
        # Test a 0d array.  Check that after reshaping to 2d, the value is the
        # same, but the shape is a row/column vector as specified.
        vec0d = np.array(1.)
        vec0d_row = util.atleast_2d_row(vec0d)
        vec0d_col = util.atleast_2d_col(vec0d)
        np.testing.assert_array_equal(vec0d, vec0d_row.squeeze())
        np.testing.assert_array_equal(vec0d, vec0d_col.squeeze())
        self.assertEqual(vec0d_row.shape, (1, 1))
        self.assertEqual(vec0d_col.shape, (1, 1))

        # Test a 1d array.  Check that after reshaping to 2d, the values are the
        # same, but the shape is a row/column vector as specified.
        vec1d = np.ones((3))
        vec1d_row = util.atleast_2d_row(vec1d)
        vec1d_col = util.atleast_2d_col(vec1d)
        np.testing.assert_array_equal(vec1d.squeeze(), vec1d_row.squeeze())
        np.testing.assert_array_equal(vec1d.squeeze(), vec1d_col.squeeze())
        self.assertEqual(vec1d.shape, (vec1d.size,))
        self.assertEqual(vec1d_row.shape, (1, vec1d.size))
        self.assertEqual(vec1d_col.shape, (vec1d.size, 1))

        # Test a 2d array.  Nothing should change about the array.
        vec2d = np.ones((3, 3))
        vec2d_row = util.atleast_2d_row(vec2d)
        vec2d_col = util.atleast_2d_col(vec2d)
        np.testing.assert_array_equal(vec2d, vec2d_row)
        np.testing.assert_array_equal(vec2d, vec2d_col)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(
        parallel.is_distributed(), 'Only save/load arrays in serial')
    def test_load_save_array_text(self):
        """Test that can read/write text arrays"""
        rows = [1, 5, 20]
        cols = [1, 4, 5, 23]
        array_path = join(self.test_dir, 'test_array.txt')
        delimiters = [',', ' ', ';']
        for delimiter in delimiters:
            for is_complex in [False, True]:
                for squeeze in [False, True]:
                    for num_rows in rows:
                        for num_cols in cols:

                            # Generate real and complex arrays
                            array = np.random.random((num_rows, num_cols))
                            if is_complex:
                                array = array + (
                                    1j * np.random.random((num_rows, num_cols)))

                            # Check row and column vectors, no squeeze (1, 1)
                            if squeeze and (num_rows > 1 or num_cols > 1):
                                array = np.squeeze(array)
                            util.save_array_text(
                                array, array_path, delimiter=delimiter)
                            array_read = util.load_array_text(
                                array_path, delimiter=delimiter,
                                is_complex=is_complex)
                            if squeeze:
                                array_read = np.squeeze(array_read)
                            np.testing.assert_equal(array_read, array)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Only load arrays in serial')
    def test_svd(self):
        # Set tolerance for testing eigval/eigvec property
        test_atol = 1e-10

        # Check tall, fat, and square arrays
        num_rows_list = [100]
        num_cols_list = [50, 100, 150]

        # Loop through different array sizes
        for num_rows in num_rows_list:
            for num_cols in num_cols_list:

                # Check real and complex data
                for is_complex in [True]:

                    # Generate a random array with elements in [0, 1]
                    array = np.random.random((num_rows, num_cols))
                    if is_complex:
                        array = array + 1j * np.random.random(
                            (num_rows, num_cols))

                    # Compute full set of singular values to help choose
                    # tolerance levels that guarantee truncation (otherwise
                    # tests won't actually check those features).
                    sing_vals_full = np.linalg.svd(array, full_matrices=0)[1]
                    atol_list = [np.median(sing_vals_full), None]
                    rtol_list = [
                        np.median(sing_vals_full) / np.max(sing_vals_full),
                        None]

                    # Loop through different tolerance cases
                    for atol in atol_list:
                        for rtol in rtol_list:

                            # For all arrays, check that the output of util.svd
                            # satisfies the definition of an SVD.  Do this by
                            # checking eigval/eigvec properties, which must be
                            # satisfied by the sing vecs and sing vals, even if
                            # there is truncation.  The fact that the singular
                            # vectors are eigenvectors of a normal array ensures
                            # that they are unitary, so we don't have to check
                            # that separately.
                            L_sing_vecs, sing_vals, R_sing_vecs = util.svd(
                                array, atol=atol, rtol=rtol)
                            np.testing.assert_allclose(
                                array.dot(array.conj().T.dot(L_sing_vecs)) -
                                L_sing_vecs.dot(np.diag(sing_vals ** 2)),
                                np.zeros(L_sing_vecs.shape),
                                atol=test_atol)
                            np.testing.assert_allclose(
                                array.conj().T.dot(array.dot(R_sing_vecs)) -
                                R_sing_vecs.dot(np.diag(sing_vals ** 2)),
                                np.zeros(R_sing_vecs.shape),
                                atol=test_atol)

                            # If either tolerance is nonzero, make sure that
                            # something is actually truncated, otherwise force
                            # test to quit.  To do this, make sure the eigvec
                            # array is not square.
                            if rtol and sing_vals.size == sing_vals_full.size:
                                raise ValueError(
                                    'Failed to choose relative tolerance that '
                                    'forces truncation.')
                            if atol and sing_vals.size == sing_vals_full.size:
                                raise ValueError(
                                    'Failed to choose absolute tolerance that '
                                    'forces truncation.')

                            # If necessary, test that tolerances are satisfied
                            if atol:
                                self.assertTrue(abs(sing_vals[-1]) > atol)
                            if rtol:
                                self.assertTrue((
                                    abs(sing_vals[0]) / abs(sing_vals[-1])
                                    > rtol))


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Only load arrays in serial')
    def test_eigh(self):
        # Set tolerance for test of eigval/eigvec properties.  Value necessary
        # for test to pass depends on array size, as well as atol and rtol
        # values
        test_atol = 1e-12

        # Generate random array
        num_rows = 100

        # Test arrays that are and are not positive definite
        for is_pos_def in [True, False]:

            # Test both real and complex data
            for is_complex in [True, False]:

                # Generate random array with values between 0 and 1
                array = np.random.random((num_rows, num_rows))
                if is_complex:
                    array = array + 1j * np.random.random((num_rows, num_rows))

                # Make array conjugate-symmetric.  Note that if the array is
                # large, for some reason an in-place operation causes the
                # operation to fail (not sure why...).  Values are still between
                # 0 and 1.
                array = 0.5 * (array + array.conj().T)

                # If necessary, make the array positive definite by first making
                # it symmetric (adding the transpose), and then making it
                # diagonally dominant (each element is less than 1, so add N * I
                # to make the diagonal dominant).  Here an in-place change seems
                # to be ok.
                if is_pos_def:
                    array = array + num_rows * np.eye(num_rows)

                    # Make sure array is positive definite, otherwise
                    # force test to quit.
                    if np.linalg.eig(array)[0].min() < 0.:
                        raise ValueError(
                            'Failed to generate positive definite array '
                            'for test.')

                # Compute full set of eigenvalues to help choose tolerance
                # levels that guarantee truncation (otherwise tests won't
                # actually check those features).
                eigvals_full = np.linalg.eig(array)[0]
                atol_list = [np.median(abs(eigvals_full)), None]
                rtol_list = [
                    np.median(abs(eigvals_full)) / abs(np.max(eigvals_full)),
                    None]

                # Loop through different tolerance values
                for atol in atol_list:
                    for rtol in rtol_list:

                        # For each case, test that returned values are in fact
                        # eigenvalues and eigenvectors of the given array.
                        # Since each pair that is returned (not truncated due to
                        # tolerances) should have this property, we can test
                        # this even if tolerances are passed in.  Compare to the
                        # zero array because then we only have to check the
                        # absolute magnitue of each term, rather than consider
                        # relative errors with respect to nonzero terms.
                        eigvals, eigvecs = util.eigh(
                            array, rtol=rtol, atol=atol,
                            is_positive_definite=is_pos_def)
                        np.testing.assert_allclose(
                            array.dot(eigvecs) -
                            eigvecs.dot(np.diag(eigvals)),
                            np.zeros(eigvecs.shape),
                            atol=test_atol)

                        # If either tolerance is nonzero, make sure that
                        # something is actually truncated, otherwise force test
                        # to quit.  To do this, make sure the eigvec array is
                        # not square.
                        if rtol and eigvals.size == eigvals_full.size:
                            raise ValueError(
                                'Failed to choose relative tolerance that '
                                'forces truncation.')
                        if atol and eigvals.size == eigvals_full.size:
                            raise ValueError(
                                'Failed to choose absolute tolerance that '
                                'forces truncation.')

                        # If the positive definite flag is passed in, make sure
                        # the returned eigenvalues are all positive
                        if is_pos_def:
                            self.assertTrue(eigvals.min() > 0)

                        # If a relative tolerance is passed in, make sure the
                        # relative tolerance is satisfied.
                        if rtol is not None:
                            self.assertTrue(
                                abs(eigvals).min() / abs(eigvals).max() > rtol)

                        # If an absolute tolerance is passed in, make sure the
                        # absolute tolerance is satisfied.
                        if atol is not None:
                            self.assertTrue(abs(eigvals).min() > atol)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Only load arrays in serial')
    def test_eig_biorthog(self):
        test_atol = 1e-10
        num_rows = 100

        # Test real and complex data
        for is_complex in [True, False]:
            array = np.random.random((num_rows, num_rows))
            if is_complex:
                array = array + 1j * np.random.random((num_rows, num_rows))

            # Test different scale choices
            for scale_choice in ['left', 'right']:
                R_eigvals, R_eigvecs, L_eigvecs = util.eig_biorthog(
                    array, scale_choice=scale_choice)

                # Check eigenvector/eigenvalue relationship (use right
                # eigenvalues only).  Test difference so that all values are
                # compared to zeros, avoiding need to check relative tolerances.
                np.testing.assert_allclose(
                    array.dot(R_eigvecs) - R_eigvecs.dot(np.diag(R_eigvals)),
                    np.zeros(array.shape),
                    atol=test_atol)
                np.testing.assert_allclose(
                    L_eigvecs.conj().T.dot(array) - np.diag(R_eigvals).dot(
                        L_eigvecs.conj().T),
                    np.zeros(array.shape),
                    atol=test_atol)

                # Check biorthogonality (take magnitudes since inner products
                # are complex in general).  Again, take difference so that all
                # test values should be zero, avoiding need for rtol
                ip_array = L_eigvecs.conj().T.dot(R_eigvecs)
                np.testing.assert_allclose(
                    np.abs(ip_array) - np.eye(num_rows),
                    np.zeros(ip_array.shape),
                    atol=test_atol)

                # Check for unit norms
                if scale_choice == 'left':
                    unit_eigvecs = R_eigvecs
                elif scale_choice == 'right':
                    unit_eigvecs = L_eigvecs
                np.testing.assert_allclose(
                    np.sqrt(np.sum(np.multiply(
                        unit_eigvecs, unit_eigvecs.conj()), axis=0)).squeeze(),
                    np.ones(R_eigvals.size))

        # Check that error is raised for invalid scale choice
        self.assertRaises(
            ValueError, util.eig_biorthog, array, **{'scale_choice':'invalid'})


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Only load data in serial')
    def test_load_impulse_outputs(self):
        """
        Test loading multiple signal files in [t sig1 sig2 ...] format.

        Creates signals, saves them, loads them, tests are equal to the
        originals.
        """
        signal_path = join(self.test_dir, 'file%03d.txt')
        for num_paths in [1, 4]:
            for num_signals in [1, 2, 4, 5]:
                for num_time_steps in [1, 10, 100]:
                    all_signals_true = np.random.random((num_paths,
                        num_time_steps, num_signals))
                    # Time steps need not be sequential
                    time_values_true = np.random.random(num_time_steps)

                    signal_paths = []
                    # Save signals to file
                    for path_num in range(num_paths):
                        signal_paths.append(signal_path%path_num)
                        data_to_save = np.concatenate( \
                          (time_values_true.reshape(len(time_values_true), 1),
                          all_signals_true[path_num]), axis=1)
                        util.save_array_text(data_to_save, signal_path%path_num)

                    time_values, all_signals = util.load_multiple_signals(
                        signal_paths)
                    np.testing.assert_allclose(all_signals, all_signals_true)
                    np.testing.assert_allclose(time_values, time_values_true)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only.')
    def test_solve_Lyapunov(self):
        """Test solution of Lyapunov w/known solution from Matlab's dlyap"""
        # Generate data
        A = np.array([
            [0.725404224946106, 0.714742903826096],
            [-0.063054873189656, -0.204966058299775]])
        Q = np.array([
            [0.318765239858981, -0.433592022305684],
            [-1.307688296305273, 0.342624466538650]])
        X_true = np.array([
            [-0.601761400231752, -0.351368789021923],
            [-1.143398707577891, 0.334986522655114]])

        # Test direct method
        X_computed = util.solve_Lyapunov_direct(A, Q)
        np.testing.assert_allclose(X_computed, X_true)

        # Test iterative method
        X_computed = util.solve_Lyapunov_iterative(A, Q)
        np.testing.assert_allclose(X_computed, X_true)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only.')
    def test_balanced_truncation(self):
        """Test balanced system is close to original."""
        for num_inputs in [1, 3]:
            for num_outputs in [1, 4]:
                for num_states in [1, 10]:
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    Ar, Br, Cr = util.balanced_truncation(A, B, C)
                    num_time_steps = 10
                    y = util.impulse(A, B, C, num_time_steps=num_time_steps)
                    yr = util.impulse(Ar, Br, Cr, num_time_steps=num_time_steps)
                    np.testing.assert_allclose(yr, y, rtol=1e-5)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only.')
    def test_drss(self):
        """Test drss gives correct array dimensions and stable dynamics."""
        for num_states in [1, 5, 14]:
            for num_inputs in [1, 3, 6]:
                for num_outputs in [1, 2, 3, 7]:
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    self.assertEqual(A.shape, (num_states,num_states))
                    self.assertEqual(B.shape, (num_states, num_inputs))
                    self.assertEqual(C.shape, (num_outputs, num_states))
                    self.assertTrue(np.amax(np.abs(np.linalg.eig(A)[0])) < 1)


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only.')
    def test_lsim(self):
        """Test that lsim has right shapes, does not test result"""
        for num_states in [1, 4, 9]:
            for num_inputs in [1, 2, 4]:
                for num_outputs in [1, 2, 3, 5]:
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    nt = 5
                    inputs = np.random.random((nt, num_inputs))
                    outputs = util.lsim(A, B, C, inputs)
                    self.assertEqual(outputs.shape, (nt, num_outputs))


    #@unittest.skip('Testing something else.')
    @unittest.skipIf(parallel.is_distributed(), 'Serial only.')
    def test_impulse(self):
        """Test impulse response of discrete system"""
        rtol = 1e-10
        atol = 1e-12
        for num_states in [1, 10]:
            for num_inputs in [1, 3]:
                for num_outputs in [1, 2, 3, 5]:
                    # Generate state-space system arrays
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)

                    # Check that can give time_steps as argument
                    outputs = util.impulse(A, B, C)
                    num_time_steps = len(outputs)
                    outputs_true = np.zeros(
                        (num_time_steps, num_outputs, num_inputs))
                    for ti in range(num_time_steps):
                        outputs_true[ti] = C.dot(
                            np.linalg.matrix_power(A, ti).dot(B))
                    np.testing.assert_allclose(
                        outputs, outputs_true, rtol=rtol, atol=atol)

                    # Check can give num_time_steps as an argument
                    outputs = util.impulse(
                        A, B, C, num_time_steps=num_time_steps)
                    np.testing.assert_allclose(
                        outputs, outputs_true, rtol=rtol, atol=atol)


    #@unittest.skip('Testing something else.')
    def test_Hankel(self):
        """Test forming Hankel array from first column and last row."""
        for num_rows in [1, 4, 6]:
            for num_cols in [1, 3, 6]:

                # Generate simple integer values so structure of array is easy
                # to see.  This doesn't affect the robustness of the test, as
                # all we are concerned about is structure.
                first_col = np.arange(1, num_rows + 1)
                last_row = np.arange(1, num_cols + 1) * 10
                last_row[0] = first_col[-1]

                # Fill in Hankel array.  Recall that along skew diagonals, i +
                # j is constant.
                Hankel_true = np.zeros((num_rows, num_cols))
                for i in range(num_rows):
                    for j in range(num_cols):

                        # Upper left triangle of values.  Fill skew diagonals
                        # until we hit the lower left corner of the array, where
                        # i + j = num_rows - 1.
                        if i + j < num_rows:
                            Hankel_true[i, j] = first_col[i + j]

                        # Lower right triangle of values.  Starting on skew
                        # diagonal just to right of lower left corner of array,
                        # fill in rest of values.
                        else:
                            Hankel_true[i, j] = last_row[i + j - num_rows + 1]

                # Compute Hankel array using util and test
                Hankel_test = util.Hankel(first_col, last_row)
                np.testing.assert_equal(Hankel_test, Hankel_true)


    #@unittest.skip('Testing something else.')
    def test_Hankel_chunks(self):
        """Test forming Hankel array using chunks."""
        chunk_num_rows = 2
        chunk_num_cols = 2
        chunk_shape = (chunk_num_rows, chunk_num_cols)
        for num_row_chunks in [1, 4, 6]:
            for num_col_chunks in [1, 3, 6]:

                # Generate simple values that make it easy to see the array
                # structure
                first_col_chunks = [
                    np.ones(chunk_shape) * (i + 1)
                    for i in range(num_row_chunks)]
                last_row_chunks = [
                    np.ones(chunk_shape) * (j + 1) * 10
                    for j in range(num_col_chunks)]
                last_row_chunks[0] = first_col_chunks[-1]

                # Fill in Hankel array chunk by chunk
                Hankel_true = np.zeros((
                    num_row_chunks * chunk_shape[0],
                    num_col_chunks * chunk_shape[1]))
                for i in range(num_row_chunks):
                    for j in range(num_col_chunks):

                        # Upper left triangle of values
                        if i + j < num_row_chunks:
                            Hankel_true[
                                i * chunk_num_rows:(i + 1) * chunk_num_rows,
                                j * chunk_num_cols:(j + 1) * chunk_num_cols] =\
                                first_col_chunks[i + j]

                        # Lower right triangle of values
                        else:
                            Hankel_true[
                                i * chunk_num_rows:(i + 1) * chunk_num_rows,
                                j * chunk_num_cols:(j + 1) * chunk_num_cols] =\
                                last_row_chunks[i + j - num_row_chunks + 1]

                # Compute Hankel array using util and test
                Hankel_test = util.Hankel_chunks(
                    first_col_chunks, last_row_chunks=last_row_chunks)
                np.testing.assert_equal(Hankel_test, Hankel_true)


if __name__ == '__main__':
    unittest.main()
