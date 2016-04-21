#!/usr/bin/env python
"""Test util module"""
from future.builtins import range
import unittest
# For deleting directories and their contents
from shutil import rmtree 
import os
from os.path import join

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
from modred import util


class TestUtil(unittest.TestCase):
    """Tests all of the functions in util.py
    
    To test all parallel features, use "mpiexec -n 2 python testutil.py"
    """    
    def setUp(self):
        self.test_dir = 'DELETE_ME_test_files_util'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if _parallel.is_rank_zero():
            if not os.path.isdir(self.test_dir):
                os.mkdir(self.test_dir)
    
    def tearDown(self):
        _parallel.barrier()
        if _parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        _parallel.barrier()
        
    @unittest.skipIf(
        _parallel.is_distributed(), 'Only save/load matrices in serial')
    def test_load_save_array_text(self):
        """Test that can read/write text matrices"""
        #tol = 1e-8
        rows = [1, 5, 20]
        cols = [1, 4, 5, 23]
        mat_path = join(self.test_dir, 'test_matrix.txt')
        delimiters = [',', ' ', ';']
        for delimiter in delimiters:
            for is_complex in [False, True]:
                for squeeze in [False, True]:
                    for num_rows in rows:
                        for num_cols in cols:
                            mat_real = np.random.random((num_rows, num_cols))
                            if is_complex:
                                mat_imag = np.random.random(
                                    (num_rows, num_cols))
                                mat = mat_real + 1J*mat_imag
                            else:
                                mat = mat_real
                            # Check row and column vectors, no squeeze (1,1)
                            if squeeze and (num_rows > 1 or num_cols > 1):
                                mat = np.squeeze(mat)
                            util.save_array_text(
                                mat, mat_path, delimiter=delimiter)
                            mat_read = util.load_array_text(
                                mat_path, delimiter=delimiter,
                                is_complex=is_complex)
                            if squeeze:
                                mat_read = np.squeeze(mat_read)
                            np.testing.assert_allclose(mat_read, mat)#,rtol=tol)
    
    @unittest.skipIf(_parallel.is_distributed(), 'Only load matrices in serial')
    def test_svd(self):
        # Set tolerance for testing eigval/eigvec property
        test_tol = 1e-10

        # Check tall, fat, and square matrices
        num_rows_list = [100]
        num_cols_list = [50, 100, 150]

        # Loop through different matrix sizes
        for num_rows in num_rows_list:
            for num_cols in num_cols_list:

                # Generate a random matrix with elements in [0, 1]
                mat = np.random.random((num_rows, num_cols))

                # Compute full set of singular values to help choose tolerance
                # levels that guarantee truncation (otherwise tests won't
                # actually check those features).
                sing_vals_full = np.linalg.svd(mat, full_matrices=0)[1]
                atol_list = [np.median(sing_vals_full), None]
                rtol_list = [
                    np.median(sing_vals_full) / np.max(sing_vals_full), None]

                # Loop through different tolerance cases
                for atol in atol_list:
                    for rtol in rtol_list:
                
                        # For all matrices, check that the output of util.svd
                        # satisfies the definition of an SVD.  Do this by
                        # checking eigval/eigvec properties, which must be
                        # satisfied by the sing vecs and sing vals, even if
                        # there is truncation.  The fact that the singular
                        # vectors are eigenvectors of a normal matrix ensures
                        # that they are unitary, so we don't have to check that
                        # separately.
                        L_sing_vecs, sing_vals, R_sing_vecs = util.svd(
                            mat, atol=atol, rtol=rtol)
                        np.testing.assert_allclose(
                            np.dot(np.dot(mat, mat.T), L_sing_vecs) - 
                            np.dot(L_sing_vecs, np.diag(sing_vals ** 2)), 
                            np.zeros(L_sing_vecs.shape),
                            atol=test_tol)
                        np.testing.assert_allclose(
                            np.dot(np.dot(mat.T, mat), R_sing_vecs) - 
                            np.dot(R_sing_vecs, np.diag(sing_vals ** 2)), 
                            np.zeros(R_sing_vecs.shape),
                            atol=test_tol)
                    
                        # If either tolerance is nonzero, make sure that
                        # something is actually truncated, otherwise force test
                        # to quit.  To do this, make sure the eigvec matrix is
                        # not square.
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
                            self.assertTrue(
                                abs(sing_vals[0]) / abs(sing_vals[-1]) > rtol)
    
    @unittest.skipIf(_parallel.is_distributed(), 'Only load matrices in serial')
    def test_eigh(self):
        # Set tolerance for test of eigval/eigvec properties.  Value necessary
        # for test to pass depends on matrix size, as well as atol and rtol 
        # values
        test_tol = 1e-12

        # Generate random matrix
        num_rows = 100

        # Test matrices that are and are not positive definite
        for is_pos_def in [True, False]:

            # Generate random matrix with values between 0 and 1 
            mat = np.random.random((num_rows, num_rows)) 
           
            # Make matrix symmetric.  Note that if the matrix is large, for
            # some reason an in-place operation causes the operation to fail
            # (not sure why...).  Values are still between 0 and 1.
            mat = 0.5 * (mat + mat.T)
 
            # If necessary, make the matrix positive definite by first making
            # it symmetric (adding the transpose), and then making it
            # diagonally dominant (each element is less than 1, so add N * I to
            # make the diagonal dominant).  Here an in-place change seems to be
            # ok.  
            if is_pos_def:
                mat = mat + num_rows * np.eye(num_rows)

                # Make sure matrix is positive definite, otherwise
                # force test to quit.
                if np.linalg.eig(mat)[0].min() < 0.:
                    raise ValueError(
                        'Failed to generate positive definite matrix '
                        'for test.')
 
            # Compute full set of eigenvalues to help choose tolerance levels
            # that guarantee truncation (otherwise tests won't actually check
            # those features).
            eigvals_full = np.linalg.eig(mat)[0]
            atol_list = [np.median(abs(eigvals_full)), None]
            rtol_list = [
                np.median(abs(eigvals_full)) / abs(np.max(eigvals_full)), None]

            # Loop through different tolerance values
            for atol in atol_list:
                for rtol in rtol_list:

                    # For each case, test that returned values are in fact
                    # eigenvalues and eigenvectors of the given matrix.  Since
                    # each pair that is returned (not truncated due to
                    # tolerances) should have this property, we can test this
                    # even if tolerances are passed in.  Compare to the zero
                    # matrix because then we only have to check the absolute
                    # magnitue of each term, rather than consider relative
                    # errors with respect to nonzero terms.
                    eigvals, eigvecs = util.eigh(
                        mat, rtol=rtol, atol=atol,
                        is_positive_definite=is_pos_def)
                    np.testing.assert_allclose(
                        np.dot(mat, eigvecs) - 
                        np.dot(eigvecs, np.diag(eigvals)), 
                        np.zeros(eigvecs.shape),
                        atol=test_tol)

                    # If either tolerance is nonzero, make sure that something
                    # is actually truncated, otherwise force test to quit.  To
                    # do this, make sure the eigvec matrix is not square.
                    if rtol and eigvals.size == eigvals_full.size:
                        raise ValueError(
                            'Failed to choose relative tolerance that forces '
                            'truncation.')
                    if atol and eigvals.size == eigvals_full.size:
                        raise ValueError(
                            'Failed to choose absolute tolerance that forces '
                            'truncation.')

                    # If the positive definite flag is passed in, make sure the
                    # returned eigenvalues are all positive
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

    @unittest.skipIf(_parallel.is_distributed(), 'Only load matrices in serial')
    def test_eig_biorthog(self):
        test_tol = 1e-10
        num_rows = 100 
        mat = np.random.random((num_rows, num_rows))
        for scale_choice in ['left', 'right']:
            R_eigvals, R_eigvecs, L_eigvecs = util.eig_biorthog(
                mat, scale_choice=scale_choice)
       
            # Check eigenvector/eigenvalue relationship (use right eigenvalues
            # only).  Test difference so that all values are compared to zeros,
            # avoiding need to check relative tolerances.
            np.testing.assert_allclose(
                np.dot(mat, R_eigvecs) - np.dot(R_eigvecs, np.diag(R_eigvals)),
                np.zeros(mat.shape),
                atol=test_tol)
            np.testing.assert_allclose(
                np.dot(L_eigvecs.conj().T, mat) - 
                np.dot(np.diag(R_eigvals), L_eigvecs.conj().T),
                np.zeros(mat.shape),
                atol=test_tol)

            # Check biorthogonality (take magnitudes since inner products are
            # complex in general).  Again, take difference so that all test
            # values should be zero, avoiding need for rtol
            ip_mat = np.dot(L_eigvecs.conj().T, R_eigvecs)
            np.testing.assert_allclose(
                np.abs(ip_mat) - np.eye(num_rows), np.zeros(ip_mat.shape), 
                atol=test_tol)

            # Check for unit norms
            if scale_choice == 'left':
                unit_eigvecs = R_eigvecs
            elif scale_choice == 'right':
                unit_eigvecs = L_eigvecs
            np.testing.assert_allclose(
                np.sqrt(np.sum(np.multiply(
                unit_eigvecs, unit_eigvecs.conj()), axis=0)).squeeze(), 
                np.ones((1, R_eigvals.size)))

        # Check that error is raised for invalid scale choice
        self.assertRaises(
            ValueError, util.eig_biorthog, mat, **{'scale_choice':'invalid'})
        
    @unittest.skipIf(_parallel.is_distributed(), 'Only load data in serial')
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
    
    
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')    
    def test_solve_Lyapunov(self):
        """Test solution of Lyapunov w/known solution from Matlab's dlyap"""
        A = np.array([[0.725404224946106, 0.714742903826096],
                    [-0.063054873189656, -0.204966058299775]])
        Q = np.array([[0.318765239858981, -0.433592022305684],
                    [-1.307688296305273, 0.342624466538650]])
        X_true = np.array([[-0.601761400231752, -0.351368789021923],
                          [-1.143398707577891, 0.334986522655114]])
        X_computed = util.solve_Lyapunov_direct(A, Q)
        np.testing.assert_allclose(X_computed, X_true)
        X_computed_mats = util.solve_Lyapunov_direct(np.mat(A), np.mat(Q))
        np.testing.assert_allclose(X_computed_mats, X_true)    

        X_computed = util.solve_Lyapunov_iterative(A, Q)
        np.testing.assert_allclose(X_computed, X_true)
        X_computed_mats = util.solve_Lyapunov_iterative(np.mat(A), np.mat(Q))
        np.testing.assert_allclose(X_computed_mats, X_true)    
    
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
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
    
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
    def test_drss(self):
        """Test drss gives correct mat dimensions and stable dynamics."""
        for num_states in [1, 5, 14]:
            for num_inputs in [1, 3, 6]:
                for num_outputs in [1, 2, 3, 7]:
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    self.assertEqual(A.shape, (num_states,num_states))
                    self.assertEqual(B.shape, (num_states, num_inputs))
                    self.assertEqual(C.shape, (num_outputs, num_states))
                    self.assertTrue(np.amax(np.abs(np.linalg.eig(A)[0])) < 1)
    
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
    def test_lsim(self):
        """Test that lsim has right shapes, does not test result"""
        for num_states in [1, 4, 9]:
            for num_inputs in [1, 2, 4]:
                for num_outputs in [1, 2, 3, 5]:
                    #print (
                    #    'num_states %d, num_inputs %d, 
                    #    num_outputs %d'%(num_states, num_inputs, num_outputs)
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    #print 'Shape of C is',C.shape
                    nt = 5
                    inputs = np.random.random((nt, num_inputs))
                    outputs = util.lsim(A, B, C, inputs)
                    self.assertEqual(outputs.shape, (nt, num_outputs))

    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
    def test_impulse(self):
        """Test impulse response of discrete system"""
        for num_states in [1, 10]:
            for num_inputs in [1, 3]:
                for num_outputs in [1, 2, 3, 5]:
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    # Check that can give time_step
                    outputs = util.impulse(A, B, C)
                    num_time_steps = len(outputs)
                    outputs_true = np.zeros(
                        (num_time_steps, num_outputs, num_inputs))
                    for ti in range(num_time_steps):
                        outputs_true[ti] = C * (A**ti) * B
                    np.testing.assert_allclose(outputs, outputs_true)
                    
                    # Check can give num_time_steps as an argument
                    outputs = util.impulse(
                        A, B, C, num_time_steps=num_time_steps)
                    np.testing.assert_allclose(outputs, outputs_true)

    def test_Hankel(self):
        """Test forming Hankel matrix from first row and last column."""
        for num_rows in [1, 4, 6]:
            for num_cols in [1, 3, 6]:
                first_row = np.random.random((num_cols))
                last_col = np.random.random((num_rows))
                last_col[0] = first_row[-1]
                Hankel_true = np.zeros((num_rows, num_cols))
                for r in range(num_rows):
                    for c in range(num_cols):
                        if r+c < num_cols:
                            Hankel_true[r,c] = first_row[r+c]
                        else:
                            Hankel_true[r,c] = last_col[r+c-num_cols+1]
                Hankel_comp = util.Hankel(first_row, last_col)
                np.testing.assert_equal(Hankel_comp, Hankel_true)


if __name__ == '__main__':
    unittest.main()
