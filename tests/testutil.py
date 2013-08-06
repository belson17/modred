#!/usr/bin/env python
"""Test util module"""

import unittest
# For deleting directories and their contents
from shutil import rmtree 
import os
from os.path import join
import numpy as N

import helper
helper.add_to_path(join(join(os.path.dirname(os.path.abspath(__file__)), 
    '..', 'src')))

import parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
import util


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
        
        
    @unittest.skipIf(_parallel.is_distributed(), 'Only save/load matrices in serial')
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
                            mat_real = N.random.random((num_rows, num_cols))
                            if is_complex:
                                mat_imag = N.random.random((num_rows, num_cols))
                                mat = mat_real + 1J*mat_imag
                            else:
                                mat = mat_real
                            # Check row and column vectors, no squeeze (1,1)
                            if squeeze and (num_rows > 1 or num_cols > 1):
                                mat = N.squeeze(mat)
                            util.save_array_text(mat, mat_path, delimiter=delimiter)
                            mat_read = util.load_array_text(mat_path, delimiter=delimiter,
                                is_complex=is_complex)
                            if squeeze:
                                mat_read = N.squeeze(mat_read)
                            N.testing.assert_allclose(mat_read, mat)#,rtol=tol)
                          
                          
    @unittest.skipIf(_parallel.is_distributed(), 'Only load matrices in serial')
    def test_svd(self):
        num_internals_list = [10, 50]
        num_rows_list = [3, 5, 40]
        num_cols_list = [1, 9, 70]
        for num_rows in num_rows_list:
            for num_cols in num_cols_list:
                for num_internals in num_internals_list:
                    left_mat = N.mat(N.random.random((num_rows, num_internals)))
                    right_mat = N.mat(N.random.random((num_internals, num_cols)))
                    full_mat = left_mat*right_mat
                    L_sing_vecs, sing_vals, R_sing_vecs = util.svd(full_mat)
                    
                    U, E, V_comp_conj = N.linalg.svd(full_mat, full_matrices=0)
                    V = N.mat(V_comp_conj).H
                    if num_internals < num_rows or num_internals < num_cols:
                        U = U[:,:num_internals]
                        V = V[:,:num_internals]
                        E = E[:num_internals]
          
                    N.testing.assert_allclose(L_sing_vecs, U)
                    N.testing.assert_allclose(sing_vals, E)
                    N.testing.assert_allclose(R_sing_vecs, V)
    
    
        
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
                    all_signals_true = N.random.random((num_paths,
                        num_time_steps, num_signals))
                    # Time steps need not be sequential
                    time_values_true = N.random.random(num_time_steps)
                    
                    signal_paths = []
                    # Save signals to file
                    for path_num in range(num_paths):
                        signal_paths.append(signal_path%path_num)
                        data_to_save = N.concatenate( \
                          (time_values_true.reshape(len(time_values_true), 1),
                          all_signals_true[path_num]), axis=1)
                        util.save_array_text(data_to_save, signal_path%path_num)
                    
                    time_values, all_signals = util.load_multiple_signals(
                        signal_paths)
                    N.testing.assert_allclose(all_signals, all_signals_true)
                    N.testing.assert_allclose(time_values, time_values_true)
    
    
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')    
    def test_solve_Lyapunov(self):
        """Test solution of Lyapunov w/known solution from Matlab's dlyap"""
        A = N.array([[0.725404224946106, 0.714742903826096],
                    [-0.063054873189656, -0.204966058299775]])
        Q = N.array([[0.318765239858981, -0.433592022305684],
                    [-1.307688296305273, 0.342624466538650]])
        X_true = N.array([[-0.601761400231752, -0.351368789021923],
                          [-1.143398707577891, 0.334986522655114]])
        X_computed = util.solve_Lyapunov_direct(A, Q)
        N.testing.assert_allclose(X_computed, X_true)
        X_computed_mats = util.solve_Lyapunov_direct(N.mat(A), N.mat(Q))
        N.testing.assert_allclose(X_computed_mats, X_true)    

        X_computed = util.solve_Lyapunov_iterative(A, Q)
        N.testing.assert_allclose(X_computed, X_true)
        X_computed_mats = util.solve_Lyapunov_iterative(N.mat(A), N.mat(Q))
        N.testing.assert_allclose(X_computed_mats, X_true)    
    
    
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
                    N.testing.assert_allclose(yr, y, rtol=1e-5)
    
    
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
                    self.assertTrue(N.amax(N.abs(N.linalg.eig(A)[0])) < 1)
    
    
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only.')
    def test_lsim(self):
        """Test that lsim has right shapes, does not test result"""
        for num_states in [1, 4, 9]:
            for num_inputs in [1, 2, 4]:
                for num_outputs in [1, 2, 3, 5]:
                    #print 'num_states %d, num_inputs %d, num_outputs %d'%(num_states, num_inputs, num_outputs)
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    #print 'Shape of C is',C.shape
                    nt = 5
                    inputs = N.random.random((nt, num_inputs))
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
                    outputs_true = N.zeros((num_time_steps, num_outputs, num_inputs))
                    for ti in range(num_time_steps):
                        outputs_true[ti] = C * (A**ti) * B
                    N.testing.assert_allclose(outputs, outputs_true)
                    
                    # Check can give num_time_steps as an argument
                    outputs = util.impulse(A, B, C, num_time_steps=num_time_steps)
                    N.testing.assert_allclose(outputs, outputs_true)
                    
    
if __name__ == '__main__':
    unittest.main()
