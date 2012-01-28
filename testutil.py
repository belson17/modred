#!/usr/bin/env python

import unittest
import subprocess as SP
import os
import numpy as N
import util

try: 
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    distributed = MPI.COMM_WORLD.Get_size() > 1
except ImportError:
    print 'Warning: without mpi4py module, only serial behavior is tested'
    distributed = False
    rank = 0

if rank==0:
    print 'To fully test, must do both:'
    print '  1) python testutil.py'
    print '  2) mpiexec -n <# procs> python testutil.py\n\n'

class TestUtil(unittest.TestCase):
    """Tests all of the functions in util.py
    
    To test all parallel features, use "mpiexec -n 2 python testutil.py"
    """    
    def setUp(self):
        self.testDir = 'files_modaldecomp_test/'
        if rank == 0:
            if not os.path.isdir(self.testDir):
                SP.call(['mkdir', self.testDir])
    
    def tearDown(self):
        if distributed:
            MPI.COMM_WORLD.barrier()
        if rank == 0:
            SP.call(['rm -rf %s/*' % self.testDir], shell=True)
        if distributed:
            MPI.COMM_WORLD.barrier()
        
        
    @unittest.skipIf(distributed, 'Only save/load matrices in serial')
    def test_load_save_mat_text(self):
        """Test that can read/write text matrices"""
        tol = 8
        rows = [1, 5, 20]
        cols = [1, 4, 5, 23]
        matPath = self.testDir+'testMatrix.txt'
        delimiters = [',',' ',';']
        for delimiter in delimiters:
            for is_complex in [False, True]:
                for squeeze in [False, True]:
                    for numRows in rows:
                        for numCols in cols:
                            mat_real = N.random.random((numRows,numCols))
                            if is_complex:
                                mat_imag = N.random.random((numRows,numCols))
                                mat = mat_real + 1J*mat_imag
                            else:
                                mat = mat_real
                            # Check row and column vectors, no squeeze (1,1)
                            if squeeze and (numRows > 1 or numCols > 1):
                                mat = N.squeeze(mat)
                            util.save_mat_text(mat, matPath, delimiter=delimiter)
                            matRead = util.load_mat_text(matPath, delimiter=delimiter,
                                is_complex=is_complex)
                            if squeeze:
                                matRead = N.squeeze(matRead)
                            N.testing.assert_array_almost_equal(matRead, mat, 
                                decimal=tol)
                          
                          
    @unittest.skipIf(distributed, 'Only load matrices in serial')
    def test_svd(self):
        num_internals_list = [10,50]
        num_rows_list = [3,5,40]
        num_cols_list = [1,9,70]
        for num_rows in num_rows_list:
            for num_cols in num_cols_list:
                for num_internals in num_internals_list:
                    left_mat = N.mat(N.random.random((num_rows, num_internals)))
                    right_mat = N.mat(N.random.random((num_internals, num_cols)))
                    full_mat = left_mat*right_mat
                    [L_sing_vecs, sing_vals, R_sing_vecs] = util.svd(full_mat)
                    
                    U, E, V_comp_conj = N.linalg.svd(full_mat, full_matrices=0)
                    V = N.mat(V_comp_conj).H
                    if num_internals < num_rows or num_internals <num_cols:
                        U = U[:,:num_internals]
                        V = V[:,:num_internals]
                        E = E[:num_internals]
          
                    N.testing.assert_array_almost_equal(L_sing_vecs, U)
                    N.testing.assert_array_almost_equal(sing_vals, E)
                    N.testing.assert_array_almost_equal(R_sing_vecs, V)
    
    
        
    @unittest.skipIf(distributed, 'Only load data in serial')
    def test_load_impulse_outputs(self):
        """
        Test loading impulse outputs in [t out1 out2 ...] format.
        
        Creates outputs, saves them, loads them from ERA instance, tests the loaded outputs
        are equal to the originals.
        Returns time_values and outputs at time values in 3D array with indices [time,output,input].
        That is, outputs[time_index] = Markov parameter at time_values[time_index].
        """
        impulse_file_path = self.testDir+'in%03d_to_outs.txt'
        num_time_steps = 150
        for num_states in [4,10]:
            for num_inputs in [1, 4]:
                for num_outputs in [1, 2, 4, 5]:
                    outputs_true = N.random.random((num_time_steps, num_outputs, num_inputs))
                    time_values_true = N.random.random(num_time_steps)
                    
                    impulse_file_paths = []
                    # Save Markov parameters to file
                    for input_num in range(num_inputs):
                        impulse_file_paths.append(impulse_file_path%input_num)
                        data_to_save = N.concatenate( \
                          (time_values_true.reshape(len(time_values_true),1),
                          outputs_true[:,:,input_num]), axis=1)
                        util.save_mat_text(data_to_save, impulse_file_path%input_num)
                    
                    time_values, outputs = util.load_impulse_outputs(impulse_file_paths)
                    N.testing.assert_allclose(outputs, outputs_true)
                    N.testing.assert_allclose(time_values, time_values_true)

    
    
    def test_solve_Lyapunov(self):
        """Test solution of Lyapunov w/known solution from Maltab's dlyap"""
        A = N.array([[1., 2.],[3., 4.]])
        Q = N.array([[4., 3.], [1., 2.]])
        X_true = N.array([[2.2777777777, -0.5],[-1.166666666666667, -0.166666666666667]])
        X_computed = util.solve_Lyapunov(A,Q)
        N.testing.assert_array_almost_equal(X_computed, X_true)
        X_computed_mats = util.solve_Lyapunov(N.mat(A), N.mat(Q))
        N.testing.assert_array_almost_equal(X_computed_mats, X_true)    
    
    
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
    
    
    def test_lsim(self):
        """Test that lsim has right shapes, does not test result"""
        for num_states in [1, 4, 9]:
            for num_inputs in [1, 2, 4]:
                for num_outputs in [1, 2, 3, 5]:
                    print 'num_states %d, num_inputs %d, num_outputs %d'%(num_states, num_inputs, num_outputs)
                    A,B,C = util.drss(num_states, num_inputs, num_outputs)
                    print 'Shape of C is',C.shape
                    inputs = N.random.random((3,num_inputs))
                    outputs = util.lsim(A,B,C,inputs)
                    self.assertEqual(outputs.shape, (3, num_outputs))
                    
                    
    
    def test_impulse(self):
        """Test impulse response of discrete system"""
        for num_states in [1, 10]:
            for num_inputs in [1, 3]:
                for num_outputs in [1, 2, 3, 5]:
                    for time_step in [1, 2, 4]:
                        A, B, C = util.drss(num_states, num_inputs, num_outputs)
                        # Check that can give time_step
                        time_steps, outputs = util.impulse(A, B, C, time_step=time_step)
                        num_time_steps = len(time_steps)
                        time_steps_true = N.arange(0,num_time_steps*time_step,time_step)
                        N.testing.assert_array_equal(time_steps, time_steps_true)
                        outputs_true = N.zeros((num_time_steps, num_outputs, num_inputs))
                        for ti,tv in enumerate(time_steps):
                            outputs_true[ti] = C*(A**tv)*B
                        N.testing.assert_array_equal(outputs, outputs_true)
                        
                        # Check can give time_steps
                        time_steps, outputs = util.impulse(A,B,C,time_steps=time_steps)
                        N.testing.assert_array_equal(time_steps, time_steps_true)
                        N.testing.assert_array_equal(outputs, outputs_true)
                        
                        # Check can give arbitrary time steps (even out of order)
                        time_steps = N.zeros(num_time_steps,dtype=int)
                        for i in range(num_time_steps):
                            time_steps[i] = int((N.random.random()*10000)%100)
                        time_steps, outputs = util.impulse(A,B,C,time_steps=time_steps)
                        outputs_true = N.zeros((num_time_steps, num_outputs, num_inputs))
                        for ti,tv in enumerate(time_steps):
                            outputs_true[ti] = C*(A**tv)*B
                        N.testing.assert_array_equal(outputs, outputs_true)
                        
        
        
    
if __name__=='__main__':
    unittest.main(verbosity=2)


