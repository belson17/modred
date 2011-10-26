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
    
    
if __name__=='__main__':
    unittest.main(verbosity=2)


