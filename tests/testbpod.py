#!/usr/bin/env python
"""Test the bpod module"""

import unittest
import copy
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path(join(join(os.path.dirname(os.path.abspath(__file__)), 
    '..', 'src')))
import parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance

from bpod import BPOD
from vectorspace import VectorSpace
import util
import vectors as V

class TestBPOD(unittest.TestCase):
    """Test the BPOD class methods """
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
    
        self.test_dir = 'DELETE_ME_test_files_bpod'
        if not os.path.isdir(self.test_dir):
            _parallel.call_from_rank_zero(os.mkdir, self.test_dir)
        
        self.mode_nums = [2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.num_direct_vecs = 40
        self.num_adjoint_vecs = 45
        self.num_states = 100
        self.index_from = 2
        
        self.my_BPOD = BPOD(N.vdot, verbosity=0)
        self.generate_data_set()
        _parallel.barrier()

    def tearDown(self):
        _parallel.barrier()
        _parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
    
    def generate_data_set(self):
        # create data set (saved to file)
        self.direct_vec_path = join(self.test_dir, 'direct_vec_%03d.txt')
        self.adjoint_vec_path = join(self.test_dir, 'adjoint_vec_%03d.txt')

        self.direct_vec_handles = [V.ArrayTextVecHandle(self.direct_vec_path%i) 
            for i in range(self.num_direct_vecs)]
        self.adjoint_vec_handles = [
            V.ArrayTextVecHandle(self.adjoint_vec_path%i) 
            for i in range(self.num_adjoint_vecs)]
        
        self.direct_vec_array = _parallel.call_and_bcast(N.random.random,
            (self.num_states, self.num_direct_vecs))
        self.adjoint_vec_array = _parallel.call_and_bcast(N.random.random,
            (self.num_states, self.num_adjoint_vecs)) 
        if _parallel.is_rank_zero():
            for i, handle in enumerate(self.direct_vec_handles):
                handle.put(self.direct_vec_array[:, i])
            for i, handle in enumerate(self.adjoint_vec_handles):
                handle.put(self.adjoint_vec_array[:, i])
        self.direct_vecs = [self.direct_vec_array[:, i] 
            for i in range(self.num_direct_vecs)]
        self.adjoint_vecs = [self.adjoint_vec_array[:, i] 
            for i in range(self.num_adjoint_vecs)]
        
        self.Hankel_mat_true = N.dot(self.adjoint_vec_array.T, 
            self.direct_vec_array)
        
        # Do the SVD on all procs.
        self.L_sing_vecs_true, self.sing_vals_true, self.R_sing_vecs_true = \
            util.svd(self.Hankel_mat_true)
        self.direct_mode_array = self.direct_vec_array * \
            N.mat(self.R_sing_vecs_true) * \
            N.mat(N.diag(self.sing_vals_true ** -0.5))
        self.adjoint_mode_array = self.adjoint_vec_array * \
            N.mat(self.L_sing_vecs_true) *\
            N.mat(N.diag(self.sing_vals_true ** -0.5))
        
        
        
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        
        def my_load(fname): pass
        def my_save(data, fname): pass 
        def my_IP(vec1, vec2): pass
        
        data_members_default = {'put_mat': util.save_array_text, 'get_mat':
             util.load_array_text,
            'verbosity': 0, 'L_sing_vecs': None, 'R_sing_vecs': None,
            'sing_vals': None, 'direct_vec_handles': None,
            'adjoint_vec_handles': None, 
            'direct_vecs': None, 'adjoint_vecs': None, 'Hankel_mat': None,
            'vec_space': VectorSpace(inner_product=my_IP, verbosity=False)}
        
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        #self.max_diff = None
        self.assertEqual(util.get_data_members(BPOD(my_IP, verbosity=False)),
            data_members_default)
        
        
        my_BPOD = BPOD(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpace(inner_product=my_IP,
            verbosity=False)
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
       
        my_BPOD = BPOD(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
 
        my_BPOD = BPOD(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
        
        max_vecs_per_node = 500
        my_BPOD = BPOD(my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * _parallel.get_num_nodes() / _parallel.\
            get_num_procs()
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
       
    #@unittest.skip('testing others')
    def test_compute_decomp(self):
        """
        Test that can take vecs, compute the Hankel and SVD matrices
        
        With previously generated random vecs, compute the Hankel
        matrix, then take the SVD. The computed matrices are saved, then
        loaded and compared to the true matrices. 
        """
        tol = 1e-6
        L_sing_vecs_path = join(self.test_dir, 'L_sing_vecs.txt')
        R_sing_vecs_path = join(self.test_dir, 'R_sing_vecs.txt')
        sing_vals_path = join(self.test_dir, 'sing_vals.txt')
        Hankel_mat_path = join(self.test_dir, 'hankel.txt')
        
        L_sing_vecs_return, sing_vals_return, R_sing_vecs_return = \
            self.my_BPOD.compute_decomp(self.direct_vec_handles, 
                self.adjoint_vec_handles)
        L_sing_vecs_return2, sing_vals_return2, R_sing_vecs_return2 = \
            self.my_BPOD.compute_decomp_in_memory(self.direct_vecs, 
                self.adjoint_vecs)
        N.testing.assert_equal(L_sing_vecs_return, L_sing_vecs_return2)
        N.testing.assert_equal(R_sing_vecs_return, R_sing_vecs_return2)
        N.testing.assert_equal(sing_vals_return, sing_vals_return2)
        
        self.my_BPOD.put_decomp(L_sing_vecs_path, sing_vals_path, 
            R_sing_vecs_path)        
        self.my_BPOD.put_Hankel_mat(Hankel_mat_path)
        
        _parallel.barrier()
        L_sing_vecs_loaded = util.load_array_text(L_sing_vecs_path)
        R_sing_vecs_loaded = util.load_array_text(R_sing_vecs_path)
        sing_vals_loaded = N.squeeze(N.array(util.load_array_text(
            sing_vals_path)))
        Hankel_mat_loaded = util.load_array_text(Hankel_mat_path)
        
        N.testing.assert_allclose(self.my_BPOD.Hankel_mat,
            self.Hankel_mat_true, rtol=tol)
        N.testing.assert_allclose(self.my_BPOD.L_sing_vecs,
            self.L_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(self.my_BPOD.R_sing_vecs,
            self.R_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(self.my_BPOD.sing_vals,
            self.sing_vals_true, rtol=tol)
        
        N.testing.assert_allclose(L_sing_vecs_return,
            self.L_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(R_sing_vecs_return,
            self.R_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(sing_vals_return,
            self.sing_vals_true, rtol=tol)
        
        N.testing.assert_allclose(Hankel_mat_loaded,
            self.Hankel_mat_true, rtol=tol)
        N.testing.assert_allclose(L_sing_vecs_loaded,
            self.L_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(R_sing_vecs_loaded,
            self.R_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(sing_vals_loaded,
            self.sing_vals_true, rtol=tol)
        

    #@unittest.skip('testing others')
    def test_compute_modes(self):
        """
        Test computing modes in serial and parallel. 
        
        This method uses the existing random data set saved to disk. It tests
        that BPOD can generate the modes, save them, and load them, then
        compares them to the known solution.
        """

        direct_mode_path = join(self.test_dir, 'direct_mode_%03d.txt')
        adjoint_mode_path = join(self.test_dir, 'adjoint_mode_%03d.txt')
        
        # starts with the CORRECT decomposition.
        self.my_BPOD.R_sing_vecs = self.R_sing_vecs_true
        self.my_BPOD.L_sing_vecs = self.L_sing_vecs_true
        self.my_BPOD.sing_vals = self.sing_vals_true
        
        direct_mode_handles = [V.ArrayTextVecHandle(direct_mode_path%i) 
            for i in self.mode_nums]
        adjoint_mode_handles = [V.ArrayTextVecHandle(adjoint_mode_path%i)
            for i in self.mode_nums]

        self.my_BPOD.compute_direct_modes(self.mode_nums, direct_mode_handles,
            index_from=self.index_from, 
            direct_vec_handles=self.direct_vec_handles)
        self.my_BPOD.compute_adjoint_modes(self.mode_nums, 
            adjoint_mode_handles,
            index_from=self.index_from, 
            adjoint_vec_handles=self.adjoint_vec_handles)
        
        my_BPOD_in_memory = BPOD(inner_product=N.vdot, verbosity=0)

        # start with the CORRECT decomposition.
        my_BPOD_in_memory.R_sing_vecs = self.R_sing_vecs_true
        my_BPOD_in_memory.L_sing_vecs = self.L_sing_vecs_true
        my_BPOD_in_memory.sing_vals = self.sing_vals_true
        
        direct_modes_returned = \
            my_BPOD_in_memory.compute_direct_modes_in_memory(
                self.mode_nums, direct_vecs=self.direct_vecs,
                index_from=self.index_from)
        adjoint_modes_returned = \
            my_BPOD_in_memory.compute_adjoint_modes_in_memory(
                self.mode_nums, adjoint_vecs=self.adjoint_vecs, 
                index_from=self.index_from)

        _parallel.barrier()
        for mode_index, mode_handle in enumerate(direct_mode_handles):
            mode = mode_handle.get()
            N.testing.assert_allclose(mode, 
                self.direct_mode_array[:,self.mode_nums[mode_index] - 
                self.index_from])
            N.testing.assert_allclose(
                direct_modes_returned[mode_index].squeeze(), 
                N.array(self.direct_mode_array[:,self.mode_nums[mode_index] - \
                    self.index_from]).squeeze())

        for mode_index, mode_handle in enumerate(adjoint_mode_handles):
            mode = mode_handle.get()
            N.testing.assert_allclose(mode, 
                self.adjoint_mode_array[:,self.mode_nums[mode_index] - \
                    self.index_from])
            N.testing.assert_allclose(
                adjoint_modes_returned[mode_index].squeeze(), 
                N.array(self.adjoint_mode_array[:,self.mode_nums[mode_index] - 
                    self.index_from]).squeeze())
        
        for direct_mode_index, direct_handle in \
            enumerate(direct_mode_handles):
            direct_mode = direct_handle.get()
            for adjoint_mode_index, adjoint_handle in \
                enumerate(adjoint_mode_handles):
                adjoint_mode = adjoint_handle.get()
                IP = self.my_BPOD.vec_space.inner_product(direct_mode, 
                    adjoint_mode)
                if self.mode_nums[direct_mode_index] != \
                    self.mode_nums[adjoint_mode_index]:
                    self.assertAlmostEqual(IP, 0.)
                else:
                    self.assertAlmostEqual(IP, 1.)
      
      
if __name__ == '__main__':
    unittest.main()

