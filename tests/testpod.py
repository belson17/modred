#!/usr/bin/env python

import unittest
import os
import numpy as N
from os.path import join
from shutil import rmtree
import copy

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

from pod import POD
from vecoperations import VecOperations
import vectors as V
import util


class TestPOD(unittest.TestCase):
    """ Test all the POD class methods """
    
    def setUp(self):
        self.test_dir ='DELETE_ME_test_files_pod'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():        
            os.mkdir(self.test_dir)
        self.mode_nums =[2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.num_vecs = 40
        self.num_states = 100
        self.index_from = 2
        
        self.my_POD = POD(inner_product=N.vdot, verbose=False)
        self.generate_data_set()
        parallel.sync()

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.sync()

    def generate_data_set(self):
        """Create data set (saved to file)"""
        self.vec_path = join(self.test_dir, 'vec_%03d.txt')
        self.vec_handles = [V.ArrayTextHandle(self.vec_path%i)
            for i in range(self.num_vecs)]
        
        if parallel.is_rank_zero():
            self.vec_mat = N.mat(N.random.random((self.num_states, 
                self.num_vecs)))
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(self.vec_mat[:, vec_index])
        else:
            self.vec_mat = None
        if parallel.is_distributed():
            self.vec_mat = parallel.comm.bcast(self.vec_mat,root=0)
         
        self.correlation_mat_true = self.vec_mat.T * self.vec_mat
        
        #Do the SVD on all procs.
        self.sing_vecs_true, self.sing_vals_true, dummy = util.svd(
            self.correlation_mat_true)
        # Use N.dot ?
        self.mode_mat = self.vec_mat * N.mat(self.sing_vecs_true) * N.mat(N.diag(
            self.sing_vals_true ** -0.5))

     
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        data_members_default = {'put_mat': util.save_array_text, 'get_mat': 
            util.load_array_text, 'parallel': parallel_mod.default_instance,
            'verbose': False, 'sing_vecs': None, 'sing_vals': None,
            'correlation_mat': None, 'vec_handles': None,
            'vec_ops': VecOperations(verbose=False)}
        
        self.assertEqual(util.get_data_members(POD(verbose=False)), 
            data_members_default)
        
        def my_load(): pass
        def my_save(): pass

        my_POD = POD(verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'] = VecOperations(verbose=False)
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
       
        my_POD = POD(get_mat=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
 
        my_POD = POD(put_mat=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
        
        max_vecs_per_node = 500
        my_POD = POD(max_vecs_per_node=max_vecs_per_node, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'].max_vecs_per_node =\
            max_vecs_per_node
        data_members_modified['vec_ops'].max_vecs_per_proc =\
            max_vecs_per_node * parallel.get_num_nodes() / parallel.\
            get_num_procs()
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
          
        
    def test_compute_decomp(self):
        """
        Test that can take vecs, compute the correlation and SVD matrices
        
        With previously generated random vecs, compute the correlation 
        matrix, then take the SVD. The computed matrices are saved, then
        loaded and compared to the true matrices. 
        """
        tol = 1e-6
        sing_vecs_path = join(self.test_dir, 'sing_vecs.txt')
        sing_vals_path = join(self.test_dir, 'sing_vals.txt')
        correlation_mat_path = join(self.test_dir, 'correlation.txt')
        
        sing_vecs_returned, sing_vals_returned = \
            self.my_POD.compute_decomp(self.vec_handles)
        self.my_POD.put_decomp(sing_vecs_path, sing_vals_path)
        self.my_POD.put_correlation_mat(correlation_mat_path)
        
        sing_vecs_loaded = util.load_array_text(sing_vecs_path)
        sing_vals_loaded = N.squeeze(N.array(util.load_array_text(
            sing_vals_path)))
        correlation_mat_loaded = util.load_array_text(correlation_mat_path)
        
        N.testing.assert_allclose(self.my_POD.correlation_mat, 
            self.correlation_mat_true, rtol=tol)
        N.testing.assert_allclose(self.my_POD.sing_vecs, 
            self.sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(self.my_POD.sing_vals, 
            self.sing_vals_true, rtol=tol)
          
        N.testing.assert_allclose(sing_vecs_returned, 
            self.sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(sing_vals_returned, 
            self.sing_vals_true, rtol=tol)
          
        N.testing.assert_allclose(correlation_mat_loaded, 
            self.correlation_mat_true, rtol=tol)
        N.testing.assert_allclose(sing_vecs_loaded, self.sing_vecs_true,
            rtol=tol)
        N.testing.assert_allclose(sing_vals_loaded, self.sing_vals_true,
            rtol=tol)
        

    def test_compute_modes(self):
        """
        Test computing modes in serial and parallel. 
        
        This method uses the existing random data set saved to disk. It tests
        that POD can generate the modes, save them, and load them, then
        compares them to the known solution.
        """
        mode_path = join(self.test_dir, 'mode_%03d.txt')
        mode_handles = [V.ArrayTextHandle(mode_path%i) for i in self.mode_nums]
        # starts with the CORRECT decomposition.
        self.my_POD.sing_vecs = self.sing_vecs_true
        self.my_POD.sing_vals = self.sing_vals_true
        
        my_POD_in_memory = POD(inner_product=N.vdot, verbose=False)
        my_POD_in_memory.sing_vecs = self.sing_vecs_true
        my_POD_in_memory.sing_vals = self.sing_vals_true
        
        self.my_POD.compute_modes(self.mode_nums, mode_handles, 
            index_from=self.index_from, vec_handles=self.vec_handles)
        
        modes_returned = my_POD_in_memory.compute_modes_and_return(
            self.mode_nums, vec_handles=self.vec_handles, 
            index_from=self.index_from) 
        
        for mode_index, mode_handle in enumerate(mode_handles):
            mode = mode_handle.get()
            N.testing.assert_allclose(mode, 
                self.mode_mat[:,self.mode_nums[mode_index]-self.index_from])
            N.testing.assert_allclose(
                modes_returned[mode_index].squeeze(), 
                N.array(self.mode_mat[:,
                    self.mode_nums[mode_index]-self.index_from]).squeeze())

        
        for mode_index1, handle1 in enumerate(mode_handles):
            mode1 = handle1.get()
            for mode_index2, handle2 in enumerate(mode_handles):
                mode2 = handle2.get()
                IP = self.my_POD.vec_ops.inner_product(mode1, mode2)
                if self.mode_nums[mode_index1] != \
                    self.mode_nums[mode_index2]:
                    self.assertAlmostEqual(IP, 0.)
                else:
                    self.assertAlmostEqual(IP, 1.)


if __name__=='__main__':
    unittest.main()
