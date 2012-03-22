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
        self.pod = POD(get_vec=util.load_mat_text, put_vec=
            util.save_mat_text, put_mat=util.save_mat_text, inner_product=
            util.inner_product, verbose=False)
        self.generate_data_set()
        parallel.sync()

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.sync()

    def generate_data_set(self):
        # create data set (saved to file)
        self.vec_path = join(self.test_dir, 'vec_%03d.txt')
        self.vec_paths = []
        
        if parallel.is_rank_zero():
            self.vec_mat = N.mat(N.random.random((self.num_states,self.
                num_vecs)))
            for vec_index in range(self.num_vecs):
                util.save_mat_text(self.vec_mat[:, vec_index], self.vec_path %
                    vec_index)
                self.vec_paths.append(self.vec_path % vec_index)
        else:
            self.vec_paths = None
            self.vec_mat = None
        if parallel.is_distributed():
            self.vec_paths = parallel.comm.bcast(self.vec_paths,root=0)
            self.vec_mat = parallel.comm.bcast(self.vec_mat,root=0)
         
        self.correlation_mat_true = self.vec_mat.T * self.vec_mat
        
        #Do the SVD on all procs.
        self.sing_vecs_true, self.sing_vals_true, dummy = util.svd(self.\
            correlation_mat_true)
        # Use N.dot ?
        self.mode_mat = self.vec_mat * N.mat(self.sing_vecs_true) * N.mat(N.diag(
            self.sing_vals_true ** -0.5))

     
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        data_members_default = {'put_mat': util.save_mat_text, 'get_mat': 
            util.load_mat_text, 'parallel': parallel_mod.default_instance,
            'verbose': False, 'sing_vecs': None, 'sing_vals': None,
            'correlation_mat': None, 'vec_sources': None,
            'vec_ops': VecOperations(get_vec=None, put_vec=None,
            inner_product=None, max_vecs_per_node=2, verbose=False)}
        
        self.assertEqual(util.get_data_members(POD(verbose=False)), 
            data_members_default)

        def my_load(fname): pass
        my_POD = POD(get_vec=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'].get_vec = my_load
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
       
        my_POD = POD(get_mat=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
 
        def my_save(data, fname): pass 
        my_POD = POD(put_vec=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'].put_vec = my_save
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
        
        my_POD = POD(put_mat=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(my_POD), data_members_modified)
        
        def my_ip(f1, f2): pass
        my_POD = POD(inner_product=my_ip, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'].inner_product = my_ip
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
        tol = 1e-8
        vec_path = join(self.test_dir, 'vec_%03d.txt')
        sing_vecs_path = join(self.test_dir, 'sing_vecs.txt')
        sing_vals_path = join(self.test_dir, 'sing_vals.txt')
        correlation_mat_path = join(self.test_dir, 'correlation.txt')
        
        self.pod.compute_decomp(self.vec_paths)
        self.pod.put_correlation_mat(correlation_mat_path)
        self.pod.put_decomp(sing_vecs_path, sing_vals_path)
        
        if parallel.is_rank_zero():
            sing_vecs_loaded = util.load_mat_text(sing_vecs_path)
            sing_vals_loaded = N.squeeze(N.array(util.load_mat_text(
                sing_vals_path)))
            correlation_mat_loaded = util.load_mat_text(correlation_mat_path)
        else:
            sing_vecs_loaded = None
            sing_vals_loaded = None
            correlation_mat_loaded = None

        if parallel.is_distributed():
            sing_vecs_loaded = parallel.comm.bcast(sing_vecs_loaded, root=0)
            sing_vals_loaded = parallel.comm.bcast(sing_vals_loaded, root=0)
            correlation_mat_loaded = parallel.comm.bcast(correlation_mat_loaded,
                root=0)
        
        N.testing.assert_allclose(self.pod.correlation_mat, 
            self.correlation_mat_true, rtol=tol)
        N.testing.assert_allclose(self.pod.sing_vecs, 
            self.sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(self.pod.sing_vals, 
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
        
        # starts with the CORRECT decomposition.
        self.pod.sing_vecs = self.sing_vecs_true
        self.pod.sing_vals = self.sing_vals_true
        
        self.pod.compute_modes(self.mode_nums, 
            [mode_path%i for i in self.mode_nums], 
            index_from=self.index_from, vec_sources=self.vec_paths)
          
        for mode_num in self.mode_nums:
            if parallel.is_rank_zero():
                mode = util.load_mat_text(mode_path % mode_num)
            else:
                mode = None
            if parallel.is_distributed():
                mode = parallel.comm.bcast(mode, root=0)
            N.testing.assert_allclose(mode, 
                self.mode_mat[:,mode_num - self.index_from])
        
        if parallel.is_rank_zero():
            for mode_num1 in self.mode_nums:
                mode1 = util.load_mat_text(mode_path % mode_num1)
                for mode_num2 in self.mode_nums:
                    mode2 = util.load_mat_text(mode_path % mode_num2)
                    IP = self.pod.vec_ops.inner_product(mode1, mode2)
                    if mode_num1 != mode_num2:
                        self.assertAlmostEqual(IP, 0.)
                    else:
                        self.assertAlmostEqual(IP, 1.)


if __name__=='__main__':
    unittest.main()
