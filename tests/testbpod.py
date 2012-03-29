#!/usr/bin/env python

import unittest
import copy
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

from bpod import BPOD
from vecoperations import VecOperations
import util
from vecdefs import ArrayText


class TestBPOD(unittest.TestCase):
    """ Test the BPOD class methods """
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
    
        self.test_dir = 'DELETE_ME_test_files_bpod'
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():        
            os.mkdir(self.test_dir)
        
        self.maxDiff = 1000
        self.mode_nums =[2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.num_direct_vecs = 40
        self.num_adjoint_vecs = 45
        self.num_states = 100
        self.index_from = 2
        
        self.my_vec_defs = ArrayText()
        self.my_BPOD = BPOD(self.my_vec_defs, 
            put_mat=util.save_mat_text, verbose=False)
        self.generate_data_set()
        parallel.sync()

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.sync()
    
    def generate_data_set(self):
        # create data set (saved to file)
        self.direct_vec_path = join(self.test_dir, 'direct_vec_%03d.txt')
        self.adjoint_vec_path = join(self.test_dir, 'adjoint_vec_%03d.txt')

        self.direct_vec_paths=[]
        self.adjoint_vec_paths=[]
        
        if parallel.is_rank_zero():
            self.direct_vec_mat = N.mat(N.random.random((self.num_states,
                self.num_direct_vecs)))
            self.adjoint_vec_mat = N.mat(N.random.random((self.num_states,
                self.num_adjoint_vecs))) 
            
            for direct_vec_index in range(self.num_direct_vecs):
                util.save_mat_text(self.direct_vec_mat[:,direct_vec_index],
                    self.direct_vec_path%direct_vec_index)
                self.direct_vec_paths.append(self.direct_vec_path % 
                    direct_vec_index)
            for adjoint_vec_index in range(self.num_adjoint_vecs):
                util.save_mat_text(self.adjoint_vec_mat[:,adjoint_vec_index],
                  self.adjoint_vec_path%adjoint_vec_index)
                self.adjoint_vec_paths.append(self.adjoint_vec_path %
                    adjoint_vec_index)
        else:
            self.direct_vec_paths=None
            self.adjoint_vec_paths=None
            self.direct_vec_mat = None
            self.adjoint_vec_mat = None
        if parallel.is_distributed():
            self.direct_vec_paths = parallel.comm.bcast(
                self.direct_vec_paths, root=0)
            self.adjoint_vec_paths = parallel.comm.bcast(
                self.adjoint_vec_paths, root=0)
            self.direct_vec_mat = parallel.comm.bcast(
                self.direct_vec_mat, root=0)
            self.adjoint_vec_mat = parallel.comm.bcast(
                self.adjoint_vec_mat, root=0)
         
        self.hankel_mat_true = self.adjoint_vec_mat.T * self.direct_vec_mat
        
        #Do the SVD on all procs.
        self.L_sing_vecs_true, self.sing_vals_true, self.R_sing_vecs_true = \
            util.svd(self.hankel_mat_true)
        self.direct_mode_mat = self.direct_vec_mat * \
            N.mat(self.R_sing_vecs_true) * \
            N.mat(N.diag(self.sing_vals_true ** -0.5))
        self.adjoint_mode_mat = self.adjoint_vec_mat * \
            N.mat(self.L_sing_vecs_true) *\
            N.mat(N.diag(self.sing_vals_true ** -0.5))
        
        #self.my_BPOD.direct_vec_paths=self.direct_vec_paths
        #self.my_BPOD.adjoint_vec_paths=self.adjoint_vec_paths
        
        
        
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        
        data_members_default = {'put_mat': util.save_mat_text, 'get_mat':
             util.load_mat_text, 'parallel': parallel_mod.default_instance,
            'verbose': False, 'L_sing_vecs': None, 'R_sing_vecs': None,
            'sing_vals': None, 'direct_vec_sources': None,
            'adjoint_vec_sources': None, 'hankel_mat': None,
            'vec_ops': VecOperations(self.my_vec_defs, verbose=False)}
        
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        self.assertEqual(util.get_data_members(BPOD(self.my_vec_defs, 
            verbose=False)), data_members_default)
        
        def my_load(fname): pass
        def my_save(data, fname): pass 
        
        my_BPOD = BPOD(self.my_vec_defs, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'] = VecOperations(self.my_vec_defs,
            verbose=False)
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
       
        my_BPOD = BPOD(self.my_vec_defs, get_mat=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
 
        my_BPOD = BPOD(self.my_vec_defs, put_mat=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
        
        max_vecs_per_node = 500
        my_BPOD = BPOD(self.my_vec_defs, max_vecs_per_node=max_vecs_per_node,
            verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'].max_vecs_per_node =\
            max_vecs_per_node
        data_members_modified['vec_ops'].max_vecs_per_proc =\
            max_vecs_per_node * parallel.get_num_nodes() / parallel.\
            get_num_procs()
        self.assertEqual(util.get_data_members(my_BPOD), data_members_modified)
       
        
    def test_compute_decomp(self):
        """
        Test that can take vecs, compute the Hankel and SVD matrices
        
        With previously generated random vecs, compute the Hankel
        matrix, then take the SVD. The computed matrices are saved, then
        loaded and compared to the true matrices. 
        """
        tol = 1e-8
        direct_vec_path = join(self.test_dir, 'direct_vec_%03d.txt')
        adjoint_vec_path = join(self.test_dir, 'adjoint_vec_%03d.txt')
        L_sing_vecs_path = join(self.test_dir, 'L_sing_vecs.txt')
        R_sing_vecs_path = join(self.test_dir, 'R_sing_vecs.txt')
        sing_vals_path = join(self.test_dir, 'sing_vals.txt')
        hankel_mat_path = join(self.test_dir, 'hankel.txt')
        
        self.my_BPOD.compute_decomp(self.direct_vec_paths, 
            self.adjoint_vec_paths, L_sing_vecs_path, sing_vals_path, 
            R_sing_vecs_path)
        L_sing_vecs_return, sing_vals_return, R_sing_vecs_return = \
            self.my_BPOD.compute_decomp_and_return(self.direct_vec_paths, 
            self.adjoint_vec_paths)
        
        self.my_BPOD.put_hankel_mat(hankel_mat_path)
        
        if parallel.is_rank_zero():
            L_sing_vecs_loaded = util.load_mat_text(L_sing_vecs_path)
            R_sing_vecs_loaded = util.load_mat_text(R_sing_vecs_path)
            sing_vals_loaded = N.squeeze(N.array(util.load_mat_text(
                sing_vals_path)))
            hankel_mat_loaded = util.load_mat_text(hankel_mat_path)
        else:
            L_sing_vecs_loaded = None
            R_sing_vecs_loaded = None
            sing_vals_loaded = None
            hankel_mat_loaded = None

        if parallel.is_distributed():
            L_sing_vecs_loaded = parallel.comm.bcast(L_sing_vecs_loaded,
                root=0)
            R_sing_vecs_loaded = parallel.comm.bcast(R_sing_vecs_loaded,
                root=0)
            sing_vals_loaded = parallel.comm.bcast(sing_vals_loaded, 
                root=0)
            hankel_mat_loaded = parallel.comm.bcast(hankel_mat_loaded, 
                root=0)
        
        N.testing.assert_allclose(self.my_BPOD.hankel_mat,
          self.hankel_mat_true, rtol=tol)
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
        
        N.testing.assert_allclose(hankel_mat_loaded,
          self.hankel_mat_true, rtol=tol)
        N.testing.assert_allclose(L_sing_vecs_loaded,
          self.L_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(R_sing_vecs_loaded,
          self.R_sing_vecs_true, rtol=tol)
        N.testing.assert_allclose(sing_vals_loaded,
          self.sing_vals_true, rtol=tol)
        


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
        
        direct_mode_paths = [direct_mode_path%i for i in self.mode_nums]
        adjoint_mode_paths = [adjoint_mode_path%i for i in self.mode_nums]

        self.my_BPOD.compute_direct_modes(self.mode_nums, direct_mode_paths,
            index_from=self.index_from, direct_vec_sources=self.direct_vec_paths)
          
        self.my_BPOD.compute_adjoint_modes(self.mode_nums, adjoint_mode_paths,
            index_from=self.index_from, adjoint_vec_sources=self.adjoint_vec_paths)
          
        for mode_num in self.mode_nums:
            if parallel.is_rank_zero():
                direct_mode = util.load_mat_text(direct_mode_path % mode_num)
                adjoint_mode = util.load_mat_text(adjoint_mode_path % mode_num)
            else:
                direct_mode = None
                adjoint_mode = None
            if parallel.is_distributed():
                direct_mode = parallel.comm.bcast(direct_mode, root=0)
                adjoint_mode = parallel.comm.bcast(adjoint_mode, root=0)
            N.testing.assert_allclose(direct_mode, 
                self.direct_mode_mat[:,mode_num-self.index_from])
            N.testing.assert_allclose(adjoint_mode, 
                self.adjoint_mode_mat[:,mode_num-self.index_from])
        
        if parallel.is_rank_zero():
            for mode_num1 in self.mode_nums:
                direct_mode = util.load_mat_text(
                  direct_mode_path%mode_num1)
                for mode_num2 in self.mode_nums:
                    adjoint_mode = util.load_mat_text(
                        adjoint_mode_path%mode_num2)
                    IP = self.my_vec_defs.inner_product(
                      direct_mode,adjoint_mode)
                    if mode_num1 != mode_num2:
                        self.assertAlmostEqual(IP, 0.)
                    else:
                        self.assertAlmostEqual(IP, 1.)
      
      
if __name__ == '__main__':
    unittest.main()

