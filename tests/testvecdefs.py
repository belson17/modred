#!/usr/bin/env python

import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

import util
import vecdefs as VD

@unittest.skipIf(parallel.is_distributed(), 'No need to test in parallel')
class TestVecDefs(unittest.TestCase):
    """ Test all the vecdef methods """
    
    def setUp(self):
        self.test_dir ='DELETE_ME_test_files_vecdefs'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():        
            os.mkdir(self.test_dir)
        self.mode_nums =[2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.num_vecs = 40
        self.num_states = 100
        self.index_from = 2
        #parallel.sync()

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)

    def test_all_get_puts(self):
        """Tests that put->get returns original vector"""
        test_path = join(self.test_dir, 'test_vec')
        vec_true = N.random.random((3,4))
        
        v_out = VD.put_vec_in_memory(N.copy(vec_true), None)
        v_out = VD.get_vec_in_memory(v_out)
        N.testing.assert_allclose(v_out, vec_true)
        
        VD.put_vec_text(N.copy(vec_true), test_path)
        v_out = VD.get_vec_text(test_path)
        N.testing.assert_allclose(v_out, vec_true)
        
        VD.put_vec_pickle(N.copy(vec_true), test_path)
        v_out = VD.get_vec_pickle(test_path)
        N.testing.assert_allclose(v_out, vec_true)

    # Test classes too, but they're very simple so no tests right now

if __name__=='__main__':
    unittest.main()    

        
        
        
        