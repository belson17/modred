#!/usr/bin/env python

import unittest
import os
from os.path import join
from shutil import rmtree
import cPickle
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

import util
import vectors as V

@unittest.skipIf(parallel.is_distributed(), 'No need to test in parallel')
class TestVectors(unittest.TestCase):
    """ Test the vector methods """
    def setUp(self):
        self.test_dir ='DELETE_ME_test_files_vectors'
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
    
    
    @unittest.skip('Testing others')
    def test_base_vec_handle(self):
        """Test base class of vector handles"""
        util.save_array_text(base_vec1, base_path1)
        util.save_array_text(base_vec2, base_path2)
        util.save_array_text(vec_true, test_path)
        
        vec_ptr = V.VecHandle(base_handle=base_path1, scale=scale)
        self.assertEqual(V.VecHandle.cached_base_handle, base_path1)
        self.assertEqual(V.VecHandle.cached_base_vec, base_vec1)
        
        vec_ptr = V.VecHandle(base_handle=base_path2)
        self.assertEqual(V.VecHandle.cached_base_handle, base_path2)
        self.assertEqual(V.VecHandle.cached_base_vec, base_vec2)
    
    def test_in_memory_handle(self):
        """Test derived vector handles get/put"""
        base_vec1 = N.random.random((3,4))
        base_vec2 = N.random.random((3,4))
        vec_true = N.random.random((3,4))
        scale = N.random.random()
        #test_path = join(self.test_dir, 'test_vec')
        #base_path1 = join(self.test_dir, 'base_vec1')
        #base_path2 = join(self.test_dir, 'base_vec2')
    
        vec_handle = V.InMemoryHandle(vec=vec_true, base_handle=V.InMemoryHandle(base_vec1),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, scale*(vec_true - base_vec1))
        
        vec_handle = V.InMemoryHandle(vec=vec_true, base_handle=V.InMemoryHandle(base_vec2),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, scale*(vec_true - base_vec2))
        
        vec_handle = V.InMemoryHandle(vec=vec_true, base_handle=V.InMemoryHandle(base_vec1))
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, vec_true - base_vec1)
        
        vec_handle = V.InMemoryHandle(vec=vec_true)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, vec_true)


    def test_array_text_handle(self):
        """Test derived vector handles get/put"""
        base_vec1 = N.random.random((3,4))
        base_vec2 = N.random.random((3,4))
        vec_true = N.random.random((3,4))
        scale = N.random.random()
        test_path = join(self.test_dir, 'test_vec')
        base_path1 = join(self.test_dir, 'base_vec1')
        base_path2 = join(self.test_dir, 'base_vec2')
        util.save_array_text(base_vec1, base_path1)
        util.save_array_text(base_vec2, base_path2)
        util.save_array_text(vec_true, test_path)
        VecHandle = V.ArrayTextHandle #, V.PickleHandle]:
        vec_handle = VecHandle(test_path, base_handle=VecHandle(base_path1),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, scale*(vec_true - base_vec1))
        
        vec_handle = VecHandle(test_path, base_handle=VecHandle(base_path2),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, scale*(vec_true - base_vec2))
        
        vec_handle = VecHandle(test_path, base_handle=VecHandle(base_path1))
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, vec_true - base_vec1)
        
        vec_handle = VecHandle(test_path)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, vec_true)


    def test_pickle_handle(self):
        """Test derived vector handles get/put"""
        VecHandle = V.PickleHandle
        base_vec1 = N.random.random((3,4))
        base_vec2 = N.random.random((3,4))
        vec_true = N.random.random((3,4))
        scale = N.random.random()
        vec_true_path = join(self.test_dir, 'test_vec')
        base_path1 = join(self.test_dir, 'base_vec1')
        base_path2 = join(self.test_dir, 'base_vec2')
        VecHandle(base_path1).put(base_vec1)
        VecHandle(base_path2).put(base_vec2)
        VecHandle(vec_true_path).put(vec_true)
        vec_handle = VecHandle(vec_true_path, base_handle=VecHandle(base_path1),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, scale*(vec_true - base_vec1))
        
        vec_handle = VecHandle(vec_true_path, base_handle=VecHandle(base_path2),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, scale*(vec_true - base_vec2))
        
        vec_handle = VecHandle(vec_true_path, base_handle=VecHandle(base_path1))
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, vec_true - base_vec1)
        
        vec_handle = VecHandle(vec_true_path)
        vec_comp = vec_handle.get()
        N.testing.assert_allclose(vec_comp, vec_true)


    def test_IP_trapz(self):
        """Test trapezoidal rule inner product for 2nd-order convergence"""
        # Known inner product of x**2 + 1.2y**2 and x**2 over interval
        # -1<x<1 and -2<y<2.
        ip_true = 5.8666666
        ip_error = []
        num_points_list = [20, 100]
        for num_points in num_points_list:
            angle = N.linspace(0, N.pi, num_points)
            x_grid = N.cos(angle)[::-1]
            y_grid = 2*N.cos(angle)[::-1]
            X, Y = N.meshgrid(x_grid, y_grid)
            v1 = X**2 + 1.2*Y**2
            v2 = X**2
            ip_comp = V.InnerProductNonUniform(x_grid, y_grid)(v1,v2)
            ip_error.append(N.abs(ip_comp-ip_true))
        convergence = (N.log(ip_error[1]) - N.log(ip_error[0]))/ \
            (N.log(num_points_list[1]) - N.log(num_points_list[0]))
        self.assertTrue(convergence < -1.9)
                


if __name__=='__main__':
    unittest.main()    

        
        
        
        
