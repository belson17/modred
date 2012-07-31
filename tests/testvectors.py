#!/usr/bin/env python
"""Test vectors module"""

import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path(join(join(os.path.dirname(os.path.abspath(__file__)), 
    '..', 'src')))
import parallel as parallel_mod
parallel = parallel_mod.parallel_default_instance

import vectors as V

@unittest.skipIf(parallel.is_distributed(), 'No need to test in parallel')
class TestVectors(unittest.TestCase):
    """Test the vector methods """
    def setUp(self):
        self.test_dir ='DELETE_ME_test_files_vectors'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():        
            os.mkdir(self.test_dir)
        parallel.barrier()
        self.mode_nums = [2, 4, 3, 6, 9, 8, 10, 11, 30]
        self.num_vecs = 40
        self.num_states = 100
        self.index_from = 2

    def tearDown(self):
        parallel.barrier()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.barrier()
    
    def test_in_memory_handle(self):
        """Test in memory and base class vector handles"""
        base_vec1 = N.random.random((3, 4))
        base_vec2 = N.random.random((3, 4))
        vec_true = N.random.random((3, 4))
        scale = N.random.random()
        
        # Test base class functionality
        vec_handle = V.InMemoryVecHandle(vec=vec_true, 
            base_vec_handle=V.InMemoryVecHandle(vec=base_vec1),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_equal(vec_comp, scale*(vec_true - base_vec1))
        
        vec_handle = V.InMemoryVecHandle(vec=vec_true, 
            base_vec_handle=V.InMemoryVecHandle(vec=base_vec2),
            scale=scale)
        vec_comp = vec_handle.get()
        N.testing.assert_equal(vec_comp, scale*(vec_true - base_vec2))
        
        vec_handle = V.InMemoryVecHandle(vec=vec_true, 
            base_vec_handle=V.InMemoryVecHandle(vec=base_vec1))
        vec_comp = vec_handle.get()
        N.testing.assert_equal(vec_comp, vec_true - base_vec1)
        
        vec_handle = V.InMemoryVecHandle(vec=vec_true)
        vec_comp = vec_handle.get()
        N.testing.assert_equal(vec_comp, vec_true)

        # Test put
        vec_handle = V.InMemoryVecHandle()
        vec_handle.put(vec_true)
        N.testing.assert_equal(vec_handle.vec, vec_true)
        
        # Test __eq__ operator
        vec_handle1 = V.InMemoryVecHandle(vec=N.ones(2))
        vec_handle2 = V.InMemoryVecHandle(vec=N.ones(2))
        vec_handle3 = V.InMemoryVecHandle(vec=N.ones(3))
        vec_handle4 = V.InMemoryVecHandle(vec=N.zeros(2))
        self.assertEqual(vec_handle1, vec_handle1)
        self.assertEqual(vec_handle1, vec_handle2)
        self.assertNotEqual(vec_handle1, vec_handle3)
        self.assertNotEqual(vec_handle1, vec_handle4)

    def test_handles_which_save(self):
        """Test handles whose get/put load/save from file"""
        base_vec1 = N.random.random((3,4))
        base_vec2 = N.random.random((3,4))
        vec_true = N.random.random((3,4))
        #scale = N.random.random()
        vec_true_path = join(self.test_dir, 'test_vec')
        vec_saved = join(self.test_dir, 'put_vec')
        base_path1 = join(self.test_dir, 'base_vec1')
        base_path2 = join(self.test_dir, 'base_vec2')
        for VecHandle in [V.ArrayTextVecHandle, V.PickleVecHandle]:
            VecHandle(base_path1).put(base_vec1)
            VecHandle(base_path2).put(base_vec2)
            VecHandle(vec_true_path).put(vec_true)
            # Test get
            vec_handle = VecHandle(vec_true_path)
            vec_comp = vec_handle.get()
            N.testing.assert_allclose(vec_comp, vec_true)
            # Test put
            vec_handle = VecHandle(vec_saved)
            vec_handle.put(vec_true)
            N.testing.assert_equal(vec_handle.get(), vec_true)
            # Test __eq__ operator
            vec_handle1 = VecHandle('a')
            vec_handle2 = VecHandle('a')
            vec_handle3 = VecHandle('aa')
            vec_handle4 = VecHandle('b')
            self.assertEqual(vec_handle1, vec_handle1)
            self.assertEqual(vec_handle1, vec_handle2)
            self.assertNotEqual(vec_handle1, vec_handle3)
            self.assertNotEqual(vec_handle1, vec_handle4)
            
        
        
    def test_IP_trapz(self):
        """Test trapezoidal rule inner product for 2nd-order convergence"""
        # Known inner product of x**2 + 1.2y**2 and x**2 over interval
        # -1<x<1 and -2<y<2.
        ip_true = 5.8666666
        ip_error = []
        num_points_list = [20, 100]
        for num_points in num_points_list:
            x_grid = N.cos(N.linspace(0, N.pi, num_points))[::-1]
            y_grid = 2*N.cos(N.linspace(0, N.pi, num_points+1))[::-1]
            # Notice order is reversed. This gives dimensions [nx, ny]
            # instead of [ny, nx]. See N.meshgrid documentation.
            Y, X = N.meshgrid(y_grid, x_grid)
            v1 = X**2 + 1.2*Y**2
            v2 = X**2
            ip_comp = V.InnerProductTrapz(x_grid, y_grid)(v1, v2)
            ip_error.append(N.abs(ip_comp-ip_true))
        convergence = (N.log(ip_error[1]) - N.log(ip_error[0]))/ \
            (N.log(num_points_list[1]) - N.log(num_points_list[0]))
        self.assertTrue(convergence < -1.9)
                


if __name__ == '__main__':
    unittest.main()    

        
        
        
        
