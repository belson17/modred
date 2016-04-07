#!/usr/bin/env python
"""Test vectorspace module"""
from __future__ import division
from future.builtins import range
import os
from os.path import join
from shutil import rmtree
import copy
import random
#import inspect #makes it possible to find information about a function
import unittest

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
from modred.vectorspace import *
import modred.vectors as V
import modred.util


class TestVectorSpaceHandles(unittest.TestCase):
    """ Tests of the VectorSpace class """
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DELETE_ME_test_files_vecoperations'
        if not os.path.isdir(self.test_dir) and _parallel.is_rank_zero():
            os.mkdir(self.test_dir)

        self.max_vecs_per_proc = 10
        self.total_num_vecs_in_mem = (
            _parallel.get_num_procs() * self.max_vecs_per_proc)

        self.my_vec_ops = VectorSpaceHandles(inner_product=np.vdot, verbosity=0)
        self.my_vec_ops.max_vecs_per_proc = self.max_vecs_per_proc

        # Default data members, verbosity set to 0 even though default is 1
        # so messages won't print during tests
        self.default_data_members = {'inner_product': np.vdot,
            'max_vecs_per_node': 10000,
            'max_vecs_per_proc': 10000 * _parallel.get_num_nodes() // \
                _parallel.get_num_procs(),
            'verbosity': 0, 'print_interval': 10, 'prev_print_time': 0.}        
        _parallel.barrier()

    def tearDown(self):
        _parallel.barrier()
        _parallel.call_from_rank_zero(rmtree, self.test_dir, ignore_errors=True)
        _parallel.barrier()
 
    #@unittest.skip('testing other things')
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly."""
        data_members_original = util.get_data_members(
            VectorSpaceHandles(inner_product=np.vdot, verbosity=0))
        self.assertEqual(data_members_original, self.default_data_members)
                
        max_vecs_per_node = 500
        my_VS = VectorSpaceHandles(inner_product=np.vdot, 
            max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members = copy.deepcopy(data_members_original)
        data_members['max_vecs_per_node'] = max_vecs_per_node
        data_members['max_vecs_per_proc'] = max_vecs_per_node * \
            _parallel.get_num_nodes() // _parallel.get_num_procs()
        self.assertEqual(util.get_data_members(my_VS), data_members)

    #@unittest.skip('testing other things')
    def test_sanity_check(self):
        """Tests correctly checks user-supplied objects and functions."""
        nx = 40
        ny = 15
        test_array = np.random.random((nx, ny))

        my_VS = VectorSpaceHandles(inner_product=np.vdot, verbosity=0)
        in_mem_handle = V.VecHandleInMemory(test_array)
        my_VS.sanity_check(in_mem_handle)
        
        # An sanity's vector that redefines multiplication to modify its data
        class SanityMultVec(V.Vector):
            def __init__(self, arr):
                self.arr = arr
            def __add__(self, obj):
                f_return = copy.deepcopy(self)
                f_return.arr += obj.arr
                return f_return
            def __mul__(self, a):
                self.arr *= a
                return self
                
        class SanityAddVec(V.Vector):
            def __init__(self, arr):
                self.arr = arr
            def __add__(self, obj):
                self.arr += obj.arr
                return self
            def __mul__(self, a):
                f_return = copy.deepcopy(self)
                f_return.arr *= a
                return f_return
        
        def my_IP(vec1, vec2):
            return np.vdot(vec1.arr, vec2.arr)
        my_VS.inner_product = my_IP
        my_sanity_mult_vec = SanityMultVec(test_array)
        self.assertRaises(ValueError, my_VS.sanity_check, 
            V.VecHandleInMemory(my_sanity_mult_vec))
        my_sanity_add_vec = SanityAddVec(test_array)
        self.assertRaises(ValueError, my_VS.sanity_check, 
            V.VecHandleInMemory(my_sanity_add_vec))
                
    def generate_vecs_modes(self, num_states, num_vecs, num_modes):
        """Generates random vecs and finds the modes. 
        
        Returns:
            vec_array: matrix in which each column is a vec (in order)
            mode_indices: unordered list of integers representing mode indices,
                each entry is unique. Mode indices are picked randomly.
            build_coeff_mat: matrix num_vecs x num_modes, random entries
            mode_array: matrix of modes, each column is a mode.
                matrix column # = mode_index
        """
        mode_indices = list(range(num_modes))
        random.shuffle(mode_indices)
        build_coeff_mat = np.mat(np.random.random((num_vecs, num_modes)))
        vec_array = np.mat(np.random.random((num_states, num_vecs)))
        mode_array = vec_array*build_coeff_mat
        return vec_array, mode_indices, build_coeff_mat, mode_array 
    
    #@unittest.skip('testing other things')
    def test_lin_combine(self):
        num_vecs_list = [1, 15, 40]
        num_states = 20
        # Test cases where number of modes:
        #   less, equal, more than num_states
        #   less, equal, more than num_vecs
        #   less, equal, more than total_num_vecs_in_mem
        num_modes_list = [1, 8, 10, 20, 25, 45, \
            int(np.ceil(self.total_num_vecs_in_mem / 2.)),\
            self.total_num_vecs_in_mem, self.total_num_vecs_in_mem * 2]
        mode_path = join(self.test_dir, 'mode_%03d.txt')
        vec_path = join(self.test_dir, 'vec_%03d.txt')
        
        for num_vecs in num_vecs_list:
            for num_modes in num_modes_list:
                #generate data and then broadcast to all procs
                #print '----- new case ----- '
                #print 'num_vecs =',num_vecs
                #print 'num_states =',num_states
                #print 'num_modes =',num_modes
                #print 'max_vecs_per_node =',max_vecs_per_node
                #print 'index_from =',index_from
                vec_handles = [V.VecHandleArrayText(vec_path%i) 
                    for i in range(num_vecs)]
                vec_array, mode_indices, build_coeff_mat, true_modes = \
                    _parallel.call_and_bcast(self.generate_vecs_modes, 
                    num_states, num_vecs, num_modes)

                if _parallel.is_rank_zero():
                    for vec_index, vec_handle in enumerate(vec_handles):
                        vec_handle.put(vec_array[:,vec_index])
                _parallel.barrier()
                mode_handles = [V.VecHandleArrayText(mode_path%mode_num)
                    for mode_num in mode_indices]
                
                # If there are more vecs than mat has rows
                build_coeff_mat_too_small = \
                    np.zeros((build_coeff_mat.shape[0]-1, 
                        build_coeff_mat.shape[1]))
                self.assertRaises(ValueError, self.my_vec_ops.\
                    lin_combine, mode_handles,
                    vec_handles, build_coeff_mat_too_small, mode_indices)
                
                # Test the case that only one mode is desired,
                # in which case user might pass in an int
                if len(mode_indices) == 1:
                    mode_indices = mode_indices[0]
                    mode_handles = mode_handles[0]
                    
                # Saves modes to files
                self.my_vec_ops.lin_combine(mode_handles,
                    vec_handles, build_coeff_mat, mode_indices)

                # Change back to list so is iterable
                if not isinstance(mode_indices, list):
                    mode_indices = [mode_indices]

                _parallel.barrier()
                #print 'mode_indices',mode_indices
                for mode_index in mode_indices:
                    computed_mode = V.VecHandleArrayText(
                        mode_path % mode_index).get()
                    #print 'mode number',mode_num
                    #print 'true mode',true_modes[:,\
                    #    mode_num-index_from]
                    #print 'computed mode',computed_mode
                    np.testing.assert_allclose(
                        computed_mode, true_modes[:,mode_index])
                        
                _parallel.barrier()
       
        _parallel.barrier()
    
    #@unittest.skip('testing others')
    @unittest.skipIf(_parallel.is_distributed(), 'Serial only')
    def test_compute_inner_product_mat_types(self):
        class ArrayTextComplexHandle(V.VecHandleArrayText):
            def get(self):
                return (1 + 1j)*V.VecHandleArrayText.get(self)
        
        num_row_vecs = 4
        num_col_vecs = 6
        num_states = 7

        row_vec_path = join(self.test_dir, 'row_vec_%03d.txt')
        col_vec_path = join(self.test_dir, 'col_vec_%03d.txt')
        
        # generate vecs and save to file
        row_vec_array = np.mat(np.random.random((num_states,
            num_row_vecs)))
        col_vec_array = np.mat(np.random.random((num_states,
            num_col_vecs)))
        row_vec_paths = []
        col_vec_paths = []
        for vec_index in range(num_row_vecs):
            path = row_vec_path % vec_index
            util.save_array_text(row_vec_array[:,vec_index], path)
            row_vec_paths.append(path)
        for vec_index in range(num_col_vecs):
            path = col_vec_path % vec_index
            util.save_array_text(col_vec_array[:,vec_index], path)
            col_vec_paths.append(path)
    
        # Compute inner product matrix and check type
        for handle, dtype in [(V.VecHandleArrayText, float), 
            (ArrayTextComplexHandle, complex)]:
            row_vec_handles = [handle(path) for path in row_vec_paths] 
            col_vec_handles = [handle(path) for path in col_vec_paths]
            inner_product_mat = self.my_vec_ops.compute_inner_product_mat(
                row_vec_handles, col_vec_handles)
            symm_inner_product_mat = \
                self.my_vec_ops.compute_symmetric_inner_product_mat(
                    row_vec_handles)
            self.assertEqual(inner_product_mat.dtype, dtype)
            self.assertEqual(symm_inner_product_mat.dtype, dtype)

    #@unittest.skip('testing other things')
    def test_compute_inner_product_mats(self):
        """Test computation of matrix of inner products.""" 
        num_row_vecs_list = [1, int(round(self.total_num_vecs_in_mem / 2.)), 
            self.total_num_vecs_in_mem, self.total_num_vecs_in_mem * 2,
            _parallel.get_num_procs() + 1]
        num_col_vecs_list = num_row_vecs_list
        num_states = 6

        row_vec_path = join(self.test_dir, 'row_vec_%03d.txt')
        col_vec_path = join(self.test_dir, 'col_vec_%03d.txt')
        
        for num_row_vecs in num_row_vecs_list:
            for num_col_vecs in num_col_vecs_list:
                # generate vecs
                _parallel.barrier()
                row_vec_array = _parallel.call_and_bcast(np.random.random, 
                    (num_states, num_row_vecs))
                col_vec_array = _parallel.call_and_bcast(np.random.random, 
                    (num_states, num_col_vecs))
                row_vec_handles = [V.VecHandleArrayText(row_vec_path%i) 
                    for i in range(num_row_vecs)]
                col_vec_handles = [V.VecHandleArrayText(col_vec_path%i) 
                    for i in range(num_col_vecs)]
                
                # Save vecs
                if _parallel.is_rank_zero():
                    for i,h in enumerate(row_vec_handles):
                        h.put(row_vec_array[:,i])
                    for i,h in enumerate(col_vec_handles):
                        h.put(col_vec_array[:,i])
                _parallel.barrier()

                # If number of rows/cols is 1, check case of passing a handle
                if len(row_vec_handles) == 1:
                    row_vec_handles = row_vec_handles[0]
                if len(col_vec_handles) == 1:
                    col_vec_handles = col_vec_handles[0]

                # Test IP computation.  
                product_true = np.dot(row_vec_array.T, col_vec_array)
                product_computed = self.my_vec_ops.compute_inner_product_mat(
                    row_vec_handles, col_vec_handles)
                row_vecs = [row_vec_array[:,i] for i in range(num_row_vecs)]
                col_vecs = [col_vec_array[:,i] for i in range(num_col_vecs)]
                np.testing.assert_allclose(product_computed, product_true)
                
                # Test symm IP computation
                product_true = np.dot(row_vec_array.T, row_vec_array)
                product_computed = \
                    self.my_vec_ops.compute_symmetric_inner_product_mat(
                        row_vec_handles)
                row_vecs = [row_vec_array[:,i] for i in range(num_row_vecs)]
                np.testing.assert_allclose(product_computed, product_true)
                
                        
if __name__=='__main__':
    unittest.main()    
