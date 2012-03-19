#!/usr/bin/env python

import os
from os.path import join
from shutil import rmtree
import copy
#import inspect #makes it possible to find information about a function
import unittest
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

from fieldoperations import FieldOperations
import util


class TestFieldOperations(unittest.TestCase):
    """ Tests of the FieldOperations class """
    
    def setUp(self):
    
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
            
        self.test_dir = 'DELETE_ME_test_files_fieldoperations'    
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():
            os.mkdir(self.test_dir)

        # Default data members, verbose set to false even though default is true
        # so messages won't print during tests
        self.default_data_members = {'get_field': None, 'put_field': None, 
            'inner_product': None, 'max_fields_per_node': 2,
            'max_fields_per_proc': 2, 'parallel':parallel_mod.default_instance,
            'verbose': False, 'print_interval':10, 'prev_print_time':0.}
       
        self.max_fields_per_proc = 10
        self.total_num_fields_in_mem = parallel.get_num_procs() * self.max_fields_per_proc

        # FieldOperations object for running tests
        self.fieldOperations = FieldOperations( 
            get_field=util.load_mat_text, 
            put_field=util.save_mat_text, 
            inner_product=util.inner_product, 
            verbose=False)
        self.fieldOperations.max_fields_per_proc = self.max_fields_per_proc
        
        parallel.sync()

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.sync()
 
 
 
    #@unittest.skip('testing other things')
    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly.
        """
        data_members_original = util.get_data_members(FieldOperations(verbose=False))
        self.assertEqual(data_members_original, self.default_data_members)
        
        def my_load(fname): pass
        my_FO = FieldOperations(get_field=my_load, verbose=False)
        data_members = copy.deepcopy(data_members_original)
        data_members['get_field'] = my_load
        self.assertEqual(util.get_data_members(my_FO), data_members)
        
        def my_save(data,fname): pass
        my_FO = FieldOperations(put_field=my_save, verbose=False)
        data_members = copy.deepcopy(data_members_original)
        data_members['put_field'] = my_save
        self.assertEqual(util.get_data_members(my_FO), data_members)
        
        def my_ip(f1,f2): pass
        my_FO = FieldOperations(inner_product=my_ip, verbose=False)
        data_members = copy.deepcopy(data_members_original)
        data_members['inner_product'] = my_ip
        self.assertEqual(util.get_data_members(my_FO), data_members)
        
        max_fields_per_node = 500
        my_FO = FieldOperations(max_fields_per_node=max_fields_per_node, verbose=False)
        data_members = copy.deepcopy(data_members_original)
        data_members['max_fields_per_node'] = max_fields_per_node
        data_members['max_fields_per_proc'] = max_fields_per_node * my_FO.parallel.get_num_nodes()/ \
            my_FO.parallel.get_num_procs()
        self.assertEqual(util.get_data_members(my_FO), data_members)

        

    #@unittest.skip('testing other things')
    def test_idiot_check(self):
        """
        Tests idiot_check correctly checks user-supplied objects and functions.
        """
        nx = 40
        ny = 15
        test_array = N.random.random((nx,ny))
        def inner_product(a,b):
            return N.sum(a.arr*b.arr)
        my_FO = FieldOperations(inner_product=util.inner_product, verbose=False)
        my_FO.idiot_check(test_obj=test_array)
        
        # An idiot's class that redefines multiplication to modify its data
        class IdiotMult(object):
            def __init__(self, arr):
                self.arr = arr
            def __add__(self, obj):
                f_return = copy.deepcopy(self)
                f_return.arr += obj.arr
                return f_return
            def __mul__(self, a):
                self.arr *= a
                return self
                
        class IdiotAdd(object):
            def __init__(self, arr):
                self.arr = arr
            def __add__(self, obj):
                self.arr += obj.arr
                return self
            def __mul__(self, a):
                f_return = copy.deepcopy(self)
                f_return.arr *= a
                return f_return
        my_FO.inner_product = inner_product
        my_idiot_mult = IdiotMult(test_array)
        self.assertRaises(ValueError,my_FO.idiot_check, test_obj=my_idiot_mult)
        my_idiot_add = IdiotAdd(test_array)
        self.assertRaises(ValueError,my_FO.idiot_check, test_obj=my_idiot_add)
                
        
        
    def generate_fields_modes(self, num_states, num_fields, num_modes, index_from=1):
        """
        Generates random fields and finds the modes. 
        
        Returns:
            field_mat: matrix in which each column is a field (in order)
            mode_nums: unordered list of integers representing mode numbers,
                each entry is unique. Mode numbers are picked randomly between
            build_coeff_mat: matrix num_fields x num_modes, random entries
            mode_mat: matrix of modes, each column is a mode.
                matrix column # = mode_number - index_from
        """
        mode_nums = []
        while len(mode_nums) < num_modes:
            mode_num = index_from+int(N.floor(N.random.random()*num_modes))
            if mode_nums.count(mode_num) == 0:
                mode_nums.append(mode_num)

        build_coeff_mat = N.mat(N.random.random((num_fields,num_modes)))
        field_mat = N.mat(N.zeros((num_states,num_fields)))
        for field_index in range(num_fields):
            field_mat[:,field_index] = N.random.random((num_states,1))
        mode_mat = field_mat*build_coeff_mat
        return field_mat,mode_nums,build_coeff_mat,mode_mat 
        
    
    
    #@unittest.skip('testing other things')
    def test_compute_modes(self):
        """
        Test that can compute modes from arguments. 
               
        Cases are tested for numbers of fields, states per field,
        mode numbers, number of fields/modes allowed in memory
        simultaneously, and indexing schemes 
        (meaning the first mode can be numbered 0, 1, or any integer).
        """
        num_fields_list = [1, 15, 40]
        num_states = 20
        # Test cases where number of modes:
        #   less, equal, more than num_states
        #   less, equal, more than num_fields
        #   less, equal, more than total_num_fields_in_mem
        num_modes_list = [1, 8, 10, 20, 25, 45, \
            int(N.ceil(self.total_num_fields_in_mem / 2.)),\
            self.total_num_fields_in_mem, self.total_num_fields_in_mem * 2]
        index_from_list = [0, 5]
        #mode_path = 'proc'+str(self.fieldOperations.parallelInstance._rank)+'/mode_%03d.txt'
        mode_path = join(self.test_dir, 'mode_%03d.txt')
        field_path = join(self.test_dir, 'field_%03d.txt')
        
        for num_fields in num_fields_list:
            for num_modes in num_modes_list:
                for index_from in index_from_list:
                    #generate data and then broadcast to all procs
                    #print '----- new case ----- '
                    #print 'num_fields =',num_fields
                    #print 'num_states =',num_states
                    #print 'num_modes =',num_modes
                    #print 'max_fields_per_node =',max_fields_per_node                          
                    #print 'index_from =',index_from
                    field_paths = [field_path % field_index \
                        for field_index in xrange(num_fields)]

                    if parallel.is_rank_zero():
                        field_mat,mode_nums, build_coeff_mat, true_modes = \
                          self.generate_fields_modes(num_states, num_fields,
                          num_modes, index_from=index_from)
                        for field_index,s in enumerate(field_paths):
                            util.save_mat_text(field_mat[:,field_index], s)
                    else:
                        mode_nums = None
                        build_coeff_mat = None
                        field_mat = None
                        true_modes = None
                        mode_paths = None
                    if parallel.is_distributed():
                        mode_nums = parallel.comm.bcast(
                            mode_nums, root=0)
                        build_coeff_mat = parallel.comm.bcast(
                            build_coeff_mat, root=0)
                        field_mat = parallel.comm.bcast(
                            field_mat, root=0)
                        true_modes = parallel.comm.bcast(
                            true_modes, root=0)
                    
                    mode_paths = [mode_path%mode_num for mode_num in
                            mode_nums]
                    # if any mode number (minus starting indxex)
                    # is greater than the number of coeff mat columns,
                    # or is less than zero
                    check_assert_raises = False
                    for mode_num in mode_nums:
                        mode_num_from_zero = mode_num-index_from
                        if mode_num_from_zero < 0 or mode_num_from_zero >=\
                            build_coeff_mat.shape[1]:
                            check_assert_raises = True
                    if check_assert_raises:
                        self.assertRaises(ValueError, self.fieldOperations.\
                            _compute_modes, mode_nums, mode_paths, 
                            field_paths, build_coeff_mat, index_from=\
                            index_from)
                    # If the coeff mat has more rows than there are 
                    # field paths
                    elif num_fields > build_coeff_mat.shape[0]:
                        self.assertRaises(ValueError, self.fieldOperations.\
                            _compute_modes, mode_nums, mode_paths,
                            field_paths, build_coeff_mat, index_from=\
                            index_from)
                    elif num_modes > num_fields:
                        self.assertRaises(ValueError,
                          self.fieldOperations._compute_modes, mode_nums,
                          mode_paths, field_paths, build_coeff_mat,
                          index_from=index_from)
                    else:
                        # Test the case that only one mode is desired,
                        # in which case user might pass in an int
                        if len(mode_nums) == 1:
                            mode_nums = mode_nums[0]
                            mode_paths = mode_paths[0]
                            
                        # Saves modes to files
                        self.fieldOperations._compute_modes(mode_nums, 
                            mode_paths,
                            field_paths, build_coeff_mat, 
                            index_from=index_from)

                        # Change back to list so is iterable
                        if isinstance(mode_nums, int):
                            mode_nums = [mode_nums]

                        parallel.sync()
                        #print 'mode_nums',mode_nums
                        #if parallel.is_rank_zero():
                        for mode_num in mode_nums:
                            computed_mode = util.load_mat_text(
                                mode_path % mode_num)
                            #print 'mode number',mode_num
                            #print 'true mode',true_modes[:,\
                            #    mode_num-index_from]
                            #print 'computed mode',computed_mode
                            N.testing.assert_allclose(
                                computed_mode, true_modes[:,mode_num-\
                                index_from])
                                
                        parallel.sync()
       
        parallel.sync()



    def test_compute_inner_product_mat_types(self):
        def get_field_as_complex(path):
            return (1 + 1j) * util.load_mat_text(path) 

        num_row_fields = 4
        num_col_fields = 6
        num_states = 7

        row_field_path = join(self.test_dir, 'row_field_%03d.txt')
        col_field_path = join(self.test_dir, 'col_field_%03d.txt')
        
        # generate fields and save to file, only do on proc 0
        parallel.sync()
        if parallel.is_rank_zero():
            row_field_mat = N.mat(N.random.random((num_states,
                num_row_fields)))
            col_field_mat = N.mat(N.random.random((num_states,
                num_col_fields)))
            row_field_paths = []
            col_field_paths = []
            for field_index in xrange(num_row_fields):
                path = row_field_path % field_index
                util.save_mat_text(row_field_mat[:,field_index],path)
                row_field_paths.append(path)
            for field_index in xrange(num_col_fields):
                path = col_field_path % field_index
                util.save_mat_text(col_field_mat[:,field_index],path)
                col_field_paths.append(path)
        else:
            row_field_mat = None
            col_field_mat = None
            row_field_paths = None
            col_field_paths = None
        if parallel.is_distributed():
            row_field_mat = parallel.comm.bcast(row_field_mat, root=0)
            col_field_mat = parallel.comm.bcast(col_field_mat, root=0)
            row_field_paths = parallel.comm.bcast(row_field_paths, root=0)
            col_field_paths = parallel.comm.bcast(col_field_paths, root=0)

        # If number of rows/cols is 1, test case that a string, not
        # a list, is passed in
        if len(row_field_paths) == 1:
            row_field_paths = row_field_paths[0]
        if len(col_field_paths) == 1:
            col_field_paths = col_field_paths[0]
    
        # Comptue inner product matrix and check type
        for load, type in [(get_field_as_complex, complex), (util.\
            load_mat_text, float)]:
            self.fieldOperations.get_field = load
            inner_product_mat = self.fieldOperations.compute_inner_product_mat(
                row_field_paths, col_field_paths)
            symm_inner_product_mat = self.fieldOperations.\
                compute_symmetric_inner_product_mat(row_field_paths)
            self.assertEqual(inner_product_mat.dtype, type)
            self.assertEqual(symm_inner_product_mat.dtype, type)



    #@unittest.skip('testing other things')
    def test_compute_inner_product_mats(self):
        """
        Test computation of matrix of inner products in memory-efficient
        chunks, both in parallel (compute_inner_product_mat).
        """ 
        num_row_fields_list =[1, int(round(self.total_num_fields_in_mem / 2.)), self.\
            total_num_fields_in_mem, self.total_num_fields_in_mem *2,
            parallel.get_num_procs()+1]
        num_col_fields_list = num_row_fields_list
        num_states = 6

        row_field_path = join(self.test_dir, 'row_field_%03d.txt')
        col_field_path = join(self.test_dir, 'col_field_%03d.txt')
        
        for num_row_fields in num_row_fields_list:
            for num_col_fields in num_col_fields_list:
                # generate fields and save to file, only do on proc 0
                parallel.sync()
                if parallel.is_rank_zero():
                    row_field_mat = N.mat(N.random.random((num_states,
                        num_row_fields)))
                    col_field_mat = N.mat(N.random.random((num_states,
                        num_col_fields)))
                    row_field_paths = []
                    col_field_paths = []
                    for field_index in xrange(num_row_fields):
                        path = row_field_path % field_index
                        util.save_mat_text(row_field_mat[:,field_index],path)
                        row_field_paths.append(path)
                    for field_index in xrange(num_col_fields):
                        path = col_field_path % field_index
                        util.save_mat_text(col_field_mat[:,field_index],path)
                        col_field_paths.append(path)
                else:
                    row_field_mat = None
                    col_field_mat = None
                    row_field_paths = None
                    col_field_paths = None
                if parallel.is_distributed():
                    row_field_mat = parallel.comm.bcast(row_field_mat, root=0)
                    col_field_mat = parallel.comm.bcast(col_field_mat, root=0)
                    row_field_paths = parallel.comm.bcast(row_field_paths, root=0)
                    col_field_paths = parallel.comm.bcast(col_field_paths, root=0)

                # If number of rows/cols is 1, test case that a string, not
                # a list, is passed in
                if len(row_field_paths) == 1:
                    row_field_paths = row_field_paths[0]
                if len(col_field_paths) == 1:
                    col_field_paths = col_field_paths[0]

                # Path list may actually be a string, in which case covert to list
                if not isinstance(row_field_paths, list):
                    row_field_paths = [row_field_paths]
                if not isinstance(col_field_paths, list):
                    col_field_paths = [col_field_paths]
    
                # True inner product matrix
                product_true = row_field_mat.T * col_field_mat
               
                # Test paralleized computation.  
                product_computed = \
                    self.fieldOperations.compute_inner_product_mat(row_field_paths,
                        col_field_paths)
                N.testing.assert_allclose(product_computed, 
                    product_true)
                        
if __name__=='__main__':
    unittest.main()    

    
    