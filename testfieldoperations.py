#!/usr/bin/env python

import subprocess as SP
import os
import copy
#import inspect #makes it possible to find information about a function
import unittest
import numpy as N
from fieldoperations import FieldOperations
import parallel as parallel_mod
import util

parallel = parallel_mod.parallel_default

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
    print '  1) python testfieldoperations.py'
    print '  2) mpiexec -n <# procs> python testfieldoperations.py\n\n'

class TestFieldOperations(unittest.TestCase):
    """ Tests of the FieldOperations class """
    
    def setUp(self):
        # Default data members, verbose set to false even though default is true
        # so messages won't print during tests
        self.default_data_members = {'load_field': None, 'save_field': None, 
            'inner_product': None, 'max_fields_per_node': 2,
            'max_fields_per_proc': 2, 'parallel':parallel_mod.parallel_default,
            'verbose': False, 'print_interval':10, 'prev_print_time':0.}
       
        self.max_fields_per_proc = 10
        self.total_num_fields_in_mem = parallel.get_num_procs() * self.max_fields_per_proc

        # FieldOperations object for running tests
        self.fieldOperations = FieldOperations( 
            load_field=util.load_mat_text, 
            save_field=util.save_mat_text, 
            inner_product=util.inner_product, 
            verbose=False)
        self.fieldOperations.max_fields_per_proc = self.max_fields_per_proc

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            SP.call(['rm -rf files_modaldecomp_test/*'], shell=True)
        parallel.sync()
 
    #@unittest.skip('testing other things')
    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly.
        """
        data_members_original = util.get_data_members(FieldOperations(verbose=False))
        self.assertEqual(data_members_original, self.default_data_members)
        
        def my_load(fname): pass
        my_FO = FieldOperations(load_field=my_load, verbose=False)
        data_members = copy.deepcopy(data_members_original)
        data_members['load_field'] = my_load
        self.assertEqual(util.get_data_members(my_FO), data_members)
        
        def my_save(data,fname): pass
        my_FO = FieldOperations(save_field=my_save, verbose=False)
        data_members = copy.deepcopy(data_members_original)
        data_members['save_field'] = my_save
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
                
        
    def generate_snaps_modes(self, num_states, num_snaps, num_modes, index_from=1):
        """
        Generates random snapshots and finds the modes. 
        
        Returns:
        snap_mat -  matrix in which each column is a snapshot (in order)
        mode_nums - unordered list of integers representing mode numbers,
          each entry is unique. Mode numbers are picked randomly between
          index_from and num_modes+index_from-1. 
        build_coeff_mat - matrix num_snaps x num_modes, random entries
        mode_mat - matrix of modes, each column is a mode.
          matrix column # = mode_number - index_from
        """
        mode_nums = []
        while len(mode_nums) < num_modes:
            mode_num = index_from+int(N.floor(N.random.random()*num_modes))
            if mode_nums.count(mode_num) == 0:
                mode_nums.append(mode_num)

        build_coeff_mat = N.mat(N.random.random((num_snaps,num_modes)))
        snap_mat = N.mat(N.zeros((num_states,num_snaps)))
        for snap_index in range(num_snaps):
            snap_mat[:,snap_index] = N.random.random((num_states,1))
        mode_mat = snap_mat*build_coeff_mat
        return snap_mat,mode_nums,build_coeff_mat,mode_mat 
        
    
    #@unittest.skip('testing other things')
    def test_compute_modes(self):
        """
        Test that can compute modes from arguments. 
        
        Parallel and serial cases need to be tested independently. 
        
        Many cases are tested for numbers of snapshots, states per snapshot,
        mode numbers, number of snapshots/modes allowed in memory
        simultaneously, and what the indexing scheme is 
        (currently supports any indexing
        scheme, meaning the first mode can be numbered 0, 1, or any integer).
        """
        num_snaps_list = [1, 15, 40]
        num_states = 20
        # Test cases where number of modes:
        #   less, equal, more than num_states
        #   less, equal, more than num_snaps
        #   less, equal, more than total_num_fields_in_mem
        num_modes_list = [1, 8, 10, 20, 25, 45, \
            int(N.ceil(self.total_num_fields_in_mem / 2.)),\
            self.total_num_fields_in_mem, self.total_num_fields_in_mem * 2]
        index_from_list = [0, 5]
        #mode_path = 'proc'+str(self.fieldOperations.parallelInstance._rank)+'/mode_%03d.txt'
        mode_path = 'files_modaldecomp_test/mode_%03d.txt'
        snap_path = 'files_modaldecomp_test/snap_%03d.txt'
        if self.fieldOperations.parallel.is_rank_zero():
            if not os.path.isdir('files_modaldecomp_test'):
                SP.call(['mkdir','files_modaldecomp_test'])
        
        for num_snaps in num_snaps_list:
            for num_modes in num_modes_list:
                for index_from in index_from_list:
                    #generate data and then broadcast to all procs
                    #print '----- new case ----- '
                    #print 'num_snaps =',num_snaps
                    #print 'num_states =',num_states
                    #print 'num_modes =',num_modes
                    #print 'max_fields_per_node =',max_fields_per_node                          
                    #print 'index_from =',index_from
                    snap_paths = [snap_path % snap_index \
                        for snap_index in xrange(num_snaps)]
                    
                    if parallel.is_rank_zero():
                        snap_mat,mode_nums, build_coeff_mat, true_modes = \
                          self.generate_snaps_modes(num_states, num_snaps,
                          num_modes, index_from=index_from)
                        for snap_index,s in enumerate(snap_paths):
                            util.save_mat_text(snap_mat[:,snap_index], s)
                    else:
                        mode_nums = None
                        build_coeff_mat = None
                        snap_mat = None
                        true_modes = None
                    if parallel.is_distributed():
                        mode_nums = parallel.comm.bcast(
                            mode_nums, root=0)
                        build_coeff_mat = parallel.comm.bcast(
                            build_coeff_mat, root=0)
                        snap_mat = parallel.comm.bcast(
                            snap_mat, root=0)
                        true_modes = parallel.comm.bcast(
                            true_modes, root=0)
                        
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
                            _compute_modes, mode_nums, mode_path, 
                            snap_paths, build_coeff_mat, index_from=\
                            index_from)
                    # If the coeff mat has more rows than there are 
                    # snapshot paths
                    elif num_snaps > build_coeff_mat.shape[0]:
                        self.assertRaises(ValueError, self.fieldOperations.\
                            _compute_modes, mode_nums, mode_path,
                            snap_paths, build_coeff_mat, index_from=\
                            index_from)
                    elif num_modes > num_snaps:
                        self.assertRaises(ValueError,
                          self.fieldOperations._compute_modes, mode_nums,
                          mode_path, snap_paths, build_coeff_mat,
                          index_from=index_from)
                    else:
                        # Test the case that only one mode is desired,
                        # in which case user might pass in an int
                        if len(mode_nums) == 1:
                            mode_nums = mode_nums[0]

                        # Saves modes to files
                        self.fieldOperations._compute_modes(mode_nums, 
                            mode_path, snap_paths, build_coeff_mat, 
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
                            N.testing.assert_array_almost_equal(
                                computed_mode, true_modes[:,mode_num-\
                                index_from])
                                
                        parallel.sync()
       
        parallel.sync()

    def test_compute_inner_product_mat_types(self):
        def load_field_as_complex(path):
            return (1 + 1j) * util.load_mat_text(path) 

        num_row_snaps = 4
        num_col_snaps = 6
        num_states = 7

        if not os.path.isdir('files_modaldecomp_test'):
            SP.call(['mkdir', 'files_modaldecomp_test'])
        row_snap_path = 'files_modaldecomp_test/row_snap_%03d.txt'
        col_snap_path = 'files_modaldecomp_test/col_snap_%03d.txt'
        
        # generate snapshots and save to file, only do on proc 0
        parallel.sync()
        if parallel.is_rank_zero():
            row_snap_mat = N.mat(N.random.random((num_states,
                num_row_snaps)))
            col_snap_mat = N.mat(N.random.random((num_states,
                num_col_snaps)))
            row_snap_paths = []
            col_snap_paths = []
            for snap_index in xrange(num_row_snaps):
                path = row_snap_path % snap_index
                util.save_mat_text(row_snap_mat[:,snap_index],path)
                row_snap_paths.append(path)
            for snap_index in xrange(num_col_snaps):
                path = col_snap_path % snap_index
                util.save_mat_text(col_snap_mat[:,snap_index],path)
                col_snap_paths.append(path)
        else:
            row_snap_mat = None
            col_snap_mat = None
            row_snap_paths = None
            col_snap_paths = None
        if parallel.is_distributed():
            row_snap_mat = parallel.comm.bcast(row_snap_mat, root=0)
            col_snap_mat = parallel.comm.bcast(col_snap_mat, root=0)
            row_snap_paths = parallel.comm.bcast(row_snap_paths, root=0)
            col_snap_paths = parallel.comm.bcast(col_snap_paths, root=0)

        # If number of rows/cols is 1, test case that a string, not
        # a list, is passed in
        if len(row_snap_paths) == 1:
            row_snap_paths = row_snap_paths[0]
        if len(col_snap_paths) == 1:
            col_snap_paths = col_snap_paths[0]
    
        # Comptue inner product matrix and check type
        for load, type in [(load_field_as_complex, complex), (util.\
            load_mat_text, float)]:
            self.fieldOperations.load_field = load
            inner_product_mat = self.fieldOperations.compute_inner_product_mat(
                row_snap_paths, col_snap_paths)
            symm_inner_product_mat = self.fieldOperations.\
                compute_symmetric_inner_product_mat(row_snap_paths)
            self.assertEqual(inner_product_mat.dtype, type)
            self.assertEqual(symm_inner_product_mat.dtype, type)

    #@unittest.skip('testing other things')
    def test_compute_inner_product_mats(self):
        """
        Test computation of matrix of inner products in memory-efficient
        chunks, both in parallel (compute_inner_product_mat).
        """ 
        def assert_equal_mat_products(mat1, mat2, paths1, paths2):
            # Path list may actually be a string, in which case covert to list
            if isinstance(paths1, str):
                paths1 = [paths1]
            if isinstance(paths2, str):
                paths2 = [paths2]

            # True inner product matrix
            product_true = mat1 * mat2
           
            # Test computation as chunk (a serial method, tested on each proc)
            #productComputedAsChunk = self.fieldOperations.\
            #    _compute_inner_product_chunk(paths1, paths2)
            #N.testing.assert_array_almost_equal(productComputedAsChunk, 
            #    product_true)

            # Test paralleized computation.  
            product_computed_as_mat = \
                self.fieldOperations.compute_inner_product_mat(paths1, paths2)
            N.testing.assert_array_almost_equal(product_computed_as_mat, 
                product_true)
            
            
            # Test computation of symmetric inner product matrix
            if paths1 == paths2:  
                # First test complete upper triangular computation
                product_computed_as_symm_mat = self.fieldOperations.\
                    compute_symmetric_inner_product_mat(paths1)
                N.testing.assert_array_almost_equal(
                    product_computed_as_symm_mat, product_true)
            
        num_row_snaps_list =[1, int(round(self.total_num_fields_in_mem / 2.)), self.\
            total_num_fields_in_mem, self.total_num_fields_in_mem *2]
        num_col_snaps_list = num_row_snaps_list
        num_states = 6

        if not os.path.isdir('files_modaldecomp_test'):
            SP.call(['mkdir', 'files_modaldecomp_test'])
        row_snap_path = 'files_modaldecomp_test/row_snap_%03d.txt'
        col_snap_path = 'files_modaldecomp_test/col_snap_%03d.txt'
        
        for num_row_snaps in num_row_snaps_list:
            for num_col_snaps in num_col_snaps_list:
                # generate snapshots and save to file, only do on proc 0
                parallel.sync()
                if parallel.is_rank_zero():
                    row_snap_mat = N.mat(N.random.random((num_states,
                        num_row_snaps)))
                    col_snap_mat = N.mat(N.random.random((num_states,
                        num_col_snaps)))
                    row_snap_paths = []
                    col_snap_paths = []
                    for snap_index in xrange(num_row_snaps):
                        path = row_snap_path % snap_index
                        util.save_mat_text(row_snap_mat[:,snap_index],path)
                        row_snap_paths.append(path)
                    for snap_index in xrange(num_col_snaps):
                        path = col_snap_path % snap_index
                        util.save_mat_text(col_snap_mat[:,snap_index],path)
                        col_snap_paths.append(path)
                else:
                    row_snap_mat = None
                    col_snap_mat = None
                    row_snap_paths = None
                    col_snap_paths = None
                if parallel.is_distributed():
                    row_snap_mat = parallel.comm.bcast(row_snap_mat, root=0)
                    col_snap_mat = parallel.comm.bcast(col_snap_mat, root=0)
                    row_snap_paths = parallel.comm.bcast(row_snap_paths, root=0)
                    col_snap_paths = parallel.comm.bcast(col_snap_paths, root=0)

                # If number of rows/cols is 1, test case that a string, not
                # a list, is passed in
                if len(row_snap_paths) == 1:
                    row_snap_paths = row_snap_paths[0]
                if len(col_snap_paths) == 1:
                    col_snap_paths = col_snap_paths[0]

                # Test different rows and cols snapshots
                assert_equal_mat_products(row_snap_mat.T, col_snap_mat,
                    row_snap_paths, col_snap_paths)
                
                # Test with only the row data, to ensure nothing is
                # goes wrong when the same list is used twice
                # (potential memory issues, or lists may accidentally
                # get altered).  Also, test symmetric computation
                # method.
                assert_equal_mat_products(row_snap_mat.T, row_snap_mat,
                    row_snap_paths, row_snap_paths)
                        
if __name__=='__main__':
    unittest.main(verbosity=2)    

    
    

