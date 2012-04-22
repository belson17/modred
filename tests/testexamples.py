#!/usr/bin/env python

import unittest
import os
from os.path import join
from shutil import rmtree
import helper
helper.add_to_path('examples')
helper.add_to_path('src')
from parallel import default_instance
parallel = default_instance

class TestExamples(unittest.TestCase):
    def setUp(self):
        cwd = os.path.dirname(__file__)
        self.test_dir = 'DELETE_ME_test_tutorial_examples'
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():        
            os.mkdir(self.test_dir)
        parallel.barrier()
        os.chdir(self.test_dir)
        self.examples_dir = join(join('..', '..'), 'examples')
    
    def tearDown(self):
        os.chdir('..')
        parallel.barrier()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.barrier()
 
    def test_tutorial_examples(self):
        """Runs all tutorial examples. If run without errors, passes test"""
        example_module = 'tutorial_ex%d'
        for example_num in range(1, 7):
            # Example 3 isn't meant to work in parallel
            if not (parallel.is_distributed() and example_num != 3):
                exec('import %s as I'%(example_module%example_num))
                I.main(verbose=False)
            
        
    @unittest.skip('Unnecessary test for user')
    def test_benchmark(self):
        import benchmark as B
        num_states = 14
        num_bases = 10
        num_sums = 5
        max_vecs_per_node = 4
        time = B.lin_combine(num_states, num_bases, num_sums, 
            max_vecs_per_node, verbose=False)
        self.assertEqual(type(time), float)
        
        num_rows = 10
        num_cols = 12
        time = B.inner_product_mat(num_states, num_rows, num_cols, 
            max_vecs_per_node, verbose=False)
        self.assertEqual(type(time), float)
        
        time = B.symmetric_inner_product_mat(num_states, num_rows, 
            max_vecs_per_node, verbose=False)
        self.assertEqual(type(time), float)
        
        
        

if __name__ == '__main__':
    unittest.main()
