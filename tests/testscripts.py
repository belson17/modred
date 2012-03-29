#!/usr/bin/env python

import unittest
import os
from os.path import join
import helper
helper.add_to_path('examples')
helper.add_to_path('src')
from parallel import default_instance
parallel = default_instance

class TestExampleScripts(unittest.TestCase):
    def setUp(self):
        cwd = os.path.dirname(__file__)
        self.examples_dir = join(join(cwd, '..'), 'examples')
    
    def test_main_bpod_disk(self):
        """Runs main_bpod_disk. If runs without error, passes test"""
        import main_bpod_disk as M
        M.main(make_plots=False, verbose=False)

    @unittest.skipIf(parallel.is_distributed(), 'Only test in serial')
    def test_main_pod_in_memory(self):
        """Runs main_simple_pod. If runs without error, passes test"""
        import main_pod_in_memory as M
        M.main(make_plots=False, verbose=False)
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
