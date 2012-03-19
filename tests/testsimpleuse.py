#!/usr/bin/env python

import unittest
import os
import numpy as N
from os.path import join
from shutil import rmtree
import copy

import helper
helper.add_to_path('examples')
helper.add_to_path('src')
import util
import simpleuse as SU

class TestGetPutVec(unittest.TestCase):
    """Test methods ``get_vec`` and ``put_vec`` which are used by all SimpleUse classes."""
    def setUp(self):
        self.num_vecs = 10
        self.num_states = 15
        self.vecs = N.random.random((self.num_states, self.num_vecs))
        
    def tearDown(self):
        pass
        
    def test_get_vec(self):
        """Check that ``get_vec`` retrieves a column of an array."""
        for vec_index in [3,1,9]:
            N.testing.assert_array_equal(self.vecs[:,vec_index], 
                SU.get_vec((self.vecs, vec_index)))
    
    def test_put_vec(self):
        """Puts vecs into another array, checks modified properly"""
        new_vecs = N.random.random((self.num_states, self.num_vecs))
        for vec_index in [3,1,9]:
            SU.put_vec(self.vecs[:,vec_index], (new_vecs, vec_index))
            N.testing.assert_array_equal(self.vecs[:,vec_index],
                new_vecs[:,vec_index])
    

class TestSimpleUsePOD(unittest.TestCase):
    def setUp(self):
        self.num_vecs = 10
        self.num_states = 15
        self.vecs = N.random.random((self.num_states, self.num_vecs))
        self.my_POD = SU.SimpleUsePOD(verbose=False)
        self.my_POD.set_vecs(self.vecs)
        
    def tearDown(self):
        pass
    
    def test_all(self):
        """Tests computation of modes from vecs. """
        num_modes = self.num_vecs/2
        sing_vecs, sing_vals = self.my_POD.compute_decomp()
        self.assertEqual(sing_vecs.shape, (self.num_vecs, self.num_vecs))
        
        modes = self.my_POD.compute_modes(num_modes)
        self.assertEqual(modes.shape, (self.num_states, num_modes))
        


class TestSimpleUseBPOD(unittest.TestCase):
    def setUp(self):
        self.num_direct_vecs = 10
        self.num_adjoint_vecs = 9
        self.num_states = 15
        self.num_direct_modes = 5
        self.num_adjoint_modes = 6
        self.direct_vecs = N.random.random((self.num_states, 
            self.num_direct_vecs))
        self.adjoint_vecs = N.random.random((self.num_states, 
            self.num_adjoint_vecs))
        self.my_BPOD = SU.SimpleUseBPOD(verbose=False)
        self.my_BPOD.set_direct_vecs(self.direct_vecs)
        self.my_BPOD.set_adjoint_vecs(self.adjoint_vecs)
        
    def tearDown(self):
        pass
    
    def test_all(self):
        """Tests computation of modes from vecs."""
        L_sing_vecs, sing_vals, R_sing_vecs = self.my_BPOD.compute_decomp()
        min_vecs = min(self.num_direct_vecs, self.num_adjoint_vecs)
        self.assertEqual(L_sing_vecs.shape, (self.num_adjoint_vecs, min_vecs))
        self.assertEqual(R_sing_vecs.shape, (self.num_direct_vecs, min_vecs))
        self.assertEqual(sing_vals.shape, (min_vecs,))
        
        direct_modes = self.my_BPOD.compute_direct_modes(self.num_direct_modes)
        adjoint_modes = self.my_BPOD.compute_adjoint_modes(self.num_adjoint_modes)
        self.assertEqual(direct_modes.shape, (self.num_states, self.num_direct_modes))
        self.assertEqual(adjoint_modes.shape, (self.num_states, self.num_adjoint_modes))
        
        

class TestSimpleUseDMD(unittest.TestCase):
    def setUp(self):
        self.num_vecs = 10
        self.num_states = 15
        self.vecs = N.random.random((self.num_states, self.num_vecs))
        self.my_DMD = SU.SimpleUseDMD(verbose=False)
        self.my_DMD.set_vecs(self.vecs)
        
    def tearDown(self):
        pass
    
    def test_all(self):
        """Tests computation of modes from vecs."""
        index_from = 2
        mode_nums = [5,6,4,2]
        ritz_vals, mode_norms = self.my_DMD.compute_decomp()
        self.assertEqual(ritz_vals.shape, ((self.num_vecs-1),))
        self.assertEqual(mode_norms.shape, ((self.num_vecs-1),))

        modes = self.my_DMD.compute_modes(mode_nums, index_from=index_from)
        self.assertEqual(modes.shape, (self.num_states, max(mode_nums)-index_from+1))
        

if __name__ == '__main__':
    unittest.main()
