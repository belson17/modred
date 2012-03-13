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

class TestGetPutField(unittest.TestCase):
    """Test methods ``get_field`` and ``put_field`` which are used by all SimpleUse classes."""
    def setUp(self):
        self.num_fields = 10
        self.num_states = 15
        self.fields = N.random.random((self.num_states, self.num_fields))
        
    def tearDown(self):
        pass
        
    def test_get_field(self):
        """Check that ``get_field`` retrieves a column of an array."""
        for field_index in [3,1,9]:
            N.testing.assert_array_equal(self.fields[:,field_index], 
                SU.get_field((self.fields, field_index)))
    
    def test_put_field(self):
        """Puts fields into another array, checks modified properly"""
        new_fields = N.random.random((self.num_states, self.num_fields))
        for field_index in [3,1,9]:
            SU.put_field(self.fields[:,field_index], (new_fields, field_index))
            N.testing.assert_array_equal(self.fields[:,field_index],
                new_fields[:,field_index])
    

class TestSimpleUsePOD(unittest.TestCase):
    def setUp(self):
        self.num_fields = 10
        self.num_states = 15
        self.fields = N.random.random((self.num_states, self.num_fields))
        self.my_POD = SU.SimpleUsePOD(verbose=False)
        self.my_POD.set_fields(self.fields)
        
    def tearDown(self):
        pass
    
    def test_all(self):
        """Tests computation of modes from fields. """
        num_modes = self.num_fields/2
        sing_vecs, sing_vals = self.my_POD.compute_decomp()
        self.assertEqual(sing_vecs.shape, (self.num_fields, self.num_fields))
        
        modes = self.my_POD.compute_modes(num_modes)
        self.assertEqual(modes.shape, (self.num_states, num_modes))
        


class TestSimpleUseBPOD(unittest.TestCase):
    def setUp(self):
        self.num_direct_fields = 10
        self.num_adjoint_fields = 9
        self.num_states = 15
        self.num_direct_modes = 5
        self.num_adjoint_modes = 6
        self.direct_fields = N.random.random((self.num_states, 
            self.num_direct_fields))
        self.adjoint_fields = N.random.random((self.num_states, 
            self.num_adjoint_fields))
        self.my_BPOD = SU.SimpleUseBPOD(verbose=False)
        self.my_BPOD.set_direct_fields(self.direct_fields)
        self.my_BPOD.set_adjoint_fields(self.adjoint_fields)
        
    def tearDown(self):
        pass
    
    def test_all(self):
        """Tests computation of modes from fields."""
        L_sing_vecs, sing_vals, R_sing_vecs = self.my_BPOD.compute_decomp()
        min_fields = min(self.num_direct_fields, self.num_adjoint_fields)
        self.assertEqual(L_sing_vecs.shape, (self.num_adjoint_fields, min_fields))
        self.assertEqual(R_sing_vecs.shape, (self.num_direct_fields, min_fields))
        self.assertEqual(sing_vals.shape, (min_fields,))
        
        direct_modes = self.my_BPOD.compute_direct_modes(self.num_direct_modes)
        adjoint_modes = self.my_BPOD.compute_adjoint_modes(self.num_adjoint_modes)
        self.assertEqual(direct_modes.shape, (self.num_states, self.num_direct_modes))
        self.assertEqual(adjoint_modes.shape, (self.num_states, self.num_adjoint_modes))
        
        

class TestSimpleUseDMD(unittest.TestCase):
    def setUp(self):
        self.num_fields = 10
        self.num_states = 15
        self.fields = N.random.random((self.num_states, self.num_fields))
        self.my_DMD = SU.SimpleUseDMD(verbose=False)
        self.my_DMD.set_fields(self.fields)
        
    def tearDown(self):
        pass
    
    def test_all(self):
        """Tests computation of modes from fields."""
        index_from = 2
        mode_nums = [5,6,4,2]
        ritz_vals, mode_norms = self.my_DMD.compute_decomp()
        self.assertEqual(ritz_vals.shape, ((self.num_fields-1),))
        self.assertEqual(mode_norms.shape, ((self.num_fields-1),))

        modes = self.my_DMD.compute_modes(mode_nums, index_from=index_from)
        self.assertEqual(modes.shape, (self.num_states, max(mode_nums)-index_from+1))
        

if __name__ == '__main__':
    unittest.main()
