
import numpy
import util
import unittest

class TestUtil(unittest.TestCase):
    """Tests all of the functions in util.py"""
    
    def setUp(self):
        pass
    
    def test_read_write_mat_text(self):
        """Test that can read/write text matrices"""
        
        #make a random matrix, write it, read it, assert almost equal
        #try with a few different arguments (like delimiters), 1D matrix
        #(vector) with all rows, all colulmns, big matrices, etc
        pass

    def test_MPI(self):
        """Test that the MPI object uses arguments correctly.
        
        Tests must be run in parallel (with mpiexec -n 2). Also test that
        when in serial, it defaults to a good behavior."""
        pass    
    
if __name__=='__main__':
    unittest.main()

