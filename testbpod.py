
import numpy as N
import bpod as BP
import unittest

class TestBPOD(unittest.TestCase):
    """ Test all the BPOD class methods """
    
    def setUp(self):
        pass
    
    def test_init(self):
        pass
    
    def test__compute_hankel_chunk(self):
        """Don't need to test this in parallel vs serial since it is only
        called on a per-processor basis """
        pass
        
    def test__compute_hankel_serial(self):
        """Tests that _compute_decomp works in serial"""
        pass
        
    def test__compute_hankel_parallel(self):
        """Tests that _compute_decomp works in parallel"""
        pass

    def test__svd(self):
        pass
        
    def test_compute_decomp(self):
        pass

    def test_compute_direct_modes(self):
        """ """"
        #Mostly need to tests the part unique to this function. The 
        #lower level functions are tested independently.
        pass
        
    def test_compute_adjoint_modes(self):
        pass
  
if __name__=='__main__':
    unittest.main()
    
        
    


