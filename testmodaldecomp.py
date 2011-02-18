
import unittest
import numpy as N
import modaldecomp as MD

#import inspect #makes it possible to find information about a function

class TestModalDecomp(unittest.TestCase):
    """ Tests of the ModalDecomp class """
    
    def setUp(self):
        pass
    
    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly"""
        def my_load(fname): return N.ones((2,2))
        pass

    def test_compute_modes_chunk(self):
        """
        Test that can compute chunks of modes from arguments.
        
        The modes are saved to file so must read them in to test. The
        compute_modes_chunk method is only ever called on a per-proc basis
        so it does not need to be tested seperately for parallel and serial.
        """
    
    def test_compute_modes_serial(self):
        """
        Test that can compute modes from the arguments.
        
        This only needs to test serial and the components of compute_modes
        that is not handled by compute_modes_chunk. compute_modes_chunk
        does most of the work for actually computing the modes."""
        pass

    def test_compute_modes_parallel(self):
        """
        Test that can compute modes from arguments. 
        
        This only needs to test serial and the components of compute_modes
        that is not handled by compute_modes_chunk. compute_modes_chunk
        does most of the work for actually computing the modes."""
        pass
        
if __name__=='__main__':
    unittest.main()    

    
    

