""" This module collects all of the tests and runs them"""

import os
import unittest

import helper
helper.add_to_path('src')
#import parallel as parallel_mod
#parallel = parallel_mod.parallel_default_instance

# Check if we have discover function, if not use unittest2 
# This might be compatible with some python versions < 2.7
"""
try:
    dummy = unittest.defaultTestLoader.discover(os.path.dirname(__file__))
except:
    import unittest2 as unittest
"""

def run():
    test_loader = unittest.defaultTestLoader
    #print 'discovering tests in path',os.path.dirname(__file__)
    test_suites = test_loader.discover(os.path.dirname(__file__))
        #, top_level_dir='../')
    unittest.TextTestRunner(buffer=True).run(test_suites)
    #parallel.barrier()

if __name__ == '__main__':
    run()
