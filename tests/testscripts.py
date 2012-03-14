
import unittest
import os
from os.path import join
import helper
helper.add_to_path('examples')

class TestExampleScripts(unittest.TestCase):
    def setUp(self):
        cwd = os.path.dirname(__file__)
        self.examples_dir = join(join(cwd, '..'), 'examples')
    
    def test_main_bpod_disk(self):
        """Runs main_bpod_disk. If runs without error, passes test"""
        import main_bpod_disk as M
        M.main(make_plots=False, verbose=False)

    def test_main_simple_pod(self):
        """Runs main_simple_pod. If runs without error, passes test"""
        import main_simple_pod as M
        M.main(make_plots=False, verbose=False)

if __name__ == '__main__':
    unittest.main()
