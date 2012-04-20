# Helper functions that are common in setting up tests.

import sys, os

def add_to_path(directory):
    """Add directory to sys.path for importing modules
    
    directory must be a string relative to the modred top level, such
    as "src" or "examples".
    """
    dir_loc = os.path.join(os.path.join(os.path.dirname(__file__), '..'),
        directory)
    if sys.path[0] != dir_loc:
        sys.path.insert(0, dir_loc)


"""
if rank==0:
    print 'To fully test, must do both:'
    print '  1) python testutil.py'
    print '  2) mpiexec -n <# procs> python testutil.py\n\n'
"""
