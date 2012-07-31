# Helper functions that are common in setting up tests.

import sys
def add_to_path(directory):
    """Add directory to sys.path for importing modules
    
    Args:
        ``directory``: a string to the directory.   
    """
    if sys.path.count(directory) == 0:
        sys.path.insert(0, directory)

"""
if rank==0:
    print 'To fully test, must do both:'
    print '  1) python testutil.py'
    print '  2) mpiexec -n <# procs> python testutil.py\n\n'
"""
