# Helper functions that are common in setting up tests.

import sys, os

def add_src_to_path():
    """Add src directory to sys.path for importing modules"""
    src_dir = os.path.join(os.path.join(os.path.dirname(__file__), '..'),
        'src')
    if sys.path[0] != src_dir:
        sys.path.insert(0, src_dir)
"""
def get_rank_distributed(): 
     
    #Returns rank and distributed boolean, needed for testing in
    #parallel.
    
    try: 
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        distributed = MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        print 'Warning: without mpi4py module, only serial behavior is tested'
        distributed = False
        rank = 0
    return rank, is_distributed
""" 

"""
if rank==0:
    print 'To fully test, must do both:'
    print '  1) python testutil.py'
    print '  2) mpiexec -n <# procs> python testutil.py\n\n'
"""
