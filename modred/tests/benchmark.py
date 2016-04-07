"""
This script is for profiling and scaling. 
There are individual functions for testing individual components of modred. 

Then in python, do this to view the results, see load_prof_parallel.py
      
benchmark.py is to be used after installing modred.
"""
from __future__ import print_function
from future import standard_library
standard_library.install_hooks()
from future.builtins import range
import os
from os.path import join
from shutil import rmtree
import argparse
import pickle
import time as T
import cProfile

import numpy as np

import modred as mr


_parallel = mr.parallel_default_instance

parser = argparse.ArgumentParser(description='Get directory in which to ' +\
    'save data.')
parser.add_argument('--outdir', default='files_benchmark', 
    help='Directory in which to save data.')
parser.add_argument('--function', choices=['lin_combine',
    'inner_product_mat', 'symmetric_inner_product_mat'], 
    help='Function to benchmark.')
args = parser.parse_args()
data_dir = args.outdir

#if data_dir[-1] != '/':
#    join(data_dir, = '/'

basis_name = 'vec_%04d'
product_name = 'product_%04d'
col_vec_name = 'col_%04d'
row_vec_name = 'row_%04d'

def generate_vecs(vec_dir, num_states, vec_handles):
    """Creates a random data set of vecs, saves to file."""
    if not os.path.exists(vec_dir) and _parallel.is_rank_zero():
        os.mkdir(vec_dir)
    _parallel.barrier()
    
    """
    # Parallelize saving of vecs (may slow down sequoia)
    proc_vec_num_asignments = \
        _parallel.find_assignments(range(num_vecs))[_parallel.getRank()]
    for vec_num in proc_vec_num_asignments:
        vec = np.random.random(num_states)
        save_vec(vec, vec_dir + vec_name%vec_num)
    """
    if _parallel.is_rank_zero():
        for handle in vec_handles:
            handle.put(np.random.random(num_states))
    
    _parallel.barrier()

def inner_product_mat(num_states, num_rows, num_cols, max_vecs_per_node, 
    verbosity=1):
    """
    Computes inner products from known vecs.
    
    Remember that rows correspond to adjoint modes and cols to direct modes
    """
    col_vec_handles = [mr.VecHandlePickle(join(data_dir, col_vec_name%col_num))
        for col_num in range(num_cols)]
    row_vec_handles = [mr.VecHandlePickle(join(data_dir, row_vec_name%row_num))
        for row_num in range(num_rows)]
    
    generate_vecs(data_dir, num_states, row_vec_handles+col_vec_handles)
    
    my_VS = mr.VectorSpaceHandles(np.vdot, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity) 
    
    prof = cProfile.Profile()
    start_time = T.time()
    prof.runcall(my_VS.compute_inner_product_mat, *(col_vec_handles,  
        row_vec_handles))
    total_time = T.time() - start_time
    prof.dump_stats('IP_mat_r%d.prof'%_parallel.get_rank())

    return total_time
    
def symmetric_inner_product_mat(num_states, num_vecs, max_vecs_per_node, 
    verbosity=1):
    """
    Computes symmetric inner product matrix from known vecs (as in POD).
    """    
    vec_handles = [mr.VecHandlePickle(join(data_dir, row_vec_name%row_num))
        for row_num in range(num_vecs)]
    
    generate_vecs(data_dir, num_states, vec_handles)
    
    my_VS = mr.VectorSpaceHandles(np.vdot, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity) 
    
    prof = cProfile.Profile()
    start_time = T.time()
    prof.runcall(my_VS.compute_symmetric_inner_product_mat, vec_handles)
    total_time = T.time() - start_time
    prof.dump_stats('IP_symmetric_mat_r%d.prof'%_parallel.get_rank())

    return total_time

def lin_combine(num_states, num_bases, num_products, max_vecs_per_node,
    verbosity=1):
    """
    Computes linear combination of vecs from saved vecs and random coeffs
    
    num_bases is number of vecs to be linearly combined
    num_products is the resulting number of vecs
    """

    basis_handles = [mr.VecHandlePickle(join(data_dir, basis_name%basis_num))
        for basis_num in range(num_bases)]
    product_handles = [mr.VecHandlePickle(join(data_dir, 
        product_name%product_num))
        for product_num in range(num_products)]

    generate_vecs(data_dir, num_states, basis_handles)
    my_VS = mr.VectorSpaceHandles(np.vdot, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity)
    coeff_mat = np.random.random((num_bases, num_products))
    _parallel.barrier()

    prof = cProfile.Profile()
    start_time = T.time()
    prof.runcall(my_VS.lin_combine, *(product_handles, basis_handles, 
        coeff_mat)) 
    total_time = T.time() - start_time
    prof.dump_stats('lincomb_r%d.prof'%_parallel.get_rank())
    return total_time
    
def clean_up():
    _parallel.barrier()
    if _parallel.is_rank_zero():
        rmtree(data_dir)

def main():
    #method_to_test = 'lin_combine'
    #method_to_test = 'inner_product_mat'
    #method_to_test = 'symmetric_inner_product_mat'
    method_to_test = args.function
    
    # Common parameters
    max_vecs_per_node = 60
    num_states = 900
    
    # Run test of choice
    if method_to_test == 'lin_combine':
        # lin_combine test
        num_bases = 800
        num_products = 400
        time_elapsed = lin_combine(
                num_states, num_bases, num_products, max_vecs_per_node)
    elif method_to_test == 'inner_product_mat':
        # inner_product_mat test
        num_rows = 1200
        num_cols = 1200
        time_elapsed = inner_product_mat(num_states, num_rows, num_cols, 
            max_vecs_per_node)

    elif method_to_test == 'symmetric_inner_product_mat':
        # symmetric_inner_product_mat test
        num_vecs = 12000
        time_elapsed = symmetric_inner_product_mat(
                num_states, num_vecs, max_vecs_per_node)
    else:
        print('Did not recognize --function argument, choose from')
        print('lin_combine, inner_product_mat, and inner_product_mat')
    #print 'Time for %s is %f'%(method_to_test, time_elapsed)
    
    _parallel.barrier()
    clean_up()
    

if __name__ == '__main__':
    main()

    
    

