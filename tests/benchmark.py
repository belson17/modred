"""
This script is for profiling and scaling. 
There are individual functions for testing individual components of modred. 

To profile, do the following::
  
  python -m cProfile -o <path_to_output_file> <path_to_python_script>
    
In parallel, simply prepend with::

  mpiexec -n <number of procs> ...

Then in python, do this to view the results::

  import pstats
  pstats.Stats(<path_to_output_file>).strip_dirs().sort_stats('cumulative').\
      print_stats(<number_significant_lines_to_print>)

Also see load_prof_parallel.py
      
benchmark.py is to be used after installing modred.
"""
import os
from os.path import join
from shutil import rmtree
import cPickle
import time as T
import numpy as N
import cProfile

import modred as MR

parallel = MR.parallel_default_instance

import argparse
parser = argparse.ArgumentParser(description='Get directory in which to ' +\
    'save data.')
parser.add_argument('--outdir', default='files_benchmark', help='Directory in ' +\
    'which to save data.')
parser.add_argument('--function', choices=['lin_combine',
    'inner_product_mat', 'symmetric_inner_product_mat'], help='Function to ' +\
    'benchmark.')
args = parser.parse_args()
data_dir = args.outdir

#if data_dir[-1] != '/':
#    join(data_dir, = '/'

basis_name = 'vec_%04d'
product_name = 'product_%04d'
col_vec_name = 'col_%04d'
row_vec_name = 'row_%04d'

def generate_vecs(num_states, num_vecs, vec_dir, vec_name):
    """
    Creates a data set of vecs, saves to file.
    
    vec_dir is the directory
    vec_name is the file name and must include a %03d type string.
    """
    if not os.path.exists(vec_dir) and parallel.is_rank_zero():
        os.mkdir(vec_dir)
    parallel.barrier()
    
    """
    # Parallelize saving of vecs (may slow down sequoia)
    proc_vec_num_asignments = \
        parallel.find_assignments(range(num_vecs))[parallel.getRank()]
    for vec_num in proc_vec_num_asignments:
        vec = N.random.random(num_states)
        save_vec(vec, vec_dir + vec_name%vec_num)
    """
    
    if parallel.is_rank_zero():
        for vec_num in xrange(num_vecs):
            vec = N.random.random(num_states)
            my_vec_defs.put_vec(vec, join(vec_dir, vec_name%vec_num))
    
    parallel.barrier()


def inner_product_mat(num_states, num_rows, num_cols, max_vecs_per_node, 
    verbosity=1):
    """
    Computes inner products from known vecs.
    
    Remember that rows correspond to adjoint modes and cols to direct modes
    """    

    col_vec_paths = [join(data_dir, col_vec_name%col_num) for col_num in range(num_cols)]
    generate_vecs(num_states, num_cols, data_dir, col_vec_name)
    
    row_vec_paths = [join(data_dir, row_vec_name%row_num) for row_num in range(num_rows)]   
    my_VS = MR.VectorSpace(my_vec_defs, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity) 
    
    start_time = T.time()
    inner_product_mat = my_VS.compute_inner_product_mat(col_vec_paths, 
        row_vec_paths)
    total_time = T.time() - start_time
    return total_time
    
    
def symmetric_inner_product_mat(num_states, num_vecs, max_vecs_per_node, 
    verbosity=1):
    """
    Computes symmetric inner product matrix from known vecs (as in POD).
    """    
    vec_name = 'vec_%04d'
    vec_paths = [join(data_dir, vec_name % vec_num) for vec_num in range(
        num_vecs)]
    
    my_VS = MR.VectorSpace(my_vec_defs, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity) 
    
    start_time = T.time()
    inner_product_mat = my_VS.compute_symmetric_inner_product_mat(vec_paths)
    total_time = T.time() - start_time
    return total_time


def lin_combine(num_states, num_bases, num_products, max_vecs_per_node,
    verbosity=1):
    """
    Computes linear combination of vecs from saved vecs and random coeffs
    
    num_bases is number of vecs to be linearly combined
    num_products is the resulting number of vecs
    """

    my_VS = MR.VectorSpace(my_vec_defs, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity)
    coeff_mat = N.random.random((num_bases, num_products))
    
    basis_paths = [join(data_dir, basis_name%basis_num) for basis_num in range(num_bases)]
    product_paths = [join(data_dir, product_name%product_num) for product_num in range \
        (num_products)]
    
    start_time = T.time()
    my_VS.lin_combine(product_paths, basis_paths, coeff_mat)
    total_time = T.time() - start_time
    return total_time
    
    
def clean_up():
    if parallel.is_rank_zero():
        rmtree(data_dir)


def main():
    #method_to_test = 'lin_combine'
    #method_to_test = 'inner_product_mat'
    #method_to_test = 'symmetric_inner_product_mat'
    method_to_test = args.function
    
    # Common parameters
    max_vecs_per_node = 5
    num_states = 900
    
    # Run test of choice
    if method_to_test == 'lin_combine':
        # lin_combine test
        num_bases = 800
        num_products = 400
        generate_vecs(num_states, num_bases, data_dir, basis_name)
        #time_elapsed = lin_combine(
        #        num_states, num_bases, num_products, max_vecs_per_node)
        cProfile.runctx('lin_combine(num_states, num_bases, num_products, '+\
            'max_vecs_per_node)', globals(), {'num_states': num_states,
            'num_bases': num_bases, 'num_products': num_products, 
            'max_vecs_per_node': max_vecs_per_node}, 'lincomb_r%d.prof'%parallel.get_rank())
    elif method_to_test == 'inner_product_mat':
        # inner_product_mat test
        num_rows = 2000
        num_cols = 2000
        generate_vecs(num_states, num_vecs, data_dir, vec_name)
        parallel.barrier()
        time_elapsed = inner_product_mat(
                num_states, num_rows, num_cols, max_vecs_per_node)
    elif method_to_test == 'inner_product_mat':
        # symmetric_inner_product_mat test
        num_vecs = 2000
        generate_vecs(num_states, num_vecs, data_dir, vec_name)
        parallel.barrier()
        time_elapsed = symmetric_inner_product_mat(
                num_states, num_vecs, max_vecs_per_node)
    else:
        print 'Did not recognize --function argument, choose from'
        print 'lin_combine, inner_product_mat, and inner_product_mat'
    #print 'Time for %s is %f'%(method_to_test, time_elapsed)
    
    parallel.barrier()
    clean_up()
    

if __name__ == '__main__':
    main()

    
    

