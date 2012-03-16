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
"""
import os
from os.path import join
from shutil import rmtree
import cPickle
import time as T
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

import fieldoperations as FO
import util


def save_pickle(obj, filename):
    fid = open(filename,'wb')
    cPickle.dump(obj,fid)
    fid.close()
    
    
def load_pickle(filename):
    fid = open(filename,'rb')
    obj = cPickle.load(fid)
    fid.close()
    return obj


save_field = save_pickle 
load_field = load_pickle
#save_field = util.save_mat_text
#load_field = util.load_mat_text
inner_product = util.inner_product


import argparse
parser = argparse.ArgumentParser(description='Get directory in which to ' +\
    'save data.')
parser.add_argument('--outdir', default='files_benchmark', help='Directory in ' +\
    'which to save data.')
parser.add_argument('--function', required=True, choices=['lin_combine',
    'inner_product_mat', 'symmetric_inner_product_mat'], help='Function to ' +\
    'benchmark.')
args = parser.parse_args()
data_dir = args.outdir

#if data_dir[-1] != '/':
#    join(data_dir, = '/'

def generate_fields(num_states, num_fields, field_dir, field_name):
    """
    Creates a data set of fields, saves to file.
    
    field_dir is the directory
    field_name is the file name and must include a %03d type string.
    """
    if not os.path.exists(field_dir) and parallel.is_rank_zero():
        os.mkdir(field_dir)
    
    """
    # Parallelize saving of fields (may slow down sequoia)
    proc_field_num_asignments = \
        parallel.find_assignments(range(num_fields))[parallel.getRank()]
    for field_num in proc_field_num_asignments:
        field = N.random.random(num_states)
        save_field(field, field_dir + field_name%field_num)
    """
    
    if parallel.is_rank_zero():
        for field_num in xrange(num_fields):
            field = N.random.random(num_states)
            save_field(field, join(field_dir, field_name%field_num))
    
    parallel.sync()


def inner_product_mat(num_states, num_rows, num_cols, max_fields_per_node):
    """
    Computes inner products from known fields.
    
    Remember that rows correspond to adjoint modes and cols to direct modes
    """    
    col_field_name = 'col_%04d.txt'
    col_field_paths = [join(data_dir, col_field_name%col_num) for col_num in range(num_cols)]
    generate_fields(num_states, num_cols, data_dir, col_field_name)
    
    row_field_name = 'row_%04d.txt'    
    row_field_paths = [join(data_dir, row_field_name%row_num) for row_num in range(num_rows)]
    generate_fields(num_states, num_rows, data_dir, row_field_name)
    
    my_FO = FO.FieldOperations(max_fields_per_node=max_fields_per_node, put_field=\
        save_field, get_field=load_field, inner_product=inner_product, 
        verbose=True) 
    
    start_time = T.time()
    inner_product_mat = my_FO.compute_inner_product_mat(col_field_paths, 
        row_field_paths)
    total_time = T.time() - start_time
    return total_time
    
    
def symmetric_inner_product_mat(num_states, num_fields, max_fields_per_node):
    """
    Computes symmetric inner product matrix from known fields (as in POD).
    """    
    field_name = 'field_%04d.txt'
    fieldPaths = [join(data_dir, field_name % field_num) for field_num in range(
        num_fields)]
    generate_fields(num_states, num_fields, data_dir, field_name)
    
    my_FO = FO.FieldOperations(max_fields_per_node=max_fields_per_node, put_field=\
        save_field, get_field=load_field, inner_product=inner_product, 
        verbose=True) 
    
    start_time = T.time()
    inner_product_mat = my_FO.compute_symmetric_inner_product_mat(fieldPaths)
    total_time = T.time() - start_time
    return total_time


def lin_combine(num_states, num_bases, num_products, max_fields_per_node):
    """
    Computes linear combination of fields from saved fields and random coeffs
    
    num_bases is number of fields to be linearly combined
    num_products is the resulting number of fields
    """
    basis_name = 'snap_%04d.txt'
    product_name = 'product_%04d.txt'
    generate_fields(num_states, num_bases, data_dir, basis_name)
    my_FO = FO.FieldOperations(max_fields_per_node=max_fields_per_node,
        put_field=save_field, get_field=load_field, inner_product=inner_product)
    coeff_mat = N.random.random((num_bases, num_products))
    
    basis_paths = [join(data_dir,  basis_name%basis_num) for basis_num in range(num_bases)]
    product_paths = [join(data_dir,  product_name%product_num) for product_num in range \
        (num_products)]
    
    start_time = T.time()
    my_FO.lin_combine(product_paths, basis_paths, coeff_mat)
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
    max_fields_per_node = 50
    num_states = 10000
    
    # Run test of choice
    if method_to_test == 'lin_combine':
        # lin_combine test
        num_bases = 2500
        num_products = 1000
        time_elapsed = lin_combine(
                num_states, num_bases, num_products, max_fields_per_node)
    elif method_to_test == 'inner_product_mat':
        # inner_product_mat test
        num_rows = 2000
        num_cols = 2000
        time_elapsed = inner_product_mat(
                num_states, num_rows, num_cols, max_fields_per_node)
    elif method_to_test == 'symmetric_inner_product_mat':
        # symmetric_inner_product_mat test
        num_fields = 2000
        time_elapsed = symmetric_inner_product_mat(
                num_states, num_fields, max_fields_per_node)
    print 'Time for %s is %f'%(method_to_test, time_elapsed)
    
    parallel.sync()
    clean_up()
    

if __name__ == '__main__':
    main()

    
    

