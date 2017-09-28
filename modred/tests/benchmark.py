"""
This script is for profiling and scaling.
There are individual functions for testing individual components of modred.

Then in python, do this to view the results, see load_prof_parallel.py

benchmark.py is to be used after installing modred.
"""
import os
from os.path import join
from shutil import rmtree
import argparse
import pickle
import time as T
import cProfile

import numpy as np

import modred as mr


parser = argparse.ArgumentParser(
    description='Get directory in which to save data.')
parser.add_argument(
    '--outdir', default='files_benchmark',
    help='Directory in which to save data.')
parser.add_argument(
    '--function',
    choices=['lin_combine', 'inner_product_array', 'symm_inner_product_array'],
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
    if not os.path.exists(vec_dir) and mr.parallel.is_rank_zero():
        os.mkdir(vec_dir)
    mr.parallel.barrier()

    """
    # Parallelize saving of vecs (may slow down sequoia)
    proc_vec_num_asignments = \
        mr.parallel.find_assignments(mr.range(num_vecs))[mr.parallel.getRank()]
    for vec_num in proc_vec_num_asignments:
        vec = np.random.random(num_states)
        save_vec(vec, vec_dir + vec_name%vec_num)
    """
    if mr.parallel.is_rank_zero():
        for handle in vec_handles:
            handle.put(np.random.random(num_states))

    mr.parallel.barrier()


def inner_product_array(
    num_states, num_rows, num_cols, max_vecs_per_node, verbosity=1):
    """
    Computes inner products from known vecs.

    Remember that rows correspond to adjoint modes and cols to direct modes
    """
    col_vec_handles = [mr.VecHandlePickle(join(data_dir, col_vec_name%col_num))
        for col_num in mr.range(num_cols)]
    row_vec_handles = [mr.VecHandlePickle(join(data_dir, row_vec_name%row_num))
        for row_num in mr.range(num_rows)]

    generate_vecs(data_dir, num_states, row_vec_handles+col_vec_handles)

    my_VS = mr.VectorSpaceHandles(
        np.vdot, max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)

    prof = cProfile.Profile()
    start_time = T.time()
    prof.runcall(
        my_VS.compute_inner_product_array, *(col_vec_handles, row_vec_handles))
    total_time = T.time() - start_time
    prof.dump_stats('IP_array_r%d.prof'%mr.parallel.get_rank())

    return total_time


def symm_inner_product_array(
    num_states, num_vecs, max_vecs_per_node, verbosity=1):
    """
    Computes symmetric inner product array from known vecs (as in POD).
    """
    vec_handles = [mr.VecHandlePickle(join(data_dir, row_vec_name%row_num))
        for row_num in mr.range(num_vecs)]

    generate_vecs(data_dir, num_states, vec_handles)

    my_VS = mr.VectorSpaceHandles(
        np.vdot, max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)

    prof = cProfile.Profile()
    start_time = T.time()
    prof.runcall(my_VS.compute_symm_inner_product_array, vec_handles)
    total_time = T.time() - start_time
    prof.dump_stats('IP_symm_array_r%d.prof'%mr.parallel.get_rank())

    return total_time


def lin_combine(
    num_states, num_bases, num_products, max_vecs_per_node, verbosity=1):
    """
    Computes linear combination of vecs from saved vecs and random coeffs

    num_bases is number of vecs to be linearly combined
    num_products is the resulting number of vecs
    """

    basis_handles = [mr.VecHandlePickle(join(data_dir, basis_name%basis_num))
        for basis_num in mr.range(num_bases)]
    product_handles = [mr.VecHandlePickle(join(data_dir,
        product_name%product_num))
        for product_num in mr.range(num_products)]

    generate_vecs(data_dir, num_states, basis_handles)
    my_VS = mr.VectorSpaceHandles(np.vdot, max_vecs_per_node=max_vecs_per_node,
        verbosity=verbosity)
    coeff_array = np.random.random((num_bases, num_products))
    mr.parallel.barrier()

    prof = cProfile.Profile()
    start_time = T.time()
    prof.runcall(my_VS.lin_combine, *(product_handles, basis_handles,
        coeff_array))
    total_time = T.time() - start_time
    prof.dump_stats('lincomb_r%d.prof'%mr.parallel.get_rank())
    return total_time


def clean_up():
    mr.parallel.barrier()
    if mr.parallel.is_rank_zero():
        try:
            rmtree(data_dir)
        except:
            pass

def main():
    #method_to_test = 'lin_combine'
    #method_to_test = 'inner_product_array'
    #method_to_test = 'symm_inner_product_array'
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
    elif method_to_test == 'inner_product_array':
        # inner_product_array test
        num_rows = 1200
        num_cols = 1200
        time_elapsed = inner_product_array(num_states, num_rows, num_cols,
            max_vecs_per_node)

    elif method_to_test == 'symm_inner_product_array':
        # symm_inner_product_array test
        num_vecs = 1200
        time_elapsed = symm_inner_product_array(
            num_states, num_vecs, max_vecs_per_node)
    else:
        print(
            'Did not recognize --function argument. Choose from: lin_combine, '
            'inner_product_array, symm_inner_product_array.')
    #print('Time for %s is %f' % (method_to_test, time_elapsed))

    mr.parallel.barrier()
    clean_up()


if __name__ == '__main__':
    main()
