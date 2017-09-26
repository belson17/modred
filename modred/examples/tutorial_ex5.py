from __future__ import print_function
from future.builtins import range
import os

import numpy as np

from modred import parallel
import modred as mr


# Create directory for output files
out_dir = 'tutorial_ex5_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Artificial sample times used as quadrature weights in POD
num_vecs = 100
quad_weights = np.logspace(1., 3., num=num_vecs)
base_vec_handle = mr.VecHandlePickle('%s/base_vec.pkl' % out_dir)
snapshots = [
    mr.VecHandlePickle(
        '%s/vec%d.pkl' % (out_dir, i),base_vec_handle=base_vec_handle,
        scale=quad_weights[i])
    for i in range(num_vecs)]

# Save arbitrary data, normally unnecessary
num_elements = 2000
if parallel.is_rank_zero():
    for snap in snapshots + [base_vec_handle]:
        snap.put(np.random.random(num_elements))
parallel.barrier()

# Compute and save POD modes
my_POD = mr.PODHandles(np.vdot)
my_POD.compute_decomp(snapshots)
my_POD.put_decomp('%s/sing_vals.txt' % out_dir, '%s/sing_vecs.txt' % out_dir)
my_POD.put_correlation_mat('%s/correlation_mat.txt' % out_dir)
mode_indices = [1, 4, 5, 0, 10]
modes = [
    mr.VecHandleArrayText('%s/mode%d.txt' % (out_dir, i)) for i in mode_indices]
my_POD.compute_modes(mode_indices, modes)

# Check that modes are orthonormal
vec_space = mr.VectorSpaceHandles(inner_product=np.vdot)
IP_mat = vec_space.compute_symm_inner_product_mat(modes)
if not np.allclose(IP_mat, np.eye(len(mode_indices))):
    print('Warning: modes are not orthonormal', IP_mat)
