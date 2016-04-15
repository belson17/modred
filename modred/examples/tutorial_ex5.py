from __future__ import print_function
from future.builtins import range
import numpy as np
import modred as mr

num_elements = 2000
num_vecs = 100

# Artificial sample times used as quadrature weights in POD.
quad_weights = np.logspace(1., 3., num=num_vecs)

base_vec_handle = mr.VecHandlePickle('base_vec.pkl')
snapshots = [mr.VecHandlePickle(
    'vec%d.pkl' %i, base_vec_handle=base_vec_handle, scale=quad_weights[i])
    for i in range(num_vecs)]
 
# Save arbitrary data, normally unnecessary.
num_elements = 2000  
parallel = mr.parallel_default_instance
if parallel.is_rank_zero():
    for snap in snapshots + [base_vec_handle]:
        snap.put(np.random.random(num_elements))
parallel.barrier()

# Compute and save POD modes.
my_POD = mr.PODHandles(np.vdot)
my_POD.compute_decomp(snapshots)
my_POD.put_decomp('sing_vals.txt', 'sing_vecs.txt')
my_POD.put_correlation_mat('correlation_mat.txt')
mode_indices = [1, 4, 5, 0, 10]
modes = [mr.VecHandleArrayText('mode%d.txt'%i) for i in mode_indices]
my_POD.compute_modes(mode_indices, modes)

# Check that modes are orthonormal
vec_space = mr.VectorSpaceHandles(inner_product=np.vdot)
IP_mat = vec_space.compute_symmetric_inner_product_mat(modes)
if not np.allclose(IP_mat, np.eye(len(mode_indices))):
    print('Warning: modes are not orthonormal', IP_mat)
