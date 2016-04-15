from __future__ import division
from __future__ import absolute_import
from future.builtins import range
import numpy as np
import modred as mr
from customvector import CustomVector, CustomVecHandle, inner_product


# Define snapshot handles.
direct_snapshots = [
    CustomVecHandle('direct_snap%d.pkl' % i, scale=np.pi) for i in range(10)]
adjoint_snapshots = [
    CustomVecHandle('adjoint_snap%d.pkl' % i, scale=np.pi) for i in range(10)]
    
# Arbitrary data.
parallel = mr.parallel_default_instance
nx = 50
ny = 30
nz = 20
x = np.linspace(0, 1, nx)
y = np.logspace(1, 2, ny)
z = np.linspace(0, 1, nz) ** 2
if parallel.is_rank_zero():
    for snap in direct_snapshots + adjoint_snapshots:
        snap.put(CustomVector([x, y, z], np.random.random((nx, ny, nz))))
parallel.barrier()

# Compute and save Balanced POD modes.
my_BPOD = mr.BPODHandles(inner_product)
my_BPOD.sanity_check(direct_snapshots[0])
sing_vals, L_sing_vecs, R_sing_vecs = my_BPOD.compute_decomp(
    direct_snapshots, adjoint_snapshots)

# less than 10% error
sing_vals_norm = sing_vals / np.sum(sing_vals)
num_modes = np.nonzero(np.cumsum(sing_vals_norm) > 0.9)[0][0] + 1
mode_nums = list(range(num_modes))

direct_modes = [CustomVecHandle('direct_mode%d.pkl'%i) for i in mode_nums] 
adjoint_modes = [CustomVecHandle('adjoint_mode%d.pkl'%i) for i in mode_nums]

my_BPOD.compute_direct_modes(mode_nums, direct_modes)
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_modes)
