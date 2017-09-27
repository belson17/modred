from future.builtins import range
import os

import numpy as np

from modred import parallel
import modred as mr


# Create directory for output files
out_dir = 'tutorial_ex4_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Define non-uniform grid and corresponding inner product weights
nx = 80
ny = 100
x_grid = 1. - np.cos(np.linspace(0, np.pi, nx))
y_grid = np.linspace(0, 1., ny) ** 2
Y, X = np.meshgrid(y_grid, x_grid)

# Create random snapshot data
num_vecs = 100
snapshots = [
    mr.VecHandlePickle('%s/vec%d.pkl' % (out_dir, i)) for i in range(num_vecs)]
if parallel.is_rank_zero():
    for i, snap in enumerate(snapshots):
        snap.put(np.sin(X * i) + np.cos(Y * i))
parallel.barrier()

# Calculate DMD modes and save them to pickle files
weighted_IP = mr.InnerProductTrapz(x_grid, y_grid)
my_DMD = mr.DMDHandles(weighted_IP)
my_DMD.compute_decomp(snapshots)
my_DMD.put_decomp(
    '%s/eigvals.txt' % out_dir, '%s/R_low_order_eigvecs.txt' % out_dir,
    '%s/L_low_order_eigvecs.txt' % out_dir,
    '%s/correlation_array_eigvals.txt' % out_dir,
    '%s/correlation_array_eigvecs.txt' % out_dir)
mode_indices = [1, 4, 5, 0, 10]
modes = [
    mr.VecHandlePickle('%s/mode%d.pkl' % (out_dir, i)) for i in mode_indices]
my_DMD.compute_exact_modes(mode_indices, modes)
