from future.builtins import range
import os

import numpy as np

import modred as mr


# Create directory for output files
out_dir = 'tutorial_ex3_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Define the handles for the snapshots
num_vecs = 30
direct_snapshots = [
    mr.VecHandleArrayText('%s/direct_vec%d.txt' % (out_dir, i))
    for i in range(num_vecs)]
adjoint_snapshots = [
    mr.VecHandleArrayText('%s/adjoint_vec%d.txt' % (out_dir, i))
    for i in range(num_vecs)]

# Save arbitrary data in text files
x = np.linspace(0, np.pi, 200)
for i, snap in enumerate(direct_snapshots):
    snap.put(np.sin(x*i))
for i, snap in enumerate(adjoint_snapshots):
    snap.put(np.cos(0.5*x*i))

# Calculate and save BPOD modes
my_BPOD = mr.BPODHandles(np.vdot, max_vecs_per_node=10)
sing_vals, L_sing_vecs, R_sing_vecs = my_BPOD.compute_decomp(
    direct_snapshots, adjoint_snapshots)

num_modes = 10
mode_nums = list(range(num_modes))
direct_modes = [
    mr.VecHandleArrayText('%s/direct_mode%d' % (out_dir, i))
    for i in mode_nums]
adjoint_modes = [
    mr.VecHandleArrayText('%s/adjoint_mode%d' % (out_dir, i))
    for i in mode_nums]
my_BPOD.compute_direct_modes(mode_nums, direct_modes)
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_modes)
