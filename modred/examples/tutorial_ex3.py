from future.builtins import range
import numpy as np
import modred as mr

# Define the handles for the snapshots
num_vecs = 30    
direct_snapshots = [
    mr.VecHandleArrayText('direct_vec%d.txt' % i) 
    for i in range(num_vecs)]
adjoint_snapshots = [
    mr.VecHandleArrayText('adjoint_vec%d.txt' % i)
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
    mr.VecHandleArrayText('direct_mode%d' % i) 
    for i in mode_nums]
adjoint_modes = [
    mr.VecHandleArrayText('adjoint_mode%d' % i) 
    for i in mode_nums]
my_BPOD.compute_direct_modes(mode_nums, direct_modes)
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_modes)
