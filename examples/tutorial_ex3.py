import numpy as N
import modred as MR

# Define the handles for the snapshots
num_vecs = 30    
direct_snapshots = [MR.VecHandleArrayText('direct_vec%d.txt' % i) 
    for i in range(num_vecs)]
adjoint_snapshots = [MR.VecHandleArrayText('adjoint_vec%d.txt' % i)
    for i in range(num_vecs)]

# Save arbitrary data in text files
x = N.linspace(0, N.pi, 200)
for i, snap in enumerate(direct_snapshots):
    snap.put([N.sin(x*i) for i in range(num_vecs)])
for i, snap in enumerate(adjoint_snapshots):
    snap.put([N.cos(0.5*x*i) for i in range(num_vecs)])

# Calculate and save BPOD modes
my_BPOD = MR.BPODHandles(N.vdot, max_vecs_per_node=10)
L_sing_vecs, sing_vals, R_sing_vecs = \
    my_BPOD.compute_decomp(direct_snapshots, adjoint_snapshots)

num_modes = 10
mode_nums = range(num_modes)  
direct_modes = [MR.VecHandleArrayText('direct_mode%d' % i) 
    for i in mode_nums]
adjoint_modes = [MR.VecHandleArrayText('adjoint_mode%d' % i) 
    for i in mode_nums]
my_BPOD.compute_direct_modes(mode_nums, direct_modes)
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_modes)
