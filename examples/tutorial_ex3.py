import numpy as N
import modred as MR

# Define the snapshots to be used
num_vecs = 30    
direct_snapshots = [MR.ArrayTextVecHandle('direct_vec%d.txt' % i) 
    for i in range(num_vecs)]
adjoint_snapshots = [MR.ArrayTextVecHandle('adjoint_vec%d.txt' % i)
    for i in range(num_vecs)]

# Save arrays in text files
# We use arbitrary fake data as a placeholder
x = N.linspace(0, N.pi, 200)
for i, snap in enumerate(direct_snapshots):
    snap.put([N.sin(x*i) for i in range(num_vecs)])
for i, snap in enumerate(adjoint_snapshots):
    snap.put([N.cos(0.5*x*i) for i in range(num_vecs)])

# Calculate BPOD modes
my_BPOD = MR.BPOD(N.vdot, max_vecs_per_node=10)
L_sing_vecs, sing_vals, R_sing_vecs = \
    my_BPOD.compute_decomp(direct_snapshots, adjoint_snapshots)

# The BPOD modes are saved to disk
num_modes = 10
mode_nums = range(num_modes)  
direct_modes = [MR.ArrayTextVecHandle('direct_mode%d' % i) 
    for i in mode_nums]
adjoint_modes = [MR.ArrayTextVecHandle('adjoint_mode%d' % i) 
    for i in mode_nums]
my_BPOD.compute_direct_modes(mode_nums, direct_modes)
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_modes)
