import numpy as N
import modred as MR

num_vecs = 30

direct_snap_handles = [MR.ArrayTextVecHandle('direct_vec%d.txt'%i) 
for i in range(num_vecs)]
adjoint_snap_handles = [MR.ArrayTextVecHandle('adjoint_vec%d.txt'%i)
for i in range(num_vecs)]

# Save arrays in text files
# We use arbitrary fake data as a placeholder
x = N.arange(0, N.pi, 10000)
for i, handle in enumerate(direct_snap_handles):
    handle.put([N.sin(x*0.1*i) for i in range(num_vecs)])
for i, handle in enumerate(adjoint_snap_handles):
    handle.put([N.cos(x*0.1*i) for i in range(num_vecs)])

my_BPOD = MR.BPOD(inner_product=N.vdot, max_vecs_per_node=10)
L_sing_vecs, sing_vals, R_sing_vecs = \
    my_BPOD.compute_decomp(direct_snap_handles, adjoint_snap_handles)

# The BPOD modes are saved to disk.
num_modes = 15
mode_nums = range(num_modes)  
direct_mode_handles = [MR.ArrayTextVecHandle('direct_mode%d'%i) for i in mode_nums]
adjoint_mode_handles = [MR.ArrayTextVecHandle('adjoint_mode%d'%i) for i in mode_nums]
my_BPOD.compute_direct_modes(mode_nums, direct_mode_handles )
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_mode_handles)
