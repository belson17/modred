import numpy as N
import modred as MR

nx = 100
ny = 200
num_inputs = 2
num_outputs = 4
num_modes = 40

# Create random modes and action on modes. Typically this data already exists and
# this section is unnecesary.
basis_vecs = [MR.VecHandlePickle('dir_mode_%02d.pkl'%i) 
    for i in range(num_modes)]
adjoint_basis_vecs = [MR.VecHandlePickle('adj_mode_%02d.pkl'%i) 
    for i in range(num_modes)]
A_on_basis_vecs = [MR.VecHandlePickle('A_on_dir_mode_%02d.pkl'%i) 
    for i in range(num_modes)]
B_on_bases = [MR.VecHandlePickle('B_on_basis_%02d.pkl'%i) 
    for i in range(num_inputs)]
C_on_basis_vecs = [N.sin(N.linspace(0, 0.1*i, num_outputs)) for i in range(num_modes)]
parallel = MR.parallel_default_instance
if parallel.is_rank_zero():
    for handle in (basis_vecs + adjoint_basis_vecs + A_on_basis_vecs + 
        B_on_bases):
        handle.put(N.random.random((nx, ny)))
parallel.barrier()

inner_product = N.vdot
LTI_proj = MR.LTIGalerkinProjectionHandles(inner_product, basis_vecs, 
    adjoint_basis_vec_handles=adjoint_basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A_on_basis_vecs, B_on_bases, C_on_basis_vecs)
LTI_proj.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')

