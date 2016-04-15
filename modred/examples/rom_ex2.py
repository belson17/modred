from future.builtins import range
import numpy as np
import modred as mr

nx = 100
ny = 200
num_inputs = 2
num_outputs = 4
num_modes = 40

# Create random modes and action on modes. Typically this data already exists and
# this section is unnecesary.
basis_vecs = [
    mr.VecHandlePickle('dir_mode_%02d.pkl'%i) 
    for i in range(num_modes)]
adjoint_basis_vecs = [
    mr.VecHandlePickle('adj_mode_%02d.pkl'%i) 
    for i in range(num_modes)]
A_on_basis_vecs = [
    mr.VecHandlePickle('A_on_dir_mode_%02d.pkl'%i) 
    for i in range(num_modes)]
B_on_bases = [
    mr.VecHandlePickle('B_on_basis_%02d.pkl'%i) 
    for i in range(num_inputs)]
C_on_basis_vecs = [
    np.sin(np.linspace(0, 0.1*i, num_outputs)) for i in range(num_modes)]
parallel = mr.parallel_default_instance
if parallel.is_rank_zero():
    for handle in (
        basis_vecs + adjoint_basis_vecs + A_on_basis_vecs + B_on_bases):
        handle.put(np.random.random((nx, ny)))
parallel.barrier()

inner_product = np.vdot
LTI_proj = mr.LTIGalerkinProjectionHandles(
    inner_product, basis_vecs, adjoint_basis_vec_handles=adjoint_basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A_on_basis_vecs, B_on_bases, C_on_basis_vecs)
LTI_proj.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')

