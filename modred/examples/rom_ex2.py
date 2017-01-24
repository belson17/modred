from future.builtins import range
import os

import numpy as np

from modred import parallel
import modred as mr


# Create directory for output files
out_dir = 'rom_ex2_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Create random modes and action on modes. Typically this data already exists
# and this section is unnecesary.
nx = 100
ny = 200
num_inputs = 2
num_outputs = 4
num_modes = 40
basis_vecs = [
    mr.VecHandlePickle('%s/dir_mode_%02d.pkl' % (out_dir, i))
    for i in range(num_modes)]
adjoint_basis_vecs = [
    mr.VecHandlePickle('%s/adj_mode_%02d.pkl' % (out_dir, i))
    for i in range(num_modes)]
A_on_basis_vecs = [
    mr.VecHandlePickle('%s/A_on_dir_mode_%02d.pkl' % (out_dir, i))
    for i in range(num_modes)]
B_on_bases = [
    mr.VecHandlePickle('%s/B_on_basis_%02d.pkl' % (out_dir, i))
    for i in range(num_inputs)]
C_on_basis_vecs = [
    np.sin(np.linspace(0, 0.1 * i, num_outputs)) for i in range(num_modes)]
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
LTI_proj.put_model(
    '%s/A_reduced.txt' % out_dir, '%s/B_reduced.txt' % out_dir,
    '%s/C_reduced.txt' % out_dir)
