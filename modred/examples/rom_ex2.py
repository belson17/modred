import os

import numpy as np

from modred import parallel
import modred as mr


# Create directory for output files
out_dir = 'rom_ex2_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Create handles for modes
num_modes = 10
basis_vecs = [
    mr.VecHandlePickle('%s/dir_mode_%02d.pkl' % (out_dir, i))
    for i in mr.range(num_modes)]
adjoint_basis_vecs = [
    mr.VecHandlePickle('%s/adj_mode_%02d.pkl' % (out_dir, i))
    for i in mr.range(num_modes)]

# Define system dimensions and create handles for action on modes
num_inputs = 2
num_outputs = 4
A_on_basis_vecs = [
    mr.VecHandlePickle('%s/A_on_dir_mode_%02d.pkl' % (out_dir, i))
    for i in mr.range(num_modes)]
B_on_bases = [
    mr.VecHandlePickle('%s/B_on_basis_%02d.pkl' % (out_dir, i))
    for i in mr.range(num_inputs)]
C_on_basis_vecs = [
    np.sin(np.linspace(0, 0.1 * i, num_outputs)) for i in mr.range(num_modes)]

# Create random modes and action on modes. Typically this data already exists
# and this section is unnecesary.
nx = 100
ny = 200
if parallel.is_rank_zero():
    for handle in (
        basis_vecs + adjoint_basis_vecs + A_on_basis_vecs + B_on_bases):
        handle.put(np.random.random((nx, ny)))
parallel.barrier()

# Perform Galerkin projection and save data to disk
inner_product = np.vdot
LTI_proj = mr.LTIGalerkinProjectionHandles(
    inner_product, basis_vecs, adjoint_basis_vec_handles=adjoint_basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A_on_basis_vecs, B_on_bases, C_on_basis_vecs)
LTI_proj.put_model(
    '%s/A_reduced.txt' % out_dir, '%s/B_reduced.txt' % out_dir,
    '%s/C_reduced.txt' % out_dir)
