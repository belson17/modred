import os

import numpy as np

import modred as mr


# Create directory for output files
out_dir = 'rom_ex1_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Create random system and modes
nx = 100
num_inputs = 2
num_outputs = 3
num_basis_vecs = 10
A = np.mat(np.random.random((nx, nx)))
B = np.mat(np.random.random((nx, num_inputs)))
C = np.mat(np.random.random((num_outputs, nx)))

basis_vecs = np.mat(np.random.random((nx, num_basis_vecs)))

LTI_proj = mr.LTIGalerkinProjectionMatrices(basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A * basis_vecs, B, C * basis_vecs)
LTI_proj.put_model(
    '%s/A_reduced.txt' % out_dir, '%s/B_reduced.txt' % out_dir,
    '%s/C_reduced.txt' % out_dir)
