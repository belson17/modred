import os

import numpy as np

import modred as mr


# Create directory for output files
out_dir = 'rom_ex1_out'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# Create random LTI system
nx = 100
num_inputs = 2
num_outputs = 3
num_basis_vecs = 10
A = np.random.random((nx, nx))
B = np.random.random((nx, num_inputs))
C = np.random.random((num_outputs, nx))

# Create random modes
basis_vecs = np.random.random((nx, num_basis_vecs))

# Perform Galerkin projection and save data
LTI_proj = mr.LTIGalerkinProjectionArrays(basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A.dot(basis_vecs), B, C.dot(basis_vecs))
LTI_proj.put_model(
    '%s/A_reduced.txt' % out_dir, '%s/B_reduced.txt' % out_dir,
    '%s/C_reduced.txt' % out_dir)
