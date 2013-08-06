import numpy as N
import modred as MR

nx = 100
num_inputs = 2
num_outputs = 3
num_basis_vecs = 10

# Create random system and modes
A = N.mat(N.random.random((nx, nx)))
B = N.mat(N.random.random((nx, num_inputs)))
C = N.mat(N.random.random((num_outputs, nx)))

basis_vecs = N.mat(N.random.random((nx, num_basis_vecs)))

LTI_proj = MR.LTIGalerkinProjectionMatrices(basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A * basis_vecs, B, C * basis_vecs)
LTI_proj.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')


