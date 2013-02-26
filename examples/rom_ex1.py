import numpy as N
import modred as MR

nx = 100
num_inputs = 2
num_outputs = 3
num_basis_vecs = 10

# Create random system and random modes
A = N.arange(nx**2).reshape((nx, nx))
B = N.arange(nx*num_inputs).reshape((nx, num_inputs))
C = N.arange(nx*num_outputs).reshape((num_outputs, nx))
basis_vec_array = N.arange(nx*num_basis_vecs).reshape((nx, num_basis_vecs))

LTI_proj = MR.LTIGalerkinProjectionArrays(basis_vec_array)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model(
    A.dot(basis_vec_array), B, C.dot(basis_vec_array))
LTI_proj.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')


