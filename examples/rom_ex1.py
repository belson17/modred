import numpy as N
import modred as MR

nx = 100
num_inputs = 2
num_outputs = 3
num_modes = 10

# Create random system and random modes
A_op = MR.MatrixOperator(N.arange(nx**2).reshape((nx, nx)))
B_op = MR.MatrixOperator(N.arange(nx*num_inputs).reshape((nx, num_inputs)))
C_op = MR.MatrixOperator(N.arange(nx*num_outputs).reshape((num_outputs, nx)))
basis_vecs = [N.sin(N.linspace(0, i+1, nx)) for i in range(num_modes)]

inner_product = N.vdot
LTI_proj = MR.LTIGalerkinProjection(inner_product, basis_vecs)
A_reduced, B_reduced, C_reduced = LTI_proj.compute_model_in_memory(
    A_op, B_op, C_op, num_inputs)
LTI_proj.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')


