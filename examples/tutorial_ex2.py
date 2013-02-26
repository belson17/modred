import numpy as N
import modred as MR

num_vecs = 30
nx = 100

# Non-uniform grid and corresponding inner product weights.
x_grid = 1. - N.cos(N.linspace(0, N.pi, nx))
x_spacing = N.diff(x_grid)
weights = N.append(N.append(x_spacing[0], 0.5*(x_spacing[:-1] + x_spacing[1:])),
    x_spacing[-1])

# Arbitrary data
vec_array = N.random.random((nx, num_vecs))

my_POD = MR.PODArrays(inner_product_weights=weights)
sing_vecs, sing_vals = my_POD.compute_decomp(vec_array)
num_modes = 10
modes = my_POD.compute_modes(range(num_modes))
