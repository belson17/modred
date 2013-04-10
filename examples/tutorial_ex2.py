import numpy as N
import modred as MR

num_vecs = 100
nx = 100

# Non-uniform grid and corresponding inner product weights.
x_grid = 1. - N.cos(N.linspace(0, N.pi, nx))
x_diff = N.diff(x_grid)
weights = 0.5*N.append(N.append(x_diff[0], x_diff[:-1]+x_diff[1:]), x_diff[-1])

# Arbitrary data
vecs = N.random.random((nx, num_vecs))
num_modes = 10
modes, sing_vals = MR.compute_POD_matrices_direct_method(vecs, range(num_modes),
    inner_product_weights=weights)
