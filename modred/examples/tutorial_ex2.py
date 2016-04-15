from future.builtins import range
import numpy as np
import modred as mr

num_vecs = 100
nx = 100

# np.n-uniform grid and corresponding inner product weights.
x_grid = 1. - np.cos(np.linspace(0, np.pi, nx))
x_diff = np.diff(x_grid)
weights = 0.5 * np.append(np.append(
    x_diff[0], x_diff[:-1] + x_diff[1:]), x_diff[-1])

# Arbitrary data
vecs = np.random.random((nx, num_vecs))
num_modes = 10
modes, sing_vals = mr.compute_POD_matrices_direct_method(
    vecs, list(range(num_modes)), inner_product_weights=weights)
