import numpy as np

import modred as mr


# Create random data
num_vecs = 100
nx = 100
vecs = np.random.random((nx, num_vecs))

# Define non-uniform grid and corresponding inner product weights
x_grid = 1. - np.cos(np.linspace(0, np.pi, nx))
x_diff = np.diff(x_grid)
weights = 0.5 * np.append(np.append(
    x_diff[0], x_diff[:-1] + x_diff[1:]), x_diff[-1])

# Compute POD
num_modes = 10
POD_res = mr.compute_POD_arrays_direct_method(
    vecs, list(mr.range(num_modes)), inner_product_weights=weights)
modes = POD_res.modes
eigvals = POD_res.eigvals
