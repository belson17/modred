from future.builtins import range

import numpy as np

import modred as mr


# Create random data
num_vecs = 30
vecs = np.random.random((100, num_vecs))

# Compute POD
num_modes = 5
POD_res = mr.compute_POD_arrays_snaps_method(
    vecs, list(range(num_modes)))
modes = POD_res.modes
eigvals = POD_res.eigvals
