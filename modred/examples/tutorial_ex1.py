from future.builtins import range

import numpy as np

import modred as mr


# Arbitrary data
num_vecs = 30
vecs = np.random.random((100, num_vecs))

num_modes = 5
modes, eigvals = mr.compute_POD_matrices_snaps_method(
    vecs, list(range(num_modes)))
