from future.builtins import range
import numpy as np
import modred as mr

num_vecs = 30
# Arbitrary data
vecs = np.random.random((100, num_vecs))
num_modes = 5
modes, eig_vals = mr.compute_POD_matrices_snaps_method(
    vecs, list(range(num_modes)))
