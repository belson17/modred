import numpy as N
import modred as MR

num_vecs = 30
# Arbitrary data
vecs = N.random.random((100, num_vecs))
num_modes = 5
modes, eig_vals = MR.compute_POD_matrices_snaps_method(vecs, range(num_modes))
