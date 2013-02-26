import numpy as N
import modred as MR

num_vecs = 30
# Arbitrary data
vec_array = N.random.random((100, num_vecs))

my_POD = MR.PODArrays()
sing_vecs, sing_vals = my_POD.compute_decomp(vec_array)
num_modes = 5
modes = my_POD.compute_modes(range(num_modes))
