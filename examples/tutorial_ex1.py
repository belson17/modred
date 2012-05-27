import numpy as N
import modred as MR

num_vecs = 30
# We use arbitrary fake data as a placeholder
x = N.linspace(0, N.pi, 100)
vecs = [N.sin(x*0.1*i) for i in range(num_vecs)]

my_POD = MR.POD(N.vdot)
sing_vecs, sing_vals = my_POD.compute_decomp_in_memory(vecs)
num_modes = 5
modes = my_POD.compute_modes_in_memory(range(num_modes))
