import numpy as N
import modred as MR

num_vecs = 30
nx = 100
ny = 45
x_grid = 1. - N.cos(N.linspace(0, N.pi, nx))
y_grid = N.linspace(0, 2., ny)**2

# We use arbitrary fake data as a placeholder
# Normally one doesn't need to do this because you have real data.
Y, X = N.meshgrid(y_grid, x_grid)
vecs = [N.sin(X*0.1*i) + N.cos(Y*0.15*i) for i in range(num_vecs)]

weighted_ip = MR.InnerProductTrapz(x_grid, y_grid)
pod = MR.POD(weighted_ip)
sing_vecs, sing_vals = pod.compute_decomp_in_memory(vecs)
num_modes = 10
modes = pod.compute_modes_in_memory(range(num_modes))
