import modred as MR
import numpy as N

num_vecs = 100
# Non-uniform grid and corresponding inner product weights.
nx = 100
ny = 100
x_grid = 1. - N.cos(N.linspace(0, N.pi, nx))
y_grid = N.linspace(0, 1., ny)**2
Y, X = N.meshgrid(y_grid, x_grid)

snapshots = [MR.VecHandlePickle('vec%d.pkl'%i) for i in range(num_vecs)]
parallel = MR.parallel_default_instance
if parallel.is_rank_zero():
    for i,snap in enumerate(snapshots):
        snap.put(N.sin(X*i) + N.cos(Y*i))
parallel.barrier()

weighted_IP = MR.InnerProductTrapz(x_grid, y_grid)

# Calculate DMD modes and save them to pickle files.
my_DMD = MR.DMDHandles(weighted_IP)
my_DMD.compute_decomp(snapshots)
my_DMD.put_decomp('ritz_vals.txt', 'mode_norms.txt', 'build_coeffs.txt')
mode_indices = [1, 4, 5, 0, 10]
modes = [MR.VecHandlePickle('mode%d.pkl'%i) for i in mode_indices]
my_DMD.compute_modes(mode_indices, modes)

