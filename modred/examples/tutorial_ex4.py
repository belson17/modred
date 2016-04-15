from future.builtins import range
import modred as mr
import numpy as np

num_vecs = 100
# np.n-uniform grid and corresponding inner product weights.
nx = 80
ny = 100
x_grid = 1. - np.cos(np.linspace(0, np.pi, nx))
y_grid = np.linspace(0, 1., ny)**2
Y, X = np.meshgrid(y_grid, x_grid)

snapshots = [mr.VecHandlePickle('vec%d.pkl'%i) for i in range(num_vecs)]
parallel = mr.parallel_default_instance
if parallel.is_rank_zero():
    for i,snap in enumerate(snapshots):
        snap.put(np.sin(X * i) + np.cos(Y * i))
parallel.barrier()

weighted_IP = mr.InnerProductTrapz(x_grid, y_grid)

# Calculate DMD modes and save them to pickle files.
my_DMD = mr.DMDHandles(weighted_IP)
my_DMD.compute_decomp(snapshots)
my_DMD.put_decomp(
    'eigvals.txt', 'R_low_order_eigvecs.txt', 'L_low_order_eigvecs.txt',
    'correlation_mat_eigvals.txt', 'correlation_mat_eigvecs.txt')
mode_indices = [1, 4, 5, 0, 10]
modes = [mr.VecHandlePickle('mode%d.pkl'%i) for i in mode_indices]
my_DMD.compute_exact_modes(mode_indices, modes)

