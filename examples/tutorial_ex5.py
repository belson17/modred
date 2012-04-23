import numpy as N
import modred as MR

num_elements = 2000
num_vecs = 100

# Sample times, used for quadrature weights in POD
quad_weights = N.logspace(1., 3., num=num_vecs)

# Define the snapshots to be used
snapshots = [MR.PickleVecHandle('vec%d.pkl' % i, scale=quad_weights[i])
             for i in range(num_vecs)]

# Save fake data. Typically the data already exists from a previous
# simulation or experiment.
parallel = MR.parallel_default_instance
if parallel.is_rank_zero():
    for i, snap in enumerate(snapshots):
        snap.put(N.random.random(num_elements))
parallel.barrier()

# Compute POD modes
pod = MR.POD(N.vdot)
pod.compute_decomp(snapshots)
pod.put_decomp('sing_vecs.txt', 'sing_vals.txt')
pod.put_correlation_mat('correlation_mat.txt')
mode_nums = [1, 4, 5, 2, 10]
modes = [MR.ArrayTextVecHandle('mode%d.txt'%i) for i in mode_nums]
pod.compute_modes(mode_nums, modes)

# Check that modes are orthonormal
vec_ops = MR.VectorSpace(inner_product=N.vdot)
IP_mat = vec_ops.compute_symmetric_inner_product_mat(modes)
if not N.allclose(IP_mat, N.eye(len(mode_nums))):
    print 'Warning: modes are not orthonormal'
