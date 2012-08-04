import modred as MR
import numpy as N

# Define the snapshots to be used
num_vecs = 100
base_vec_handle = MR.PickleVecHandle('base_vec.pkl')
snapshots = [MR.PickleVecHandle('vec%d.pkl'%i, base_vec_handle=base_vec_handle)
             for i in range(num_vecs)]
 
# Save fake data. Typically the data already exists from a previous
# simulation or experiment.
num_elements = 2000  
parallel = MR.parallel_default_instance
if parallel.is_rank_zero():
    for snap in snapshots + [base_vec_handle]:
        snap.put(N.random.random(num_elements))
parallel.barrier()

# Calculate DMD modes, saving to Pickle files
my_DMD = MR.DMD(N.vdot)
my_DMD.compute_decomp(snapshots)
my_DMD.put_decomp('ritz_vals.txt', 'mode_norms.txt', 'build_coeffs.txt')
mode_nums = [1, 4, 5, 2, 10]
modes = [MR.PickleVecHandle('mode%d.pkl'%i) for i in mode_nums]
my_DMD.compute_modes(mode_nums, modes)
