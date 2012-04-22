import modred as MR
import numpy as N
parallel = MR.parallel.default_instance

def main(verbose=True):
    num_elements = 2000  
    num_vecs = 100
    base_vec_handle = MR.PickleVecHandle('base_vec.pkl')
    vec_handles = [MR.PickleVecHandle('vec%d.pkl'%i, base_vec_handle=base_vec_handle)
        for i in range(num_vecs)]
     
    # Save fake data. Typically the data already exists from a previous
    # simulation or experiment.
    if parallel.is_rank_zero():
        # A base vector to be subtracted off from each vector as it is loaded.
        base_vec = N.random.random(num_elements)
        for handle in vec_handles + [base_vec_handle]:
            handle.put(N.random.random(num_elements))
    parallel.barrier()
    
    my_DMD = MR.DMD(inner_product=N.vdot, verbose=verbose)
    my_DMD.compute_decomp(vec_handles)
    my_DMD.put_decomp('ritz_vals.txt', 'mode_norms.txt', 'build_coeffs.txt')
    mode_nums = [1, 4, 5, 2, 10]
    mode_handles = [MR.PickleVecHandle('mode%d.pkl'%i) for i in mode_nums]

if __name__ == '__main__':
    main()
