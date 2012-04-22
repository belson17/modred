import numpy as N
import modred as MR

def main(verbose=True):
    num_elements = 2000
    num_vecs = 100
    
    # Sample times, used for quadrature weights in POD
    quad_weights = N.logspace(1., 3., num=num_vecs)
    
    vec_handles = [MR.PickleVecHandle('vec%d.pkl'%i, scale=quad_weights[i])
        for i in range(num_vecs)]
    
    # Save fake data. Typically the data already exists from a previous
    # simulation or experiment.
    parallel = MR.parallel.default_instance
    if parallel.is_rank_zero():
        for i, handle in enumerate(vec_handles):
            handle.put(N.random.random(num_elements))
    parallel.barrier()
    
    my_POD = MR.POD(inner_product=N.vdot, verbose=verbose)  
    my_POD.compute_decomp(vec_handles)
    my_POD.put_decomp('sing_vecs.txt', 'sing_vals.txt')
    my_POD.put_correlation_mat('correlation_mat.txt')
    mode_nums = [1, 4, 5, 2, 10]
    mode_handles = [MR.ArrayTextVecHandle('mode%d.txt'%i) for i in mode_nums]
    my_POD.compute_modes(mode_nums, mode_handles)
    
    # Check that modes are orthonormal
    my_vec_ops = MR.VecOperations(inner_product=N.vdot, verbose=verbose)
    IP_mat = my_vec_ops.compute_symmetric_inner_product_mat(mode_handles)
    if not N.allclose(IP_mat, N.eye(len(mode_nums))) and verbose:
        print 'Warning: modes are not orthonormal'
        
if __name__ == '__main__':
    main()
