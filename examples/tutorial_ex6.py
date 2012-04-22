import cPickle
import numpy as N
import modred as MR
from copy import deepcopy
parallel = MR.parallel.default_instance

class CustomVector(MR.Vector):
    def __init__(self, grids, data_array):
        self.grids = grids
        self.data_array = data_array
        self.weighted_ip = MR.InnerProductTrapz(*self.grids)

    def __add__(self, other):
        """Return a new object that is the sum of self and other"""
        sum_vec = deepcopy(self)
        sum_vec.data_array = self.data_array + other.data_array
        return sum_vec

    def __mul__(self, scalar):
        """Return a new object that is ``self * scalar`` """
        mult_vec = deepcopy(self)
        mult_vec.data_array = mult_vec.data_array * scalar
        return mult_vec

    def inner_product(self, other):
        return self.weighted_ip(self.data_array, other.data_array)

class CustomVecHandle(MR.VecHandle):
    def __init__(self, vec_path, base_handle=None, scale=None):
        MR.VecHandle.__init__(self, base_handle, scale)
        self.vec_path = vec_path

    def _get(self):
        file_id = open(self.vec_path, 'rb')
        grids = cPickle.load(file_id)
        data_array = cPickle.load(file_id)
        file_id.close()
        return CustomVector(grids, data_array)

    def _put(self, vec):
        file_id = open(self.vec_path, 'wb')
        cPickle.dump(vec.grids, file_id)
        cPickle.dump(vec.data_array, file_id)
        file_id.close()
        
def inner_product(v1, v2):
    return v1.inner_product(v2)

def main():
    # Set vec handles (assuming existing saved data)
    direct_snapshots = [CustomVecHandle('direct_snap%d.pkl'%i,
        scale=N.pi) for i in range(10)]
    adjoint_snapshots = [CustomVecHandle('adjoint_snap%d.pkl'%i,
        scale=N.pi) for i in range(10)]
        
    # generate random data
    nx = 50
    ny = 30
    nz = 20
    x = N.linspace(0, 1, nx)
    y = N.logspace(1, 2, ny)
    z = N.linspace(0, 1, nz)**2
    
    if parallel.is_rank_zero():
        for snap in direct_snapshots + adjoint_snapshots:
            snap.put(CustomVector([x, y, z], 
                                  N.random.random((nx, ny, nz))))
    parallel.barrier()
    
    bpod = MR.BPOD(inner_product=inner_product)
    bpod.sanity_check(direct_snapshots[0])
    L_sing_vecs, sing_vals, R_sing_vecs = \
        bpod.compute_decomp(direct_snapshots, adjoint_snapshots)
    
    # Model error less than ~10%
    sing_vals_norm = sing_vals / N.sum(sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > 0.9)[0][0] + 1
    mode_nums = range(num_modes)
    
    direct_modes = [CustomVecHandle('direct_mode%d.pkl'%i) 
                    for i in mode_nums] 
    adjoint_modes = [CustomVecHandle('adjoint_mode%d.pkl'%i) 
                     for i in mode_nums]
    
    bpod.compute_direct_modes(mode_nums, direct_modes)
    bpod.compute_adjoint_modes(mode_nums, adjoint_modes)

if __name__ == "__main__":
    main()

