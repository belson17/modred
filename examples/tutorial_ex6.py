import cPickle
import numpy as N
import modred as MR
from copy import deepcopy
parallel = MR.parallel.default_instance

class CustomVector(MR.Vector):
    def __init__(self, grids=None, data_array=None):
        self.grids = grids
        self.data_array = data_array
        self.my_trapz_IP = None
    def __add__(self, other):
        """Return a new object that is the sum of self and other"""
        sum_vec = deepcopy(self)
        sum_vec.data_array = self.data_array + other.data_array
        return sum_vec
    def __mul__(self, scalar):
        """Return a new object that is ``self * scalar`` """
        mult_vec = deepcopy(self)
        mult_vec.data_array = mult_vec.data_array*scalar
        return mult_vec
    def inner_product(self, other):
        if self.my_trapz_IP is None:
            if self.grids is not None:
                self.my_trapz_IP = MR.InnerProductTrapz(*self.grids)
            else:
                raise MR.UndefinedError('grids are not specified')
        return self.my_trapz_IP(self.data_array, other.data_array)

class CustomVecHandle(MR.VecHandle):
    def __init__(self, vec_path, base_handle=None, scale=None):
        MR.VecHandle.__init__(self, base_handle, scale)
        self.vec_path = vec_path
    def _get(self):
        file_id = open(self.vec_path, 'rb')
        grids = cPickle.load(file_id)
        data_array = cPickle.load(file_id)
        file_id.close()
        v = CustomVector(grids=grids, data_array=data_array)
        return v
    def _put(self, vec):
        file_id = open(self.vec_path, 'wb')
        cPickle.dump(vec.grids, file_id)
        cPickle.dump(vec.data_array, file_id)
        file_id.close()
        
def inner_product(v1, v2):
    return v1.inner_product(v2)

def main(verbose=True):
    # Set vec handles (assuming existing saved data)
    direct_snap_handles = [CustomVecHandle('direct_snap%d.pkl'%i,
        scale=N.pi) for i in range(10)]
    adjoint_snap_handles = [CustomVecHandle('adjoint_snap%d.pkl'%i,
        scale=N.pi) for i in range(10)]
        
    # generate random data
    nx = 50
    ny = 30
    nz = 20
    x = N.linspace(0, 1, nx)
    y = N.logspace(1, 2, ny)
    z = N.linspace(0, 1, nz)**2
    
    if parallel.is_rank_zero():
        for handle in direct_snap_handles + adjoint_snap_handles:
            handle.put(CustomVector(grids=[x, y, z], 
                data_array=N.random.random((nx, ny, nz))))
    parallel.barrier()
    
    my_BPOD = MR.BPOD(inner_product=inner_product, verbose=verbose)
    my_BPOD.sanity_check(direct_snap_handles[0])
    L_sing_vecs, sing_vals, R_sing_vecs = \
        my_BPOD.compute_decomp(direct_snap_handles, adjoint_snap_handles)
    
    # Model error less than ~10%
    sing_vals_norm = sing_vals/N.sum(sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > 0.9)[0][0] + 1
    mode_nums = range(num_modes)
    
    direct_mode_handles = [CustomVecHandle('direct_mode%d.pkl'%i) 
        for i in mode_nums] 
    adjoint_mode_handles = [CustomVecHandle('adjoint_mode%d.pkl'%i) 
        for i in mode_nums]
    
    my_BPOD.compute_direct_modes(mode_nums, direct_mode_handles)
    my_BPOD.compute_adjoint_modes(mode_nums, adjoint_mode_handles)
