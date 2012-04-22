import modred as MR
import numpy as N
import cPickle

class CustomVector(MR.Vector):
    def __init__(self, path=None):
        if path is not None:
            self.load(path)
        self.my_trapz_IP = None
    def load(self, path):
        file_id = open(path, 'rb')
        self.x, self.y, self.z = cPickle.load(file_id)
        self.data_array = cPickle.load(file_id)
        file_id.close()
    def save(self, path):
        file_id = open(path, 'wb')
        cPickle.dump((self.x, self.y, self.z), file_id)
        cPickle.dump(self.data_array, file_id)
        file_id.close()
    def copy(self):
        """Returns a copy of self"""
        from copy import deepcopy
        return deepcopy(self)
    def __add__(self, other):
        """Return a new object that is the sum of self and other"""
        sum_vec = self.copy()
        sum_vec.data_array = self.data_array + other.data_array
        return sum_vec
    def __mul__(self, scalar):
        """Return a new object that is ``self * scalar`` """
        mult_vec = self.copy()
        mult_vec.data_array = mult_vec.data_array*scalar
    def inner_product(self, other):
        if self.my_trapz_IP is None:
            self.my_trapz_IP = MR.InnerProductTrapz(self.x, self.y, self.z)
        return self.my_trapz_IP(self.data_array, other.data_array)

class CustomVecHandle(MR.VecHandle):
    def __init__(self, vec_path, base_handle=None, scale=None):
        MR.VecHandle.__init__(self, base_handle, scale)
        self.vec_path = vec_path
    def _get(self):
        return CustomVector(self.vec_path)
    def _put(self, vec):
        vec.save(self.vec_path)
def inner_product(v1, v2):
    return v1.inner_product(v2)
    
# Set vec handles (assuming existing saved data)
direct_snap_handles = [CustomVecHandle(vec_path='direct_snap%d.pkl'%i,
    scale=N.pi) for i in range(10)]
adjoint_snap_handles = [CustomVecHandle(vec_path='adjoint_snap%d.pkl'%i,
    scale=N.pi) for i in range(10)]

# generate some random data
num_elements = 50
for snap in direct_snap_handles + adjoint_snap_handles:
    # seems like the following would be a good way to build a CustomVector
    # snap.put(CustomVector(N.random.random(num_elements)))

my_BPOD = MR.BPOD(inner_product=inner_product)
sing_vecs, sing_vals = my_BPOD.compute_decomp(direct_snap_handles, 
    adjoint_snap_handles)
num_modes = 5
mode_nums = range(num_modes)  
direct_mode_handles = [CustomVecHandle('direct_mode%d.pkl'%i) 
    for i in mode_nums] 
adjoint_mode_handles = [CustomVecHandle('adjoint_mode%d.pkl'%i) 
    for i in mode_nums]

my_BPOD.compute_direct_modes(mode_nums, direct_mode_handles)
my_BPOD.compute_adjoint_modes(mode_nums, adjoint_mode_handles)
