from future import standard_library
standard_library.install_hooks()
import pickle
from copy import deepcopy

import numpy as np

import modred as mr


class CustomVector(mr.Vector):
    def __init__(self, grids, data_array):
        self.grids = grids
        self.data_array = data_array
        self.weighted_ip = mr.InnerProductTrapz(*self.grids)

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

class CustomVecHandle(mr.VecHandle):
    def __init__(self, vec_path, base_handle=None, scale=None):
        mr.VecHandle.__init__(self, base_handle, scale)
        self.vec_path = vec_path

    def _get(self):
        file_id = open(self.vec_path, 'rb')
        grids = pickle.load(file_id)
        data_array = pickle.load(file_id)
        file_id.close()
        return CustomVector(grids, data_array)

    def _put(self, vec):
        file_id = open(self.vec_path, 'wb')
        pickle.dump(vec.grids, file_id)
        pickle.dump(vec.data_array, file_id)
        file_id.close()
        
def inner_product(v1, v2):
    return v1.inner_product(v2)
