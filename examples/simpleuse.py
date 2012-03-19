
"""A collection of classes that make it even simpler to use modred.

Everything is done in memory, and methods return modes and matrices.
"""

import numpy as N

import sys, os
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(__file__), '..'),
    'src'))
import pod
import bpod
import dmd
import util
 
def get_vec(array_and_index):
    array = array_and_index[0]
    index = array_and_index[1]
    return array[:,index]

def put_vec(vec, array_and_index):
    array = array_and_index[0]
    index = array_and_index[1]
    array[:,index] = vec


class SimpleUsePOD(object):
    """Simple class which keeps all data in memory and uses the POD class.
    
    The purpose of this class is to get quick results for simple data.
    This class wraps the POD class so that modes and matrices are returned. 
    
    Usage::
      
      # Put your vecs into columns of an array, "vecs"
      my_POD = SimpleUsePOD(vecs=vecs)
      num_modes = 20
      modes = my_POD.compute_modes(num_modes)
      # To look at singulaar values:
      sing_vals = my_POD.sing_vals
    
    More fine-grain control is available through the other methods.
    
    For the more curious user, this module and class supply ``get_vec`` and 
    ``put_vec`` functions that get/put vecs from/into columns of numpy arrays.
    By default, the default inner product is ``(fiedl1*vec2.conj()).sum()``.
    """
    def __init__(self, inner_product=util.inner_product, verbose=True, 
        vecs=None):
        self.inner_product = inner_product
        self.verbose = verbose
        self.POD = pod.POD(inner_product=inner_product, get_vec=get_vec,
            put_vec=put_vec, max_vecs_per_node=1000, verbose=verbose)
        self._called_compute_decomp = False
        if vecs is not None:
            self.set_vecs(vecs)
            
        
    def set_vecs(self, vecs):
        """Sets the vecs used in POD.
        
        Args:
            vecs: a 2D array with columns of vecs, ``X'' in literature.
        """ 
        self.vecs = vecs
        self.num_vecs = vecs.shape[1]

    
    def compute_decomp(self, vecs=None):
        """Computes correlation matrix and its SVD.
        
        Computes correlation matrix with (i,j) entry <vec_i, vec_j>,
        then U E V^* = correlation_mat.
        
        Kwargs:
            vecs: a 2D array with columns of vecs, ``X'' in literature.
        
        Returns:
            singular vectors (U, which equals V)
            
            singular values (E)             
        """
        self._called_compute_decomp = True
        
        if vecs is not None:
            self.vecs = vecs
        
        self.vec_sources = [(self.vecs, index) 
            for index in xrange(0, self.num_vecs)]
        self.POD.compute_decomp(self.vec_sources)
        self.correlation_mat = self.POD.correlation_mat
        self.sing_vals = self.POD.sing_vals
        return self.POD.sing_vecs, self.POD.sing_vals
    
    
    def compute_modes(self, num_modes):
        """Computes ``num_modes`` modes
        
        Args:
            num_modes: integer number of modes to compute
        
        Returns:
            modes: 2D array with modes as columns
        """
        if not self._called_compute_decomp:
            self.compute_decomp()

        self.mode_nums = range(1,num_modes+1)
        self.modes = N.zeros((self.vecs.shape[0], num_modes))
        mode_dests = [(self.modes, mode_index) for mode_index in range(num_modes)]
        self.POD.compute_modes(self.mode_nums, mode_dests)
        return self.modes




class SimpleUseBPOD(object):
    """Simple class which keeps all data in memory and uses the BPOD class.
    
    The purpose of this class is to get quick results for simple data.
    This class wraps the BPOD class so that modes and matrices are returned. 
    
    Usage::

      # Put your vecs into columns of arrays direct_vecs and adjoint_vecs
      my_BPOD = SimpleUseBPOD(direct_vecs=direct_vecs, adjoint_vecs=adjoint_vecs)
      num_modes = 20
      direct_modes = my_BPOD.compute_direct_modes(num_modes)
      adjoint_modes = my_BPOD.compute_adjoing_modes(num_modes)
      sing_vals = my_BPOD.sing_vals
    
    More fine-grain control is available through the other methods.
    
    For the more curious user, this module and class supply ``get_vec`` and 
    ``put_vec`` functions that get/put vecs from/into columns of numpy arrays.
    By default, the default inner product is ``(fiedl1*vec2.conj()).sum()``.
    """    
    def __init__(self, inner_product=util.inner_product, verbose=True,
        direct_vecs=None, adjoint_vecs=None):
        self.inner_product = inner_product
        self.BPOD = bpod.BPOD(inner_product=inner_product, get_vec=get_vec,
            put_vec=put_vec, max_vecs_per_node=1000, verbose=verbose)
        self._called_compute_decomp = False
        if direct_vecs is not None:
            self.set_direct_vecs(direct_vecs)
        if adjoint_vecs is not None:
            self.set_adjoint_vecs(adjoint_vecs)
            
    def set_direct_vecs(self, direct_vecs):
        """Sets the direct vecs.
        
        Args:
            direct_vecs: a 2D array with columns of vecs, ``X'' in literature.
        """ 
        self.direct_vecs = direct_vecs
        self.num_direct_vecs = self.direct_vecs.shape[1]

    def set_adjoint_vecs(self, adjoint_vecs):
        """Sets the adjoint vecs.
        
        Args:
            adjoint_vecs: a 2D array with columns of vecs, ``Y'' in literature.
        """ 
        self.adjoint_vecs = adjoint_vecs
        self.num_adjoint_vecs = self.adjoint_vecs.shape[1]

    
    def compute_decomp(self, direct_vecs=None, adjoint_vecs=None):
        """Computes Hankel matrix and its SVD.
        
        Computes Hankel matrix with (i,j) entry <adjoint_vec_i, direct_vec_j>,
        then U E V^* = hankel_mat.
        
        Kwargs:
           direct_vecs: a 2D array with columns of vecs, ``X'' in literature.
           
           adjoint_vecs: a 2D array with columns of vecs, ``Y'' in literature.
        
        Returns:
            left singular vectors (U)
            
            singular values (E) 
            
            right singular vectors (V)
        """
        self._called_compute_decomp = True
        if direct_vecs is not None:
            self.direct_vecs = direct_vecs
        
        if adjoint_vecs is not None:
            self.adjoint_vecs = adjoint_vecs
        
        self.direct_vec_sources = [(self.direct_vecs, index) 
            for index in xrange(0, self.num_direct_vecs)]
        self.adjoint_vec_sources = [(self.adjoint_vecs, index) 
            for index in xrange(0, self.num_adjoint_vecs)]

        self.BPOD.compute_decomp(self.direct_vec_sources, self.adjoint_vec_sources)
        self.hankel_mat = self.BPOD.hankel_mat
        self.sing_vals = self.BPOD.sing_vals
        return self.BPOD.L_sing_vecs, self.BPOD.sing_vals, self.BPOD.R_sing_vecs
    
    
    def compute_direct_modes(self, num_modes):
        """Computes ``num_modes`` direct modes.
        
        Args:
            num_modes: integer number of direct modes to compute
        
        Returns:
            2D array with direct modes as columns
        """
        if not self._called_compute_decomp:
            self.compute_decomp()

        self.mode_nums = range(1,num_modes+1)
        self.direct_modes = N.zeros((self.direct_vecs.shape[0], num_modes))
        mode_dests = [(self.direct_modes, mode_index) for mode_index in range(num_modes)]
        self.BPOD.compute_direct_modes(self.mode_nums, mode_dests)
        return self.direct_modes
        
    def compute_adjoint_modes(self, num_modes):
        """Computes ``num_modes`` adjoint modes.
        
        Args:
            num_modes: integer number of adjoint modes to compute
        
        Returns:
            2D array with adjoint modes as columns
        """
        if not self._called_compute_decomp:
            self.compute_decomp()

        self.mode_nums = range(1,num_modes+1)
        self.adjoint_modes = N.zeros((self.adjoint_vecs.shape[0], num_modes))
        mode_dests = [(self.adjoint_modes, mode_index) for mode_index in range(num_modes)]
        self.BPOD.compute_adjoint_modes(self.mode_nums, mode_dests)
        return self.adjoint_modes
    

class SimpleUseDMD(object):
    """Simple class which keeps all data in memory and uses the DMD class.
    
    The purpose of this class is to get quick results for simple data.
    This class wraps the DMD class so that modes and matrices are returned. 
    
    Usage::

      # Put your vecs into columns of an array, "vecs"
      my_DMD = SimpleUseDMD(vecs)
      num_modes = 20
      modes = my_DMD.compute_modes(num_modes)
      ritz_vals = my_DMD.ritz_vals
    
    More fine-grain control is available through the other methods.
    
    For the more curious user, this module and class supply ``get_vec`` and 
    ``put_vec`` functions that get/put vecs from/into columns of numpy arrays.
    By default, the default inner product is ``(fiedl1*vec2.conj()).sum()``.
    """    
    def __init__(self, inner_product=util.inner_product, verbose=True, vecs=None):
        self.inner_product = inner_product
        self.DMD = dmd.DMD(inner_product=inner_product, get_vec=get_vec,
            put_vec=put_vec, max_vecs_per_node=1000, verbose=verbose)
        self._called_compute_decomp = False
        if vecs is not None:
            self.set_vecs(vecs)
        
    def set_vecs(self, vecs):
        """Sets the vecs used in DMD.
        
        Args:
            vecs: a 2D array with columns of vecs, ``X'' in literature.
        """ 
        self.vecs = vecs
        self.num_vecs = vecs.shape[1]

    
    def compute_decomp(self, vecs=None):
        """Computes decomposition.
        
        Kwargs:
            vecs: a 2D array with columns of vecs.
        
        Returns:
            Ritz values
            
            mode norms
        """
        if vecs is not None:
            self.vecs = vecs
        
        self._called_compute_decomp = True
        
        self.vec_sources = [(self.vecs, index) 
            for index in xrange(0, self.num_vecs)]
        self.DMD.compute_decomp(self.vec_sources)
        self.ritz_vals = self.DMD.ritz_vals
        self.mode_norms = self.DMD.mode_norms
        
        return self.DMD.ritz_vals, self.DMD.mode_norms
        
    
    def compute_modes(self, mode_nums, index_from=1):
        """Computes modes
                
        Args:
            mode_nums: list of integer number of modes to compute
        
        Kwargs:
            index_from: number corresponding to the first DMD mode (0, 1, other).
        
        Returns:
            modes: 2D array with modes as columns. 
                The first column corresponds
                to mode with number ``index_from``. There are columns of zeros
                for the modes that were not requested in mode_nums. 
        """
        num_modes = len(mode_nums)
        
        if not self._called_compute_decomp:
            self.compute_decomp()
        
        # TODO (Jon Tu): The modes are complex even when the data is real. 
        # This seems impossible. The imaginary parts are not all nearly zero.
        self.modes = N.zeros((self.vecs.shape[0], max(mode_nums)-index_from+1), 
            dtype=complex)
        mode_dests = [(self.modes, mode_index-index_from) for mode_index in mode_nums]
        self.DMD.compute_modes(mode_nums, mode_dests, index_from=index_from)
        return self.modes
    
    
    
