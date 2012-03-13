
"""A collection of classes that make it even simpler to use modred.

Everything is done in memory.
"""

import numpy as N

import sys, os
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(__file__), '..'),
    'src'))
import pod
import bpod
import dmd
import util
 
def get_field(array_and_index):
    array = array_and_index[0]
    index = array_and_index[1]
    return array[:,index]

def put_field(field, array_and_index):
    array = array_and_index[0]
    index = array_and_index[1]
    array[:,index] = field


class SimpleUsePOD(object):
    """Simple class which keeps all data in memory and uses the POD class.
    
    The purpose of this class is to get quick results for simple data.
    This class wraps the POD class so that modes and matrices are returned. 
    
    Usage::
      
      my_POD = SimpleUsePOD()
      my_POD.set_fields(fields)
      num_modes = 20
      modes = my_POD.compute_modes(num_modes)
      sing_vals = my_POD.sing_vals
    
    More fine-grain control is available through the other methods.
    
    For the more curious user, this module and class supply ``get_field`` and 
    ``put_field`` functions that get/put fields from/into columns of numpy arrays.
    By default, the default inner product is ``(fiedl1*field2.conj()).sum()``.
    """
    def __init__(self, inner_product=util.inner_product, verbose=True):
        self.inner_product = inner_product
        self.verbose = verbose
        self.POD = pod.POD(inner_product=inner_product, get_field=get_field,
            put_field=put_field, max_fields_per_node=1000, verbose=verbose)
        self._called_compute_decomp = False
        
        
    def set_fields(self, fields):
        """Sets the fields used in POD.
        
        Args:
            fields: a 2D array with columns of fields, ``X'' in literature.
        """ 
        self.fields = fields
        self.num_fields = fields.shape[1]

    
    def compute_decomp(self, fields=None):
        """Computes correlation matrix and its SVD.
        
        Computes correlation matrix with (i,j) entry <field_i, field_j>,
        then U E V^* = correlation_mat.
        
        Kwargs:
            fields: a 2D array with columns of fields, ``X'' in literature.
        
        Returns:
            singular vectors (U, which equals V)
            
            singular values (E)             
        """
        self._called_compute_decomp = True
        
        if fields is not None:
            self.fields = fields
        
        self.field_sources = [(self.fields, index) 
            for index in xrange(0, self.num_fields)]
        self.POD.compute_decomp(self.field_sources)
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
        self.modes = N.zeros((self.fields.shape[0], num_modes))
        mode_dests = [(self.modes, mode_index) for mode_index in range(num_modes)]
        self.POD.compute_modes(self.mode_nums, mode_dests)
        return self.modes




class SimpleUseBPOD(object):
    """Simple class which keeps all data in memory and uses the BPOD class.
    
    The purpose of this class is to get quick results for simple data.
    This class wraps the BPOD class so that modes and matrices are returned. 
    
    Usage::
      
      my_BPOD = SimpleUseBPOD()
      my_BPOD.set_fields(fields)
      num_modes = 20
      direct_modes = my_BPOD.compute_direct_modes(num_modes)
      adjoint_modes = my_BPOD.compute_adjoing_modes(num_modes)
      sing_vals = my_BPOD.sing_vals
    
    More fine-grain control is available through the other methods.
    
    For the more curious user, this module and class supply ``get_field`` and 
    ``put_field`` functions that get/put fields from/into columns of numpy arrays.
    By default, the default inner product is ``(fiedl1*field2.conj()).sum()``.
    """    
    def __init__(self, inner_product=util.inner_product, verbose=True):
        self.inner_product = inner_product
        self.BPOD = bpod.BPOD(inner_product=inner_product, get_field=get_field,
            put_field=put_field, max_fields_per_node=1000, verbose=verbose)
        self._called_compute_decomp = False
        
    def set_direct_fields(self, direct_fields):
        """Sets the direct fields.
        
        Args:
            direct_fields: a 2D array with columns of fields, ``X'' in literature.
        """ 
        self.direct_fields = direct_fields
        self.num_direct_fields = self.direct_fields.shape[1]

    def set_adjoint_fields(self, adjoint_fields):
        """Sets the adjoint fields.
        
        Args:
            adjoint_fields: a 2D array with columns of fields, ``Y'' in literature.
        """ 
        self.adjoint_fields = adjoint_fields
        self.num_adjoint_fields = self.adjoint_fields.shape[1]

    
    def compute_decomp(self, direct_fields=None, adjoint_fields=None):
        """Computes Hankel matrix and its SVD.
        
        Computes Hankel matrix with (i,j) entry <adjoint_field_i, direct_field_j>,
        then U E V^* = hankel_mat.
        
        Kwargs:
           direct_fields: a 2D array with columns of fields, ``X'' in literature.
           
           adjoint_fields: a 2D array with columns of fields, ``Y'' in literature.
        
        Returns:
            left singular vectors (U)
            
            singular values (E) 
            
            right singular vectors (V)
        """
        self._called_compute_decomp = True
        if direct_fields is not None:
            self.direct_fields = direct_fields
        
        if adjoint_fields is not None:
            self.adjoint_fields = adjoint_fields
        
        self.direct_field_sources = [(self.direct_fields, index) 
            for index in xrange(0, self.num_direct_fields)]
        self.adjoint_field_sources = [(self.adjoint_fields, index) 
            for index in xrange(0, self.num_adjoint_fields)]

        self.BPOD.compute_decomp(self.direct_field_sources, self.adjoint_field_sources)
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
        self.direct_modes = N.zeros((self.direct_fields.shape[0], num_modes))
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
        self.adjoint_modes = N.zeros((self.adjoint_fields.shape[0], num_modes))
        mode_dests = [(self.adjoint_modes, mode_index) for mode_index in range(num_modes)]
        self.BPOD.compute_adjoint_modes(self.mode_nums, mode_dests)
        return self.adjoint_modes
    

class SimpleUseDMD(object):
    """Simple class which keeps all data in memory and uses the DMD class.
    
    The purpose of this class is to get quick results for simple data.
    This class wraps the DMD class so that modes and matrices are returned. 
    
    Usage::
      
      my_DMD = SimpleUsePOD()
      my_DMD.set_fields(fields)
      num_modes = 20
      modes = my_DMD.compute_modes(num_modes)
      ritz_vals = my_DMD.ritz_vals
    
    More fine-grain control is available through the other methods.
    
    For the more curious user, this module and class supply ``get_field`` and 
    ``put_field`` functions that get/put fields from/into columns of numpy arrays.
    By default, the default inner product is ``(fiedl1*field2.conj()).sum()``.
    """    
    def __init__(self, inner_product=util.inner_product, verbose=True):
        self.inner_product = inner_product
        self.DMD = dmd.DMD(inner_product=inner_product, get_field=get_field,
            put_field=put_field, max_fields_per_node=1000, verbose=verbose)
        self._called_compute_decomp = False
        
    def set_fields(self, fields):
        """Sets the fields used in DMD.
        
        Args:
            fields: a 2D array with columns of fields, ``X'' in literature.
        """ 
        self.fields = fields
        self.num_fields = fields.shape[1]

    
    def compute_decomp(self, fields=None):
        """Computes decomposition.
        
        Kwargs:
            fields: a 2D array with columns of fields.
        
        Returns:
            Ritz values
            
            mode norms
        """
        if fields is not None:
            self.fields = fields
        
        self._called_compute_decomp = True
        
        self.field_sources = [(self.fields, index) 
            for index in xrange(0, self.num_fields)]
        self.DMD.compute_decomp(self.field_sources)
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
        self.modes = N.zeros((self.fields.shape[0], max(mode_nums)-index_from+1), 
            dtype=complex)
        mode_dests = [(self.modes, mode_index-index_from) for mode_index in mode_nums]
        self.DMD.compute_modes(mode_nums, mode_dests, index_from=index_from)
        return self.modes
    
    
    
