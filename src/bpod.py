"""BPOD class"""

import numpy as N
from vecoperations import VecOperations
import util
import parallel
import vectors as V

class BPOD(object):
    """Balanced Proper Orthogonal Decomposition
    
    Kwargs:
        inner_product: Function to take inner products
        
        put_mat: Function to put a matrix out of modred
      	
      	get_mat: Function to get a matrix into modred
      	                
        verbose: print more information about progress and warnings
        
        max_vecs_per_node: max number of vectors in memory per node.
    
    Computes direct and adjoint modes from direct and adjoint vecs.
    It uses VecOperations for low level functions.
    
    Usage::
    
      myBPOD = BPOD(inner_product=my_inner_product, max_vecs_per_node=500)
      myBPOD.compute_decomp(direct_vec_handles, adjoint_vec_handles)
      myBPOD.compute_direct_modes(range(50), direct_mode_handles)
      myBPOD.compute_adjoint_modes(range(50), adjoint_mode_handles)

    """
    
    def __init__(self, inner_product=None, 
        put_mat=util.save_array_text, get_mat=util.load_array_text,
        max_vecs_per_node=None, verbose=True):
        """Constructor """
        # Class that contains all of the low-level vec operations
        # and parallelizes them.
        self.vec_ops = VecOperations(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbose=verbose)
        self.parallel = parallel.default_instance
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbose = verbose
        self.L_sing_vecs = None
        self.R_sing_vecs = None
        self.sing_vals = None
        self.direct_vec_handles = None
        self.adjoint_vec_handles = None
        self.hankel_mat = None
        
        
    def idiot_check(self, test_vec_handle, max_handle_size=None):
        """See VecOperations documentation"""
        self.vec_ops.idiot_check(test_vec_handle, 
        	max_handle_size=max_handle_size)

    def get_decomp(self, L_sing_vecs_source, sing_vals_source, 
        R_sing_vecs_source):
        """Gets the decomposition matrices from elsewhere (memory or file)."""
        if self.get_mat is None:
            raise util.UndefinedError('Must specify a get_mat function')
        if self.parallel.is_rank_zero():
            self.L_sing_vecs = self.get_mat(L_sing_vecs_source)
            self.sing_vals = N.squeeze(N.array(self.get_mat(sing_vals_source)))
            self.R_sing_vecs = self.get_mat(R_sing_vecs_source)
        else:
            self.L_sing_vecs = None
            self.sing_vals = None
            self.R_sing_vecs = None
        if self.parallel.is_distributed():
            self.L_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs,
                root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals,
                root=0)
            self.R_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs, 
                root=0)
    
    
    def put_hankel_mat(self, hankel_mat_dest):
        """Put Hankel mat"""
        if self.put_mat is None:
            raise util.UndefinedError('put_mat not specified')
        elif self.parallel.is_rank_zero():
            self.put_mat(self.hankel_mat, hankel_mat_dest)           
    
    
    def put_L_sing_vecs(self, dest):
        """Put left singular vectors of SVD, V"""
        if self.put_mat is None:
            raise util.UndefinedError("put_mat not specified")
        elif self.parallel.is_rank_zero():
            self.put_mat(self.L_sing_vecs, dest)
        
    def put_R_sing_vecs(self, dest):
        """Put right singular vectors of SVD, U"""
        if self.put_mat is None:
            raise util.UndefinedError("put_mat not specified")
        elif self.parallel.is_rank_zero():
            self.put_mat(self.R_sing_vecs, dest)
    
    def put_sing_vals(self, dest):
        """Put singular values of SVD, E"""
        if self.put_mat is None:
            raise util.UndefinedError("put_mat not specified")
        elif self.parallel.is_rank_zero():
            self.put_mat(self.sing_vals, dest)
   
    
    def put_decomp(self, L_sing_vecs_dest, sing_vals_dest, R_sing_vecs_dest):
        """Save the decomposition matrices to file."""
        self.put_L_sing_vecs(L_sing_vecs_dest)
        self.put_R_sing_vecs(R_sing_vecs_dest)
        self.put_sing_vals(sing_vals_dest)
    
    
    def compute_decomp(self, direct_vec_handles, adjoint_vec_handles):
        """Finds Hankel mat and its SVD.
        
        Args:
            direct_vec_handles: list of handles for direct vecs
            
            adjoint_vec_handles: list of handles for adjoint vecs
        
        Returns:
            L_sing_vecs: matrix of left singular vectors (U in UEV*=H)
        
            sing_vals: 1D array of singular values (E in UEV*=H)
            
            R_sing_vecs: matrix of right singular vectors (V in UEV*=H)
        """
        self.direct_vec_handles = direct_vec_handles
        self.adjoint_vec_handles = adjoint_vec_handles
        self.hankel_mat = self.vec_ops.compute_inner_product_mat(
            self.adjoint_vec_handles, self.direct_vec_handles)
        self.compute_SVD()
        return self.L_sing_vecs, self.sing_vals, self.R_sing_vecs
        

    def compute_SVD(self):
        """Takes the SVD of the Hankel matrix.
        
        Useful if you already have the Hankel mat and want to skip 
        recomputing it. Intead, set ``self.hankel_mat``, and call this.
        """
        if self.parallel.is_rank_zero():
            self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = \
                util.svd(self.hankel_mat)
        else:
            self.L_sing_vecs = None
            self.R_sing_vecs = None
            self.sing_vals = None
        if self.parallel.is_distributed():
            self.L_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs,
                root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals,
                root=0)
            self.R_sing_vecs = self.parallel.comm.bcast(self.R_sing_vecs,
                root=0)
    
    
    
    def _compute_direct_modes_helper(self, direct_vec_handles=None):
        """Helper for ``compute_direct_modes`` and 
        ``compute_direct_modes_and_return`` see their docs for more info.
        """
        #self.R_sing_vecs and self.sing_vals must exist, else UndefinedError.
        
        if self.R_sing_vecs is None:
            raise util.UndefinedError('Must define self.R_sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
            
        if direct_vec_handles is not None:
            self.direct_vec_handles = direct_vec_handles
        if self.direct_vec_handles is None:
            raise util.UndefinedError('Must specify direct_vec_handles')
        # Switch to N.dot...
        build_coeff_mat = N.mat(self.R_sing_vecs) * \
            N.mat(N.diag(self.sing_vals**-0.5))
        return build_coeff_mat
        
    def compute_direct_modes_and_return(self, mode_nums, 
        direct_vec_handles=None, index_from=0):
        """Computes direct modes and returns them in a list.
        
        See ``compute_direct_modes`` for details.
        
        Returns:
            a list of modes
            
        In parallel, each MPI worker is returned a complete list of modes
        """
        build_coeff_mat = self._compute_direct_modes_helper(direct_vec_handles)
        return self.vec_ops.compute_modes_and_return(mode_nums, 
            self.direct_vec_handles, build_coeff_mat, index_from=index_from)
            
    def compute_direct_modes(self, mode_nums, mode_handles,
        direct_vec_handles=None, index_from=0):
        """Computes direct modes and calls ``self.put_vec`` on them.
        
        Args:
          mode_nums: Mode numbers to compute. 
              Examples are ``range(10)`` or ``[3,1,6,8]``. 
              The mode numbers need not be sorted,
              and sorting does not increase efficiency. 
              
          mode_handles: list of handles for modes.
          
        Kwargs:
          index_from: Index modes starting from 0, 1, or other.
          
          direct_vec_handles: list of handles for direct vecs. 
              Optional if already given when calling ``self.compute_decomp``.
        """
        build_coeff_mat = self._compute_direct_modes_helper(direct_vec_handles)
        self.vec_ops.compute_modes(mode_nums, mode_handles, 
            self.direct_vec_handles, build_coeff_mat, index_from=index_from)
        
        
    
    def _compute_adjoint_modes_helper(self, adjoint_vec_handles=None):
        """Helper for ``compute_adjoint_modes`` and 
        ``compute_adjoint_modes_and_return``, see those docs for more info.
        """
        #self.L_sing_vecs and self.sing_vals must exist, else UndefinedError.
        
        if self.L_sing_vecs is None:
            raise util.UndefinedError('Must define self.L_sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
        if adjoint_vec_handles is not None:
            self.adjoint_vec_handles = adjoint_vec_handles
        if self.adjoint_vec_handles is None:
            raise util.UndefinedError('Must specify adjoint_vec_handles')

        self.sing_vals = N.squeeze(N.array(self.sing_vals))
        
        build_coeff_mat = N.dot(self.L_sing_vecs, N.diag(self.sing_vals**-0.5))
        return build_coeff_mat      
    
    def compute_adjoint_modes(self, mode_nums, mode_handles,
        adjoint_vec_handles=None,  index_from=0):
        """Computes the adjoint modes ``self.put_vec`` s them.
        
        Args:
            mode_nums: Mode numbers to compute. 
                Examples are ``range(10)`` or ``[3, 1, 6, 8]``. 
                The mode numbers need not be sorted,
                and sorting does not increase efficiency. 
                
            mode_handles: list of handles for modes
            
        Kwargs:
            index_from: Index modes starting from 0, 1, or other.
                
            adjoint_vec_handles: list of handles for adjoint vecs. 
	      		Optional if already given when calling ``self.compute_decomp``.
        """
        build_coeff_mat = self._compute_adjoint_modes_helper(adjoint_vec_handles)
        self.vec_ops.compute_modes(mode_nums, mode_handles, 
            self.adjoint_vec_handles, build_coeff_mat, index_from=index_from)
        
    def compute_adjoint_modes_and_return(self, mode_nums, 
        adjoint_vec_handles=None, index_from=0):
        """Computes the adjoint modes returns them.
        
        See ``compute_adjoint_modes`` for details.

        Returns:
            a list of modes
            
        In parallel, each MPI worker returns a complete list of modes.
        """
        build_coeff_mat = self._compute_adjoint_modes_helper(adjoint_vec_handles)
        return self.vec_ops.compute_modes_and_return(mode_nums, 
            self.adjoint_vec_handles, build_coeff_mat, index_from=index_from)
        
