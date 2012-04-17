"""POD class"""
import numpy as N

from vecoperations import VecOperations
import util
import vectors as V
import parallel

class POD(object):
    """Proper Orthogonal Decomposition.
    
    Kwargs:
        inner_product: Fucntion to find inner product of two vector objects.
        
        put_mat: Function to put a matrix out of modred
      	
      	get_mat: Function to get a matrix into modred
      	
        verbose: print more information about progress and warnings
        
        max_vecs_per_node: max number of vectors in memory per node.

    Computes orthonormal POD modes from vecs.  
    
    Usage::
      
      myPOD = POD(inner_product=my_inner_product)
      myPOD.compute_decomp(vec_handles)
      myPOD.compute_modes(range(10), mode_handles)
      
    """
        
    def __init__(self, inner_product=None, 
        get_mat=util.load_array_text, put_mat=util.save_array_text, 
        max_vecs_per_node=None, verbose=True, 
        print_interval=10):
        """Constructor """
        self.vec_ops = VecOperations(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, 
            verbose=verbose, print_interval=print_interval)
        self.parallel = parallel.default_instance
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbose = verbose
        self.sing_vecs = None
        self.sing_vals = None
        self.correlation_mat = None
        self.vec_handles = None

     
    def idiot_check(self, test_vec_handle):
        """See VecOperations documentation"""
        self.vec_ops.idiot_check(test_vec_handle)

     
    def get_decomp(self, sing_vecs_source, sing_vals_source):
        """Gets the decomposition matrices from sources (memory or file)"""
        if self.get_mat is None:
            raise util.UndefinedError('Must specify a get_mat function')
        if self.parallel.is_rank_zero():
            self.sing_vecs = self.get_mat(sing_vecs_source)
            self.sing_vals = N.squeeze(N.array(self.get_mat(sing_vals_source)))
        else:
            self.sing_vecs = None
            self.sing_vals = None
        if self.parallel.is_distributed():
            self.sing_vecs = self.parallel.comm.bcast(self.sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
        
    def put_decomp(self, sing_vecs_dest, sing_vals_dest):
        """Put the decomposition matrices to file or memory."""
        self.put_sing_vecs(sing_vecs_dest)
        self.put_sing_vals(sing_vals_dest)
        
    def put_sing_vecs(self, dest):
        """Put singular vectors, U (==V)"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined")
            
        if self.parallel.is_rank_zero():
            self.put_mat(self.sing_vecs, dest)

    def put_sing_vals(self, dest):
        """Put singular values, E"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined")
            
        if self.parallel.is_rank_zero():
            self.put_mat(self.sing_vals, dest)

    def put_correlation_mat(self, correlation_mat_dest):
        """Put correlation matrix"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined")
        if self.parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, correlation_mat_dest)


    def compute_decomp(self, vec_handles):
        """Computes correlation mat X*X, then the SVD of this matrix.
        
        Args:
            vec_handles: list of handles for vecs
            
        Returns:
            sing_vecs: matrix of singular vectors (U, ==V, in UEV*=H)
        
            sing_vals: 1D array of singular values (E in UEV*=H) 
        """
        self.vec_handles = vec_handles
        self.correlation_mat = self.vec_ops.\
            compute_symmetric_inner_product_mat(self.vec_handles)
        #self.correlation_mat = self.vec_ops.\
        #    compute_inner_product_mat(self.vec_handles, self.vec_handles)
        self.compute_SVD()        
        return self.sing_vecs, self.sing_vals
       
        
        
    def compute_SVD(self):
        """Compute SVD, UEV*=correlation_mat"""
        if self.parallel.is_rank_zero():
            self.sing_vecs, self.sing_vals, dummy = \
                util.svd(self.correlation_mat)
        else:
            self.sing_vecs = None
            self.sing_vals = None
        if self.parallel.is_distributed():
            self.sing_vecs = self.parallel.comm.bcast(self.sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
            
            
            
            
    def _compute_modes_helper(self, vec_handles=None):
        """Helper for ``compute_modes`` and ``compute_modes_and_return``."""
        #self.sing_vecs, self.sing_vals must exist or an UndefinedError.
        if self.sing_vecs is None:
            raise util.UndefinedError('Must define self.sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
        if vec_handles is not None:
            self.vec_handles = vec_handles
        build_coeff_mat = N.dot(self.sing_vecs, N.diag(self.sing_vals**-0.5))
        return build_coeff_mat
    
    def compute_modes(self, mode_nums, mode_handles,
        vec_handles=None, index_from=0):
        """Computes the modes and calls ``put`` on them.
        
        Args:
            mode_nums: Mode numbers to compute. 
              Examples are ``range(10`` or ``[3,1,6,8]``. 
              The mode numbers need not be sorted,
              and sorting does not increase efficiency. 
              
            mode_handles: list of handles for modes
            
        Kwargs:
            index_from: Index modes starting from 0, 1, or other.
              
            vec_handles: list of handles for vectors. 
	            Optional if already given when calling ``self.compute_decomp``.
        """
        build_coeff_mat = self._compute_modes_helper(vec_handles)
        self.vec_ops.compute_modes(mode_nums, mode_handles,
             self.vec_handles, build_coeff_mat, index_from=index_from)
    
    def compute_modes_and_return(self, mode_nums, vec_handles=None, 
    	index_from=0):
        """Computes modes and returns them in a list.
        
        See ``compute_modes`` for details.
        
        Returns:
            a list of modes
            
        In parallel, each MPI worker is returned a complete list of modes
        """
        build_coeff_mat = self._compute_modes_helper(vec_handles)
        return self.vec_ops.compute_modes_and_return(mode_nums,
             self.vec_handles, build_coeff_mat, index_from=index_from)
    


