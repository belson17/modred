"""POD class"""
import numpy as N

from vecoperations import VecOperations
import util
import parallel

class POD(object):
    """Proper Orthogonal Decomposition
    
    Computes orthonormal modes from vecs.  
    
    Usage::
      
      myPOD = POD(...)
      myPOD.compute_decomp(vec_sources=my_vec_sources)
      myPOD.compute_modes(range(1,100), ['mode%d.txt'%i for i in range(1,100)])
    """
        
    def __init__(self, get_vec=None, put_vec=None, 
        get_mat=util.load_mat_text, put_mat=util.save_mat_text, 
        inner_product=None, max_vecs_per_node=None, verbose=True, 
        print_interval=10):
        """Constructor
        
        Kwargs:
            get_vec: function to get a vec from elsewhere (memory or a file).
            
            put_vec: function to put a vec elsewhere (to memory or a file).
            
            put_mat: function to put a matrix (to memory or file).
            
            inner_product: function to take inner product of two vecs.
            
            verbose: print more information about progress and warnings.
            
            max_vecs_per_node: max number of vectors in memory per node.
            
        Returns:
            POD instance
        """
        self.vec_ops = VecOperations(get_vec=get_vec, 
            put_vec=put_vec, inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, 
            verbose=verbose, print_interval=print_interval)
        self.parallel = parallel.default_instance

        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbose = verbose
        self.sing_vecs = None
        self.sing_vals = None
        self.correlation_mat = None
        self.vec_sources = None
     
    def idiot_check(self, test_obj=None, test_obj_source=None):
        """See VecOperations documentation"""
        return self.vec_ops.idiot_check(test_obj, test_obj_source)

     
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
 
    def put_correlation_mat(self, correlation_mat_dest):
        """Put correlation matrix"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined")
        if self.parallel.is_rank_zero():
            self.put_mat(self.correlation_mat, correlation_mat_dest)
        
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


    
    def compute_decomp(self, vec_sources):
        """Computes correlation mat X*X, then the SVD of this matrix."""
        self.vec_sources = vec_sources
        self.correlation_mat = self.vec_ops.\
            compute_symmetric_inner_product_mat(self.vec_sources)
        #self.correlation_mat = self.vec_ops.\
        #    compute_inner_product_mat(self.vec_sources, self.vec_sources)
        self.compute_SVD()
        
        
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
            
            
    def compute_modes(self, mode_nums, mode_dests, index_from=1,
        vec_sources=None):
        """Computes the modes and calls ``self.put_vec`` on them.
        
        Args:
            mode_nums: Mode numbers to compute. 
              Examples are [1,2,3,4,5] or [3,1,6,8]. 
              The mode numbers need not be sorted,
              and sorting does not increase efficiency. 
              
            mode_dests: list of destinations to put modes (memory or file)
        
        Kwargs:
            index_from: Index modes starting from 0, 1, or other.
              
            vec_sources: Paths to vecs. Optional if already given when calling 
                ``self.compute_decomp``.


        self.sing_vecs, self.sing_vals must exist or an UndefinedError.
        """
        if self.sing_vecs is None:
            raise util.UndefinedError('Must define self.sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
        if vec_sources is not None:
            self.vec_sources = vec_sources

        build_coeff_mat = N.dot(self.sing_vecs, N.diag(self.sing_vals**-0.5))

        self.vec_ops.compute_modes(mode_nums, mode_dests,
             self.vec_sources, build_coeff_mat, index_from=index_from)
    


