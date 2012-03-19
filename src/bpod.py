
import numpy as N
from vecoperations import VecOperations
import util
import parallel

class BPOD(object):
    """Balanced Proper Orthogonal Decomposition
    
    Computes direct and adjoint modes from direct and adjoint vecs.
    It uses VecOperations for low level functions.
    
    Usage::
    
      myBPOD = BPOD(get_vec=my_get_vec, put_vec=my_put_vec,
          inner_product=my_inner_product, max_vecs_per_node=500)
      myBPOD.compute_decomp(direct_vec_sources, adjoint_vec_sources)      
      myBPOD.compute_direct_modes(range(1, 50), 'bpod_direct_mode_%03d.txt')
      myBPOD.compute_adjoint_modes(range(1, 50), 'bpod_adjoint_mode_%03d.txt')
    """
    
    def __init__(self, get_vec=None, put_vec=None, 
        put_mat=util.save_mat_text, get_mat=util.load_mat_text,
        inner_product=None, max_vecs_per_node=2, verbose=True):
        """Constructor
        
        Kwargs:
            get_vec: Function to get a vec from elsewhere (memory or file).
            
            put_vec: Function to put a vec elsewhere (memory or file).
            
            put_mat: Function to put a matrix elsewhere (memory or file).
            
            inner_product: Function to take inner product of two vecs.
            
            verbose: print more information about progress and warnings
        """
        # Class that contains all of the low-level vec operations
        # and parallelizes them.
        self.vec_ops = VecOperations(get_vec=get_vec, 
            put_vec=put_vec, inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbose=verbose)
        self.parallel = parallel.default_instance

        self.get_mat = get_mat
        self.put_mat = put_mat
        self.verbose = verbose
        
    def idiot_check(self, test_obj=None, test_obj_source=None):
        return self.vec_ops.idiot_check(test_obj, test_obj_source)

    def get_decomp(self, L_sing_vecs_source, sing_vals_source, R_sing_vecs_source):
        """Gets the decomposition matrices from elsewhere (memory or file)."""
        if self.get_mat is None:
            raise UndefinedError('Must specify a get_mat function')
        if self.parallel.is_rank_zero():
            self.L_sing_vecs = self.get_mat(L_sing_vecs_source)
            self.sing_vals = N.squeeze(N.array(self.get_mat(sing_vals_source)))
            self.R_sing_vecs = self.get_mat(R_sing_vecs_source)
        else:
            self.L_sing_vecs = None
            self.sing_vals = None
            self.R_sing_vecs = None
        if self.parallel.is_distributed():
            self.L_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
            self.R_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs, root=0)
    
    
    def put_hankel_mat(self, hankel_mat_dest):
        if self.put_mat is None:
            raise util.UndefinedError('put_mat not specified')
        elif self.parallel.is_rank_zero():
            self.put_mat(self.hankel_mat, hankel_mat_dest)           
    
    
    def put_L_sing_vecs(self, dest):
        if self.put_mat is None:
            raise util.UndefinedError("put_mat not specified")
        elif self.parallel.is_rank_zero():
            self.put_mat(self.L_sing_vecs, dest)
        
    def put_R_sing_vecs(self, dest):
        if self.put_mat is None:
            raise util.UndefinedError("put_mat not specified")
        elif self.parallel.is_rank_zero():
            self.put_mat(self.R_sing_vecs, dest)
    
    def put_sing_vals(self, dest):
        if self.put_mat is None:
            raise util.UndefinedError("put_mat not specified")
        elif self.parallel.is_rank_zero():
            self.put_mat(self.sing_vals, dest)
   
    
    def put_decomp(self, L_sing_vecs_dest, sing_vals_dest, R_sing_vecs_dest):
        """Save the decomposition matrices to file."""
        self.put_L_sing_vecs(L_sing_vecs_dest)
        self.put_R_sing_vecs(R_sing_vecs_dest)
        self.put_sing_vals(sing_vals_dest)
        
        
    def compute_decomp(self, direct_vec_sources, adjoint_vec_sources):
        """Compute BPOD from given vecs.
        
        Computes the Hankel mat Y*X, then takes the SVD of this matrix.
        """        
        self.direct_vec_sources = direct_vec_sources
        self.adjoint_vec_sources = adjoint_vec_sources
        # Do Y.conj()*X
        self.hankel_mat = self.vec_ops.compute_inner_product_mat(
            self.adjoint_vec_sources, self.direct_vec_sources)
        self.compute_SVD()        
        #self.parallel.evaluate_and_bcast([self.L_sing_vecs,self.sing_vals,self.\
        #    R_sing_vecs], util.svd, arguments = [self.hankel_mat])


    def compute_SVD(self):
        """Takes the SVD of the Hankel matrix.
        
        Useful if you already have the Hankel mat and want to skip 
        recomputing it. Intead, set self.hankel_mat, and call this.
        """
        if self.parallel.is_rank_zero():
            self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = \
                util.svd(self.hankel_mat)
        else:
            self.L_sing_vecs = None
            self.R_sing_vecs = None
            self.sing_vals = None
        if self.parallel.is_distributed():
            self.L_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
            self.R_sing_vecs = self.parallel.comm.bcast(self.R_sing_vecs, root=0)
        

    def compute_direct_modes(self, mode_nums, mode_dests, index_from=1,
        direct_vec_sources=None):
        """Computes the direct modes and ``self.put_vec``s them.
        
        Args:
          mode_nums: Mode numbers to compute. 
              Examples are [1,2,3,4,5] or [3,1,6,8]. 
              The mode numbers need not be sorted,
              and sorting does not increase efficiency. 
              
          mode_dest: list of modes' destinations (file or memory)
          
        Kwargs:
          index_from:
              Index modes starting from 0, 1, or other.
          
          direct_vec_sources: sources to direct vecs. 
              Optional if already given when calling ``self.compute_decomp``.
            
        self.R_sing_vecs and self.sing_vals must exist, else UndefinedError.
        """
        
        if self.R_sing_vecs is None:
            raise util.UndefinedError('Must define self.R_sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
            
        if direct_vec_sources is not None:
            self.direct_vec_sources = direct_vec_sources
        if self.direct_vec_sources is None:
            raise util.UndefinedError('Must specify direct_vec_sources')
        # Switch to N.dot...
        build_coeff_mat = N.mat(self.R_sing_vecs)*N.mat(N.diag(self.sing_vals**-0.5))
        self.vec_ops._compute_modes(mode_nums, mode_dests, 
            self.direct_vec_sources, build_coeff_mat, index_from=index_from)
    
    def compute_adjoint_modes(self, mode_nums, mode_dests, index_from=1,
        adjoint_vec_sources=None):
        """Computes the adjoint modes ``self.put_vec``s them.
        
        Args:
            mode_nums: Mode numbers to compute. 
                Examples are [1,2,3,4,5] or [3,1,6,8]. 
                The mode numbers need not be sorted,
                and sorting does not increase efficiency. 
                
            mode_dest: list of modes' destinations (file or memory).
        
        Kwargs:
            index_from: Index modes starting from 0, 1, or other.
                
            adjoint_vec_sources: sources of adjoint vecs. 
            		Optional if already given when calling ``self.compute_decomp``.
            
        self.L_sing_vecs and self.sing_vals must exist, else UndefinedError.
        """
        
        if self.L_sing_vecs is None:
            raise UndefinedError('Must define self.L_sing_vecs')
        if self.sing_vals is None:
            raise UndefinedError('Must define self.sing_vals')
        if adjoint_vec_sources is not None:
            self.adjoint_vec_sources=adjoint_vec_sources
        if self.adjoint_vec_sources is None:
            raise util.UndefinedError('Must specify adjoint_vec_sources')

        self.sing_vals = N.squeeze(N.array(self.sing_vals))
        
        build_coeff_mat = N.dot(self.L_sing_vecs, N.diag(self.sing_vals**-0.5))
                 
        self.vec_ops._compute_modes(mode_nums, mode_dests,
            self.adjoint_vec_sources, build_coeff_mat, index_from=index_from)
    
