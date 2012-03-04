
import numpy as N

from fieldoperations import FieldOperations
import util
import parallel

class POD(object):
    """Proper Orthogonal Decomposition
    
    Computes orthonormal modes from fields.  
    
    Usage::
      
      myPOD = POD(...)
      myPOD.compute_decomp(field_paths=my_field_paths)
      myPOD.compute_modes(range(1,100), mode_path)
    """
        
    def __init__(self, get_field=None, put_field=None, 
        load_mat=util.load_mat_text, save_mat=util.save_mat_text, 
        inner_product=None, max_fields_per_node=None, verbose=True, 
        print_interval=10):
        """Constructor
        
        Kwargs:
            get_field 
                Function to get a field from elsewhere (memory or a file).
            put_field 
                Function to put a field elsewhere (to memory or a file).
            save_mat
                Function to save a matrix.
            inner_product
                Function to take inner product of two fields.
            verbose 
                True means print more information about progress and warnings.
        Returns:
            POD instance
        """
        self.field_ops = FieldOperations(get_field=get_field, 
            put_field=put_field, inner_product=inner_product, 
            max_fields_per_node=max_fields_per_node, 
            verbose=verbose, print_interval=print_interval)
        self.parallel = parallel.default_instance

        self.load_mat = load_mat
        self.save_mat = save_mat
        self.verbose = verbose
     
    def load_decomp(self, sing_vecs_path, sing_vals_path):
        """Loads the decomposition matrices from file. """
        if self.load_mat is None:
            raise UndefinedError('Must specify a load_mat function')
        if self.parallel.is_rank_zero():
            self.sing_vecs = self.load_mat(sing_vecs_path)
            self.sing_vals = N.squeeze(N.array(self.load_mat(sing_vals_path)))
        else:
            self.sing_vecs = None
            self.sing_vals = None
        if self.parallel.is_distributed():
            self.sing_vecs = self.parallel.comm.bcast(self.sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
 
    def save_correlation_mat(self, correlation_mat_path):
        if self.save_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("save_mat is undefined, can't save")
        if self.parallel.is_rank_zero():
            self.save_mat(self.correlation_mat, correlation_mat_path)
        
    def save_decomp(self, sing_vecs_path, sing_vals_path):
        """Save the decomposition matrices to file."""
        self.save_sing_vecs(sing_vecs_path)
        self.save_sing_vals(sing_vals_path)
        
        
    def save_sing_vecs(self, path):
        if self.save_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("save_mat is undefined, can't save")
            
        if self.parallel.is_rank_zero():
            self.save_mat(self.sing_vecs, path)

    def save_sing_vals(self, path):
        if self.save_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("save_mat is undefined, can't save")
            
        if self.parallel.is_rank_zero():
            self.save_mat(self.sing_vals, path)


    
    def compute_decomp(self, field_paths):
        """Computes correlation mat X*X, then the SVD of this matrix."""
        self.field_paths = field_paths
        self.correlation_mat = self.field_ops.\
            compute_symmetric_inner_product_mat(self.field_paths)
        #self.correlation_mat = self.field_ops.\
        #    compute_inner_product_mat(self.field_paths, self.field_paths)
        self.compute_SVD()
        
        
    def compute_SVD(self):
        if self.parallel.is_rank_zero():
            self.sing_vecs, self.sing_vals, dummy = \
                util.svd(self.correlation_mat)
        else:
            self.sing_vecs = None
            self.sing_vals = None
        if self.parallel.is_distributed():
            self.sing_vecs = self.parallel.comm.bcast(self.sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
            
            
    def compute_modes(self, mode_nums, mode_path, index_from=1, field_paths=None):
        """Computes the modes and calls ``self.put_field`` on them.
        
        Args:
            mode_nums: Mode numbers to compute. 
              Examples are [1,2,3,4,5] or [3,1,6,8]. 
              The mode numbers need not be sorted,
              and sorting does not increase efficiency. 
              
            mode_path:
              Full path to mode location, e.g. /home/user/mode_%d.txt.
        
        
        Kwargs:
            index_from: Index modes starting from 0, 1, or other.
              
            field_paths: Paths to fields. Optional if already given when calling 
                ``self.compute_decomp``.


        self.sing_vecs, self.sing_vals must exist or an UndefinedError.
        """
        if self.sing_vecs is None:
            raise util.UndefinedError('Must define self.sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
        if field_paths is not None:
            self.field_paths = field_paths

        build_coeff_mat = N.mat(self.sing_vecs) * \
            N.mat(N.diag(self.sing_vals**-0.5))

        self.field_ops._compute_modes(mode_nums, mode_path,
             self.field_paths, build_coeff_mat, index_from=index_from)
    


