
import numpy as N
from fieldoperations import FieldOperations
import util
import parallel

class BPOD(object):
    """
    Balanced Proper Orthogonal Decomposition
    
    Generate direct and adjoint modes from direct and adjoint simulation 
    snapshots. BPOD uses FieldOperations for low level
    functions.
    
    Usage::
    
      import bpod      
      myBPOD = bpod.BPOD(load_field = my_load_field, save_field=my_save_field,
          inner_product = my_inner_product, max_fields_per_node = 500)
      myBPOD.compute_decomp(direct_snap_paths, adjoint_snap_paths)      
      myBPOD.save_hankel_mat(hankelPath)
      myBPOD.save_decomp(L_sing_vecs_path, sing_vals_path, R_sing_vecs_path)
      myBPOD.compute_direct_modes(range(1, 50), 'bpod_direct_mode_%03d.txt')
      myBPOD.compute_adjoint_modes(range(1, 50), 'bpod_adjoint_mode_%03d.txt')
    """
    
    def __init__(self, load_field=None, save_field=None, 
        save_mat=util.save_mat_text, load_mat=util.load_mat_text,
        inner_product=None, max_fields_per_node=2, verbose=True):
        """
        BPOD constructor
        
          load_field 
            Function to load a snapshot given a filepath.
          save_field 
            Function to save a mode given data and an output path.
          save_mat
            Function to save a matrix.
          inner_product
            Function to take inner product of two snapshots.
          verbose 
            True means print more information about progress and warnings
        """
        # Class that contains all of the low-level field operations
        # and parallelizes them.
        self.field_ops_slave = FieldOperations(load_field=load_field, 
            save_field=save_field, inner_product=inner_product, 
            max_fields_per_node=max_fields_per_node, verbose=verbose)
        self.parallel = parallel.parallel_default

        self.load_mat = load_mat
        self.save_mat = save_mat
        self.verbose = verbose
 

    def load_decomp(self, L_sing_vecs_path, sing_vals_path, R_sing_vecs_path):
        """
        Loads the decomposition matrices from file. 
        """
        if self.load_mat is None:
            raise UndefinedError('Must specify a load_mat function')
        if self.parallel.is_rank_zero():
            self.L_sing_vecs = self.load_mat(L_sing_vecs_path)
            self.sing_vals = N.squeeze(N.array(self.load_mat(sing_vals_path)))
            self.R_sing_vecs = self.load_mat(R_sing_vecs_path)
        else:
            self.L_sing_vecs = None
            self.sing_vals = None
            self.R_sing_vecs = None
        if self.parallel.is_distributed():
            self.L_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs, root=0)
            self.sing_vals = self.parallel.comm.bcast(self.sing_vals, root=0)
            self.R_sing_vecs = self.parallel.comm.bcast(self.L_sing_vecs, root=0)
    
    
    def save_hankel_mat(self, hankel_mat_path):
        if self.save_mat is None:
            raise util.UndefinedError('save_mat not specified')
        elif self.parallel.is_rank_zero():
            self.save_mat(self.hankel_mat, hankel_mat_path)           
    
    
    def save_decomp(self, L_sing_vecs_path, sing_vals_path, R_sing_vecs_path):
        """Save the decomposition matrices to file."""
        if self.save_mat is None:
            raise util.UndefinedError("save_mat not specified")
        elif self.parallel.is_rank_zero():
            self.save_mat(self.L_sing_vecs, L_sing_vecs_path)
            self.save_mat(self.R_sing_vecs, R_sing_vecs_path)
            self.save_mat(self.sing_vals, sing_vals_path)

        
    def compute_decomp(self, direct_snap_paths, adjoint_snap_paths):
        """
        Compute BPOD decomposition from given data.
        
        First computes the Hankel mat Y*X, then the SVD of this matrix.
        """        
        self.direct_snap_paths = direct_snap_paths
        self.adjoint_snap_paths = adjoint_snap_paths
        # Do Y.conj()*X
        self.hankel_mat = self.field_ops_slave.compute_inner_product_mat(
            self.adjoint_snap_paths, self.direct_snap_paths)
        self.compute_SVD()        
        #self.parallel.evaluate_and_bcast([self.L_sing_vecs,self.sing_vals,self.\
        #    R_sing_vecs], util.svd, arguments = [self.hankel_mat])


    def compute_SVD(self):
        """Assumes the hankel matrix is in memory, takes the SVD
        
        This is especially useful if you already have the hankel mat and 
        only want to compute the SVD. You can skip using compute_decomp,
        and instead load the hankel mat, set self.hankel_mat, and call this
        function."""
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
        

    def compute_direct_modes(self, mode_nums, mode_path, index_from=1,
        direct_snap_paths=None):
        """
        Computes the direct modes and saves them to file.
        
        mode_nums
          Mode numbers to compute on this processor. This 
          includes the index_from, so if index_from=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 

        mode_path
          Full path to mode location, e.g /home/user/mode_%d.txt.

        index_from
          Choose to index modes starting from 0, 1, or other.
        
        self.R_sing_vecs & self.sing_vals must exist, else an UndefinedError.
        """
        if self.R_sing_vecs is None:
            raise util.UndefinedError('Must define self.R_sing_vecs')
        if self.sing_vals is None:
            raise util.UndefinedError('Must define self.sing_vals')
            
        if direct_snap_paths is not None:
            self.direct_snap_paths = direct_snap_paths
        if self.direct_snap_paths is None:
            raise util.UndefinedError('Must specify direct_snap_paths')
        # Switch to N.dot...
        build_coeff_mat = N.mat(self.R_sing_vecs)*N.mat(N.diag(self.sing_vals**-0.5))

        self.field_ops_slave._compute_modes(mode_nums, mode_path, 
            self.direct_snap_paths, build_coeff_mat, index_from=index_from)
    
    def compute_adjoint_modes(self, mode_nums, mode_path, index_from=1,
        adjoint_snap_paths=None):
        """
        Computes the adjoint modes and saves them to file.
        
        mode_nums
          Mode numbers to compute on this processor. This 
          includes the index_from, so if index_from=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work.

        mode_path
          Full path to mode location, e.g /home/user/mode_%d.txt.

        index_from
          Choose to index modes starting from 0, 1, or other.
        
        self.L_sing_vecs, self.sing_vals must exist or an UndefinedError.
        """
        if self.L_sing_vecs is None:
            raise UndefinedError('Must define self.L_sing_vecs')
        if self.sing_vals is None:
            raise UndefinedError('Must define self.sing_vals')
        if adjoint_snap_paths is not None:
            self.adjoint_snap_paths=adjoint_snap_paths
        if self.adjoint_snap_paths is None:
            raise util.UndefinedError('Must specify adjoint_snap_paths')

        self.sing_vals = N.squeeze(N.array(self.sing_vals))
        # Switch to N.dot...
        build_coeff_mat = N.mat(self.L_sing_vecs) * \
            N.mat(N.diag(self.sing_vals**-0.5))
                 
        self.field_ops_slave._compute_modes(mode_nums, mode_path,
            self.adjoint_snap_paths, build_coeff_mat, index_from=index_from)
    
