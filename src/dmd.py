
import numpy as N
from fieldoperations import FieldOperations
from pod import POD
import util
import parallel

class DMD(object):
    """
    Dynamic Mode Decomposition/Koopman Mode Decomposition
        
    Generate Ritz vectors from simulation snapshots.
    
    """

    def __init__(self, load_field=None, save_field=None, 
        load_mat=util.load_mat_text, save_mat=util.save_mat_text,
        inner_product=None, 
        max_fields_per_node=None, POD=None, verbose=True):
        """
        DMD constructor
        """
        self.field_ops = FieldOperations(load_field=load_field,\
            save_field=save_field, inner_product=inner_product,
            max_fields_per_node=max_fields_per_node, verbose=\
            verbose)
        self.parallel = parallel.default_instance

        self.load_mat = load_mat
        self.save_mat = save_mat
        self.POD = POD
        self.verbose = verbose

    def load_decomp(self, ritz_vals_path, mode_norms_path, build_coeffs_path):
        """
        Loads the decomposition matrices from file. 
        """
        if self.load_mat is None:
            raise UndefinedError('Must specify a load_mat function')
        if self.parallel.is_rank_zero():
            self.ritz_vals = N.squeeze(N.array(self.load_mat(ritz_vals_path)))
            self.mode_norms = N.squeeze(N.array(self.load_mat(mode_norms_path)))
            self.build_coeffs = self.load_mat(build_coeffs_path)
        else:
            self.ritz_vals = None
            self.mode_norms = None
            self.build_coeffs = None
        if self.parallel.is_distributed():
            self.ritz_vals = self.parallel.comm.bcast(self.ritz_vals, root=0)
            self.mode_norms = self.parallel.comm.bcast(self.mode_norms, root=0)
            self.build_coeffs = self.parallel.comm.bcast(self.build_coeffs, root=0)
            
    def save_decomp(self, ritz_vals_path, mode_norms_path, build_coeffs_path):
        """Save the decomposition matrices to file."""
        self.save_ritz_vals(ritz_vals_path)
        self.save_mode_norms(mode_norms_path)
        self.save_build_coeffs(build_coeffs_path)
        
    def save_ritz_vals(self, path):
        if self.save_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("save_mat is undefined, can't save")
        if self.parallel.is_rank_zero():
            self.save_mat(self.ritz_vals, path)

    def save_mode_norms(self, path):
        if self.save_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("save_mat is undefined, can't save")
        if self.parallel.is_rank_zero():
            self.save_mat(self.mode_norms, path)

    def save_build_coeffs(self, path):
        if self.save_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("save_mat is undefined, can't save")
        if self.parallel.is_rank_zero():
            self.save_mat(self.build_coeffs, path)



    def compute_decomp(self, snap_paths):
        """
        Compute DMD decomposition
        """
         
        if snap_paths is not None:
            self.snap_paths = snap_paths
        if self.snap_paths is None:
            raise util.UndefinedError('snap_paths is not given')

        # Compute POD from snapshots (excluding last snapshot)
        if self.POD is None:
            self.POD = POD(load_field=self.field_ops.load_field, 
                inner_product=self.field_ops.inner_product, 
                max_fields_per_node=self.field_ops.max_fields_per_node, 
                verbose=self.verbose)
            self.POD.compute_decomp(snap_paths=self.snap_paths[:-1])
        elif self.snaplist[:-1] != self.POD.snaplist or len(snap_paths) !=\
            len(self.POD.snap_paths)+1:
            raise RuntimeError('Snapshot mismatch between POD and DMD '+\
                'objects.')     
        _pod_sing_vals_sqrt_mat = N.mat(
            N.diag(N.array(self.POD.sing_vals).squeeze() ** -0.5))

        # Inner product of snapshots w/POD modes
        num_snaps = len(self.snap_paths)
        pod_modes_star_times_snaps = N.mat(N.empty((num_snaps-1, num_snaps-1)))
        pod_modes_star_times_snaps[:, :-1] = self.POD.correlation_mat[:,1:]  
        pod_modes_star_times_snaps[:, -1] = self.field_ops.\
            compute_inner_product_mat(self.snap_paths[:-1], self.snap_paths[
            -1])
        pod_modes_star_times_snaps = _pod_sing_vals_sqrt_mat * self.POD.\
            sing_vecs.H * pod_modes_star_times_snaps
            
        # Reduced order linear system
        low_order_linear_map = pod_modes_star_times_snaps * self.POD.sing_vecs * \
            _pod_sing_vals_sqrt_mat
        self.ritz_vals, low_order_eig_vecs = N.linalg.eig(low_order_linear_map)
        
        # Scale Ritz vectors
        ritz_vecs_star_times_init_snap = low_order_eig_vecs.H * _pod_sing_vals_sqrt_mat * \
            self.POD.sing_vecs.H * self.POD.correlation_mat[:,0]
        ritz_vec_scaling = N.linalg.inv(low_order_eig_vecs.H * low_order_eig_vecs) *\
            ritz_vecs_star_times_init_snap
        ritz_vec_scaling = N.mat(N.diag(N.array(ritz_vec_scaling).squeeze()))

        # Compute mode energies
        self.build_coeffs = self.POD.sing_vecs * _pod_sing_vals_sqrt_mat *\
            low_order_eig_vecs * ritz_vec_scaling
        self.mode_norms = N.diag(self.build_coeffs.H * self.POD.\
            correlation_mat * self.build_coeffs).real
        
    def compute_modes(self, mode_nums, mode_path, index_from=1, snap_paths=None):
        if self.build_coeffs is None:
            raise util.UndefinedError('Must define self.build_coeffs')
        # User should specify ALL snapshots, even though all but last are used
        if snap_paths is not None:
            self.snap_paths = snap_paths
        self.field_ops._compute_modes(mode_nums, mode_path, self.\
            snap_paths[:-1], self.build_coeffs, index_from=index_from)

        
