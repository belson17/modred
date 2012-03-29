"""DMD class"""

import numpy as N
from vecoperations import VecOperations
import pod
import util
import parallel

class DMD(object):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition.
        
    Args:
        vec_defs: Class or module w/functions ``get_vec``, ``put_vec``,
        ``inner_product``

    Kwargs:        
        put_mat: Function to put a matrix out of modred
      	
      	get_mat: Function to get a matrix into modred
               
        verbose: Print more information about progress and warnings
        
        max_vecs_per_node: max number of vectors in memory per node.
    
    Computes Ritz vectors from vecs.
    
    Usage::
    
      myDMD = DMD(vec_defs, max_vecs_per_node=500)
      myDMD.compute_decomp(sources)
      myDMD.compute_modes(range(1, 51), ['mode_%02d.txt'%i for i in range(1,51)])
    
    """
    def __init__(self, vec_defs, 
        get_mat=util.load_mat_text, put_mat=util.save_mat_text,
        max_vecs_per_node=None, POD=None, verbose=True):
        """Constructor"""
        self.vec_ops = VecOperations(vec_defs, 
            max_vecs_per_node=max_vecs_per_node, verbose=verbose)
        self.parallel = parallel.default_instance
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.POD = POD
        self.verbose = verbose
        self.ritz_vals = None
        self.build_coeffs = None
        self.mode_norms = None
        self.vec_sources = None



    def idiot_check(self, test_obj=None, test_obj_source=None):
        """See VecOperations documentation"""
        return self.vec_ops.idiot_check(test_obj, test_obj_source)


    def get_decomp(self, ritz_vals_source, mode_norms_source, 
        build_coeffs_source):
        """Retrieves the decomposition matrices from a source. """
        if self.get_mat is None:
            raise util.UndefinedError('Must specify a get_mat function')
        if self.parallel.is_rank_zero():
            self.ritz_vals = N.squeeze(N.array(self.get_mat(ritz_vals_source)))
            self.mode_norms = N.squeeze(N.array(self.get_mat(mode_norms_source)))
            self.build_coeffs = self.get_mat(build_coeffs_source)
        else:
            self.ritz_vals = None
            self.mode_norms = None
            self.build_coeffs = None
        if self.parallel.is_distributed():
            self.ritz_vals = self.parallel.comm.bcast(self.ritz_vals, root=0)
            self.mode_norms = self.parallel.comm.bcast(self.mode_norms, root=0)
            self.build_coeffs = self.parallel.comm.bcast(self.build_coeffs, root=0)
            
    def put_decomp(self, ritz_vals_dest, mode_norms_dest, build_coeffs_dest):
        """Puts the decomposition matrices in dest."""
        self.put_ritz_vals(ritz_vals_dest)
        self.put_mode_norms(mode_norms_dest)
        self.put_build_coeffs(build_coeffs_dest)
        
    def put_ritz_vals(self, dest):
        """Puts the Ritz values"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined, can't put")
        if self.parallel.is_rank_zero():
            self.put_mat(self.ritz_vals, dest)

    def put_mode_norms(self, dest):
        """Puts the mode norms"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined, can't put")
        if self.parallel.is_rank_zero():
            self.put_mat(self.mode_norms, dest)

    def put_build_coeffs(self, dest):
        """Puts the build coeffs"""
        if self.put_mat is None and self.parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined, can't put")
        if self.parallel.is_rank_zero():
            self.put_mat(self.build_coeffs, dest)



    def _compute_decomp(self, vec_sources):
        """Compute decomposition"""
         
        if vec_sources is not None:
            self.vec_sources = vec_sources
        if self.vec_sources is None:
            raise util.UndefinedError('vec_sources is not given')

        # Compute POD from vecs (excluding last vec)
        if self.POD is None:
            self.POD = pod.POD(self.vec_ops.vec_defs, 
                max_vecs_per_node=self.vec_ops.max_vecs_per_node, 
                verbose=self.verbose)
            # Don't use the returned mats, get them later from POD instance.
            dum, dum = self.POD.compute_decomp_and_return(
                vec_sources=self.vec_sources[:-1])
        elif self.vec_sources[:-1] != self.POD.vec_sources or \
            len(vec_sources) != len(self.POD.vec_sources)+1:
            raise RuntimeError('vec mismatch between POD and DMD '+\
                'objects.')
        pod_sing_vecs = self.POD.sing_vecs
        pod_sing_vals = self.POD.sing_vals
        _pod_sing_vals_sqrt_mat = N.mat(
            N.diag(N.array(pod_sing_vals).squeeze() ** -0.5))

        # Inner product of vecs w/POD modes
        num_vecs = len(self.vec_sources)
        pod_modes_star_times_vecs = N.mat(N.empty((num_vecs-1, num_vecs-1)))
        pod_modes_star_times_vecs[:,:-1] = self.POD.correlation_mat[:,1:]  
        pod_modes_star_times_vecs[:,-1] = \
            self.vec_ops.compute_inner_product_mat(self.vec_sources[:-1], 
                self.vec_sources[-1])
        pod_modes_star_times_vecs = _pod_sing_vals_sqrt_mat * pod_sing_vecs.H *\
            pod_modes_star_times_vecs
            
        # Reduced order linear system
        low_order_linear_map = pod_modes_star_times_vecs * pod_sing_vecs * \
            _pod_sing_vals_sqrt_mat
        self.ritz_vals, low_order_eig_vecs = N.linalg.eig(low_order_linear_map)
        
        # Scale Ritz vectors
        ritz_vecs_star_times_init_vec = low_order_eig_vecs.H * _pod_sing_vals_sqrt_mat * \
            pod_sing_vecs.H * self.POD.correlation_mat[:,0]
        ritz_vec_scaling = N.linalg.inv(low_order_eig_vecs.H * low_order_eig_vecs) *\
            ritz_vecs_star_times_init_vec
        ritz_vec_scaling = N.mat(N.diag(N.array(ritz_vec_scaling).squeeze()))

        # Compute mode energies
        self.build_coeffs = pod_sing_vecs * _pod_sing_vals_sqrt_mat *\
            low_order_eig_vecs * ritz_vec_scaling
        self.mode_norms = N.diag(self.build_coeffs.H * 
            self.POD.correlation_mat * self.build_coeffs).real
            
            
    def compute_decomp(self, vec_sources, ritz_vals_dest, mode_norms_dest, 
        build_coeffs_dest):
        """Computes decomposition and puts mats with ``put_mat``.
        
        Args:
            vec_sources: list of sources from which vecs are retrieved
            
            ritz_vals_dest: destination to which ritz_vals are ``put_mat``.
            
            mode_norms_dest: destination to which mode_norms are ``put_mat``.
            
            build_coeffs_dest: destination to which build_coeffs are ``put_mat``.
        """
        self._compute_decomp(vec_sources)
        self.put_decomp(ritz_vals_dest, mode_norms_dest, build_coeffs_dest)
        
    def compute_decomp_and_return(self, vec_sources):
        """Computes decomposition and returns.
        
        Returns:
            vec_sources, ritz_vals, mode_norms, build_coeffs
        """
        self._compute_decomp(vec_sources)
        return self.ritz_vals, self.mode_norms, self.build_coeffs
        
        
        
        
    def _compute_modes_helper(self, vec_sources=None):
        """Helper for ``compute_modes`` and ``compute_modes_helper``"""
        if self.build_coeffs is None:
            raise util.UndefinedError('Must define self.build_coeffs')
        # User should specify ALL vecs, even though all but last are used
        if vec_sources is not None:
            self.vec_sources = vec_sources
        
    def compute_modes(self, mode_nums, mode_dests, vec_sources=None, 
        index_from=0):
        """Computes modes
        
        Args:
            mode_nums: list of mode numbers, ``range(10)`` or ``[3, 2, 5]`` 
            
            mode_dests: destinations to ``put_vec`` the modes
            
        Kwargs:
            index_from: integer to start numbering modes from, 0, 1, or other.
            
            vec_sources: sources to vecs, can omit if given in ``compute_decomp``.
        """
        self._compute_modes_helper(vec_sources=vec_sources)
        self.vec_ops.compute_modes(mode_nums, mode_dests, 
            self.vec_sources[:-1], self.build_coeffs, index_from=index_from)
        
    def compute_modes_and_return(self, mode_nums, vec_sources=None, 
        index_from=0):
        """Computes modes and returns
        
        See ``compute_modes`` for details.

        Returns:
            a list of modes.

        In parallel, each MPI worker is returned a complete list of modes
        """
        self._compute_modes_helper(vec_sources=vec_sources)
        return self.vec_ops.compute_modes_and_return(mode_nums, 
            self.vec_sources[:-1], self.build_coeffs, index_from=index_from)
 
 