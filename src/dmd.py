"""DMD class"""

import numpy as N
from vectorspace import VectorSpace
import pod
import util
import parallel as parallel_mod
parallel = parallel_mod.parallel_default_instance
import vectors as V

class DMD(object):
    """Dynamic Mode Decomposition/Koopman Mode Decomposition.
        
    Kwargs:        
        inner_product: Function to compute inner product.
        
        put_mat: Function to put a matrix out of modred.
      	
      	get_mat: Function to get a matrix into modred.
               
        max_vecs_per_node: max number of vectors in memory per node.

        verbosity: 0 prints almost nothing, 1 prints progress and warnings
        
        POD: POD object to use for computations.
        
    Computes Ritz vectors from vecs.
    
    Usage::
    
      myDMD = DMD(my_inner_product)
      myDMD.compute_decomp(vec_handles)
      myDMD.compute_modes(range(50), mode_handles)
    
    """
    def __init__(self, inner_product, 
        get_mat=util.load_array_text, put_mat=util.save_array_text,
        max_vecs_per_node=None, POD=None, verbosity=1):
        """Constructor"""
        self.vec_space = VectorSpace(inner_product=inner_product, 
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.get_mat = get_mat
        self.put_mat = put_mat
        self.POD = POD
        self.verbosity = verbosity
        self.ritz_vals = None
        self.build_coeffs = None
        self.mode_norms = None
        self.vec_handles = None
        self.vecs = None



    def sanity_check(self, test_vec_handle):
        """Check user-supplied vector handle.
        
        Args:
            test_vec_handle: a vector handle.
        
        See :py:meth:`vectorspace.VectorSpace.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)

    def sanity_check_in_memory(self, test_vec):
        """Check user-supplied vector object.
        
        Args:
            test_vec: a vector.
        
        See :py:meth:`vectorspace.VectorSpace.sanity_check_in_memory`.
        """
        self.vec_space.sanity_check_in_memory(test_vec_handle)


    def get_decomp(self, ritz_vals_source, mode_norms_source, 
        build_coeffs_source):
        """Retrieves the decomposition matrices from a source. """
        if self.get_mat is None:
            raise util.UndefinedError('Must specify a get_mat function')
        if parallel.is_rank_zero():
            self.ritz_vals = N.squeeze(N.array(self.get_mat(ritz_vals_source)))
            self.mode_norms = N.squeeze(N.array(self.get_mat(mode_norms_source)))
            self.build_coeffs = self.get_mat(build_coeffs_source)
        else:
            self.ritz_vals = None
            self.mode_norms = None
            self.build_coeffs = None
        if parallel.is_distributed():
            self.ritz_vals = parallel.comm.bcast(self.ritz_vals, root=0)
            self.mode_norms = parallel.comm.bcast(self.mode_norms, root=0)
            self.build_coeffs = parallel.comm.bcast(self.build_coeffs, root=0)
            
    def put_decomp(self, ritz_vals_dest, mode_norms_dest, build_coeffs_dest):
        """Puts the decomposition matrices in dest."""
        self.put_ritz_vals(ritz_vals_dest)
        self.put_mode_norms(mode_norms_dest)
        self.put_build_coeffs(build_coeffs_dest)
        
    def put_ritz_vals(self, dest):
        """Puts the Ritz values"""
        if self.put_mat is None and parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined, can't put")
        if parallel.is_rank_zero():
            self.put_mat(self.ritz_vals, dest)
        parallel.barrier()
        
    def put_mode_norms(self, dest):
        """Puts the mode norms"""
        if self.put_mat is None and parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined, can't put")
        if parallel.is_rank_zero():
            self.put_mat(self.mode_norms, dest)
        parallel.barrier()
        
    def put_build_coeffs(self, dest):
        """Puts the build coeffs"""
        if self.put_mat is None and parallel.is_rank_zero():
            raise util.UndefinedError("put_mat is undefined, can't put")
        if parallel.is_rank_zero():
            self.put_mat(self.build_coeffs, dest)
        parallel.barrier()
            
    def compute_decomp(self, vec_handles):
        """Computes decomposition and returns SVD matrices.
        
        Args:
            vec_handles: list of handles for the vecs.
                    
        Returns:
            vec_handles, ritz_vals, mode_norms, build_coeffs.
        """
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)
        if self.vec_handles is None:
            raise util.UndefinedError('vec_handles is not given')

        # Compute POD from vecs (excluding last vec)
        if self.POD is None:
            self.POD = pod.POD(inner_product=self.vec_space.inner_product, 
                max_vecs_per_node=self.vec_space.max_vecs_per_node, 
                verbosity=self.verbosity)
            # Don't use the returned mats, get them later from POD instance.
            dum, dum = self.POD.compute_decomp(
                vec_handles=self.vec_handles[:-1])
        elif self.vec_handles[:-1] != self.POD.vec_handles or \
            len(vec_handles) != len(self.POD.vec_handles)+1:
            raise RuntimeError('vec mismatch between POD and DMD '+\
                'objects.')
        pod_sing_vecs = self.POD.sing_vecs
        pod_sing_vals = self.POD.sing_vals
        _pod_sing_vals_sqrt_mat = N.mat(N.diag(pod_sing_vals** -0.5))


        # Inner product of vecs w/POD modes
        num_vecs = len(self.vec_handles)
        pod_modes_star_times_vecs = N.mat(N.empty((num_vecs-1, num_vecs-1)))
        pod_modes_star_times_vecs[:,:-1] = self.POD.correlation_mat[:,1:]  
        pod_modes_star_times_vecs[:,-1] = \
            self.vec_space.compute_inner_product_mat(self.vec_handles[:-1], 
                self.vec_handles[-1])
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
        
        ritz_vec_scaling = N.mat(N.diag(N.array(N.array(
            ritz_vec_scaling).squeeze(),ndmin=1)))

        # Compute mode energies
        self.build_coeffs = pod_sing_vecs * _pod_sing_vals_sqrt_mat *\
            low_order_eig_vecs * ritz_vec_scaling
        self.mode_norms = N.diag(self.build_coeffs.H * 
            self.POD.correlation_mat * self.build_coeffs).real
        return self.ritz_vals, self.mode_norms, self.build_coeffs
        
    def compute_decomp_in_memory(self, vecs):
        """Same as ``compute_decomp`` but takes vecs instead of handles."""
        self.vecs = util.make_list(vecs)
        vec_handles = [V.InMemoryVecHandle(v) for v in self.vecs]
        return self.compute_decomp(vec_handles)
        
    
    def compute_modes(self, mode_nums, mode_handles, vec_handles=None, 
        index_from=0):
        """Computes modes and calls ``mode_handle.put`` on them.
        
        Args:
            mode_nums: list of mode numbers, ``range(10)`` or ``[3, 2, 5]``.
            
            mode_handles: list of handles for modes.
            
        Kwargs:
            vec_handles: list of handles for vecs, can omit if given in
            :py:meth:`compute_decomp`.

            index_from: integer to start numbering modes from, 0, 1, or other.
        """
        if self.build_coeffs is None:
            raise util.UndefinedError('Must define self.build_coeffs')
        # User should specify ALL vecs, even though all but last are used
        if vec_handles is not None:
            self.vec_handles = util.make_list(vec_handles)
        self.vec_space.compute_modes(mode_nums, mode_handles, 
            self.vec_handles[:-1], self.build_coeffs, index_from=index_from)
        
    def compute_modes_in_memory(self, mode_nums, vecs=None, 
        index_from=0):
        """Computes modes and returns them.
        
        Args:
            mode_nums: list of mode numbers, ``range(10)`` or ``[3, 2, 5]``.
            
        Kwargs:
            vecs: list of handles for vecs, can omit if given in
            :py:meth:`compute_decomp`.

            index_from: integer to start numbering modes from, 0, 1, or other.
        
        Returns:
            a list of all modes.

        In parallel, each MPI worker returns all modes.
        See :py:meth:`compute_modes`.
        """
        if self.build_coeffs is None:
            raise util.UndefinedError('Must define self.build_coeffs')
        # User should specify ALL vecs, even though all but last are used
        if vecs is not None:
            self.vecs = util.make_list(vecs)
        return self.vec_space.compute_modes_in_memory(mode_nums, 
            self.vecs[:-1], self.build_coeffs, index_from=index_from)
 
 
