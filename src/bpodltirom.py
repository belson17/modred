"""Class for finding BPOD models for LTI plants in parallel"""

import util
from vectors import InMemoryVecHandle
from vectorspace import VectorSpace
import parallel as parallel_mod
parallel = parallel_mod.parallel_default_instance

def compute_derivs(vec_handles, adv_vec_handles, deriv_vec_handles, dt):
    """Computes 1st-order time derivatives of vectors. 
    
    Args:
        vec_handles: list of handles of vecs.
        
        adv_vec_handles: list of handles of vecs advanced ``dt`` in time.
        
        deriv_vec_handles: list of handles for time derivatives of vecs.
        
        dt: time step of ``adv_vec_handles``
        
    Computes d(vec)/dt = (vec(t=dt) - vec(t=0)) / dt.
    """
    num_vecs = len(vec_handles)
    if num_vecs != len(adv_vec_handles) or \
        num_vecs != len(deriv_vec_handles):
        raise RuntimeError('Number of vectors not equal')
    
    vec_index_tasks = parallel.find_assignments(range(num_vecs))[
        parallel.get_rank()]
    
    for i in vec_index_tasks:
        vec = vec_handles[i].get()
        vec_dt = adv_vec_handles[i].get()
        deriv_vec_handles[i].put((vec_dt - vec)*(1./dt))
    parallel.barrier()

def compute_derivs_in_memory(vecs, adv_vecs, dt):
    """Computes 1st-order time derivatives of vectors. 
    
    Args:
        vecs: list of vecs.
        
        adv_vecs: list of vecs advanced ``dt`` in time.
        
        dt: time step of ``adv_vecs``.
                
    Returns:
        deriv_vecs: list of time-derivs of vectors.
        
    In parallel, each processor returns all derivatives.
    """
    vec_handles = [InMemoryVecHandle(v) for v in vecs]
    adv_vec_handles = [InMemoryVecHandle(v) for v in adv_vecs]
    deriv_vec_handles = [InMemoryVecHandle() for i in xrange(len(adv_vecs))]
    compute_derivs(vec_handles, adv_vec_handles, deriv_vec_handles, dt)
    deriv_vecs = [v.get() for v in deriv_vec_handles]
    if parallel.is_distributed():
        # Remove empty entries            
        for i in range(deriv_vecs.count(None)):
            deriv_vecs.remove(None)
        all_deriv_vecs = util.flatten_list(
            parallel.comm.allgather(deriv_vecs))
        return all_deriv_vecs
    else:
        return deriv_vecs
    

class BPODROM(object):
    """Computes the ROM matrices from BPOD modes for an LTI plant.
    
    Kwargs:       
        inner_product: Function to take inner products.
        
        put_mat: Function to put a matrix elsewhere (memory or file).
        
        verbosity: 0 prints almost nothing, 1 prints progress and warnings
        
        max_vecs_per_node: max number of vectors in memory per node.
    
    First compute the BPOD modes with the BPOD class.
    Then, this class creates either discrete or continuous time models.
    
    To find a discrete time model, advance the direct modes forward in 
    time by a time step ``dt`` and set ``A_times_direct_modes_handles`` with
    the handles to the advanced direct modes.
    
    For a continuous time ROM, you have a few options.
    First, you can compute d(mode)/dt yourself.
    Or, you can advance the direct modes one time step and 
    approximate a first-order time derivative with method
    :py:meth:`compute_derivs`.
        
    Usage::

      myBPODROM = BPODROM(...)
      A = myBPODROM.compute_A(A_times_direct_mode_handles, adjoint_mode_handles)
      B = myBPODROM.compute_B(B_vec_handles, adjoint_mode_handles)
      C = myBPODROM.compute_C(C_vec_handles, direct_mode_handles)

    """

    def __init__(self, inner_product, put_mat=util.save_array_text,
        verbosity=1, max_vecs_per_node=10000):
        """Constructor"""
        self.inner_product = inner_product
        self.put_mat = put_mat
        self.vec_space = VectorSpace(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        parallel = parallel_mod.parallel_default_instance
        self.num_modes = None
        self.verbosity = verbosity
        self.A = None
        self.B = None
        self.C = None

    def put_A(self, A_dest):
        """Put reduced A matrix to ``A_dest``"""
        if parallel.is_rank_zero():
            self.put_mat(self.A, A_dest)
    def put_B(self, B_dest):
        """Put reduced B matrix to ``B_dest``"""
        if parallel.is_rank_zero():
            self.put_mat(self.B, B_dest)
    def put_C(self, C_dest):
        """Put reduced C matrix to ``C_dest``"""
        if parallel.is_rank_zero():
            self.put_mat(self.C, C_dest)
            
    def put_model(self, A_dest, B_dest, C_dest):
        """Put reduced matrices A, B, and C to ``A_dest``, ``B_dest``, and
        ``C_dest``."""
        self.put_A(A_dest)
        self.put_B(B_dest)
        self.put_C(C_dest)
        
    def compute_model(self, A_times_direct_modes_handles, B_vec_handles, 
        C_vec_handles, direct_mode_handles, adjoint_mode_handles, 
        num_modes=None):
        """Computes and returns the reduced-order model matrices.
        
            A_times_direct_modes_handles: list of handles to "A * direct modes"
                That is, the direct modes operated on by the full A matrix. 
                For a discrete time system, these are the 
                handles of the direct modes that have been advanced one
                time step.
                For continuous time systems, these are the handles of the
                time derivatives of the direct modes (see also 
                :py:meth:`compute_derivs`).
            
            B_vec_handles: list of handles to B vecs.
                These are spatial representations of the B matrix in the 
                full system.
            
            C_vec_handles: list of handles to C vecs.
                These are spatial representations of the C matrix in the full 
                system.
            
            direct_mode_handles: list of handles to the direct modes
            
            adjoint_mode_handles: list of handles to the adjoint modes
            
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given. Default is maximum possible.
        """
        self.compute_A(A_times_direct_modes_handles, adjoint_mode_handles,
            num_modes=num_modes)
        self.compute_B(B_vec_handles, adjoint_mode_handles, 
            num_modes=num_modes)
        self.compute_C(C_vec_handles, direct_mode_handles, 
            num_modes=num_modes)
        return self.A, self.B, self.C
        
    def compute_model_in_memory(self, A_times_direct_modes, 
        B_vecs, C_vecs, direct_modes, adjoint_modes, 
        num_modes=None):
        """See :py:meth:`compute_model`, but takes vecs instead of handles."""
        
        self.compute_A_in_memory(A_times_direct_modes, adjoint_modes,
            num_modes=num_modes)
        self.compute_B_in_memory(B_vecs, adjoint_modes, num_modes=num_modes)
        self.compute_C_in_memory(C_vecs, direct_modes, num_modes=num_modes)
        return self.A, self.B, self.C
        
    
    def compute_A(self, A_times_direct_modes_handles, adjoint_mode_handles, 
        num_modes=None):
        """Computes and returns the continous or discrete time A matrix.
        
        Args:
            A_times_direct_modes_handles: list of handles to "A * direct modes"
                That is, the direct modes operated on by the full A matrix. 
                For a discrete time system, these are the 
                handles of the direct modes that have been advanced one
                time step.
                For continuous time systems, these are the handles of the
                time derivatives of the direct modes (see also 
                :py:meth:`compute_derivs`).
            
            adjoint_mode_handles: list of handles to the adjoint modes
            
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given. Default is maximum possible.
        
        Returns:
            A: reduced A matrix.
        """
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = min(len(A_times_direct_modes_handles),
                len(adjoint_mode_handles))
        self.A = self.vec_space.compute_inner_product_mat(
            adjoint_mode_handles[:self.num_modes],
            A_times_direct_modes_handles[:self.num_modes])
        return self.A
        
        
        
    def compute_A_in_memory(self, A_times_direct_modes, 
        adjoint_modes, num_modes=None):
        """Computes and returns the continous or discrete time A matrix.
        
        Args:
            A_times_direct_modes: list of "A * direct modes"
                That is, the direct modes operated on by the full A matrix. 
                For a discrete time system, these are the 
                handles of the direct modes that have been advanced one
                time step.
                For continuous time systems, these are the handles of the
                time derivatives of the direct modes (see also 
                :py:meth:`compute_derivs`).
                
            adjoint_modes: list of adjoint modes
            
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given. Default is maximum possible.

        Returns:
            A: reduced A matrix

        See :py:meth:`compute_A`.
        """
        A_times_direct_modes_handles = [InMemoryVecHandle(v) for v in 
            A_times_direct_modes]
        adjoint_mode_handles = [InMemoryVecHandle(v) for v in 
            adjoint_modes]
        self.compute_A(A_times_direct_modes_handles, 
            adjoint_mode_handles, num_modes=num_modes)
        return self.A


    
    def compute_B(self, B_vec_handles, adjoint_mode_handles, num_modes=None):
        """Computes and returns the reduced B matrix.
        
        Args:		
            B_vec_handles: list of handles to B vecs.
                These are spatial representations of the B matrix in the 
                full system.
                
            adjoint_mode_handles: list of handles to the adjoint modes.
        		
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given.
		
		Returns:
		    B: Reduced B matrix
		    
        Computes inner products of adjoint modes with B vecs.
        
        Tip: If you found the modes via sampling IC responses to B and C*,
        then your impulse responses may be missing a factor of 1/dt
        (where dt is the first simulation time step).
        This can be remedied by multiplying the reduced B by dt.
        """
        # TODO: Check this description, then move to docstring
        #To see this dt effect, consider:
        #
        #dx/dt = Ax+Bu, approximate as (x^(k+1)-x^k)/dt = Ax^k + Bu^k.
        #Rearranging terms, x^(k+1) = (dt*I+A)x^k + dt*Bu^k.
        #The impulse response is: x^0=0, u^0=1, and u^k=0 for k>=1.
        #Thus x^1 = dt*B, x^2 = dt*(I+dt*A)*B, ...
        #and y^1 = dt*C*B, y^2 = dt*C*(I+dt*A)*B, ...
        #However, the impulse response to the true discrete-time system is
        #x^1 = B, x^2 = A_d*B, ...
        #and y^1 = CB, y^2 = CA_d*B, ...
        #(where I+dt*A ~ A_d)
        #The important thing to see is the factor of dt difference.

        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(adjoint_mode_handles)
        num_inputs = len(B_vec_handles)
        self.B = self.vec_space.compute_inner_product_mat(
            adjoint_mode_handles[:self.num_modes], B_vec_handles)
        return self.B

    def compute_B_in_memory(self, B_vecs, adjoint_modes, num_modes=None):
        """Computes and returns the reduced B matrix.
        
        Args:		
            B_vecs: list of B vecs.
                These are spatial representations of the B matrix in the 
                full system.
                
            adjoint_modes: list of adjoint modes.
        		
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given.
		
		Returns:
		    B: Reduced B matrix
		
		See :py:meth:`compute_B`.
        """
        B_vec_handles = [InMemoryVecHandle(v) for v in B_vecs]
        adjoint_mode_handles = [InMemoryVecHandle(v) for v in 
            adjoint_modes]
        self.compute_B(B_vec_handles, adjoint_mode_handles, num_modes=num_modes)
        return self.B


        
    def compute_C(self, C_vec_handles, direct_mode_handles, num_modes=None):
        """Computes and returns the reduced C matrix.
               
        Args: 
            C_vec_handles: list of handles to C vecs.
                These are spatial representations of the C matrix in the full 
                system.
                        
            direct_mode_handles: list of handles to the direct modes
        		
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given.
        
        Returns:
            Reduced C matrix
        
        Computes inner products of adjoint mode with C vecs.
        """
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(direct_mode_handles)
        self.C = self.vec_space.compute_inner_product_mat(
            C_vec_handles, direct_mode_handles[:self.num_modes])
        return self.C

    def compute_C_in_memory(self, C_vecs, direct_modes, num_modes=None):
        """Computes and returns the reduced C matrix.
               
        Args: 
            C_vec_handles: list of handles to C vecs.
                These are spatial representations of the C matrix in the full 
                system.
                        
            direct_mode_handles: list of handles to the direct modes
                
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given.
        
        Returns:
            Reduced C matrix
        
        Computes inner products of adjoint mode with C vecs.
        
        See :py:meth:`compute_C`.
        
        """
        C_vec_handles = [InMemoryVecHandle(v) for v in C_vecs]
        direct_mode_handles = [InMemoryVecHandle(v) for v in 
            direct_modes]
        self.compute_C(C_vec_handles, direct_mode_handles, 
            num_modes=num_modes)
        return self.C
        