"""Class for finding BPOD models for LTI plants in parallel"""

import util
from vectors import InMemoryVecHandle
from vecoperations import VecOperations
import parallel as parallel_mod

class BPODROM(object):
    """Computes the ROM matrices from BPOD modes for an LTI plant.
    
    Kwargs:       
        inner_product: Function to take inner products.
        
        put_mat: Function to put a matrix elsewhere (memory or file).
        
        verbose: print more information about progress and warnings
        
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

    def __init__(self, inner_product=None, put_mat=util.save_array_text,
        verbose=True, max_vecs_per_node=10000):
        """Constructor"""
        self.inner_product = inner_product
        self.put_mat = put_mat
        self.vec_ops = VecOperations(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node, verbose=verbose)
        self.parallel = parallel_mod.default_instance
        self.num_modes = None
        self.verbose = verbose
        self.A = None
        self.B = None
        self.C = None

    def put_A(self, A_dest):
        """Put reduced A matrix to ``A_dest``"""
        if self.parallel.is_rank_zero():
            self.put_mat(self.A, A_dest)
    def put_B(self, B_dest):
        """Put reduced B matrix to ``B_dest``"""
        if self.parallel.is_rank_zero():
            self.put_mat(self.B, B_dest)
    def put_C(self, C_dest):
        """Put reduced C matrix to ``C_dest``"""
        if self.parallel.is_rank_zero():
            self.put_mat(self.C, C_dest)
            
    def put_model(self, A_dest, B_dest, C_dest):
        """Put reduced matrices A, B, and C to ``A_dest``, ``B_dest``, and
        ``C_dest``."""
        self.put_A(A_dest)
        self.put_B(B_dest)
        self.put_C(C_dest)
        
      
    def compute_derivs(self, vec_handles, adv_vec_handles, 
        deriv_vec_handles, dt):
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
        
        vec_index_tasks = self.parallel.find_assignments(range(num_vecs))[
            self.parallel.get_rank()]
        
        for i in vec_index_tasks:
            vec = vec_handles[i].get()
            vec_dt = adv_vec_handles[i].get()
            deriv_vec_handles[i].put((vec_dt - vec)*(1./dt))
        self.parallel.barrier()
    
    def compute_derivs_in_memory(self, vecs, adv_vecs, dt):
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
        self.compute_derivs(vec_handles, adv_vec_handles, deriv_vec_handles,
            dt)
        deriv_vecs = [v.get() for v in deriv_vec_handles]
        if self.parallel.is_distributed():
            # Remove empty entries            
            for i in range(deriv_vecs.count(None)):
                deriv_vecs.remove(None)
            all_deriv_vecs = util.flatten_list(
                self.parallel.comm.allgather(deriv_vecs))
            return all_deriv_vecs
        else:
            return deriv_vecs
    
    def compute_A(self, A_times_direct_modes_handles, 
        adjoint_mode_handles, num_modes=None):
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
        self.A = self.vec_ops.compute_inner_product_mat(
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
        self.B = self.vec_ops.compute_inner_product_mat(
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
        self.C = self.vec_ops.compute_inner_product_mat(
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
        