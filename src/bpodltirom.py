"""Class for finding BPOD models for LTI plants.

Currently not parallelized."""

import util
import numpy as N
import vectors as V

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
    time by a time step ``dt`` and set ``A_full_direct_mode_handles`` with
    the handles to the advanced direct modes.
    
    For a continuous time ROM, you have a few options.
    First, you can compute d(mode)/dt yourself.
    Or, you can advance the direct modes one time step and 
    approximate a first-order time derivative with ``compute_derivs``.
        
    Usage::

      myBPODROM = BPODROM(...)
      myBPODROM.compute_A(A_dest, deriv_direct_mode_handles, adjoint_mode_handles)
      myBPODROM.compute_B(B_dest, input_vec_handles, adjoint_mode_handles)
      myBPODROM.compute_C(C_dest, output_vec_handles, direct_mode_handles)

    """

    def __init__(self, inner_product=None, put_mat=util.save_array_text,
        verbose=True, max_vecs_per_node=10000):
        """Constructor"""
        self.inner_product = inner_product
        self.put_mat = put_mat
        self.max_vecs_per_node = max_vecs_per_node
        self.max_vecs_per_proc = self.max_vecs_per_node
        self.num_modes = None
        self.verbose = verbose
        self.A = None
        self.B = None
        self.C = None

      
    def compute_derivs(self, vec_handles, vec_adv_handles, 
        vec_deriv_handles, dt):
        """Computes 1st-order time derivatives of vectors. 
        
        Args:
			vec_handles: list of handles of vecs.
			
			vec_dt_handles: list of handles of vecs advanced dt
			in time
			
			vec_deriv_handle: list of handles for time derivatives of vecs.
        
        Returns:
        	Outputs of vec_deriv_handles.put()
        
        Computes d(mode)/dt = (mode(t=dt) - mode(t=0)) / dt.
        """
        num_vecs = min(len(vec_handles), len(vec_adv_handles), 
            len(vec_deriv_dests))
            
        put_outputs = [None]*len(vec_adv_handles)
        for vec_index in xrange(len(vec_adv_handles)):
            vec = vec_handles[vec_index].get()
            vec_dt = vec_adv_handles[vec_index].get()
            put_outputs[vec_index] = vec_deriv_handles.put(
            	(vec_dt - vec)*(1./dt), vec_deriv_dests[vec_index])
        return put_outputs
    
    
    def compute_A(self, A_dest, A_full_direct_mode_handles, 
        adjoint_mode_handles, num_modes=None):
        """Forms the continous or discrete time A matrix.
        
        Args:
			A_dest: where the reduced A matrix will be put
			
			A_full_direct_mode_handles: list of handles to direct modes
			that have been operated on by the full A matrix. 
			For a discrete time system, these are the 
			handles of the direct modes that have been advanced one
			time step.
			For continuous time systems, these are the handles of the
			time derivatives of the direct modes (see also 
			``compute_derivs``).
	
			adjoint_mode_handles: list of handles to the adjoint modes
			
        Kwargs:
        		num_modes: number of modes/states to keep in the ROM. 
        		    Can omit if already given. Default is maximum possible.
        """
        self._compute_A(A_full_direct_mode_handles, 
            adjoint_mode_handles, num_modes=num_modes)
        self.put_mat(self.A, A_dest)
        
    def compute_A_and_return(self, A_full_direct_mode_handles, 
        adjoint_mode_handles, num_modes=None):
        """Forms the continous or discrete time A matrix and returns it
        
        See ``compute_A`` for details.
        
        Returns:
            A: reduced model matrix
        """
        self._compute_A(A_full_direct_mode_handles, 
            adjoint_mode_handles, num_modes=num_modes)
        return self.A
    
    def _compute_A(self, A_full_direct_mode_handles, 
        adjoint_mode_handles, num_modes=None):
        """Forms the continous or discrete time A matrix.
        
        See ``compute_A`` for details.
        
        TODO: Parallelize this if it ever becomes a bottleneck.
        """            
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = min(len(A_full_direct_mode_handles),
                len(adjoint_mode_handles))
        
        self.A = N.zeros((self.num_modes, self.num_modes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        num_rows_per_chunk = self.max_vecs_per_proc - 1
        
        for start_row_num in range(0, self.num_modes, num_rows_per_chunk):
            end_row_num = min(start_row_num+num_rows_per_chunk, self.num_modes)
            #a list of 'row' adjoint modes (row because Y' has rows 
            #   of adjoint vecs)
            adjoint_modes = [adjoint_handle.get() 
                for adjoint_handle in 
                adjoint_mode_handles[start_row_num:end_row_num]] 
              
            #now read in each column (direct modes advanced dt or time deriv)
            for col_num, advanced_handle in enumerate(
                A_full_direct_mode_handles[:self.num_modes]):
                advanced_mode = advanced_handle.get()
                for row_num in range(start_row_num, end_row_num):
                    self.A[row_num,col_num] = \
                        self.inner_product(
                            adjoint_modes[row_num - start_row_num],
                            advanced_mode)
    
    
    def compute_B(self, B_dest, input_vec_handles, adjoint_mode_handles, 
        num_modes=None):
        """Forms the B matrix.
        
        Computes inner products of adjoint modes with input vecs (B in Ax+Bu).
        
        Args:
			B_dest: where the reduced B matrix will be put
			
			input_vec_handles: list of handles to input vecs.
					These are spatial representations of the B matrix in the 
					full system.
					
			adjoint_mode_handles: list of handles to the adjoint modes
        		
        Kwargs:
			num_modes: number of modes/states to keep in the ROM. 
				Can omit if already given.
				
        Tip: If you found the modes via sampling IC responses to B and C*,
        then your impulse responses may be missing a factor of 1/dt
        (where dt is the first simulation time step).
        This can be remedied by multiplying the reduced B by dt.
        """
        self._compute_B(input_vec_handles, adjoint_mode_handles, 
            num_modes=num_modes)
        self.put_mat(self.B, B_dest)

    def compute_B_and_return(self, input_vec_handles, adjoint_mode_handles, 
        num_modes=None):
        """Forms the B matrix and returns it
        
        See ``compute_B`` for details.
        
        Returns:
            B matrix
        """
        self._compute_B(input_vec_handles, adjoint_mode_handles, 
            num_modes=num_modes)
        return self.B
      
    def _compute_B(self, input_vec_handles, adjoint_mode_handles, 
        num_modes=None):
        """Forms the B matrix
        
        See ``compute_B`` for details.
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
            
        num_inputs = len(input_vec_handles)
        self.B = N.zeros((self.num_modes, num_inputs))
        
        num_rows_per_chunk = self.max_vecs_per_proc - 1
        for start_row_num in range(0, self.num_modes, num_rows_per_chunk):
            end_row_num = min(start_row_num + num_rows_per_chunk,
                self.num_modes)
            # list of adjoint modes, which correspond to rows of B
            adjoint_modes = [adjoint_handle.get() \
                for adjoint_handle in 
                adjoint_mode_handles[start_row_num:end_row_num]] 
                
            # now get the input vecs, which correspond to columns of B
            for col_num, input_handle in enumerate(input_vec_handles):
                input_vec = input_handle.get()
                for row_num in range(start_row_num, end_row_num):
                    self.B[row_num, col_num] = \
                      self.inner_product(adjoint_modes[row_num-start_row_num],
                          input_vec)
    
    
    def compute_C_and_return(self, output_vec_handles, direct_mode_handles,
        num_modes=None):
        """Forms the C matrix and returns it
        
        See ``compute_C`` for details.
        
        Returns:
            C matrix
        """
        self._compute_C(output_vec_handles, direct_mode_handles, 
            num_modes=num_modes)
        return self.C
        
    
    def compute_C(self, C_dest, output_vec_handles, direct_mode_handles,
        num_modes=None):
        """Forms the C matrix, either continuous or discrete.
        
        Computes inner products of adjoint mode with output vecs (C in y=Cx).
        
        Args: 
			C_dest: where the reduced C matrix will be put.
			
			output_vec_handles: list of handles to output vecs.
				These are spatial representations of the C matrix in the full 
				system.
        				
          	direct_mode_handles: list of handles to the direct modes
        		
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given.
        """
        self._compute_C(output_vec_handles, direct_mode_handles,
            num_modes=num_modes)
        self.put_mat(self.C, C_dest)
    
    
    def _compute_C(self, output_vec_handles, direct_mode_handles,
        num_modes=None):
        """Forms the C matrix
        
        See ``compute_C`` for details.
        """
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(direct_mode_handles)
        
        num_outputs = len(output_vec_handles)
        self.C = N.zeros((num_outputs, self.num_modes))
        num_cols_per_chunk = self.max_vecs_per_proc - 1
        
        for start_col_num in range(0, self.num_modes, num_cols_per_chunk):
            end_col_num = min(start_col_num+num_cols_per_chunk, self.num_modes)
            
            # list of direct modes, which correspond to columns of C
            direct_modes = [direct_handle.get() for direct_handle in 
                direct_mode_handles[start_col_num:end_col_num]]
            
            # get the output vecs, which correspond to rows of C
            for row_num, output_handle in enumerate(output_vec_handles):
                output_vec = output_handle.get()
                for col_num in range(start_col_num, end_col_num):
                    self.C[row_num,col_num] = \
                        self.inner_product(output_vec, 
                            direct_modes[col_num-start_col_num])      
    
