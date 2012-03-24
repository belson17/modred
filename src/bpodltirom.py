"""Class for finding BPOD models for LTI plants.

Currently not parallelized."""

import util
import numpy as N

class BPODROM(object):
    """Computes the ROM matrices from BPOD modes for an LTI plant.
    
    First compute the BPOD modes with the BPOD class.
    Then, you can create either discrete or continuous time models.
    Usage::

      myBPODROM = BPODROM(...)
      myBPODROM.form_A(A_dest, deriv_direct_mode_sources, adjoint_mode_sources)
      myBPODROM.form_B(B_dest, input_vec_sources, adjoint_mode_sources)
      myBPODROM.form_C(C_dest, output_vec_sources, direct_mode_sources)

    To find a discrete time model, advance the direct modes forward in 
    time by a time step dt and replace ``deriv_direct_mode_sources`` with
    the sources to the advanced direct modes.
    
    For a continuous time ROM, you have a few options.
    First, you can compute d(mode)/dt yourself.
    Or, you can advance the direct modes one time step and 
    approximate a first-order time derivative with ``compute_derivs``.
    """

    def __init__(self, inner_product=None, put_mat=util.save_mat_text,
        get_vec=None, put_vec=None, verbose=True, max_vecs_per_node=2):
        """Constructor"""
        self.inner_product = inner_product
        self.put_mat = put_mat
        self.get_vec = get_vec
        self.put_vec = put_vec
        self.max_vecs_per_node = max_vecs_per_node
        self.max_vecs_per_proc = self.max_vecs_per_node
        self.num_modes = None
        self.verbose = verbose
        self.dt = None
        self.A = None
        self.B = None
        self.C = None

      
    def compute_derivs(self, vec_sources, vec_adv_sources, 
        vec_deriv_dests, dt):
        """Computes 1st-order time derivatives of vectors. 
        
        Args:
        		vec_sources: list of sources of vecs.
        		
        		vec_dt_sources: list of sources of vecs advanced dt
        		in time
        		
        		vec_deriv_dests: list of destinations for time derivatives of
        		vecs.
        
        Computes d(mode)/dt = (mode(t=dt) - mode(t=0)) / dt.
        """
        if (self.get_vec is None):
            raise util.UndefinedError('no get_vec defined')
        if (self.put_vec is None):
            raise util.UndefinedError('no put_vec defined')
        
        num_vecs = min(len(vec_sources), len(vec_adv_sources), 
            len(vec_deriv_dests))
       
        for vec_index in xrange(len(vec_adv_sources)):
            vec = self.get_vec(vec_sources[vec_index])
            vec_dt = self.get_vec(vec_adv_sources[vec_index])
            self.put_vec((vec_dt - vec)*(1./dt), vec_deriv_dests[vec_index])
            
    
    def compute_A(self, A_dest, A_full_direct_mode_sources, 
        adjoint_mode_sources, num_modes=None):
        """Form the continous or discrete time A matrix.
        
        Args:
		        A_dest: where the reduced A matrix will be put
		        
		        A_full_direct_mode_sources: list of sources to direct modes
		        that have been operated on by the full A matrix. 
		            For a discrete time system, these are the 
								sources of the direct modes that have been advanced one
								time step.
								For continuous time systems, these are the sources of the
								time derivatives of the direct modes (see also 
								``compute_derivs``).
        
        		adjoint_mode_sources: list of sources to the adjoint modes
        		
        Kwargs:
        		num_modes: number of modes/states to keep in the ROM. 
        		    Can omit if already given. Default is maximum possible.
        		
        		
        TODO: Parallelize this if it ever becomes a bottleneck.
        """            
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = min(len(A_full_direct_mode_sources),
                len(adjoint_mode_sources))
        
        self.A = N.zeros((self.num_modes, self.num_modes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        num_rows_per_chunk = self.max_vecs_per_proc - 1
        
        for start_row_num in range(0, self.num_modes, num_rows_per_chunk):
            end_row_num = min(start_row_num+num_rows_per_chunk, self.num_modes)
            #a list of 'row' adjoint modes (row because Y' has rows 
            #   of adjoint vecs)
            adjoint_modes = [self.get_vec(adjoint_source) \
                for adjoint_source in 
                adjoint_mode_sources[start_row_num:end_row_num]] 
              
            #now read in each column (direct modes advanced dt or time deriv)
            for col_num, advanced_source in enumerate(
                A_full_direct_mode_sources[:self.num_modes]):
                advanced_mode = self.get_vec(advanced_source)
                for row_num in range(start_row_num, end_row_num):
                    self.A[row_num,col_num] = \
                        self.inner_product(
                            adjoint_modes[row_num - start_row_num],
                            advanced_mode)

        self.put_mat(self.A, A_dest)
        if self.verbose:
            print 'A matrix put to %s'%A_dest

      
    def compute_B(self, B_dest, input_vec_sources, adjoint_mode_sources, 
        num_modes=None):
        """Forms the B matrix.
        
        Computes inner products of adjoint modes with input vecs (B in Ax+Bu).
        
        Args:
		        B_dest: where the reduced B matrix will be put
		        
        		input_vec_sources: list of sources to input vecs.
        				These are spatial representations of the B matrix in the 
        				full system.
        				
          	adjoint_mode_sources: list of sources to the adjoint modes
        		
        Kwargs:
        		num_modes: number of modes/states to keep in the ROM. 
        		    Can omit if already given.
        		    
        Tip: If you found the modes via IC responses to B and C*,
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
            self.num_modes = len(adjoint_mode_sources)
            
        num_inputs = len(input_vec_sources)
        self.B = N.zeros((self.num_modes, num_inputs))
        
        num_rows_per_chunk = self.max_vecs_per_proc - 1
        for start_row_num in range(0, self.num_modes, num_rows_per_chunk):
            end_row_num = min(start_row_num + num_rows_per_chunk,
                self.num_modes)
            # list of adjoint modes, which correspond to rows of B
            adjoint_modes = [self.get_vec(adjoint_source) \
                for adjoint_source in 
                adjoint_mode_sources[start_row_num:end_row_num]] 
                
            # now get the input vecs, which correspond to columns of B
            for col_num, input_source in enumerate(input_vec_sources):
                input_vec = self.get_vec(input_source)
                for row_num in range(start_row_num, end_row_num):
                    self.B[row_num, col_num] = \
                      self.inner_product(adjoint_modes[row_num-start_row_num],
                          input_vec)
        
        #if self.dt != 0:
        #    self.B *= self.dt
            
        self.put_mat(self.B, B_dest)
        if self.verbose:
            print 'B matrix put to', B_dest
        
    
    def compute_C(self, C_dest, output_vec_sources, direct_mode_sources,
        num_modes=None):
        """Forms the C matrix, either continuous or discrete.
        
        Computes inner products of adjoint mode with output vecs (C in y=Cx).
        
        Args: 
		        C_dest: where the reduced C matrix will be put.
		        
        		output_vec_sources: list of sources to output vecs.
        				These are spatial representations of the C matrix in the full 
        				system.
        				
          	direct_mode_sources: list of sources to the direct modes
        		
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. 
                Can omit if already given.
        """
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(direct_mode_sources)
        
        num_outputs = len(output_vec_sources)
        self.C = N.zeros((num_outputs, self.num_modes))
        num_cols_per_chunk = self.max_vecs_per_proc - 1
        
        for start_col_num in range(0, self.num_modes, num_cols_per_chunk):
            end_col_num = min(start_col_num+num_cols_per_chunk, self.num_modes)
            
            # list of direct modes, which correspond to columns of C
            direct_modes = [self.get_vec(direct_source) \
                for direct_source in 
                direct_mode_sources[start_col_num:end_col_num]]
            
            # get the output vecs, which correspond to rows of C
            for row_num, output_source in enumerate(output_vec_sources):
                output_vec = self.get_vec(output_source)
                for col_num in range(start_col_num, end_col_num):
                    self.C[row_num,col_num] = \
                        self.inner_product(output_vec, 
                            direct_modes[col_num-start_col_num])      
  
        self.put_mat(self.C, C_dest)
        if self.verbose:
            print 'C matrix put to %s'%C_dest
    
    
