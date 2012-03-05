
import util
import numpy as N


class BPODROM(object):
    """Computes the ROM matrices from BPOD modes for an LTI plant.
    
    To use it, you must first have BPOD modes.
    Then, you must advance the direct modes forward in time by a time step dt.
    For a discrete time ROM, this will be the time step of the system. 
    For a continuous time ROM, a first-order approximation of the derivative of
    the direct modes is made with
    d(mode)/dt = (mode(t=dt) - mode(t=0)) / dt, see ``self.compute_mode_derivs``.
    
    Usage::

      myBPODROM = BPODROM(...)
      # For continuous time systems
      myBPODROM.compute_mode_derivs(mode_sources, mode_dt_sources, mode_deriv_sources,1e-4)
      # For continuous time systems, set dt=0
      myBPODROM.form_A(A_dest, direct_mode_advanced_sources, adjoint_mode_sources, dt)
      myBPODROM.form_B(B_dest, input_field_sources, adjoint_mode_sources, dt)
      myBPODROM.form_C(C_dest, output_field_sources, direct_mode_sources)
    
    """

    def __init__(self,adjoint_mode_sources=None,
      direct_mode_sources=None,
      direct_dt_mode_sources=None,  
      direct_deriv_mode_sources=None,
      discrete = None,
      dt = None,
      inner_product=None,
      put_mat=util.save_mat_text,
      get_field=None,
      put_field=None,
      num_modes=None,
      verbose = True,
      max_fields_per_node=2):
          self.dt = dt
          self.inner_product = inner_product
          self.put_mat = put_mat
          self.get_field = get_field
          self.put_field = put_field
          self.max_fields_per_node = max_fields_per_node
          self.max_fields_per_proc = self.max_fields_per_node
          self.num_modes = num_modes
          self.verbose = verbose
      
    def compute_mode_derivs(self,mode_sources, mode_dt_sources, mode_deriv_dests, dt):
        """Computes 1st-order time derivatives of modes. 
        
        Args:
        		mode_sources: list of sources of direct modes.
        		
        		mode_dt_sources: list of sources of direct modes advanced dt in time
        		
        		mode_deriv_dests: list of destinations for time derivatives of direct modes.
        		
        Requires both __mul__ and __add__ to be defined for the mode object.
        """
        if (self.get_field is None):
            raise util.UndefinedError('no get_field defined')
        if (self.put_field is None):
            raise util.UndefinedError('no put_field defined')
        
        num_modes = min(len(mode_sources),len(mode_dt_sources),len(mode_deriv_dests))
        if self.verbose:
		        print 'Computing derivatives of',num_modes,'modes'
        
        for mode_index in xrange(len(mode_dt_sources)):
            mode = self.get_field(mode_sources[mode_index])
            modeDt = self.get_field(mode_dt_sources[mode_index])
            self.put_field((modeDt - mode)*(1./dt), mode_deriv_dests[mode_index])
        
    
    def form_A(self, A_dest, direct_mode_advanced_sources, adjoint_mode_sources, dt,
    		num_modes=None):
        """
        Computes the continous or discrete time A matrix
        
        Args:
		        A_dest: where the reduced A matrix will be saved
		        
		        direct_mode_advanced_sources: list of sources of direct modes advanced. 
								For a discrete time system, these are the 
								sources of the direct modes that have been advanced a time dt.
								For continuous time systems, these are the sources of the time
								derivatives of the direct modes.
        
        		adjoint_mode_sources: list of sources to the adjoint modes
        		
        		dt: The time step of the ROM, for continuous time it is 0.
        
        Kwargs:
        		num_modes: number of modes/states to keep in the ROM. Can omit if already given.
        		
        TODO: Parallelize this if it ever becomes a bottleneck.
        """
        self.dt = dt
        if self.verbose:
            if self.dt == 0:
                print 'Computing the continuous-time A matrix'
            else:
                print 'Computing the discrete-time A matrix with time step',self.dt
        
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = min(len(direct_mode_advanced_sources), len(adjoint_mode_sources))
        
        self.A = N.zeros((self.num_modes, self.num_modes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        num_rows_per_chunk = self.max_fields_per_proc - 1
        
        for start_row_num in range(0,self.num_modes, num_rows_per_chunk):
            end_row_num = min(start_row_num+num_rows_per_chunk, self.num_modes)
            #a list of 'row' adjoint modes (row because Y' has rows of adjoint snaps)
            adjoint_modes = [self.get_field(adjoint_source) \
                for adjoint_source in adjoint_mode_sources[start_row_num:end_row_num]] 
              
            #now read in each column (direct modes advanced dt or time deriv)
            for col_num,advanced_source in enumerate(direct_mode_advanced_sources[:self.num_modes]):
                advanced_mode = self.get_field(advanced_source)
                for row_num in range(start_row_num,end_row_num):
                  self.A[row_num,col_num] = \
                      self.inner_product(adjoint_modes[row_num-start_row_num], advanced_mode)

        self.put_mat(self.A, A_dest)
        if self.verbose:
            print '----- A matrix put to',A_dest,'------'

      
    def form_B(self, B_dest, input_field_sources, adjoint_mode_sources, dt, num_modes=None):
        """Forms the B matrix, either continuous or discrete time.
        
        Computes inner products of adjoint mode with input fields (B in Ax+Bu).
        
        Args:
		        B_dest: where the reduced B matrix will be put (memory or file)
		        
        		input_field_sources: list of input fields' sources.
        				These are spatial representations of the B matrix in the full system.
        				
          	adjoint_mode_sources: list of sources to the adjoint modes
        		
        		dt: discrete time step. Set dt = 0 for continuous time systems.
        
        Kwargs:
        		num_modes: number of modes/states to keep in the ROM. Can omit if already given.
        """
        
        if self.dt is None:
            self.dt = dt
        elif self.dt != dt:
            print "WARNING: dt values are inconsistent, using new value",dt
            self.dt = dt
        
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(adjoint_mode_sources)
            
        num_inputs = len(input_field_sources)
        self.B = N.zeros((self.num_modes,num_inputs))
        
        num_rows_per_chunk = self.max_fields_per_proc - 1
        for start_row_num in range(0,self.num_modes,num_rows_per_chunk):
            end_row_num = min(start_row_num+num_rows_per_chunk,self.num_modes)
            #a list of 'row' adjoint modes
            adjoint_modes = [self.get_field(adjoint_source) \
                for adjoint_source in adjoint_mode_sources[start_row_num:end_row_num]] 
                
            #now read in each column (actuator fields)
            for col_num,input_source in enumerate(input_field_sources):
                input_field = self.get_field(input_source)
                for row_num in range(start_row_num, end_row_num):
                    self.B[row_num, col_num] = \
                      self.inner_product(adjoint_modes[row_num-start_row_num], input_field)
        
        if self.dt != 0:
            self.B *= self.dt
            
        self.put_mat(self.B, B_dest)
        if self.verbose:
            if self.dt!=0:
                print '----- B matrix, discrete-time, put to',B_dest,'-----'
            else:
                print '----- B matrix, continuous-time, put to',B_dest,'-----'
        
    
    def form_C(self, C_dest, output_field_sources, direct_mode_sources, num_modes=None):
        """Forms the C matrix, either continuous or discrete.
        
        Computes inner products of adjoint mode with output fields (C in y=Cx).
        
        Args:
		        C_dest: where the reduced C matrix will be put (memory or file)
		        
        		output_field_sources: list of output fields' sources.
        				These are spatial representations of the C matrix in the full system.
        				
          	direct_mode_sources: list of sources to the direct modes
        		
        Kwargs:
            num_modes: number of modes/states to keep in the ROM. Can omit if already given.
        
        (Time step dt is irrelevant for the C matrix.)
        """
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(direct_mode_sources)
        
        num_outputs = len(output_field_sources)
        self.C = N.zeros((num_outputs, self.num_modes))
        num_cols_per_chunk = self.max_fields_per_proc - 1
        
        for start_col_num in range(0,self.num_modes, num_cols_per_chunk):
            end_col_num = min(start_col_num+num_cols_per_chunk, self.num_modes)
            
            #a list of 'column' direct modes
            direct_modes = [self.get_field(direct_source) \
                for direct_source in direct_mode_sources[start_col_num:end_col_num]]
            
            #now read in each row (outputs)
            for row_num,output_source in enumerate(output_field_sources):
                output_field = self.get_field(output_source)
                for col_num in range(start_col_num, end_col_num):
                    self.C[row_num,col_num] = \
                        self.inner_product(output_field, direct_modes[col_num-start_col_num])      
  
        self.put_mat(self.C, C_dest)
        if self.verbose:
            print '----- C matrix put to',C_dest,'-----'
    
    
