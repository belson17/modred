
import util
import numpy as N


class BPODROM(object):
    """
    Computes the ROM matrices from BPOD modes for an LTI plant.
    
    To use it, you must first use the BPOD class to form the BPOD modes.
    Then, before using this class, you must advance the direct modes
    forward in time by a time step dt. For a discrete time ROM, this will
    be the time step of the system. For a continuous time ROM, a first-order
    approximation of the derivative of the direct modes is made with
    d(mode)/dt = (mode(t=dt) - mode(t=0)) / dt, see compute_mode_derivs.
    
    Usage::

      myBPODROM = bpodltirom.BPODROM(...)
      # For continuous time systems
      myBPODROM.compute_mode_derivs(mode_paths, mode_dt_paths, mode_deriv_paths,1e-4)
      # For continuous time systems, set dt=0
      myBPODROM.form_A(A_Path, direct_mode_advanced_paths, adjoint_mode_paths, dt)
      myBPODROM.form_B(B_Path, input_paths, adjoint_mode_paths, dt)
      myBPODROM.form_C(C_Path, output_paths, direct_mode_paths)
    
    Eventually it should be made a derived class of
    a ROM class that computes Galerkin projections for generic PDEs. Each
    PDE would be a derived class
    """

    def __init__(self,adjoint_mode_paths=None,
      direct_mode_paths=None,
      direct_dt_mode_paths=None,  
      direct_deriv_mode_paths=None,
      discrete = None,
      dt = None,
      inner_product=None,
      save_mat=util.save_mat_text,
      get_field=None,
      put_field=None,
      num_modes=None,
      verbose = True,
      max_fields_per_node=2):
          self.dt = dt
          self.inner_product = inner_product
          self.save_mat = save_mat
          self.get_field = get_field
          self.put_field = put_field
          self.max_fields_per_node = max_fields_per_node
          self.max_fields_per_proc = self.max_fields_per_node
          self.num_modes = num_modes
          self.verbose = verbose
      
    def compute_mode_derivs(self,mode_paths,mode_dt_paths,mode_deriv_paths,dt):
        """
        Computes time derivatives of modes. dt=1e-4 is a good first choice.
        
        It reads in modes from mode_paths and mode_dt_paths, and simply
        subtracts them and divides by dt. This requires both __mul__
        and __add__ to be defined for the mode object.
        """
        if (self.get_field is None):
            raise util.UndefinedError('no get_field defined')
        if (self.put_field is None):
            raise util.UndefinedError('no put_field defined')
        
        num_modes = min(len(mode_paths),len(mode_dt_paths),len(mode_deriv_paths))
        print 'Computing derivatives of',num_modes,'modes'
        
        for mode_index in xrange(len(mode_dt_paths)):
            mode = self.get_field(mode_paths[mode_index])
            modeDt = self.get_field(mode_dt_paths[mode_index])
            self.put_field((modeDt - mode)*(1./dt), mode_deriv_paths[mode_index])
        
    
    def form_A(self, A_Path, direct_mode_advanced_paths, adjoint_mode_paths, dt, num_modes=None):
        """
        Computes the continouso or discrete time A matrix
        
        A_Path
          where the A matrix will be saved
        direct_mode_advanced_paths
          For a discrete time system, this is the 
          paths to the direct modes that have been advanced a time dt.
          For continuous time systems, this should be paths to the time
          derivatives of the direct modes.
        adjoint_mode_paths
          Paths to the adjoint modes
        dt
          The associated time step of the ROM, for continuous time it is 0.
        num_modes
          number of modes/states to keep in the ROM.
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
            self.num_modes = min(len(direct_mode_advanced_paths), len(adjoint_mode_paths))
        
        self.A = N.zeros((self.num_modes, self.num_modes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        num_rows_per_chunk = self.max_fields_per_proc - 1
        
        for start_row_num in range(0,self.num_modes, num_rows_per_chunk):
            end_row_num = min(start_row_num+num_rows_per_chunk, self.num_modes)
            #a list of 'row' adjoint modes (row because Y' has rows of adjoint snaps)
            adjoint_modes = [self.get_field(adjoint_path) \
                for adjoint_path in adjoint_mode_paths[start_row_num:end_row_num]] 
              
            #now read in each column (direct modes advanced dt or time deriv)
            for col_num,advanced_path in enumerate(direct_mode_advanced_paths[:self.num_modes]):
                advanced_mode = self.get_field(advanced_path)
                for row_num in range(start_row_num,end_row_num):
                  self.A[row_num,col_num] = \
                      self.inner_product(adjoint_modes[row_num-start_row_num], advanced_mode)

        self.save_mat(self.A, A_Path)
        if self.verbose:
            print '----- A matrix saved to',A_Path,'------'

      
    def form_B(self, B_Path, input_paths, adjoint_mode_paths, dt, num_modes=None):
        """
        Forms the B matrix, either continuous or discrete time.
        
        Computes inner products of adjoint mode with sensor inputs.
        
        B_Path 
          is where the B matrix will be saved
        input_paths 
          is a list of the actuator fields' files (spatial representation
          of the B matrix in the full system).
          THE ORDER IS IMPORTANT. The order of the input files determines 
          the order of the actuators in the ROM and must be kept track of.
        adjoint_mode_paths
          is a list of paths to the adjoint modes
        dt
          Set dt = 0 for continuous time systems.
        num_modes
          number of modes/states to keep in the ROM.
        """
        
        if self.dt is None:
            self.dt = dt
        elif self.dt != dt:
            print "WARNING: dt values are inconsistent, using new value",dt
            self.dt = dt
        
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(adjoint_mode_paths)
            
        num_inputs = len(input_paths)
        self.B = N.zeros((self.num_modes,num_inputs))
        
        num_rows_per_chunk = self.max_fields_per_proc - 1
        for start_row_num in range(0,self.num_modes,num_rows_per_chunk):
            end_row_num = min(start_row_num+num_rows_per_chunk,self.num_modes)
            #a list of 'row' adjoint modes
            adjoint_modes = [self.get_field(adjoint_path) \
                for adjoint_path in adjoint_mode_paths[start_row_num:end_row_num]] 
                
            #now read in each column (actuator fields)
            for col_num,input_path in enumerate(input_paths):
                input_field = self.get_field(input_path)
                for row_num in range(start_row_num, end_row_num):
                    self.B[row_num, col_num] = \
                      self.inner_product(adjoint_modes[row_num-start_row_num], input_field)
        
        if self.dt != 0:
            self.B *= self.dt
            
        self.save_mat(self.B, B_Path)
        if self.verbose:
            if self.dt!=0:
                print '----- B matrix, discrete-time, saved to',B_Path,'-----'
            else:
                print '----- B matrix, continuous-time, saved to',B_Path,'-----'
        
    
    def form_C(self, C_Path, output_paths, direct_mode_paths, num_modes=None):
        """
        Forms the C matrix, either continuous or discrete.
        
        Computes inner products of adjoint mode with sensor inputs.
        
        C_Path 
          is where the C matrix will be saved
        output_paths
          is a list of the senor fields' files (spatial representation
          of the C matrix in the full system).
          THE ORDER IS IMPORTANT. The order of the output files determines 
          the order of the sensors in the ROM and must be kept track of.
        direct_mode_paths 
          is a list of paths to the direct modes
        num_modes
          number of modes/states to keep in the ROM.
        
        Note: dt does not matter for the C matrix.
        """
        if num_modes is not None:
            self.num_modes = num_modes
        if self.num_modes is None:
            self.num_modes = len(direct_mode_paths)
        
        num_outputs = len(output_paths)
        self.C = N.zeros((num_outputs, self.num_modes))
        num_cols_per_chunk = self.max_fields_per_proc - 1
        
        for start_col_num in range(0,self.num_modes, num_cols_per_chunk):
            end_col_num = min(start_col_num+num_cols_per_chunk, self.num_modes)
            
            #a list of 'row' adjoint modes
            direct_modes = [self.get_field(direct_path) \
                for direct_path in direct_mode_paths[start_col_num:end_col_num]]
            
            #now read in each row (outputs)
            for row_num,output_path in enumerate(output_paths):
                output_field = self.get_field(output_path)
                for col_num in range(start_col_num, end_col_num):
                    self.C[row_num,col_num] = \
                        self.inner_product(output_field, direct_modes[col_num-start_col_num])      
  
        self.save_mat(self.C, C_Path)
        if self.verbose:
            print '----- C matrix saved to',C_Path,'-----'
    
    
