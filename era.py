
import numpy as N
import util

class ERA(object):
    """Forms ERA ROM for discrete-time system from impulse output data 

    The simplest way to use this class is ::
    
      output_paths =['/path/in1ToOuts.txt', '/path/in2ToOuts.txt']
      myERA = ERA()
      time_values, outputs = util.load_impulse_outputs(output_paths)
      myERA.set_impulse_outputs(time_values, outputs)
      myERA.compute_decomp()
      myERA.save_decomp('H.txt','H2.txt','U.txt','E.txt','V.txt')
      myERA.compute_ROM(50)
      myERA.save_ROM('A.txt','B.txt','C.txt')
      
    This would generate a 50-state LTI ROM with A,B,C matrices saved in text
    format.
    
    The above usage uses util.load_impulse_outputs, which assumes a 
    format of text files (see documentation for this function).
    Alternatively, the impulse response output signals can be set directly
    with set_impulse_outputs. For details, see documentation for set_impulse_outputs.
    
    The output times must be in one of two formats::
    
      t0 + dt_system*[0, 1, P, P+1, 2P, 2P+1, ... ]
    
    Or a special case where P=2::
    
      t0 + dt_system*[0, 1, 2, 3, 4, ... ]
    
    Currently *only the second format* appears to give balanced models with good accuracy.
    It's possible that the first format does as well, but we aren't sure.
    For now, it's recommended to use only the special case format, P=2.
    
    Not intended to be parallelized for distributed memory.
    
    See Ma et al. 2011 TCFD for more details.
    """
    
    def __init__(self, save_mat=util.save_mat_text, load_mat=util.load_mat_text, mc=None, 
        mo=None, verbose=True):
        """
        Construct an ERA class. 
        
        Variables mc and mo are the number of Markov parameters to use for the Hankel matrix.
        The default is to use all the data and set mc = mo.
        """
        self.save_mat = save_mat
        self.load_mat = load_mat
        self.mc = mc
        self.mo = mo
        self.outputs = None
        self.verbose = verbose
        # Used for error checking and determing file formats
        self.dt_tol = 1e-6
     
    def set_impulse_outputs(self, time_values, outputs):
        """
        Set the signals from each output to each input.
        
        Input arguments:
        outputs: 
          3D array of Markov parameters with indices [time_interval#, output#, input#],
          so that outputs[snap_num] is the Markov parameter C A**i B at that time interval number.
          outputs = [CB, CAB, CA**PB, CA**(P+1)B, ...]   OR 
          output = [CB, CAB, CA**2B, ...]
        
        time_values: 
          corresponding to outputs
          For sequence 1, this would be dt*[0, 1, P, P+1, 2P, 2P+1, ...].
          For sequence 2, this would be dt*[0, 1, 2, 3, 4, ...].
        
        This function is particularly useful if the data is not saved in the default
        format (which can be loaded with self.load_impulse_outputs).
        
        Sets self.outputs, self.time_steps, and self.dt_system.
        """
        self.time_steps, self.outputs, self.dt_system = \
            self._correct_format(time_values, outputs[:time_values.shape[0]]) 
        
        
    """ 
    Moved to util, since it isn't a necessary component of ERA.
    It's a convenience.
    Not sure what the right place for it is though.
    
    def load_impulse_outputs(self, output_paths):
        
        Loads impulse outputs with format [t out1 out2 ...].
        
        The output times must be in one of two formats:
        t0, t0+dt_system, t0+dt_sample, t0+dt_sample+dt_system, t0+2*dt_sample, ...
        Or, if dt_system and dt_sample are equal:
        t0, t0+dt_sample, t0+2*dt_sample, ...
        
        output_paths is a list of the files, each corresponding to a particular
        input's impulse response.
        
        num_inputs = len(output_paths)
        
        # Read the first file to get parameters
        raw_data = self.load_mat(output_paths[0])
        num_outputs = raw_data.shape[1]-1
        if num_outputs == 0:
            raise ValueError('Data must have at least two columns')
        time_values = raw_data[:, 0]
        num_time_values = len(time_values)
        
        # Now allocate array and read all of the output data
        outputs_read = N.zeros((num_time_values, num_outputs, num_inputs))
        
        # Load all of the outputs, make sure time_values match for each input impulse file
        for input_num,output_path in enumerate(output_paths):
            raw_data = self.load_mat(output_path)
            time_values_read = raw_data[:,0]
            if N.amax(N.abs(time_values_read - time_values)) > self.dt_tol:
                raise ValueError('Time values in %s are inconsistent with other files'%output_path)   
            outputs_read[:,:,input_num] = raw_data[:,1:]
                
        self.time_steps, self.outputs, self.dt_system = self._correct_format(time_values, 
            outputs_read)
    """           
    
    
    def compute_decomp(self, mc=None, mo=None):
        """
        Assembles the Hankel matrices from outputs and takes SVD
        
        Assumes that the impulse output signals are set as self.outputs
        L_sing_vecs*N.mat(N.diag(sing_vals))*R_sing_vecs.H = Hankel_mat
        """
        if mc is not None: self.mc = mc
        if mo is not None: self.mo = mo
        if self.outputs is None:
            raise util.UndefinedError('No output impulse data exists in instance')
               
        self._assemble_Hankel()
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = util.svd(self.Hankel_mat) 
    
       
        
    def compute_ROM(self, num_states, num_inputs=None, num_outputs=None, dt_system=None):
        """
        Computes the A,B,C LTI ROM matrices from the SVD matrices.
        
        num_states:
          the number of states to be found for the ROM.
        If using loaded SVD data, then must give the num_inputs, num_outputs, and dt_system.
        If using impulse response data, this is unnecessary.
        
        For discrete time systems the, impulse is applied over a time interval dt_system and so
        has a time-integral 1*dt_system rather than 1. This means the B matrix is "off" by a
        factor of dt_system. This is accounted for by multiplying B by dt_system.
        """
      
        if self.outputs is not None:
            num_time_steps, num_outputs, num_inputs = self.outputs.shape
        elif dt_system is not None:
            self.dt_system = dt_system
        elif num_outputs is None or num_inputs is None or self.dt_system is None:
            raise util.UndefinedError('Specify number of outputs, inputs, and dt_system')
        
        # Truncated matrices
        Ur = N.mat(self.L_sing_vecs[:,:num_states])
        Er = N.squeeze(self.sing_vals[:num_states])
        Vr = N.mat(self.R_sing_vecs[:,:num_states])
        
        self.A = N.mat(N.diag(Er**-.5)) * Ur.H * self.Hankel_mat2 * Vr * N.mat(N.diag(Er**-.5))
        self.B = (N.mat(N.diag(Er**.5)) * (Vr.H)[:,:num_inputs]) * self.dt_system 
        # !! NEED * dt above!!
        self.C = Ur[:num_outputs,:] * N.mat(N.diag(Er**.5))
        
        if (N.linalg.eig(self.A)[0] >= 1.).any() and self.verbose:
            print 'Warning: Unstable eigenvalues of ROM matrix A'

 
 
    def save_decomp(self, Hankel_mat_path, Hankel_mat2_path, L_sing_vecs_path, sing_valsPath,
        R_sing_vecs_path):
        """Saves the decomposition matrices"""
        self.save_mat(self.Hankel_mat, Hankel_mat_path)
        self.save_mat(self.Hankel_mat2,Hankel_mat2_path)
        self.save_mat(self.L_sing_vecs,L_sing_vecs_path)
        self.save_mat(self.sing_vals,sing_valsPath)
        self.save_mat(self.R_sing_vecs,R_sing_vecs_path)
 
 
        
    def load_decomp(self, Hankel_mat_path, Hankel_mat2_path, L_sing_vecs_path, sing_valsPath,
        R_sing_vecs_path):
        """Loads the decomposition matrices, computes SVD if necessary"""
        self.L_sing_vecs=self.load_mat(L_sing_vecs_path)
        self.R_sing_vecs=self.load_mat(R_sing_vecs_path)    
        self.sing_vals=self.load_mat(sing_valsPath)    
        self.Hankel_mat=self.load_mat(Hankel_mat_path)
        self.Hankel_mat2=self.load_mat(Hankel_mat2_path)
        if self.Hankel_mat.shape != self.Hankel_mat2.shape:
            raise RuntimeError('Sizes of Hankel and Hankel2 matrices differ')
                
 
 
    def save_ROM(self, APath, BPath, CPath):
        """Saves the A, B, and C LTI matrices to file"""  
        self.save_mat(self.A, APath)
        self.save_mat(self.B, BPath)
        self.save_mat(self.C, CPath)
        print 'Saved ROM matrices to:'
        print APath
        print BPath
        print CPath
    
 
 
    def _assemble_Hankel(self):
        """
        Assembles and sets self.Hankel_mat and self.Hankel_mat2 (H and H' in Ma 2011).        
        
        Chooses mc and mo to be equal and maximal for number of time steps.
        Consider (let sample_interval=P=1 for clarity) sequences
        [0 1 1 2 2 3] => H = [[0 1][1 2]]   H2 = [[1 2][2 3]], mc=mo=1, nt=6
        [0 1 1 2 2 3 3 4 4 5] => H=[[0 1 2][1 2 3][2 3 4]] H2=[[1 2 3][2 3 4][3 4 5]], mc=mo=2,nt=10
        Thus, using all available data means mc=mo=(nt-2)/4.
        For additional output times, (nt-2)/4 rounds down, so we use this formula.
        """
        num_time_steps, num_outputs, num_inputs = N.shape(self.outputs)
        assert(num_time_steps%2==0)
        
        if self.mo is None or self.mc is None:
            # Set mo and mc, time_steps is always in format [0, 1, P, P+1, ...]
            self.mo = (num_time_steps-2)/4
            self.mc = self.mo
                
        if (self.mo + self.mc) +2 > num_time_steps:
            raise ValueError('mo+mc+2=%d and must be <= than the number of samples %d'%(
                self.mo+self.mc+2, num_time_steps))
        
        self.Hankel_mat = N.zeros((num_outputs*self.mo, num_inputs*self.mc))
        self.Hankel_mat2 = N.zeros(self.Hankel_mat.shape)
        #outputs_flattened = self.outputs.swapaxes(0,1).reshape((num_outputs, -1))
        for row in range(self.mo):
            row_start = row*num_outputs
            row_end = row_start + num_outputs
            for col in range(self.mc):
                col_start = col*num_inputs
                col_end = col_start + num_inputs
                self.Hankel_mat[row_start:row_end, col_start:col_end] = self.outputs[2*(row+col)]
                self.Hankel_mat2[row_start:row_end, col_start:col_end] = self.outputs[2*(row+col)+1]
                
        
        
    def _correct_format(self, time_values, outputs):
        """
        Returns time_steps and outputs in format [CB, CAB, CA**P, CA**(P+1)B, ...]
        
        This converts the case of [CB, CAB, CA**2B, ...] into
        the general form, ie [CB, CAB, CA**PB, CA**(P+1)B, CA**(2P)B, ...].
        This means there can be repeated entries. 
        """
        if time_values.ndim != 1:
            raise RuntimeError('time_values must be a 1D array')
        
        ndims = outputs.ndim
        if ndims == 1:
            print 'Warning: Reshaping outputs from 1D array to 3D'
            outputs = outputs.reshape((outputs.shape[0],1,1))
        elif ndims == 2:
            print 'Warning: Reshaping outputs from 2D array to 3D'
            outputs = outputs.reshape((outputs.shape[0],outputs.shape[1],1))
        elif ndims > 3: 
            raise RuntimeError('outputs can have 1, 2, or 3 dimn, yours had %d'%ndims)
            
        num_time_steps, num_outputs, num_inputs = outputs.shape
        if num_time_steps != time_values.shape[0]:
            raise RuntimeError('Size of time and output arrays differ')

        dt_system = time_values[1] - time_values[0]
        dt_sample = time_values[2] - time_values[0]
        time_steps = N.round((time_values-time_values[0])/dt_system)
        
        if N.abs(2*dt_system - dt_sample) < self.dt_tol:
            # Format [0, 1, 2, 3, ...], requires conversion
            num_time_steps_corr = (num_time_steps - 1)*2
            outputs_corr = N.zeros((num_time_steps_corr, num_outputs, num_inputs))
            time_steps_corr = N.zeros(num_time_steps_corr)
            outputs_corr[::2] = outputs[:-1]
            outputs_corr[1::2] = outputs[1:]
            time_steps_corr[::2] = time_steps[:-1]
            time_steps_corr[1::2] = time_steps[1:]
            true_time_steps = N.zeros(num_time_steps_corr)
            true_time_steps[::2] = N.arange(num_time_steps_corr/2)
            true_time_steps[1::2] = 1 + N.arange(num_time_steps_corr/2)
        else:
            # Format [0, 1, P, P+1, ...]
            # Make number of time steps even, using integer division
            num_time_steps_corr = (num_time_steps/2) * 2
            time_steps_corr = time_steps[:num_time_steps_corr]
            outputs_corr = outputs[:num_time_steps_corr]
            true_time_steps = N.zeros(num_time_steps_corr)
            true_time_steps[::2] = round(dt_sample/dt_system) * N.arange(num_time_steps_corr/2)
            true_time_steps[1::2] = 1 + round(dt_sample/dt_system) * N.arange(num_time_steps_corr/2)
        
        
        if (time_steps_corr != true_time_steps).any():
            raise ValueError('Time values do not make sense')
        
        return time_steps_corr, outputs_corr, dt_system
