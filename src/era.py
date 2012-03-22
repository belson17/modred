"""Functions and classes for ERA models"""
import numpy as N
import util

def make_time_steps(num_steps, interval):
    """Returns int array of time steps [0 1 interval interval+1 ...]
    
    Args:
        num_steps: integer, number of time steps to create.
        
        interval: interval between pairs of time steps, as shown above.
    
    Returns:
        time_steps: array of integers, time steps.
    """
    if num_steps % 2 != 0:
        raise ValueError('num_steps must be even, you gave %d'%num_steps)
    interval = int(interval)
    time_steps = N.zeros(num_steps, dtype=int)
    time_steps[::2] = interval*N.arange(num_steps/2)
    time_steps[1::2] = 1 + interval*N.arange(num_steps/2)
    return time_steps


def make_sampled_format(times, outputs):
    """Converts equally spaced samples into sampled format [0 1 P P+1 ...].
    
    Args:
        times: an array of time values or time steps.
        
        outputs: an array of output values at the ``times`` specified.
    
    Returns:
        time_steps: array with integer time steps
        
        outputs: outputs at the time_steps
        
        dt: the associated value of time between each time step
    """
    num_time_steps, num_outputs, num_inputs = outputs.shape
    if num_time_steps != times.shape[0]:
        raise RuntimeError('Size of time and output arrays differ')

    dt_system = times[1] - times[0]
    dt_sample = times[2] - times[0]
    time_steps = N.round((times-times[0])/dt_system)

    if N.abs(2*dt_system - dt_sample) > 1e-8:
        raise ValueError('Data is already in a sampled format, '
            'not equally spaced')
    
    # Format [0, 1, 2, 3, ...], requires conversion
    num_time_steps_corr = (num_time_steps - 1)*2
    outputs_corr = N.zeros((num_time_steps_corr, num_outputs, num_inputs))
    time_steps_corr = N.zeros(num_time_steps_corr, dtype=int)
    outputs_corr[::2] = outputs[:-1]
    outputs_corr[1::2] = outputs[1:]
    time_steps_corr[::2] = time_steps[:-1]
    time_steps_corr[1::2] = time_steps[1:]
    true_time_steps = make_time_steps(num_time_steps_corr, 1)
        
    if (time_steps_corr != true_time_steps).any():
        raise ValueError('Time values do not make sense')
    
    return true_time_steps, outputs_corr, dt_system


def compute_ROM(outputs, num_states, dt=None):
    """Returns A, B, and C matrices, default settings for convenience."""
    my_ERA = ERA()
    my_ERA.set_outputs(outputs, dt=dt)
    my_ERA.compute_ROM(num_states)
    return my_ERA.A, my_ERA.B, my_ERA.C
    


class ERA(object):
    """Forms ERA ROM for discrete-time system from impulse output data 

    Usage::
    
      output_paths =['/path/in1ToOuts.txt', '/path/in2ToOuts.txt']
      myERA = ERA()
      time_values, outputs = util.load_impulse_outputs(output_paths)
      # Optional for equally spaced time data
      time_steps, outputs, dt = make_sampled_format(time_values, outputs)
      myERA.set_outputs(outputs, dt)
      myERA.compute_ROM(50)
      myERA.put_ROM('A.txt','B.txt','C.txt')
      
    The above usage uses util.load_impulse_outputs, which assumes a 
    format of text files (see documentation for this function).
        
    The output times must be in the format:
    t0 + dt_system*[0, 1, P, P+1, 2P, 2P+1, ... ]
    
    The special case where P=2 can be recast into P=1 using make_sampled_format().
    t0 + dt_system*[0, 1, 2, 3, 4, ... ]
      
    See Ma et al. 2011 TCFD for ERA details.
    """
    
    def __init__(self, put_mat=util.save_mat_text, get_mat=util.load_mat_text,
        mc=None, mo=None, verbose=True):
        """Constructor
        
        Kwargs:
            mc: number of Markov parameters for controllable dimension.
                Default is mc and mo equal and maximal for a balanced model.
            mo: number of Markov parameters for observable dimension.
                Default is mc and mo equal and maximal for a balanced model.
                
        Variables mc and mo are the number of Markov parameters to use for
        the Hankel matrix.
        The default is to use all the data and set mc = mo.
        """
        self.put_mat = put_mat
        self.get_mat = get_mat
        self.mc = mc
        self.mo = mo
        self.outputs = None
        self.verbose = verbose
        # Used for error checking and determing file formats
        self.dt_tol = 1e-6
        self.A = None
        self.B = None
        self.C = None
        self.sing_vals = None
        self.L_sing_vecs = None
        self.R_sing_vecs = None
        self.num_outputs = None
        self.num_inputs = None
        self.num_time_steps = None
        self.dt = None
        self.Hankel_mat = None
        self.Hankel_mat2 = None
     

    def set_outputs(self, outputs, dt=None):
        """Set the signals from each output to each input.
        
        Args:
            outputs: array of Markov params w/indices 
                (time_interval#, output#, input#),
                so that outputs[vec_num] is the Markov parameter C A**i B.
                outputs = [CB, CAB, CA**PB, CA**(P+1)B, ...]
        Kwargs:
            dt: time step of the system, used to scale B in later step.
        """
        if dt is None:
            self.dt = 1.
        else:
            self.dt = dt
            
        self.outputs = outputs
        ndims = self.outputs.ndim
        if ndims == 1:
            self.outputs = self.outputs.reshape((self.outputs.shape[0], 1, 1))
        elif ndims == 2:
            self.outputs = self.outputs.reshape((self.outputs.shape[0], 
                self.outputs.shape[1], 1))
        elif ndims > 3: 
            raise RuntimeError('outputs can have 1, 2, or 3 dims, '
                'yours had %d'%ndims)
        
        self.num_time_steps, self.num_outputs, self.num_inputs = \
            self.outputs.shape
        
        # Must have an even number of time steps, remove last entry if odd.
        if self.num_time_steps % 2 != 0:
            self.num_time_steps -= 1
            self.outputs = self.outputs[:-1]
            
    
    
    def compute_ROM(self, num_states, dt=None, mc=None, mo=None):
        """Computes the A, B, and C LTI ROM matrices.

        Assembles the Hankel matrices from self.outputs and takes SVD.
        
        Args:
            num_states: number of states to be found for the ROM.
            
        Kwargs:
            dt: time step of the discrete system, *not* the sampling period ("P").
            
            mc: number of Markov parameters for controllable dimension.
            		Default is mc and mo equal and maximal for a balanced model.
            
            mo: number of Markov parameters for observable dimension.
            		Default is mc and mo equal and maximal for a balanced model.
                
        For discrete time systems the impulse is applied over a time interval
        dt and so has a time-integral 1*dt rather than 1. 
        This means the B matrix is "off" by a factor of dt. 
        This is accounted for by multiplying B by dt.
        This is the only reason dt is an argument.
        
        SVD is ``L_sing_vecs*N.mat(N.diag(sing_vals))*R_sing_vecs.H = Hankel_mat``
        """

        if self.outputs is None:
            raise util.UndefinedError('No output impulse data in instance')
        
        self.mc = mc
        self.mo = mo

        self._assemble_Hankel()
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = \
            util.svd(self.Hankel_mat) 

        # Truncate matrices
        Ur = N.mat(self.L_sing_vecs[:,:num_states])
        Er = N.squeeze(self.sing_vals[:num_states])
        Vr = N.mat(self.R_sing_vecs[:,:num_states])
        
        self.A = N.mat(N.diag(Er**-.5)) * Ur.H * self.Hankel_mat2 * Vr * \
            N.mat(N.diag(Er**-.5))
        self.B = (N.mat(N.diag(Er**.5)) * (Vr.H)[:,:self.num_inputs]) * self.dt 
        # !! NEED * dt above!!
        self.C = Ur[:self.num_outputs,:] * N.mat(N.diag(Er**.5))
        
        if (N.linalg.eig(self.A)[0] >= 1.).any() and self.verbose:
            print 'Warning: Unstable eigenvalues of ROM matrix A'

 
    def put_ROM(self, A_dest, B_dest, C_dest):
        """Puts the A, B, and C LTI matrices to destination"""  
        self.put_mat(self.A, A_dest)
        self.put_mat(self.B, B_dest)
        self.put_mat(self.C, C_dest)
        if self.verbose:
            print 'Put ROM matrices to:'
            print A_dest
            print B_dest
            print C_dest
     
 
    def put_decomp(self, Hankel_mat_dest, Hankel_mat2_dest, L_sing_vecs_dest, 
        sing_vals_dest, R_sing_vecs_dest):
        """Saves the decomposition and Hankel matrices"""
        self.put_mat(self.Hankel_mat, Hankel_mat_dest)
        self.put_mat(self.Hankel_mat2, Hankel_mat2_dest)
        self.put_mat(self.L_sing_vecs, L_sing_vecs_dest)
        self.put_mat(self.sing_vals, sing_vals_dest)
        self.put_mat(self.R_sing_vecs, R_sing_vecs_dest)
    
    
    def put_sing_vals(self, sing_vals_dest):
        """Saves just the singular values"""
        self.put_mat(self.sing_vals, sing_vals_dest)
      
 
    def _assemble_Hankel(self):
        """Assembles and sets self.Hankel_mat and self.Hankel_mat2 
        
        (H and H' in Ma 2011).        
        """
        # To understand the default choice of mc and mo, 
        # consider (let sample_interval=P=1 w.l.o.g.)
        # sequences
        # [0 1 1 2 2 3] => H = [[0 1][1 2]]   H2 = [[1 2][2 3]], mc=mo=1, nt=6
        # [0 1 1 2 2 3 3 4 4 5] => H=[[0 1 2][1 2 3][2 3 4]] 
        # and H2=[[1 2 3][2 3 4][3 4 5]], mc=mo=2,nt=10
        # Thus, using all available data means mc=mo=(nt-2)/4.
        # For additional output times, (nt-2)/4 rounds down, so
        # we use this formula.
        
        if self.mo is None or self.mc is None:
            # Set mo and mc, time_steps is always in format [0, 1, P, P+1, ...]
            self.mo = (self.num_time_steps-2)/4
            self.mc = self.mo
                
        if (self.mo + self.mc) +2 > self.num_time_steps:
            raise ValueError('mo+mc+2=%d and must be <= than the number of '
                'samples %d'%(self.mo+self.mc+2, self.num_time_steps))
        
        self.Hankel_mat = N.zeros((self.num_outputs*self.mo, 
            self.num_inputs*self.mc))
        self.Hankel_mat2 = N.zeros(self.Hankel_mat.shape)
        #outputs_flattened = \
        #    self.outputs.swapaxes(0,1).reshape((num_outputs, -1))
        for row in range(self.mo):
            row_start = row*self.num_outputs
            row_end = row_start + self.num_outputs
            for col in range(self.mc):
                col_start = col*self.num_inputs
                col_end = col_start + self.num_inputs
                self.Hankel_mat[row_start:row_end, col_start:col_end] = \
                    self.outputs[2*(row+col)]
                self.Hankel_mat2[row_start:row_end, col_start:col_end] = \
                    self.outputs[2*(row+col)+1]
                
                
