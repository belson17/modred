
import numpy as N
import util

class ERA(object):
    """ Forms the ERA ROM, following Ma 2010 

    The simplest way to use this class is to call 
    
    Usage::

      myERA = ERA()
      myERA.load_impulse_outputs(['/path/input1ToOutputs.txt',
        '/path/input2ToOutputs.txt',...])
      myERA.compute_decomp()
      myERA.save_decomp('H.txt','H2.txt','U.txt','E.txt','V.txt')
      myERA.compute_ROM(50)
    
    This would generate a 50-state LTI ROM with A,B,C matrices saved in text
    format. 
    
    The above usage makes use of load_impulse_outputs, which assumes a 
    particular format of text files (see documentation for this function).
    If the data is not saved in this way, the impulse response output signals
    can be set directly with set_impulse_outputs. For details, see 
    the documentation of set_impulse_outputs.
    
    Currently everything only works in serial, but ERA is fast once the 
    input-output signals are given. In the future this class could be extended
    to compute input-to-output signals from saved snapshots as well.
    """  
    def __init__(self, save_mat=util.save_mat_text, \
        load_mat=util.load_mat_text,
        dt_sample=None, dt_model=None, mc=None, mo=None, num_states=100):
        
        self.dtTol = 1e-6
        self.save_mat = save_mat
        self.load_mat = load_mat
        self.dt_sample = dt_sample 
        self.dt_model = dt_model
        self.mc=mc
        self.mo=mo
        self.num_states = num_states

    def compute_decomp(self, mc=None, mo=None):
        """
        Forms Hankel matrices, takes the SVD, and saves the resulting matrices
        
        Assumes that the impulse output signals are provided to the class as
        self.IO_signals_sampled and IO_signals_advanced_dt
        
        L_sing_vecs*N.mat(N.diag(sing_vals))*R_sing_vecs.H = hankel_mat
        """
        if mc is not None: self.mc = mc
        if mo is not None: self.mo = mo
        
        if self.IO_signals_sampled is None or self.IO_signals_advanced_dt is None:
            raise util.UndefinedError('No output impulse data exists in ERA instance')
        
        # IOSignalsDt* now contains the impulse response data
        # num_snaps,num_outputs,num_inputs,self.dt are determined from IOSignals.
        
        # Form the Hankel matrices with self.IOSignals*
        self._compute_hankel()
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = util.svd(self.hankel_mat) 
    
    def compute_ROM(self, num_states, dt_sample=None, dt_model=None):
        """
        Computes the A,B,C ROM discrete-time matrices, with dt_model time step.
        
        dt_sample 
          is the time step between the snapshots. If it is not given,
          assume it is already known.
        dt_model 
          time step after snapshots to A*snapshot. If not given, 
          assume already known.
        num_states
          the number of states in the ROM.
        """
        self.num_states = num_states
        if dt_sample is not None: self.dt_sample = dt_sample
        if dt_model is not None: self.dt_model = dt_model
        if self.dt_sample is None or self.dt_model is None:
            raise util.UndefinedError('No time step was given')
        # Have all of the decomposition matrices required, find ROM   
        self._compute_ROM_matrices()
        
      
    def set_impulse_outputs(self, IO_signals_sampled, IO_signals_advanced_dt):
        """
        Set the signals from each output to each input.
        
        IOSignals* are 3D arrays with indices [output#, input#, snap#]
        The self.IOSignals* variables must be set with the data before
        self.compute_decomp can be called. This function is particularly
        useful if the data is not saved in the default format (which can
        be loaded with self.load_impulse_outputs).
        In this way, any data format can be used, the computation/loading
        of the impulse output data done independently, and passed into
        the ERA class instance.
        """
        if not isinstance(IO_signals_sampled, type(N.zeros(1))):
            raise RuntimeError('IO_signals_sampled must be a numpy array')
        if not isinstance(IO_signals_advanced_dt, type(N.zeros(1))):
            raise RuntimeError('IO_signals_advanced_dt must be a numpy array')
        if len(IO_signals_sampled.shape) != 3 or len(IO_signals_advanced_dt.shape) != 3:
            raise RuntimeError('IO_signals_sampled and AdvancedDt must '+\
              'be 3D numpy arrays')
        
        self.IO_signals_sampled = IO_signals_sampled
        self.IO_signals_advanced_dt = IO_signals_advanced_dt
        self.num_outputs,self.num_inputs,self.num_snaps = N.shape(self.IO_signals_sampled)
        if self.IO_signals_sampled.shape != self.IO_signals_advanced_dt.shape:
            raise RuntimeError('IO_signals_sampled and IO_signals_advanced_dt are not '+\
              'the same size')
 
  
    def load_impulse_outputs(self, IOPaths):
        """
        Reads impulse output signals, forms IOSignals array.
        
        This is a convenience function that assumes the form of the files
        to have columns
        [t output1 output2 ...] 
        for each impulse response in separate files. The times must be:
        
        t0, t0+dt_model, t0+dt_sample, t0+dt_sample+dt_model, t0+2*dt_sample, ...
        
        OR, if dt_model and dt_sample are equal
        
        t0, t0+dt_sample, t0+2*dt_sample, ...
        
        IOPaths is a list of the files, each file corresponds to a particular
        impulse response.
        
        If the impulse output signals are not saved in this format, one
        can form the  required IOSignals array with data stored as
        
        [output #, input #, snap #]
        
        and pass that array to the ERA instance with
        self.set_impulse_outputs(IOSignals)
        
        Usage:
        myera = ERA()
        myera.load_impulse_outputs(['input1ToOutputs.txt','input2ToOutputs.txt'])
        myera.compute_decomp()
        """
        num_inputs = len(IOPaths)       

        # Read the first to get some of the parameters
        rawData = self.load_mat(IOPaths[0],delimiter=' ')
        time = rawData[:,0]
        outputSignals = rawData[:,1:]
        dummy,num_outputs = N.shape(outputSignals)

        tiSample = 0
        t0 = time[0]
        if abs((time[2]-t0) - (time[1]-t0)*2) < self.dtTol:
            # dt_model and dt_sample are equal, special case!
            self.dt_sample = time[1] - t0
            self.dt_model = self.dt_sample
        else:
            self.dt_sample = time[2] - t0
            self.dt_model = time[1] - t0
        
        for ti,tv in enumerate(time[:-1]):
            if abs(t0 + tiSample*self.dt_sample - tv) < self.dtTol:
                # Then on a sampled dt value
                if abs(t0 + tiSample*self.dt_sample + self.dt_model - time[ti+1]) > self.dtTol:
                    raise ValueError('Time spacing, dt_model, is wrong in '+IOPaths[0]+\
                    '; expected dt_sample '+str(self.dt_sample)+' and dt_model '+\
                    str(self.dt_model))
                else:
                    tiSample += 1
        
        # tiSample is now the number of snapshots at the sampled values
        # This includes the IC, if it was in the impulse response output file
        num_snaps = tiSample
        
        self.IO_signals_sampled = N.zeros((num_outputs,num_inputs,num_snaps))
        self.IO_signals_advanced_dt = N.zeros((num_outputs,num_inputs,num_snaps))
        
        # Load all of the IOFiles and store the output signals
        for input_num,IOPath in enumerate(IOPaths):
            rawData = self.load_mat(IOPath,delimiter=' ')
            timeIO = rawData[:,0]
            outputSignals = rawData[:,1:]
            if len(time) != len(timeIO):
                raise ValueError('Number of impulse output signals differs between'+\
                  ' files '+IOPaths[0]+' and '+IOPath)
            elif N.amax(abs(time - timeIO)) > self.dtTol:
                raise ValueError('Times differ between files '+IOPaths[0]+\
                  ' and '+IOPath)
            tiSample = 0
            for ti,tv in enumerate(timeIO[:-1]):
                if abs(t0 + tiSample*self.dt_sample - tv) < self.dtTol:
                    # Then on a sampled dt value
                    if abs(t0 + tiSample*self.dt_sample + self.dt_model - rawData[ti+1,0]) > self.dtTol:
                        raise ValueError('Time spacing, dt_model, is wrong in '+\
                          IOPaths[0]+'; expected dt_sample '+\
                          str(self.dt_sample)+' and dt_model '+\
                          str(self.dt_model))
                    else:
                        self.IO_signals_sampled[:,input_num,tiSample] = \
                          outputSignals[ti,:].T
                        self.IO_signals_advanced_dt[:,input_num,tiSample] = \
                          outputSignals[ti+1,:].T
                        tiSample += 1
                  
        # self.IO_signals_sampled and Model have shape:
        # (i,j,k) where i is the OUTPUT, j is input,
        # and k is snapshot number. All are 0-indexed. 
        # For example,  the first input's
        # impulse response output signals are in [:,0,:].   
 
    def save_decomp(self, hankel_mat_path, hankel_mat2_path, 
      L_sing_vecs_path, sing_valsPath, R_sing_vecs_path):
        """Saves the decomposition matrices"""
        self.save_mat(self.hankel_mat, hankel_mat_path)
        self.save_mat(self.hankel_mat2,hankel_mat2_path)
        self.save_mat(self.L_sing_vecs,L_sing_vecs_path)
        self.save_mat(self.sing_vals,sing_valsPath)
        self.save_mat(self.R_sing_vecs,R_sing_vecs_path)
    
        
    def load_decomp(self, hankel_mat_path, hankel_mat2_path, \
      L_sing_vecs_path, sing_valsPath, R_sing_vecs_path,
      num_inputs = None, num_outputs = None):
        """Loads the decomposition matrices, computes SVD if necessary"""
        if num_inputs is not None: self.num_inputs = num_inputs
        if num_outputs is not None: self.num_outputs = num_outputs
        if self.num_outputs is None or self.num_inputs is None:
            raise util.UndefinedError(\
              'Specify number of outputs and inputs when loading decomp')
              
        self.L_sing_vecs=self.load_mat(L_sing_vecs_path)
        self.R_sing_vecs=self.load_mat(R_sing_vecs_path)    
        self.sing_vals=self.load_mat(sing_valsPath)    
        self.hankel_mat=self.load_mat(hankel_mat_path)
        self.hankel_mat2=self.load_mat(hankel_mat2_path)
        
        s = self.hankel_mat.shape
        self.mo = s[0]/self.num_outputs
        self.mc = s[1]/self.num_inputs
        if self.hankel_mat.shape != self.hankel_mat2.shape:
            raise RuntimeError('Sizes of hankel and hankel2 matrices differ')
                
  
    def save_ROM(self, APath, BPath, CPath):
        """Saves the A, B, and C LTI matrices to file"""  
        self.save_mat(self.A, APath)
        self.save_mat(self.B, BPath)
        self.save_mat(self.C, CPath)
        print 'Saved ROM matrices to:'
        print APath
        print BPath
        print CPath
    
  
    def _compute_hankel(self):
        """
        Computes the Hankel and A*Hankel matrix (H and H' in Ma 2010).
        
        Stores them internally as self.hankel_mat and self.hankel_mat2.
        Requires that self.IO_signals_sampled and AdvanceDt be set only
        """
        self.num_outputs,self.num_inputs,self.num_snaps = N.shape(self.IO_signals_sampled)
        #print 'num of snaps, including IC, is ',self.num_snaps

        if self.mo is None or self.mc is None:
            self.mo = ((self.num_snaps-1)/2)
            self.mc = ((self.num_snaps-1)/2)
            #print 'Assuming square Hankel matrix with max number of snapshots'
        #print 'mo, mc, and num_snapshots are', self.mo,self.mc,num_snaps
 
        if (self.mo + self.mc + 1) > self.num_snaps:
            raise ValueError('mo+mc+1 must be less than the number of snapshots')
        
        #Now form the hankel matrix, H and H' from Ma 2010
        self.hankel_mat = N.zeros((self.num_outputs*(self.mo),
          self.num_inputs*(self.mc)))
        self.hankel_mat2 = N.zeros(N.shape(self.hankel_mat))
        #print 'forming hankel mat and mo,mc is',self.mo,self.mc
        #print 'hankel matrices have size',self.hankel_mat.shape
        #print 'shape of iosignals is',self.IO_signals_sampled.shape
        for no in range(self.mo):
            for nc in range(self.mc):
                #print 'shape of H',N.shape(H[no:no+num_outputs,nc:nc+num_inputs])
                #print 'shape of IOSignals',N.shape(IOSignals[:,:,no+nc])
                #print no,nc
                self.hankel_mat[no*self.num_outputs:(no+1)*self.num_outputs,\
                  nc*self.num_inputs:(nc+1)*self.num_inputs]=self.IO_signals_sampled[:,:,no+nc]
                self.hankel_mat2[no*self.num_outputs:(no+1)*self.num_outputs,\
                  nc*self.num_inputs:(nc+1)*self.num_inputs]=self.IO_signals_advanced_dt[:,:,no+nc]
        #computed Hankel matrices and SVD as data members
    
    
    def _compute_ROM_matrices(self):
        """ Creates the A,B,C LTI ROM matrices from the SVD matrices.
        
        See Ma et al 2010 for derivation of matrix equations.
        Requires that num_inputs, num_outputs, num_states, and the SVD matrices be
        set."""
        if self.num_inputs is None or self.num_outputs is None:
            raise UndefinedError('Specify number of inputs and outputs. '+\
            'If you loaded the SVD matrices from file, also give optional\n'+\
            'arguments num_inputs and num_outputs to load_decomp')
        # Truncated matrices
        Ur = N.mat(self.L_sing_vecs[:,:self.num_states])
        Er = N.squeeze(self.sing_vals[:self.num_states])
        Vr = N.mat(self.R_sing_vecs[:,:self.num_states])
        
        #print 'sizes of matrices to be multiplied are'
        #print N.matrix(N.diag(Er**-.5)).shape
        #print Ur.H.shape
        #print self.hankel_mat2.shape
        #print Vr.shape
        #print N.matrix(N.diag(Er**-.5)).shape
        self.A = N.matrix(N.diag(Er**-.5)) * Ur.H * self.hankel_mat2 * \
          Vr * N.matrix(N.diag(Er**-.5))
        self.B = (N.matrix(N.diag(Er**.5)) * (Vr.H)[:,:self.num_inputs]) * self.dt_model 
        # !! NEED * dt above!!
        # ERA finds a system that reproduces the impulse responses. However,
        # as far as ERA is concerned, the impulse is "applied" over a 
        # time interval dt_model and so has a time-integral
        # 1*dt_model rather than just 1. This means the B matrix in the ERA system
        # is "off" by a factor of dt_model. When the dt factor is included, 
        # the ERA and BPOD
        # systems are approximately the same.
        self.C = (Ur[:self.num_outputs,:]*N.matrix(N.diag(Er**.5)))
        
        
      
      
    
  
