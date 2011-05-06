
import numpy as N
import util

class ERA(object):
    """ Forms the ERA ROM, following Ma 2010 
    
    Usage:
    The simplest way to use this class is to call 
    myERA = ERA()
    myERA.load_impulse_outputs(['/path/input1ToOutputs.txt',\
      '/path/input2ToOutputs.txt',...])
    myERA.compute_decomp()
    myERA.save_decomp('H.txt','H2.txt','U.txt','E.txt','V.txt')
    myERA.compute_ROM(50, APath='A.txt',BPath='B.txt',CPath='C.txt')
    
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
        dtSample=None, dtModel=None, mc=None, mo=None, numStates=100):
        
        self.dtTol = 1e-6
        self.save_mat = save_mat
        self.load_mat = load_mat
        self.dtSample = dtSample 
        self.dtModel = dtModel
        self.mc=mc
        self.mo=mo
        self.numStates = numStates

    def compute_decomp(self, mc=None, mo=None):
        """
        Forms Hankel matrices, takes the SVD, and saves the resulting matrices
        
        Assumes that the impulse output signals are provided to the class as
        self.IOSignalsSampled and IOSignalsAdvancedDt
        
        LSingVecs*N.mat(N.diag(singVals))*RSingVecs.H = hankelMat
        """
        if mc is not None: self.mc = mc
        if mo is not None: self.mo = mo
        
        if self.IOSignalsSampled is None or self.IOSignalsAdvancedDt is None:
            raise util.UndefinedError('No output impulse data exists in ERA instance')
        
        # IOSignalsDt* now contains the impulse response data
        # numSnaps,numOutputs,numInputs,self.dt are determined from IOSignals.
        
        # Form the Hankel matrices with self.IOSignals*
        self._compute_hankel()
        self.LSingVecs, self.singVals, self.RSingVecs = util.svd(self.hankelMat) 
    
    def compute_ROM(self, numStates, dtSample=None, dtModel=None):
        """
        Computes the A,B,C ROM discrete-time matrices, with dtModel time step.
        
        dtSample - is the time step between the snapshots. If it is not given,
        assume it is already known.
        dtModel - time step after snapshots to A*snapshot. If not given, 
        assume already known.
        numStates is the number of states in the ROM.
        """
        self.numStates = numStates
        if dtSample is not None: self.dtSample = dtSample
        if dtModel is not None: self.dtModel = dtModel
        if self.dtSample is None or self.dtModel is None:
            raise util.UndefinedError('No time step was given')
        # Have all of the decomposition matrices required, find ROM   
        self._compute_ROM_matrices()
        
      
    def set_impulse_outputs(self, IOSignalsSampled, IOSignalsAdvancedDt):
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
        if not isinstance(IOSignalsSampled, type(N.zeros(1))):
            raise RuntimeError('IOSignalsSampled must be a numpy array')
        if not isinstance(IOSignalsAdvancedDt, type(N.zeros(1))):
            raise RuntimeError('IOSignalsAdvancedDt must be a numpy array')
        if len(IOSignalsSampled.shape) != 3 or len(IOSignalsAdvancedDt.shape) != 3:
            raise RuntimeError('IOSignalsSampled and AdvancedDt must '+\
              'be 3D numpy arrays')
        
        self.IOSignalsSampled = IOSignalsSampled
        self.IOSignalsAdvancedDt = IOSignalsAdvancedDt
        self.numOutputs,self.numInputs,self.numSnaps = N.shape(self.IOSignalsSampled)
        if self.IOSignalsSampled.shape != self.IOSignalsAdvancedDt.shape:
            raise RuntimeError('IOSignalsSampled and IOSignalsAdvancedDt are not '+\
              'the same size')
 
  
    def load_impulse_outputs(self, IOPaths):
        """
        Reads impulse output signals, forms IOSignals array.
        
        This is a convenience function that assumes the form of the files
        to have columns
        [t output1 output2 ...] 
        for each impulse response in separate files. The times must be:
        t0, t0+dtModel, t0+dtSample, t0+dtSample+dtModel, t0+2*dtSample, ...
        OR, if dtModel and dtSample are equal
        t0, t0+dtSample, t0+2*dtSample, ...
        
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
        numInputs = len(IOPaths)       

        # Read the first to get some of the parameters
        rawData = self.load_mat(IOPaths[0],delimiter=' ')
        time = rawData[:,0]
        outputSignals = rawData[:,1:]
        dummy,numOutputs = N.shape(outputSignals)

        tiSample = 0
        t0 = time[0]
        if abs((time[2]-t0) - (time[1]-t0)*2) < self.dtTol:
            # dtModel and dtSample are equal, special case!
            self.dtSample = time[1] - t0
            self.dtModel = self.dtSample
        else:
            self.dtSample = time[2] - t0
            self.dtModel = time[1] - t0
        
        for ti,tv in enumerate(time[:-1]):
            if abs(t0 + tiSample*self.dtSample - tv) < self.dtTol:
                # Then on a sampled dt value
                if abs(t0 + tiSample*self.dtSample + self.dtModel - time[ti+1]) > self.dtTol:
                    raise ValueError('Time spacing, dtModel, is wrong in '+IOPaths[0]+\
                    '; expected dtSample '+str(self.dtSample)+' and dtModel '+\
                    str(self.dtModel))
                else:
                    tiSample += 1
        
        # tiSample is now the number of snapshots at the sampled values
        # This includes the IC, if it was in the impulse response output file
        numSnaps = tiSample
        
        self.IOSignalsSampled = N.zeros((numOutputs,numInputs,numSnaps))
        self.IOSignalsAdvancedDt = N.zeros((numOutputs,numInputs,numSnaps))
        
        # Load all of the IOFiles and store the output signals
        for inputNum,IOPath in enumerate(IOPaths):
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
                if abs(t0 + tiSample*self.dtSample - tv) < self.dtTol:
                    # Then on a sampled dt value
                    if abs(t0 + tiSample*self.dtSample + self.dtModel - rawData[ti+1,0]) > self.dtTol:
                        raise ValueError('Time spacing, dtModel, is wrong in '+\
                          IOPaths[0]+'; expected dtSample '+\
                          str(self.dtSample)+' and dtModel '+\
                          str(self.dtModel))
                    else:
                        self.IOSignalsSampled[:,inputNum,tiSample] = \
                          outputSignals[ti,:].T
                        self.IOSignalsAdvancedDt[:,inputNum,tiSample] = \
                          outputSignals[ti+1,:].T
                        tiSample += 1
                  
        # self.IOSignalsSampled and Model have shape:
        # (i,j,k) where i is the OUTPUT, j is input,
        # and k is snapshot number. All are 0-indexed. 
        # For example,  the first input's
        # impulse response output signals are in [:,0,:].   
 
    def save_decomp(self, hankelMatPath, hankelMat2Path, 
      LSingVecsPath, singValsPath, RSingVecsPath):
        """Saves the decomposition matrices"""
        self.save_mat(self.hankelMat, hankelMatPath)
        self.save_mat(self.hankelMat2,hankelMat2Path)
        self.save_mat(self.LSingVecs,LSingVecsPath)
        self.save_mat(self.singVals,singValsPath)
        self.save_mat(self.RSingVecs,RSingVecsPath)
    
        
    def load_decomp(self, hankelMatPath, hankelMat2Path, \
      LSingVecsPath, singValsPath, RSingVecsPath,
      numInputs = None, numOutputs = None):
        """Loads the decomposition matrices, computes SVD if necessary"""
        if numInputs is not None: self.numInputs = numInputs
        if numOutputs is not None: self.numOutputs = numOutputs
        if self.numOutputs is None or self.numInputs is None:
            raise util.UndefinedError(\
              'Specify number of outputs and inputs when loading decomp')
              
        self.LSingVecs=self.load_mat(LSingVecsPath)
        self.RSingVecs=self.load_mat(RSingVecsPath)    
        self.singVals=self.load_mat(singValsPath)    
        self.hankelMat=self.load_mat(hankelMatPath)
        self.hankelMat2=self.load_mat(hankelMat2Path)
        
        s = self.hankelMat.shape
        self.mo = s[0]/self.numOutputs
        self.mc = s[1]/self.numInputs
        if self.hankelMat.shape != self.hankelMat2.shape:
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
        
        Stores them internally as self.hankelMat and self.hankelMat2.
        Requires that self.IOSignalsSampled and AdvanceDt be set only
        """
        self.numOutputs,self.numInputs,self.numSnaps = N.shape(self.IOSignalsSampled)
        #print 'num of snaps, including IC, is ',self.numSnaps

        if self.mo is None or self.mc is None:
            self.mo = ((self.numSnaps-1)/2)
            self.mc = ((self.numSnaps-1)/2)
            #print 'Assuming square Hankel matrix with max number of snapshots'
        #print 'mo, mc, and numSnapshots are', self.mo,self.mc,numSnaps
 
        if (self.mo + self.mc + 1) > self.numSnaps:
            raise ValueError('mo+mc+1 must be less than the number of snapshots')
        
        #Now form the hankel matrix, H and H' from Ma 2010
        self.hankelMat = N.zeros((self.numOutputs*(self.mo),
          self.numInputs*(self.mc)))
        self.hankelMat2 = N.zeros(N.shape(self.hankelMat))
        #print 'forming hankel mat and mo,mc is',self.mo,self.mc
        #print 'hankel matrices have size',self.hankelMat.shape
        #print 'shape of iosignals is',self.IOSignalsSampled.shape
        for no in range(self.mo):
            for nc in range(self.mc):
                #print 'shape of H',N.shape(H[no:no+numOutputs,nc:nc+numInputs])
                #print 'shape of IOSignals',N.shape(IOSignals[:,:,no+nc])
                #print no,nc
                self.hankelMat[no*self.numOutputs:(no+1)*self.numOutputs,\
                  nc*self.numInputs:(nc+1)*self.numInputs]=self.IOSignalsSampled[:,:,no+nc]
                self.hankelMat2[no*self.numOutputs:(no+1)*self.numOutputs,\
                  nc*self.numInputs:(nc+1)*self.numInputs]=self.IOSignalsAdvancedDt[:,:,no+nc]
        #computed Hankel matrices and SVD as data members
    
    
    def _compute_ROM_matrices(self):
        """ Creates the A,B,C LTI ROM matrices from the SVD matrices.
        
        See Ma et al 2010 for derivation of matrix equations.
        Requires that numInputs, numOutputs, numStates, and the SVD matrices be
        set."""
        if self.numInputs is None or self.numOutputs is None:
            raise UndefinedError('Specify number of inputs and outputs. '+\
            'If you loaded the SVD matrices from file, also give optional\n'+\
            'arguments numInputs and numOutputs to load_decomp')
        # Truncated matrices
        Ur = N.mat(self.LSingVecs[:,:self.numStates])
        Er = N.squeeze(self.singVals[:self.numStates])
        Vr = N.mat(self.RSingVecs[:,:self.numStates])
        
        #print 'sizes of matrices to be multiplied are'
        #print N.matrix(N.diag(Er**-.5)).shape
        #print Ur.H.shape
        #print self.hankelMat2.shape
        #print Vr.shape
        #print N.matrix(N.diag(Er**-.5)).shape
        self.A = N.matrix(N.diag(Er**-.5)) * Ur.H * self.hankelMat2 * \
          Vr * N.matrix(N.diag(Er**-.5))
        self.B = (N.matrix(N.diag(Er**.5)) * (Vr.H)[:,:self.numInputs]) * self.dtModel 
        # !! NEED * dt above!!
        # ERA finds a system that reproduces the impulse responses. However,
        # as far as ERA is concerned, the impulse is "applied" over a 
        # time interval dtModel and so has a time-integral
        # 1*dtModel rather than just 1. This means the B matrix in the ERA system
        # is "off" by a factor of dtModel. When the dt factor is included, 
        # the ERA and BPOD
        # systems are approximately the same.
        self.C = (Ur[:self.numOutputs,:]*N.matrix(N.diag(Er**.5)))
        
        
      
      
    
  