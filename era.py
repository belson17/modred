
import numpy as N
import util

class ERA(object):
    """ Forms the ERA ROM, following Ma 2010 
    
    Usage:
    The simplest way to use this class is to call 
    myERA = ERA(dt=2.5)
    myERA.load_impulse_outputs(['/path/input1ToOutputs.txt',\
      '/path/input2ToOutputs.txt',...])
    myERA.compute_decomp()
    myERA.compute_ROM(numStates=50)
    
    This would generate a 50-state LTI ROM with A,B,C matrices saved in text
    format in the outputDir. 
    
    The above usage makes use of load_impulse_outputs, which assumes a 
    particular format of text files (see documentation for this function).
    If the data is not saved in this way, the impulse response output signals
    can be set directly with set_impulse_outputs. For details, see 
    the documentation of set_impulse_outputs.
    
    Currently everything only works in serial, but ERA is fast once the 
    input-output signals are given. In the future this class could be extended
    to compute input-to-output signals from saved snapshots as well.
    """
  
    def __init__(self, IOSignals = None, LSingVecsPath=None,
        singValsPath=None, RSingVecsPath=None,
        hankelMatPath=None, hankelMat2Path=None,  
        save_mat=util.save_mat_text, load_mat=util.load_mat_text,
        dt=None, mc=None, mo=None, numStates=100):

        self.IOSignals = IOSignals
        self.LSingVecsPath = LSingVecsPath
        self.RSingVecsPath = RSingVecsPath
        self.singValsPath = singValsPath
        self.hankelMatPath = hankelMatPath
        self.hankelMat2Path = hankelMat2Path
        self.save_mat = save_mat
        self.load_mat = load_mat
        self.dt=dt
        self.mc=mc
        self.mo=mo
        self.numStates=numStates
        self.numOutputs = None
        self.numInputs = None
        self.numSnaps = None

    def compute_decomp(self, mc=None, mo=None,
      hankelMatPath=None,
      hankelMat2Path=None, LSingVecsPath=None,
      singValsPath=None, RSingVecsPath=None):
        """
        Forms Hankel matrices, takes the SVD, and saves the resulting matrices
        
        - IOPaths is a list of file names, each containing the outputs from 
        a single impulse response of an input.
        These files are read with the function self._loadInputToOutputs,
        or will be read 
        by a default function that assumes a text file with columns:
        [t output1 output2 ...]
        - RSingVecsPath is the location of the V matrix in the SVD of
        the Hankel matrix, U E V* = H
        LSingVecs*N.mat(N.diag(singVals))*RSingVecs.H = hankelMat
        """        
        if mc is not None: self.mc=mc
        if mo is not None: self.mo=mo
        
        if self.IOSignals is None:
            raise UndefinedError('No output impulse data exists in ERA instance')
        
        # self.IOSignals now contains the impulse response data
        # numSnaps,numOutputs,numInputs,self.dt are determined from IOSignals.
        
        # Form the Hankel matrices with self.IOSignals
        self._compute_hankel()
        self.LSingVecs, self.singVals, self.RSingVecs = util.svd(self.hankelMat) 

        self.save_decomp(hankelMatPath=hankelMatPath,\
          hankelMat2Path=hankelMat2Path, LSingVecsPath=LSingVecsPath,\
          singValsPath=singValsPath, RSingVecsPath=RSingVecsPath)   
    
    def compute_ROM(self,numStates=None,dt=None,
      APath='A_ERA_disc.txt',BPath='B_ERA_disc.txt',CPath='C_ERA_disc.txt'):
        """
        Computes the A,B,C ROM matrices and saves to file.
        
        dt - is the time step between the snapshots. If it is not given, assume it
        was determined by computeDecomp and saved as a data member.
        numStates is the number of states in the ROM.
        APath, BPath, CPath - the filenames of where to save the ROM matrices.
        They are discrete time matrices, with the associated timestep 'dt'.
        """        
        if numStates is not None: 
            self.numStates=numStates
        if dt is not None: 
            self.dt=dt
                
        # Have all of the decomposition matrices required, find ROM   
        self._compute_ROM_matrices()
        self.save_ROM(APath=APath, BPath=BPath, CPath=CPath)
        
      
    def set_impulse_outputs(self,IOSignals):
        """
        Set the signals from each output to each input.
        
        IOSignals is a 3D array with indices [output#, input#, snap#]
        The self.IOSignals variable must be set with the data before
        self.compute_decomp can be called. This function is particularly
        useful if the data is not saved in the default format (which can
        be loaded and set to IOSignals with self.load_impulse_outputs).
        In this way, any data format can be used, the computation/loading
        of the impulse output data done independently, and passed into
        the ERA class instance.
        """
        if not isintance(IOSignals,N.zeros(1)):
            raise RuntimeError('IOSignals must be a numpy array')
        if len(IOSignals.shape) != 3:
            raise RuntimeError('IOSignals must be a 3D numpy array')
        
        self.IOSignals = IOSignals
 
  
    def load_impulse_outputs(self,IOPaths):
        """
        Reads impulse output signals, forms IOSignals array.
        
        This is a convenience function that assumes the form of the files
        to have columns
        [t output1 output2 ...] 
        for each impulse response in separate files. 
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
        dat = self.load_mat(IOPaths[0],delimiter=' ')
        time = dat[:,0]
        outputSignals = dat[:,1:]
        self.dt = time[1]-time[0]
        for ti in xrange(len(time)-1):
            if abs(time[ti+1]-time[ti]-self.dt)>self.dt*1e-3:
                raise ValueError('dt must be constant, isnt in file '+IOPaths[0])  
        numSnaps,numOutputs = N.shape(outputSignals)
        
        self.IOSignals = N.zeros((numOutputs,numInputs,numSnaps))
        self.IOSignals[:,0,:] = outputSignals.T
        
        # Load all of the other IOFiles
        for inputNum,IOPath in enumerate(IOPaths[1:]):
            dat = self.load_mat(IOPath,delimiter=' ')
            if N.abs((dat[1,0]-dat[0,0])-self.dt)>self.dt*1e-3:
                raise ValueError('dt is not consistent for file '+IOPath)
            self.IOSignals[:,inputNum+1,:] = dat[:,1:].T
        
        #self.IOSignals has shape: (i,j,k) where i is the OUTPUT, j is input,
        #and k is snapshot number. All are 0-indexed, obviously.   
 
    def save_decomp(self,hankelMatPath=None, \
      hankelMat2Path=None, LSingVecsPath=None, \
      singValsPath=None, RSingVecsPath=None):
        """Saves the decomposition matrices"""
        if LSingVecsPath is not None: self.LSingVecsPath=LSingVecsPath
        if RSingVecsPath is not None: self.RSingVecsPath=RSingVecsPath
        if singValsPath is not None: self.singValsPath=singValsPath
        if hankelMatPath is not None: self.hankelMatPath=hankelMatPath
        if hankelMat2Path is not None: self.hankelMat2Path=hankelMat2Path 
        
        if self.hankelMatPath is not None:  
            self.save_mat(self.hankelMat,self.hankelMatPath)
        else:
            print 'WARNING - No hankel matrix path given, not saving it'
        
        if self.hankelMat2Path is not None:  
            self.save_mat(self.hankelMat2,self.hankelMat2Path)
        else:
            print 'WARNING - No A*(hankel matrix) path given, not saving it'
        
        if self.LSingVecsPath is not None:  
            self.save_mat(self.LSingVecs,self.LSingVecsPath)
        else:
            print 'WARNING - No left singular vectors path given, not saving it'
        
        if self.singValsPath is not None:  
            self.save_mat(self.singVals,self.singValsPath)
        else:
            print 'WARNING - No singular values path given, not saving it'
        
        if self.RSingVecsPath is not None:  
            self.save_mat(self.RSingVecs,self.RSingVecsPath)
        else:
            print 'WARNING - No right singular vectores path given, not saving it'
    
        
    def load_decomp(self,hankelMatPath=None,
        hankelMat2Path=None, LSingVecsPath=None,
        singValsPath=None, RSingVecsPath=None,
        numInputs=None,numOutputs=None):
        """Loads the decomposition matrices, computes SVD if necessary"""
        if LSingVecsPath is not None: self.LSingVecsPath=LSingVecsPath
        if RSingVecsPath is not None: self.RSingVecsPath=RSingVecsPath
        if singValsPath is not None: self.singValsPath=singValsPath
        if hankelMatPath is not None: self.hankelMatPath=hankelMatPath
        if hankelMat2Path is not None: self.hankelMat2Path=hankelMat2Path 
        if numInputs is not None: self.numInputs = numInputs
        if numOutputs is not None: self.numOutputs = numOutputs
        
        if self.LSingVecsPath is not None:
            self.LSingVecs=self.load_mat(self.LSingVecsPath)
        
        if self.RSingVecsPath is not None:
            self.RSingVecs=self.load_mat(self.RSingVecsPath)    
        
        if self.singValsPath is not None:
            self.singVals=self.load_mat(self.singValsPath)    
        
        if self.hankelMatPath is not None:
            self.hankelMat=self.load_mat(self.hankelMatPath)
        if self.hankelMat is None:
            raise UndefinedError('No hankel matrix data is given, required')
        
        if self.hankelMat2Path is not None:
            self.hankelMat2=self.load_mat(self.hankelMat2Path)
        if self.hankelMat2 is None:
            raise UndefinedError('No hankel matrix 2 data is given, required')  
        
        self.mo,self.mc = N.shape(self.hankelMat)
        if (self.mo,self.mc) != self.hankelMat2.shape:
            raise RuntimeError('Sizes of hankel and hankel2 matrices differ')
        
        if self.RSingVecs is None or self.LSingVecs is None or \
          self.singVals is None:
            print 'Paths to the SVD decomp matrices were not given so '+\
              'computing the SVD now' 
            self.LSingVecs, self.singVals, self.RSingVecs = util.svd(self.hankelMat)
        
            
  
    def save_ROM(self,APath,BPath,CPath):
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
        """
        self.numOutputs,self.numInputs,self.numSnaps = N.shape(self.IOSignals)
        if self.mo is None or self.mc is None:
            self.mo = ((self.numSnaps-2)/2)
            self.mc = ((self.numSnaps-2)/2)
            #print 'Assuming square Hankel matrix with max number of snapshots'
        #print 'mo, mc, and numSnapshots are', self.mo,self.mc,numSnaps
 
        if (self.mo + self.mc)+2 > self.numSnaps:
            raise ValueError('mo+mc+2 must be less than the number of snapshots')
        
        #Now form the hankel matrix, H and H' from Ma 2010
        self.hankelMat = N.zeros((self.numOutputs*(self.mo+1),
          self.numInputs*(self.mc+1)))
        self.hankelMat2 = N.zeros(N.shape(self.hankelMat))
        for no in range(self.mo+1):
            for nc in range(self.mc+1):
                #print 'shape of H',N.shape(H[no:no+numOutputs,nc:nc+numInputs])
                #print 'shape of IOSignals',N.shape(IOSignals[:,:,no+nc])
                #print no,nc
                self.hankelMat[no*self.numOutputs:(no+1)*self.numOutputs,\
                  nc*self.numInputs:(nc+1)*self.numInputs]=self.IOSignals[:,:,no+nc]
                self.hankelMat2[no*self.numOutputs:(no+1)*self.numOutputs,\
                  nc*self.numInputs:(nc+1)*self.numInputs]=self.IOSignals[:,:,no+nc+1]
        #computed Hankel matrices and SVD as data members
    
    
    def _compute_ROM_matrices(self):
        """ Creates the A,B,C LTI ROM matrices from the SVD matrices.
        
        See Ma et al 2010 for derivation of matrix equations"""
        if self.numInputs is None or self.numOutputs is None:
            raise UndefinedError('Specify number of inputs and outputs. '+\
            'If you loaded the SVD matrices from file, also give optional\n'+\
            'arguments numInputs and numOutputs to load_decomp')    
        # Truncated matrices
        Ur = self.LSingVecs[:,:self.numStates]
        Er = self.singVals[:self.numStates]
        Vr = self.RSingVecs[:,:self.numStates]
        
        self.A = N.matrix(N.diag(Er**-.5)) * Ur.H * self.hankelMat2 * \
          Vr * N.matrix(N.diag(Er**-.5))
        self.B = (N.matrix(N.diag(Er**.5)) * (Vr.H)[:,:self.numInputs]) * self.dt 
        # !! NEED * dt above!!
        # ERA finds a system that reproduces the impulse responses. However,
        # the impulse is applied over a time interval dt and so has an integral
        # 1*dt rather than just 1. This means the B matrix in the ERA system
        # is "off" by a factor of dt. When the dt factor is included, 
        # the ERA and BPOD
        # systems are approximately the same.
        self.C = (Ur[:self.numOutputs,:]*N.matrix(N.diag(Er**.5)))
        
        
      
      
    
  