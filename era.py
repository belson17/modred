
import numpy as N
import util

#an error to raise when an undefined function or variable is used
class UndefinedError(Exception):
    pass

########################################################################

class ERA(object):
    """ Forms the ERA ROM, following Ma 2010 
    
    The simplest way to use this class is to call 
    myEra = ERA()
    myEra.computeDecomp(IOPaths=['/path/file1','/path/file2',...])
    myEra.computeROM(numStates=50)
    
    This would generate a 50-state ROM with A,B,C matrices saved in text
    format in the outputDir. 
    
    Currently everything only works in serial, but ERA is fast once the 
    input-output signals are given. In the future this class could be extended
    to compute input-to-output signals from saved snapshots as well.
    """
  
    def __init__(self,IOPaths=None,LSingVecsPath='LSingVecs.txt',
        singValsPath='singVals.txt',RSingVecsPath='RSingVecs.txt',
        hankelMatPath='hankelMat.txt',hankelMat2Path='hankelMat2.txt',  
        saveMatrix=util.write_mat_text,loadMatrix=util.read_mat_text,dt=None,
        mc=None,mo=None,numStates=100):
        
        self._IOPaths=IOPaths
        self._outputDir = util.fixDirSlash(outputDir)
        self._LSingVecsPath=LSingVecsPath
        self._RSingVecsPath=RSingVecsPath
        self._hankelMatPath=hankelMatPath
        self._hankelMat2Path=hankelMat2Path
        self._save_matrix = saveMatrix
        self._load_matrix = loadMatrix
        self._dt=dt
        self._mc=mc
        self._mo=mo
        self._numStates=numStates
    
  #def setOutputDir(self,outputDir):
  #  self._outputDir = util.fixDirSlash(outputDir)
  
    def set_IOSignals(self,IOSignals):
        self._IOSignals = IOSignals
  
    def set_IOPaths(self,IOPaths):
        self._IOPaths = IOPaths   
  
  #other set and get functions...
  
    def compute_decomp(self,IOPaths=None,LSingVecsPath=None,singValsPath=None,
        RSingVecsPath=None,hankelMatPath=None,hankelMat2Path=None,
        mc=None,mo=None):
    
        """ This method forms the Hankel matrices H and H2 (Ma 2010 notation), 
        and finds the SVD
        
        - IOPaths is a list of files containing impulse response from each input.
        They are read with the function self._loadInputToOutputs, or will be read 
        by a default function that assumes a text file with columns [t y1 y2 y3 ...]
        - RSingVecsPath is the location of the W matrix in the SVD of the Hankel matrix,
        LSingVecs*singVals*RSingVecs.H = hankelMat
        """
        
        if mc is not None: self._mc=mc
        if mo is not None: self._mo=mo
        if numStates!=None: self._numStates=numStates
        if LSingVecsPath is not None: self._LSingVecsPath=LSingVecsPath
        if RSingVecsPath is not None: self._RSingVecsPath=RSingVecsPath
        if singValsPath is not None: self._singValsPath=singValsPath
        if hankelMatPath is not None: self._hankelMatPath=hankelMatPath
        if hankelMat2Path is not None: self._hankelMat2Path=hankelMat2Path 
        
        #Read the files to form self._IOSignals[numOutputs,numInputs,numSnapshots]
        # IOSignals must be a 3D Numpy array.
        # each output,input pair defines a 1D array, which is the time-series of the 
        #impulse response signal.
        
        #priority order for the input-output impulse response data is: 
        # - IOPaths argument, load in the files
        # - self._IOSignals, no data is given to the function, so use what exists
        # - self._IOPaths - No data or files exist, use the known file locations.
        
        if IOPaths is not None:
            self._read_files(IOPaths) #overwrites self._IOSignals
        elif self._IOSignals is not None:
            pass
        elif self._IOPaths is not None:
            self._read_files(self._IOPaths)
        else:
            raise UndefinedError('No input-output impulse data or files given')
        
        #numSnaps,numOutputs,numInputs,self._dt are determined from files.
        #self._IOSignals now has the impulse data
        
        #Now form the Hankel matrices with self._IOSignals, writes to file the 
        # SVD matrices (hankelMat=RSingVecs,singVals,LSingVecs), and the Hankel matrix itself,
        # hankelMat. It leaves these matrices as data members for use later.
        self._form_hankel()
        self._save_matrix(self._hankelMat,self._hankelMatPath)
        self._save_matrix(self._hankelMat2,self._hankelMat2Path)
        
        self._svd()
        
        #save these matrices to file, by default text files in current dir.
        #save the full svd information so can compute more modes/states later.
        self._save_matrix(self._LSingVecs,self._LSingVecsPath)
        self._save_matrix(self._RSingVecs,self._RSingVecsPath)
        self._save_matrix(self._singVals,self._singValsPath)
    
  ######################################################################  
  
    def compute_ROM(self,numStates=None,dt=None,LSingVecsFile=None,
        singValsPath=None,RSingVecsFile=None,hankelMatFile=None,
        hankelMatFile2=None,APath='A-era-disc.txt',BPath='B-era-disc.txt',
        CPath='C-era-disc.txt'):
        """Computes the A,B,C ROM matrices using numStates number of ROM states.
        
        dt - is the time step between the snapshots. If it is not given, assume it
        was determined by computeDecomp and saved as a data member.
        LSingVecsFile, etc - Files where these matrices are saved. If these
        arguments are given, then the matrices are read from file. If not, then
        the matrices in memory are used (if they exist). 
        AFile,BFile,CFile - the filenames of where to save the ROM matrices.
        They are discrete time matrices, with the associated timestep 'dt'.
        
        """
        
        if numStates is not None: self._numStates=numStates
        if dt!=None: self._dt=dt
        
        if LSingVecsPath is not None:
            self._LSingVecs=self._load_matrix(LSingVecsPath)
        elif self._LSingVecs is None:
            raise UndefinedError('No left singular vector data is given')
        
        if RSingVecsPath is not None:
            self._RSingVecs=self._load_matrix(RSingVecsPath)    
        elif self._RSingVecs is None:
            raise UndefinedError('No right singular vector data is given')
        
        if singValsPath is not None:
            self._singVals=self._load_matrix(singValsPath)    
        elif self._RSingVecs is None:
            raise UndefinedError('No singular value data is given')    
        
        if hankelMatPath is not None:
            self._hankelMat=self._load_matrix(hankelMatPath)
        elif self._hankelMat is None:
            raise UndefinedError('No hankel matrix data is given')
        
        if hankelMat2Path is not None:
            self._hankelMat2=self._load_matrix(hankelMat2Path)
        elif self._hankelMat2 is None:
            raise UndefinedError('No hankel matrix 2 data is given')  
        
        #Now have all of the matrices required, can find ROM.    
        self._A,self._B,self._C = self._compute_ROM_matrices()
        
        #Now write these matrices to files
        self._save_matrix(self._A,APath)
        self._save_matrix(self._B,BPath)
        self._save_matrix(self._C,CPath)
    
  ######################################################################
  
    def _read_files(self,IOPaths):
        """Reads a list of files and forms the IOSignals numpy array
        containing the input-output data"""
        numInputs = len(IOPaths)
          
        #No custom file, instead use simple matrix loading and assumed format
        # of cols [t y1 y2 ...]
        #read the first to get the dimensions
        dat = self._load_matrix(IOPaths[0])
        time = dat[:,0]
        outputSignals = dat[:,1:]
        self._dt = time[1]-time[0]
        for ti in range(len(time)-1):
            if abs(time[ti+1]-time[ti]-self._dt)>self._dt*1e-3:
                raise ValueError('dt must be constant, isnt in file '+IOPaths[0])
          
        numOutputs,numSnaps = N.shape(outputSignals)
        #Now load all of the rest of the files
        #Load all of the other IOFiles
        for inputNum,IOPath in enumerate(IOPaths[1:]):
            dat = self._load_matrix(IOPath) #first column is time
            if N.abs(dat[1,0]-dat[0,0]-self._dt)>self._dt*1e-3:
                raise ValueError('dt is not constant for file '+IOPath)
            self._IOSignals[:,inputNum+1,:] = dat[:,1:]
        
        #self._IOSignals has shape: (i,j,k) where i is the OUTPUT, j is input,
        #and k is snapshot number. All are 0-indexed, obviously.
        """
        #If using a non-standard format and so a custom load function
        if self._loadInputToOutputs != None: #Use the given function
          #read the first file to discover the dimensions
          #each file is assumed to be for one input and contain all output signals 
          #from an impulse response for the one input    
          time,outputSignals = self._loadInputToOutputs(IOFilenames[0])
          self._dt = time[1]-time[0]
          for ti in range(len(time)-1):
            if abs(time[ti+1]-time[ti]-self._dt)>self._dt*1e-3:
              raise ValueError('dt must be constant, isnt in file '+IOFilenames[0])
            
          self.numOutputs,self.numSnaps = N.shape(outputSignals)
          
          self._IOSignals = N.zeros((self.numOutputs,self.numInputs,self.numSnaps))
          self._IOSignals[:,0,:] = outputSignals
          #Now load all of the rest of the files
          #Load all of the other IOFiles
          for inputNum,IOFilename in enumerate(IOFilenames[1:]):
            time,self._IOSignals[:,inputNum+1,:] = self._loadInputToOutputs(IOFilename)
            #just check that the first times are correct, assume others are.
            if abs(time[2]-time[1]-dt)>dt*1e-3:
              raise ValueError('dt must be the same in all files, not the same in '+IOFilename)
        """
  
  
    def _form_hankel(self):
        """ Computes the Hankel and A*Hankel matrix (H and H' in Ma 2010).
        It puts them in self._hankelMat and self._hankelMat2.
        """
        if (numOutputs,numInputs,numSnaps)!=N.shape(self._IOSignals):
            print 'WARNING - IOSignals array size seems to have changed!'  
      
        if (self._mo+self._mc)+2>numSnapshots:
            raise ValueError('mo+mc+2 must be less than the number of snapshots')
        
        if self._mo==None and self._mc==None:
            self._mo = ((numSnaps-2)/2)
            self._mc = ((numSnaps-2)/2)
            #print 'Assuming a square Hankel matrix with maximum number of snapshots'
        print 'mo, mc, and numSnapshots are', self._mo,self._mc,numSnaps
        
        #Now form the hankel matrix, H and H' from Ma 2010
        self._hankelMat = N.zeros((numOutputs*(self._mo+1),numInputs*(self._mc+1)))
        self._hankelMat2 = N.zeros(N.shape(self._hankelMat))
        for no in range(self._mo+1):
            for nc in range(self._mc+1):
                #print 'shape of H',N.shape(H[no:no+numOutputs,nc:nc+numInputs])
                #print 'shape of IOSignals',N.shape(IOSignals[:,:,no+nc])
                #print no,nc
                self._hankelMat[no*numOutputs:(no+1)*numOutputs,\
                  nc*numInputs:(nc+1)*numInputs]=self._IOSignals[:,:,no+nc]
                self._hankelMat2[no*numOutputs:(no+1)*numOutputs,\
                  nc*numInputs:(nc+1)*numInputs]=self._IOSignals[:,:,no+nc+1]
        #computed Hankel matrices and SVD as data members
    
    def _svd(self):
        """ Computes the SVD of the matrix self._hankelMat """
        
        self._LSingVecs,self._singVals,RSingVecsStar =\
          N.linalg.svd(self._hankelMat,full_matrices=0) #only computes nonzero SVD values
        
        self._LSingVecs = N.matrix(self._LSingVecs)
        self._RSingVecs = N.matrix(RSingVecsStar).H
    
    def _compute_ROM_matrices(self):
        """ This method creates the A,B,C matrices for the ROM discrete time
        system. It uses the number of states (ie modes) in self._numStates"""
        
        #truncated matrices
        Ur = self._LSingVecs[:,:self._numStates]
        Er = self._singVals[:self._numStates]
        Vr = self._RSingVecs[:,:self._numStates]
        
        self._A = N.matrix(N.diag(Er**-.5)) * Ur.H * H2 * Vr * N.matrix(N.diag(Er**-.5))
        self._B = (N.matrix(N.diag(Er**.5)) * Vr.H)[:,:] * self._dt ##NEED * dt,
        # ERA finds a system that reproduces the impulse responses. However,
        #the impulse is applied over a time interval dt and so has an integral
        # 1*dt rather than just 1. This means the B matrix in the ERA system
        # is "off" by a factor of dt. When the dt factor is included, the ERA and BPOD
        # systems are approximately the same.
        self._C = (Ur*N.matrix(N.diag(Er**.5)))[:,:]
        
        #no return, computed matrices as data members
    
  