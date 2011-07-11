
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
      myBPODROM.compute_mode_derivs(modePaths, modeDtPaths, modeDerivPaths,1e-4)
      # For continuous time systems, set dt=0
      myBPODROM.form_A(APath, directModeAdvancedPaths, adjointModePaths, dt)
      myBPODROM.form_B(BPath, inputPaths, adjointModePaths, dt)
      myBPODROM.form_C(CPath, outputPaths, directModePaths)
    
    Eventually it should be made a derived class of
    a ROM class that computes Galerkin projections for generic PDEs. Each
    PDE would be a derived class
    """

    def __init__(self,adjointModePaths=None,
      directModePaths=None,
      directDtModePaths=None,  
      directDerivModePaths=None,
      discrete = None,
      dt = None,
      inner_product=None,
      save_mat=util.save_mat_text,
      load_field=None,
      save_field=None,
      numModes=None,
      verbose = True,
      maxFieldsPerNode=2):
          self.dt = dt
          self.inner_product=inner_product
          self.save_mat=save_mat
          self.load_field=load_field
          self.save_field=save_field
          self.maxFieldsPerNode=maxFieldsPerNode
          self.maxFieldsPerProc = self.maxFieldsPerNode
          self.numModes = numModes
          self.verbose = verbose
      
    def compute_mode_derivs(self,modePaths,modeDtPaths,modeDerivPaths,dt):
        """
        Computes time derivatives of modes. dt=1e-4 is a good first choice.
        
        It reads in modes from modePaths and modeDtPaths, and simply
        subtracts them and divides by dt. This requires both __mul__
        and __add__ to be defined for the mode object.
        """
        if (self.load_field is None):
            raise util.UndefinedError('no load_field defined')
        if (self.save_field is None):
            raise util.UndefinedError('no save_field defined')
        
        numModes = min(len(modePaths),len(modeDtPaths),len(modeDerivPaths))
        print 'Computing derivatives of',numModes,'modes'
        
        for modeIndex in xrange(len(modeDtPaths)):
            mode = self.load_field(modePaths[modeIndex])
            modeDt = self.load_field(modeDtPaths[modeIndex])
            self.save_field((modeDt - mode)*(1./dt), modeDerivPaths[modeIndex])
        
    
    def form_A(self, APath, directModeAdvancedPaths, adjointModePaths, dt, numModes=None):
        """
        Computes the continouso or discrete time A matrix
        
        APath
          where the A matrix will be saved
        directModeAdvancedPaths
          For a discrete time system, this is the 
          paths to the direct modes that have been advanced a time dt.
          For continuous time systems, this should be paths to the time
          derivatives of the direct modes.
        adjointModePaths
          Paths to the adjoint modes
        dt
          The associated time step of the ROM, for continuous time it is 0.
        numModes
          number of modes/states to keep in the ROM.
        """
        self.dt = dt
        if self.verbose:
            if self.dt == 0:
                print 'Computing the continuous-time A matrix'
            else:
                print 'Computing the discrete-time A matrix with time step',self.dt
        
        if numModes is not None:
            self.numModes=numModes
        if self.numModes is None:
            self.numModes = min(len(directModeAdvancedPaths), len(adjointModePaths))
        
        self.A = N.zeros((self.numModes,self.numModes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        numRowsPerChunk = self.maxFieldsPerProc - 1
        
        for startRowNum in range(0,self.numModes,numRowsPerChunk):
            endRowNum = min(startRowNum+numRowsPerChunk,self.numModes)
            #a list of 'row' adjoint modes (row because Y' has rows of adjoint snaps)
            adjointModes = [self.load_field(adjointPath) \
                for adjointPath in adjointModePaths[startRowNum:endRowNum]] 
              
            #now read in each column (direct modes advanced dt or time deriv)
            for colNum,advancedPath in enumerate(directModeAdvancedPaths[:self.numModes]):
                advancedMode = self.load_field(advancedPath)
                for rowNum in range(startRowNum,endRowNum):
                  self.A[rowNum,colNum] = \
                      self.inner_product(adjointModes[rowNum-startRowNum], advancedMode)

        self.save_mat(self.A, APath)
        print '----- A matrix saved to',APath,'------'

      
    def form_B(self, BPath, inputPaths, adjointModePaths, dt, numModes=None):
        """
        Forms the B matrix, either continuous or discrete time.
        
        Computes inner products of adjoint mode with sensor inputs.
        
        BPath 
          is where the B matrix will be saved
        inputPaths 
          is a list of the actuator fields' files (spatial representation
          of the B matrix in the full system).
          THE ORDER IS IMPORTANT. The order of the input files determines 
          the order of the actuators in the ROM and must be kept track of.
        adjointModePaths
          is a list of paths to the adjoint modes
        dt
          Set dt = 0 for continuous time systems.
        numModes
          number of modes/states to keep in the ROM.
        """
        
        if self.dt is None:
            self.dt = dt
        elif self.dt != dt:
            print "WARNING: dt values are inconsistent, using new value",dt
            self.dt = dt
        
        if numModes is not None:
            self.numModes = numModes
        if self.numModes is None:
            self.numModes = len(adjointModePaths)
            
        numInputs = len(inputPaths)
        self.B = N.zeros((self.numModes,numInputs))
        
        numRowsPerChunk = self.maxFieldsPerProc - 1
        for startRowNum in range(0,self.numModes,numRowsPerChunk):
            endRowNum = min(startRowNum+numRowsPerChunk,self.numModes)
            #a list of 'row' adjoint modes
            adjointModes = [self.load_field(adjointPath) \
                for adjointPath in adjointModePaths[startRowNum:endRowNum]] 
                
            #now read in each column (actuator fields)
            for colNum,inputPath in enumerate(inputPaths):
                inputField = self.load_field(inputPath)
                for rowNum in range(startRowNum,endRowNum):
                    self.B[rowNum,colNum] = \
                      self.inner_product(adjointModes[rowNum-startRowNum],inputField)
        
        if self.dt != 0:
            self.B *= self.dt
            
        self.save_mat(self.B, BPath)
        if self.dt!=0:
            print '----- B matrix, discrete-time, saved to',BPath,'-----'
        else:
            print '----- B matrix, continuous-time, saved to',BPath,'-----'
        
    
    def form_C(self, CPath, outputPaths, directModePaths, numModes=None):
        """
        Forms the C matrix, either continuous or discrete.
        
        Computes inner products of adjoint mode with sensor inputs.
        
        CPath 
          is where the C matrix will be saved
        outputPaths
          is a list of the senor fields' files (spatial representation
          of the C matrix in the full system).
          THE ORDER IS IMPORTANT. The order of the output files determines 
          the order of the sensors in the ROM and must be kept track of.
        directModePaths 
          is a list of paths to the direct modes
        numModes
          number of modes/states to keep in the ROM.
        
        Note: dt does not matter for the C matrix.
        """
        if numModes is not None:
            self.numModes = numModes
        if self.numModes is None:
            self.numModes = len(directModePaths)
        
        numOutputs = len(outputPaths)
        self.C = N.zeros((numOutputs,self.numModes))
        numColsPerChunk = self.maxFieldsPerProc - 1
        
        for startColNum in range(0,self.numModes,numColsPerChunk):
            endColNum = min(startColNum+numColsPerChunk,self.numModes)
            
            #a list of 'row' adjoint modes
            directModes = [self.load_field(directPath) \
                for directPath in directModePaths[startColNum:endColNum]]
            
            #now read in each row (outputs)
            for rowNum,outputPath in enumerate(outputPaths):
                outputField = self.load_field(outputPath)
                for colNum in range(startColNum,endColNum):
                    self.C[rowNum,colNum] = \
                        self.inner_product(outputField,directModes[colNum-startColNum])      
  
        self.save_mat(self.C, CPath)
        print '----- C matrix saved to',CPath,'-----'
    
    