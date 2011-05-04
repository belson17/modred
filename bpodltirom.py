
import util
import numpy as N
""" This is a class which computes the ROM matrices from BPOD
modes and assuming an LTI plant. Eventually it should be made a derived class of
a ROM class that computes Galerkin projections for generic PDEs. Each
PDE would be a derived class"""


class BPODROM(object):
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
      maxFieldsPerNode=2,
      numNodes=1):
        
        numProcs = 1 # not parallelized yet
        self.adjointModePaths=adjointModePaths
        self.directModePaths=directModePaths
        self.directDerivModePaths=directDerivModePaths
        self.directDtModePaths=directDtModePaths
        self.discrete = discrete
        self.dt = dt
        self.inner_product=inner_product
        self.save_mat=save_mat
        self.load_field=load_field
        self.save_field=save_field
        self._numProcs=numProcs
        self.maxFieldsPerNode=maxFieldsPerNode
        self.numNodes = 1
        self.maxFieldsPerProc = self.maxFieldsPerNode*self.numNodes/numProcs
        self.numModes = numModes
    
    def compute_mode_derivs(self,modePaths,modeDtPaths,modeDerivPaths,dt):
        """
        Computes time derivatives of modes.
        
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
        
    
    def form_A(self, APath, directDerivModePaths=None, directDtModePaths=None,
      adjointModePaths=None, numModes=None):
        """
        Computes the A matrix, either discrete or continuous time
        
        If directDerivModePaths is define and directDtModePaths is not,
        then the A matrix will be continuous time. In the other cases,
        the A matrix will be discrete time with the associated time step
        being the time step taken to compute the advanced modes saved at
        directDtModePaths from the direct modes saved at directModePaths.
        """
        
        if directDerivModePaths is not None:
            self.directDerivModePaths=directDerivModePaths
        if directDtModePaths is not None:
            self.directDtModePaths=directDtModePaths
        if adjointModePaths is not None:
            self.adjointModePaths=adjointModePaths
        
        if (self.adjointModePaths is None):
            raise util.UndefinedError('No adjoint mode paths given')    
        # Determine if finding the discrete or continuous time ROM matrix A
        if (self.directDerivModePaths is None and self.directDtModePaths is None):
            raise util.UndefinedError('No dt or deriv of direct mode paths given')
        elif (self.directDerivModePaths is not None and self.directDtModePaths is None):
            self.discrete = False
        elif (self.directDerivModePaths is None and self.directDtModePaths is not None):
            self.discrete = True
        else:
            print 'Warning: Both deriv and dt of direct modes are given '+\
                'assuming discrete, so using the dt information and not deriv'
            self.discrete = True
        
        # Create a new variable that contains either Dt or Deriv paths.
        if self.discrete:
            directAdvancedModePaths = self.directDtModePaths
        else: 
            directAdvancedModePaths = self.directDerivModePaths
        
        if numModes is not None:
            self.numModes=numModes
        if self.numModes is None:
            self.numModes = min(len(directAdvancedModePaths),
            len(adjointModePaths))
        
        if self.discrete:
            print 'Computing',self.numModes,'modes of the discrete-time A matrix'
        else:
            print 'Computing',self.numModes,'modes of the continuous-time A matrix'
        self.A = N.zeros((self.numModes,self.numModes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        numRowsPerChunk = self.maxFieldsPerProc - 1
        
        for startRowNum in range(0,self.numModes,numRowsPerChunk):
            endRowNum = min(startRowNum+numRowsPerChunk,self.numModes)
            adjointModes = [] #a list of 'row' adjoint modes
            for adjointPath in self.adjointModePaths[startRowNum:endRowNum]:
                adjointModes.append(self.load_field(adjointPath))
              
            #now read in each column (forward time modes advanced dt)
            for colNum,derivPath in enumerate(directAdvancedModePaths[:self.numModes]):
                derivMode = self.load_field(derivPath)
                for rowNum in range(startRowNum,endRowNum):
                  self.A[rowNum,colNum] = self.inner_product(adjointModes[rowNum-startRowNum],
                  derivMode)

        self.save_mat(self.A,APath)
        print '----- A matrix saved to',APath,'------'

      
      
    def form_B(self,BPath,inputPaths,adjointModePaths=None,numModes=None,
        discrete = None, dt=None):
        """Forms the B matrix, inner product of adjoint mode with sensor input
        inpuPaths is a list of the input files, representing the actuator field.
        THE ORDER IS IMPORTANT HERE. The order of the input files determines 
        the order
        of the actuators in the ROM and must be kept track of."""
        
        if adjointModePaths is not None:
            self.adjointModePaths = adjointModePaths
        if self.adjointModePaths is None:
            raise util.UndefinedError('No adjoint mode paths given')
        if numModes is not None:
            self.numModes = numModes
        if self.numModes is None:
            self.numModes = len(adjointModePaths)
        if discrete is not None:
            self.discrete = discrete
        if self.discrete is None:
            print 'Warning: did not specify discrete or continuous time, assuming '+\
            'discrete'
            self.discrete = True
        if dt is not None: self.dt = dt
        if self.discrete and self.dt is None:
            raise ValueError('Must specify a dt for a discrete-time B matrix')
            
        numInputs = len(inputPaths)
        self.B = N.zeros((self.numModes,numInputs))
        
        numRowsPerChunk = self.maxFieldsPerProc - 1
        for startRowNum in range(0,self.numModes,numRowsPerChunk):
            endRowNum = min(startRowNum+numRowsPerChunk,self.numModes)
            adjointModes = [] #a list of 'row' adjoint modes
            for adjointPath in self.adjointModePaths[startRowNum:endRowNum]:
                adjointModes.append(self.load_field(adjointPath))
              
            #now read in each column (forward time modes advanced dt)
            for colNum,inputPath in enumerate(inputPaths):
                inputField = self.load_field(inputPath)
                for rowNum in range(startRowNum,endRowNum):
                    self.B[rowNum,colNum] = \
                      self.inner_product(adjointModes[rowNum-startRowNum],inputField)
        if self.discrete:
            self.B *= self.dt
            
        self.save_mat(self.B,BPath)
        if self.discrete:
            print '----- B matrix, discrete-time, saved to',BPath,'-----'
        else:
            print '----- B matrix, continuous-time, saved to',BPath,'-----'
        
    
    def form_C(self,CPath,outputPaths,directModePaths=None,numModes=None):
        """Forms the C matrix, multiplication of adjoint mode with sensor input
        outputFiles is a list of the output files, representing the sensor fields.
        THE ORDER IS IMPORTANT HERE. The order of the output files determines the order
        of the sensors in the ROM and must be kept track of."""
        
        if directModePaths is not None:
            self.directModePaths = directModePaths
        if self.directModePaths is None:
            raise util.UndefinedError('No adjoint mode paths given')
        if numModes is not None:
            self.numModes = numModes
        if self.numModes is None:
            self.numModes = len(directModePaths)
        
        numOutputs = len(outputPaths)
        self.C = N.zeros((numOutputs,self.numModes))
        numColsPerChunk = self.maxFieldsPerProc - 1
        
        for startColNum in range(0,self.numModes,numColsPerChunk):
            endColNum = min(startColNum+numColsPerChunk,self.numModes)
            
            directModes = [] #a list of 'row' adjoint modes
            for directPath in self.directModePaths[startColNum:endColNum]:
                directModes.append(self.load_field(directPath))
              
            #now read in each row (outputs)
            for rowNum,outputPath in enumerate(outputPaths):
                outputField = self.load_field(outputPath)
                for colNum in range(startColNum,endColNum):
                    self.C[rowNum,colNum] = \
                      self.inner_product(outputField,directModes[colNum-startColNum])      
  
        self.save_mat(self.C,CPath)
        print '----- C matrix saved to',CPath,'-----'
    
    