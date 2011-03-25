#!/usr/bin/env python

import util
import numpy as N
""" This is a class which computes the ROM matrices from BPOD
modes and assuming an LTI plant. Eventually it should be made a derived class of
a ROM class that computes Galerkin projections for generic PDEs. Each
PDE would be a derived class"""


class BPODROM(object):
    def __init__(self,adjointModePaths=None,directModePaths=None,
      directDerivModePaths=None,inner_product=None,save_mat=util.save_mat_text,
      load_mode=None,save_mode=None,numModes=None,maxSnapsInMem=100,numProcs=1):
        
        self.adjointModePaths=adjointModePaths
        self.directModePaths=directModePaths
        self.directDerivModePaths=directDerivModePaths
        self.inner_product=inner_product
        self.save_mat=save_mat
        self.load_mode=load_mode
        self.save_mode=save_mode
        self._numProcs=numProcs
        self.maxSnapsInMem=maxSnapsInMem
        self.numModes = numModes
    
    def compute_mode_derivs(self,modePaths,modeDtPaths,modeDerivPaths,dt):
        """
        Computes time derivatives of modes.
        
        It reads in modes from modePaths and modeDtPaths, and simply
        subtracts them and divides by dt. This requires both __mul__
        and __add__ to be defined for the mode object.
        """
        if (self.load_mode is None):
            raise util.UndefinedError('no load_mode defined')
        if (self.save_mode is None):
            raise util.UndefinedError('no save_mode defined')
        
        numModes = min(len(modePaths),len(modeDtPaths),len(modeDerivPaths))
        print 'Computing derivatives of',numModes,'modes'
        
        for modeIndex in xrange(len(modeDtPaths)):
            mode = self.load_mode(modePaths[modeIndex])
            modeDt = self.load_mode(modeDtPaths[modeIndex])
            self.save_mode((modeDt + (mode*(-1.))*(1./dt), modeDerivPaths[modeIndex])            
        
    
    def form_A(self,APath,directDerivModePaths=None,adjointModePaths=None,
      numModes=None):
        if directDerivModePaths is not None:
            self.directDerivModePaths=directDerivModePaths
        if adjointModePaths is not None:
            self.adjointModePaths=adjointModePaths
            
        if (self.directDerivModePaths is None):
            raise util.UndefinedError('No derivative of direct mode paths given')
        if (self.adjointModePaths is None):
            raise util.UndefinedError('No adjoint mode paths given')
            
        if numModes is not None:
            self.numModes=numModes
        if self.numModes is None:
            self.numModes = min(len(directDerivModePaths),
            len(adjointModePaths))
          
        print 'Computing',self.numModes,'modes of the A matrix'
        
        self.A = N.zeros((self.numModes,self.numModes))
        
        #reading in sets of modes to form A in chunks rather than all at once
        #Assuming all column chunks are 1 long
        numRowsPerChunk = self.maxSnapsInMem - 1
        
        for startRowNum in range(0,self.numModes,numRowsPerChunk):
            endRowNum = min(startRowNum+numRowsPerChunk,self.numModes)
            adjointModes = [] #a list of 'row' adjoint modes
            for adjointPath in self.adjointModePaths[startRowNum:endRowNum]:
                adjointModes.append(self.load_mode(adjointPath))
              
            #now read in each column (forward time modes advanced dt)
            for colNum,derivPath in enumerate(self.directDerivModePaths[:self.numModes]):
                derivMode = self.load_mode(derivPath)
                for rowNum in range(startRowNum,endRowNum):
                  self.A[rowNum,colNum] = self.inner_product(adjointModes[rowNum],
                  derivMode)

        self.save_mat(self.A,APath)
        print '---- A matrix formed, saved to',APath,'-----'

      
      
    def form_B(self,BPath,inputPaths,adjointModePaths=None,numModes=None):
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
        
        numInputs = len(inputPaths)
        self.B = N.zeros((self.numModes,numInputs))
        
        numRowsPerChunk = self.maxSnapsInMem - 1
        for startRowNum in range(0,self.numModes,numRowsPerChunk):
            endRowNum = min(startRowNum+numRowsPerChunk,self.numModes)
            adjointModes = [] #a list of 'row' adjoint modes
            for adjointPath in self.adjointModePaths[startRowNum:endRowNum]:
                adjointModes.append(self.load_mode(adjointPath))
              
            #now read in each column (forward time modes advanced dt)
            for colNum,inputPath in enumerate(inputPaths):
                inputField = self.load_mode(inputPath)
                for rowNum in range(startRowNum,endRowNum):
                    self.B[rowNum,colNum] = \
                      self.inner_product(adjointModes[rowNum],inputField)
      
        self.save_mat(self.B,BPath)
        print '----- B matrix is formed and saved to',BPath,'-----'
      
    
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
        numColsPerChunk = self.maxSnapsInMem - 1
        
        for startColNum in range(0,self.numModes,numColsPerChunk):
            endColNum = min(startColNum+numColsPerChunk,self.numModes)
            
            directModes = [] #a list of 'row' adjoint modes
            for directPath in self.directModePaths[startColNum:endColNum]:
                directModes.append(self.load_mode(directPath))
              
            #now read in each row (outputs)
            for rowNum,outputPath in enumerate(outputPaths):
                outputField = self.load_mode(outputPath)
                for colNum in range(startColNum,endColNum):
                    self.C[rowNum,colNum] = \
                      self.inner_product(outputField,directModes[colNum])      
  
        self.save_mat(self.C,CPath)
        print '----- C matrix is formed and saved to',CPath,'-----'
    
    