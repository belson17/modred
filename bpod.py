from modaldecomp import ModalDecomp
import util
import numpy as N

# Derived class
class BPOD(ModalDecomp):
    """
    Balanced Proper Orthogonal Decomposition
    
    Generate direct and adjoint modes from direct and adjoint simulation 
    snapshots.
    
    """
    
    def __init__(self, load_snap=None, save_mode=None, 
        save_mat=util.write_mat_text, inner_product=None, directSnapPaths=None, 
        adjointSnapPaths=None, LSingVecs=None, singVals=None, RSingVecs=None,
        hankelMat=None):
        """
        BPOD constructor
        
        load_snap - Function to load a snapshot given a filepath.
        save_mode - Function to save a mode given data and an output path.
        save_mat - Function to save a matrix.
        inner_product - Function to take inner product of two snapshots.
        directSnapPaths - List of filepaths from which direct snapshots can be
            loaded.
        adjointSnapPaths - List of filepaths from which direct snapshots can be
            loaded.
        """
        # Base class constructor defines common data members
        ModalDecomp.__init__(self, load_snap=load_snap, save_mode=save_mode, 
            save_mat=save_mat, inner_product=inner_product)

        # Additional data members
        self.directSnapPaths = directSnapPaths
        self.adjointSnapPaths = adjointSnapPaths
        
        # Data members that will be set after computation
        self.LSingVecs = LSingVecs
        self.singVals = singVals
        self.RSingVecs = RSingVecs
        self.hankelMat = hankelMat

        print 'BPOD constructor.'
        
    def compute_decomp(self, LSingVecsPath=None, singValsPath=None, 
        RSingVecsPath=None):
        """
        Compute BPOD decomposition
        
        LSingVecsPath - Output path for matrix of left singular vectors from 
            Hankel matrix SVD.
        singValsPath - Output path for singular values from Hankel matrix SVD.
        RSingVecsPath - Output path for matrix of right singular vectors from
            Hankel matrix SVD.
            
        """
        # Compute Hankel matrix decomposition
        # Then save the decomposition matrices as needed, to file/data members
        self.hankelMat = self._compute_hankel(self.directSnapPaths,
                                              self.adjointSnapPaths)
                                              
        self.LSingVecs, self.singVals, self.RSingVecs = self._svd( 
            self.hankelMat )
        
        print 'Computing BPOD decomposition.'
            
    def compute_direct_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for direct modes
        self.LSingVecs = 1
        self.singVals = 1
        ModalDecomp._compute_modes(self, modeNumList, modePath, 
            self.directSnapPaths, self.LSingVecs*self.singVals, 
            indexFrom )
        print 'Implemented to compute direct modes.'
    
    def compute_adjoint_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for adjoint modes
        self.RSingVecs = 1
        self.singVals = 1
        ModalDecomp._compute_modes(self, modeNumList, modePath, 
            self.directSnapPaths,self.RSingVecs*self.singVals, 
            indexFrom=indexFrom )
        print 'Implemented to compute adjoint modes.'

    def _compute_hankel(self,directSnapPaths,adjointSnapPaths):
        """ Computes the hankel matrix (in parallel) and returns it.
        
        This method assigns the task of computing the Hankel matrix
        into pieces for each processor, then passes this on to
        self._compute_hankel_chunk(...)."""
        
        numDirectSnaps = len(directSnapPaths)
        numAdjointSnaps = len(adjointSnapPaths)
        
        numRows = numAdjointSnaps
        numCols = numDirectSnaps
        
        rowNumProcAssignments = self.mpi.find_consec_proc_assignments(numRows)
        numRowsPerProc = rowNumProcAssignments[1]-rowNumProcAssignments[0]
        
        if self.mpi.rank == 0:
            print 'Each processor is responsible for',numRowsPerProc
            print 'rows. The assignments are',rowNumProcAssignments
      
            if numRowsPerProc > self.maxSnapsInMem:
                print 'Each processor will have to read the number of direct'
                print 'snapshots = ',str(numForwardSnapshots),'multiple times,'
                print 'increase num CPUs to',int(N.ceil(numRows/self.maxSnapsInMem))
                print 'to avoid this and get a big speedup'
         
        #Find all chunks of the Hankel matrix (if one proc, the whole matrix)
        hankelMatChunk=self._compute_hankel_chunk(directSnapPaths,
            adjointSnapPaths[rowNumProcAssignments[self.mpi.rank]:\
            rowNumProcAssignments[self.mpi.rank+1]])
                       
        #Gather list of chunks from each processor, ordered by rank
        hankelMatChunkList = self.mpi.comm.gather(hankelMatChunk,root=0)
        if self.mpi.rank==0:
            hankelMat = N.zeros((numAdjointSnaps,numDirectSnaps))
            for CPUNum in xrange(self.mpi.numCPUs):
            #concatenate the chunks of Hankel matrix
                hankelMat[rowNumProcAssignments[CPUNum]:\
                    rowNumProcAssignments[CPUNum+1],:] = \
                    hankelMatChunkList[CPUNum]
        return hankelMat
    
    def _svd(self,hankelMat):
        LSingVecs,singVals,RSingVecsStar = \
            N.linalg.svd(hankelMat,full_matrices=0)
        LSingVecs = N.matrix(LSingVecs)
        RSingVecs = N.matrix(RSingVecsStar).H
        if (len(singVals) < numModes): 
            #the min is in case there are diff nums of snapshots
            raise RuntimeError('Too few non-zero singular values for the number'+\
              'of modes!')
        singVals = N.matrix(singVals)
        return LSingVecs,singVals,RSingVecs
               
        #truncated matrices
        #V1 = N.mat(Vstar[0:numModes,:]).H
        #E1 = E[0:numModes]
        #U1 = N.mat(U[:,0:numModes])
  
        
    def _compute_hankel_chunk(self,directSnapPaths,adjointSnapPaths):
        """ Computes a chunk of the Hankel matrix prescribed by the path lists.
        
        A helper function that actually does the inner products and forms 
        a part of the Hankel matrix. The part it computes has
        # rows= number of adjoint snapshot files passed in
        # columns = number direct snapshot files passed in
        It returns a matrix with the above dimensions.
        """
        
        numDirectSnaps = len(directSnapPaths)
        numAdjointSnaps = len(adjointSnapPaths)
        
        #easier to think in terms of rows and cols
        numRows = numAdjointSnaps
        numCols = numDirectSnaps
        
        #These two variables set the chunks of the X and Y matrices that are read in
        #at each step.
        if self.maxSnapsInMem > numAdjointSnaps:
            numRowsPerChunk = numAdjointSnaps
        else:
            numRowsPerChunk = self.maxSnapsInMem - 1 #adjoint snapshots
        numColsPerChunk = 1 #forward snapshots per chunk in memory at once
        
        hankelMatChunk = N.mat(N.zeros((numRows,numCols)))
         
        startColNum = 0
        startRowNum = 0
         
        while startRowNum < numAdjointSnaps: #read in another set of snaps
            if startRowNum + numRowsPerChunk > numAdjointSnapshots:
                #then a typical "chunk" is too large, go only to the end.
                endRowNum = numRows
            else:
                endRowNum = startRowNum + numRowsPerChunk
          
            #load in the snapshots from startColNum
            adjointSnaps = [] # a list of snapshot objects
            for adjointFile in adjointFiles[startRowNum:endRowNum]:
                #print 'reading snapshot',file
                adjointSnaps.append(self.load_snap(adjointFile))
         
            while startColNum < numDirectSnaps:
                if startColNum + numColsPerChunk > numDirectSnaps:
                    endColNum = numDirectSnaps
                else:
                    endColNum = startColNum + numColsPerChunk
                directSnaps = []
                for directFile in directFiles[startColNum:endColNum]:
                    directSnaps.append(self.load_snap(directFile))
              
                #With the chunks of the "X" and "Y" matrices, find chunk of hankel
                for rowNum in range(startRowNum,endRowNum):
                    for colNum in range(startColNum,endColNum):
                        hankelMatChunk[rowNum,colNum] = self.inner_product( \
                        adjointSnaps[rowNum-startRowNum],
                        directSnaps[colNum-startColNum])
                        #print 'formed H['+str(rowNum)+','+str(colNum)+'] of'+
                        #  'H['+str(numRows)+','+str(numCols)+']'
       
            print '---- Formed a',numRows,'by',numCols,'chunk of the Hankel matrix ----'
        return hankelMatChunk
  
  
