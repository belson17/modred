
import util
import numpy as N


# Base class
class ModalDecomp(object):
    """
    Modal Decomposition base class

    This parent class is designed for the implementation of algorithms that take 
    a  set of data (snapshots) and turn them into modes.  Each class will 
    implement a method to perform some sort of decomposition, e.g. using POD, 
    BPOD, or DMD.  The results of this decomposition will be used to construct 
    modes by taking linear combinations of the original data snapshots.

    """
    
    def __init__(self,load_snap=None, save_mode=None, save_mat=None, inner_product=None,
                maxSnapsInMem=100,numProcs=None):
        """
        Modal decomposition constructor.
    
        This constructor sets the default values for data members common to all
        derived classes.  All derived classes should be sure to invoke this
        constructor explicitly.
        """
        self.load_snap = load_snap
        self.save_mode = save_mode
        self.save_mat = save_mat
        self.inner_product = inner_product
        self.maxSnapsInMem=maxSnapsInMem
        self.mpi = util.MPI(numProcs=numProcs)
    
    def _compute_inner_product_chunk(self,rowSnapPaths,colSnapPaths):
        """ Computes the inner products of snapshots in memory-efficient chunks
        
        The 'chunk' refers to the fact that within this method, the snapshots
        are read in memory-efficient ways such that they are not all in memory at 
        once. This results in finding 'chunks' of the eventual matrix 
        that is returned. 
        It is also true that this function is meant to be 
        used in parallel - individually on each processor. In this case,
        each processor has different lists of snapshots to take inner products
        of.
        rows = number of row snapshot files passed in (BPOD adjoint snaps)
        columns = number column snapshot files passed in (BPOD direct snaps)
        Currently this method only supports finding rectangular chunks.
        In the future this method can be expanded to find more general shapes
        of the matrix.
        It returns a matrix with the above number of rows and columns.
        """
        
        numRowSnaps = len(rowSnapPaths)
        numColSnaps = len(colSnapPaths)
        #enforce that there are more columns than rows for efficiency
        if numRowSnaps > numColSnaps:
            transpose = True
            temp = rowSnapPaths
            rowSnapPaths = colSnapPaths
            colSnapPaths = temp
            temp = numRowSnaps
            numRowSnaps = numColSnaps
            numColSnaps = temp
                
        #These two variables set the chunks of the X and Y matrices that are read in
        #at each step.
        if self.maxSnapsInMem > numrowSnaps:
            numRowsPerChunk = numrowSnaps
        else:
            numRowsPerChunk = self.maxSnapsInMem - 1 #row snapshots
        numColsPerChunk = 1 #forward snapshots per chunk in memory at once
        
        innerProductMatChunk = N.mat(N.zeros((numRows,numCols)))
        
        startColNum = 0
        startRowNum = 0
         
        while startRowNum < numrowSnaps: #read in another set of snaps
            if startRowNum + numRowsPerChunk > numrowSnapshots:
                #then a typical "chunk" is too large, go only to the end.
                endRowNum = numRows
            else:
                endRowNum = startRowNum + numRowsPerChunk
          
            #load in the snapshots from startColNum
            rowSnaps = [] # a list of snapshot objects
            for rowFile in rowFiles[startRowNum:endRowNum]:
                #print 'reading snapshot',file
                rowSnaps.append(self.load_snap(rowFile))
         
            while startColNum < numcolSnaps:
                if startColNum + numColsPerChunk > numcolSnaps:
                    endColNum = numcolSnaps
                else:
                    endColNum = startColNum + numColsPerChunk
                colSnaps = []
                for colFile in colFiles[startColNum:endColNum]:
                    colSnaps.append(self.load_snap(colFile))
              
                #With the chunks of the "X" and "Y" matrices, find chunk
                for rowNum in range(startRowNum,endRowNum):
                    for colNum in range(startColNum,endColNum):
                        innerProductMatChunk[rowNum,colNum] = \
                          self.inner_product(rowSnaps[rowNum-startRowNum],
                          colSnaps[colNum-startColNum])
                        #print 'formed H['+str(rowNum)+','+str(colNum)+'] of'+
                        #  'H['+str(numRows)+','+str(numCols)+']'
       
            print '---- Formed a',numRows,'by',numCols,'chunk of the Hankel matrix ----'
            if transpose: innerProductMatChunk=innerProductMatChunk.T
        return innerProductMatChunk
    
    # Common method for computing modes from snapshots and coefficients
    def _compute_modes(self,modeNumList,modePath,snapPaths,buildCoeffMat,
        indexFrom=1):
        """
        A common method to compute and save modes from snapshots.
        
        modeNumList - Indices of modes to compute.
        modePath - Full path to mode location, e.g /home/tmp/u_%d.out.
        indexFrom - Choose to index modes starting from 0 or 1.
        snapPaths - A list paths to files from which snapshots can be loaded.
        buildCoeffMat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth mode.
        
        This method works in parallel to compute the modes evenly among
        all of the processors available (see self.mpi).
        It also will never have more than self.maxSnaps snapshots/modes
        in memory simultaneously.
        """
        #Determine the processor mode assignments.

        if len(snapPaths) > buildCoeffMat.shape[0]:
            raise ValueError('coefficient matrix has fewer rows than number'+
              'of snap paths')
        
        for modeNum in modeNumList:
            if modeNum-indexFrom > len(snapPaths):
                raise ValueError('Cannot form more modes than number of '+
                  'snapshots')

        if len(modeNumList) < self.mpi.numProcs:
            raise util.MPIError('Cannot find fewer modes than number of procs, '+
              'lower the number of procs')       
              
        modeNumProcAssignments = self.mpi.find_proc_assignments(modeNumList)
        
        #Pass the work to individual processors. Each computes and saves
        #the mode numbers from rowNumProcAssigments[n] (a list of mode numbers)
        # There is no required mpi.gather command, modes are saved to file directly.
        self._compute_modes_chunk(modeNumProcAssignments[self.mpi.rank],
            modePath,snapPaths,buildCoeffMat,indexFrom=indexFrom)
  
    def _compute_modes_chunk(self,modeNumList,modePath,snapPaths,buildCoeffMat,
      indexFrom=1):
        """
        Computes a given set of modes in memory-efficient chunks.
        
        The size of the chunks used depends on the parameter self.maxSnaps,
        which is the maximum number of snapshots that will fit in memory 
        simultaneously. In parallel, this method is executed by each processor
        with a different modeNumList.
        """
       
        numSnaps = len(snapPaths)
        numModes = len(modeNumList)

        if numSnaps > buildCoeffMat.shape[0]:
            raise ValueError('coefficient matrix has fewer rows than number '+
              'of snap paths')
        for modeNum in modeNumList:
            if modeNum-indexFrom > len(snapPaths):
                raise ValueError('Cannot form more modes than number of '+
                  'snapshots')
        if numSnaps < buildCoeffMat.shape[0]:
            print 'Warning - fewer snapshot paths than rows in the coeff matrix'
            print '  some rows of coeff matrix will not be used!'
        
        #The truncated matrices, not sure where they belong right now.
        #V1 = N.mat(Vstar[0:numModes,:]).H
        #E1 = E[0:numModes]
        #U1 = N.mat(U[:,0:numModes])
        
        if numModes >= self.maxSnapsInMem: #more modes than can be in memory
            numModesPerChunk = self.maxSnapsInMem - 1
        else:
            numModesPerChunk = numModes
       
        if numModes < self.maxSnapsInMem: #form all modes at once
            numSnapsPerChunk = 1
            numModesPerChunk = numModes
        else:
            numSnapsPerChunk = 1
            numModesPerChunk = self.maxSnapsInMem-numSnapsPerChunk
        
        #Loop over each chunk
        for startModeIndex in range(0,numModes,numModesPerChunk):
            endModeIndex = min(startModeIndex+numModesPerChunk,numModes)
            modesChunk = [] #List of modes that are being computed
            
            # Sweep through all snapshots, adding "levels", ie adding 
            # the contribution
            # of each snapshot to each mode. After one sweep of all
            # snapshots,
            # the current modesChunk is complete with modes numbered:
            # modeList[startModeIndex:endModeIndex], which are then saved to disk.
            # This process is repeated until all modes are computed and saved.
            for startSnapNum in xrange(0,numSnaps,numSnapsPerChunk):
                endSnapNum = min(startSnapNum+numSnapsPerChunk,numSnaps)
                snaps=[]
                
                for snapNum in xrange(startSnapNum,endSnapNum):
                    snaps.append(self.load_snap(snapPaths[snapNum]))
                    #Might be able to eliminate this loop for array 
                    #multiplication (after tested)
                    #But this could increase memory usage, be careful
                for modeIndex in xrange(startModeIndex,endModeIndex):
                    for snapNum in xrange(startSnapNum,endSnapNum):
                        modeLevel = snaps[snapNum-startSnapNum]*\
                          buildCoeffMat[snapNum, \
                            modeNumList[modeIndex]-indexFrom]
                        if modeIndex-startModeIndex>=len(modesChunk): 
                            #the mode list isn't full, must be created
                            modesChunk.append(modeLevel) 
                        else: #mode list exists
                            modesChunk[modeIndex-startModeIndex] += modeLevel
            #after summing all snapshots contributions to current modes, 
            #save modes
            for modeIndex in xrange(startModeIndex,endModeIndex):
                self.save_mode(modesChunk[modeIndex-startModeIndex],
                  modePath%(modeNumList[modeIndex]))
            
            
        
        