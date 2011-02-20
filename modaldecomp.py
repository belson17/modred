
import util
import numpy as N


# Base class
class ModalDecomp(object):
    """
    Modal Decomposition base class

    This parent class is designed for the implementation of algorithms that
    take 
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
        """ Computes inner products of snapshots in memory-efficient chunks
        
        The 'chunk' refers to the fact that within this method, the 
        snapshots
        are read in memory-efficient ways such that they are not all in
        memory at 
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
        
        numRows = len(rowSnapPaths)
        numCols = len(colSnapPaths)
        #enforce that there are more columns than rows for efficiency
        if numRows > numCols:
            transpose = True
            temp = rowSnapPaths
            rowSnapPaths = colSnapPaths
            colSnapPaths = temp
            temp = numRows
            numRows = numCols
            numCols = temp
        else: 
            transpose = False
                
        #These two variables set the chunks of the matrices that are read in
        #at each step.
        numColsPerChunk = 1
        numRowsPerChunk = self.maxSnapsInMem-numColsPerChunk         
        
        innerProductMatChunk = N.mat(N.zeros((numRows,numCols)))
        
        for startRowNum in range(0,numRows,numRowsPerChunk):
            endRowNum = min(numRows,startRowNum+numRowsPerChunk)
            
            rowSnaps = []
            for rowPath in rowSnapPaths[startRowNum:endRowNum]:
                rowSnaps.append(self.load_snap(rowPath))
         
            for startColNum in range(0,numCols,numColsPerChunk):
                endColNum = min(numCols,startColNum+numColsPerChunk)

                colSnaps = []
                for colPath in colSnapPaths[startColNum:endColNum]:
                    colSnaps.append(self.load_snap(colPath))
              
                #With the chunks of the row and column matrices,
                #find inner products
                for rowNum in range(startRowNum,endRowNum):
                    for colNum in range(startColNum,endColNum):
                        innerProductMatChunk[rowNum,colNum] = \
                          self.inner_product(rowSnaps[rowNum-startRowNum],
                          colSnaps[colNum-startColNum])
            #print 'formed ['+str(rowNum+1)+','+str(colNum+1)+'] of'+\
            #    '['+str(numRows)+','+str(numCols)+'], completed '+\
            #    str(round(100.*(rowNum+1)*(colNum+1)/(numRows*numCols)))+\
            #'% of mat'
       
        if transpose: innerProductMatChunk=innerProductMatChunk.T
        return innerProductMatChunk
    
    
    # Common method for computing modes from snapshots and coefficients
    def _compute_modes(self,modeNumList,modePath,snapPaths,buildCoeffMat,
        indexFrom=1):
        """
        A common method to compute and save modes from snapshots.
        
        modeNumList - mode numbers to compute on this processor. This 
          includes the indexFrom, so if indexFrom=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 
        modePath - Full path to mode location, e.g /home/user/mode_%d.txt.
        indexFrom - Choose to index modes starting from 0, 1, or other.
        snapPaths - A list paths to files from which snapshots can be loaded.
        buildCoeffMat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth mode.
        
        This methods primary purpose is to evenly divide the tasks for each
        processor in parallel (see util.MPI). It then calls _compute_modes_chunk.
        Each processor then computes and saves
        the mode numbers assigned to it
        """
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

        #Pass the work to individual processors..
        self._compute_modes_chunk(modeNumProcAssignments[self.mpi.rank],
            modePath,snapPaths,buildCoeffMat,indexFrom=indexFrom)
        #There is no required mpi.gather command, modes are saved to file 

    def _compute_modes_chunk(self,modeNumList,modePath,snapPaths,buildCoeffMat,
      indexFrom=1):
        """
        Computes a given set of modes in memory-efficient chunks.
        
        modeNumList - mode numbers to compute on this processor. This 
          includes the indexFrom, so if indexFrom=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 
        modePath - Full path to mode location, e.g /home/user/mode_%d.txt.
        indexFrom - Choose to index modes starting from 0, 1, or other.
        snapPaths - A list paths to files from which snapshots can be loaded.
          This must be the FULL list of snapshots, even in parallel.
        buildCoeffMat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth mode.
            Like snapPaths, this must be the FULL buildCoeffMat, even in
            parallel.
        
        The 'chunk' refers to the fact that within this method, the snapshots
        are read in memory-efficient ways such that they are not all in 
        memory at 
        once. This results in finding 'chunks' of the eventual modes 
        that are saved to disk.
        It is also true that this function is meant to be 
        used in parallel - individually on each processor. In this case,
        each processor has different lists of snapshots to form modes from.
        The size of the chunks used depends on the parameter self.maxSnapsInMem
        which is the maximum number of snapshots that will fit in memory 
        simultaneously. In parallel, this method is executed by each processor
        with a different modeNumList. In this way, this method operates in
        a single-processor sense.
        
        
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
        
        numSnapsPerChunk = 1
        numModesPerChunk = self.maxSnapsInMem-numSnapsPerChunk
                
        #Loop over each chunk
        for startModeIndex in range(0,numModes,numModesPerChunk):
            endModeIndex = min(startModeIndex+numModesPerChunk,numModes)
            modesChunk = [] #List of modes that are being computed
            
            # Sweep through all snapshots, adding "levels", ie adding 
            # the contribution of each snapshot
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
            
            
        
        