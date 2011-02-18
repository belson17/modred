
import util

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
                maxSnapsInMem=100,numCPUs=1):
        """
        Modal decomposition constructor.
    
        This constructor sets the default values for data members common to all
        derived classes.  All derived classes should be sure to invoke this
        constructor explicitly.
        """
        # TO DO: Set default values for various data members.
        print 'Modal decomposition constructor.'
        self.load_snap = load_snap
        self.save_mode = save_mode
        self.save_mat = save_mat
        self.inner_product = inner_product
        self.maxSnapsInMem=maxSnapsInMem
        self.mpi = util.MPI(numCPUs=numCPUs)
        
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
        # TO DO: add load/save functions as optional arguments here?
        # TO DO: add name for saving build coefficients?
        # The following function must be defined in each derived class!
        if indexFrom != 1: 
            raise UndefinedError('Not supported, set indexFrom to 1 please')
        #Determine the processor mode assignments.
        rowNumProcAssignments = self.mpi.findProcAssignments(modeNumList)
        
        #Pass the work to individual processors. Each computes and saves
        #the mode numbers from rowNumProcAssigments[n] (a list of mode numbers)
        # There is no required mpi.gather command, modes are saved to file directly.
        self._compute_modes_chunk(rowNumProcAssignments[self.mpi.rank],
            modePath,snapPaths,buildCoeffMat)
  
    def _compute_modes_chunk(self,modeNumList,modePath,snapPaths,buildCoeffMat):
        """
        Computes a given set of modes in memory-efficient chunks.
        
        The size of the chunks used depends on the parameter self.maxSnaps,
        which is the maximum number of snapshots that will fit in memory 
        simultaneously. In parallel, this method is executed by each processor
        with a different modeNumList.
        """
       
        numDirectSnaps = len(directSnapPaths)
        numAdjointSnaps = len(adjointSnapPaths)
        numModes = len(modeNumList)
        
        #The truncated matrices, not sure where they belong right now.
        #V1 = N.mat(Vstar[0:numModes,:]).H
        #E1 = E[0:numModes]
        #U1 = N.mat(U[:,0:numModes])
        
        if numModes >= self.maxSnapsInMem: #more modes than can be in memory
            numModesPerChunk = self.maxSnapsInMem - 1
        else:
            numModesPerChunk = numModes
       
        if numModes < self.maxSnaps: #form all modes at once
            numSnapsPerChunk = 1
            numModesPerChunk = numModes
        else:
            numSnapsPerChunk = 1
            numModesPerChunk = self.maxSnapsInMem-numSnapsPerChunk
        
        #Loop over each chunk
        startModeNum=0
        while startModeNum < numModes:
            modesChunk = [] #List of modes that are being computed
            if startModeNum + numModesPerChunks < numModes:
                endModeNum = startModeNum+numModesPerChunk
            else:
                endModeNum = numModes
            
            startSnapNum=0
            while startSnapNum < numSnaps:    
                if startSnapNum + numSnapsPerChunk < numSnaps:
                    endSnapNum = startSnapNum+numSnapsPerChunk
                else:
                    endSnapNum = numSnaps
                
                #Sweep through all snapshots, adding "levels", ie adding 
                # contributions
                # of each snapshot to each mode. After one sweep of all
                # snapshots,
                # the current modesChunk is complete with modes from
                # startModeNum to endModeNum, which are then saved to disk.
                # This process is repeated until all modes are done.
                for snapNum,snapPath in enumerate(snapPaths[startSnapNum:endSnapNum]):
                    snap = self.load_snap(snapPath)
                    #Might be able to eliminate this loop for array multiplication (after tested)
                    for modeNum in xrange(startModeNum,endModeNum):
                        if snapNum==0: 
                            #the mode list is empty, must be created with appends
                            modesChunk.append(snap * buildCoeffMat[snapNum,modeNum])
                        else: #mode list exists
                            modesChunk[modeNum] += snap*buildCoeffMat[snapNum,modeNum]
                print 'Created the "level" of current mode chunk that is due to'
                print 'snapshot numbers',startSnapNum,endSnapNum
                
                startSnapNum = endSnapsPerChunk
            
            for modeNum in xrange(startModeNum,endModeNum):
                self.write_mode(modesChunk[modeNum],modePath%modeNum) #interface might be wrong!
            startModeNum = endModeNum
            
            
        
        