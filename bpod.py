

from modaldecomp import ModalDecomp
import util
import numpy as N
import util 

class BPOD(ModalDecomp):
    """
    Balanced Proper Orthogonal Decomposition
    
    Generate direct and adjoint modes from direct and adjoint simulation 
    snapshots. BPOD inherits from ModalDecomp and uses it for low level
    functions.
    
    """
    
    def __init__(self, load_snap=None, save_mode=None, 
        save_mat=util.save_mat_text, load_mat=util.load_mat_text, 
        inner_product=None,
        maxSnapsInMem=100, numProcs=None, directSnapPaths=None, 
        adjointSnapPaths=None, LSingVecs=None, 
        singVals=None, RSingVecs=None, hankelMat=None, verbose=False):
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
        SVD of hankelMat gives:
        LSingVecs * singVals * RSingVecs* = hankelMat = "Y* X"
        hankelMat is adjoint modes "multiplied by" direct modes.
        """
        # Base class constructor defines common data members
        ModalDecomp.__init__(self, load_snap=load_snap, save_mode=save_mode, 
            save_mat=save_mat, inner_product=inner_product, maxSnapsInMem=\
            maxSnapsInMem, numProcs=numProcs,verbose=verbose)

        # Additional data members
        self.directSnapPaths = directSnapPaths
        self.adjointSnapPaths = adjointSnapPaths
        
        # Data members that will be set after computation
        self.LSingVecs = LSingVecs
        self.singVals = singVals
        self.RSingVecs = RSingVecs
        self.hankelMat = hankelMat
        self.load_mat = load_mat
        
    def load_decomp(self, LSingVecsPath, singValsPath, RSingVecsPath,
        load_mat=None):
        """
        Loads the decomposition matrices from file. 
        """
        if self.load_mat is None:
            raise UndefinedError('Must specify a load_mat function')
        if self.mpi._rank == 0:
            self.LSingVecs = self.load_mat(LSingVecsPath)
            self.singVals = N.squeeze(N.array(self.load_mat(singValsPath)))
            self.RSingVecs = self.load_mat(RSingVecsPath)
        else:
            self.LSingVecs = None
            self.singVals = None
            self.RSingVecs = None
        if self.mpi.parallel:
            self.LSingVecs = self.mpi.comm.bcast(self.LSingVecs, root=0)
            self.singVals = self.mpi.comm.bcast(self.singVals, root=0)
            self.RSingVecs = self.mpi.comm.bcast(self.LSingVecs, root=0)
    
    def save_decomp(self, hankelMatPath, LSingVecsPath, singValsPath, 
                    RSingVecsPath):
        """Save the decomposition matrices to file."""
        if self.mpi._rank == 0:
            self.save_mat(self.hankelMat,hankelMatPath)
            self.save_mat(self.LSingVecs,LSingVecsPath)
            self.save_mat(self.RSingVecs,RSingVecsPath)
            self.save_mat(self.singVals,singValsPath)
        
    
    def compute_decomp(self, hankelMatPath='hankelMat.txt',
      LSingVecsPath='LSingVecs.txt', 
      singValsPath='singVals.txt', RSingVecsPath='RSingVecs.txt',
      directSnapPaths=None,
      adjointSnapPaths=None):
        """
        Compute BPOD decomposition, forms the Hankel matrix and its SVD
        
        LSingVecsPath - Output path for matrix of left singular vectors from 
            Hankel matrix SVD.
        singValsPath - Output path for singular values from Hankel matrix SVD.
        RSingVecsPath - Output path for matrix of right singular vectors from
            Hankel matrix SVD.
        """
        
        if directSnapPaths is not None:
            self.directSnapPaths=directSnapPaths
        if adjointSnapPaths is not None:
            self.adjointSnapPaths=adjointSnapPaths

        if self.directSnapPaths is None:
            raise util.UndefinedError('directSnapPaths is not given')
        if self.adjointSnapPaths is None:
            raise util.UndefinedError('adjointSnapPaths is not given')
            
        ## TESTING, hankelmat is set already, UNCOMMENT LATER
        self.hankelMat = self._compute_hankel(
          self.directSnapPaths, self.adjointSnapPaths)

        if self.save_mat is not None and hankelMatPath is not None and \
          self.mpi._rank==0:
            self.save_mat(self.hankelMat,hankelMatPath)
        
        self.LSingVecs, self.singVals, self.RSingVecs = util.svd(
          self.hankelMat)
        
        if self.save_mat is not None:
            self.save_decomp(hankelMatPath, LSingVecsPath, singValsPath,
              RSingVecsPath)
        

    def compute_direct_modes(self, modeNumList, modePath, indexFrom=1,
      directSnapPaths=None):
        """
        Computes the direct modes and saves them to file.
        
        modeNumList - mode numbers to compute on this processor. This 
          includes the indexFrom, so if indexFrom=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 
        modePath - Full path to mode location, e.g /home/user/mode_%d.txt.
        indexFrom - Choose to index modes starting from 0, 1, or other.
        self.RSingVecs, self.singVals must exist or an UndefinedError.
        """
        if self.RSingVecs is None:
            raise util.UndefinedError('Must define self.RSingVec')
        if self.singVals is None:
            raise util.UndefinedError('Must define self.singVals')
            
        if directSnapPaths is not None:
            self.directSnapPaths=directSnapPaths
        
        buildCoeffMat = N.mat(self.RSingVecs) * \
          N.mat(N.diag(self.singVals**(-0.5)))

        self._compute_modes(modeNumList, modePath, 
            self.directSnapPaths, buildCoeffMat, 
            indexFrom=indexFrom )
    
    def compute_adjoint_modes(self, modeNumList, modePath, indexFrom=1,
      adjointSnapPaths=None):
        """
        Computes the adjoint modes and saves them to file.
        
        modeNumList - mode numbers to compute on this processor. This 
          includes the indexFrom, so if indexFrom=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 
        modePath - Full path to mode location, e.g /home/user/mode_%d.txt.
        indexFrom - Choose to index modes starting from 0, 1, or other.
        self.LSingVecs, self.singVals must exist or an UndefinedError.
        """
        
        if self.LSingVecs is None:
            raise UndefinedError('Must define self.LSingVecs')
        if self.singVals is None:
            raise UndefinedError('Must define self.singVals')
        if adjointSnapPaths is not None:
            self.adjointSnapPaths=adjointSnapPaths
        self.singVals = N.squeeze(N.array(self.singVals))
        
        buildCoeffMat = N.mat(self.LSingVecs) * \
          N.mat(N.diag(self.singVals**(-0.5)))
                 
        self._compute_modes(modeNumList, modePath, 
            self.adjointSnapPaths,buildCoeffMat,
            indexFrom=indexFrom )

    def _compute_hankel(self,directSnapPaths,adjointSnapPaths):
        """ Computes the hankel matrix and returns it.
        
        This method assigns the task of computing the Hankel matrix
        into pieces for each processor, then passes this on to
        self._compute_hankel_chunk(...). After _compute_hankel_chunk
        returns chunks of the Hankel matrix, they are concatenated
        into a completed, single, matrix on processor 0. This completed
        matrix is broadcast to all other processors (if parallel).
        
        REMEMBER - adjoint snaps correspond to ROWS, direct snaps to COLUMNS
        of the Hankel matrix.
        """
        
        numDirectSnaps = len(directSnapPaths)
        numAdjointSnaps = len(adjointSnapPaths)
        
        adjointSnapProcAssignments = \
          self.mpi.find_proc_assignments(range(numAdjointSnaps))
        
        if self.mpi._rank == 0 and \
          adjointSnapProcAssignments[0][-1]-adjointSnapProcAssignments[0][0] > \
          self.maxSnapsInMem:
              print 'Each processor will have to read the number of direct'
              print 'snapshots (',numDirectSnaps,') multiple times.'
              print 'Increase num processors to at least',\
              int(N.ceil(numDirectSnaps/self.maxSnapsInMem))
              print 'to avoid this and get a big speedup'
                      
        hankelMatChunk=self._compute_inner_product_chunk(
          adjointSnapPaths[adjointSnapProcAssignments[self.mpi._rank][0]: \
          adjointSnapProcAssignments[self.mpi._rank][-1]+1],
          directSnapPaths)
                       
        #Gather list of chunks from each processor, ordered by rank
        if self.mpi.parallel:
            hankelMatChunkList = self.mpi.comm.gather(hankelMatChunk,root=0)
            if self.mpi._rank==0:
                hankelMat = N.mat(N.zeros((numAdjointSnaps,numDirectSnaps)))
                for procNum in xrange(self.mpi._numProcs):
                #concatenate the chunks of Hankel matrix
                    hankelMat[adjointSnapProcAssignments[procNum][0]:\
                      adjointSnapProcAssignments[procNum][-1]+1] = \
                      hankelMatChunkList[procNum]
            else:
                hankelMat = None
            hankelMat = self.mpi.comm.bcast(hankelMat,root=0)
        else:
            hankelMat = hankelMatChunk
        
        return hankelMat
    

  
      
