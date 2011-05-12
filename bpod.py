from fieldoperations import FieldOperations
import util
import numpy as N
import util 

class BPOD(object):
    """
    Balanced Proper Orthogonal Decomposition
    
    Generate direct and adjoint modes from direct and adjoint simulation 
    snapshots. BPOD inherits from ModalDecomp and uses it for low level
    functions.
    
    """
    
    def __init__(self, load_field=None, save_field=None, save_mat=util.\
        save_mat_text, load_mat=util.load_mat_text, inner_product=None,
        maxFieldsPerNode=2, numNodes=1, verbose=True):
        """
        BPOD constructor
        
        load_field - Function to load a snapshot given a filepath.
        save_field - Function to save a mode given data and an output path.
        save_mat - Function to save a matrix.
        inner_product - Function to take inner product of two snapshots.
        verbose - True means print more information about progress and warnings
        numNodes - number of nodes used in this computation
        """
        self.verbose = verbose
        self.load_mat = load_mat
        self.save_mat = save_mat
        # Class that contains all of the low-level field operations
        # and parallelizes them.
        self.fieldOperations = FieldOperations(load_field=load_field, save_field=save_field,
            save_mat=save_mat, inner_product=inner_product, maxFieldsPerNode=\
            maxFieldsPerNode, numNodes=numNodes, verbose=verbose)
        self.mpi = util.MPIInstance
    

    def load_decomp(self, LSingVecsPath, singValsPath, RSingVecsPath):
        """
        Loads the decomposition matrices from file. 
        """
        if self.load_mat is None:
            raise UndefinedError('Must specify a load_mat function')
        if self.mpi.isRankZero():
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
    
    def save_hankel_mat(self, hankelMatPath):
        if self.save_mat is None:
            raise util.UndefinedError('save_mat not specified')
        elif self.mpi.isRankZero():
            self.save_mat(self.hankelMat, hankelMatPath)           
    
    def save_decomp(self, LSingVecsPath, singValsPath, RSingVecsPath):
        """Save the decomposition matrices to file."""
        if self.save_mat is None:
            raise util.UndefinedError('save_mat is undefined, cant save')
        elif self.mpi.isRankZero():
            self.save_mat(self.LSingVecs, LSingVecsPath)
            self.save_mat(self.RSingVecs, RSingVecsPath)
            self.save_mat(self.singVals, singValsPath)
                
    def compute_decomp(self, directSnapPaths=None, adjointSnapPaths=None,
        hankelMatPath=None):
        """
        Compute BPOD decomposition from given data.
        
        If directSnapPaths and adjointSnapPaths are given, then the Hankel
        matrix is formed, then the SVD is taken.
        If hankelMatPath is given, then the Hankel matrix is read from
        file and the SVD is taken.
        If no arguments are given, then the function fails.
        If both the paths and the hankelMatPath are given, the HankelMatPath
        is given priority since it requires less computation.
        """
        
        if hankelMatPath is not None:
            if self.load_mat is None:
                raise util.UndefinedError('load_mat is undefined')
            elif self.mpi.isRankZero():
                self.hankelMat = self.load_mat(hankelMatPath)
        elif directSnapPaths is not None and adjointSnapPaths is not None:
            self.directSnapPaths = directSnapPaths
            self.adjointSnapPaths = adjointSnapPaths
            self.hankelMat = self.fieldOperations.compute_inner_product_mat(\
                self.adjointSnapPaths, self.directSnapPaths)
        else:
            raise util.UndefinedError('Must provide either snap paths or '+\
                'hankel matrix path to bpod.compute_decomp')
                
        if self.mpi.isRankZero():
            self.LSingVecs, self.singVals, self.RSingVecs = util.svd(self.\
                hankelMat)
        else:
            self.LSingVecs = None
            self.RSingVecs = None
            self.singVals = None
        if self.mpi.isParallel():
            self.LSingVecs = self.mpi.comm.bcast(self.LSingVecs, root=0)
            self.singVals = self.mpi.comm.bcast(self.singVals, root=0)
            self.RSingVecs = self.mpi.comm.bcast(self.RSingVecs, root=0)
        
        #self.mpi.evaluate_and_bcast([self.LSingVecs,self.singVals,self.RSingVecs],\
        #  util.svd, arguments = [self.hankelMat])

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
            raise util.UndefinedError('Must define self.RSingVecs')
        if self.singVals is None:
            raise util.UndefinedError('Must define self.singVals')
            
        if directSnapPaths is not None:
            self.directSnapPaths = directSnapPaths
        if self.directSnapPaths is None:
            raise util.UndefinedError('Must specify directSnapPaths')
        
        buildCoeffMat = N.mat(self.RSingVecs) * N.mat(N.diag(self.singVals **\
            -0.5))

        self.fieldOperations._compute_modes(modeNumList, modePath, 
            self.directSnapPaths, buildCoeffMat, indexFrom=indexFrom)
    
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
        if self.adjointSnapPaths is None:
            raise util.UndefinedError('Must specify adjointSnapPaths')

        self.singVals = N.squeeze(N.array(self.singVals))
        
        buildCoeffMat = N.mat(self.LSingVecs) * N.mat(N.diag(self.singVals **\
            -0.5))
                 
        self.fieldOperations._compute_modes(modeNumList, modePath,
            self.adjointSnapPaths, buildCoeffMat, indexFrom=indexFrom)
    
