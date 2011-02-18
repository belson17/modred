# Parent class
from modaldecomp import ModalDecomp

# Common functions
import util

# Derived class
class BPOD(ModalDecomp):
    """
    Balanced Proper Orthogonal Decomposition
    
    Generate direct and adjoint modes from direct and adjoint simulation 
    snapshots.
    
    """
    
    def __init__(self, load_snap=None, save_mode=None, 
        save_mat=util.write_mat_text, inner_product=None, directSnapPaths=None, 
        adjointSnapPaths=None, LSingVecs=None, singVals=None, RSingVecs=None):
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
        self.LSingVecs, self.singVals, self.RSingVecs = self._compute_decomp( 
            self.directSnapPaths, self.adjointSnapPaths )
        
        print 'Computing BPOD decomposition.'
    
    # Private method takes two lists of snapshots and decomposes Hankel matrix
    def _compute_decomp(self, directSnapPaths, adjointSnapPaths):
        print 'Private implementation of Hankel matrix decomposition'
        # Temporary values
        LSingVecs = 1
        singVals = 1
        RSingVecs = 1
        return LSingVecs, singVals, RSingVecs
        
    def compute_direct_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for direct modes
        self.LSingVecs = 1
        self.singVals = 1
        ModalDecomp.compute_modes(self, modeNumList, modePath, 
            self.directSnapPaths, self.LSingVecs*self.singVals, 
            indexFrom )
        print 'Implemented to compute direct modes.'
    
    def compute_adjoint_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for adjoint modes
        self.RSingVecs = 1
        self.singVals = 1
        ModalDecomp.compute_modes(self, modeNumList, modePath, 
            self.directSnapPaths,self.RSingVecs*self.singVals, 
            indexFrom=indexFrom )
        print 'Implemented to compute adjoint modes.'

