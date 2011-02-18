# Parent class
from modaldecomp import ModalDecomp
from pod import POD

# Common functions
import util

# Derived class
class DMD(ModalDecomp):
    """
    Dynamic Mode Decomposition/Koopman Mode Decomposition
        
    Generate Ritz vectors from simulation snapshots.
    
    """
    
    def __init__(self, load_snap=None, save_mode=None, 
        save_mat=util.write_mat_text, inner_product=None, snapPaths=None,
        ritzVals=None, ritzVecs=None, ritzVecScaling=None, pod=None):
        """
        DMD constructor
        
        loadSnap - Function to load a snapshot given a filepath.
        saveMode - Function to save a mode given data and an output path.
        saveMat - Function to save a matrix.
        
        """
        # Base class constructor defines common data members
        ModalDecomp.__init__(self, load_snap=load_snap, save_mode=save_mode, 
            save_mat=save_mat, inner_product=inner_product)

        # Additional data members
        self.snapPaths = snapPaths
        
        # Data members that will be set after computation
        self.ritzVals = ritzVals
        self.ritzVecs = ritzVecs
        self.ritzVecScaling = ritzVecScaling
        self.pod = pod

        # TO DO: DMD objects should contain an internal POD object.
        print 'DMD constructor.'
        
    def compute_decomp(self, ritzEigvalsPath=None, ritzVecsPath=None, 
        ritzVecScalingPath=None, ):
        """
        Compute DMD decomposition
        
        ritzEigvalsPath - Output path for Ritz eigenvalues.
        ritzVecNormsPath - Output path for matrix of Ritz vectors.
            
        """
        # Check if a pod object has been loaded yet, then compute as necessary
        print 'Computing DMD decomposition.'
    
    def compute_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for modes
        self.ritzVecs = 1
        self.ritzVecScaling = 1
        ModalDecomp.compute_modes(self, modeNumList, modePath, 
            self.snapPaths,self.ritzVecs*self.ritzVecScaling, 
            indexFrom=indexFrom )
        print 'Implemented to compute DMD modes.'
        
        