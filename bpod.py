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
    
    def __init__(self,loadSnap=None,saveMode=None,saveMat=util.writeMatToText):
        """
        BPOD constructor
        
        loadSnap - Function to load a snapshot given a filepath.
        saveMode - Function to save a mode given data and an output path.
        saveMat - Function to save a matrix.
        
        """
        # TO DO: add more optional arguments?  
        # TO DO: Set default values for various data members.
        print 'BPOD constructor.'
        
    def computeDecomp(self, directSnaps=None, directSnapPaths=None, 
        adjointSnaps=None, adjointSnapPaths=None, LSingVecsPath=None,
        singValsPath=None, RSingVecsPath=None):
        """
        Compute BPOD decomposition
        
        directSnaps - Iterable container of direct snapshots.
        directSnapPaths - List of paths to files containing direct snapshots.
        adjointSnaps - Iterable container of adjoint snapshots.
        adjointSnapPaths - List of paths to files containing adjoint snapshots.
        LSingVecsPath - Output path for matrix of left singular vectors from 
            Hankel matrix SVD.
        singValsPath - Output path for singular values from Hankel matrix SVD.
        RSingVecsPath - Output path for matrix of right singular vectors from
            Hankel matrix SVD.
            
        """
        # TO DO: add load/save functions as optional arguments here?
        # TO DO: add name for saving build coefficients?
        print 'Computing BPOD decomposition.'