# Parent class
from modaldecomp import ModalDecomp

# Common functions
import util

# Derived class
class DMD(ModalDecomp):
    """
    Dynamic Mode Decomposition/Koopman Mode Decomposition
        
    Generate Ritz vectors from simulation snapshots.
    
    """
    
    def __init__(self,loadSnap=None,saveMode=None,saveMat=util.writeMatToText):
        """
        DMD constructor
        
        loadSnap - Function to load a snapshot given a filepath.
        saveMode - Function to save a mode given data and an output path.
        saveMat - Function to save a matrix.
        
        """
        # TO DO: add more optional arguments?  
        # TO DO: set default values for various data members.
        # TO DO: DMD objects should contain an internal POD object.
        print 'DMD constructor.'
        
    def computeDecomp(self, snaps=None, snapPaths=None, ritzEigvalsPath=None,
        ritzVecNormsPath=None):
        """
        Compute DMD decomposition
        
        snaps - Iterable container of snapshots.
        snapPaths - List of paths to files containing snapshots.
        ritzEigvalsPath - Output path for Ritz eigenvalues.
        ritzVecNormsPath - Output path for matrix of Ritz vectors.
            
        """
        # TO DO: add load/save functions as optional arguments here?
        # TO DO: add name for saving build coefficients?
        print 'Computing DMD decomposition.'