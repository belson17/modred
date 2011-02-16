# Parent class
from bpod import BPOD

# Common functions
import util

# Derived class
class POD(BPOD):
    """
    Proper Orthogonal Decomposition
    
    Generate orthonormal modes from simulation snapshots.
    
    """
    
    def __init__(self,loadSnap=None,saveMode=None,saveMat=util.writeMatToText):
        """
        POD constructor
        
        loadSnap - Function to load a snapshot given a filepath.
        saveMode - Function to save a mode given data and an output path.
        saveMat - Function to save a matrix.
        
        """
        # TO DO: add more optional arguments?  
        # TO DO: set default values for various data members.
        print 'POD constructor.'
        
    def computeDecomp(self, snaps=None, snapPaths=None, LSingVecsPath=None,
        singValsPath=None):
        """
        Compute POD decomposition
        
        snaps - Iterable container of snapshots.
        snapPaths - List of paths to files containing snapshots.
        LSingVecsPath - Output path for matrix of left singular vectors from 
            correlation matrix SVD.
        singValsPath - Output path for singular values from correlation matrix 
        SVD.
            
        """
        # TO DO: add load/save functions as optional arguments here?
        # TO DO: add name for saving build coefficients?
        print 'Computing POD decomposition.'
        BPOD.computeDecomp(self)