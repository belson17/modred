# Common functions
import util

# Base class
class ModalDecomp:
    """
    Modal Decomposition base class

    This parent class is designed for the implementation of algorithms that take 
    a  set of data (snapshots) and turn them into modes.  Each class will 
    implement a method to perform some sort of decomposition, e.g. using POD, 
    BPOD, or DMD.  The results of this decomposition will be used to construct 
    modes by taking linear combinations of the original data snapshots.

    """
    def __init__(self):
        """
        Modal decomposition constructor.
    
        """
        # TO DO: Set default values for various data members.
        print 'Modal decomposition constructor.'
        
    def computeModes(self, modeNumList, modePath, indexFrom=1, snaps=None,
        snapPaths=None, coeffMat=None, coeffMatPath=None):
        """
        A common method to compute modes from snapshots.
        
        modeNumList - Indices of modes to compute.
        modePath - Full path to mode location, e.g /home/tmp/u_%d.out.
        indexFrom - Choose to index modes starting from 0 or 1.
        snaps - An iterable container of snapshots.
        snapPaths - A list paths to files from which snapshots can be loaded.
        coeffMat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth mode.
        coeffMatPath - Path to file from which coefficient matrix can be loaded.

        """
        # TO DO: add load/save functions as optional arguments here?
        # TO DO: add name for saving build coefficients?
        print 'Compute modes using build coefficient matrix.'


    