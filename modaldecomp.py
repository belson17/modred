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
    
    # Constructor
    def __init__(self,load_snap=None, save_mode=None, save_mat=None, inner_product=None):
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
        
    # Common method for computing modes from snapshots and coefficients
    def compute_modes(self, modeNumList, modePath, snapPaths, coeffMat, 
        indexFrom=1):
        """
        A common method to compute modes from snapshots.
        
        modeNumList - Indices of modes to compute.
        modePath - Full path to mode location, e.g /home/tmp/u_%d.out.
        indexFrom - Choose to index modes starting from 0 or 1.
        snapPaths - A list paths to files from which snapshots can be loaded.
        coeffMat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth mode.

        """
        # TO DO: add load/save functions as optional arguments here?
        # TO DO: add name for saving build coefficients?
        # The following function must be defined in each derived class!
        print 'General method to build modes from coefficients.'

    