# Parent class
from modaldecomp import ModalDecomp
from bpod import BPOD

# Common functions
import util

# Derived class
class POD(BPOD):
    """
    Proper Orthogonal Decomposition
    
    Generate orthonormal modes from simulation snapshots.
    
    """
        
    def __init__(self, load_snap=None, save_mode=None, save_mat=\
        util.save_mat_text, inner_product=None, maxSnapsInMem=100, numProcs=\
        None, snapPaths=None, singVals=None, singVecs=None, correlationMat=
        None):
        """
        POD constructor
        
        The POD class is a subclass of the BPOD class, where we take the 
        adjoint snapshots to be the direct snapshots.
        
        load_snap - Function to load a snapshot given a filepath.
        save_mode - Function to save a mode given data and an output path.
        save_mat - Function to save a matrix.
        inner_product - Function to take inner product of two snapshots.
        snapPaths - List of filepaths from which snapshots can be loaded.

        """
        ModalDecomp.__init__(self, load_snap=load_snap, save_mode=save_mode, 
            save_mat=save_mat, inner_product=inner_product, maxSnapsInMem=\
            maxSnapsInMem, numProcs=numProcs)

        self.snapPaths = snapPaths        
        self.singVecs = singVecs
        self.singVals = singVals
        self.correlationMat = correlationMat
  
        
    def compute_decomp(self, SingVecsPath=None, singValsPath=None ):
        """
        Compute POD decomposition
        
        singVecsPath - Output path for matrix of left singular vectors from 
            Hankel matrix SVD.
        singValsPath - Output path for singular values from Hankel matrix SVD.
            
        """
        # Compute Hankel matrix decomposition, using dir snaps as adj snaps
        # Then save the decomposition matrices as needed, to file/data members
        # Only one set of sing vecs needs to be saved for POD (symmetry)
        self.correlationMat = self._compute_hankel(self.snapPaths,
            self.snapPaths)
                                              
        self.singVecs, self.singVals, dummy = self._svd( 
            self.correlationMat )
        del dummy
        
        print 'Computing POD decomposition.'

    def compute_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for modes
        self.singVecs = 1
        self.singVals = 1
        ModalDecomp.compute_modes(self, modeNumList, modePath, self.snapPaths,
            self.singVals*self.singVecs, indexFrom=indexFrom )
        print 'Implemented to compute POD modes.'
