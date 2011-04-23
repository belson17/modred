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
        
    def __init__(self, load_field=None, save_field=None, save_mat=\
        util.save_mat_text, inner_product=None, maxFieldsPerNode=None, 
        numNodes=None, snapPaths=None, singVals=None, singVecs=None, 
        correlationMat=None, verbose=False):
        """
        POD constructor
        
        The POD class is a subclass of the BPOD class, where we take the 
        adjoint snapshots to be the direct snapshots.
        """
        ModalDecomp.__init__(self, load_field=load_field, save_field=save_field,
            save_mat=save_mat, inner_product=inner_product, maxFieldsPerNode=\
            maxFieldsPerNode, numNodes=numNodes, verbose=verbose)

        self.snapPaths = snapPaths        
        self.singVecs = singVecs
        self.singVals = singVals
        self.correlationMat = correlationMat
  
        
    def compute_decomp(self, singVecsPath=None, singValsPath=None, 
        snapPaths=None):
        """
        Compute POD decomposition
        """
        if snapPaths is not None:
            self.snapPaths = snapPaths
        if self.snapPaths is None:
            raise util.UndefinedError('snapPaths is not given')

        # Compute Hankel matrix decomposition, using dir snaps as adj snaps
        # Then save the decomposition matrices as needed, to file/data members
        # Only one set of sing vecs needs to be saved for POD (symmetry)
        self.correlationMat = self._compute_hankel(self.snapPaths,
            self.snapPaths)
        self.singVecs, self.singVals, dummy = util.svd(self.correlationMat)
        del dummy
        
    def compute_modes(self, modeNumList, modePath, indexFrom=1, snapPaths=\
        None):
        raise util.UndefinedError('This functionality has not yet been '+\
            'implemented')
        """
        if snapPaths is not None:
            self.snapPaths = snapPaths
        if self.snapPaths is None:
            raise util.UndefinedError('snapPaths is not given')

        # Call base class method w/correct coefficients for modes
        self.singVecs = 1
        self.singVals = 1
        ModalDecomp.compute_modes(self, modeNumList, modePath, self.snapPaths,
            self.singVals*self.singVecs, indexFrom=indexFrom )
        print 'Implemented to compute POD modes.
        """
