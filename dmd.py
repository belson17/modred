# Parent class
from modaldecomp import ModalDecomp
from pod import POD
import numpy as N

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
        buildCoeff=None, ritzVals=None, lowOrderEigVecs=None, ritzVecScaling=None, 
        modeEnergies=None, pod=None):
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
        self.buildCoeff = buildCoeff
        
        # Data members that will be set after computation
        self.ritzVals = ritzVals
        self.lowOrderEigVecs = lowOrderEigVecs
        self.ritzVecScaling = ritzVecScaling
        self.modeEnergies = modeEnergies
        self.pod = pod
        
        print 'DMD constructor.'
        
    def compute_decomp(self, ritzEigvalsPath=None, ritzVecsPath=None, 
        ritzVecScalingPath=None, ):
        """
        Compute DMD decomposition
        
        ritzEigvalsPath - Output path for Ritz eigenvalues.
        ritzVecNormsPath - Output path for matrix of Ritz vectors.
            
        """
        
        # Compute POD from snapshots (excluding last snapshot)
        if self.pod is None:
            self.pod = POD(self.load_snap, self.inner_product, 
                self.snapPaths[:-1])
            self.pod.compute_decomp()
        elif self.snaplist[:-1] != self.podsnaplist or len(snapPaths) !=\
            len(self.pod.snapPaths)+1 ):
            raise RuntimeError('Snapshot mistmatch between POD and DMD '+\
                'objects.')      
        _podSingValsMat = N.mat(N.diag(N.array(self.pod.singVals).squeeze()))
                
        # Inner product of snapshots w/POD modes
        numSnaps = len(self.SnapPaths)
        podModesStarTimesSnaps = N.mat(N.empty((numSnaps-1, numSnaps-1)))
        podModesStarTimesSnaps[:,0:-1] = self.pod.correlationMatrix[:,1:]
        podModesStarTimesSnaps[:,-1] = self._compute_inner_product_chunk( self.\
            snapPaths[:-1], self.snapPaths[-1] )
        podModesStarTimesSnaps = (_podSingValsMat ** -0.5) * self.pod.\
            singVecs.H * modesStarTimesSnaps
            
        # Reduced order linear system
        self.lowOrderLinearMap = podModesStarTimesSnaps * self.pod.singVecs * \
            (_podSingValsMat ** -0.5) 
        self.ritzValues, self.lowOrderEigVecs = N.linalg.eig(lowOrderLinearMap)
        
        # Scale Ritz vectors
        ritzVecsStarTimesInitSnap = self.lowOrderEigVecs.H * (_podSingValsMat \
            ** -0.5 ) * self.pod.singVecs.H, self.pod.correlationMat[:,0]
        self.ritzVecScaling = N.linalg.inv(self.lowOrderEigVecs.H, self.\
            lowOrderEigVecs) * ritzVecsStarTimesInitSnap

        # Compute mode energies
        self.buildCoeff = self.pod.singVecs * (_podSingValsMat ** -0.5) *\
            self.lowOrderEigVecs * N.diag(self.d)
        self.modeEnergies = N.sqrt(N.diag(buildCoeff.H * self.pod.\
            correlationMatXTX * buildCoeff)).real
        
    def compute_modes(self, modeNumList, modePath, indexFrom=1 ):
        # Call base class method w/correct coefficients for modes
        """ Add error checking for definition of various matrices"""
        ModalDecomp.compute_modes(self, modeNumList, modePath, 
            self.buildCoeff, indexFrom=indexFrom )
        print 'Implemented to compute DMD modes.'
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        