# Parent class
from fieldoperations import FieldOperations
from pod import POD
import numpy as N

# Common functions
import util

# Derived class
class DMD(object):
    """
    Dynamic Mode Decomposition/Koopman Mode Decomposition
        
    Generate Ritz vectors from simulation snapshots.
    
    """

    def __init__(self, load_field=None, save_field=None, save_mat=util.\
        save_mat_text, inner_product=None, maxFieldsPerNode=None, numNodes=1, 
        pod=None, verbose=True):
        """
        DMD constructor
        """
        # Base class constructor defines common data members
        self.fieldOperations = FieldOperations(self, load_field=load_field,\
            save_field=save_field,
            save_mat=save_mat, inner_product=inner_product,\
            maxFieldsPerNode=maxFieldsPerNode, numNodes=numNodes, \
            verbose=verbose)

        self.verbose = verbose
        self.load_mat = load_mat
        self.save_mat = save_mat

        # Data members that will be set after computation
        self.pod = pod
        
        
    def compute_decomp(self, ritzValsPath=None, modeNormsPath=None, 
        buildCoeffPath=None, snapPaths=None):
        """
        Compute DMD decomposition
        """
         
        if snapPaths is not None:
            self.snapPaths = snapPaths
        if self.snapPaths is None:
            raise util.UndefinedError('snapPaths is not given')

        # Compute POD from snapshots (excluding last snapshot)
        if self.pod is None:
            self.pod = POD(load_field=self.load_field, inner_product=self.\
                inner_product, snapPaths=self.snapPaths[:-1], maxFieldsPerNode=\
                self.maxFieldsPerNode, numNodes=self.numNodes, verbose=self.\
                verbose)
            self.pod.compute_decomp()
        elif self.snaplist[:-1] != self.podsnaplist or len(snapPaths) !=\
            len(self.pod.snapPaths)+1:
            raise RuntimeError('Snapshot mistmatch between POD and DMD '+\
                'objects.')     
        _podSingValsSqrtMat = N.mat(N.diag(N.array(self.pod.singVals).\
            squeeze() ** -0.5))

        # Inner product of snapshots w/POD modes
        numSnaps = len(self.snapPaths)
        podModesStarTimesSnaps = N.mat(N.empty((numSnaps-1, numSnaps-1)))
        podModesStarTimesSnaps[:, :-1] = self.pod.correlationMat[:,1:]  
        podModesStarTimesSnaps[:, -1] = self.compute_inner_product_matrix(self.\
            snapPaths[:-1], self.snapPaths[-1])
        podModesStarTimesSnaps = _podSingValsSqrtMat * self.pod.\
            singVecs.H * podModesStarTimesSnaps
            
        # Reduced order linear system
        lowOrderLinearMap = podModesStarTimesSnaps * self.pod.singVecs * \
            _podSingValsSqrtMat
        self.ritzVals, lowOrderEigVecs = N.linalg.eig(lowOrderLinearMap)
        
        # Scale Ritz vectors
        ritzVecsStarTimesInitSnap = lowOrderEigVecs.H * _podSingValsSqrtMat * \
            self.pod.singVecs.H * self.pod.correlationMat[:,0]
        ritzVecScaling = N.linalg.inv(lowOrderEigVecs.H * lowOrderEigVecs) *\
            ritzVecsStarTimesInitSnap
        ritzVecScaling = N.mat(N.diag(N.array(ritzVecScaling).squeeze()))

        # Compute mode energies
        self.buildCoeff = self.pod.singVecs * _podSingValsSqrtMat *\
            lowOrderEigVecs * ritzVecScaling
        self.modeNorms = N.diag(self.buildCoeff.H * self.pod.\
            correlationMat * self.buildCoeff).real

        # Save data 
        if self.save_mat is not None and self.mpi._rank==0:
            if ritzValsPath is not None:
                self.save_mat(self.ritzVals, ritzValsPath)
            if buildCoeffPath is not None:
                self.save_mat(self.buildCoeff, buildCoeffPath)
            if modeNormsPath is not None:
                self.save_mat(self.modeNorms, modeNormsPath)

    def compute_modes(self, modeNumList, modePath, indexFrom=1, snapPaths=None):
        if self.buildCoeff is None:
            raise util.UndefinedError('Must define self.buildCoeff')
        # User should specify ALL snapshots, even though all but last are used
        if snapPaths is not None:
            self.snapPaths = snapPaths
        ModalDecomp._compute_modes(self, modeNumList, modePath, self.\
            snapPaths[:-1], self.buildCoeff, indexFrom=indexFrom)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
