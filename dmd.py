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

    def __init__(self, load_field=None, save_field=None, load_mat=util.\
        load_mat_text, save_mat=util.save_mat_text, inner_product=None, 
        maxFieldsPerNode=None, numNodes=1, pod=None, verbose=True):
        """
        DMD constructor
        """
        self.fieldOperations = FieldOperations(load_field=load_field,\
            save_field=save_field, inner_product=inner_product,
            maxFieldsPerNode=maxFieldsPerNode, numNodes=numNodes, verbose=\
            verbose)
        self.mpi = util.MPIInstance

        self.load_mat = load_mat
        self.save_mat = save_mat
        self.pod = pod
        self.verbose = verbose

    def load_decomp(self, ritzValsPath, modeNormsPath, buildCoeffPath):
        """
        Loads the decomposition matrices from file. 
        """
        if self.load_mat is None:
            raise UndefinedError('Must specify a load_mat function')
        if self.mpi.isRankZero():
            self.ritzVals = N.squeeze(N.array(self.load_mat(ritzValsPath)))
            self.modeNorms = N.squeeze(N.array(self.load_mat(modeNormsPath)))
            self.buildCoeff = self.load_mat(buildCoeffPath)
        else:
            self.ritzVals = None
            self.modeNorms = None
            self.buildCoeff = None
        if self.mpi.parallel:
            self.ritzVals = self.mpi.comm.bcast(self.ritzVals, root=0)
            self.modeNorms = self.mpi.comm.bcast(self.modeNorms, root=0)
            self.buildCoeff = self.mpi.comm.bcast(self.buildCoeff, root=0)
            
    def save_decomp(self, ritzValsPath, modeNormsPath, buildCoeffPath):
        """Save the decomposition matrices to file."""
        if self.save_mat is None and self.mpi.isRankZero():
            raise util.UndefinedError("save_mat is undefined, can't save")
            
        if self.mpi.isRankZero():
            self.save_mat(self.ritzVals, ritzValsPath)
            self.save_mat(self.modeNorms, modeNormsPath)
            self.save_mat(self.buildCoeff, buildCoeffPath)

    def compute_decomp(self, snapPaths):
        """
        Compute DMD decomposition
        """
         
        if snapPaths is not None:
            self.snapPaths = snapPaths
        if self.snapPaths is None:
            raise util.UndefinedError('snapPaths is not given')

        # Compute POD from snapshots (excluding last snapshot)
        if self.pod is None:
            self.pod = POD(load_field=self.fieldOperations.load_field, 
                inner_product=self.fieldOperations.inner_product, 
                maxFieldsPerNode=self.fieldOperations.maxFieldsPerNode, 
                numNodes=self.fieldOperations.numNodes, verbose=self.verbose)
            self.pod.compute_decomp(snapPaths=self.snapPaths[:-1])
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
        podModesStarTimesSnaps[:, -1] = self.fieldOperations.\
            compute_inner_product_mat(self.snapPaths[:-1], self.snapPaths[
            -1])
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
        
    def compute_modes(self, modeNumList, modePath, indexFrom=1, snapPaths=None):
        if self.buildCoeff is None:
            raise util.UndefinedError('Must define self.buildCoeff')
        # User should specify ALL snapshots, even though all but last are used
        if snapPaths is not None:
            self.snapPaths = snapPaths
        self.fieldOperations._compute_modes(modeNumList, modePath, self.\
            snapPaths[:-1], self.buildCoeff, indexFrom=indexFrom)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
