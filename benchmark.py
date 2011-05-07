
"""This script is to be used to determine how fast the code is for profiling
and scaling. There are individual functions for testing each component of
modaldecomp. """

import os
import subprocess as SP
import util
import modaldecomp as MD

mpi = util.MPI(verbose = True)

save_field = util.save_mat_text
load_field = util.load_mat_text

def generate_fields(numStates, numFields, fieldDir, fieldName):
    """
    Creates a data set of fields, saves to file.
    
    fieldDir is the directory
    fieldName is the name and must include a %03d type string.
    """
    
    if not os.path.exists(fieldDir):
        SP.call(['mkdir', fieldDir])
    
    if mpi.isRankZero():
        for fieldNum in range(numFields):
            field = N.random.random(numStates)
            save_field(field, fieldDir + fieldName%fieldNum)

def inner_product_mat():
    numStates = 10000
    numDirectModes = 100
    numAdjointModes = 100
    
    dataDir = './files_benchmark/'
    directModeName = 'direct_mode_%04d.txt'
    adjointModeName = 'adjoint_mode_%04d.txt'
    
    directModePaths = []
    for modeNum in range(numDirectModes):
        directModePaths.append(dataDir + directModeName%modeNum)

    for modeNum in range(numAdjointModes):
        adjointModePaths.append(dataDir + adjointModeName%modeNum)
    
    generate_fields(numStates, numDirectModes, dataDir, directModeName)
    generate_fields(numStates, numDirectModes, dataDir, adjointModeName)
    
    myMD = MD.ModalDecomp()
    # Timer
    innerProductMat = myMD.compute_inner_product_matrix(directModePaths, \
      adjointModePaths)
    # Timer  
        
    
    
    
    

