
"""This script is to be used to determine how fast the code is for profiling
and scaling. There are individual functions for testing each component of
modaldecomp. """

import os
import subprocess as SP
import numpy as N
import util
import modaldecomp as MD
import time as T

mpi = util.MPI(verbose = True)

save_field = util.save_mat_text
load_field = util.load_mat_text
inner_product = util.inner_product
dataDir = './files_benchmark/'

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

def inner_product_mat(numStates, numRows, numCols, maxFieldsPerNode):
    """
    Computes inner products from known fields.
    
    Remember that rows correspond to adjoint modes and cols to direct modes
    """
    
    colFieldName = 'col_%04d.txt'
    rowFieldName = 'row_%04d.txt'
    
    rowFieldPaths = []
    for rowNum in range(numRows):
        rowFieldPaths.append(dataDir + rowFieldName%rowNum)
    
    colFieldPaths = []
    for colNum in range(numCols):
        colFieldPaths.append(dataDir + colFieldName%colNum)
    
    generate_fields(numStates, numRows, dataDir, rowFieldName)
    generate_fields(numStates, numCols, dataDir, colFieldName)
    
    myMD = MD.ModalDecomp(maxFieldsPerNode = maxFieldsPerNode,
        save_field = save_field, load_field=load_field, 
        inner_product=inner_product) 
    
    startTime = T.time()
    innerProductMat = myMD.compute_inner_product_matrix(colFieldPaths, \
      rowFieldPaths)
    totalTime = T.time() - startTime
    return totalTime


def lin_combine_fields(numStates, numBases, numProducts, maxFieldsPerNode):
    """
    Computes linear combination of fields from saved fields and random coeffs
    lin_combine_fields(self,outputFieldPaths,inputFieldPaths,fieldCoeffMat):
    """
    # numBases is number of fields to be linearly combined
    # numProducts is the resulting number of fields


    basisName = 'snap_%04d.txt'
    productName = 'product_%04d.txt'
    generate_fields(numStates, numBases, dataDir, basisName)
    myMD = MD.ModalDecomp(maxFieldsPerNode = maxFieldsPerNode,
    save_field = save_field, load_field=load_field, inner_product=inner_product)
    coeffMat = N.random.random((numBases, numProducts))
    
    basisPaths = []
    for basisNum in range(numBases):
        basisPaths.append(dataDir + basisName%basisNum)
    
    productPaths = []
    for productNum in range(numProducts):
        productPaths.append(dataDir + productName%productNum)
    
    startTime = T.time()
    myMD.lin_combine_fields(productPaths, basisPaths, coeffMat)
    totalTime = T.time() - startTime
    return totalTime
    
def clean_up():
    SP.call(['rm -rf '+dataDir+'*'], shell=True)


def main():
    numStates = 100
    numBases = 100
    numProducts = 10
    maxFieldsPerNode = 50
    t= lin_combine_fields(numStates, numBases, numProducts, maxFieldsPerNode)
    print 'time for lin_combine_fields is',t
    t= inner_product_mat(numStates, 10, 10, 50)
    print 'time for inner_product_mat is',t
    clean_up()


if __name__ == '__main__':
    main()

        
    
    
    
    

