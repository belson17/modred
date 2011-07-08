
"""This script is to be used to determine how fast the code is for profiling
and scaling. There are individual functions for testing each component of
modaldecomp. """

import os
import subprocess as SP
import numpy as N
import util
import time as T
import parallel as parallel_mod
import fieldoperations as FO
import cPickle

parallel = parallel_mod.parallelInstance
def save_pickle(obj, filename):
    fid = open(filename,'wb')
    cPickle.dump(obj,fid)
    fid.close()
    
def load_pickle(filename):
    fid = open(filename,'rb')
    obj = cPickle.load(fid)
    fid.close()
    return obj

save_field = save_pickle 
load_field = load_pickle
#save_field = util.save_mat_text
#load_field = util.load_mat_text
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
    
    if parallel.isRankZero():
        for fieldNum in range(numFields):
            field = N.random.random(numStates)
            save_field(field, fieldDir + fieldName%fieldNum)
    parallel.sync()

def inner_product_mat(numStates, numRows, numCols, maxFieldsPerNode):
    """
    Computes inner products from known fields.
    
    Remember that rows correspond to adjoint modes and cols to direct modes
    """    
    colFieldName = 'col_%04d.txt'
    rowFieldName = 'row_%04d.txt'
    
    rowFieldPaths = [dataDir + rowFieldName%rowNum for rowNum in range(numRows)]

    colFieldPaths = [dataDir + colFieldName%colNum for colNum in range(numCols)]
    
    generate_fields(numStates, numRows, dataDir, rowFieldName)
    generate_fields(numStates, numCols, dataDir, colFieldName)
    
    myFO = FO.FieldOperations(maxFieldsPerNode = maxFieldsPerNode,
        save_field = save_field, load_field=load_field, 
        inner_product=inner_product, verbose=True) 
    
    startTime = T.time()
    innerProductMat = myFO.compute_inner_product_mat(colFieldPaths, \
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
    myFO = FO.FieldOperations(maxFieldsPerNode = maxFieldsPerNode,
    save_field = save_field, load_field=load_field, inner_product=inner_product)
    coeffMat = N.random.random((numBases, numProducts))
    
    basisPaths = [dataDir + basisName%basisNum for basisNum in range(numBases)]
    productPaths = [dataDir + productName%productNum for productNum in range(numProducts)]
    
    startTime = T.time()
    myFO.lin_combine(productPaths, basisPaths, coeffMat)
    totalTime = T.time() - startTime
    return totalTime
    
def clean_up():
    SP.call(['rm -rf '+dataDir+'*'], shell=True)


def main():
    numStates = 2000
    numBases = 100
    numProducts = 10
    maxFieldsPerNode = 50
    #t= lin_combine_fields(numStates, numBases, numProducts, maxFieldsPerNode)
    #print 'time for lin_combine_fields is',t
    numRows = 2000
    numCols = 500
    t= inner_product_mat(numStates, numRows, numCols, maxFieldsPerNode)
    print 'time for inner_product_mat is',t
    clean_up()


if __name__ == '__main__':
    main()

    
    

