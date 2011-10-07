"""
This script is to be used to determine how fast the code is for profiling
and scaling. There are individual functions for testing each component of
modaldecomp. 

To profile, do the following:
    python -m cProfile -o <pathToOutputFile> <pathToPythonScript>
    
In parallel, simply prepend with:
    mpiexec -n <number of procs> ...

Then in python, do this to view the results:
    import pstats
    pstats.Stats(<pathToOutputFile>).strip_dirs().sort_stats('cumulative').\
        print_stats(<numberOfSignificantLinesToPrint>)
"""

import os
import subprocess as SP
import numpy as N
import util
import time as T
import fieldoperations as FO
import cPickle
import parallel as parallel_mod
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



import argparse
parser = argparse.ArgumentParser(description='Get directory in which to ' +\
    'save data.')
parser.add_argument('--outdir', default='./files_benchmark', help='Path in ' +\
    'which to save data for benchmark.')
parser.add_argument('--benchmarkfunc', required=True, choices=['lin_combine',
    'inner_product_mat', 'symmetric_inner_product_mat'], help='Function to ' +\
    'benchmark.')
args = parser.parse_args()
dataDir = args.outdir
if dataDir[-1] != '/':
    dataDir += '/'

def generate_fields(numStates, numFields, fieldDir, fieldName):
    """
    Creates a data set of fields, saves to file.
    
    fieldDir is the directory
    fieldName is the name and must include a %03d type string.
    """
    if not os.path.exists(fieldDir):
        SP.call(['mkdir', fieldDir])
    
    """
    # Parallelize saving of fields (may slow down sequoia)
    procFieldNumAssignments = \
        parallel.find_assignments(range(numFields))[parallel.getRank()]
    for fieldNum in procFieldNumAssignments:
        field = N.random.random(numStates)
        save_field(field, fieldDir + fieldName%fieldNum)
    """
    
    if parallel.isRankZero():
        for fieldNum in xrange(numFields):
            field = N.random.random(numStates)
            save_field(field, fieldDir + fieldName % fieldNum)
    
    parallel.sync()


def inner_product_mat(numStates, numRows, numCols, maxFieldsPerNode):
    """
    Computes inner products from known fields.
    
    Remember that rows correspond to adjoint modes and cols to direct modes
    """    
    colFieldName = 'col_%04d.txt'
    colFieldPaths = [dataDir + colFieldName%colNum for colNum in range(numCols)]
    generate_fields(numStates, numCols, dataDir, colFieldName)
    
    rowFieldName = 'row_%04d.txt'    
    rowFieldPaths = [dataDir + rowFieldName%rowNum for rowNum in range(numRows)]
    generate_fields(numStates, numRows, dataDir, rowFieldName)
    
    myFO = FO.FieldOperations(maxFieldsPerNode=maxFieldsPerNode, save_field=\
        save_field, load_field=load_field, inner_product=inner_product, 
        verbose=True) 
    
    startTime = T.time()
    innerProductMat = myFO.compute_inner_product_mat(colFieldPaths, 
        rowFieldPaths)
    totalTime = T.time() - startTime
    return totalTime
    
    
def symmetric_inner_product_mat(numStates, numFields, maxFieldsPerNode):
    """
    Computes symmetric inner product matrix from known fields (as in POD).
    """    
    fieldName = 'field_%04d.txt'
    fieldPaths = [dataDir + fieldName % fieldNum for fieldNum in range(
        numFields)]
    generate_fields(numStates, numFields, dataDir, fieldName)
    
    myFO = FO.FieldOperations(maxFieldsPerNode=maxFieldsPerNode, save_field=\
        save_field, load_field=load_field, inner_product=inner_product, 
        verbose=True) 
    
    startTime = T.time()
    innerProductMat = myFO.compute_symmetric_inner_product_mat(fieldPaths)
    totalTime = T.time() - startTime
    return totalTime


def lin_combine(numStates, numBases, numProducts, maxFieldsPerNode):
    """
    Computes linear combination of fields from saved fields and random coeffs
    
    numBases is number of fields to be linearly combined
    numProducts is the resulting number of fields
    """
    basisName = 'snap_%04d.txt'
    productName = 'product_%04d.txt'
    generate_fields(numStates, numBases, dataDir, basisName)
    myFO = FO.FieldOperations(maxFieldsPerNode = maxFieldsPerNode,
    save_field = save_field, load_field=load_field, inner_product=inner_product)
    coeffMat = N.random.random((numBases, numProducts))
    
    basisPaths = [dataDir + basisName%basisNum for basisNum in range(numBases)]
    productPaths = [dataDir + productName%productNum for productNum in range \
        (numProducts)]
    
    startTime = T.time()
    myFO.lin_combine(productPaths, basisPaths, coeffMat)
    totalTime = T.time() - startTime
    return totalTime
    
    
def clean_up():
    SP.call(['rm -rf ' + dataDir + '*'], shell=True)


def main():
    #methodToTest = 'lin_combine'
    #methodToTest = 'inner_product_mat'
    #methodToTest = 'symmetric_inner_product_mat'
    methodToTest = args.benchmarkfunc
    
    # Common parameters
    maxFieldsPerNode = 50
    numStates = 8000
    
    # Run test of choice
    if methodToTest == 'lin_combine':
        # lin_combine test
        numBases = 2500
        numProducts = 1000
        t = lin_combine(numStates, numBases, numProducts, maxFieldsPerNode)
    elif methodToTest == 'inner_product_mat':
        # inner_product_mat test
        numRows = 2000
        numCols = 2000
        t= inner_product_mat(numStates, numRows, numCols, maxFieldsPerNode)
    elif methodToTest == 'symmetric_inner_product_mat':
        # symmetric_inner_product_mat test
        numFields = 2000
        t= symmetric_inner_product_mat(numStates, numFields, maxFieldsPerNode)
    print 'Time for ' + methodToTest + ' is %f' % t
    
    # Delete files
    clean_up()
    

if __name__ == '__main__':
    main()

    
    

