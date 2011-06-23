#!/usr/bin/env python

import subprocess as SP
import multiprocessing
import os
import copy
import inspect #makes it possible to find information about a function
import unittest
import numpy as N
import fieldoperations as FO
import util

class TestFieldOperations(unittest.TestCase):
    """ Tests of the self.fieldOperations class """
    
    def setUp(self):
       
        self.maxFields = 10
        self.totalNumFieldsInMem = 1 * self.maxFields

        # FieldOperations object for running tests
        self.fieldOperations = FO.FieldOperations( 
            load_field=util.load_mat_text, 
            save_field=util.save_mat_text, 
            inner_product=util.inner_product,
            verbose=False, 
            maxFields = self.maxFields)

    def tearDown(self):
        SP.call(['rm -rf files_modaldecomp_test/*'], shell=True)       
 
    #@unittest.skip('testing other things')
    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly
        """
        dataMembersDefault = {'load_field': None, 'save_field': None, 
            'inner_product': None, 'maxFields': 2,
            'verbose': False}
        dataMembersObserved = util.get_data_members(FO.FieldOperations(verbose=\
            False))
        self.assertEqual(dataMembersObserved, dataMembersDefault)
        
        def my_load(fname): pass
        myFO = FO.FieldOperations(load_field=my_load, verbose=False)
        dataMembersModified = copy.deepcopy(dataMembersDefault)
        dataMembersModified['load_field'] = my_load
        dataMembersObserved = util.get_data_members(myFO)
        self.assertEqual(dataMembersObserved, dataMembersModified)
        
        def my_save(data,fname): pass
        myFO = FO.FieldOperations(save_field=my_save, verbose=False)
        dataMembersModified = copy.deepcopy(dataMembersDefault)
        dataMembersModified['save_field'] = my_save
        dataMembersObserved = util.get_data_members(myFO)
        self.assertEqual(dataMembersObserved, dataMembersModified)
        
        def my_ip(f1,f2): pass
        myFO = FO.FieldOperations(inner_product=my_ip, verbose=False)
        dataMembersModified = copy.deepcopy(dataMembersDefault)
        dataMembersModified['inner_product'] = my_ip
        dataMembersObserved = util.get_data_members(myFO)
        self.assertEqual(dataMembersObserved, dataMembersModified)
        
        maxFields = 500
        myFO = FO.FieldOperations(maxFields=maxFields, verbose=False)
        dataMembersModified = copy.deepcopy(dataMembersDefault)
        dataMembersModified['maxFields'] = maxFields
        dataMembersObserved = util.get_data_members(myFO)
        self.assertEqual(dataMembersObserved, dataMembersModified)
      

    #@unittest.skip('testing other things')
    def test_idiot_check(self):
        """
        Tests that the idiot check correctly checks user supplied objects/
        functions.
        """
        nx = 40
        ny = 15
        testArray = N.random.random((nx,ny))
        def inner_product(a,b):
            return N.sum(a.arr*b.arr)
        myFO = FO.FieldOperations(inner_product=util.inner_product, verbose=False)
        myFO.idiot_check(testObj=testArray)
        
        # An idiot's class that redefines multiplication to modify its data
        class IdiotMult(object):
            def __init__(self,arr):
                self.arr = arr #array
            def __add__(self,obj):
                fReturn = copy.deepcopy(self)
                fReturn.arr+=obj.arr
                return fReturn
            def __mul__(self,a):
                self.arr*=a
                return self
                
        class IdiotAdd(object):
            def __init__(self,arr):
                self.arr = arr #array
            def __add__(self,obj):
                self.arr += obj.arr
                return self
            def __mul__(self,a):
                fReturn = copy.deepcopy(self)
                fReturn.arr*=a
                return fReturn
        myFO.inner_product = inner_product
        myIdiotMult = IdiotMult(testArray)
        self.assertRaises(ValueError,myFO.idiot_check,testObj=myIdiotMult)
        myIdiotAdd = IdiotAdd(testArray)
        self.assertRaises(ValueError,myFO.idiot_check,testObj=myIdiotAdd)
                
        
    def generate_snaps_modes(self,numStates,numSnaps,numModes,indexFrom=1):
        """
        Generates random snapshots and finds the modes. 
        
        Returns:
        snapMat -  matrix in which each column is a snapshot (in order)
        modeNumList - unordered list of integers representing mode numbers,
          each entry is unique. Mode numbers are picked randomly between
          indexFrom and numModes+indexFrom-1. 
        buildCoeffMat - matrix numSnaps x numModes, random entries
        modeMat - matrix of modes, each column is a mode.
          matrix column # = modeNumber - indexFrom
        """
        modeNumList=[]
        while len(modeNumList) < numModes:
            modeNum = indexFrom+int(N.floor(N.random.random()*numModes))
            if modeNumList.count(modeNum) == 0:
                modeNumList.append(modeNum)

        buildCoeffMat = N.mat(N.random.random((numSnaps,numModes)))
        snapMat = N.mat(N.zeros((numStates,numSnaps)))
        for snapIndex in range(numSnaps):
            snapMat[:,snapIndex] = N.random.random((numStates,1))
        modeMat = snapMat*buildCoeffMat
        modeArray = N.array(modeMat)
        return snapMat,modeNumList,buildCoeffMat,modeArray 
        
    
    # def test__compute_modes_chunk(self):
        """
        Test that can compute chunks of modes from arguments.
        
        Currently this is tested almost entirely within test_compute_modes,
        therefore nothing is tested here. In the future, the error checking
        in _compute_modes_chunk, but really _compute_modes and
        _compute_modes_chunk do only one task and need only one test.
        """
    #    pass        
    
    #@unittest.skip('testing other things')
    def test_compute_modes(self):
        """
        Test that can compute modes from arguments. 
        
        This tests the _compute_modes and _compute_modes_chunk functions.
        Parallel and serial cases need to be tested independently. 
        
        Many cases are tested for numbers of snapshots, states per snapshot,
        mode numbers, number of snapshots/modes allowed in memory
        simultaneously, and what the indexing scheme is 
        (currently supports any indexing
        scheme, meaning the first mode can be numbered 0, 1, or any integer).
        """
        numSnapsList = [1, 15, 40]
        numStates = 20
        # Test cases where number of modes:
        #   less, equal, more than numStates
        #   less, equal, more than numSnaps
        #   less, equal, more than totalNumFieldsInMem
        numModesList = [1, 8, 10, 20, 25, 45, int(N.ceil(self.\
            totalNumFieldsInMem / 2.)), self.totalNumFieldsInMem, self.\
            totalNumFieldsInMem * 2]
        indexFromList = [0, 5]
        #modePath = 'proc'+str(self.fieldOperations.mpi._rank)+'/mode_%03d.txt'
        modePath = 'files_modaldecomp_test/mode_%03d.txt'
        snapPath = 'files_modaldecomp_test/snap_%03d.txt'
        if not os.path.isdir('files_modaldecomp_test'):
            SP.call(['mkdir','files_modaldecomp_test'])
        
        for numSnaps in numSnapsList:
            for numModes in numModesList:
                for indexFrom in indexFromList:
                    for sharedMemLoad in [True, False]:
                        for sharedMemSave in [True, False]:
                            #generate data and then broadcast to all procs
                            #print '----- new case ----- '
                            #print 'numSnaps =',numSnaps
                            #print 'numStates =',numStates
                            #print 'numModes =',numModes
                            #print 'maxFields =',maxFields                          
                            #print 'indexFrom =',indexFrom
                            snapPaths = [snapPath % snapIndex for snapIndex in \
                                range(numSnaps)]
                            
                            snapMat,modeNumList, buildCoeffMat, trueModes = \
                                self.generate_snaps_modes(numStates, numSnaps,
                                numModes, indexFrom=indexFrom)
                            for snapIndex,s in enumerate(snapPaths):
                                util.save_mat_text(snapMat[:,snapIndex], s)
                                
                            # if any mode number (minus starting indxex)
                            # is greater than the number of coeff mat columns,
                            # or is less than zero
                            checkAssertRaises = False
                            for modeNum in modeNumList:
                                modeNumFromZero = modeNum-indexFrom
                                if modeNumFromZero < 0 or modeNumFromZero >=\
                                    buildCoeffMat.shape[1]:
                                    checkAssertRaises = True
                            if checkAssertRaises:
                                self.assertRaises(ValueError, self.\
                                    fieldOperations._compute_modes, 
                                    modeNumList, modePath, snapPaths, 
                                    buildCoeffMat, indexFrom=indexFrom)
                                    
                            # If the coeff mat has more rows than there are 
                            # snapshot paths
                            elif numSnaps > buildCoeffMat.shape[0]:
                                self.assertRaises(ValueError, self.\
                                    fieldOperations._compute_modes, modeNumList,
                                    modePath, snapPaths, buildCoeffMat, 
                                    indexFrom=indexFrom)
                            elif numModes > numSnaps:
                                self.assertRaises(ValueError,
                                    self.fieldOperations._compute_modes, 
                                    modeNumList, modePath, snapPaths, 
                                    buildCoeffMat, indexFrom=indexFrom)
                                  
                            else:
                                # Test the case that only one mode is desired,
                                # in which case user might pass in an int
                                if len(modeNumList) == 1:
                                    modeNumList = modeNumList[0]

                                # Saves modes to files
                                self.fieldOperations._compute_modes(modeNumList, 
                                    modePath, snapPaths, buildCoeffMat, 
                                    indexFrom=indexFrom)

                                # Change back to list so is iterable
                                if isinstance(modeNumList, int):
                                    modeNumList = [modeNumList]
                                
                                # all computed modes are saved
                                for modeNum in modeNumList:
                                    computedMode = util.load_mat_text(
                                        modePath % modeNum)
                                    #print 'mode number',modeNum
                                    #print 'true mode',trueModes[:,
                                        #modeNum-indexFrom]
                                    #print 'computed mode',computedMode
                                    
                                    N.testing.assert_array_almost_equal(
                                        computedMode.squeeze(), trueModes[:,
                                        modeNum-indexFrom].squeeze())
                                
                                
    #@unittest.skip('testing other things')
    def test_compute_inner_product_mats(self):
        """
        Test computation of matrix of inner products in memory-efficient
        chunks, both in parallel (compute_inner_product_matrix) and serial
        (compute_inner_product_chunk)
        """ 
        
        def assert_equal_mat_products(mat1, mat2, paths1, paths2):
            # Path list may actually be a string, in which case covert to list
            if isinstance(paths1, str):
                paths1 = [paths1]
            if isinstance(paths2, str):
                paths2 = [paths2]

            # True inner product matrix
            productTrue = mat1 * mat2
           
            # Test computation as chunk (a serial method, tested on each proc)
            #productComputedAsChunk = self.fieldOperations.\
            #    _compute_inner_product_chunk(paths1, paths2)
            #N.testing.assert_array_almost_equal(productComputedAsChunk, 
            #    productTrue)

            # Test parallelized computation.  
            for sharedMemLoad in [True, False]:
                for sharedMemInnerProduct in [True, False]:
                    productComputedAsMat = self.fieldOperations.\
                        compute_inner_product_mat(paths1, paths2, sharedMemLoad=\
                        sharedMemLoad, sharedMemInnerProduct=\
                        sharedMemInnerProduct)
                    N.testing.assert_array_almost_equal(productComputedAsMat, 
                        productTrue)
                
                # Test computation of upper triangular inner product mat chunk
                if paths1 == paths2:
                    # Test computation as chunk (serial).  
                    # First test complete upper triangular computation
                    productComputedAsFullSymmChunk = self.fieldOperations.\
                        _compute_upper_triangular_inner_product_chunk(paths1, 
                            paths2, sharedMemLoad=sharedMemLoad, 
                            sharedMemInnerProduct=sharedMemInnerProduct)
                    N.testing.assert_array_almost_equal(
                        productComputedAsFullSymmChunk, N.triu(productTrue))

                    # Also test non-square upper triangular computation
                    numRows = int(N.ceil(len(paths1) / 2.))
                    productComputedAsPartialSymmChunk = self.fieldOperations.\
                        _compute_upper_triangular_inner_product_chunk(paths1[
                            :numRows], paths2, sharedMemLoad=sharedMemLoad, 
                            sharedMemInnerProduct=sharedMemInnerProduct)
                    N.testing.assert_array_almost_equal(
                        productComputedAsPartialSymmChunk, N.triu(productTrue[
                        :numRows, :]))
                    
                    # Test computation in parallel
                    productComputedAsSymmMat = self.fieldOperations.\
                        compute_symmetric_inner_product_mat(paths1, 
                            sharedMemLoad=sharedMemLoad, sharedMemInnerProduct=\
                            sharedMemInnerProduct)
                    N.testing.assert_array_almost_equal(productComputedAsSymmMat, 
                        productTrue)
                # If lists are not the same, should return an error
                else:
                    self.assertRaises(ValueError, self.fieldOperations.\
                        _compute_upper_triangular_inner_product_chunk, paths1, 
                        paths2)
            
        numRowSnapsList =[1, int(round(self.totalNumFieldsInMem / 2.)), self.\
            totalNumFieldsInMem, self.totalNumFieldsInMem *2]
        numColSnapsList = numRowSnapsList
        numStates = 6

        if not os.path.isdir('files_modaldecomp_test'):
            SP.call(['mkdir', 'files_modaldecomp_test'])
        rowSnapPath = 'files_modaldecomp_test/row_snap_%03d.txt'
        colSnapPath = 'files_modaldecomp_test/col_snap_%03d.txt'

        
        for numRowSnaps in numRowSnapsList:
            for numColSnaps in numColSnapsList:
                # print '---- Case with numRowSnaps =',numRowSnaps,'and numColSnaps =',numColSnaps
                # generate snapshots and save to file
                rowSnapMat = N.mat(N.random.random((numStates,
                    numRowSnaps)))
                colSnapMat = N.mat(N.random.random((numStates,
                    numColSnaps)))
                rowSnapPaths = []
                colSnapPaths = []
                for snapIndex in xrange(numRowSnaps):
                    path = rowSnapPath % snapIndex
                    util.save_mat_text(rowSnapMat[:,snapIndex],path)
                    rowSnapPaths.append(path)
                for snapIndex in xrange(numColSnaps):
                    path = colSnapPath % snapIndex
                    util.save_mat_text(colSnapMat[:,snapIndex],path)
                    colSnapPaths.append(path)

                # If number of rows/cols is 1, test case that a string, not
                # a list, is passed in
                if len(rowSnapPaths) == 1:
                    rowSnapPaths = rowSnapPaths[0]
                if len(colSnapPaths) == 1:
                    colSnapPaths = colSnapPaths[0]

                # Test different rows and cols snapshots
                assert_equal_mat_products(rowSnapMat.T, colSnapMat,
                    rowSnapPaths, colSnapPaths)
                
                # Test with only the row data, to ensure nothing is
                # goes wrong when the same list is used twice
                # (potential memory issues, or lists may accidentally
                # get altered).  Also, test symmetric computation
                # method.
                assert_equal_mat_products(rowSnapMat.T, rowSnapMat,
                    rowSnapPaths, rowSnapPaths)
        
if __name__=='__main__':
    unittest.main(verbosity=2)    

    
    

