#!/usr/bin/env python
import unittest
import numpy as N
from modaldecomp import ModalDecomp
import util
import subprocess as SP
import os
import copy
import inspect #makes it possible to find information about a function

try:
    from mpi4py import MPI
    parallel = MPI.COMM_WORLD.Get_size() >=2
    rank = MPI.COMM_WORLD.Get_rank()
    numProcs = MPI.COMM_WORLD.Get_size()
except ImportError:
    parallel = False
    numProcs = 1
    rank = 0

print 'To test fully, remember to do both:'
print '    1) python testmodaldecomp.py'
print '    2) mpiexec -n <# procs> python testmodaldecomp.py'

class TestModalDecomp(unittest.TestCase):
    """ Tests of the self.modalDecomp class """
    
    def setUp(self):
        # Default data members (verbose set to false even though default is true
        # so messages won't print during tests
        self.defaultMPI = util.MPI() 
        self.defaultDataMembers = {'load_field': None, 'save_field': None, 
            'save_mat': None, 'inner_product': None, 'maxFieldsPerNode': 2,
            'maxFieldsPerProc': 2, 'mpi': self.defaultMPI, 'numNodes': 1, 
            'verbose': False}
       
        # Parallel stuff
        self.maxFieldsPerProc = 10
        self.totalNumFieldsInMem = numProcs * self.maxFieldsPerProc

        # ModalDecomp object for running tests
        self.modalDecomp = ModalDecomp(load_field=util.load_mat_text, 
            save_field=util.save_mat_text, inner_product=util.inner_product, 
            verbose=False)
        self.modalDecomp.maxFieldsPerProc = self.maxFieldsPerProc

    def tearDown(self):
        self.modalDecomp.mpi.sync()
        if self.modalDecomp.mpi.isRankZero():
            SP.call(['rm -rf files_modaldecomp_test/*'],shell=True)
        self.modalDecomp.mpi.sync()
 
    #@unittest.skip('testing other things')
    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly
        """
        
        dataMembersOriginal = util.get_data_members(ModalDecomp(verbose=False))
        self.assertEqual(dataMembersOriginal, self.defaultDataMembers)
        
        def my_load(fname): pass
        myMD = ModalDecomp(load_field=my_load, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['load_field'] = my_load
        self.assertEqual(util.get_data_members(myMD), dataMembers)
        
        def my_save(data,fname): pass
        myMD = ModalDecomp(save_field=my_save, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['save_field'] = my_save
        self.assertEqual(util.get_data_members(myMD), dataMembers)
        
        myMD = ModalDecomp(save_mat=my_save, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['save_mat'] = my_save
        self.assertEqual(util.get_data_members(myMD), dataMembers)
        
        def my_ip(f1,f2): pass
        myMD = ModalDecomp(inner_product=my_ip, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['inner_product'] = my_ip
        self.assertEqual(util.get_data_members(myMD), dataMembers)
        
        maxFieldsPerNode = 500
        myMD = ModalDecomp(maxFieldsPerNode=maxFieldsPerNode, verbose=False)
        dataMembers = copy.deepcopy(dataMembersOriginal)
        dataMembers['maxFieldsPerNode'] = maxFieldsPerNode
        dataMembers['maxFieldsPerProc'] = maxFieldsPerNode * myMD.numNodes /\
            myMD.mpi.getNumProcs()
        self.assertEqual(util.get_data_members(myMD), dataMembers)
        
        self.assertRaises(util.MPIError, ModalDecomp, numNodes=numProcs + 1, 
            verbose=False)
        #MPI class is tested by utils.   
        

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
        myMD = ModalDecomp(inner_product=util.inner_product, verbose=False)
        myMD.idiot_check(testObj=testArray)
        
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
        myMD.inner_product = inner_product
        myIdiotMult = IdiotMult(testArray)
        self.assertRaises(ValueError,myMD.idiot_check,testObj=myIdiotMult)
        myIdiotAdd = IdiotAdd(testArray)
        self.assertRaises(ValueError,myMD.idiot_check,testObj=myIdiotAdd)
                
        
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
        return snapMat,modeNumList,buildCoeffMat,modeMat 
        
    
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
        #modePath = 'proc'+str(self.modalDecomp.mpi._rank)+'/mode_%03d.txt'
        modePath = 'files_modaldecomp_test/mode_%03d.txt'
        snapPath = 'files_modaldecomp_test/snap_%03d.txt'
        if self.modalDecomp.mpi.isRankZero():
            if not os.path.isdir('files_modaldecomp_test'):
                SP.call(['mkdir','files_modaldecomp_test'])
        
        for numSnaps in numSnapsList:
            for numModes in numModesList:
                for indexFrom in indexFromList:
                    #generate data and then broadcast to all procs
                    #print '----- new case ----- '
                    #print 'numSnaps =',numSnaps
                    #print 'numStates =',numStates
                    #print 'numModes =',numModes
                    #print 'maxFieldsPerNode =',maxFieldsPerNode                          
                    #print 'indexFrom =',indexFrom
                    snapPaths = []
                    for snapIndex in range(numSnaps):
                        snapPaths.append(snapPath % snapIndex)
                    
                    if self.modalDecomp.mpi.isRankZero():
                        snapMat,modeNumList, buildCoeffMat, trueModes = \
                          self.generate_snaps_modes(numStates, numSnaps,
                          numModes, indexFrom=indexFrom)
                        for snapIndex,s in enumerate(snapPaths):
                            util.save_mat_text(snapMat[:,snapIndex], s)
                    else:
                        modeNumList = None
                        buildCoeffMat = None
                        snapMat = None
                        trueModes = None
                    if self.modalDecomp.mpi.isParallel():
                        modeNumList = self.modalDecomp.mpi.comm.bcast(
                            modeNumList, root=0)
                        buildCoeffMat = self.modalDecomp.mpi.comm.bcast(
                            buildCoeffMat,root=0)
                        snapMat = self.modalDecomp.mpi.comm.bcast(
                            snapMat, root=0)
                        trueModes = self.modalDecomp.mpi.comm.bcast(
                            trueModes, root=0)
                        
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
                        self.assertRaises(ValueError, self.modalDecomp.\
                            _compute_modes, modeNumList, modePath, 
                            snapPaths, buildCoeffMat, indexFrom=\
                            indexFrom)
                    # If the coeff mat has more rows than there are 
                    # snapshot paths
                    elif numSnaps > buildCoeffMat.shape[0]:
                        self.assertRaises(ValueError, self.modalDecomp.\
                            _compute_modes, modeNumList, modePath,
                            snapPaths, buildCoeffMat, indexFrom=\
                            indexFrom)
                    elif numModes > numSnaps:
                        self.assertRaises(ValueError,
                          self.modalDecomp._compute_modes, modeNumList,
                          modePath, snapPaths, buildCoeffMat,
                          indexFrom=indexFrom)
                    # If more processors than number of snaps available,
                    # then some procs will not have a task, not allowed.
                    elif self.modalDecomp.mpi.getNumProcs() > numSnaps:
                        self.assertRaises(util.MPIError, self.\
                            modalDecomp._compute_modes, modeNumList, 
                            modePath, snapPaths, buildCoeffMat, 
                            indexFrom=indexFrom)
                    else:
                        # Test the case that only one mode is desired,
                        # in which case user might pass in an int
                        if len(modeNumList) == 1:
                            modeNumList = modeNumList[0]

                        # Saves modes to files
                        self.modalDecomp._compute_modes(modeNumList, 
                            modePath, snapPaths, buildCoeffMat, 
                            indexFrom=indexFrom)

                        # Change back to list so is iterable
                        if isinstance(modeNumList, int):
                            modeNumList = [modeNumList]

                        self.modalDecomp.mpi.sync()
                        
                        # Do tests on processor 0
                        if self.modalDecomp.mpi.isRankZero():
                            for modeNum in modeNumList:
                                computedMode = util.load_mat_text(
                                    modePath % modeNum)
                                #print 'mode number',modeNum
                                #print 'true mode',trueModes[:,
                                    #modeNum-indexFrom]
                                #print 'computed mode',computedMode
                                N.testing.assert_array_almost_equal(
                                    computedMode, trueModes[:,modeNum-\
                                    indexFrom])
                                
                        self.modalDecomp.mpi.sync()
       
        self.modalDecomp.mpi.sync()

    #@unittest.skip('testing other things')
    def test_compute_inner_product_matrices(self):
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
            productComputedAsChunk = self.modalDecomp.\
                _compute_inner_product_chunk(paths1, paths2)
            N.testing.assert_array_almost_equal(productComputedAsChunk, 
                productTrue)

            # Test paralleized computation.  
            productComputedAsMat = self.modalDecomp.\
                compute_inner_product_matrix(paths1, paths2)
            N.testing.assert_array_almost_equal(productComputedAsMat, 
                productTrue)
           
            # Test computation of upper triangular inner product matrix chunk
            if paths1 == paths2:
                # Test computation as chunk (serial).  
                # First test complete upper triangular computation
                productComputedAsFullSymmChunk = self.modalDecomp.\
                    _compute_upper_triangular_inner_product_chunk(paths1, 
                        paths2)
                N.testing.assert_array_almost_equal(
                    productComputedAsFullSymmChunk, N.triu(productTrue))

                # Also test non-square upper triangular computation
                numRows = int(N.ceil(len(paths1) / 2.))
                productComputedAsPartialSymmChunk = self.modalDecomp.\
                    _compute_upper_triangular_inner_product_chunk(paths1[
                        :numRows], paths2)
                N.testing.assert_array_almost_equal(
                    productComputedAsPartialSymmChunk, N.triu(productTrue[
                    :numRows, :]))
                
                # Test computation in parallel
                productComputedAsSymmMat = self.modalDecomp.\
                    compute_symmetric_inner_product_matrix(paths1)
                N.testing.assert_array_almost_equal(productComputedAsSymmMat, 
                    productTrue)
            # If lists are not the same, should return an error
            else:
                self.assertRaises(ValueError, self.modalDecomp.\
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
                # generate snapshots and save to file, only do on proc 0
                self.modalDecomp.mpi.sync()
                if self.modalDecomp.mpi.isRankZero():
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
                else:
                    rowSnapMat = None
                    colSnapMat = None
                    rowSnapPaths = None
                    colSnapPaths = None
                if self.modalDecomp.mpi.isParallel():
                    rowSnapMat = self.modalDecomp.mpi.comm.bcast(
                        rowSnapMat, root=0)
                    colSnapMat = self.modalDecomp.mpi.comm.bcast(
                        colSnapMat, root=0)
                    rowSnapPaths = self.modalDecomp.mpi.comm.bcast(
                        rowSnapPaths, root=0)
                    colSnapPaths = self.modalDecomp.mpi.comm.bcast(
                        colSnapPaths, root=0)

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

    
    

