#!/usr/bin/env python

import unittest
import numpy as N
import modaldecomp as MD
import util
import subprocess as SP
import os

#import inspect #makes it possible to find information about a function
try:
    from mpi4py import MPI
    parallel = MPI.COMM_WORLD.Get_size() >=2
    rank = MPI.COMM_WORLD.Get_rank()
    numProcs = MPI.COMM_WORLD.Get_size()
except ImportError:
    parallel = False
    numProcs = 1

if parallel:
    if MPI.COMM_WORLD.Get_rank()==0:
        print 'Remember to test in serial also with command:'
        print 'python testmodaldecomp.py'
else:
    print 'Remember to test in parallel also with command:'
    print 'mpiexec -n <numProcs> python testmodaldecomp.py' 


class TestModalDecomp(unittest.TestCase):
    """ Tests of the self.modalDecomp class """
    
    def setUp(self):
        self.modalDecomp = MD.ModalDecomp()
    
    def tearDown(self):
        self.modalDecomp.mpi.sync()
        if self.modalDecomp.mpi._rank == 0:
            SP.call(['rm -rf testfiles/*'],shell=True)
        self.modalDecomp.mpi.sync()

    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly"""
          
        def my_load(fname): 
            return 0
        myMD = MD.ModalDecomp(load_snap=my_load)
        self.assertEqual(myMD.load_snap,my_load)
        
        def my_save(data,fname):
            pass 
        myMD = MD.ModalDecomp(save_mode=my_save)
        self.assertEqual(myMD.save_mode,my_save)
        
        myMD = MD.ModalDecomp(save_mat=my_save)
        self.assertEqual(myMD.save_mat,my_save)
        
        def my_ip(f1,f2): return 0
        myMD = MD.ModalDecomp(inner_product=my_ip)
        self.assertEqual(myMD.inner_product,my_ip)
        
        maxSnaps = 500
        myMD = MD.ModalDecomp(maxSnapsInMem=maxSnaps)
        self.assertEqual(myMD.maxSnapsInMem,maxSnaps)
        self.assertRaises(util.MPIError,MD.ModalDecomp,
          numProcs=numProcs+1)
        #MPI class is tested by utils.   
        
        
    def generate_snaps_modes(self,numStates,numSnaps,numModes,indexFrom=1):
        """
        Generates random snapshots and finds the modes. 
        
        Returns:
        snapMat -  matrix in which each column is a snapshot in order
        modeNumList - unordered list of integers representing mode numbers
          each entry is unique. Mode numbers are picked randomly between
          indexFrom and numModes+indexFrom-1. The algorithm to compute modes
          actually allows for repeated mode numbers in serial. In parallel,
          this is problematic because different processors can try to write
          to the same file.
        buildCoeffMat - matrix numSnaps x numModes, random
        modeMat - matrix of modes, each column is a mode.
          matrix column # = modeNumber - indexFrom
        """
        modeNumList=[]
        modeIndex=0
        while modeIndex < numModes:
            modeNum = indexFrom+int(N.floor(N.random.random()*numModes))
            if modeNumList.count(modeNum) == 0:
                modeNumList.append(modeNum)
                modeIndex+=1
        buildCoeffMat = N.mat(N.random.random((numSnaps,numModes)))
        snapMat = N.mat(N.zeros((numStates,numSnaps)))
        for snapNum in range(numSnaps):
            snapMat[:,snapNum] = N.random.random((numStates,1))
        modeMat = snapMat*buildCoeffMat
        return snapMat,modeNumList,buildCoeffMat,modeMat 
        
    def helper_compute_modes(self,compute_modes_func):
        """
        A helper function used by test__compute_modes* tests.
        
        This function takes an argument
        compute_modes_func that should be a member
        function of self.modalDecomp. It is always called with the args -
        [modeNumList,modePath,snapPaths,buildCoeffMat,indexFrom=indexFrom].
        If this is not what compute_modes_func takes as arguments, then this 
        helper function will fail. This could be changed in the future.
        This function is used by test__compute_modes* currently. 
        It exists to prevent duplicate code related to calculating modes 
        for many different
        Many cases are tested for numbers of snapshots, states per snapshot,
        mode numbers, number of snapshots/modes allowed in memory
        simultaneously,
        and what the indexing scheme is (currently supports any indexing
        scheme,
        meaning the first mode can be numbered 0, 1, or any integer).
        
        
        cases. It only works because each compute_modes_func currently being
        tested take exactly the same arguments!
        In the future there might need to be a more flexible solution.
        
        """
        
        def load_snap(snapNum): #returns a precomputed, random, vector
            #argument snapNum is actually an integer.
            return snapMat[:,snapNum]
        numSnapsList = [1,3,15,40]
        numStatesList = [1,10,21]
        numModesList = [1,2,15,50]
        maxSnapsInMemList = [8,20,10000]
        indexFromList = [0,1,5]
        #modePath = 'proc'+str(self.modalDecomp.mpi._rank)+'/mode_%03d.txt'
        modePath = 'testfiles/mode_%03d.txt'
        if self.modalDecomp.mpi._rank == 0:
            if not os.path.isdir('testfiles'):
                SP.call(['mkdir','testfiles'])
        
        self.modalDecomp.load_snap=load_snap
        self.modalDecomp.save_mode=util.save_mat_text
        self.modalDecomp.inner_product=util.inner_product
        for numSnaps in numSnapsList:
            for numStates in numStatesList:
                for numModes in numModesList:
                    for maxSnapsInMem in maxSnapsInMemList: 
                        self.modalDecomp.maxSnapsInMem=maxSnapsInMem
                        for indexFrom in indexFromList:
                            #generate data and then broadcast to all procs
                            if self.modalDecomp.mpi._rank == 0:
                                snapMat,modeNumList,buildCoeffMat,trueModes = \
                                  self.generate_snaps_modes(numStates,numSnaps,
                                  numModes,indexFrom=indexFrom)
                            else:
                                modeNumList = None
                                buildCoeffMat = None
                                snapMat = None
                                trueModes = None
                            if self.modalDecomp.mpi.parallel:
                                modeNumList = self.modalDecomp.mpi.comm.bcast(
                                    modeNumList, root=0)
                                buildCoeffMat = self.modalDecomp.mpi.comm.bcast(
                                    buildCoeffMat,root=0)
                                snapMat = self.modalDecomp.mpi.comm.bcast(
                                    snapMat, root=0)
                                trueModes = self.modalDecomp.mpi.comm.bcast(
                                    trueModes, root=0)
                            snapPaths = range(numSnaps)
                            # if a mode number (minus starting indxex)
                            # is greater than the number of coeff mat columns,
                            # or is less than zero
                            checkError = False
                            for modeNum in modeNumList:
                                modeNumFromZero = modeNum-indexFrom
                                if modeNumFromZero < 0 or modeNumFromZero >\
                                    buildCoeffMat.shape[1]-1:
                                    checkError=True
                            # or if coeff mat has more rows than there are 
                            # snapshot paths
                            if checkError or numSnaps > buildCoeffMat.shape[0]:
                                self.assertRaises(ValueError, 
                                    compute_modes_func, modeNumList, modePath,
                                    snapPaths, buildCoeffMat, indexFrom=\
                                    indexFrom)
                            elif self.modalDecomp.mpi._numProcs > numModes:
                                self.assertRaises(util.MPIError, 
                                    compute_modes_func, modeNumList, modePath,
                                    snapPaths, buildCoeffMat, indexFrom=\
                                    indexFrom)
                            else:
                                #test the case that only one mode is desired,
                                #in which case user might pass in an int
                                if len(modeNumList) == 1:
                                    modeNumList = modeNumList[0]

                                #saves modes to files
                                compute_modes_func(modeNumList, modePath,
                                    snapPaths, buildCoeffMat, indexFrom=\
                                    indexFrom)

                                #change back to list so is iterable
                                if isinstance(modeNumList, int):
                                    modeNumList = [modeNumList]

                                #mpi barrier to sync procs
                                self.modalDecomp.mpi.sync()

                                #do tests on processor 0
                                if self.modalDecomp.mpi._rank == 0:
                                    for modeNum in modeNumList:
                                        computedMode = util.load_mat_text(
                                            modePath % modeNum)
                                        N.testing.assert_array_almost_equal(
                                            computedMode, trueModes[:,modeNum-\
                                            indexFrom])            
                                self.modalDecomp.mpi.sync()

    @unittest.skipIf(parallel,'This is a serial test')
    def test__compute_modes_chunk(self):
        """
        Test that can compute chunks of modes from arguments.
        
        The modes are computed, saved to file, and read in to test. The
        compute_modes_chunk method is only ever called on a per-proc basis
        so it does not need to be tested seperately for parallel and serial.
        It happens that this test passes in parallel as well, but this is
        not required.
        """
        self.helper_compute_modes(self.modalDecomp._compute_modes_chunk)      
        #self.modalDecomp.mpi.sync()
    
    def test__compute_modes(self):
        """
        Test that can compute modes from arguments. 
        
        This tests the _compute_modes function, which primarily breaks 
        up the tasks for each processor and passes off the computation to
        _compute_modes_chunk. Parallel and serial cases need to be tested
        independently currently (because of file saving/loading/deleting
        problems in the hlper_compute_modes which are fixable). This test
        mostly is a call to helper_compute_modes with the function
        modalDecomp._compute_modes"""
        
        self.helper_compute_modes(self.modalDecomp._compute_modes)
       
        self.modalDecomp.mpi.sync()
        
    @unittest.skipIf(parallel,'Test in serial, method runs on per-proc basis')
    def test__compute_inner_product_chunk(self):
        """
        Test computation of matrix of innerproducts in memory-efficient chunks
        """
        
        def assert_equal_mat_products(mat1,mat2,paths1,paths2):
            productTrue = mat1*mat2
            productComputed = self.modalDecomp._compute_inner_product_chunk(
              paths1,paths2)
            N.testing.assert_array_almost_equal(productComputed,productTrue)
        
        numRowSnapsList =[1,3,20,100]
        numColSnapsList = [1,2,4,20,99]
        numStatesList = [1,10,25]
        maxSnapsInMemList = [6,20,10000]
        if not os.path.isdir('testfiles'):
            SP.call(['mkdir','testfiles'])
        rowSnapPath = 'testfiles/row_snap_%03d.txt'
        colSnapPath = 'testfiles/col_snap_%03d.txt'
        
        self.modalDecomp.load_snap=util.load_mat_text
        self.modalDecomp.save_mode=util.save_mat_text
        self.modalDecomp.inner_product=util.inner_product
        
        for numRowSnaps in numRowSnapsList:
            for numColSnaps in numColSnapsList:
                for numStates in numStatesList:
                    # generate snapshots and save to file
                    rowSnapMat = \
                      N.mat(N.random.random((numStates,numRowSnaps)))
                    colSnapMat = \
                      N.mat(N.random.random((numStates,numColSnaps)))
                    rowSnapPaths = []
                    colSnapPaths = []
                    for snapNum in xrange(numRowSnaps):
                        path = rowSnapPath%snapNum
                        util.save_mat_text(rowSnapMat[:,snapNum],path)
                        rowSnapPaths.append(path)
                    for snapNum in xrange(numColSnaps):
                        path = colSnapPath%snapNum
                        util.save_mat_text(colSnapMat[:,snapNum],path)
                        colSnapPaths.append(path)
                    if len(rowSnapPaths) == 1:
                        rowSnapPaths = rowSnapPaths[0]
                    if len(colSnapPaths) == 1:
                        colSnapPaths = colSnapPaths[0]

                    for maxSnapsInMem in maxSnapsInMemList: 
                        self.modalDecomp.maxSnapsInMem=maxSnapsInMem
                        # Test different rows and cols snapshots
                        assert_equal_mat_products(rowSnapMat.T,colSnapMat,
                          rowSnapPaths,colSnapPaths)
                        #Test with only the row data
                        assert_equal_mat_products(rowSnapMat.T,rowSnapMat,
                          rowSnapPaths,rowSnapPaths)
                        #Test with only the col data
                        assert_equal_mat_products(colSnapMat.T,colSnapMat,
                          colSnapPaths,colSnapPaths)
                        
                    SP.call(['rm -f testfiles/*snap*.txt'],shell=True)
                
    
    
if __name__=='__main__':
    unittest.main(verbosity=2)    

    
    

