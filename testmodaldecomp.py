
import unittest
import numpy as N
import modaldecomp as MD
import util
import subprocess as SP

#import inspect #makes it possible to find information about a function

class TestModalDecomp(unittest.TestCase):
    """ Tests of the ModalDecomp class """
    
    def setUp(self):
        pass
    
    def test_init(self):
        """
        Test arguments passed to the constructor are assigned properly"""
        
        #args = {'load_snap':None,'save_mode':None,'save_mat':None,
        #  'inner_product':None,'maxSnapsInMem':100,'numCPUs':1}
          
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
        
        #no test for numCPUs, this is handled by the util.MPI class


    def helper_compute_modes(self,modalDecomp,compute_modes_func):
        """
        This is a very helpful helper function that takes a modaldecomp
        object modalDecomp. The function compute_modes_func should be a member
        function of modaldecomp. It is called with the same set of args -
        [modeNumList,modePath,snapPaths,buildCoeffMat,indexFrom=indexFrom].
        If this is not what compute_modes_func takes as arguments, then this 
        helper function will fail. 
        This helper function is used by test__compute_modes_* currently. 
        It exists to save a lot
        of duplicate code related to calculating modes for many different
        cases. It only works because each compute_modes_func currently being
        tested take exactly the same arguments!
        """
        
        def load_snap(snapPath): #returns a precomputed, random, vector
            return snapMat[:,snapPath]
        
        numSnapsList = [1,3,20,100]
        numStatesList = [1,10,25]
        numModesList = [1,2,15,80]
        maxSnapsInMemList = [4,20,10000]
        indexFromList = [0,1,5]
        modePath = 'mode_%03d.txt'
        
        modalDecomp.load_snap=load_snap
        modalDecomp.save_mode=util.save_mat_text
        modalDecomp.inner_product=util.inner_product
        
        for numSnaps in numSnapsList:
            for numStates in numStatesList:
                for numModes in numModesList:
                    for maxSnapsInMem in maxSnapsInMemList: 
                        modalDecomp.maxSnapsInMem=maxSnapsInMem
                        for indexFrom in indexFromList:
                          #form the mode list randomly, between indexFrom and numModes+indexFrom-1
                          # the indexFrom only matters when saving and laoding modes
                          modeNumList=[]
                          modeIndex=0
                          while modeIndex < numModes:
                              #currently allows for repeat mode numbers in the list
                              #the function _compute_modes_chunk handles this case.
                              modeNum = indexFrom+int(N.floor(N.random.random()*numModes))
                              if modeNumList.count(modeNum) == 0:
                                  modeNumList.append(modeNum)
                                  modeIndex+=1
                              #else: print 'retrying random modeNum, modeNumList=',modeNumList,'numModes=',numModes,'modeNum=',modeNum,modeNumList.count(modeNum)
                          snapPaths = range(numSnaps)
                          buildCoeffMat = N.mat(N.random.random((numSnaps,numModes)))
                          snapMat = N.mat(N.zeros((numStates,numSnaps)))
                          for snapNum in range(numSnaps):
                              snapMat[:,snapNum] = N.random.random((numStates,1))
                          
                          if numModes > numSnaps or numSnaps > buildCoeffMat.shape[0]:
                              self.assertRaises(ValueError,compute_modes_func,modeNumList,modePath,snapPaths,
                                buildCoeffMat,indexFrom=indexFrom)
                          else:
                              compute_modes_func(modeNumList,modePath,snapPaths,
                                buildCoeffMat,indexFrom=indexFrom)
                              #saves modes to files mode_001.txt, mode_002.txt, etc        
                              #calculate modes via simple matrix eqn
                              trueModes = snapMat*buildCoeffMat
                              for modeNum in modeNumList:
                                  computedMode=(util.load_mat_text(modePath%modeNum))
                                  N.testing.assert_array_almost_equal(computedMode,trueModes[:,modeNum-indexFrom])            
        """
        if modalDecomp.mpi.parallel:
            waitRank = modalDecomp.mpi.rank
            sumOfRanksTrue = 0
            sumOfRanksMeasured = 0
            for r in xrange(modalDecomp.mpi.numCPUs):
                sumOfRanksTrue +=r
            waitRankList=modalDecomp.mpi.comm.gather(waitRank,root=0)
            for r in waitRankList:
                sumOfRanksMeasured +=r
            self.assertEqual(sumOfRanksTrue,sumOfRanksMeasured)
        """
        SP.call(['rm -f mode_*.txt'],shell=True)
        


    def test__compute_modes_chunk(self):
        """
        Test that can compute chunks of modes from arguments.
        
        The modes are saved to file so must read them in to test. The
        compute_modes_chunk method is only ever called on a per-proc basis
        so it does not need to be tested seperately for parallel and serial.
        """
        modalDecomp = MD.ModalDecomp()
        self.helper_compute_modes(modalDecomp,modalDecomp._compute_modes_chunk)      
        
    
    def test__compute_modes_serial(self):
        """
        Test that can compute modes from the arguments.
        
        This only needs to test serial and the components of compute_modes
        that is not handled by compute_modes_chunk. compute_modes_chunk
        does most of the work for actually computing the modes."""
        modalDecomp = MD.ModalDecomp()
        self.helper_compute_modes(modalDecomp,modalDecomp._compute_modes)
        

    def test__compute_modes_parallel(self):
        """
        Test that can compute modes from arguments. 
        
        This only needs to test serial and the components of compute_modes
        that is not handled by compute_modes_chunk. compute_modes_chunk
        does most of the work for actually computing the modes."""
        pass
        
    def test__compute_inner_product_chunk(self):
        """
        Test computation of matrix of inner products via memory-efficient chunks
        """
        pass
    
    
if __name__=='__main__':
    unittest.main()    

    
    

