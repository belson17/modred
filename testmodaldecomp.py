
import unittest
import numpy as N
import modaldecomp as MD
import util

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

    def test__compute_modes_chunk(self):
        """
        Test that can compute chunks of modes from arguments.
        
        The modes are saved to file so must read them in to test. The
        compute_modes_chunk method is only ever called on a per-proc basis
        so it does not need to be tested seperately for parallel and serial.
        """
        
        #modeNumList,modePath,snapPaths,buildCoeffMat
        numSnaps = 3
        numStates = 5
        numModes = 3
        maxSnapsInMem = 2
        modeNumList = range(1,numModes+1)
        modePath = 'mode_%03d'
        snapPaths = range(numSnaps)
        buildCoeffMat = N.mat(N.random.random((numSnaps,numModes)))
        
        snaps = []        
        X = N.mat(N.zeros((numStates,numSnaps)))
        for snapNum in range(numSnaps):
            snaps.append(N.random.random((numStates,1)))
            X[:,snapNum] = snaps[snapNum]
            print 'snapnum in test',snapNum
            print 'snap in test',snaps[snapNum]
        def load_snap(snapPath): #returns a precomputed, random, vector
            return snaps[snapPath]
                        
        myMD = MD.ModalDecomp(load_snap=load_snap,save_mode=util.save_mat_text,
          save_mat=util.save_mat_text,inner_product=util.inner_product)
        myMD.maxSnapsInMem=maxSnapsInMem
        myMD._compute_modes_chunk(modeNumList,modePath,snapPaths,
          buildCoeffMat)
        #saves modes to files mode_001.txt, mode_002.txt, etc
        
        trueModes = X*buildCoeffMat
        print 'true modes'
        print trueModes
        for modeNum in modeNumList:
            print 'modeNum',modeNum
            computedMode=(util.load_mat_text(modePath%modeNum))
            N.testing.assert_array_almost_equal(computedMode,trueModes[:,modeNum-1]) #-1?            
            print computedMode
        #calculate all modes via matrix equations
        
        
        
        
    
    def test__compute_modes_serial(self):
        """
        Test that can compute modes from the arguments.
        
        This only needs to test serial and the components of compute_modes
        that is not handled by compute_modes_chunk. compute_modes_chunk
        does most of the work for actually computing the modes."""
        pass

    def test__compute_modes_parallel(self):
        """
        Test that can compute modes from arguments. 
        
        This only needs to test serial and the components of compute_modes
        that is not handled by compute_modes_chunk. compute_modes_chunk
        does most of the work for actually computing the modes."""
        pass
        
if __name__=='__main__':
    unittest.main()    

    
    

