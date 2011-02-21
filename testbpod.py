
import numpy as N
import bpod as BP
import unittest
import util
import subprocess as SP

try:
    from mpi4py import MPI
    parallel = MPI.COMM_WORLD.Get_size() >=2
except ImportError:
    parallel = False

if parallel:
    if MPI.COMM_WORLD.Get_rank()==0:
        print 'Remember to test in serial also with command:'
        print 'python bpod.py'
else:
    print 'Remember to test in parallel also with command:'
    print 'mpiexec -n <numProcs> python testbpod.py' 

class TestBPOD(unittest.TestCase):
    """ Test all the BPOD class methods """
    
    def setUp(self):
        self.bpod = BP.BPOD()
        self.modeNumList =[1,4,3,6]
        self.numDirectSnaps = 6
        self.numAdjointSnaps = 7
        self.numStates = 7
        self.bpod.save_mat=util.save_mat_text
        self.bpod.load_snap=util.load_mat_text
        self.bpod.inner_product=util.inner_product
        self.bpod.save_mode = util.save_mat_text
        self.indexFrom=1
        self.bpod.indexFrom=self.indexFrom
        
        self.generate_data_set()
        
    def generate_data_set(self):
        #create data set (saved to file)
        self.directSnapPath ='testfiles/direct_snap_%03d.txt'
        self.adjointSnapPath ='testfiles/adjoint_snap_%03d.txt'

        self.directSnapPaths=[]
        self.adjointSnapPaths=[]
        self.directSnapMat = N.mat(
          N.random.random((self.numStates,self.numDirectSnaps)))
        self.adjointSnapMat = N.mat(
          N.random.random((self.numStates,self.numAdjointSnaps))) 
        for directSnapNum in range(self.numDirectSnaps):
            util.save_mat_text(self.directSnapMat[:,directSnapNum],
              self.directSnapPath%directSnapNum)
            self.directSnapPaths.append(self.directSnapPath%directSnapNum)
        for adjointSnapNum in range(self.numAdjointSnaps):
            util.save_mat_text(self.adjointSnapMat[:,adjointSnapNum],
              self.adjointSnapPath%adjointSnapNum)
            self.adjointSnapPaths.append(self.adjointSnapPath%adjointSnapNum)
        
        self.hankelMatTrue=self.adjointSnapMat.T*self.directSnapMat
        self.LSingVecsTrue,self.singValsTrue,self.RSingVecsTrue=\
          util.svd(self.hankelMatTrue)
        
        self.directModeMat = self.directSnapMat * N.mat(self.RSingVecsTrue)*\
          N.mat(N.diag(self.singValsTrue**(-0.5)))
        self.adjointModeMat = self.adjointSnapMat*N.mat(self.LSingVecsTrue)*\
          N.mat(N.diag(self.singValsTrue**(-0.5)))
        
        self.bpod.directSnapPaths=self.directSnapPaths
        self.bpod.adjointSnapPaths=self.adjointSnapPaths
        
        
    def tearDown(self):
        SP.call(['rm -f testfiles/*'],shell=True)
    
    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
          
        def my_load(fname): 
            return 0
        myBPOD = BP.BPOD(load_snap=my_load)
        self.assertEqual(myBPOD.load_snap,my_load)
        
        def my_save(data,fname):
            pass 
        myBPOD = BP.BPOD(save_mode=my_save)
        self.assertEqual(myBPOD.save_mode,my_save)
        
        myBPOD = BP.BPOD(save_mat=my_save)
        self.assertEqual(myBPOD.save_mat,my_save)
        
        def my_ip(f1,f2): return 0
        myBPOD = BP.BPOD(inner_product=my_ip)
        self.assertEqual(myBPOD.inner_product,my_ip)
        
        maxSnaps = 500
        myBPOD = BP.BPOD(maxSnapsInMem=maxSnaps)
        self.assertEqual(myBPOD.maxSnapsInMem,maxSnaps)
        
        directSnapPaths=['a','b']
        myBPOD = BP.BPOD(directSnapPaths = directSnapPaths)
        self.assertEqual(myBPOD.directSnapPaths,directSnapPaths)
        
        adjointSnapPaths=['a','c']
        myBPOD = BP.BPOD(adjointSnapPaths = adjointSnapPaths)
        self.assertEqual(myBPOD.adjointSnapPaths,adjointSnapPaths)
        
        LSingVecs=N.mat(N.random.random((2,2)))
        myBPOD = BP.BPOD(LSingVecs = LSingVecs)
        N.testing.assert_array_almost_equal(myBPOD.LSingVecs,LSingVecs)
        
        RSingVecs = N.mat(N.random.random((2,2)))
        myBPOD = BP.BPOD(RSingVecs = RSingVecs)
        N.testing.assert_array_almost_equal(myBPOD.RSingVecs,RSingVecs)
        
        singVals = N.mat(N.random.random((2,2)))
        myBPOD = BP.BPOD(singVals = singVals)
        N.testing.assert_array_almost_equal(myBPOD.singVals,singVals)
        
        hankelMat = N.mat(N.random.random((2,2)))
        myBPOD = BP.BPOD(hankelMat = hankelMat)
        N.testing.assert_array_almost_equal(myBPOD.hankelMat,hankelMat)
        
    
    def test__compute_hankel_serial(self):
        """Tests that _compute_decomp works in serial"""
        pass
        
    def test__compute_hankel_parallel(self):
        """Tests that _compute_decomp works in parallel"""
        pass

        
    def test_compute_decomp(self):
        """Test that can take snapshots and compute the SVD matrices
        
        The testing is minimal here because the SVD is thoroughly tested
        in testutil.py
        """
        tol = 8
        directSnapPath = 'testfiles/direct_snap_%03d.txt'
        adjointSnapPath = 'testfiles/adjoint_snap_%03d.txt'
        LSingVecsPath ='testfiles/lsingvecs.txt'
        RSingVecsPath ='testfiles/rsingvecs.txt'
        singValsPath ='testfiles/singvals.txt'
        hankelMatPath='testfiles/hankel.txt'
        
        self.bpod.compute_decomp(RSingVecsPath=RSingVecsPath,
          LSingVecsPath=LSingVecsPath,singValsPath=singValsPath,
          hankelMatPath=hankelMatPath)
                
        LSingVecsLoaded = util.load_mat_text(LSingVecsPath)
        RSingVecsLoaded = util.load_mat_text(RSingVecsPath)
        singValsLoaded = N.squeeze(N.array(
          util.load_mat_text(singValsPath)))
        hankelMatLoaded = util.load_mat_text(hankelMatPath)
        
        N.testing.assert_array_almost_equal(self.bpod.hankelMat,
          self.hankelMatTrue,decimal=tol)
        N.testing.assert_array_almost_equal(self.bpod.LSingVecs,
          self.LSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(self.bpod.RSingVecs,
          self.RSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(self.bpod.singVals,
          self.singValsTrue,decimal=tol)
          
        N.testing.assert_array_almost_equal(hankelMatLoaded,
          self.hankelMatTrue,decimal=tol)
        N.testing.assert_array_almost_equal(LSingVecsLoaded,
          self.LSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(RSingVecsLoaded,
          self.RSingVecsTrue,decimal=tol)
        N.testing.assert_array_almost_equal(singValsLoaded,
          self.singValsTrue,decimal=tol)        

    def test_compute_modes(self):
        """
        Only testing unique aspects, assuming ModalDecomp is tested.
        """
        #compute_direct_modes takes arguments modeNumList, modePath, indexFrom
        #AND requires self.directSnapPaths to be set. 

        directModePath ='testfiles/direct_mode_%03d.txt'
        adjointModePath ='testfiles/adjoint_mode_%03d.txt'
        
        self.bpod.RSingVecs=self.RSingVecsTrue
        self.bpod.LSingVecs=self.LSingVecsTrue
        self.bpod.singVals=self.singValsTrue
        self.bpod.compute_direct_modes(self.modeNumList,directModePath,
          indexFrom=self.indexFrom)
        self.bpod.compute_adjoint_modes(self.modeNumList,adjointModePath,
          indexFrom=self.indexFrom)
          
        directModesComp = []
        adjointModesComp = []
        for modeNum in self.modeNumList:
            snap = util.load_mat_text(directModePath%modeNum)
            N.testing.assert_array_almost_equal(snap,self.directModeMat[:,modeNum-self.indexFrom])
            snap=util.load_mat_text(adjointModePath%modeNum)
            N.testing.assert_array_almost_equal(snap,self.adjointModeMat[:,modeNum-self.indexFrom])
        
        for modeNum1 in self.modeNumList:
            directMode = util.load_mat_text(
              directModePath%modeNum1)
            for modeNum2 in self.modeNumList:
                adjointMode = util.load_mat_text(
                  adjointModePath%modeNum2)
                innerProduct = self.bpod.inner_product(
                  directMode,adjointMode)
                if modeNum1 != modeNum2:
                    self.assertAlmostEqual(innerProduct,0.)
                else:
                    self.assertAlmostEqual(innerProduct,1.)
  
if __name__=='__main__':
    unittest.main()
    
        
    


