# Example main routine for testing modules
import util
from modaldecomp import ModalDecomp
from bpod import BPOD
from pod import POD
from dmd import DMD

if __name__=='__main__':
    # Test BPOD class
    print '\n% --- Testing BPOD class --- %'
    bpodObj = BPOD()
    bpodObj.directSnapPaths=[]
    bpodObj.adjointSnapPaths=[]
    #bpodObj.compute_decomp()
    #bpodObj.compute_direct_modes(1,'')
    #bpodObj.compute_adjoint_modes(1,'')
   
    # Test POD class
    print '\n% --- Testing POD class: --- %'
    podObj = POD()
    podObj.snapPaths=[]
    podObj.compute_decomp()
    podObj.compute_modes(1,'')
    
    # Test DMD class
    print '\n% --- Testing DMD class: --- %'
    dmdObj = DMD()
    dmdObj.compute_decomp()
    dmdObj.compute_modes(1,'')
    print ''
    
    
    # TO DO: replace this file with unittests
