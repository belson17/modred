# Example main routine for testing modules
import util
from modaldecomp import ModalDecomp
from bpod import BPOD
from pod import POD
from dmd import DMD

if __name__=='__main__':
    # Test constructors
    bpodObj = BPOD()
    podObj = POD()
    dmdObj = DMD()
    
    # Test decomposition
    bpodObj.compute_decomp()
    podObj.compute_decomp()
    dmdObj.compute_decomp()
    
    # Test mode construction
    bpodObj.compute_modes(1,'')
    podObj.compute_modes(1,'')
    dmdObj.compute_modes(1,'')
    
    # TO DO: replace this file with unittests
