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
    bpodObj.computeDecomp()
    podObj.computeDecomp()
    dmdObj.computeDecomp()
    
    # Test mode construction
    bpodObj.computeModes(1,'')
    podObj.computeModes(1,'')
    dmdObj.computeModes(1,'')
    
    # TO DO: replace this file with unittests