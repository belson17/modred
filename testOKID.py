
import unittest
import numpy as N
from OKID import *
# Scipy for simulating LTI
import scipy.signal as SS
import odesolve 
import matplotlib.pyplot as PLT
import util

def diff(arr_measured, arr_true, normalize = False):
    err= N.mean((arr_measured-arr_true)**2)
    if normalize:
        return err/N.mean(arr_measured**2)
    else:
        return err
    

class TestOKID(unittest.TestCase):
    def setUp(self):
        pass
               
    def test_OKID(self):
        A = N.mat([[0.4933,  -0.4114],[0.4114,0.4933]])
        B = N.mat([[-1.164],[.4948]])
        C = N.mat([-1.216, 1.591])
        
        for nt in [50]:
            for num_states in [5]:
                for num_inputs in [1]:
                    for num_outputs in [1]:
                        num_Markovs = nt
                        A,B,C = util.drss(num_states, num_inputs, num_outputs)
                        time_steps = N.arange(nt, dtype=int)
                        
                        time_steps, output_impulse = util.impulse(A, B, C, time_steps=time_steps)
                        inputs = N.concatenate((N.random.random((nt/2,num_inputs)),
                            N.zeros((nt/2,num_inputs))), axis=0)
                        #print 'num states',num_states,'num inputs',num_inputs
                        #print 'B.shape',B.shape
                        outputs = util.lsim(A, B, C, inputs)
                        #print 'input signal',inputs.shape
                        #print 'output signal',outputs.shape
                        
                        output_impulse_OKID = OKID(inputs, outputs, num_Markovs)
                        """
                        for input_num in range(num_inputs):
                            PLT.figure()
                            PLT.hold(True)
                            PLT.plot(time_steps, output_impulse[:,:,input_num])
                            PLT.plot(time_steps, output_impulse_OKID[:,:,input_num])
                            PLT.plot(time_steps, N.abs(output_impulse - output_impulse_OKID)[:,:,input_num])
                            PLT.legend(['true','OKID','error'])
                        PLT.show()
                        """
                        #print 'error is',N.amax(N.abs(output_impulse-output_impulse_OKID))
                        #print output_impulse[1:-1]/output_impulse_OKID[1:-1]
                        #print 'output signal',outputs
                        #print 'output impulse',output_impulse
                        #print 'output OKID',output_impulse_OKID
                        N.testing.assert_allclose(output_impulse_OKID.squeeze(), 
                            output_impulse[:num_Markovs].squeeze(), atol=1e-1, rtol=1e-1)

  
if __name__ == '__main__':
    unittest.main()
  

