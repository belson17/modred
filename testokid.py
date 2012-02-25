# Currently the tests compare to a Matlab implementation's results.
# It is hard to judge a correct OKID output. The estimated
# Markov parameters are approximate with different levels of errors for different systems
# and choices of parameters. Generally one needs to tune the parameters for each system to get
# accurate Markov parameters.

import unittest
import numpy as N
from okid import *
import matplotlib.pyplot as PLT
import util

def diff(arr_measured, arr_true, normalize=False):
    err= N.mean((arr_measured-arr_true)**2)
    if normalize:
        return err/N.mean(arr_measured**2)
    else:
        return err
        

class TestOKID(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'files_okid_test/'
        
        
    def test_OKID(self):
        A = util.load_mat_text(self.test_dir+'A.txt')
        B = util.load_mat_text(self.test_dir+'B.txt')
        C = util.load_mat_text(self.test_dir+'C.txt')
        D = util.load_mat_text(self.test_dir+'D.txt')
        num_inputs = B.shape[1]
        num_outputs = C.shape[0]
        num_states = A.shape[0]
        inputs = util.load_mat_text(self.test_dir+'inputs.txt')
        outputs = util.load_mat_text(self.test_dir+'outputs.txt')
        
        nt = inputs.shape[1]
        assert(outputs.shape[1] == nt)
        
        Markovs_true = N.zeros((num_outputs, num_inputs, nt))
        
        temp = util.load_mat_text(self.test_dir+'Markovs_Matlab_output1.txt')
        temp = temp.reshape((num_inputs, -1))
        num_Markovs_OKID = temp.shape[1]
        Markovs_Matlab = N.zeros((num_outputs, num_inputs, num_Markovs_OKID))
                
        for iOut in range(num_outputs):
            Markovs_Matlab[iOut] = util.load_mat_text(
            		self.test_dir+'Markovs_Matlab_output%d.txt'%(iOut+1)).reshape(
            				(num_inputs,num_Markovs_OKID))
            Markovs_true[iOut] = util.load_mat_text(
            		self.test_dir+'Markovs_true_output%d.txt'%(iOut+1)).reshape(
            				(num_inputs,nt))
        
        Markovs_python = OKID(inputs, outputs, num_Markovs_OKID)

        PLT.figure(figsize=(14,10))
        for output_num in range(num_outputs):
            for input_num in range(num_inputs):
                PLT.subplot(num_outputs, num_inputs, output_num*(num_inputs) + input_num + 1)
                PLT.hold(True)
                PLT.plot(Markovs_true[output_num,input_num],'k*-')
                PLT.plot(Markovs_Matlab[output_num,input_num],'b--')
                PLT.plot(Markovs_python[output_num,input_num],'r.')
                PLT.legend(['True','Matlab OKID','Python OKID'])
                PLT.title('Input %d to output %d'%(input_num+1,output_num+1))
        PLT.show()
        #print 'Diff between matlab and python is',diff(Markovs_Matlab, Markovs_python)
        N.testing.assert_allclose(Markovs_python, Markovs_Matlab, atol=1e-1, rtol=1e-1)
  
if __name__ == '__main__':
    unittest.main()
  

