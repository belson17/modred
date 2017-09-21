#!/usr/bin/env python
"""Test OKID"""
from __future__ import division
from future.builtins import range
import os
from os.path import join
import unittest

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
from modred.okid import OKID
from modred import util


# Useful for debugging, makes plots
plot = False
if plot:
    try:
        import matplotlib.pyplot as PLT
    except:
        plot = False


def diff(arr_measured, arr_true, normalize=False):
    err = np.mean((arr_measured-arr_true)**2)
    if normalize:
        return err/np.mean(arr_measured**2)
    else:
        return err


@unittest.skipIf(_parallel.is_distributed(), 'Only test OKID in serial')
class TestOKID(unittest.TestCase):
    def setUp(self):
        self.test_dir = join(os.path.dirname(__file__), 'files_OKID')

    def tearDown(self):
        pass

    def test_OKID(self):
        for case in ['SISO', 'SIMO', 'MISO', 'MIMO']:
            inputs = util.load_array_text(
                join(join(self.test_dir, case), 'inputs.txt'))
            outputs = util.load_array_text(
                join(join(self.test_dir, case), 'outputs.txt'))
            (num_inputs, nt) = inputs.shape
            (num_outputs, nt2) = outputs.shape

            assert(nt2 == nt)

            Markovs_true = np.zeros((nt, num_outputs, num_inputs))

            temp = util.load_array_text(join(join(self.test_dir, case),
                'Markovs_Matlab_output1.txt'))
            temp = temp.reshape((num_inputs, -1))
            num_Markovs_OKID = temp.shape[1]
            Markovs_Matlab = np.zeros(
                (num_Markovs_OKID, num_outputs, num_inputs))

            for iOut in range(num_outputs):
                data = util.load_array_text(
                    join(join( self.test_dir, case),
                    'Markovs_Matlab_output%d.txt'%(iOut+1)))
                if num_inputs > 1:
                    data = np.swapaxes(data, 0, 1)
                Markovs_Matlab[:,iOut,:] = data
                data = util.load_array_text(join(join(
                    self.test_dir, case), 'Markovs_true_output%d.txt'%(iOut+1)))
                if num_inputs > 1:
                    data = np.swapaxes(data, 0, 1)
                Markovs_true[:,iOut,:] = data

            Markovs_python = OKID(inputs, outputs, num_Markovs_OKID)

            if plot:
                PLT.figure(figsize=(14,10))
                for output_num in range(num_outputs):
                    for input_num in range(num_inputs):
                        PLT.subplot(num_outputs, num_inputs,
                            output_num*(num_inputs) + input_num + 1)
                        PLT.hold(True)
                        PLT.plot(Markovs_true[:,output_num,input_num],'k*-')
                        PLT.plot(Markovs_Matlab[:,output_num,input_num],'b--')
                        PLT.plot(Markovs_python[:,output_num,input_num],'r.')
                        PLT.legend(['True', 'Matlab OKID', 'Python OKID'])
                        PLT.title('Input %d to output %d'%(input_num+1,
                            output_num+1))
                PLT.show()
            #print (
            #    'Diff between matlab and python is',
            #    diff(Markovs_Matlab, Markovs_python)
            np.testing.assert_allclose(
                Markovs_python, Markovs_Matlab, atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(
                Markovs_python, Markovs_true[:num_Markovs_OKID], atol=1e-3,
                rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
