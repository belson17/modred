#!/usr/bin/env python
"""Test OKID"""
import os
from os.path import join
import unittest

import numpy as np

import modred.parallel as parallel
from modred.okid import OKID
from modred import util
from modred.py2to3 import range


# Useful for debugging, makes plots
plot = False
if plot:
    try:
        import matplotlib.pyplot as plt
    except:
        plot = False


def diff(arr_measured, arr_true, normalize=False):
    err = np.mean((arr_measured - arr_true)**2)
    if normalize:
        return err / np.mean(arr_measured ** 2)
    else:
        return err


@unittest.skipIf(parallel.is_distributed(), 'Only test OKID in serial')
class TestOKID(unittest.TestCase):
    def setUp(self):
        self.test_dir = join(os.path.dirname(__file__), 'files_OKID')


    def tearDown(self):
        pass


    def test_OKID(self):
        rtol = 1e-8
        atol = 1e-10

        for case in ['SISO', 'SIMO', 'MISO', 'MIMO']:
            inputs = util.load_array_text(
                join(join(self.test_dir, case), 'inputs.txt'))
            outputs = util.load_array_text(
                join(join(self.test_dir, case), 'outputs.txt'))
            (num_inputs, nt) = inputs.shape
            (num_outputs, nt2) = outputs.shape

            assert(nt2 == nt)

            Markovs_true = np.zeros((nt, num_outputs, num_inputs))

            tmp = util.load_array_text(
                join(join(self.test_dir, case), 'Markovs_Matlab_output1.txt'))
            tmp = tmp.reshape((num_inputs, -1))
            num_Markovs_OKID = tmp.shape[1]
            Markovs_Matlab = np.zeros(
                (num_Markovs_OKID, num_outputs, num_inputs))

            for i_out in range(num_outputs):
                data = util.load_array_text(
                    join(join( self.test_dir, case),
                    'Markovs_Matlab_output%d.txt' % (i_out + 1)))
                if num_inputs > 1:
                    data = np.swapaxes(data, 0, 1)
                Markovs_Matlab[:, i_out, :] = data
                data = util.load_array_text(join(
                    join(self.test_dir, case),
                    'Markovs_true_output%d.txt' % (i_out + 1)))
                if num_inputs > 1:
                    data = np.swapaxes(data, 0, 1)
                Markovs_true[:,i_out,:] = data

            Markovs_python = OKID(inputs, outputs, num_Markovs_OKID)

            if plot:
                plt.figure(figsize=(14,10))
                for output_num in range(num_outputs):
                    for input_num in range(num_inputs):
                        plt.subplot(num_outputs, num_inputs,
                            output_num*(num_inputs) + input_num + 1)
                        plt.hold(True)
                        plt.plot(Markovs_true[:,output_num,input_num],'k*-')
                        plt.plot(Markovs_Matlab[:,output_num,input_num],'b--')
                        plt.plot(Markovs_python[:,output_num,input_num],'r.')
                        plt.legend(['True', 'Matlab OKID', 'Python OKID'])
                        plt.title('Input %d to output %d'%(input_num+1,
                            output_num+1))
                plt.show()

            np.testing.assert_allclose(
                Markovs_python.squeeze(), Markovs_Matlab.squeeze(),
                rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                Markovs_python.squeeze(),
                Markovs_true[:num_Markovs_OKID].squeeze(),
                rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
