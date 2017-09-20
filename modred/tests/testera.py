#!/usr/bin/env python
"""Test era module"""
from __future__ import division
from future.builtins import range
import unittest
import os
from os.path import join
from shutil import rmtree

import numpy as np

import modred.parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance
from modred import era
from modred import util


def make_time_steps(num_steps, interval):
    """Helper function to find array of integer time steps.

    Args:
        num_steps: integer number of time steps to create.

        interval: interval between pairs of time steps, as shown above.

    Returns:
        time_steps: array of integers, time steps [0 1 interval interval+1 ...]
    """
    if num_steps % 2 != 0:
        raise ValueError('num_steps, %d, must be even'%num_steps)
    interval = int(interval)
    time_steps = np.zeros(num_steps, dtype=int)
    time_steps[::2] = interval*np.arange(num_steps/2)
    time_steps[1::2] = 1 + interval*np.arange(num_steps/2)
    return time_steps


@unittest.skipIf(_parallel.is_distributed(), 'Only test ERA in serial')
class testERA(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DELETE_ME_test_files_era'
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.impulse_file_path = join(self.test_dir, 'impulse_input%03d.txt')

    def tearDown(self):
        """Deletes all of the matrices created by the tests"""
        rmtree(self.test_dir, ignore_errors=True)

    #@unittest.skip('Testing others')
    def test_make_sampled_format(self):
        """
        Test that can give time_values and outputs in either format.

        First tests format [0, 1, P, P+1, ...] and if there is a wrong time
        value.
        Then tests [0, 1, 2, 3, ...] format.
        """
        for num_inputs in [1, 3]:
            for num_outputs in [1, 2, 4]:
                for num_time_steps in [4, 10, 12]:
                    sample_interval = 2
                    # P=2 format [0, 1, 2, 3, ...]
                    dt_system = np.random.random()
                    dt_sample = sample_interval*dt_system
                    outputs = np.random.random((num_time_steps, num_outputs,
                        num_inputs))
                    time_steps = make_time_steps(num_time_steps,
                        sample_interval)
                    time_values = time_steps*dt_system
                    my_ERA = era.ERA()
                    time_steps_computed, outputs_computed = \
                         era.make_sampled_format(time_values, outputs)
                    #self.assertEqual(dt_system_computed, dt_system)
                    num_time_steps_true = (num_time_steps - 1)*2
                    time_steps_true = make_time_steps(num_time_steps_true, 1)
                    outputs_true = np.zeros((num_time_steps_true, num_outputs,
                        num_inputs))
                    outputs_true[::2] = outputs[:-1]
                    outputs_true[1::2] = outputs[1:]
                    np.testing.assert_allclose(time_steps_computed,
                        time_steps_true)
                    np.testing.assert_allclose(outputs_computed, outputs_true)

                    # Test that if there is a wrong time value, get an error
                    time_values[num_time_steps // 2] = -1
                    self.assertRaises(ValueError, era.make_sampled_format,
                        time_values, outputs)

    #@unittest.skip("testing others")
    def test_assemble_Hankel(self):
        """ Tests Hankel mats are symmetric given
        ``[CB CAB CA**P CA**(P+1)B ...]``."""
        for num_inputs in [1,3]:
            for num_outputs in [1, 2, 4]:
                for sample_interval in [1]:
                    num_time_steps = 50
                    num_states = 5
                    A,B,C = util.drss(num_states, num_inputs, num_outputs)
                    time_steps = make_time_steps(num_time_steps,
                        sample_interval)
                    Markovs = util.impulse(A, B, C, time_steps[-1]+1)
                    Markovs = Markovs[time_steps]

                    if sample_interval == 2:
                        time_steps, Markovs = era.make_sampled_format(
                            time_steps, Markovs)

                    myERA = era.ERA(verbosity=0)
                    myERA._set_Markovs(Markovs)
                    myERA._assemble_Hankel()
                    H = myERA.Hankel_mat
                    Hp = myERA.Hankel_mat2

                    for row in range(myERA.mc):
                        for col in range(myERA.mo):
                            np.testing.assert_allclose(
                                H[row*num_outputs:(row+1)*num_outputs,
                                    col*num_inputs:(col+1)*num_inputs],
                                H[col*num_outputs:(col+1)*num_outputs,
                                    row*num_inputs:(row+1)*num_inputs])
                            np.testing.assert_allclose(
                                Hp[row*num_outputs:(row+1)*num_outputs,
                                    col*num_inputs:(col+1)*num_inputs],
                                Hp[col*num_outputs:(col+1)*num_outputs,
                                    row*num_inputs:(row+1)*num_inputs])
                            np.testing.assert_allclose(
                                H[row*num_outputs:(row+1)*num_outputs,
                                    col*num_inputs:(col+1)*num_inputs],
                                C*(A**time_steps[(row+col)*2])*B)
                            np.testing.assert_allclose(
                                Hp[row*num_outputs:(row+1)*num_outputs,
                                    col*num_inputs:(col+1)*num_inputs],
                                C*(A**time_steps[(row+col)*2 + 1])*B)

    #@unittest.skip('testing others')
    def test_compute_model(self):
        """
        Test ROM Markov params similar to those given

        - generates data
        - assembles Hankel matrix
        - computes SVD
        - forms the ROM discrete matrices A, B, and C (D=0)
        - Tests Markov parameters from ROM are approx. equal to full plant's
        """
        num_time_steps = 40
        num_states_plant = 12
        num_states_model = num_states_plant // 3
        for num_inputs in [1, 3]:
            for num_outputs in [1, 2]:
                for sample_interval in [1, 2, 4]:
                    time_steps = make_time_steps(num_time_steps,
                        sample_interval)
                    A, B, C = util.drss(num_states_plant, num_inputs,
                        num_outputs)
                    myERA = era.ERA(verbosity=0)
                    Markovs = util.impulse(A, B, C, time_steps[-1]+1)
                    Markovs = Markovs[time_steps]

                    if sample_interval == 2:
                        time_steps, Markovs = \
                            era.make_sampled_format(time_steps, Markovs)
                    num_time_steps = time_steps.shape[0]

                    A_path_computed = join(self.test_dir, 'A_computed.txt')
                    B_path_computed = join(self.test_dir, 'B_computed.txt')
                    C_path_computed = join(self.test_dir, 'C_computed.txt')

                    A, B, C = myERA.compute_model(Markovs, num_states_model)
                    myERA.put_model(A_path_computed, B_path_computed,
                        C_path_computed)
                    #sing_vals = myERA.sing_vals[:num_states_model]

                    # Flatten vecs into 2D X and Y mats:
                    # [B AB A**PB A**(P+1)B ...]
                    #direct_vecs_flat = np.mat(
                    #    direct_vecs.swapaxes(0,1).reshape(
                    #    (num_states_model,-1)))

                    # Exact grammians from Lyapunov eqn solve
                    #gram_cont = util.solve_Lyapunov(A, B*B.H)
                    #gram_obs = util.solve_Lyapunov(A.H, C.H*C)
                    #print np.sort(np.linalg.eig(gram_cont)[0])[::-1]
                    #print sing_vals
                    #np.testing.assert_allclose(gram_cont.diagonal(),
                    #    sing_vals, atol=.1, rtol=.1)
                    #np.testing.assert_allclose(gram_obs.diagonal(),
                    #   sing_vals, atol=.1, rtol=.1)
                    #np.testing.assert_allclose(np.sort(np.linalg.eig(
                    #   gram_cont)[0])[::-1], sing_vals,
                    #    atol=.1, rtol=.1)
                    #np.testing.assert_allclose(np.sort(np.linalg.eig(
                    #   gram_obs)[0])[::-1], sing_vals,
                    #    atol=.1, rtol=.1)

                    # Check that the diagonals are largest entry on each row
                    #self.assertTrue((np.max(np.abs(gram_cont),axis=1) ==
                    #    np.abs(gram_cont.diagonal())).all())
                    #self.assertTrue((np.max(np.abs(gram_obs),axis=1) ==
                    #    np.abs(gram_obs.diagonal())).all())

                    # Check the ROM Markov params match the full plant's
                    Markovs_model = np.zeros(Markovs.shape)
                    for ti, tv in enumerate(time_steps):
                        Markovs_model[ti] = C * (A**tv) * B
                        #print 'computing ROM Markov param at time step %d'%tv
                    """
                    import matplotlib.pyplot as PLT
                    for input_num in range(num_inputs):
                        PLT.figure()
                        PLT.hold(True)
                        for output_num in range(num_outputs):
                            PLT.plot(time_steps[:50],
                            #   Markovs_model[:50, output_num,input_num], 'ko')
                            PLT.plot(time_steps[:50],Markovs[:50,
                            #   output_num, input_num],'rx')
                            PLT.plot(time_steps_dense[:50],
                            #   Markovs_dense[:50, output_num, input_num],'b--')
                            PLT.title('input %d to outputs'%input_num)
                            PLT.legend(['ROM','Plant','Dense plant'])
                        PLT.show()
                    """
                    np.testing.assert_allclose(
                        Markovs_model, Markovs, rtol=0.5, atol=0.5)
                    np.testing.assert_allclose(
                        util.load_array_text(A_path_computed), A)
                    np.testing.assert_allclose(
                        util.load_array_text(B_path_computed), B)
                    np.testing.assert_allclose(
                        util.load_array_text(C_path_computed), C)


if __name__ == '__main__':
    unittest.main()
