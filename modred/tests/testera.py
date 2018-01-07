#!/usr/bin/env python
"""Test era module"""
import unittest
import os
from os.path import join
from shutil import rmtree

import numpy as np
import scipy.signal
import scipy

from modred import era, parallel, util
from modred.py2to3 import range


def make_time_steps(num_steps, interval):
    """Helper function to find array of integer time steps.

    Args:
        num_steps: integer number of time steps to create.

        interval: interval between pairs of time steps in return value.

    Returns:
        time_steps: array of integer time steps with len==num_steps, [0 1 interval interval+1 ...]
    """
    if num_steps % 2 != 0:
        raise ValueError('num_steps, %d, must be even'%num_steps)
    interval = int(interval)
    time_steps = np.zeros(num_steps, dtype=int)
    time_steps[::2] = interval*np.arange(num_steps/2)
    time_steps[1::2] = 1 + interval*np.arange(num_steps/2)
    return time_steps


@unittest.skipIf(parallel.is_distributed(), 'Only test ERA in serial')
class testERA(unittest.TestCase):
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'files_ERA_DELETE_ME'
        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)
        self.impulse_file_path = join(self.test_dir, 'impulse_input%03d.txt')


    def tearDown(self):
        """Deletes all of the arrays created by the tests"""
        rmtree(self.test_dir, ignore_errors=True)


    # @unittest.skip('Testing others')
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
                    # Generate data
                    # P=2 format [0, 1, 2, 3, ...]
                    sample_interval = 2
                    dt_system = np.random.random()
                    dt_sample = sample_interval * dt_system
                    outputs = np.random.random(
                        (num_time_steps, num_outputs, num_inputs))
                    time_steps = make_time_steps(
                        num_time_steps, sample_interval)
                    time_values = time_steps * dt_system

                    # Compute using modred
                    my_ERA = era.ERA()
                    time_steps_computed, outputs_computed =\
                        era.make_sampled_format(time_values, outputs)
                    #self.assertEqual(dt_system_computed, dt_system)

                    # Reference values
                    num_time_steps_true = (num_time_steps - 1) * 2
                    time_steps_true = make_time_steps(num_time_steps_true, 1)
                    outputs_true = np.zeros(
                        (num_time_steps_true, num_outputs, num_inputs))
                    outputs_true[::2] = outputs[:-1]
                    outputs_true[1::2] = outputs[1:]

                    # Compare values
                    np.testing.assert_equal(
                        time_steps_computed, time_steps_true)
                    np.testing.assert_equal(outputs_computed, outputs_true)

                    # Test that if there is a wrong time value, get an error
                    time_values[num_time_steps // 2] = -1
                    self.assertRaises(
                        ValueError, era.make_sampled_format, time_values,
                        outputs)


    # @unittest.skip("testing others")
    def test_assemble_Hankel(self):
        """ Tests Hankel arrays are symmetric and accurate given Markov params
        ``[CB CAB CA**P CA**(P+1)B ...]``."""
        rtol = 1e-10
        atol = 1e-12
        for num_inputs in [1, 3]:
            for num_outputs in [1, 2, 4]:
                for sample_interval in [1]:
                    num_time_steps = 50
                    num_states = 5
                    A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    time_steps = make_time_steps(
                        num_time_steps, sample_interval)
                    impulse_response = util.impulse(A, B, C, time_steps[-1] + 1)
                    Markovs = impulse_response[time_steps]

                    if sample_interval == 2:
                        time_steps, Markovs = era.make_sampled_format(
                            time_steps, Markovs)

                    my_ERA = era.ERA(verbosity=0)
                    my_ERA._set_Markovs(Markovs)
                    my_ERA._assemble_Hankel()
                    H = my_ERA.Hankel_array
                    Hp = my_ERA.Hankel_array2

                    for row in range(my_ERA.mc):
                        for col in range(my_ERA.mo):
                            # Test values in H are accurate using that, roughly, H[r,c] = C * A^(r+c) * B.
                            np.testing.assert_allclose(
                                H[row * num_outputs:(row + 1) * num_outputs,
                                col * num_inputs:(col + 1) * num_inputs],
                                C.dot(
                                    np.linalg.matrix_power(
                                        A, time_steps[(row + col) * 2]).dot(
                                        B)),
                                rtol=rtol, atol=atol)
                            # Test values in H are accurate using that, roughly, H[r,c] = C * A^(r+c+1) * B.
                            np.testing.assert_allclose(
                                Hp[row * num_outputs:(row + 1) * num_outputs,
                                   col * num_inputs:(col + 1) * num_inputs],
                                C.dot(
                                    np.linalg.matrix_power(
                                        A, time_steps[(row + col) * 2 + 1]).dot(
                                            B)),
                                rtol=rtol, atol=atol)
                            # Test H is block symmetric
                            np.testing.assert_equal(
                                H[row * num_outputs:(row + 1) * num_outputs,
                                  col * num_inputs:(col + 1) * num_inputs],
                                H[col * num_outputs:(col + 1) * num_outputs,
                                  row * num_inputs:(row + 1) * num_inputs])
                            # Test Hp is block symmetric
                            np.testing.assert_equal(
                                Hp[row * num_outputs:(row + 1) * num_outputs,
                                   col * num_inputs:(col + 1) * num_inputs],
                                Hp[col * num_outputs:(col + 1) * num_outputs,
                                   row * num_inputs:(row + 1) * num_inputs])


    # @unittest.skip('testing others')
    def test_compute_model(self):
        """
        Test ROM Markov params similar to those given when the reduced system has
        the same number of states as the full system.

        - generates data
        - assembles Hankel array
        - computes SVD
        - forms the ROM discrete arrays A, B, and C (D = 0)
        - Tests Markov parameters from ROM are approx. equal to full plant's

        Also, unrelated:
        - Tests that saved ROM mats are equal to those returned in memory
        """
        dt = 1
        inf_norm_tol = 0.1
        num_time_steps = 40
        # Using more than 8 states causes poorly conditioned TF coeffs (https://github.com/scipy/scipy/issues/2980)
        num_states_plant = 8
        num_states_model = num_states_plant
        for num_inputs in [1, 3]:
            for num_outputs in [1, 2]:
                for sample_interval in [1, 2, 4]:
                    time_steps = make_time_steps(
                        num_time_steps, sample_interval)
                    A, B, C = util.drss(
                        num_states_plant, num_inputs, num_outputs)
                    my_ERA = era.ERA(verbosity=0)

                    Markovs = util.impulse(A, B, C, time_steps[-1] + 1)
                    Markovs = Markovs[time_steps]

                    if sample_interval == 2:
                        time_steps_sampled, Markovs_sampled =\
                            era.make_sampled_format(time_steps, Markovs)
                    # else:
                    #     time_steps_sampled = time_steps
                    #     Markovs_sampled = Markovs
                    num_time_steps = time_steps.shape[0]

                    A_path_computed = join(self.test_dir, 'A_computed.txt')
                    B_path_computed = join(self.test_dir, 'B_computed.txt')
                    C_path_computed = join(self.test_dir, 'C_computed.txt')

                    A_reduced, B_reduced, C_reduced = \
                        my_ERA.compute_model(Markovs, num_states_model)
                    my_ERA.put_model(
                        A_path_computed, B_path_computed, C_path_computed)

                    # Scipy can't handle MIMO transfer functions,
                    # so for each input-output pair, find the inf norm of TF_full - TF_model.
                    inf_norm_error = 0
                    for input_index in range(num_inputs):
                        for output_index in range(num_outputs):
                            print(np.mat(B)[:, input_index], np.mat(C)[output_index, :])
                            tf_full = scipy.signal.StateSpace(
                                A, np.mat(B)[:, input_index], np.mat(C)[output_index, :], 0, dt=dt).to_tf()
                            tf_red = scipy.signal.StateSpace(
                                A_reduced, np.mat(B_reduced)[:, input_index], np.mat(C_reduced)[output_index, :], 0,
                                dt=dt).to_tf()
                            tf_diff = util.sub_transfer_functions(tf_full, tf_red, dt=dt)
                            inf_norm_error = util.compute_inf_norm_discrete(tf_diff, dt)
                            self.assertTrue(inf_norm_error < inf_norm_tol +
                                            inf_norm_tol * util.compute_inf_norm_discrete(tf_full, dt))

                    # Check normalized Markovs
                    # Markovs_reduced = util.impulse(A_reduced, B_reduced, C_reduced, time_steps[-1] + 1)
                    # Markovs_reduced = Markovs_reduced[time_steps]
                    # max_Markov = np.amax(Markovs)
                    # np.testing.assert_allclose(Markovs_reduced/max_Markov, Markovs/max_Markov, rtol=rtol, atol=atol)

                    # Also test that saved reduced model mats are equal to those returned in memory
                    np.testing.assert_equal(
                        util.load_array_text(A_path_computed), A_reduced)
                    np.testing.assert_equal(
                        util.load_array_text(B_path_computed), B_reduced)
                    np.testing.assert_equal(
                        util.load_array_text(C_path_computed), C_reduced)



if __name__ == '__main__':
    unittest.main()
