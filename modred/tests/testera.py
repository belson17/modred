#!/usr/bin/env python
"""Test era module"""
import unittest
import os
from os.path import join
from shutil import rmtree

import numpy as np
import scipy.signal
import scipy
import matplotlib.pyplot as plt

from modred import era, parallel, util
from modred.py2to3 import range


def make_time_steps(num_steps, interval):
    """Helper function to find array of integer time steps.

    Args:
        num_steps: integer number of time steps to create.

        interval: interval between pairs of time steps in return value.

    Returns:
        time_steps: array of integer time steps with len==num_steps,
        [0 1 interval interval+1 ...]
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
        self.data_dir = 'files_ERA'
        self.out_dir = 'files_ERA_DELETE_ME'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.impulse_file_path = join(self.out_dir, 'impulse_input%03d.txt')


    def tearDown(self):
        """Deletes all of the arrays created by the tests"""
        rmtree(self.out_dir, ignore_errors=True)


    # @unittest.skip('Testing others')
    def test_make_sampled_format(self):
        """
        Test that can give time_values and outputs in either format.

        First tests format [0, 1, P, P+1, ...] and if there is a wrong time
        value.  Then tests [0, 1, 2, 3, ...] format.
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
                    num_states = 8
                    # A, B, C = util.drss(num_states, num_inputs, num_outputs)
                    time_steps = make_time_steps(
                        num_time_steps, sample_interval)
                    A = util.load_array_text(
                        join(self.data_dir, 'A_in%d_out%d.txt') % (
                            num_inputs, num_outputs))
                    B = util.load_array_text(
                        join(self.data_dir, 'B_in%d_out%d.txt') % (
                            num_inputs, num_outputs))
                    C = util.load_array_text(
                        join(self.data_dir, 'C_in%d_out%d.txt') % (
                            num_inputs, num_outputs))
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
                            # Test values in H are accurate using that, roughly,
                            # H[r,c] = C * A^(r+c) * B.
                            np.testing.assert_allclose(
                                H[row * num_outputs:(row + 1) * num_outputs,
                                col * num_inputs:(col + 1) * num_inputs],
                                C.dot(
                                    np.linalg.matrix_power(
                                        A, time_steps[(row + col) * 2]).dot(
                                        B)),
                                rtol=rtol, atol=atol)

                            # Test values in H are accurate using that, roughly,
                            # Hp[r,c] = C * A^(r+c+1) * B.
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
        Test ROM Markov params similar to those given when the reduced system
        has the same number of states as the full system.

        - generates data
        - assembles Hankel array
        - computes SVD
        - forms the ROM discrete arrays A, B, and C (D = 0)
        - Tests Markov parameters from ROM are approx. equal to full plant's

        Also, unrelated:
        - Tests that saved ROM mats are equal to those returned in memory
        """
        # Set test tolerances (for infinity norm of transfer function
        # difference)
        tf_abs_tol = 1e-6
        tf_rel_tol = 1e-4

        # Set time parameters for discrete-time simulation
        dt = 0.1
        num_time_steps = 1000

        # Set size of plant and model. For test, don't reduce the system, just
        # check that it comes back close to the original plant.  Also, note that
        # using more than 8 states causes poorly conditioned TF coeffs
        # (https://github.com/scipy/scipy/issues/2980)
        num_states_plant = 8
        num_states_model = num_states_plant

        # Loop through different numbers of inputs, numbers of outputs, and
        # sampling intervals
        for num_inputs in [1, 3]:
            for num_outputs in [1, 2]:
                for sample_interval in [1, 2, 4]:
                    # Define time steps at which to save data.  These will be of
                    # the form [0, 1, p, p + 1, 2p, 2p + 1, ...] where p is the
                    # sample interval.
                    time_steps = make_time_steps(
                        num_time_steps, sample_interval)
                    # # Create a state space system
                    # A_plant, B_plant, C_plant = util.drss(
                    #     num_states_plant, num_inputs, num_outputs)
                    A_plant = util.load_array_text(
                        join(self.data_dir, 'A_in%d_out%d.txt') % (
                            num_inputs, num_outputs))
                    B_plant = util.load_array_text(
                        join(self.data_dir, 'B_in%d_out%d.txt') % (
                            num_inputs, num_outputs))
                    C_plant = util.load_array_text(
                        join(self.data_dir, 'C_in%d_out%d.txt') % (
                            num_inputs, num_outputs))

                    # Simulate an impulse response using the state space system.
                    # This will generate Markov parameters at all timesteps [0,
                    # 1, 2, 3, ...].  Only keep data at the desired time steps,
                    # which are separated by a sampling interval (see above
                    # comment).
                    Markovs = util.impulse(
                        A_plant, B_plant, C_plant,
                        time_steps[-1] + 1)[time_steps]

                    # Compute a model using ERA
                    my_ERA = era.ERA(verbosity=0)
                    A_model, B_model, C_model = my_ERA.compute_model(
                        Markovs, num_states_model)

                    # Save ERA model to disk
                    A_path_computed = join(self.out_dir, 'A_computed.txt')
                    B_path_computed = join(self.out_dir, 'B_computed.txt')
                    C_path_computed = join(self.out_dir, 'C_computed.txt')
                    my_ERA.put_model(
                        A_path_computed, B_path_computed, C_path_computed)

                    # Check normalized Markovs
                    rtol = 1e-5  # 1e-6
                    atol = 1e-5  # 1e-10
                    Markovs_model = util.impulse(
                        A_model, B_model, C_model,
                        time_steps[-1] + 1)[time_steps]
                    max_Markov = np.amax(Markovs)
                    eigs_plant = np.linalg.eig(A_plant)[0]
                    eigs_model = np.linalg.eig(A_model)[0]
                    # print 'markovs shape', Markovs.shape
                    # print 'max plant eig', np.abs(eigs_plant).max()
                    # print 'max model eig', np.abs(eigs_model).max()
                    # print 'max plant markov', max_Markov
                    # print 'max model markov', np.amax(Markovs_model)
                    # print 'markov diffs', (
                    #     Markovs - Markovs_model).squeeze().max()

                    '''
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.semilogy(np.abs(Markovs).squeeze(), 'b')
                    plt.semilogy(np.abs(Markovs_model).squeeze(), 'r--')
                    plt.axis(
                        [0, time_steps[-1], Markovs.min(), Markovs.max()])
                    '''

                    np.testing.assert_allclose(
                        Markovs_model.squeeze(),
                        Markovs.squeeze(),
                        rtol=rtol, atol=atol)


                    # plt.show()
                    '''
                    # Use Scipy to check that transfer function of ERA model is
                    # close to transfer function of full model.  Do so by
                    # computing the infinity norm (H_inf) of the difference
                    # between the transfer functions. Since Scipy can't handle
                    # MIMO transfer functions, loop through each input-output
                    # pair individually.
                    for input_idx in range(num_inputs):
                        for output_idx in range(num_outputs):

                            # Compute transfer functions
                            tf_plant = scipy.signal.StateSpace(
                                A_plant, B_plant[:, input_idx:input_idx + 1],
                                C_plant[output_idx:output_idx + 1, :],
                                0, dt=dt).to_tf()
                            tf_model = scipy.signal.StateSpace(
                                A_model,
                                B_model[:, input_idx:input_idx + 1],
                                C_model[output_idx:output_idx + 1, :],
                                0, dt=dt).to_tf()
                            tf_diff = util.sub_transfer_functions(
                                tf_plant, tf_model, dt=dt)

                            # Compute transfer function norms
                            tf_plant_inf_norm = util.compute_inf_norm_discrete(
                                tf_plant, dt)
                            tf_diff_inf_norm = util.compute_inf_norm_discrete(
                                tf_diff, dt)

                            # Test values
                            print 'err_frac', (
                                tf_diff_inf_norm / tf_plant_inf_norm)
                            self.assertTrue(
                                tf_diff_inf_norm / tf_plant_inf_norm <
                                tf_rel_tol)
                    '''

                    # Also test that saved reduced model mats are equal to those
                    # returned in memory
                    np.testing.assert_equal(
                        util.load_array_text(A_path_computed), A_model)
                    np.testing.assert_equal(
                        util.load_array_text(B_path_computed), B_model)
                    np.testing.assert_equal(
                        util.load_array_text(C_path_computed), C_model)



if __name__ == '__main__':
    unittest.main()
