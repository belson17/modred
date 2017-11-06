#!/usr/bin/env python
"""Test era module"""
import unittest
import os
from os.path import join
from shutil import rmtree

import numpy as np
import scipy.signal

from modred import era, parallel, util
from modred.py2to3 import range


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


# @unittest.skipIf(parallel.is_distributed(), 'Only test ERA in serial')
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


    #@unittest.skip("testing others")
    def test_assemble_Hankel(self):
        """ Tests Hankel arrays are symmetric given
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
                    impulse_response = util.impulse(A, B, C, time_steps[-1] + 2)
                    Markovs = impulse_response[time_steps]
                    print("time steps", time_steps)
                    print("Markovs", Markovs)
                    # ss = scipy.signal.StateSpace(A, B, C, 0, dt=1)
                    # dum, Markovs = scipy.signal.dimpulse(ss, t=time_steps)
                    # Markovs = np.array(Markovs).squeeze()

                    if sample_interval == 2:
                        time_steps, Markovs = era.make_sampled_format(
                            time_steps, Markovs)

                    my_ERA = era.ERA(verbosity=0)
                    my_ERA._set_Markovs(Markovs)
                    my_ERA._assemble_Hankel()
                    H = my_ERA.Hankel_array
                    Hp = my_ERA.Hankel_array2
                    print("Hankel", H)
                    print("Hankel2", Hp)

                    for row in range(my_ERA.mc):
                        for col in range(my_ERA.mo):
                            np.testing.assert_equal(
                                H[row * num_outputs:(row + 1) * num_outputs,
                                  col * num_inputs:(col + 1) * num_inputs],
                                H[col * num_outputs:(col + 1) * num_outputs,
                                  row * num_inputs:(row + 1) * num_inputs])
                            np.testing.assert_equal(
                                Hp[row * num_outputs:(row + 1) * num_outputs,
                                   col * num_inputs:(col + 1) * num_inputs],
                                Hp[col * num_outputs:(col + 1) * num_outputs,
                                   row * num_inputs:(row + 1) * num_inputs])
                            np.testing.assert_allclose(
                                H[row * num_outputs:(row + 1) * num_outputs,
                                  col * num_inputs:(col + 1) * num_inputs],
                                C.dot(
                                    np.linalg.matrix_power(
                                        A, time_steps[(row + col) * 2]).dot(
                                            B)),
                                rtol=rtol, atol=atol)
                            np.testing.assert_allclose(
                                Hp[row * num_outputs:(row + 1) * num_outputs,
                                   col * num_inputs:(col + 1) * num_inputs],
                                C.dot(
                                    np.linalg.matrix_power(
                                        A, time_steps[(row + col) * 2 + 1]).dot(
                                            B)),
                                rtol=rtol, atol=atol)


    @unittest.skip('testing others')
    def test_compute_model(self):
        """
        Test ROM Markov params similar to those given

        - generates data
        - assembles Hankel array
        - computes SVD
        - forms the ROM discrete arrays A, B, and C (D = 0)
        - Tests Markov parameters from ROM are approx. equal to full plant's
        """
        num_time_steps = 40
        num_states_plant = 12
        num_states_model = num_states_plant // 3
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
                    # ss = scipy.signal.StateSpace(A, B, C, 0, dt=1)
                    # dum, Markovs = scipy.signal.dimpulse(ss, t=time_steps)
                    # Markovs = np.array(Markovs).squeeze()

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
                    #sing_vals = my_ERA.sing_vals[:num_states_model]

                    # Flatten vecs into 2D X and Y arrays:
                    # [B AB A**PB A**(P+1)B ...]
                    #direct_vecs_flat = direct_vecs.swapaxes(0,1).reshape(
                    #    (num_states_model,-1)))

                    # Exact grammians from Lyapunov eqn solve
                    #gram_cont = util.solve_Lyapunov(A, B*B.H)
                    #gram_obs = util.solve_Lyapunov(A.H, C.H*C)
                    #print(np.sort(np.linalg.eig(gram_cont)[0])[::-1])
                    #print(sing_vals)
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

                    # Markovs_model = np.zeros(Markovs.shape)
                    # for ti, tv in enumerate(time_steps):
                    #     Markovs_model[ti] = C.dot(
                    #         np.linalg.matrix_power(A, tv).dot(
                    #             B))
                        #print 'computing ROM Markov param at time step %d'%tv
                    Markovs_model = util.impulse(A_reduced, B_reduced, C_reduced, time_steps[-1] + 1)
                    Markovs_model = Markovs_model[time_steps]
                    # ss_reduced = scipy.signal.StateSpace(
                    #     A_reduced, B_reduced, C_reduced, 0, dt=1)
                    # dum, Markovs_model = scipy.signal.dimpulse(ss_reduced, t=time_steps)
                    # Markovs_model = np.array(Markovs_model).squeeze()
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
                    # np.testing.assert_allclose(Markovs_model, Markovs,
                    #     rtol=0.5, atol=0.5)
                    # Check normalized Markovs
                    max_Markov = np.amax(Markovs)
                    np.testing.assert_allclose(Markovs_model / max_Markov, Markovs / max_Markov,
                                               rtol=1e-2, atol=1e-2)

                    np.testing.assert_equal(
                        util.load_array_text(A_path_computed), A_reduced)
                    np.testing.assert_equal(
                        util.load_array_text(B_path_computed), B_reduced)
                    np.testing.assert_equal(
                        util.load_array_text(C_path_computed), C_reduced)


    @unittest.skip('testing others')
    def test_error_bounds(self):
        """
        cont SS -> cont TF and disc SS.
        :return:
        """
        num_time_steps = 40
        num_states_plant = 12
        num_states_model = num_states_plant // 3
        for num_inputs in [1]:#, 3]:
            for num_outputs in [1]:#, 2]:
                for sample_interval in [1]:#, 2, 4]:
                    time_steps = make_time_steps(
                        num_time_steps, sample_interval)
                    A, B, C = util.drss(
                        num_states_plant, num_inputs, num_outputs)
                    my_ERA = era.ERA(verbosity=0)
                    Markovs = util.impulse(A, B, C, time_steps[-1] + 1)
                    Markovs = Markovs[time_steps]
                    # ss = scipy.signal.StateSpace(A, B, C, 0, dt=1)
                    # dum, Markovs = scipy.signal.dimpulse(ss, t=time_steps)
                    # Markovs = np.array(Markovs).squeeze()

                    if sample_interval == 2:
                        time_steps, Markovs =\
                            era.make_sampled_format(time_steps, Markovs)
                    num_time_steps = time_steps.shape[0]
                    # A_path_computed = join(self.test_dir, 'A_computed.txt')
                    # B_path_computed = join(self.test_dir, 'B_computed.txt')
                    # C_path_computed = join(self.test_dir, 'C_computed.txt')
                    Ar, Br, Cr = my_ERA.compute_model(Markovs, num_states_model)
                    tf_full = scipy.signal.ss2tf(
                        scipy.signal.StateSpace(
                            A, B, C, np.zeros((C.shape[0], B.shape[1])), dt=1))
                    tf_red = scipy.signal.ss2tf(
                        scipy.signal.StateSpace(
                            Ar, Br, Cr, np.zeros((Cr.shape[0], Br.shape[1])), dt=1))
                    tf_diff = util.sub_transfer_functions(tf_full, tf_red, dt=1)
                    ss_diff = scipy.signal.tf2ss(tf_diff)
                    print("SS diff", ss_diff)
                    inf_norm_error = self.compute_inf_norm_system(ss_diff)
                    balanced_trunc_max_error = 2 * self.sing_vals[num_states_model+1:].sum()
                    print("err", inf_norm_error, "max err bal tr", balanced_trunc_max_error)



    def compute_inf_norm_system(self, A, B, C):
        min_inf_norm = 1e-10
        max_inf_norm = 1e10
        tol = 1e-10
        i = 0
        while (max_inf_norm - min_inf_norm > tol):
            mid_inf_norm = (min_inf_norm + max_inf_norm) / 2
            valid = self.is_valid_inf_norm(mid_inf_norm, A, B, C)
            if valid:
                min_inf_norm = mid_inf_norm
            else:
                max_inf_norm = mid_inf_norm
            print(i, valid, min_inf_norm, max_inf_norm, mid_inf_norm)
            i += 1
        return mid_inf_norm



    def is_valid_inf_norm(self, inf_norm, A, B, C):
        sys_mat = np.concatenate((np.concatenate((A, B.dot(B.H) / inf_norm ** 2), axis=1),
                                  np.concatenate((-C.H.dot(C), -A.H), axis=1)), axis=0)
        eig_vals = np.linalg.eigvals(np.mat(sys_mat))
        print(eig_vals)
        is_imag_eig_val = ((np.abs(eig_vals.imag) > 1e-12) & (np.abs(eig_vals.real) < 1e-12)).any()
        return is_imag_eig_val



if __name__ == '__main__':
    unittest.main()
