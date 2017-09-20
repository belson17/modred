"""Functions and classes for ERA models. See paper by Ma et al. 2011, TCFD."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future.builtins import range
from future.builtins import object

import numpy as np

from . import util


def make_sampled_format(times, Markovs, dt_tol=1e-6):
    """Converts samples at [0 1 2 ...] into samples at [0 1 1 2 2 3 ...].

    Args:
        ``times``: Array of time values or time step indices.

        ``Markovs``: Array of Markov parameters with indices [time, output,
        input]. ``Markovs[i]`` is the Markov parameter :math:`C A^i B`.

    Kwargs:
        ``dt_tol``: Allowable deviation from uniform time steps.

    Returns:
        ``time_steps``: Array of time step indices, [0 1 1 2 2 3 ...].

        ``Markovs``: Output array at the time step indices.

        ``dt``: Time interval between each time step.

    Takes a series of data at times :math:`dt*[0 1 2 3 ...]` and duplicates
    entries so that the result is sampled at :math:`dt*[0 1 1 2 2 3 ...]`.
    When the second format is used in the eigensystem realization algorithm
    (ERA), the resulting model has a time step of :math:`dt` rather than
    :math:`2*dt`.
    """
    num_time_steps, num_Markovs, num_inputs = Markovs.shape
    if num_time_steps != times.shape[0]:
        raise RuntimeError('Size of time and output arrays differ')

    dt = times[1] - times[0]
    if np.amax(np.abs(np.diff(times)-dt)) > dt_tol:
        raise ValueError('Data is not equally spaced in time')

    time_steps = np.round((times-times[0])/dt)
    num_time_steps_corr = (num_time_steps - 1)*2
    Markovs_corr = np.zeros((num_time_steps_corr, num_Markovs, num_inputs))
    time_steps_corr = np.zeros(num_time_steps_corr, dtype=int)
    Markovs_corr[::2] = Markovs[:-1]
    Markovs_corr[1::2] = Markovs[1:]
    time_steps_corr[::2] = time_steps[:-1]
    time_steps_corr[1::2] = time_steps[1:]
    return time_steps_corr, Markovs_corr

def compute_ERA_model(Markovs, num_states):
    """Convenience function to compute linear time-invariant (LTI)
    reduced-order model (ROM) matrices A, B, and C using the eigensystem
    realization algorithm (ERA) with default settings.

    Args:
        ``Markovs``: Array of Markov parameters with indices [time, output,
        input]. ``Markovs[i]`` is the Markov parameter :math:`C A^i B`.

        ``num_states``: Number of states in reduced-order model.

    Returns:
        ``A``: A matrix of reduced-order model.

        ``B``: B matrix of reduced-order model.

        ``C``: C matrix of reduced-order model.

    Usage::

      # Obtain ``Markovs`` array w/indicies [time, output, input]
      num_states = 20
      A, B, C = compute_ERA_model(Markovs, num_states)

    Notes:

    - Markov parameters are defined as
      :math:`[CB, CAB, CA^PB, CA^(P+1)B, ...]`

    - The functions :py:func:`util.load_signals` and
      :py:func:`util.load_multiple_signals` are often useful.
    """
    my_ERA = ERA()
    return my_ERA.compute_model(Markovs, num_states)


class ERA(object):
    """Eigensystem realization algorithm (ERA), implemented for discrete-time
    systems.

    Kwargs:
        ``put_mat``: Function to put a matrix out of modred, e.g., write it to
        file.

        ``mc``: Number of Markov parameters for controllable dimension of
        Hankel matrix.

        ``mo``: Number of Markov parameters for observable dimension of Hankel
        matrix.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.

    Computes reduced-order model (ROM) of a discrete-time system, using output
    data from an impulse response.

    Simple usage::

      # Obtain array "Markovs" with dims [time, output, input]
      myERA = ERA()
      A, B, C = myERA.compute_model(Markovs, 50)
      sing_vals = myERA.sing_vals

    Another example::

      # Obtain Markov parameters
      myERA = ERA()
      myERA.compute_model(Markovs, 50)
      myERA.put_model('A.txt', 'B.txt', 'C.txt')

    Notes:

    - Default values of ``mc`` and ``mo`` are equal and maximal for a balanced
      model.

    - The Markov parameters are to be given in the time-sampled format:
      :math:`dt*[0, 1, P, P+1, 2P, 2P+1, ... ]`

    - The special case where :math:`P=2` results in
      :math:`dt*[0, 1, 2, 3, ... ]`, see :py:func:`make_sampled_format`.

    - The functions :py:func:`util.load_signals` and
      :py:func:`util.load_multiple_signals` are often useful.

    - See convenience function :py:func:`compute_ERA_model`.
    """
    def __init__(self, put_mat=util.save_array_text, mc=None, mo=None,
        verbosity=1):
        """Constructor"""
        self.put_mat = put_mat
        self.mc = mc
        self.mo = mo
        self.outputs = None
        self.verbosity = verbosity
        self.A = None
        self.B = None
        self.C = None
        self.sing_vals = None
        self.L_sing_vecs = None
        self.R_sing_vecs = None
        self.num_outputs = None
        self.num_inputs = None
        self.num_time_steps = None
        self.Hankel_mat = None
        self.Hankel_mat2 = None
        self.num_Markovs = None
        self.Markovs = None

    def compute_model(self, Markovs, num_states, mc=None, mo=None):
        """Computes the A, B, and C matrices of the linear time-invariant (LTI)
        reduced-order model (ROM).

        Args:
            ``Markovs``: Array of Markov parameters with indices [time, output,
            input]. ``Markovs[i]`` is the Markov parameter :math:`C A^i B`.

            ``num_states``: Number of states in reduced-order model.

        Kwargs:
            ``mc``: Number of Markov parameters for controllable dimension of
            Hankel matrix.

            ``mo``: Number of Markov parameters for observable dimension of
            Hankel matrix.

        Assembles the Hankel matrices from self.Markovs and computes a singular
        value decomposition. Uses the results to form the A, B, and C matrices.

        Note that the default values of ``mc`` and ``mo`` are equal and maximal
        for a balanced model.

        Tip: For discrete time systems the impulse is applied over a time
        interval :math:`dt` and so has a time-integral :math:`1*dt` rather than
        :math:`1`.  This means the reduced B matrix is "off" by a factor of
        :math:`dt`.  You can account for this by multiplying B by :math:`dt`.
        """
        #SVD is ``L_sing_vecs*np.mat(np.diag(sing_vals))*\
        #    R_sing_vecs.H = Hankel_mat``
        self._set_Markovs(Markovs)
        self.mc = mc
        self.mo = mo

        self._assemble_Hankel()
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs = \
            util.svd(self.Hankel_mat)

        # Truncate matrices
        Ur = np.mat(self.L_sing_vecs[:, :num_states])
        Er = np.squeeze(self.sing_vals[:num_states])
        Vr = np.mat(self.R_sing_vecs[:, :num_states])

        self.A = np.mat(np.diag(Er**-.5)) * Ur.H * self.Hankel_mat2 * Vr * \
            np.mat(np.diag(Er**-.5))
        self.B = (np.mat(np.diag(Er**.5)) * (Vr.H)[:, :self.num_inputs])
        # *dt above is removed, users must do this themselves.
        # It is explained in the docs.

        self.C = Ur[:self.num_Markovs] * np.mat(np.diag(Er**.5))

        if (np.abs(np.linalg.eigvals(self.A)) >= 1.).any() and self.verbosity:
            print('Warning: Unstable eigenvalues of reduced A matrix')
            print('eig vals are', np.linalg.eigvals(self.A))
        return self.A, self.B, self.C

    def put_model(self, A_dest, B_dest, C_dest):
        """Puts the A, B, and C matrices of the linear time-invariant (LTI)
        reduced-order model (ROM) in destinations (file or memory).

        Args:
            ``A_dest``: Destination in which to put A matrix of reduced-order
            model.

            ``B_dest``: Destination in which to put B matrix of reduced-order
            model.

            ``C_dest``: Destination in which to put C matrix of reduced-order
            model.
        """
        self.put_mat(self.A, A_dest)
        self.put_mat(self.B, B_dest)
        self.put_mat(self.C, C_dest)
        if self.verbosity:
            print('Put ROM matrices to:')
            print(A_dest)
            print(B_dest)
            print(C_dest)

    def put_decomp(
        self, sing_vals_dest, L_sing_vecs_dest, R_sing_vecs_dest,
        Hankel_mat_dest, Hankel_mat2_dest):
        """Puts the decomposition matrices and Hankel matrices in destinations
        (file or memory).

        Args:
            ``sing_vals_dest``: Destination in which to put Hankel singular
            values.

            ``L_sing_vecs_dest``: Destination in which to put left singular
            vectors of Hankel matrix.

            ``R_sing_vecs_dest``: Destination in which to put right singular
            vectors of Hankel matrix.

            ``Hankel_mat_dest``: Destination in which to put Hankel matrix.

            ``Hankel_mat2_dest``: Destination in which to put second Hankel
            matrix.
        """
        self.put_mat(self.Hankel_mat, Hankel_mat_dest)
        self.put_mat(self.Hankel_mat2, Hankel_mat2_dest)
        self.put_mat(self.L_sing_vecs, L_sing_vecs_dest)
        self.put_mat(self.sing_vals, sing_vals_dest)
        self.put_mat(self.R_sing_vecs, R_sing_vecs_dest)

    def put_sing_vals(self, sing_vals_dest):
        """Puts the singular values to ``sing_vals_dest``."""
        self.put_mat(self.sing_vals, sing_vals_dest)

    def _set_Markovs(self, Markovs):
        """Sets the Markov params to ``self.Markovs`` and error checks.

        Args:
            ``Markovs``: Array of Markov params w/indices [time, output, input].

        ``Markovs[i]`` is the Markov parameter C*A**i*B.
        """
        self.Markovs = Markovs
        ndims = self.Markovs.ndim
        if ndims == 1:
            self.Markovs = self.Markovs.reshape((self.Markovs.shape[0], 1, 1))
        elif ndims == 2:
            self.Markovs = self.Markovs.reshape((self.Markovs.shape[0],
                self.Markovs.shape[1], 1))
        elif ndims > 3:
            raise RuntimeError('Markovs can have 1, 2, or 3 dims, '
                'yours had %d'%ndims)

        self.num_time_steps, self.num_Markovs, self.num_inputs = \
            self.Markovs.shape

        # Must have an even number of time steps, remove last entry if odd.
        if self.num_time_steps % 2 != 0:
            self.num_time_steps -= 1
            self.Markovs = self.Markovs[:-1]

    def _assemble_Hankel(self):
        """Assembles and sets ``self.Hankel_mat`` and ``self.Hankel_mat2``

        See variables H and H' in Ma 2011.
        """
        # To understand the default choice of mc and mo,
        # consider (let sample_interval=P=1 w.l.o.g.)
        # sequences
        # [0 1 1 2 2 3] => H = [[0 1][1 2]]   H2 = [[1 2][2 3]], mc=mo=1, nt=6
        # [0 1 1 2 2 3 3 4 4 5] => H=[[0 1 2][1 2 3][2 3 4]]
        # and H2=[[1 2 3][2 3 4][3 4 5]], mc=mo=2,nt=10
        # Thus, using all available data means mc=mo=(nt-2)/4.
        # For additional output times, (nt-2)/4 rounds down, so
        # we use this formula.

        if self.mo is None or self.mc is None:
            # Set mo and mc, time_steps is always in format [0, 1, P, P+1, ...]
            self.mo = (self.num_time_steps-2)//4
            self.mc = self.mo

        if (self.mo + self.mc) +2 > self.num_time_steps:
            raise ValueError('mo+mc+2=%d and must be <= than the number of '
                'samples %d'%(self.mo+self.mc+2, self.num_time_steps))

        self.Hankel_mat = np.zeros((self.num_Markovs*self.mo,
            self.num_inputs*self.mc))
        self.Hankel_mat2 = np.zeros(self.Hankel_mat.shape)
        #Markovs_flattened = \
        #    self.Markovs.swapaxes(0,1).reshape((num_Markovs, -1))
        for row in range(self.mo):
            row_start = row*self.num_Markovs
            row_end = row_start + self.num_Markovs
            for col in range(self.mc):
                col_start = col*self.num_inputs
                col_end = col_start + self.num_inputs
                self.Hankel_mat[row_start:row_end, col_start:col_end] = \
                    self.Markovs[2*(row+col)]
                self.Hankel_mat2[row_start:row_end, col_start:col_end] = \
                    self.Markovs[2*(row+col)+1]
