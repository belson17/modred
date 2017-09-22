"""A group of useful functions"""
from __future__ import print_function
from future.builtins import range
from future.builtins import object
import inspect
import os

import numpy as np


class UndefinedError(Exception): pass


'''
def make_mat(array):
    """Makes 1D or 2D arrays into matrices. 1D arrays become matrices with one
    column."""
    if array.ndim == 1:
        array = array.reshape((array.shape[0], 1))
    return np.mat(array)


def make_list(arg):
    #Returns the argument as a list. If already a list, ``arg`` is returned.
    #
    if not isinstance(arg, list):
        arg = [arg]
    return arg
'''


def atleast_2d_col(array):
    """Converts 1d arrays to 2d arrays, but always as column vectors"""
    if len(array.shape) == 1:
        return np.atleast_2d(array).T
    else:
        return array


def make_iterable(arg):
    """Checks if ``arg`` is iterable. If not, makes it a one-element list.
    Otherwise returns ``arg``."""
    try:
        iterator = iter(arg)
        return arg
    except TypeError:
        return [arg]


def flatten_list(my_list):
    """Flatten a list of lists into a single list."""
    return [num for elem in my_list for num in elem]


def save_array_text(array, file_name, delimiter=None):
    """Saves a 1D or 2D array to a text file.

    Args:
        ``array``: 1D or 2D or array to save to file.

        ``file_name``: Filepath to location where data is to be saved.

    Kwargs:
        ``delimiter``: Delimiter in file. Default is same as ``numpy.savetxt``.

    Format of saved files is::

      2.3 3.1 2.1 ...
      5.1 2.2 9.8 ...
      7.6 3.1 5.5 ...
      0.1 1.9 9.1 ...
      ...

    Complex data is saved in the following format (as floats)::

      real[0,0] imag[0,0] real[0,1] imag[0,1] ...
      real[1,0] imag[1,0] real[1,1] imag[1,1] ...
      ...

    Files can be read in Matlab with the provided functions or often
    with Matlab's ``load``.
    """
    # If one-dimensional array, then make a vector of many rows, 1 column
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    elif array.ndim > 2:
        raise RuntimeError('Cannot save an array with >2 dimensions')
    if delimiter is None:
        np.savetxt(file_name, array.view(float))
    else:
        np.savetxt(file_name, array.view(float), delimiter=delimiter)


def load_array_text(file_name, delimiter=None, is_complex=False):
    """Reads data saved in a text file, returns an array.

    Args:
        ``file_name``: Name of file from which to load data.

    Kwargs:
        ``delimiter``: Delimiter in file. Default is same as ``numpy.loadtxt``.

        ``is_complex``: Boolean describing whether the data to be loaded is
        complex valued.

    Returns:
        ``array``: 2D array containing loaded data.

    See :py:func:`save_array_text` for the format used by this function.
    """
    if is_complex:
        dtype = complex
    else:
        dtype = float
    array = np.loadtxt(file_name, delimiter=delimiter) #, ndmin=2)
    ## This section reproduces behavior of ndmin=2 option of np.loadtxt
    if array.ndim == 1:
        with open(file_name) as f:
            num_rows = sum(1 for line in f)
        if num_rows > 1:
            array = array.reshape((-1, 1))
        else:
            array = array.reshape((1, -1))
    ## End work around for ndmin=2 option.

    if is_complex and array.shape[1] % 2 != 0:
        raise ValueError(
            ('Cannot load complex data, file %s has an odd number of columns. '
            'Maybe it has real data.') % file_name)

    # Cast as an array, copies to make it C-contiguous memory
    return np.array(array.view(dtype))


def get_file_list(directory, file_extension=None):
    """Returns list of files in ``directory`` with ``file_extension``."""
    files = os.listdir(directory)
    if file_extension is not None:
        if len(file_extension) == 0:
            print('Warning: gave an empty file extension')
        filtered_files = []
        for f in files:
            if f[-len(file_extension):] == file_extension:
                filtered_files.append(f)
        return filtered_files
    else:
        return files


def get_data_members(obj):
    """Returns a dictionary containing data members of ``obj``."""
    data_members = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            data_members[name] = value
    return data_members


def sum_arrays(arr1, arr2):
    """Used for ``allreduce`` command."""
    return arr1 + arr2


def sum_lists(list1, list2):
    """Sums the elements of each list, returns a new list.

    This function is used in MPI reduce commands, but could be used
    elsewhere too."""
    assert len(list1) == len(list2)
    return [list1[i] + list2[i] for i in range(len(list1))]


def smart_eq(arg1, arg2):
    """Checks for equality, accounting for the fact that numpy's ``==`` doesn't
    return a bool. In that case, returns True only if all elements are equal."""
    eq = (arg1 == arg2)
    if isinstance(eq, np.ndarray):
        return eq.all()
    return eq


class InnerProductBlock(object):
    """Only used in tests. Takes inner product of all vectors."""
    def __init__(self, inner_product):
        self.inner_product = inner_product


    def __call__(self, vecs1, vecs2):
        n1 = len(vecs1)
        n2 = len(vecs2)
        IP_array = np.zeros((n1,n2),
            dtype=type(self.inner_product(vecs1[0], vecs2[0])))
        for i in range(n1):
            for j in range(n2):
                IP_array[i, j] = self.inner_product(vecs1[i], vecs2[j])
        return IP_array


def svd(array, atol=1e-13, rtol=None):
    """Wrapper for ``numpy.linalg.svd``, computes the singular value
    decomposition of an array.

    Args:
        ``array``: Array to take singular value decomposition of.

    Kwargs:
        ``atol``: Level below which singular values are truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        singular values.  Smaller ones are truncated.

    Returns:
        ``U``: Array whose columns are left singular vectors.

        ``S``: 1D array of singular values.

        ``V``: Array whose columns are right singular vectors.

    Truncates ``U``, ``S``, and ``V`` such that the singular values
    obey both ``atol`` and ``rtol``.
    """
    U, S, V_comp_conj = np.linalg.svd(array, full_matrices=0)
    V = V_comp_conj.conj().T

    # Figure out how many singular values satisfy the tolerances
    if atol is not None:
        num_nonzeros_atol = (abs(S) > atol).sum()
    else:
        num_nonzeros_atol = S.size
    if rtol is not None:
        num_nonzeros_rtol = (
            abs(S[:num_nonzeros_atol]) / abs(S[0]) > rtol).sum()
        num_nonzeros = min(num_nonzeros_atol, num_nonzeros_rtol)
    else:
        num_nonzeros = num_nonzeros_atol

    # Truncate arrays according to tolerances
    U = U[:, :num_nonzeros]
    V = V[:, :num_nonzeros]
    S = S[:num_nonzeros]

    return U, S, V


def eigh(array, atol=1e-13, rtol=None, is_positive_definite=False):
    """Wrapper for ``numpy.linalg.eigh``. Computes eigendecomposition of a
    Hermitian array.

    Args:
        ``array``: Array to take eigendecomposition of.

        ``atol``: Value below which eigenvalues (and corresponding
        eigenvectors) are truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        eigenvalues.  Smaller ones are truncated.

        ``is_positive_definite``: If true, array being decomposed will be
        assumed to be positive definite.  Tolerance will be automatically
        adjusted (if necessary) so that only positive eigenvalues are returned.

    Returns:
        ``eigvals``: 1D array of eigenvalues, sorted in descending order (of
        magnitude).

        ``eigvecs``: Array whose columns are eigenvectors.
    """
    # Compute eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(array)

    # Sort the vecs and eigvals by eigval magnitude.  The first element will
    # have the largest magnitude and the last element will have the smallest
    # magnitude.
    sort_indices = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[:, sort_indices]

    # Adjust absolute tolerance for positive definite case if there are negative
    # eigenvalues and the most negative one has magnitude greater than the
    # given tolerance.  In that case, we assume the given tolerance is too
    # samll (relative to the accuracy of the computation) and increase it to at
    # least filter out negative eigenvalues.
    if is_positive_definite and eigvals.min() < 0 and abs(eigvals.min()) > atol:
        atol = abs(eigvals.min())

    # Filter out small and negative eigenvalues, if necessary
    if atol is not None:
        num_nonzeros_atol = (abs(eigvals) > atol).sum()
    else:
        num_nonzeros_atol = eigvals.size
    if rtol is not None:
        num_nonzeros_rtol = (
            abs(eigvals[:num_nonzeros_atol]) / abs(eigvals[0]) > rtol).sum()
        num_nonzeros = min(num_nonzeros_atol, num_nonzeros_rtol)
    else:
        num_nonzeros = num_nonzeros_atol
    eigvals = eigvals[:num_nonzeros]
    eigvecs = eigvecs[:, :num_nonzeros]
    return eigvals, eigvecs


def eig_biorthog(array, scale_choice='left'):
    """Wrapper for ``numpy.linalg.eig`` that returns both left and right
    eigenvectors. Eigenvalues and eigenvectors are sorted and scaled so that
    the left and right eigenvector arrays are orthonormal.

    Args:
        ``array``: Array to take eigendecomposition of.

    Kwargs:
        ``scale_choice``: Determines whether 'left' (default) or 'right'
        eigenvectors will be scaled to yield a biorthonormal set.  The other
        eigenvectors will be left unscaled, leaving them with unit norms.

    Returns:
        ``evals``: 1D array of eigenvalues.

        ``R_evecs``: Array whose columns are right eigenvectors.

        ``L_evecs``: Array whose columns are left eigenvectors.
    """
    # Compute eigendecompositions
    R_evals, R_evecs= np.linalg.eig(array)
    L_evals_conj, L_evecs= np.linalg.eig(array.conj().T)
    L_evals = L_evals_conj.conj()

    # Sort the evals
    R_sort_indices = np.argsort(R_evals)
    L_sort_indices = np.argsort(L_evals)
    R_evals = R_evals[R_sort_indices]
    L_evals = L_evals[L_sort_indices]
    L_evals_conj = L_evals.conj()

    # Check that evals are the same
    if not np.allclose(L_evals, R_evals, rtol=1e-10, atol=1e-13):
        raise ValueError('Left and right eigenvalues do not match.')

    # Sort the evecs
    R_evecs = R_evecs[:, R_sort_indices]
    L_evecs = L_evecs[:, L_sort_indices]

    # Scale the evecs to get a biorthogonal set
    scale_factors = np.diag(np.dot(L_evecs.conj().T, R_evecs))
    if scale_choice.lower() == 'left':
        L_evecs /= scale_factors.conj()
    elif scale_choice.lower() == 'right':
        R_evecs /= scale_factors
    else:
        raise ValueError('Invalid scale choice.  Must be LEFT or RIGHT.')

    return R_evals, R_evecs, L_evecs


def solve_Lyapunov_direct(A, Q):
    """Solves discrete Lyapunov equation :math:`AXA' - X + Q = 0` for
    :math:`X`, given :math:`A` and :math:`Q`.

    This function may not be as computationally efficient or stable as
    Matlab's ``dylap``.

    See also :py:func:`solve_Lyapunov_iterative` and
    http://en.wikipedia.org/wiki/Lyapunov_equation
    """
    if A.shape != Q.shape:
        raise ValueError("A and Q don't have same shape")
    #A_flat = A.flatten()
    Q_flat = Q.flatten()
    kron_AA = np.kron(A, A)
    X_flat = np.linalg.solve(np.identity(kron_AA.shape[0]) - kron_AA, Q_flat)
    X = X_flat.reshape((A.shape))
    return X


def solve_Lyapunov_iterative(A, Q, max_iters=10000, tol=1e-8):
    """Solves discrete Lyapunov equation :math:`AXA' - X + Q = 0` for
    :math:`X`, given :math:`A` and :math:`Q`.

    This method is based on the iterative discrete-time Lyapunov solver
    described in Davinson and Man, "The Numerical Solution of A'Q+QA=-C,"
    IEEE Transactions on Automatic Control, volume 13, issue 4, August 1968,
    p. 448.

    This function may not be as computationally efficient or stable as
    Matlab's ``dylap``.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be square.')
    if A.shape != Q.shape:
        raise ValueError('A and Q must have the same shape.')

    if np.amax(np.abs(np.linalg.eig(A)[0])) > 1.:
        raise ValueError(
            'A must have stable eigenvalues (in the unit circle).')

    X = np.copy(Q)
    AP = np.copy(A)
    AT = np.copy(A.transpose())
    APT = np.copy(A.transpose())
    error = np.inf
    iter = 0
    while error > tol and iter < max_iters:
        change = AP.dot(Q).dot(APT)
        X += change
        AP = AP.dot(A)
        APT = APT.dot(AT)
        error = np.abs(change).max()
        iter += 1

    if iter >= max_iters:
        print('Warning: did not converge to solution. Error is %f.'%error)
    return X


def balanced_truncation(
    A, B, C, order=None, return_sing_vals=False, iterative_solver=True):
    """Balance and truncate discrete-time linear time-invariant (LTI) system
    defined by A, B, C arrays.

    Args:
        ``A``, ``B``, ``C``: LTI discrete-time arrays.

    Kwargs:
        ``order``: Order (number of states) of truncated system. Default is to
        use the maximal possible value (can truncate system afterwards).

    Returns:
        ``A_balanced``, ``B_balanced``, ``C_balanced``: LTI discrete-time
        arrays of balanced system.

        If ``return_sing_vals`` is True, also returns:

        ``sing_vals``: Hankel singular values.

    Notes:

    - ``D`` is unchanged by balanced truncation.
    - This function may not be as computationally efficient or stable as
      Matlab's ``balancmr``.
    """
    if iterative_solver:
        gram_cont = solve_Lyapunov_iterative(A, B.dot(B.transpose().conj()))
        gram_obsv = solve_Lyapunov_iterative(A.transpose().conj(),
            C.transpose().conj().dot(C))
    else:
        gram_cont = solve_Lyapunov_direct(A, B.dot(B.transpose().conj()))
        gram_obsv = solve_Lyapunov_direct(A.transpose().conj(),
            C.transpose().conj().dot(C))
    Uc, Ec, Vc = svd(gram_cont)
    Uo, Eo, Vo = svd(gram_obsv)
    Lc = Uc.dot(np.diag(Ec**0.5))
    Lo = Uo.dot(np.diag(Eo**0.5))
    U, E, V = svd(Lo.transpose().dot(Lc))
    if order is None:
        order = len(E)
    SL = Lo.dot(U[:,:order]).dot(np.diag(E**-0.5))
    SR = Lc.dot(V[:,:order]).dot(np.diag(E**-0.5))
    A_bal_trunc = SL.transpose().dot(A).dot(SR)
    B_bal_trunc = SL.transpose().dot(B)
    C_bal_trunc = C.dot(SR)
    if return_sing_vals:
        return A_bal_trunc, B_bal_trunc, C_bal_trunc, E
    else:
        return A_bal_trunc, B_bal_trunc, C_bal_trunc


def drss(num_states, num_inputs, num_outputs):
    """Generates a discrete-time random state-space system.

    Args:
        ``num_states``: Number of states.

        ``num_inputs``: Number of inputs.

        ``num_outputs``: Number of outputs.

    Returns:
        ``A``, ``B``, ``C``: State-space arrays of discrete-time system.

    By construction, all eigenvalues are real and stable.
    """
    eig_vals = np.linspace(.9, .95, num_states)
    eig_vecs = np.random.normal(0, 2., (num_states, num_states))
    A = np.real(
        np.linalg.inv(eig_vecs).dot(np.diag(eig_vals).dot(eig_vecs)))
    B = np.random.normal(0, 1., (num_states, num_inputs))
    C = np.random.normal(0, 1., (num_outputs, num_states))
    return A, B, C


def rss(num_states, num_inputs, num_outputs):
    """Generates a continuous-time random state-space system.

    Args:
        ``num_states``: Number of states.

        ``num_inputs``: Number of inputs.

        ``num_outputs``: Number of outputs.

    Returns:
        ``A``, ``B``, ``C``: State-space arrays of continuous-time system.

    By construction, all eigenvalues are real and stable.
    """
    e_vals = -np.random.random(num_states)
    transformation = np.random.random((num_states, num_states))
    A = np.linalg.inv(transformation).dot(np.diag(e_vals)).dot(transformation)
    B = np.random.random((num_states, num_inputs))
    C = np.random.random((num_outputs, num_states))
    return A, B, C


def lsim(A, B, C, inputs, initial_condition=None):
    """Simulates a discrete-time system with arbitrary inputs.

    :math:`x(n+1) = Ax(n) + Bu(n)`

    :math:`y(n) = Cx(n)`

    Args:
        ``A``, ``B``, and ``C``: State-space system arrays.

        ``inputs``: Array of inputs :math:`u`, with dimensions
        ``[num_time_steps, num_inputs]``.

    Kwargs:
        ``initial_condition``: Initial condition :math:`x(0)`.

    Returns:
        ``outputs``: Array of outputs :math:`y`, with dimensions
        ``[num_time_steps, num_outputs]``.

    ``D`` array is assumed to be zero.
    """
    #D = 0
    if inputs.ndim == 1:
        inputs = inputs.reshape((len(inputs), 1))
    num_steps, num_inputs = inputs.shape
    num_outputs = C.shape[0]
    num_states = A.shape[0]
    if A.shape != (num_states, num_states):
        raise ValueError('A has the wrong shape ', A.shape)
    if B.shape != (num_states, num_inputs):
        raise ValueError('B has the wrong shape ', B.shape)
    if C.shape != (num_outputs, num_states):
        raise ValueError('C has the wrong shape ', C.shape)
    #if D == 0:
    #    D = np.zeros((num_outputs, num_inputs))
    #if D.shape != (num_outputs, num_inputs):
    #    raise ValueError('D has the wrong shape, D=', D)
    if initial_condition is not None:
        if initial_condition.shape[0] != num_states or \
            initial_condition.ndim != 1:
            raise ValueError('initial_condition has the wrong shape')
    else:
        initial_condition = np.zeros(num_states)
    state = initial_condition
    outputs = np.zeros((num_steps, num_outputs))
    for ti in range(num_steps):
        outputs[ti] = np.dot(C, state)
        state = np.dot(A, state) + np.dot(B, inputs[ti])
    return outputs


def impulse(A, B, C, num_time_steps=None):
    """Generates impulse response outputs for a discrete-time system.

    Args:
        ``A``, ``B``, ``C``: State-space system arrays.

    Kwargs:
        ``num_time_steps``: Number of time steps to simulate.
        By default, automatically chooses a value between 20 and 2000, stopping
        either when 2000 timesteps have elapsed, or when the outputs decay to
        below a magnitude of 1e-6.

    Returns:
        ``outputs``: Impulse response outputs, with indices corresponding to
        [time step, output, input].

    No D array is included, but one can simply be prepended to the output if
    it is non-zero.
    """
    num_inputs = B.shape[1]
    num_outputs = C.shape[0]
    A_powers = np.identity(A.shape[0])
    outputs = []

    if num_time_steps is None:
        tol = 1e-6
        min_time_steps = 20
        max_time_steps = 2000
        ti = 0
        continue_sim = True
        while continue_sim and ti < max_time_steps:
            outputs.append(C.dot(A_powers.dot(B)))
            A_powers = np.dot(A_powers, A)
            ti += 1
            if ti > min_time_steps:
                # PA: I changed that since it is strange and it gives
                # TypeError: unorderable types: list() < float()
                # with python 3
                # if (np.abs(outputs[-min_time_steps:] < tol)).all():
                if (np.abs(outputs[-1] < tol)).all():
                    continue_sim = False
    else:
        outputs = np.zeros((num_time_steps, num_outputs, num_inputs))
        for ti in range(num_time_steps):
            outputs[ti] = C.dot(A_powers.dot(B))
            A_powers = A_powers.dot(A)
    return outputs


def load_signals(signal_path, delimiter=None):
    """Loads signals from text files with columns [t signal1 signal2 ...].

    Args:
        ``signal_paths``: List of filepaths to files containing signals.

    Returns:
        ``time_values``: 1D array of time values.

        ``signals``: Array of signals with dimensions [time, signal].

    Convenience function. Example file has format::

      0 0.1 0.2
      1 0.2 0.46
      2 0.2 1.6
      3 0.6 0.1
    """
    raw_data = load_array_text(signal_path, delimiter=delimiter)
    num_signals = raw_data.shape[1] - 1
    if num_signals == 0:
        raise ValueError('Data must have at least two columns')
    time_values = raw_data[:, 0]
    signals = raw_data[:, 1:]
    # Guarantee that signals is 2D
    if signals.ndim == 1:
        signals = signals.reshape((signals.shape[0], 1))
    return time_values, signals


def load_multiple_signals(signal_paths, delimiter=None):
    """Loads multiple signal files from text files with columns [t channel1
    channel2 ...].

    Args:
        ``signal_paths``: List of filepaths to files containing signals.

    Returns:
        ``time_values``: 1D array of time values.

        ``all_signals``: Array of signals with indices [path, time, signal].

    See :py:func:`load_signals`.
    """
    num_signal_paths = len(signal_paths)

    # Read the first file to get parameters
    time_values, signals = load_signals(signal_paths[0], delimiter=delimiter)
    num_time_values = len(time_values)

    num_signals = signals.shape[1]

    # Now allocate array and read all of the signals
    all_signals = np.zeros((num_signal_paths, num_time_values, num_signals))

    # Set the signals we already loaded
    all_signals[0] = signals

    # Load all remaining files
    for path_num, signal_path in enumerate(signal_paths):
        time_values_read, signals = load_signals(signal_path,
            delimiter=delimiter)
        if not np.allclose(time_values_read, time_values):
            raise ValueError('Time values in %s are inconsistent with '
                'other files')
        all_signals[path_num] = signals

    return time_values, all_signals


def Hankel(first_col, last_row=None):
    """
    Construct a Hankel array, whose skew diagonals are constant.

    Args:
        ``first_col``: 1D array corresponding to first column of Hankel array.

    Kwargs:
        ``last_row``: 1D array corresponding to the last row of Hankel array.
        First element will be ignored.  Default is an array of zeros of the same
        size as ``first_col``.

    Returns:
        Hankel: 2D array with dimensions ``[len(first_col), len(last_row)]``.
    """
    first_col = first_col.flatten()
    if last_row is None:
        last_row = np.zeros(first_col.shape)
    else:
        last_row = last_row.flatten()

    unique_vals = np.concatenate((first_col, last_row[1:]))
    a, b = np.ogrid[0:len(first_col), 0:len(last_row)]
    indices = a + b
    return unique_vals[indices]


def Hankel_chunks(first_col_chunks, last_row_chunks=None):
    """
    Construct a Hankel array using chunks, whose elements have Hankel structure
    at the chunk level (constant along skew diagonals), rather than at the
    element level.

    Args:
        ``first_col_chunks``: List of 2D arrays corresponding to the first
        column of Hankel array chunks.

    Kwargs:
        ``last_row_chunks``: List of 2D arrays corresponding to the last row of
        Hankel array chunks.  Default is a list of arrays of zeros.

    Returns:
        Hankel:  2D array with dimension
        ``[len(first_col) * first_col[0].shape[0],
        len(last_row) * last_row[0].shape[1]]``.
    """
    # If nothing is passed in for last row, use a list of chunks where each
    # chunk is an array of zeros, and each chunk has the same size as the chunks
    # in the first column.
    if last_row_chunks is None:
        last_row_chunks = [
            np.zeros(first_col_chunks[0].shape)] * len(first_col_chunks)

    # Gather the unique chunks in one list
    unique_chunks = first_col_chunks + last_row_chunks[1:]

    # Use a list comprehension to create a list where each element of the list
    # is an array corresponding a all the chunks in a row.  To get that array,
    # slice the list of unique chunks using the right index and call hstack on
    # it.  Finally, call vstack on the list comprehension to get the whole
    # Hankel array.
    return np.vstack([np.hstack(
        unique_chunks[idx:idx + len(last_row_chunks)])
        for idx in range(len(first_col_chunks))])
