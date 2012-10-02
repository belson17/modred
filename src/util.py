"""A group of useful functions"""

import os
import numpy as N
import inspect

class UndefinedError(Exception): 
    """Error when something has not been defined."""
    pass

def make_list(arg):
    """Returns the argument as a list. If a list, arg is returned."""
    if not isinstance(arg, list):
        arg = [arg]
    return arg
    
def flatten_list(my_list):
    """Flatten a list of lists into a single list"""
    return [num for elem in my_list for num in elem]
    
def save_array_text(array, file_name, delimiter=' '):
    """Saves a 1D or 2D array or matrix to a text file.
    
    Args:
        ``array``: Matrix or array to save to file (1D or 2D).
        
        ``file_name``: Path to save to, string.
        
    Kwargs:   
        ``delimeter``: Delimeter in file, default is a whitespace.
    
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
    # Cast into an array. Also makes it memory C-contiguous.
    array_save = N.array(array)
    
    # If one-dimensional array, then make a vector of many rows, 1 column
    if array_save.ndim == 1:
        array_save = array_save.reshape((-1, 1))
    elif array_save.ndim > 2:
        raise RuntimeError('Cannot save an array with >2 dimensions')

    N.savetxt(file_name, array_save.view(float), delimiter=delimiter)
    
    
def load_array_text(file_name, delimiter=' ', is_complex=False):
    """Loads a text file, returns an array.
    
    Args:
        ``file_name``: Name of file to load.
    
    Kwargs:
        ``is_complex``: Bool, if the data saved is complex then use ``True``.
    
    Returns:
        ``array``: 2D numpy array.

    See :py:func:`save_array_text` for the format used by this function.
    """
    if is_complex:
        dtype = complex
    else:
        dtype = float
    array = N.loadtxt(file_name, delimiter=delimiter) #, ndmin=2)
    ## This section reproduces behavior of ndmin=2 option of N.loadtxt
    if array.ndim == 1:
        num_rows = sum(1 for line in open(file_name))
        if num_rows > 1:
            array = array.reshape((-1, 1))
        else:
            array = array.reshape((1, -1))
    ## End work around for ndmin=2 option.
    
    if is_complex and array.shape[1] % 2 != 0:
        raise ValueError(('Cannot load complex data, file %s '%file_name)+\
            'has an odd number of columns. Maybe it has real data.')
            
    # Cast as an array, copies to make it C-contiguous memory
    return N.array(array.view(dtype))


def inner_product(vec1, vec2):
    """A default inner product for n-dimensional numpy arrays. """
    return N.vdot(vec1, vec2)

    
def svd(mat, tol=1e-13):
    """An SVD that better meets our needs. U E V* = mat. 
    
    Args:
        ``mat``: Array or matrix to take SVD of.
    
    Kwargs:
        ``tol``: Level at which singular values are truncated.
    
    Returns:
        ``U``: Matrix of left singular vectors.
        
        ``E``: 1D array of singular values.
        
        ``V``: Matrix of right singular vectors.
    
    Truncates U, E, and V such that there are no ~0 singular values.
    """
    U, E, V_comp_conj = N.linalg.svd(N.mat(mat), full_matrices=0)
    V = N.mat(V_comp_conj).H
    U = N.mat(U)
    
    # Only return sing vals above the tolerance
    num_nonzeros = (abs(E) > tol).sum()
    if num_nonzeros > 0:
        U = U[:, :num_nonzeros]
        V = V[:, :num_nonzeros]
        E = E[:num_nonzeros]

    return U, E, V

def eigh(mat, tol=1e-12, is_positive_definite=False):
    """Computes the eigenvalues and vecs of Hermitian matrix/array.
    
    Args:
        ``mat``: To take eigen decomposition of.
        
        ``tol``: Value at which to truncate eigenvalues and vectors.
            Give ``None`` for no truncation.

        ``is_positive_definite``: If true, matrix being decomposed will be 
            assumed to be positive definite.  Tolerance will be automatically 
            adjusted (if necessary) so that only positive eigenvalues are 
            returned.
    
    Returns:
        ``evals``: Eigenvalues in a 1D array, sorted in descending order.
        
        ``evecs``: Eigenvectors, columns of matrix/array, sorted by evals.
    """
    evals, evecs = N.linalg.eigh(mat)

    # Sort the vecs and vals by eval magnitude
    sort_indices = N.argsort(N.abs(evals))[::-1]
    evals = evals[sort_indices]
    evecs = evecs[:, sort_indices]

    # Filter eigenvalues, if necessary
    if tol is not None:
        # Adjust tolerance for pos def case if necessary. Do so if there are
        # negative eigenvalues and the most negative one has magnitude greater
        # than the tolerance.
        if is_positive_definite and evals.min() < 0 and abs(evals.min()) > tol:
            tol = abs(evals.min())
        num_nonzeros = (abs(evals) > tol).sum()
        evals = evals[:num_nonzeros]
        evecs = evecs[:,:num_nonzeros]    
    return evals, evecs


def get_file_list(directory, file_extension=None):
    """Returns list of files in ``directory`` with ``file_extension``."""
    files = os.listdir(directory)
    if file_extension is not None:
        if len(file_extension) == 0:
            print 'Warning: gave an empty file extension'
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
    """Used for allreduce command."""
    return arr1 + arr2

    
def sum_lists(list1, list2):
    """Sum the elements of each list, return a new list.
    
    This function is used in MPI reduce commands, but could be used
    elsewhere too."""
    assert len(list1) == len(list2)
    return [list1[i] + list2[i] for i in xrange(len(list1))]


def solve_Lyapunov(A, Q):
    """Solves equation of form AXA' - X + Q = 0 for X given A and Q.
    
    See http://en.wikipedia.org/wiki/Lyapunov_equation
    """
    A = N.array(A)
    Q = N.array(Q)
    if A.shape != Q.shape:
        raise ValueError('A and Q dont have same shape')
    #A_flat = A.flatten()
    Q_flat = Q.flatten()
    kron_AA = N.kron(A, A)
    X_flat = N.linalg.solve(N.identity(kron_AA.shape[0]) - kron_AA, Q_flat)
    X = X_flat.reshape((A.shape))
    return X


def drss(num_states, num_inputs, num_outputs):
    """Generates a discrete-time random state space system.
    
    Args:
        ``num_states``: Number of states.
        
        ``num_inputs``: Number of inputs.
        
        ``num_outputs``: Number of outputs.
    
    Returns:
        ``A``, ``B``, and ``C`` matrices of system.
        
    All eigenvalues are real and stable.
    """
    eig_vals = N.linspace(.9, .95, num_states) 
    eig_vecs = N.random.normal(0, 2., (num_states, num_states))
    A = N.mat(N.real(N.dot(N.dot(N.linalg.inv(eig_vecs), 
        N.diag(eig_vals)), eig_vecs)))
    B = N.mat(N.random.normal(0, 1., (num_states, num_inputs)))
    C = N.mat(N.random.normal(0, 1., (num_outputs, num_states)))
    return A, B, C

def rss(num_states, num_inputs, num_outputs):
    """Generates a continuous-time random state space system.

    Args:
        ``num_states``: Number of states.
        
        ``num_inputs``: Number of inputs.
        
        ``num_outputs``: Number of outputs.
    
    Returns:
        ``A``, ``B``, and ``C`` matrices of system.
        
    All eigenvalues are real and stable.
    """
    e_vals = -N.random.random(num_states)
    transformation = N.random.random((num_states, num_states))
    A = N.dot(N.dot(N.linalg.inv(transformation), N.diag(e_vals)),
        transformation)
    B = N.random.random((num_states, num_inputs))
    C = N.random.random((num_outputs, num_states))
    return A, B, C
        
        
def lsim(A, B, C, inputs, initial_condition=None):
    """Simulates a discrete time system with arbitrary inputs. 
    
    x(n+1) = Ax(n) + Bu(n); y(n) = Cx(n)
    
    Args:
        ``A``, ``B``, and ``C``
        
        ``inputs``: Array with indices [num_time_steps, num_inputs], u.
    
    Kwargs:
        ``initial_condition``: Initial condition, x(0).
    
    Returns:
        ``outputs``: Array with indices [num_time_steps, num_outputs], y.
    
    D matrix is assumed to be zero.
    """
    #D = 0
    A_arr = N.array(A)
    B_arr = N.array(B)
    C_arr = N.array(C)
    if inputs.ndim == 1:
        inputs = inputs.reshape((len(inputs), 1))
    num_steps, num_inputs = inputs.shape
    num_outputs = C.shape[0]
    num_states = A.shape[0]
    if A_arr.shape != (num_states, num_states):
        raise ValueError('A has the wrong shape ', A.shape)
    if B_arr.shape != (num_states, num_inputs):
        raise ValueError('B has the wrong shape ', B.shape)
    if C_arr.shape != (num_outputs, num_states):
        raise ValueError('C has the wrong shape ', C.shape)
    #if D == 0:
    #    D = N.zeros((num_outputs, num_inputs))
    #if D.shape != (num_outputs, num_inputs):
    #    raise ValueError('D has the wrong shape, D=', D)
    if initial_condition is not None:
        if initial_condition.shape[0] != num_states or initial_condition.ndim != 1:
            raise ValueError('initial_condition has the wrong shape')
    else:
        initial_condition = N.zeros(num_states)
    state = initial_condition
    outputs = N.zeros((num_steps, num_outputs)) 
    for ti in xrange(num_steps):
        outputs[ti] = N.dot(C_arr, state)
        state = N.dot(A_arr, state) + N.dot(B_arr, inputs[ti])
    return outputs

    
def impulse(A, B, C, num_time_steps=None):
    """Generates impulse response outputs for a discrete-time system.
    
    Args:
        ``A, B, C``: State-space system matrices (numpy arrays or matrices).
        
        ``num_time_steps``: Number of time steps to simulate.
    
    Returns:
        ``outputs``: Response outputs, indices [time step, output, input].
    
    No D matrix is included, but can simply be prepended to the output if it is
    non-zero. 
    """
    A_arr = N.array(A)
    B_arr = N.array(B)
    C_arr = N.array(C)
    num_inputs = B.shape[1]
    num_outputs = C.shape[0]
    A_powers = N.identity(A_arr.shape[0])
    outputs = []
    
    if num_time_steps is None:
        tol = 1e-6
        min_time_steps = 20
        max_time_steps = 2000
        ti = 0
        continue_sim = True
        while continue_sim and ti < max_time_steps:
            outputs.append(N.dot(N.dot(C_arr, A_powers), B_arr))
            A_powers = N.dot(A_powers, A_arr)
            ti += 1
            if ti > min_time_steps:
                if (N.abs(outputs[-min_time_steps:] < tol)).all():
                    continue_sim = False
        outputs = N.array(outputs)
    else:
        outputs = N.zeros((num_time_steps, num_outputs, num_inputs))
        for ti in range(num_time_steps):
            outputs[ti] = N.dot(N.dot(C_arr, A_powers), B_arr)
            A_powers = N.dot(A_powers, A_arr) 
    return outputs



def load_signals(signal_path, delimiter=' '):
    """Loads signals from text files with columns [t signal1 signal2 ...].     
    
    Args:
        ``signal_paths``: List of paths to signals, strings.
    
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



def load_multiple_signals(signal_paths, delimiter=' '):
    """Loads multiple signal files w/columns [t channel1 channel2 ...].
    
    Args:
        ``signal_paths``: List of paths to signals, strings.
    
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
    all_signals = N.zeros((num_signal_paths, num_time_values, num_signals))    
    
    # Set the signals we already loaded
    all_signals[0] = signals
    
    # Load all remaining files
    for path_num, signal_path in enumerate(signal_paths):
        time_values_read, signals = load_signals(signal_path, 
            delimiter=delimiter)
        if not N.allclose(time_values_read, time_values):
            raise ValueError('Time values in %s are inconsistent with '
                'other files')
        all_signals[path_num] = signals 

    return time_values, all_signals

def smart_eq(arg1, arg2):
    """Checks if equal, accounting for numpy's ``==`` not returning a bool."""
    eq = (arg1 == arg2)
    if isinstance(eq, N.ndarray):
        return eq.all()
    return eq
        
