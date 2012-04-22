"""A group of useful functions"""

import os
import numpy as N
import inspect

class UndefinedError(Exception): 
    """Error when something has not been defined"""
    pass

def make_list(arg):
    """Makes argument a list, if not already a list"""
    if not isinstance(arg, list):
        arg = [arg]
    return arg
    
def flatten_list(my_list):
    """Flatten a list of lists into a single list"""
    return [num for elem in my_list for num in elem]
    
def save_array_text(array, filename, delimiter=' '):
    """Saves a 1D or 2D array or matrix to a text file
    
    Args:
        mat: matrix or array to save to file (1D or 2D)
        
        filename: path to save to, string.
        
    Kwargs:   
        delimeter: delimeter in file, default is a whitespace.
    
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

    N.savetxt(filename, array_save.view(float), delimiter=delimiter)
    
    
def load_array_text(file_name, delimiter=' ', is_complex=False):
    """Loads a text file, returns an array.
    
    See :py:func:`save_array_text` for the format used by this function.
    
    Kwargs:
        is_complex: if the data saved is complex, then set  to ``True``.
    
    Returns:
        array: a numpy array, always 2D.
    """
    # Check the version of numpy, requires version >= 1.6 for ndmin option
    #if int(N.version.version[2]) < 6:
    #    print ('Warning: load_array_text requires numpy version >= 1.6 '
    #        'but you are running version %s'%N.version.version)
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
    """ A default inner product for n-dimensional numpy arrays """
    #return (vec1 * vec2.conj()).sum()
    return N.vdot(vec1, vec2)

    
def svd(mat, tol=1e-13):
    """An SVD that better meets our needs.
    
    Args:
        mat: array or matrix to take SVD of.
    
    Returns:
        U, E, V: U.E.V* = mat. U and V are matrices, E is a 1D array.
    
    Truncates the U, E, and V such that there are no ~0 singular values.
    """
    U, E, V_comp_conj = N.linalg.svd(N.mat(mat), full_matrices=0)
    V = N.mat(V_comp_conj).H
    U = N.mat(U)
    
    # Only return sing vals above the tolerance
    num_nonzeros = (abs(E) > tol).sum()
    if num_nonzeros > 0:
        U = U[:,:num_nonzeros]
        V = V[:,:num_nonzeros]
        E = E[:num_nonzeros]

    return U, E, V


def get_file_list(directory, file_extension=None):
    """Returns list of files in directory with file_extension"""
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
    """Returns a dictionary containing data members of an object"""
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not inspect.ismethod(value):
            pr[name] = value
    return pr


def sum_arrays(arr1, arr2):
    """Used for allreduce command"""
    return arr1 + arr2

    
def sum_lists(list1, list2):
    """Sum the elements of each list, return a new list.
    
    This function is used in MPI reduce commands, but could be used
    elsewhere too"""
    assert len(list1) == len(list2)
    return [list1[i] + list2[i] for i in xrange(len(list1))]


def solve_Lyapunov(A, Q):
    """Solves equation of form AXA' - X + Q = 0 for X given A and Q
    
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
    """Generates a discrete-time random state space system
    
    Args:
        number of states, inputs, and outputs.
    
    Returns:
        A, B, and C matrices of system.
    
    All e-vals are real.
    """
    eig_vals = N.linspace(.9, .95, num_states) 
    eig_vecs = N.random.normal(0, 2., (num_states, num_states))
    A = N.mat(N.real(N.dot(N.dot(N.linalg.inv(eig_vecs), 
        N.diag(eig_vals)), eig_vecs)))
    B = N.mat(N.random.normal(0, 1., (num_states, num_inputs)))
    C = N.mat(N.random.normal(0, 1., (num_outputs, num_states)))
    return A, B, C

def rss(num_states, num_inputs, num_outputs):
    """Generates a continuous-time random state space systme.

    Args:
        number of states, inputs, and outputs.
    
    Returns:
        A, B, and C matrices of system.
        
    All e-vals are real.
    """
    e_vals = -N.random.random(num_states)
    transformation = N.random.random((num_states, num_states))
    A = N.dot(N.dot(N.linalg.inv(transformation), N.diag(e_vals)),
        transformation)
    B = N.random.random((num_states, num_inputs))
    C = N.random.random((num_outputs, num_states))
    return A, B, C
        
        
def lsim(A, B, C, inputs):
    """Simulates a discrete time system with arbitrary inputs. 
    
    Args:
        inputs: array with indices [num_time_steps, num_inputs]
    
    Returns:
        outputs: array with indicies [num_time_steps, num_outputs]
    
    Currently D matrix is not used, assumed to be zero.
    """
    D = 0
    if inputs.ndim == 1:
        inputs = inputs.reshape((len(inputs), 1))
    num_steps, num_inputs = inputs.shape
    num_outputs = C.shape[0]
    num_states = A.shape[0]
    #print 'num_states is',num_states,'num inputs',num_inputs,'B shape',B.shape
    if B.shape != (num_states, num_inputs):
        raise ValueError('B has the wrong shape ', B.shape)
    if A.shape != (num_states, num_states):
        raise ValueError('A has the wrong shape ', A.shape)
    if C.shape != (num_outputs, num_states):
        raise ValueError('C has the wrong shape ', C.shape)
    if D == 0:
        D = N.zeros((num_outputs, num_inputs))
    if D.shape != (num_outputs, num_inputs):
        raise ValueError('D has the wrong shape, D=', D)
    
    outputs = [] 
    state = N.mat(N.zeros((num_states, 1)))
    
    for input in inputs:
        #print 'assigning',N.dot(C, state).shape,'into',outputs[time].shape
        input_reshape = input.reshape((num_inputs, 1))
        outputs.append((C*state).squeeze())
        #print 'shape of D*input',N.dot(D,input_reshape).shape
        #Astate = A*state
        #print 'shape of B is',B.shape,
        #print 'and shape of input is',input.reshape((num_inputs,1)).shape
        state = A*state + B*input_reshape
    
    outputs_array = N.zeros((num_steps, num_outputs))
    for t, out in enumerate(outputs):
        #print 'assigning out.shape',out.shape,'into',outputs_array[t].shape
        #print 'num_outputs',num_outputs
        outputs_array[t] = out

    return outputs_array

    
def impulse(A, B, C, time_steps=None):
    """Generates impulse response outputs for a discrete-time system.
    
    Args:
        A, B, C: state-space system matrices
        
        time_steps: 1D array of integers specifying the time steps
        
    No D matrix is included, but can simply be prepended to the output if it is
    non-zero. 
    """
    #num_states = A.shape[0]
    num_inputs = B.shape[1]
    num_outputs = C.shape[0]
    if time_steps is None:
        tol = 1e-6
        max_time_steps = 2000
        Markovs = [C*B]
        time_steps = [0]
        while (N.amax(abs(Markovs[-1])) > tol or len(Markovs) < 20) and \
            len(Markovs) < max_time_steps:
            time_steps.append(time_steps[-1] + 1)
            Markovs.append(C * (A**time_steps[-1]) * B)
    else:
        Markovs = []
        for ts in time_steps:
            Markovs.append(C*(A**ts)*B)

    outputs = N.zeros((len(Markovs), num_outputs, num_inputs))
    for time_step, Markov in enumerate(Markovs):
        outputs[time_step] = Markov
    time_steps = N.array(time_steps)
    
    return time_steps, outputs



def load_signals(signal_path, delimiter=' '):
    """Loads signals with columns [t signal1 signal2 ...].
    
    Args:
        signal_paths: list of paths to signals, strings
    
    Returns:
        time_values: 1D array of time values
        
        signals: array of signals with dimensions [time#, signal#]
    
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
    signals = raw_data[:,1:]
    # Guarantee that signals is 2D
    if signals.ndim == 1:
        signals = signals.reshape((signals.shape[0], 1))
    return time_values, signals



def load_multiple_signals(signal_paths, delimiter=' '):
    """Loads multiple signal files w/columns [t channel1 channel2 ...].
    
    Args:
        signal_paths: list of paths to signals, strings
    
    Returns:
        time_values: 1D array of time values
        
        all_signals: array of signals with dimensions [path#, time#, signal#]
    
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
        time_values_read, signals = load_signals(signal_path, delimiter=delimiter)
        if not N.allclose(time_values_read, time_values):
            raise ValueError('Time values in %s are inconsistent with '
                'other files')
        all_signals[path_num] = signals 

    return time_values, all_signals

