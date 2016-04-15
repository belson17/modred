"""OKID function. (Book: Applied System Identification, Jer-Nan Juang, 1994)"""
from future.builtins import range

import numpy as np


def OKID(inputs, outputs, num_Markovs):
    """Approximates Markov paramters using arbitrary input and output data.
    
    Args:
        ``inputs``: Array of input signals. Each row corresponds to a different
        input, each each column to a different time.
        
        ``outputs``: Array of output signals. Each row corresponds to a
        different output, each each column to a different time.
        
        ``num_Markovs``: Number of Markov parameters to estimate.
            
    Returns:
        ``Markovs_est``: Array of Markov paramemters. Array dimensions
        correspond to times, outputs, and inputs, respectively.  Thus
        ``Markovs_est[ti]`` is the Markov parameter at time index ``ti``.
    
    OKID can be sensitive to the choice of parameters. A few tips:

    - Use a tail (input=0) for your input/output data, otherwise the Markov
      parameters might grow rather than decay at large times.
    - If necessary, artificially append your data with zero input and 
      exponentially decaying output.
    - Set ``num_Markovs`` less than or equal to half of the number of samples
      (:math:`num_Markovs <= num_samples / 2`).  Estimating too many Markov
      parameters can produce spurious oscillations.    
    - Data with more than one input tends to be harder to work with. 
    """    
    # Some internal comments and variables refer to textbook (J.-np. Juang
    # 1994).
    # Force arrays to be 2 dimensional
    if inputs.ndim == 1:
        inputs = inputs.reshape((1, inputs.shape[0]))
    if outputs.ndim == 1:
        outputs = outputs.reshape((1, outputs.shape[0]))
    
    num_inputs, num_samples = inputs.shape
    num_outputs, num_samples_outputs = outputs.shape
    if num_samples != num_samples_outputs:
        raise ValueError(('number of samples in input and output differ'+
        ', %d != %d'%(num_samples, num_samples_outputs)))
    
    # Convenience variable
    num_inouts = num_inputs + num_outputs
    
    V = np.zeros((num_inputs + num_inouts*num_Markovs, num_samples))
    V[:num_inputs] = inputs
    
    inputs_outputs = np.concatenate((inputs, outputs), axis=0)
    for i in range(1, num_Markovs+1):
        V[num_inputs + (i-1)*num_inouts:num_inputs + i*num_inouts, i:] = \
            inputs_outputs[:, :num_samples-i]
    
    # Ybar in book
    #Markov_aug = np.dot(outputs, np.array(np.linalg.pinv(V, cutoff)))
    # Using least squares for better numerical conditioning? outputs = M V, so
    # outputs.T = V.T M.T, solve for M = Markovs_aug.
    Markov_aug = np.linalg.lstsq(V.T, outputs.T)[0].T

    D = Markov_aug[:, :num_inputs]
    
    # Convenience variable
    Markov_aug_noD = Markov_aug[:, num_inputs:]
    
    # Break the augmented system Markov parameters into manageable lists
    # Ybar1 and Ybar2 in book:
    Markov_aug_input = [D]
    Markov_aug_output = [D]

    for Markov_num in range(num_Markovs):
        Markov_aug_input.append(
            Markov_aug_noD[:, 
                Markov_num*num_inouts:Markov_num*num_inouts+num_inputs])
        Markov_aug_output.append(
            -Markov_aug_noD[:,Markov_num*num_inouts+num_inputs:
                Markov_num*num_inouts+num_inouts])
    
    # Estimate the Markov parameters of the system
    Markov_est = np.empty((num_Markovs, num_outputs, num_inputs))
    Markov_est[0] = D
    
    for Markov_num in range(1, num_Markovs):
        summation_term = np.zeros((num_outputs, num_inputs))
        for i in range(1, Markov_num+1):
            summation_term += np.dot(Markov_aug_output[i], 
                Markov_est[Markov_num-i,:,:])
            #print 'k=%d, i=%d'%(Markov_num,i)
        Markov_est[Markov_num] = Markov_aug_input[Markov_num] - \
            summation_term
   
    return Markov_est
    
