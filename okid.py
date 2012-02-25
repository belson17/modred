
import numpy as N
import util

def OKID(inputs, outputs, num_Markovs, cutoff=1e-6):
    """
    Returns the Markov paramters from arbitrary inputs and outputs
    
    inputs: [num_inputs, num_samples] array
    outputs: [num_outputs, num_samples] array
    num_Markovs: integer, number of approx Markov parameters to return.
      Typically 5x the number of ROM states is good.
    cutoff: condition number used for the pseudo-inverse
    
    OKID is very sensitive to the parameters used. 
    For example, if the outputs do not have a long tail, the Markov parameters could be
    growing rather than decaying.
    Also, if you try to estimate too many Markov parameters, then it appears that they *all* 
    suffer in decreased accuracy and spurious oscillations.
    A few tips:
      - Use a long tail for your input/output data.
      - In the tail, have input=0.
      - If necessary, artificially append your data with zero input and exponentially decaying
          output.
      - Estimate fewer Markov parameters if oscillations are seen, typically less than 
          half of the number of samples.
      - Data with more than one input is the most sensitive and hardest to work with. The
          number of outputs tends not to require one to be as careful picking parameters.
          
    The original paper is 1991 NASA TM-104069 by Juang, Phan, Horta and Longman
    Some comments refer to notation from this paper.
    """
    
    # Force to be 2 dimensional
    if inputs.ndim == 1:
    		inputs = inputs.reshape((1,inputs.shape[0]))
    if outputs.ndim == 1:
    		outputs = outputs.reshape((1,outputs.shape[0]))
    
    num_inputs, num_samples = inputs.shape
    num_outputs, num_samples_outputs = outputs.shape
    if num_samples != num_samples_outputs:
        raise ValueError(('number of samples in input and output differ'+
        ', %d != %d'%(num_samples, num_samples_outputs)))
    
    # Convenience variable
    num_inouts = num_inputs + num_outputs
    
    V = N.zeros((num_inputs + num_inouts*num_Markovs, num_samples))
    V[:num_inputs] = inputs
    
    inputs_outputs = N.concatenate((inputs, outputs), axis=0)
    for i in range(1,num_Markovs+1):
        V[num_inputs+(i-1)*num_inouts:num_inputs+i*num_inouts,i:] = \
            inputs_outputs[:, :num_samples-i]
    
    # Hbar in paper, Ybar in matlab script OKID.m
    Markov_aug = N.dot(outputs, N.array(N.linalg.pinv(V, cutoff)))

    D = Markov_aug[:,:num_inputs]
    Markov_aug_noD = Markov_aug[:,num_inputs:]
    
    # Break the augmented system Markov parameters into manageable lists
    # Hbar1 in paper, Ybar1 in matlab
    Markov_aug_input = []
    # Hbar2 in paper, Ybar2 in matlab
    Markov_aug_output = []
    for Markov_num in range(num_Markovs):
        Markov_aug_input.append(
            Markov_aug_noD[:,Markov_num*num_inouts:Markov_num*num_inouts+num_inputs])
        Markov_aug_output.append(
            Markov_aug_noD[:,Markov_num*num_inouts+num_inputs:Markov_num*num_inouts+num_inouts])
    
    # Estimate the Markov parameters of the system
    Markov_est = N.empty((num_outputs, num_inputs, num_Markovs))
    Markov_est[:,:,0] = Markov_aug_input[0] + N.dot(Markov_aug_output[0], D)
    
    for Markov_num in range(1, num_Markovs):
    		sum_term = N.zeros((num_outputs,num_inputs))
    		for i in range(Markov_num):
    				sum_term += N.dot(Markov_aug_output[i], Markov_est[:,:,Markov_num-i-1])
    				print 'k=%d, i=%d'%(Markov_num,i)
    		Markov_est[:,:,Markov_num] = Markov_aug_input[Markov_num]+\
    		N.dot(Markov_aug_output[Markov_num], D) + sum_term
    
    Markov_est[:,:,1:] = Markov_est[:,:,:-1]
    Markov_est[:,:,0] = D
    
    return Markov_est
    
