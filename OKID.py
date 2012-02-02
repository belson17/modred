
import numpy as N
import util

def OKID(inputs, outputs, num_Markovs, cutoff=1e-3):
    """
    Returns the Markov paramters (impulse outputs) from arbitrary inputs and outputs
    
    inputs: [num_samples, num_inputs] array
    outputs: [num_samples, num_inputs] array
    num_Markovs: integer, number of approx Markov parameters to return.
      Typically 5x the number of ROM states is good.
    cutoff: condition number used for the pseudo-inverse
    
    Original paper is 1991 NASA TM-104069 by Juang, Phan, Horta and Longman
    Some notation is used from this paper.
    lowercase u,y indicate sampled data
    double uppercase UU, YY indicate bold-faced quantities in paper
    single uppercase U, Y indicate script quantities in paper
    """
    
    # Take transpose to better match paper
    # Also force to be 2 dimensional
    inputs = N.array(N.copy(N.squeeze(inputs)).T, ndmin=2)
    outputs = N.array(N.copy(N.squeeze(outputs)).T, ndmin=2)
    #print 'inputs',inputs.shape
    #print 'outputs',outputs.shape
    num_inputs, num_samples = inputs.shape
    num_outputs, num_samples_outputs = outputs.shape
    if num_samples != num_samples_outputs:
        raise ValueError(('number of samples in input and output differ'+
        ', %d != %d'%(num_samples, num_samples_outputs)))
    
    # Convenience variable
    num_inouts = num_inputs + num_outputs
    
    #print num_inputs, num_outputs, num_Markovs, num_samples
    V = N.zeros((num_inputs + num_inouts*num_Markovs, num_samples))
    V[:num_inputs] = inputs
    
    input_output = N.concatenate((inputs, outputs), axis=0)
    #print 'shape of input output',input_output.shape
    for i in range(1,num_Markovs+1):
        #print 'shape of V assigned to is',V[num_inputs+(Markov_num-1)*num_inouts:num_inputs+Markov_num*num_inouts,Markov_num:].shape
        V[num_inputs+(i-1)*num_inouts:num_inputs+i*num_inouts,i:] = \
            input_output[:, :num_samples-i]
    
    #print 'V',V
    #util.save_mat_text(V,'V.txt')
    #print 'pinv',N.linalg.pinv(V,cutoff)
    Markov_aug = N.dot(outputs, N.array(N.linalg.pinv(V, cutoff)))
    #print 'y',outputs
    #print 'Ybar',Markov_aug   
    #D = N.zeros((num_outputs, num_inputs))
    D = Markov_aug[:,:num_inputs]
    Markov_aug_noD = Markov_aug[:,num_inputs:]
    
    impulse_outputs = N.empty((num_outputs, num_inputs, num_Markovs))
    #print 'assigning D w/ shape',D.shape,'into',impulse_outputs[:,:,0].shape
    impulse_outputs[:,:,0] = D
    
    # Break the augmented system Markov parameters into manageable lists
    Markov_aug_input = []
    Markov_aug_output = []
    for Markov_num in range(num_Markovs):
        Markov_aug_input.append(
            Markov_aug_noD[:,Markov_num*num_inouts:Markov_num*num_inouts+num_inputs])
        #print 'max col accessed is',num_inputs + Markov_num*num_inouts + num_inputs
        Markov_aug_output.append(
            Markov_aug_noD[:,Markov_num*num_inouts+num_inputs:Markov_num*num_inouts+num_inouts])
        #print 'max col accessed is',num_inputs + Markov_num*num_inouts + num_inputs + num_outputs
    #print 'Ybar1',Markov_aug_input
    #print 'Ybar2',Markov_aug_output
    #temp = Markov_aug_input
    #Markov_aug_input = Markov_aug_output
    #Markov_aug_output = temp
    # Calculate the Markov parameters of the original system
    for Markov_num in range(num_Markovs):
        sum_term = N.zeros((num_outputs,num_inputs))
        if Markov_num > 0:
            for i in range(Markov_num):
                #print 'size of sum_term is',sum_term.shape,'adding to it shape',\    
                #    N.dot(Markov_aug_output[i], impulse_outputs[:,:,Markov_num-i-1]).shape
                sum_term += N.dot(Markov_aug_output[i], impulse_outputs[:,:,Markov_num-i-1])
                #print 'k=%d, i=%d'%(Markov_num,i)
                #print 'combining Hbar %d and H %d'%(i,Markov_num-i-1)
        #print 'assigning into',impulse_outputs[:,:,Markov_num].shape
        #print 'assigning',(Markov_aug_input[Markov_num] + 
        #    N.dot(Markov_aug_output[Markov_num], D) + sum_term).squeeze().shape
        impulse_outputs[:,:,Markov_num] = (Markov_aug_input[Markov_num] + 
            N.dot(Markov_aug_output[Markov_num], D) + sum_term)
    
    #impulse_outputs[:,Markov_num] = Markov_aug_input + N.dot(Markov_aug_output, D) + \
    #    N.dot(Markov_aug_output, impulse_outputs[:,Markov_num-1])
    #print Markov_aug
    #print Markov_aug_input
    #print Markov_aug_output
    
    return impulse_outputs.swapaxes(1,2).swapaxes(0,1)
    