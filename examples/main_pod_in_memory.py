"""Example script using SimpleUsePOD.

The vecs and modes are always in memory and are returned
by the instance of SimpleUsePOD.

This demonstrates a typical usage when the inner product is 
a simple function (or even the default one), and the 
vecs are easily stacked as columns of an array.

However, this is somewhat against the philosophy of modred of not
using single large arrays and only works for limited simple cases.
Modred is more flexible and general than is seen in this example.
For anything more complicated, look at other examples.

This script assumes that modred has been installed or is otherwise
available to be imported.
"""
import numpy as N
import modred

def main(verbose=True, make_plots=True):
    num_states = 50
    num_vecs = 25
    
    vecs = [N.random.random((num_states)) for i in range(num_vecs)]
    
    my_POD = modred.POD(modred.vecdefs.ArrayInMemory(), 
        max_vecs_per_node=1000, verbose=verbose)
    sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vecs)

    # Want to capture 90% of the energy, so:
    energy = 0.9
    sing_vals_norm = sing_vals/N.sum(sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > energy)[0][0] + 1

    modes = my_POD.compute_modes(range(1, 1+num_modes), [None]*num_modes)
    
    # Make plots of leading modes if have matplotlib
    if make_plots:
        try:
            import matplotlib.pyplot as PLT
            PLT.figure()
            PLT.hold(True)
            for mode_index in range(min(num_modes/4, 3)):
                PLT.plot(modes[mode_index])
            PLT.legend(['POD mode %d'%(mode_index+1) 
                for mode_index in range(min(num_modes/4, 3))])
            PLT.show()
        except: pass
if __name__ == '__main__':
    main()
    
