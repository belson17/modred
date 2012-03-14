
"""Example script using SimpleUsePOD.

The fields and modes are always in memory and are returned
by the instance of SimpleUsePOD.

This demonstrates a typical usage when the inner product is 
a simple function (or even the default one), and the 
fields are easily stacked as columns of an array.

However, this is somewhat against the philosophy of modred of not
using single large arrays and only works for limited simple cases.
Modred is more flexible and general than is seen in this example.
For anything more complicated, look at other examples.

This script assumes that modred has been installed or is otherwise
available to be imported.
"""

import modred
import numpy as N

def main(verbose=True, make_plots=True):
    # Collect the fields and place in columns of ``fields`` array.
    # Here we just use random data.
    num_states = 100
    num_fields = 30
    
    fields = N.random.random((num_states, num_fields))
    
    # Create an instance of SimpleUsePOD.
    # The default inner_product is used, but we could specify our own
    # with an optional argument to the constructor.
    my_simple_POD = modred.SimpleUsePOD(verbose=verbose)
    
    # Set the fields to find the modes of.
    my_simple_POD.set_fields(fields)
    
    # An optional step, returns the SVD matrices so we can determine
    # how many modes to keep.
    sing_vecs, sing_vals = my_simple_POD.compute_decomp()
    
    # Want to capture 90% of the energy, so:
    energy = 0.9
    sing_vals_norm = sing_vals/N.sum(sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > energy)[0][0] + 1
    
    # Compute the first ``num_modes`` modes. The array ``modes`` has columns of modes.
    modes = my_simple_POD.compute_modes(num_modes)
    # One could skip the compute_decomp step and call compute_modes
    # directly, but then it's not possible to determine how many modes
    # capture 90% of the energy until after (with my_simple_POD.sing_vals).
    
    # Make plots of leading modes if have matplotlib
    if make_plots:
        try:
            import matplotlib.pyplot as PLT
            PLT.figure()
            PLT.hold(True)
            for mode_index in range(min(num_modes/4, 3)):
                PLT.plot(modes[:,mode_index])
            PLT.legend(['POD mode %d'%(mode_index+1) for mode_index in range(min(num_modes/4, 3))])
            PLT.show()
        except:
            pass
    
if __name__ == '__main__':
    main()