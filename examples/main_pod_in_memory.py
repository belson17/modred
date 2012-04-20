"""Example script.

This demonstrates a typical usage when the inner product and vectors
are small and simple.

This script can be run in parallel with::

  mpiexec -n 4 python main_pod_in_memory.py

This script assumes that modred has been installed or is otherwise
available to be imported.
"""
import numpy as N
import modred as MR

def main(verbose=True, make_plots=False):
    num_states = 50
    num_vecs = 20
    x = N.linspace(0., 1., num_states)
    # Make up data as a placeholder
    vecs = [N.cos(i*x)**i for i in range(num_vecs)]
    
    my_POD = MR.POD(inner_product=N.vdot, verbose=verbose)
    sing_vecs, sing_vals = my_POD.compute_decomp_in_memory(vecs)

    # Want to capture 90% of the energy, so:
    energy = 0.9
    sing_vals_norm = sing_vals/N.sum(sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > energy)[0][0] + 1

    modes = my_POD.compute_modes_in_memory(range(num_modes))
    
    # Make plots of leading modes if desired
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
        except ImportError:
            print 'matplotlib not found'
            
if __name__ == '__main__':
    main(make_plots=False)
    
