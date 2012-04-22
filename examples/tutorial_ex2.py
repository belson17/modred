import numpy as N
import modred as MR

def main(verbose=True):
    num_vecs = 30
    nx = 100
    ny = 45
    x_grid = 1. - N.cos(N.linspace(0, N.pi, nx))
    y_grid = N.linspace(0, 2., ny)**2
    
    # We use arbitrary fake data as a placeholder
    # Normally one doesn't need to do this because you have real data.
    Y, X = N.meshgrid(y_grid, x_grid)
    vecs = [N.sin(X*0.1*i) + N.cos(Y*0.15*i) for i in range(num_vecs)]
    
    my_IP_trapz = MR.InnerProductTrapz(x_grid, y_grid)
    my_POD = MR.POD(inner_product=my_IP_trapz, verbose=verbose)
    sing_vecs, sing_vals = my_POD.compute_decomp_in_memory(vecs)
    num_modes = 10
    modes = my_POD.compute_modes_in_memory(range(num_modes))

if __name__ == '__main__':
    main()