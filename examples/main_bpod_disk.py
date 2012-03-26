#!/usr/bin/env python

"""Example of BPOD on 2D vecs.

This example has most of the complexities we anticipated BPOD to handle, 
and demonstrates a typical usage.

The vecs are larger (2D), so loading them 
all into memory simultaneously may be inefficient or impossible.
Instead, they are loaded from disk as needed. 

Further, the inner product is more complicated.
The vecs are on a non-uniform grid, and so the inner product uses
trapezoidal rule as an approximation.

The loading of vecs is also non-trivial.
In this case, a base vec is subtracted from the 2D vecs.
In practice this base vec might be the equilibrium about which the
governing equations are linearized, and so the saved vecs could be
from some simulation software.
The modes, which are saved to disk, do not include the base vec.

Another benefit of having the vecs saved to disk is BPOD can then be
used in a distributed memory parallel setting.
In fact, this script can be run in parallel with, e.g.::

  mpiexec -n 4 python main_bpod_disk.py

This script assumes that modred has been installed or is otherwise
available to be imported.
"""

import os
from os.path import join
from shutil import rmtree
import copy
import numpy as N
import modred
from parallel import default_instance
parallel = default_instance

class Vec(object):
    """The vec objects used will be instances of this class"""
    def __init__(self, x_grid, y_grid, path=None):
        self.x_grid = x_grid
        self.y_grid = y_grid
        if path is not None:
            self.load(path)
        else:
            self.data = None
    def save(self, path):
        """Save vec to text format"""
        modred.save_mat_text(self.data, path)
    
    def load(self, path):
        """Load vec from text format, still with base vec"""
        self.data = modred.load_mat_text(path)
    
    def inner_product(self, vec2):
        return N.trapz(N.trapz(self.data * vec2.data, x=self.y_grid),
            x=self.x_grid) 
    
    def __mul__(self, a):
        vec_return = copy.deepcopy(self)
        vec_return.data *= a
        return vec_return
    def __rmul__(self, a):
        return self.__mul__(a)
    def __lmul__(self, a):
        return self.__mul__(a)
        
    def __add__(self, other):
        vec_return = copy.deepcopy(self)
        vec_return.data += other.data
        return vec_return
    def __eq__(self, other):
        return N.allclose(self.data, other.data)
    def __sub__(self, other):
        return self + (-1.*other)

class VecDefs(object):
    """Simple wrapper to use ``Vec`` in modred"""
    def __init__(self, x_grid, y_grid, base_vec):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.base_vec = base_vec
    def get_vec(self, vec_source):
        return Vec(self.x_grid, self.y_grid, vec_source) - self.base_vec    
    def put_vec(self, vec, vec_dest):
        vec.save(vec_dest)
    def inner_product(self, vec1, vec2):
        return vec1.inner_product(vec2)
       
       
def main(verbose=True, make_plots=True):        
    # Define some parameters
    make_plots = False
    nx = 20
    ny = 30
    x_grid = 1 + N.sin(N.linspace(-N.pi, N.pi, nx))
    y_grid = 1 + N.sin(N.linspace(-N.pi, N.pi, ny))
    num_direct_vecs = 30
    num_adjoint_vecs = 25
    save_dir = join(os.path.dirname(__file__), 'DELETE_ME_bpod_example_files')
    
    # Create the directory for example files only on processor 0.
    if not os.path.exists(save_dir) and parallel.is_rank_zero():
        os.mkdir(save_dir)
    # Wait for processor 0 to finish making the directory.
    parallel.sync()
    
    base_vec = Vec(x_grid, y_grid)
    base_vec.data = N.random.random((nx, ny))
    
    # Create random data and save to disk
    direct_vec_paths = [join(save_dir, 'direct_vec_%02d.txt'%i)
        for i in xrange(num_direct_vecs)]
    adjoint_vec_paths = [join(save_dir, 'adjoint_vec_%02d.txt'%i)
        for i in xrange(num_adjoint_vecs)]
    for path in direct_vec_paths:
        modred.save_mat_text(N.random.random((nx,ny)), path)
    for path in adjoint_vec_paths:
        modred.save_mat_text(N.random.random((nx,ny)), path)
    
    # Create an instance of BPOD.
    my_BPOD = modred.BPOD(VecDefs(x_grid, y_grid, base_vec),
        max_vecs_per_node=20, verbose=verbose)
    
    # Quick check that functions are ok.
    my_BPOD.idiot_check(test_obj_source=direct_vec_paths[0])
    
    my_BPOD.compute_decomp(direct_vec_paths, adjoint_vec_paths)
    
    # Model error less than ~10%
    sing_vals_norm = my_BPOD.sing_vals/N.sum(my_BPOD.sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > 0.9)[0][0] + 1
    
    # Compute the first ``num_modes`` modes, save to file.
    # The "+1"s are because we index modes from 1.
    mode_nums = range(1, num_modes+1)
    direct_mode_paths = [join(save_dir, 'direct_mode_%02d.txt'%i)
        for i in mode_nums]
    adjoint_mode_paths = [join(save_dir, 'adjoint_mode_%02d.txt'%i)
        for i in mode_nums]
    
    my_BPOD.compute_direct_modes(mode_nums, direct_mode_paths)
    my_BPOD.compute_adjoint_modes(mode_nums, adjoint_mode_paths)
    
    # Make plots of leading modes if have matplotlib. 
    # They are meaningless for the random data, of course.
    if make_plots:
        try:
            import matplotlib.pyplot as PLT
            X,Y = N.meshgrid(x_grid, y_grid)
            PLT.figure()
            PLT.contourf(X, Y, modred.load_mat_text(direct_mode_paths[0]).T)
            PLT.colorbar()
            PLT.title('Direct mode 1')
            
            PLT.figure()
            PLT.contourf(X, Y, modred.load_mat_text(adjoint_mode_paths[0]).T)
            PLT.colorbar()
            PLT.title('Adjoint mode 1')
            
            PLT.show()
        except:
            print "Need matplotlib for plots"
    
    # Delete the save_dir with all vec and mode files
    parallel.sync()
    if parallel.is_rank_zero():
        rmtree(save_dir)

if __name__ == '__main__':
    main()
