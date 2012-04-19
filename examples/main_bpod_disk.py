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
In fact, this script can be run in parallel with::

  mpiexec -n 4 python main_bpod_disk.py

This script assumes that modred has been installed or is otherwise
available to be imported.
"""

import os
from os.path import join
from shutil import rmtree
import copy
import numpy as N
import modred as MR
from parallel import default_instance
parallel = default_instance

class Vec(MR.Vector):
    """The vec objects used will be instances of this class"""
    def __init__(self, path=None, x_grid=None, y_grid=None, data=None):
        if x_grid is not None and y_grid is not None:
            self._inner_product = MR.InnerProductTrapz(x_grid, y_grid)
        if path is not None:
            self.load(path)
        else:
            self.data = None
        if data is not None:
            self.data = data
    def save(self, path):
        """Save vec to text format"""
        MR.save_array_text(self.data, path)
    
    def load(self, path):
        """Load vec from text format, still with base vec"""
        self.data = MR.load_array_text(path)
    
    def inner_product(self, other):
        return self._inner_product(self.data, other.data) 
    
    def __mul__(self, a):
        vec_return = copy.deepcopy(self)
        vec_return.data *= a
        return vec_return
    def __add__(self, other):
        vec_return = copy.deepcopy(self)
        vec_return.data += other.data
        return vec_return
    def __eq__(self, other):
        return N.allclose(self.data, other.data)
            
class VecHandle(MR.VecHandle):
    """Vector handle that supports a required grid"""
    x_grid = None
    y_grid = None
    def __init__(self, vec_path, x_grid=None, y_grid=None, base_vec_handle=None,
        scale=1):
        MR.VecHandle.__init__(self, base_vec_handle, scale)
        self.vec_path = vec_path
        if N.array(x_grid != VecHandle.x_grid).any():
            VecHandle.x_grid = x_grid
        if N.array(y_grid != VecHandle.y_grid).any():
            VecHandle.y_grid = y_grid
    def _get(self):
        return Vec(path=self.vec_path, x_grid=VecHandle.x_grid, 
            y_grid=VecHandle.y_grid)    
    def _put(self, vec):
        vec.save(self.vec_path)
        
def inner_product(v1, v2):
    return v1.inner_product(v2)
       
def main(verbose=True, make_plots=True):        
    # Define some parameters
    make_plots = False
    nx = 20
    ny = 30
    x_grid = 1 + N.sin(N.linspace(-N.pi, N.pi, nx))
    y_grid = 1 + N.sin(N.linspace(-N.pi, N.pi, ny))
    X, Y = N.meshgrid(x, y)
    num_direct_vecs = 30
    num_adjoint_vecs = 25
    save_dir = 'DELETE_ME_bpod_example_files'
    
    base_vec_data_path = join(save_dir, 'base_vec.txt')
    base_vec_handle = VecHandle(base_vec_data_path, x_grid, y_grid)
    direct_vec_handles = [VecHandle(join(save_dir, 'direct_vec_%02d.txt'%i),
        x_grid=x_grid, y_grid=y_grid, base_vec_handle=base_vec_handle) 
        for i in xrange(num_direct_vecs)]
    adjoint_vec_handles = [VecHandle(join(save_dir, 'adjoint_vec_%02d.txt'%i),
        x_grid=x_grid, y_grid=y_grid, base_vec_handle=base_vec_handle)
        for i in xrange(num_adjoint_vecs)]
    
    ###
    ### This section creates random data as a placeholder. Typically this
    ### data would already exist and there's no need for these steps.
    ###
    # Create the directory for example files only on processor 0.
    if not os.path.exists(save_dir) and parallel.is_rank_zero():
        os.mkdir(save_dir)
    # Wait for processor 0 to finish making the directory.
    parallel.sync()
    
    base_vec_data = 0.5*N.ones((nx, ny))
    if parallel.is_rank_zero():
        base_vec_handle.put(Vec(data=base_vec_data))
    parallel.sync()
    # Create random data and save to disk
    if parallel.is_rank_zero():
        for handle in direct_vec_handles:
            handle.put(Vec(data=N.random.random((nx,ny))+base_vec_data))
        for handle in adjoint_vec_handles:
            handle.put(Vec(data=N.random.random((nx,ny))+base_vec_data))
    parallel.sync()
    ###
    ### End of data creation section ###
    ###
    
    # Create an instance of BPOD.
    my_BPOD = MR.BPOD(inner_product=inner_product, max_vecs_per_node=20, 
        verbose=verbose)
    
    # Check that functions are ok.
    my_BPOD.sanity_check(direct_vec_handles[0])
    
    L_sing_vecs, sing_vals, R_sing_vecs = my_BPOD.compute_decomp(
        direct_vec_handles, adjoint_vec_handles)
    
    # Model error less than ~10%
    sing_vals_norm = sing_vals/N.sum(sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > 0.9)[0][0] + 1
    
    # Compute the first ``num_modes`` modes, save to file.
    mode_nums = range(num_modes)
    direct_mode_handles = [VecHandle(join(save_dir, 'direct_mode_%02d.txt'%i))
        for i in mode_nums]
    adjoint_mode_handles = [VecHandle(join(save_dir, 'adjoint_mode_%02d.txt'%i))
        for i in mode_nums]
    
    my_BPOD.compute_direct_modes(mode_nums, direct_mode_handles)
    my_BPOD.compute_adjoint_modes(mode_nums, adjoint_mode_handles)
    
    # Make plots of leading modes if have matplotlib. 
    # They are meaningless for the random data, of course.
    if make_plots:
        try:
            import matplotlib.pyplot as PLT
            X,Y = N.meshgrid(x_grid, y_grid)
            PLT.figure()
            PLT.contourf(X, Y, direct_mode_handles[0].get().data.T)
            PLT.colorbar()
            PLT.title('Direct mode 1')
            
            PLT.figure()
            PLT.contourf(X, Y, adjoint_mode_handles[0].get().data.T)
            PLT.colorbar()
            PLT.title('Adjoint mode 1')
            
            PLT.show()
        except:
            print "Need matplotlib for plots"
    
    # Clean up. Delete the save_dir with all vec and mode files
    parallel.sync()
    if parallel.is_rank_zero():
        rmtree(save_dir)

if __name__ == '__main__':
    main()
