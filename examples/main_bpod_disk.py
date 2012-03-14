#!/usr/bin/env python

"""Example script using BPOD on 2D fields.

This example has most of the complexities we anticipated BPOD to handle, 
and demonstrates a typical usage.

The fields are larger (2D), so loading them 
all into memory simultaneously may be inefficient or impossible.
Instead, they are loaded from disk as needed. 

Further, the inner product is more complicated.
The fields are on a non-uniform grid, and so the inner product uses
trapezoidal rule as an approximation.

The loading of fields is also non-trivial.
In this case, a base field is subtracted from the 2D fields.
In practice this base field might be the equilibrium about which the
governing equations are linearized, and so the saved fields could be
from some simulation software.
The modes, which are saved to disk, do not include the base field.

Another benefit of having the fields saved to disk is BPOD can then be
used in a distributed memory parallel setting.
In fact, this script can be run in parallel with, e.g.::

  mpiexec -n 4 python main_bpod_disk.py

This script assumes that modred has been installed or is otherwise
available to be imported.
"""

import copy
import os
from os.path import join
from shutil import rmtree
import numpy as N
import modred
import modred.util as util

class Field(object):
    """The field objects used will be instances of this class"""
    def __init__(self, path=None):
        if path is not None:
            self.load(path)
        else:
            self.data = None
    
    def save(self, path):
        """Save field to text format"""
        util.save_mat_text(self.data, path)
    
    def load(self, path):
        """Load field from text format, still with base field"""
        self.data = util.load_mat_text(path)
    
    def __mul__(self, a):
        field_return = copy.deepcopy(self)
        field_return.data *= a
        return field_return
    def __rmul__(self, a):
        return self.__mul__(a)
    def __lmul__(self, a):
        return self.__mul__(a)
        
    def __add__(self, other):
        field_return = copy.deepcopy(self)
        field_return.data += other.data
        return field_return
    def __sub__(self, other):
        return self + (-1.*other)
        
def main(verbose=True, make_plots=True):        
    # Define some parameters
    make_plots = False
    nx = 20
    ny = 30
    x_grid = 1 + N.sin(N.linspace(-N.pi, N.pi, nx))
    y_grid = 1 + N.sin(N.linspace(-N.pi, N.pi, ny))
    num_direct_fields = 30
    num_adjoint_fields = 25
    save_dir = join(os.path.dirname(__file__), 'DELETE_ME_bpod_example_files')
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    base_field = Field()
    base_field.data = N.random.random((nx,ny))
    
    # Now create the wrappers for use in the BPOD class.
    def get_field(path):
        """Load the field and remove the base field"""
        field = Field(path)
        return field - base_field
    
    def put_field(field, path):
        """Save the field"""
        field.save(path)
    
    def inner_product(field1, field2):
        return N.trapz(N.trapz(field1.data * field2.data, x=y_grid), 
            x=x_grid) 
    
    # Create random data and save to disk
    direct_field_paths = [join(save_dir, 'direct_field_%02d.txt'%i)
        for i in xrange(num_direct_fields)]
    adjoint_field_paths = [join(save_dir, 'adjoint_field_%02d.txt'%i)
        for i in xrange(num_adjoint_fields)]
    for path in direct_field_paths:
        util.save_mat_text(N.random.random((nx,ny)), path)
    for path in adjoint_field_paths:
        util.save_mat_text(N.random.random((nx,ny)), path)
    
    # Create an instance of BPOD.
    my_BPOD = modred.BPOD(put_field=put_field, get_field=get_field,
        inner_product=inner_product, max_fields_per_node=20, verbose=verbose)
    
    # Quick check that functions are ok.
    # You should always write tests for your get/put_field and inner product
    # functions.
    my_BPOD.idiot_check(test_obj_source=direct_field_paths[0])
    
    # Find the Hankel matrix and take its SVD
    my_BPOD.compute_decomp(direct_field_paths, adjoint_field_paths)
    
    # Want to capture 90%, so:
    sing_vals_norm = my_BPOD.sing_vals/N.sum(my_BPOD.sing_vals)
    num_modes = N.nonzero(N.cumsum(sing_vals_norm) > 0.9)[0][0] + 1
    
    # Compute the first ``num_modes`` modes, save to file.
    # The "+1"s are because we index modes from 1.
    direct_mode_paths = [join(save_dir, 'direct_mode_%02d.txt'%i)
        for i in range(1,num_modes+1)]
    adjoint_mode_paths = [join(save_dir, 'adjoint_mode_%02d.txt'%i)
        for i in range(1,num_modes+1)]
    
    my_BPOD.compute_direct_modes(range(1, num_modes+1), direct_mode_paths)
    my_BPOD.compute_adjoint_modes(range(1, num_modes+1), adjoint_mode_paths)
    
    # Make plots of leading modes if have matplotlib. 
    # They are meaningless for the random data, of course.
    if make_plots:
        try:
            import matplotlib.pyplot as PLT
            X,Y = N.meshgrid(x_grid, y_grid)
            PLT.figure()
            PLT.contourf(X, Y, util.load_mat_text(direct_mode_paths[0]).T)
            PLT.colorbar()
            PLT.title('Direct mode 1')
            
            PLT.figure()
            PLT.contourf(X, Y, util.load_mat_text(adjoint_mode_paths[0]).T)
            PLT.colorbar()
            PLT.title('Adjoint mode 1')
            
            PLT.show()
        except:
            pass
    
    # Delete the save_dir with all field and mode files
    rmtree(save_dir)

if __name__ == '__main__':
    main()