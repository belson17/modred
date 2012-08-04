#!/usr/bin/env python
"""Test dmd module"""

import copy
import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path(join(join(os.path.dirname(os.path.abspath(__file__)), 
    '..', 'src')))
import parallel as parallel_mod
_parallel = parallel_mod.parallel_default_instance

from dmd import DMD
from pod import POD
from vectorspace import VectorSpace
import vectors as V
import util


class TestDMD(unittest.TestCase):
    """Test all the DMD class methods 
    
    Since most of the computations of DMD are done by POD methods
    currently, there should be less to test here"""
    
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir = 'DELETE_ME_test_files_dmd'
        if not os.path.isdir(self.test_dir) and _parallel.is_rank_zero():
            os.mkdir(self.test_dir)
        
        self.num_vecs = 6
        self.num_states = 12
        self.index_from = 2

        self.my_DMD = DMD(N.vdot, verbosity=0)
        self.my_DMD_in_memory = DMD(N.vdot, verbosity=0)
        self.generate_data_set()
        _parallel.barrier()
        
    
    def tearDown(self):
        _parallel.barrier()
        if _parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        _parallel.barrier()
        
        
    def generate_data_set(self):
        """ Create data set of vecs and save to file"""
        self.vec_path = join(self.test_dir, 'dmd_vec_%03d.pkl')
        self.true_mode_path = join(self.test_dir, 'dmd_truemode_%03d.pkl')
        self.vec_handles = [V.PickleVecHandle(self.vec_path%i) 
            for i in range(self.num_vecs)]
       
        # Generate vecs if we are on the first processor
        # A random matrix of data (#cols = #vecs)
        self.vec_array = _parallel.call_and_bcast(N.random.random, 
            ((self.num_states, self.num_vecs)))
        if _parallel.is_rank_zero():
            for vec_index, handle in enumerate(self.vec_handles):
                handle.put(N.array(self.vec_array[:, vec_index]).squeeze())
        self.vecs = [self.vec_array[:, i] for i in range(self.num_vecs)]

        # Do direct DMD decomposition on all processors
        # This goes against our convention, try to give U, Sigma, and W names!
        # In BPOD, L_sing_vec, sing_vals, R_sing_vecs.
        U, Sigma, W = util.svd(self.vec_array[:, :-1])
        Sigma_mat = N.mat(N.diag(Sigma))
        self.ritz_vals_true, eig_vecs = N.linalg.eig(U.H * 
            self.vec_array[:,1:] * W * (Sigma_mat ** -1))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U * eig_vecs
        scaling = N.linalg.lstsq(ritz_vecs, self.vec_array[:, 0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        self.ritz_vecs_true = ritz_vecs * scaling
        self.build_coeffs_true = W * (Sigma_mat ** -1) * eig_vecs * scaling
        self.mode_norms_true = N.zeros(self.ritz_vecs_true.shape[1])
        for i in xrange(self.ritz_vecs_true.shape[1]):
            self.mode_norms_true[i] = self.my_DMD.vec_space.inner_product(
                N.array(self.ritz_vecs_true[:, i]), 
                N.array(self.ritz_vecs_true[:, i])).real

        # Generate modes if we are on the first processor
        if _parallel.is_rank_zero():
            for i in xrange(self.ritz_vecs_true.shape[1]):
                V.PickleVecHandle(self.true_mode_path%(i+1)).put(
                    self.ritz_vecs_true[:, i])



    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        
        def my_load(fname): pass
        def my_save(data, fname): pass
        def my_IP(vec1, vec2): pass
        
        data_members_default = {'put_mat': util.save_array_text, 'get_mat':
             util.load_array_text,
            'verbosity': 0, 'ritz_vals': None, 'build_coeffs': None,
            'mode_norms': None, 'vec_handles': None, 'vecs': None, 
            'vec_space': VectorSpace(my_IP, verbosity=False)}
        
        # Get default data member values
        # Set verbosity to false, to avoid printing warnings during tests
        self.assertEqual(util.get_data_members(DMD(inner_product=my_IP, 
            verbosity=0)), data_members_default)
        
        
        my_DMD = DMD(my_IP, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'] = VectorSpace(inner_product=my_IP, 
            verbosity=0)
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
       
        my_DMD = DMD(my_IP, get_mat=my_load, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
 
        my_DMD = DMD(my_IP, put_mat=my_save, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
        
        max_vecs_per_node = 500
        my_DMD = DMD(my_IP, max_vecs_per_node=max_vecs_per_node, verbosity=0)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_space'].max_vecs_per_node = \
            max_vecs_per_node
        data_members_modified['vec_space'].max_vecs_per_proc = \
            max_vecs_per_node * _parallel.get_num_nodes() / \
            _parallel.get_num_procs()
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
       

    def test_compute_decomp(self):
        """ 
        Only tests the part unique to computing the decomposition.
        """
        # Depending on vecs generated, test will fail if tol = 8, so use 7
        tol = 1e-7 

        # Run decomposition and save matrices to file
        ritz_vals_path = join(self.test_dir, 'dmd_ritz_vals.txt')
        mode_norms_path = join(self.test_dir, 'dmd_mode_energies.txt')
        build_coeffs_path = join(self.test_dir, 'dmd_build_coeffs.txt')

        ritz_vals_returned, mode_norms_returned, build_coeffs_returned = \
            self.my_DMD.compute_decomp(self.vec_handles)
        ritz_vals_returned2, mode_norms_returned2, build_coeffs_returned2 = \
            self.my_DMD_in_memory.compute_decomp_in_memory(self.vecs)
        N.testing.assert_equal(ritz_vals_returned, ritz_vals_returned2)
        N.testing.assert_equal(mode_norms_returned, mode_norms_returned2)
        N.testing.assert_equal(build_coeffs_returned, build_coeffs_returned2)
        
        self.my_DMD.put_decomp(ritz_vals_path, mode_norms_path, 
            build_coeffs_path)
        _parallel.barrier()
        
        # Test that matrices were correctly computed
        N.testing.assert_allclose(self.my_DMD.ritz_vals, 
            self.ritz_vals_true, rtol=tol)
        N.testing.assert_allclose(self.my_DMD.build_coeffs, 
            self.build_coeffs_true, rtol=tol)
        N.testing.assert_allclose(self.my_DMD.mode_norms, 
            self.mode_norms_true, rtol=tol)

        N.testing.assert_allclose(ritz_vals_returned, 
            self.ritz_vals_true, rtol=tol)
        N.testing.assert_allclose(build_coeffs_returned, 
            self.build_coeffs_true, rtol=tol)
        N.testing.assert_allclose(mode_norms_returned, 
            self.mode_norms_true, rtol=tol)

        # Test that matrices were correctly stored
        ritz_vals_loaded = N.array(util.load_array_text(ritz_vals_path,
            is_complex=True)).squeeze()
        build_coeffs_loaded = util.load_array_text(build_coeffs_path, 
            is_complex=True)
        mode_norms_loaded = N.array(
            util.load_array_text(mode_norms_path).squeeze())
    
        N.testing.assert_allclose(ritz_vals_loaded, self.ritz_vals_true, 
            rtol=tol)
        N.testing.assert_allclose(build_coeffs_loaded, self.build_coeffs_true, 
            rtol=tol)
        N.testing.assert_allclose(mode_norms_loaded, self.mode_norms_true, 
            rtol=tol)

    def test_compute_modes(self):
        """
        Test building of modes, reconstruction formula.
        """
        tol = 1e-7

        mode_path = join(self.test_dir, 'dmd_mode_%03d.pkl')
        self.my_DMD.build_coeffs = self.build_coeffs_true
        mode_nums = list(N.array(range(self.num_vecs-1))+self.index_from)
        self.my_DMD.compute_modes(mode_nums, 
            [V.PickleVecHandle(mode_path%i) for i in mode_nums],
            index_from=self.index_from, vec_handles=self.vec_handles)
        
        self.my_DMD_in_memory.build_coeffs = self.build_coeffs_true
        modes_returned = self.my_DMD_in_memory.compute_modes_in_memory(
            mode_nums, vecs=self.vecs, index_from=self.index_from)
            
        # Load all modes into matrix
        mode_array = N.array(N.zeros((self.num_states, self.num_vecs-1)), 
            dtype=complex)
        for i in range(self.num_vecs-1):
            mode_array[:, i] = V.PickleVecHandle(
                mode_path%(i+self.index_from)).get()
        N.testing.assert_allclose(mode_array, self.ritz_vecs_true, 
            rtol=tol)
        for mode_index in range(len(mode_nums)):
            N.testing.assert_allclose(modes_returned[mode_index].squeeze(),
                N.array(self.ritz_vecs_true[:, mode_index]).squeeze(), rtol=tol)
        
        vandermonde_mat = N.fliplr(N.vander(self.ritz_vals_true, 
            self.num_vecs-1))
        N.testing.assert_allclose(self.vec_array[:, :-1],
            self.ritz_vecs_true * vandermonde_mat, rtol=tol)

        util.save_array_text(vandermonde_mat, join(self.test_dir,
            'dmd_vandermonde.txt'))

if __name__ == '__main__':
    unittest.main()

