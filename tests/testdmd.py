#!/usr/bin/env python

import copy
import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_to_path('src')
import parallel as parallel_mod
parallel = parallel_mod.default_instance

from dmd import DMD
from pod import POD
from vecoperations import VecOperations
from vecdefs import ArrayText
import util


class TestDMD(unittest.TestCase):
    """ Test all the DMD class methods 
    
    Since most of the computations of DMD are done by POD methods
    currently, there should be less to test here"""
    
    def setUp(self):
        if not os.access('.', os.W_OK):
            raise RuntimeError('Cannot write to current directory')
        self.test_dir ='DELETE_ME_test_files_dmd'
        if not os.path.isdir(self.test_dir) and parallel.is_rank_zero():
            os.mkdir(self.test_dir)
        
        self.num_vecs = 6 
        self.num_states = 7 
        self.index_from = 2

        self.my_vec_defs = ArrayText()
        self.my_DMD = DMD(self.my_vec_defs, 
            put_mat=util.save_mat_text, verbose=False)

        self.generate_data_set()
        parallel.sync()
        
    
    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            rmtree(self.test_dir, ignore_errors=True)
        parallel.sync()
        
        
    def generate_data_set(self):
        """ Create data set of vecs and save to file"""
        self.vec_path = join(self.test_dir, 'dmd_vec_%03d.txt')
        self.true_mode_path = join(self.test_dir, 'dmd_truemode_%03d.txt')
        self.vec_paths = []
       
        # Generate modes if we are on the first processor
        if parallel.is_rank_zero():
            # A random matrix of data (#cols = #vecs)
            self.vec_mat = N.mat(N.random.random((self.num_states, self.\
                num_vecs)))
            
            for vec_num in range(self.num_vecs):
                util.save_mat_text(self.vec_mat[:,vec_num], self.vec_path %\
                    vec_num)
                self.vec_paths.append(self.vec_path % vec_num) 
        else:
            self.vec_paths=None
            self.vec_mat = None
        if parallel.is_distributed():
            self.vec_paths = parallel.comm.bcast(self.vec_paths, 
                root=0)
            self.vec_mat = parallel.comm.bcast(self.vec_mat, root=0)

        # Do direct DMD decomposition on all processors
        # This goes against our convention, try to give U, Sigma, and W names!
        # In BPOD, L_sing_vec, sing_vals, R_sing_vecs.
        U, Sigma, W = util.svd(self.vec_mat[:,:-1])
        Sigma_mat = N.mat(N.diag(Sigma))
        self.ritz_vals_true, eig_vecs = N.linalg.eig(U.H * self.vec_mat[:,1:] * \
            W * (Sigma_mat ** -1))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U * eig_vecs
        scaling = N.linalg.lstsq(ritz_vecs, self.vec_mat[:,0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        self.ritz_vecs_true = ritz_vecs * scaling
        self.build_coeffs_true = W * (Sigma_mat ** -1) * eig_vecs * scaling
        self.mode_norms_true = N.zeros(self.ritz_vecs_true.shape[1])
        for i in xrange(self.ritz_vecs_true.shape[1]):
            self.mode_norms_true[i] = self.my_vec_defs.inner_product(N.\
                array(self.ritz_vecs_true[:,i]), N.array(self.ritz_vecs_true[:, 
                i])).real

        # Generate modes if we are on the first processor
        if parallel.is_rank_zero():
            for i in xrange(self.ritz_vecs_true.shape[1]):
                util.save_mat_text(self.ritz_vecs_true[:,i], self.true_mode_path %\
                    (i+1))




    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        
        data_members_default = {'put_mat': util.save_mat_text, 'get_mat':
             util.load_mat_text, 'parallel': parallel_mod.default_instance,
            'verbose': False, 'ritz_vals': None, 'build_coeffs': None,
            'mode_norms': None, 'vec_sources': None, 'POD': None,
            'vec_ops': VecOperations(self.my_vec_defs, verbose=False)}
        
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        self.assertEqual(util.get_data_members(DMD(self.my_vec_defs, 
            verbose=False)), data_members_default)
        
        def my_load(fname): pass
        def my_save(data, fname): pass 
        
        my_DMD = DMD(self.my_vec_defs, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'] = VecOperations(self.my_vec_defs,
            verbose=False)
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
       
        my_DMD = DMD(self.my_vec_defs, get_mat=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
 
        my_DMD = DMD(self.my_vec_defs, put_mat=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(my_DMD), data_members_modified)
        
        max_vecs_per_node = 500
        my_DMD = DMD(self.my_vec_defs, max_vecs_per_node=max_vecs_per_node,
            verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['vec_ops'].max_vecs_per_node =\
            max_vecs_per_node
        data_members_modified['vec_ops'].max_vecs_per_proc =\
            max_vecs_per_node * parallel.get_num_nodes() / parallel.\
            get_num_procs()
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

        self.my_DMD.compute_decomp(self.vec_paths, ritz_vals_path, 
            mode_norms_path, build_coeffs_path)
        
        ritz_vals_returned, mode_norms_returned, build_coeffs_returned = \
            self.my_DMD.compute_decomp_and_return(self.vec_paths)
        
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
        if parallel.is_rank_zero():
            ritz_vals_loaded = N.array(util.load_mat_text(ritz_vals_path,
                is_complex=True)).squeeze()
            build_coeffs_loaded = util.load_mat_text(build_coeffs_path, 
                is_complex=True)
            mode_norms_loaded = N.array(util.load_mat_text(mode_norms_path).\
                squeeze())
        else:   
            ritz_vals_loaded = None
            build_coeffs_loaded = None
            mode_norms_loaded = None

        if parallel.is_distributed():
            ritz_vals_loaded = parallel.comm.bcast(ritz_vals_loaded, root=0)
            build_coeffs_loaded = parallel.comm.bcast(build_coeffs_loaded, root=0)
            mode_norms_loaded = parallel.comm.bcast(mode_norms_loaded,
                root=0)

        N.testing.assert_allclose(ritz_vals_loaded, 
            self.ritz_vals_true, rtol=tol)
        N.testing.assert_allclose(build_coeffs_loaded, 
            self.build_coeffs_true, rtol=tol)
        N.testing.assert_allclose(mode_norms_loaded, 
            self.mode_norms_true, rtol=tol)

    def test_compute_modes(self):
        """
        Test building of modes, reconstruction formula.
        """
        tol = 1e-8

        mode_path = join(self.test_dir, 'dmd_mode_%03d.txt')
        self.my_DMD.build_coeffs = self.build_coeffs_true
        mode_nums = list(N.array(range(self.num_vecs-1))+self.index_from)
        self.my_DMD.compute_modes(mode_nums, 
            [mode_path%i for i in mode_nums],
            index_from=self.index_from, vec_sources=self.vec_paths)
            
        # Load all vecs into matrix
        if parallel.is_rank_zero():
            mode_mat = N.mat(N.zeros((self.num_states, self.num_vecs-1)), dtype=\
                complex)
            for i in range(self.num_vecs-1):
                mode_mat[:,i] = util.load_mat_text(mode_path % (i+self.index_from),
                    is_complex=True)
        else:
            mode_mat = None
        if parallel.is_distributed():
            mode_mat = parallel.comm.bcast(mode_mat,root=0)
        N.testing.assert_allclose(mode_mat,self.ritz_vecs_true, rtol=\
            tol)

        vandermonde_mat = N.fliplr(N.vander(self.ritz_vals_true, self.num_vecs-1))
        N.testing.assert_allclose(self.vec_mat[:,:-1],
            self.ritz_vecs_true * vandermonde_mat, rtol=tol)

        util.save_mat_text(vandermonde_mat, join(self.test_dir,
            'dmd_vandermonde.txt'))

if __name__=='__main__':
    unittest.main()




