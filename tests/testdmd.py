#!/usr/bin/env python

import copy
import unittest
import os
from os.path import join
from shutil import rmtree
import numpy as N

import helper
helper.add_src_to_path()
import parallel as parallel_mod
parallel = parallel_mod.default_instance

from dmd import DMD
from pod import POD
from fieldoperations import FieldOperations
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
        
        self.num_snaps = 6 # number of snapshots to generate
        self.num_states = 7 # dimension of state vector
        self.index_from = 2
        self.DMD = DMD(get_field=util.load_mat_text, put_field=util.\
            save_mat_text, put_mat=util.save_mat_text, inner_product=util.\
            inner_product, verbose=False)
        self.generate_data_set()
        parallel.sync()
        
    
    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            #pass
            rmtree(self.test_dir, ignore_errors=True)
        parallel.sync()
        
        
    def generate_data_set(self):
        """ Create data set of snapshots and save to file"""
        self.snap_path = join(self.test_dir, 'dmd_snap_%03d.txt')
        self.true_mode_path = join(self.test_dir, 'dmd_truemode_%03d.txt')
        self.snap_paths = []
       
        # Generate modes if we are on the first processor
        if parallel.is_rank_zero():
            # A random matrix of data (#cols = #snapshots)
            self.snap_mat = N.mat(N.random.random((self.num_states, self.\
                num_snaps)))
            
            for snap_num in range(self.num_snaps):
                util.save_mat_text(self.snap_mat[:,snap_num], self.snap_path %\
                    snap_num)
                self.snap_paths.append(self.snap_path % snap_num) 
        else:
            self.snap_paths=None
            self.snap_mat = None
        if parallel.is_distributed():
            self.snap_paths = parallel.comm.bcast(self.snap_paths, 
                root=0)
            self.snap_mat = parallel.comm.bcast(self.snap_mat, root=0)

        # Do direct DMD decomposition on all processors
        # This goes against our convention, try to give U, Sigma, and W names!
        # In BPOD, L_sing_vec, sing_vals, R_sing_vecs.
        U, Sigma, W = util.svd(self.snap_mat[:,:-1])
        Sigma_mat = N.mat(N.diag(Sigma))
        self.ritz_vals_true, eig_vecs = N.linalg.eig(U.H * self.snap_mat[:,1:] * \
            W * (Sigma_mat ** -1))
        eig_vecs = N.mat(eig_vecs)
        ritz_vecs = U * eig_vecs
        scaling = N.linalg.lstsq(ritz_vecs, self.snap_mat[:,0])[0]
        scaling = N.mat(N.diag(N.array(scaling).squeeze()))
        self.ritz_vecs_true = ritz_vecs * scaling
        self.build_coeffs_true = W * (Sigma_mat ** -1) * eig_vecs * scaling
        self.mode_norms_true = N.zeros(self.ritz_vecs_true.shape[1])
        for i in xrange(self.ritz_vecs_true.shape[1]):
            self.mode_norms_true[i] = self.DMD.field_ops.inner_product(N.\
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
        
        data_members_default = {'put_mat': util.save_mat_text, 
            'get_mat': util.load_mat_text, 'POD': None,
            'parallel': parallel_mod.default_instance,
            'verbose': False, 'field_ops': FieldOperations(get_field=None, 
            put_field=None, inner_product=None, max_fields_per_node=2,
            verbose=False)}

        self.assertEqual(util.get_data_members(DMD(verbose=False)), \
            data_members_default)

        def my_load(fname): pass
        myDMD = DMD(get_field=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['field_ops'].get_field = my_load
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)
        
        myDMD = DMD(get_mat=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['get_mat'] = my_load
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)

        def my_save(data,fname): pass 
        myDMD = DMD(put_field=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['field_ops'].put_field = my_save
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)
        
        myDMD = DMD(put_mat=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['put_mat'] = my_save
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)
        
        def my_ip(f1, f2): pass
        myDMD = DMD(inner_product=my_ip, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['field_ops'].inner_product = my_ip
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)

        max_fields_per_node = 500
        myDMD = DMD(max_fields_per_node=max_fields_per_node, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['field_ops'].max_fields_per_node =\
            max_fields_per_node
        data_members_modified['field_ops'].max_fields_per_proc =\
            max_fields_per_node * parallel.get_num_nodes() / parallel.\
            get_num_procs()
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)
       
       

    def test_compute_decomp(self):
        """ 
        Only tests the part unique to computing the decomposition.
        """
        # Depending on snapshots generated, test will fail if tol = 8, so use 7
        tol = 1e-7 

        # Run decomposition and save matrices to file
        ritz_vals_path = join(self.test_dir, 'dmd_ritz_vals.txt')
        mode_norms_path = join(self.test_dir, 'dmd_mode_energies.txt')
        build_coeffs_path = join(self.test_dir, 'dmd_build_coeffs.txt')

        self.DMD.compute_decomp(self.snap_paths)
        self.DMD.put_decomp(ritz_vals_path, mode_norms_path, build_coeffs_path)
       
        # Test that matrices were correctly computed
        N.testing.assert_allclose(self.DMD.ritz_vals, 
            self.ritz_vals_true, rtol=tol)
        N.testing.assert_allclose(self.DMD.build_coeffs, 
            self.build_coeffs_true, rtol=tol)
        N.testing.assert_allclose(self.DMD.mode_norms, 
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
        self.DMD.build_coeffs = self.build_coeffs_true
        mode_nums = list(N.array(range(self.num_snaps-1))+self.index_from)
        self.DMD.compute_modes(mode_nums, mode_path, index_from=self.index_from, 
            field_sources=self.snap_paths)
       
        # Load all snapshots into matrix
        if parallel.is_rank_zero():
            mode_mat = N.mat(N.zeros((self.num_states, self.num_snaps-1)), dtype=\
                complex)
            for i in range(self.num_snaps-1):
                mode_mat[:,i] = util.load_mat_text(mode_path % (i+self.index_from),
                    is_complex=True)
        else:
            mode_mat = None
        if parallel.is_distributed():
            mode_mat = parallel.comm.bcast(mode_mat,root=0)
        N.testing.assert_allclose(mode_mat,self.ritz_vecs_true, rtol=\
            tol)

        vandermonde_mat = N.fliplr(N.vander(self.ritz_vals_true, self.num_snaps-1))
        N.testing.assert_allclose(self.snap_mat[:,:-1],
            self.ritz_vecs_true * vandermonde_mat, rtol=tol)

        util.save_mat_text(vandermonde_mat, join(self.test_dir,
            'dmd_vandermonde.txt'))

if __name__=='__main__':
    unittest.main()




