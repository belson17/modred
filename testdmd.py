#!/usr/bin/env python
import numpy as N
from dmd import DMD
from pod import POD
from fieldoperations import FieldOperations
import unittest
import util
import subprocess as SP
import os
import copy
import parallel as parallel_mod

parallel = parallel_mod.default_instance

if parallel.is_rank_zero():
    print 'To test fully, remember to do both:'
    print '    1) python testdmd.py'
    print '    2) mpiexec -n <# procs> python testdmd.py\n'

class TestDMD(unittest.TestCase):
    """ Test all the DMD class methods 
    
    Since most of the computations of DMD are done by POD methods
    currently, there should be less to test here"""
    
    def setUp(self):
        if not os.path.isdir('files_modaldecomp_test'):        
            SP.call(['mkdir','files_modaldecomp_test'])
        self.num_snaps = 6 # number of snapshots to generate
        self.num_states = 7 # dimension of state vector
        self.index_from = 2
        self.dmd = DMD(load_field=util.load_mat_text, save_field=util.\
            save_mat_text, save_mat=util.save_mat_text, inner_product=util.\
            inner_product, verbose=False)
        self.generate_data_set()
   
    def generate_data_set(self):
        # create data set (saved to file)
        self.snap_path = 'files_modaldecomp_test/dmd_snap_%03d.txt'
        self.true_mode_path = 'files_modaldecomp_test/dmd_truemode_%03d.txt'
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
        self.build_coeff_true = W * (Sigma_mat ** -1) * eig_vecs * scaling
        self.mode_norms_true = N.zeros(self.ritz_vecs_true.shape[1])
        for i in xrange(self.ritz_vecs_true.shape[1]):
            self.mode_norms_true[i] = self.dmd.field_ops.inner_product(N.\
                array(self.ritz_vecs_true[:,i]), N.array(self.ritz_vecs_true[:, 
                i])).real

        # Generate modes if we are on the first processor
        if parallel.is_rank_zero():
            for i in xrange(self.ritz_vecs_true.shape[1]):
                util.save_mat_text(self.ritz_vecs_true[:,i], self.true_mode_path %\
                    (i+1))

    def tearDown(self):
        parallel.sync()
        if parallel.is_rank_zero():
            SP.call(['rm -rf files_modaldecomp_test/*'],shell=True)
        parallel.sync()

    def test_init(self):
        """Test arguments passed to the constructor are assigned properly"""
        # Get default data member values
        # Set verbose to false, to avoid printing warnings during tests
        
        data_members_default = {'save_mat': util.save_mat_text, 
            'load_mat': util.load_mat_text, 'POD': None,
            'parallel': parallel_mod.default_instance,
            'verbose': False, 'field_ops': FieldOperations(load_field=None, 
            save_field=None, inner_product=None, max_fields_per_node=2,
            verbose=False)}

        self.assertEqual(util.get_data_members(DMD(verbose=False)), \
            data_members_default)

        def my_load(fname): pass
        myDMD = DMD(load_field=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['field_ops'].load_field = my_load
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)
        
        myDMD = DMD(load_mat=my_load, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['load_mat'] = my_load
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)

        def my_save(data,fname): pass 
        myDMD = DMD(save_field=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['field_ops'].save_field = my_save
        self.assertEqual(util.get_data_members(myDMD), data_members_modified)
        
        myDMD = DMD(save_mat=my_save, verbose=False)
        data_members_modified = copy.deepcopy(data_members_default)
        data_members_modified['save_mat'] = my_save
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
        Only tests the part unique to this function.
        """
        # Depending on snapshots generated, test will fail if tol = 8, so use 7
        tol = 7 

        # Run decomposition and save matrices to file
        ritz_vals_path = 'files_modaldecomp_test/dmd_ritz_vals.txt'
        mode_norms_path = 'files_modaldecomp_test/dmd_modeenergies.txt'
        build_coeff_path = 'files_modaldecomp_test/dmd_build_coeff.txt'

        self.dmd.compute_decomp(self.snap_paths)
        self.dmd.save_decomp(ritz_vals_path, mode_norms_path, build_coeff_path)
       
        # Test that matrices were correctly computed
        N.testing.assert_array_almost_equal(self.dmd.ritz_vals, 
            self.ritz_vals_true, decimal=tol)
        N.testing.assert_array_almost_equal(self.dmd.build_coeffs, 
            self.build_coeff_true, decimal=tol)
        N.testing.assert_array_almost_equal(self.dmd.mode_norms, 
            self.mode_norms_true, decimal=tol)

        # Test that matrices were correctly stored
        if parallel.is_rank_zero():
            ritz_vals_loaded = N.array(util.load_mat_text(ritz_vals_path,
                is_complex=True)).squeeze()
            build_coeff_loaded = util.load_mat_text(build_coeff_path, 
                is_complex=True)
            mode_norms_loaded = N.array(util.load_mat_text(mode_norms_path).\
                squeeze())
        else:   
            ritz_vals_loaded = None
            build_coeff_loaded = None
            mode_norms_loaded = None

        if parallel.is_distributed():
            ritz_vals_loaded = parallel.comm.bcast(ritz_vals_loaded, root=0)
            build_coeff_loaded = parallel.comm.bcast(build_coeff_loaded, root=0)
            mode_norms_loaded = parallel.comm.bcast(mode_norms_loaded,
                root=0)

        N.testing.assert_array_almost_equal(ritz_vals_loaded, 
            self.ritz_vals_true, decimal=tol)
        N.testing.assert_array_almost_equal(build_coeff_loaded, 
            self.build_coeff_true, decimal=tol)
        N.testing.assert_array_almost_equal(mode_norms_loaded, 
            self.mode_norms_true, decimal=tol)

    def test_compute_modes(self):
        """
        Test building of modes, reconstruction formula.

        """
        tol = 8

        mode_path ='files_modaldecomp_test/dmd_mode_%03d.txt'
        self.dmd.build_coeffs = self.build_coeff_true
        mode_nums = list(N.array(range(self.num_snaps-1))+self.index_from)
        self.dmd.compute_modes(mode_nums, mode_path, index_from=self.index_from, 
            snap_paths=self.snap_paths)
       
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
        N.testing.assert_array_almost_equal(mode_mat,self.ritz_vecs_true, decimal=\
            tol)

        vandermonde_mat = N.fliplr(N.vander(self.ritz_vals_true, self.num_snaps-1))
        N.testing.assert_array_almost_equal(self.snap_mat[:,:-1],
            self.ritz_vecs_true * vandermonde_mat, decimal=tol)

        util.save_mat_text(vandermonde_mat, 'files_modaldecomp_test/' +\
            'dmd_vandermonde.txt')

if __name__=='__main__':
    unittest.main(verbosity=2)




