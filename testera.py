#!/usr/bin/env python

import subprocess as SP
import era
import util
import numpy as N
import unittest
import copy

class testERA(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = 'files_era_test/'
        self.IO_paths = [self.test_dir+'input1_impulse.txt', \
          self.test_dir+'input2_impulse.txt']
        raw_data = util.load_mat_text(self.IO_paths[0],delimiter=' ')
        output_signals = raw_data[:,1:]
        self.dt = 25.
        self.num_snaps,self.num_outputs = N.shape(output_signals[:-1,:])
        self.num_inputs = len(self.IO_paths)
        
        self.IO_signals_sampled = N.zeros((self.num_outputs, self.num_inputs,self.num_snaps))
        self.IO_signals_advanced_dt = N.zeros((self.num_outputs, self.num_inputs,self.num_snaps))
        self.time = output_signals[:,0]
        for input_index, IO_path in enumerate(self.IO_paths):
            raw_data = util.load_mat_text(IO_path, delimiter=' ')
            output_signals = raw_data[:,1:]
            self.IO_signals_sampled[:,input_index,:] = output_signals[:-1,:].T
            self.IO_signals_advanced_dt[:,input_index,:] = output_signals[1:,:].T
        
        self.hankel_mat_known = \
          N.mat(util.load_mat_text(self.test_dir+'hankel_mat_known.txt'))
        self.hankel_mat2_known = \
          N.mat(util.load_mat_text(self.test_dir+'hankel_mat2_known.txt'))
        self.L_sing_vecs_known = \
          N.mat(util.load_mat_text(self.test_dir+'L_sing_vecs_known.txt'))
        self.sing_vals_known =  \
          N.squeeze(util.load_mat_text(self.test_dir+'sing_vals_known.txt'))
        self.R_sing_vecs_known = \
          N.mat(util.load_mat_text(self.test_dir+'R_sing_vecs_known.txt'))
                
        self.A_path_known = self.test_dir+'A_known.txt'
        self.B_path_known = self.test_dir+'B_known.txt'
        self.C_path_known = self.test_dir+'C_known.txt'
        self.num_states = 50
        self.A_known = util.load_mat_text(self.A_path_known)
        self.B_known = util.load_mat_text(self.B_path_known)
        self.C_known = util.load_mat_text(self.C_path_known)
        
    def tearDown(self):
        """Deletes all of the matrices created by the tests"""
        SP.call(['rm -f '+self.test_dir+'*computed*'], shell=True)
        SP.call(['rm -f '+self.test_dir+'*delete_me*'], shell=True)

    def generate_unequal_dt_data(self, num_inputs, num_outputs, num_snaps, \
      dt_sample, dt_model, t0):
        """Generates data that has unequal dt_model and dt_sample"""
        time = N.array([t0,t0+dt_model])
        for snap_num in range(1,num_snaps):
            time = N.append(time, t0 + snap_num*dt_sample)
            time = N.append(time, t0 + snap_num*dt_sample + dt_model)
        #print 'length of time is',len(time)
        #print 'time is',time
        IO_signals_all = N.random.random((num_outputs, num_inputs, 2*num_snaps))
        IO_signals_sampled = IO_signals_all[:,:,::2]
        IO_signals_advanced_dt = IO_signals_all[:,:,1::2]
        impulse_file_paths = []
        impulse_file_path = self.test_dir+'delete_me_impulse%03d.txt'
        for input_num in range(num_inputs):
            impulse_file_paths.append(impulse_file_path%input_num)
            raw_data = N.concatenate( \
              (time.reshape(len(time),1),IO_signals_all[:,input_num,:].T), axis=1)
            util.save_mat_text(raw_data,impulse_file_path%input_num)
        return IO_signals_sampled, IO_signals_advanced_dt, time, impulse_file_paths


    def test_init(self):
        """Tests the constructor"""
        
        # Should test others
        
    def test_set_impulse_outputs(self):
        """Test setting the impulse data"""
        num_inputsList = [1,3,2]
        num_outputsList = [1,5,3]
        num_snapsList = [1, 100, 21]
        for num_outputs in num_outputsList:
            for num_inputs in num_inputsList:
                for num_snaps in num_snapsList:
                    IO_signals_sampled = N.random.random((num_outputs,num_inputs,num_snaps))%100
                    IO_signals_advanced_dt = N.random.random((num_outputs,num_inputs,num_snaps))%100
                    myERA = era.ERA(dt_model=self.dt, dt_sample=self.dt)
                    myERA.set_impulse_outputs(IO_signals_sampled, IO_signals_advanced_dt)
                    self.assertEqual(myERA.num_outputs, num_outputs)
                    self.assertEqual(myERA.num_inputs, num_inputs)
                    self.assertEqual(myERA.num_snaps, num_snaps)
                    N.testing.assert_array_almost_equal(\
                      myERA.IO_signals_sampled, IO_signals_sampled)
                    N.testing.assert_array_almost_equal(\
                      myERA.IO_signals_advanced_dt, IO_signals_advanced_dt)
    
    
    def test_load_impulse_outputs(self):
        """Test that can load in a list of impulse output signals from file"""
        myERA = era.ERA()
        myERA.load_impulse_outputs(self.IO_paths)
        N.testing.assert_array_almost_equal(myERA.IO_signals_sampled,self.IO_signals_sampled)
        N.testing.assert_array_almost_equal(myERA.IO_signals_advanced_dt,self.IO_signals_advanced_dt)
        
        # Test unequal dt spacing
        num_inputs = 2
        num_outputs = 3
        num_snaps = 10
        dt_sample = 1.5
        dt_model = .1
        t0 = 10.
        
        IO_signals_sampled_true, IO_signals_advanced_dt_true, timeTrue, \
          impulse_file_pathsTrue = \
          self.generate_unequal_dt_data(num_inputs, num_outputs, num_snaps, \
          dt_sample, dt_model, t0)
        
        myERA = era.ERA()
        myERA.load_impulse_outputs(impulse_file_pathsTrue)
        self.assertAlmostEqual(myERA.dt_sample, dt_sample)
        self.assertAlmostEqual(myERA.dt_model, dt_model)
        N.testing.assert_array_almost_equal(\
          myERA.IO_signals_sampled, IO_signals_sampled_true)
        N.testing.assert_array_almost_equal(\
          myERA.IO_signals_advanced_dt, IO_signals_advanced_dt_true)
        
        
    
    def test__compute_hankel(self):
        """Test that with given signals, compute correct hankel matrix"""
        myERA = era.ERA()
        myERA.IO_signals_sampled = self.IO_signals_sampled
        myERA.IO_signals_advanced_dt= self.IO_signals_advanced_dt
        myERA._compute_hankel()
        N.testing.assert_array_almost_equal(myERA.hankel_mat, \
          self.hankel_mat_known[:myERA.hankel_mat.shape[0],:myERA.hankel_mat.shape[1]])
        N.testing.assert_array_almost_equal(myERA.hankel_mat2, \
          self.hankel_mat2_known[:myERA.hankel_mat2.shape[0],:myERA.hankel_mat2.shape[1]])           
        
        
    def test_compute_decomp(self):
        myERA = era.ERA()
        myERA.IO_signals_sampled = self.IO_signals_sampled
        myERA.IO_signals_advanced_dt= self.IO_signals_advanced_dt
        hankel_mat_path = self.test_dir+'hankel_mat_computed.txt'
        hankel_mat2_path = self.test_dir+'hankel_mat2_computed.txt'
        L_sing_vecs_path = self.test_dir+'L_sing_vecs_computed.txt'
        sing_vals_path = self.test_dir+'sing_vals_computed.txt'
        R_sing_vecs_path = self.test_dir+'R_sing_vecs_computed.txt'
        myERA.compute_decomp()
        myERA.save_decomp(hankel_mat_path,
          hankel_mat2_path , L_sing_vecs_path,
          sing_vals_path, R_sing_vecs_path)
        s = myERA.hankel_mat.shape
        N.testing.assert_array_almost_equal(myERA.hankel_mat, \
          self.hankel_mat_known[:s[0],:s[1]])
        N.testing.assert_array_almost_equal(myERA.hankel_mat2, \
          self.hankel_mat2_known[:s[0], :s[1]])
        s = myERA.L_sing_vecs.shape
        
        #print 'max error between L_sing_vecs is', N.amax(N.abs(myERA.L_sing_vecs - self.L_sing_vecs_known[:s[0],:s[1]]))
        N.testing.assert_array_almost_equal(myERA.L_sing_vecs, \
          self.L_sing_vecs_known[:s[0],:s[1]], decimal=5)
          
        s = N.squeeze(myERA.sing_vals).shape
        N.testing.assert_array_almost_equal(N.squeeze(myERA.sing_vals),
          N.squeeze(self.sing_vals_known)[:s[0]])
        s= myERA.R_sing_vecs.shape
        N.testing.assert_array_almost_equal(myERA.R_sing_vecs, \
          self.R_sing_vecs_known[:s[0],:s[1]],decimal=4)
        
        # Load in saved decomp matrices, check they are the same
        hankel_mat_loaded = util.load_mat_text(hankel_mat_path)
        hankel_mat2_loaded = myERA.load_mat(hankel_mat2_path)
        L_sing_vecs_loaded = myERA.load_mat(L_sing_vecs_path)
        R_sing_vecs_loaded = myERA.load_mat(R_sing_vecs_path)
        sing_vals_loaded = myERA.load_mat(sing_vals_path)

        N.testing.assert_array_almost_equal(hankel_mat_loaded, self.hankel_mat_known)
        N.testing.assert_array_almost_equal(hankel_mat2_loaded, self.hankel_mat2_known)
        N.testing.assert_array_almost_equal(L_sing_vecs_loaded, self.L_sing_vecs_known,decimal=5)
        N.testing.assert_array_almost_equal(N.squeeze(sing_vals_loaded), \
          N.squeeze(self.sing_vals_known))
        N.testing.assert_array_almost_equal(R_sing_vecs_loaded,self.R_sing_vecs_known,decimal=4)
    
    
    def test_compute_ROM(self):
        """Test forming the ROM matrices from decomp matrices"""
        myERA = era.ERA(num_states = self.num_states)
        myERA.num_inputs = 2
        myERA.num_outputs = 2
        myERA.hankel_mat = self.hankel_mat_known
        myERA.hankel_mat2 = self.hankel_mat2_known
        myERA.L_sing_vecs = self.L_sing_vecs_known
        myERA.R_sing_vecs = self.R_sing_vecs_known
        myERA.sing_vals = self.sing_vals_known
        
        A_path_computed = self.test_dir+'A_computed.txt'
        B_path_computed = self.test_dir+'B_computed.txt'
        C_path_computed = self.test_dir+'C_computed.txt'
        # Gives an error if there is no time step specified or read from file
        self.assertRaises(util.UndefinedError,myERA.compute_ROM,self.num_states)
        
        myERA.dt_sample = self.dt
        myERA.dt_model = self.dt
        myERA.compute_ROM(self.num_states)
        myERA.save_ROM(A_path_computed, B_path_computed, C_path_computed)
        
        N.testing.assert_array_almost_equal(myERA.A, self.A_known)
        N.testing.assert_array_almost_equal(myERA.B, self.B_known)
        N.testing.assert_array_almost_equal(myERA.C, self.C_known)
        
        A_loaded = myERA.load_mat(A_path_computed)
        B_loaded = myERA.load_mat(B_path_computed)
        C_loaded = myERA.load_mat(C_path_computed)
        
        N.testing.assert_array_almost_equal(A_loaded, self.A_known)
        N.testing.assert_array_almost_equal(B_loaded, self.B_known)
        N.testing.assert_array_almost_equal(C_loaded, self.C_known)

    def test_save_load_decomp(self):
        """Test that properly saves and loads decomp matrices"""
        myERA = era.ERA(num_states = self.num_states)
        myERA.hankel_mat = copy.deepcopy(self.hankel_mat_known)
        myERA.hankel_mat2 = copy.deepcopy(self.hankel_mat2_known)
        myERA.L_sing_vecs = copy.deepcopy(self.L_sing_vecs_known)
        myERA.sing_vals = copy.deepcopy(self.sing_vals_known)
        myERA.R_sing_vecs = copy.deepcopy(self.R_sing_vecs_known)
        
        hankel_mat_path = self.test_dir+'hankel_mat_computed.txt'
        hankel_mat2_path = self.test_dir+'hankel_mat2_computed.txt'
        L_sing_vecs_path = self.test_dir+'L_sing_vecs_computed.txt'
        sing_vals_path = self.test_dir+'sing_vals_computed.txt'
        R_sing_vecs_path = self.test_dir+'R_sing_vecs_computed.txt'
        
        myERA.save_decomp(hankel_mat_path,
            hankel_mat2_path, L_sing_vecs_path,
            sing_vals_path , R_sing_vecs_path)
          
        myERA.load_decomp(hankel_mat_path,
            hankel_mat2_path , L_sing_vecs_path,
            sing_vals_path, R_sing_vecs_path,
            num_inputs = self.num_inputs, num_outputs=self.num_outputs)
        
        N.testing.assert_array_almost_equal(myERA.hankel_mat, self.hankel_mat_known)
        N.testing.assert_array_almost_equal(myERA.hankel_mat2, self.hankel_mat2_known)
        N.testing.assert_array_almost_equal(myERA.L_sing_vecs, self.L_sing_vecs_known)
        N.testing.assert_array_almost_equal(N.squeeze(myERA.sing_vals),
          N.squeeze(self.sing_vals_known))
        N.testing.assert_array_almost_equal(myERA.R_sing_vecs, self.R_sing_vecs_known)
        
        """
        # Old test, used to allow only loading the Hankel matrices and not 
        # the SVD matrices. Now only the entire set can be loaded. Might
        # include this feature again in the future, but hardly used.
        myERANoSVD = era.ERA()
        myERANoSVD.load_decomp(hankel_mat_path,
          hankel_mat2_path,num_inputs = self.num_inputs, 
          num_outputs=self.num_outputs)
        
        N.testing.assert_array_almost_equal(myERANoSVD.hankel_mat,self.hankel_mat_known)
        N.testing.assert_array_almost_equal(myERANoSVD.hankel_mat2,self.hankel_mat2_known)
        N.testing.assert_array_almost_equal(myERANoSVD.L_sing_vecs,self.L_sing_vecs_known)
        N.testing.assert_array_almost_equal(N.squeeze(myERANoSVD.sing_vals),
          N.squeeze(self.sing_vals_known))
        N.testing.assert_array_almost_equal(myERANoSVD.R_sing_vecs,self.R_sing_vecs_known)     
        """
        
        
    def test_save_ROM(self):
        """Test can save ROM matrices A,B,C correctly"""
        myERA = era.ERA()
        myERA.A = copy.deepcopy(self.A_known)
        myERA.B = copy.deepcopy(self.B_known)
        myERA.C = copy.deepcopy(self.C_known)
        A_path_computed = self.test_dir+'A_computed.txt'
        B_path_computed = self.test_dir+'B_computed.txt'
        C_path_computed = self.test_dir+'C_computed.txt'
        myERA.save_ROM(A_path_computed,B_path_computed,C_path_computed)
        
        A_loaded = myERA.load_mat(A_path_computed)
        B_loaded = myERA.load_mat(B_path_computed)
        C_loaded = myERA.load_mat(C_path_computed)
        
        N.testing.assert_array_almost_equal(A_loaded,self.A_known)
        N.testing.assert_array_almost_equal(B_loaded,self.B_known)
        N.testing.assert_array_almost_equal(C_loaded,self.C_known)

if __name__ =='__main__':
    unittest.main(verbosity=2)
        
        
    


