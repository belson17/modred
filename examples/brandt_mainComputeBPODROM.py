#!/usr/bin/env python
""" Computes the BPOD ROM matrices. It requires that the modes are already
computed, and that the direct modes are advanced 1 small time step forward
to calculate the derivatives. """

import os
import subprocess as SP
import sys
sys.path.append('../modaldecomp')
import util
import myutils
import field as F
import numpy as N
import bpodltirom

class Args(object):
  """ Class which contains parameters needed by main function."""
  def __init__(self):
    xPosition = '350'
    #modeDirectory must contain bla.i input file and the direct modes
    self.modeDirectory = '../shervin09/CL_rom_C2_'+xPosition+'/rom/modes/'
    #modeDtDirectory is where the advanced modes will be saved to
    self.modeDtDirectory='../shervin09/CL_rom_C2_'+xPosition+'/rom/modes_dt/'
    # The ROM matrices will be saved to this file
    self.romDirectory = '../shervin09/CL_rom_C2_'+xPosition+'/rom/'
    
    self.snapDirectory = '../shervin09/snapshots_dt2.5/'
    self.inputPaths = [self.snapDirectory+'/B1_snapshots/input.u',
      self.snapDirectory+'/B2_snapshots/input.u']
    self.outputPaths = [self.snapDirectory+'/C1_snapshots/output.u',
      self.snapDirectory+'C2output_relocated/C2output_'+xPosition+'.u']
    
    # ROM matrices will contain this in their name
    # Should contain sensor location, ERA v BPOD, and continuous or discrete
    # time.
    self.caseName = 'C2_'+xPosition+'_BPOD_cont'
    
    self.numModes = 100
    self.dt = 0.0001
        
    #name of the direct modes, must contain %03d type format to insert
    #the mode number
    self.directModeName = 'bpod_direct_mode%03d.u'
    self.directDtModeName = 'bpod_direct_mode%03d_dt.u'
    self.directDerivModeName = 'bpod_direct_mode%03d_deriv.u'
    self.adjointModeName = 'bpod_adjoint_mode%03d.u'
  
    # Error checking
    self.modeDirectory = myutils.addSlash(self.modeDirectory)
    self.modeDtDirectory = myutils.addSlash(self.modeDtDirectory)
    self.romDirectory = myutils.addSlash(self.romDirectory)
    self.snapDirectory = myutils.addSlash(self.snapDirectory) 

    
def main():
  myArgs = Args()
  if len(sys.argv) ==2:
    myArgs.numModes = int(sys.argv[1])    
  elif len(sys.argv)>2:
    raise RuntimeError('Only give argument <numModes>')
  
  directModePaths = []
  directDerivModePaths = []
  directDtModePaths = []
  adjointModePaths = []
  for modeNum in range(1,myArgs.numModes+1):
    directModePaths.append(myArgs.modeDirectory+myArgs.directModeName%modeNum)
    directDerivModePaths.append(myArgs.modeDtDirectory + 
      myArgs.directDerivModeName%modeNum)
    directDtModePaths.append(myArgs.modeDtDirectory + 
      myArgs.directDtModeName%modeNum)
    adjointModePaths.append(myArgs.modeDirectory + myArgs.adjointModeName%modeNum)
  
  def load_mode(filename):
    return F.Field(dataFile = filename)
  
  def save_mode(mode, filename):
    mode.write(filename)
  
  def inner_product(f1, f2):
    return f1.innerProduct(f2)
  
    
  bpodrom = bpodltirom.BPODROM(adjointModePaths=adjointModePaths,
    directModePaths=directModePaths,
    directDerivModePaths=directDerivModePaths,
    inner_product=inner_product,
    load_mode=load_mode,save_mode=save_mode,
    numModes=myArgs.numModes,maxSnapsInMem=200)
    
  bpodrom.compute_mode_derivs(directModePaths,directDtModePaths,
    directDerivModePaths,myArgs.dt)
  
  bpodrom.form_A(myArgs.romDirectory+'A_'+myArgs.caseName+'.txt')
  bpodrom.form_B(myArgs.romDirectory+'B_'+myArgs.caseName+'.txt',myArgs.inputPaths)
  bpodrom.form_C(myArgs.romDirectory+'C_'+myArgs.caseName+'.txt',myArgs.outputPaths)
  
  

if __name__ == '__main__':
  main()





