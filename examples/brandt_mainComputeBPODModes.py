#!/usr/bin/env python
#compute the BPOD modes using the modaldecomp library

import os
import subprocess as SP
import sys
sys.path.append('../modaldecomp')
import argparse
import util
import myutils
import field as F
import numpy as N
import bpod as BP

class Args(object):
  def __init__(self):
    self.snapBaseDirectory = '../shervin09/snapshots_dt2.5/'
    #self.snapBaseDirectory = '/work/01225/bbelson/shervin09/snapshots_dt2.5/'
    self.directSnapDirectories = ['B1_snapshots', 'B2_snapshots']
    self.adjointSnapDirectories = ['C1_snapshots','C2_snapshots_350']
    #self.romDirectory = '/work/01225/bbelson/shervin09/CL_rom_C2_350_noICs/rom/'
    self.romDirectory = '../shervin09/CL_rom_C2_350/rom/'
    self.directModeName = 'bpod_direct_mode%03d.u'
    self.adjointModeName = 'bpod_adjoint_mode%03d.u'
    self.tStart = 0.
    self.tEnd = 4000.
    self.dt = 2.5
    self.numModes = 100
    self.IC = False
    
    # error checks
    self.snapBaseDirectory=myutils.addSlash(self.snapBaseDirectory)
    for i in range(len(self.directSnapDirectories)):
      self.directSnapDirectories[i] = myutils.addSlash(self.directSnapDirectories[i])
      
    for i in range(len(self.adjointSnapDirectories)):
      self.adjointSnapDirectories[i] = myutils.addSlash(self.adjointSnapDirectories[i])
    
    self.romDirectory=myutils.addSlash(self.romDirectory)
    
    if self.numModes < 0 or self.numModes > 500:
      raise ValueError('You asked to calculate '+str(self.numModes)+' modes')  
    if self.tStart < self.dt:
      self.tStart = self.dt # the IC is taken care of separately
    if self.tStart > self.tEnd:
      raise ValueError('tStart is greater than tEnd '+str(self.tStart)+
        ' '+str(self.tEnd))
        

def main():
  """
  if args is None:
    parser = argparse.ArgumentParser(description='Compute BPOD modes from snapshots')
    
    parser.add_argument('--snapBaseDir', required=False, default='../shervin09/snapshots_dt2.5/',
      help='Path to directory which contains the impulse response snapshot directories')
    parser.add_argument('--directSnapDirs', required=False, nargs='+',
      default = ['B1_snapshots_nz1', 'B2_snapshots_nz1'],
      help='Directories containing the direct impulse response snapshots')
    parser.add_argument('--adjointSnapDirs', required=False, nargs='+', 
      default = ['C1_snapshots_nz1','C2_snapshots_nz1'],
      help='Directories containing the adjoint impulse response snapshots')
    parser.add_argument('--romDir', required=False,
      default = '../shervin09/CL_rom_C2_300_test/',
      help='Directory to which the decomposition matrices and modes are saved')
    parser.add_argument('--tStart', required=False, type=float, default=0., 
      help='Start time of the snapshots')
    parser.add_argument('--dt', required=False, type=float, default=2.5, 
      help='time step of the snapshots')
    parser.add_argument('--tEnd', required=False, type=float, default=4000., 
      help='End time of the snapshots')
    parser.add_argument('--numModes', required=True, type=int, 
      help='Number of modes to compute')
    parser.add_argument('--IC',required=False,type=int, default=1, 
      help='Use ICs? True is 1, False is 0')
    args = parser.parse_args()
 
  # Error checking
  args.snapBaseDir=myutils.addSlash(args.snapBaseDir)
  for i in range(len(args.directSnapDirs)):
    args.directSnapDirs[i] = myutils.addSlash(args.directSnapDirs[i])
  for i in range(len(args.adjointSnapDirs)):
    args.adjointSnapDirs[i] = myutils.addSlash(args.adjointSnapDirs[i])
  args.romDir=myutils.addSlash(args.romDir)
  if args.numModes < 0 or args.numModes > 500:
    raise ValueError('You asked to calculate '+str(args.numModes)+' modes')  
  if args.tStart < args.dt:
    args.tStart = args.dt # the IC is taken care of separately
  if args.tStart > args.tEnd:
    raise ValueError('tStart is greater than tEnd '+str(args.tStart)+
      ' '+str(args.tEnd))
  """
  
  myArgs = Args()
  
  if not os.path.isdir(myArgs.romDirectory):
    SP.call(['mkdir','-p',myArgs.romDirectory])
  
  print 'Snapshots will be read from:'
  for p in myArgs.directSnapDirectories:
    print myArgs.snapBaseDirectory+p
  for p in myArgs.adjointSnapDirectories:
    print myArgs.snapBaseDirectory+p

  print 'Modes and decomp matrices will be saved to',myArgs.romDirectory
  
  numInputs = len(myArgs.directSnapDirectories)
  directSnapPaths=[]
  for inputNum in range(numInputs):
    #IC is the input and is the first snapshot
    if myArgs.IC:
      directSnapPaths.append(myArgs.snapBaseDirectory + 
        myArgs.directSnapDirectories[inputNum]+'input.u')
    for t in N.arange(myArgs.tStart,myArgs.tEnd+myArgs.dt*.5,myArgs.dt):
      directSnapPaths.append(myArgs.snapBaseDirectory + 
        myArgs.directSnapDirectories[inputNum]+'t'+str(t)+'.u')
    
  numOutputs = len(myArgs.adjointSnapDirectories)
  adjointSnapPaths=[]
  for outputNum in range(numOutputs):
    #IC is the input and is the first snapshot
    if myArgs.IC:
      adjointSnapPaths.append(myArgs.snapBaseDirectory + 
        myArgs.adjointSnapDirectories[outputNum]+'output.u')
    for t in N.arange(myArgs.tStart,myArgs.tEnd+myArgs.dt*.5,myArgs.dt):
      adjointSnapPaths.append(myArgs.snapBaseDirectory + 
        myArgs.adjointSnapDirectories[outputNum]+'t-'+str(t)+'.u')
 
  def load_snap(snapPath):
    f = F.Field()
    f.readBlaNoShift(snapPath)
    #f = F.Field(dataFile = snapPath)
    return f
  
  def save_mode(mode, modePath):
    #shift now since haven't shifted during read. Really this should be
    # changed in the Field class so that it is possible to read and write
    # without after shifting. Currently writeBla always shifts and scales
    # the data, meaning it assumes it was converted to BL scaling and 
    # had the fringe moved to the end of the domain.
    mode.shiftAfterRead()
    mode.params['t'] = 0.
    mode.write(modePath)
  
  def inner_product(field1, field2):
    return field1.innerProduct(field2)
    
  bpod = BP.BPOD(load_snap=load_snap, save_mode=save_mode, 
                  maxSnapsInMem=100, inner_product=inner_product,
                 directSnapPaths=directSnapPaths,
                 adjointSnapPaths=adjointSnapPaths,verbose=True)
  hankelMatPath = myArgs.romDirectory + 'hankelMat.txt'
  LSingVecsPath = myArgs.romDirectory + 'LSingVecs.txt'
  RSingVecsPath = myArgs.romDirectory + 'RSingVecs.txt'
  singValsPath = myArgs.romDirectory + 'singVals.txt'
  

  bpod.compute_decomp(hankelMatPath=hankelMatPath, LSingVecsPath=LSingVecsPath, 
      singValsPath=singValsPath, RSingVecsPath=RSingVecsPath)
   
  # If already computed the decomp matrices and saved them, use this instead
  #bpod.load_decomp(LSingVecsPath, singValsPath, RSingVecsPath)
  
  modeNumList = range(1,myArgs.numModes+1)
  
  print 'Computing the direct modes'
  directModePath = myArgs.romDirectory + myArgs.directModeName
  bpod.compute_direct_modes(modeNumList, directModePath, directSnapPaths = 
    directSnapPaths)
  
  print 'Computing the adjoint modes'
  adjointModePath = myArgs.romDirectory + myArgs.adjointModeName
  bpod.compute_adjoint_modes(modeNumList, adjointModePath,
    adjointSnapPaths=adjointSnapPaths)               
                 
  print 'Done computing all modes, saved to directory',myArgs.romDirectory
  
if __name__ == '__main__':
  main()
  
  