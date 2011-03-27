#!/usr/bin/env python

"""
Computes the ERA LTI ROM from input-output impulse response signals.
"""
import sys
import os
import myutils
import era

class Args(object):
  """Contains all parameters that change for individual cases"""
  def __init__(self):
    xPosition = '300'
    caseName = xPosition+'_ERA_disc'
    # the directory that has the impulse response data
    inputOutputDirectory = '../shervin09/snapshots_dt2.5/C2output_relocated/'
    # A list of the files which have the impulse response data
    # The directory where all ROM results will be saved (decomposition, matrices)
    baseDirectory = '../shervin09/CL_rom_C2_'+xPosition+'/rom/'
    baseDirectory = myutils.addSlash(baseDirectory)
    
    self.IOPaths = [inputOutputDirectory+'B1ToOutputs_'+xPosition+'.txt', \
      inputOutputDirectory+'B2ToOutputs_'+xPosition+'.txt']
    
    self.LSingVecsPath = baseDirectory + 'LSingVecs_'+caseName+'.txt'
    self.singValsPath = baseDirectory + 'singVals_' + caseName+'.txt'
    self.RSingVecsPath = baseDirectory + 'RSingVecs_'+caseName+'.txt'
    self.hankelMatPath = baseDirectory + 'hankelMat_'+caseName+'.txt'
    self.hankelMat2Path = baseDirectory + 'hankelMat2_'+caseName+'.txt'
    self.APath = baseDirectory+'A_'+caseName+'.txt'
    self.BPath = baseDirectory+'B_'+caseName+'.txt'
    self.CPath = baseDirectory+'C_'+caseName+'.txt'
    self.numStates = 100
    self.dt = 2.5
    self.check()
    
  def check(self):
    """Check the arguments make sense"""
    if self.numStates > 500 or self.numStates < 1:
      raise ValueError('You asked for a ROM with '+str(self.numStates)+' states')
    if self.dt < 0:
      raise ValueError('You gave a time step of '+str(self.dt))
      
    
def main():
  myArgs = Args()
  if len(sys.argv) == 2:
    myArgs.numStates = int(sys.argv[1])
  elif len(sys.argv) > 2:
    raise RuntimeError('Give only the number of states <numStates>')
  
  myERA = era.ERA(hankelMatPath=myArgs.hankelMatPath,
    hankelMat2Path=myArgs.hankelMat2Path,
    LSingVecsPath=myArgs.LSingVecsPath,
    singValsPath = myArgs.singValsPath,
    RSingVecsPath=myArgs.RSingVecsPath)
  print 'Done constructor, reading the impulse output signals' 
  myERA.load_impulse_outputs(myArgs.IOPaths)
  print 'Done loading impulse response data, computing SVD'
  myERA.compute_decomp()    
  print 'Done SVD, forming the ROM matrices'  
  myERA.compute_ROM(numStates=myArgs.numStates,APath=myArgs.APath,
    BPath = myArgs.BPath, CPath = myArgs.CPath)

if __name__=='__main__':
  main()
  


