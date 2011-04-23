
import sys  
import copy
import util
import numpy as N


class ModalDecomp(object):
    """
    Modal Decomposition base class

    This parent class contains implementations of algorithms that
    take as input a set of data (snapshots) and return as output sets of
    modes.  Derived classes should exist for unique modal decompositions, 
    e.g. POD, BPOD, and DMD. 
    """
    
    def __init__(self, load_field=None, save_field=None, save_mat=None, 
        inner_product=None, maxFieldsPerNode=None, numNodes=1, verbose=False):
        """
        Modal decomposition constructor.
    
        This constructor sets the default values for data members common to all
        derived classes.  All derived classes should be sure to invoke this
        constructor explicitly.
        """
        self.mpi = util.MPI(verbose=verbose)
        if maxFieldsPerNode is None:
            maxFieldsPerNode = 2
            if self.mpi.isRankZero() and verbose:
                print 'Warning - only loading 2 fields/processor,',\
                'increase maxFieldsPerNode for a speedup'
        self.maxFieldsPerNode = maxFieldsPerNode
        self.numNodes = numNodes
        self.load_field = load_field
        self.save_field = save_field
        self.save_mat = save_mat
        self.inner_product = inner_product
        if (self.numNodes > self.mpi.getNumProcs()):
            raise util.MPIError('More nodes than processors, ' + \
              '#nodes='+str(self.numNodes) + ' #procs=' + \
              str(self.mpi.getNumProcs())) 
              
        self.maxFieldsPerProc = \
          max(2, maxFieldsPerNode / (self.mpi.getNumProcs() / self.numNodes))
        self.verbose = verbose
        

        
    def idiot_check(self, testObj=None, testObjPath=None):
        """Checks that the user-supplied objects and functions work properly.
        
        The arguments are for a test object or the path to one (loaded with load_field)
        One of these should be supplied for thorough testing. The add and
        mult functions are tested for the generic object.
        This is not a complete testing, but catches some common mistakes.
        
        Other things which could be tested:
          reading/writing doesnt effect other snaps/modes (memory problems)
          subtraction, division (currently not used for modaldecomp)
        """
        tol = 1e-10
        if testObjPath is not None:
          testObj = self.load_field(testObjPath)
        if testObj is None:
          raise RuntimeError('Supply snap (or mode) object or path to one for idiot check!')
        objCopy = copy.deepcopy(testObj)
        objCopyMag2 = self.inner_product(objCopy,objCopy)
        
        factor = 2.
        objMult = testObj * factor
        
        if abs(self.inner_product(objMult,objMult)-objCopyMag2*factor**2)>tol:
          raise ValueError('Multiplication of snap/mode object failed')
        
        if abs(self.inner_product(testObj,testObj)-objCopyMag2)>tol:  
          raise ValueError('Original object modified by multiplication!')        
        
        objAdd = testObj + testObj
        if abs(self.inner_product(objAdd,objAdd) - objCopyMag2*4)>tol:
          raise ValueError('Addition does not give correct result')
        
        if abs(self.inner_product(testObj,testObj)-objCopyMag2)>tol:  
          raise ValueError('Original object modified by addition!')       
        
        objAddMult = testObj*factor + testObj
        if abs(self.inner_product(objAddMult,objAddMult)-objCopyMag2*(factor+1)**2)>tol:
          raise ValueError('Multiplication and addition of snap/mode are inconsistent')
        
        if abs(self.inner_product(testObj,testObj)-objCopyMag2)>tol:  
          raise ValueError('Original object modified by combo of mult/add!') 
        
        #objSub = 3.5*testObj - testObj
        #N.testing.assert_array_almost_equal(objSub,2.5*testObj)
        #N.testing.assert_array_almost_equal(testObj,objCopy)
        if self.verbose:
            print 'Passed the idiot check'
    
    
    def _compute_inner_product_chunk(self,rowFieldPaths,colFieldPaths,verbose=None):
        """ Computes inner products of snapshots in memory-efficient chunks
        
        The 'chunk' refers to the fact that within this method, the 
        snapshots
        are read in memory-efficient ways such that they are not all in
        memory at 
        once. This results in finding 'chunks' of the eventual matrix 
        that is returned. 
        It is also true that this function is meant to be 
        used in parallel - individually on each processor. In this case,
        each processor has different lists of snapshots to take inner products
        of.
        rows = number of row snapshot files passed in (BPOD adjoint snaps)
        columns = number column snapshot files passed in (BPOD direct snaps)
        Currently this method only supports finding rectangular chunks.
        In the future this method can be expanded to find more general shapes
        of the matrix.
        It returns a matrix with the above number of rows and columns.
        verbose - If verobse is True, then print the progress of inner products
        """
        if isinstance(rowFieldPaths,str):
            rowFieldPaths = [rowFieldPaths]
        if isinstance(colFieldPaths,str):
            colFieldPaths = [colFieldPaths]
        
        numRows = len(rowFieldPaths)
        numCols = len(colFieldPaths)
        
        #enforce that there are more columns than rows for efficiency
        if numRows > numCols:
            transpose = True
            temp = rowFieldPaths
            rowFieldPaths = colFieldPaths
            colFieldPaths = temp
            temp = numRows
            numRows = numCols
            numCols = temp
        else: 
            transpose = False
       
        if verbose is not None:
            self.verbose = verbose
        if self.verbose:
            printAfterNumCols = (numCols/5)+1 # Print after this many cols are computed
       
        numColsPerChunk = 1
        numRowsPerChunk = self.maxFieldsPerProc-numColsPerChunk         
        
        innerProductMatChunk = N.mat(N.zeros((numRows,numCols)))
        
        for startRowIndex in range(0,numRows,numRowsPerChunk):
            endRowIndex = min(numRows,startRowIndex+numRowsPerChunk)
            
            rowSnaps = []
            for rowPath in rowFieldPaths[startRowIndex:endRowIndex]:
                rowSnaps.append(self.load_field(rowPath))
         
            for startColIndex in range(0,numCols,numColsPerChunk):
                endColIndex = min(numCols,startColIndex+numColsPerChunk)

                colSnaps = []
                for colPath in colFieldPaths[startColIndex:endColIndex]:
                    colSnaps.append(self.load_field(colPath))
              
                #With the chunks of the row and column matrices,
                #find inner products
                for rowIndex in range(startRowIndex,endRowIndex):
                    for colIndex in range(startColIndex,endColIndex):
                        innerProductMatChunk[rowIndex,colIndex] = \
                          self.inner_product(rowSnaps[rowIndex-startRowIndex],
                          colSnaps[colIndex-startColIndex])
                if self.verbose and (endColIndex%printAfterNumCols==0 or \
                  endColIndex==numCols): 
                    numCompletedIPs = startRowIndex*numCols + \
                      (endRowIndex-startRowIndex)*endColIndex
                    print >> sys.stderr, 'Processor', self.mpi.getRank(),\
                        'completed', 'hankelMat[:' + str(endRowIndex) + ',:' +\
                        str(endColIndex)+']', 'out of hankelMat[' +\
                        str(numRows)+','+str(numCols)+']\n    ',\
                        int(1000.*numCompletedIPs/(1.*numCols*numRows))/10.,\
                        '% complete inner products on this processor'
       
        if transpose: innerProductMatChunk=innerProductMatChunk.T
        return innerProductMatChunk
        
           
            
    
    def _compute_modes(self,modeNumList,modePath,snapPaths,fieldCoeffMat,
        indexFrom=1):
        """
        A common method to compute and save modes from snapshots.
        
        modeNumList - mode numbers to compute on this processor. This 
          includes the indexFrom, so if indexFrom=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 
        modePath - Full path to mode location, e.g /home/user/mode_%03d.txt.
        indexFrom - Choose to index modes starting from 0, 1, or other.
        snapPaths - A list paths to files from which snapshots can be loaded.
        fieldCoeffMat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth index mode, ie
            indexFrom+k mode number. ith row contains coefficients to multiply
            corresponding to snapshot i.

        This methods primary purpose is to recast the problem as a simple
        linear combination of elements. It then calls lin_combine_fields.
        This mostly consists of rearranging the coeff matrix so that
        the first column corresponds to the first mode number in modeNumList.
        For more details on how the modes are formed, see doc on
        lin_combine_fields,
        where the outputFields are the modes and the inputFields are the 
        snapshots.
        """
        
        if self.save_field is None:
            raise UndefinedError('save_field is undefined')
                    
        if isinstance(modeNumList, int):
            modeNumList = [modeNumList]
        if isinstance(snapPaths, type('a_string')):
            snapPaths = [snapPaths]
        
        numModes = len(modeNumList)
        numSnaps = len(snapPaths)
        
        if numModes > numSnaps:
            raise ValueError('cannot compute more modes than number of snapshots')
                   
        for modeNum in modeNumList:
            if modeNum < indexFrom:
                raise ValueError('Cannot compute if mode number is less than '+\
                    'indexFrom')
            elif modeNum-indexFrom >= fieldCoeffMat.shape[1]:
                raise ValueError('Cannot compute if mode index is greater '+\
                    'than number of columns in the build coefficient matrix')

        if numSnaps < self.mpi.getNumProcs():
            raise util.MPIError('Cannot find modes when fewer snapshots '+\
               'than number of processors')
        
        
        # Construct fieldCoeffMat and outputPaths for lin_combine_fields
        modeNumListFromZero = []
        for modeNum in modeNumList:
            modeNumListFromZero.append(modeNum-indexFrom)
        fieldCoeffMatReordered = fieldCoeffMat[:,modeNumListFromZero]
        modePaths = []
        for modeNum in modeNumList:
            modePaths.append(modePath%modeNum)
        self.lin_combine_fields(modePaths,snapPaths,fieldCoeffMatReordered)
      
    
    
    
    def lin_combine_fields(self,outputFieldPaths,inputFieldPaths,fieldCoeffMat):
        """
        Linearly combines the input fields and saves them.
        
        outputFieldPaths is a list of the files where the linear combinations
          will be saved.
        inputFieldPaths is a list of files where the basis fields will
          be read from.
        fieldCoeffMat is a matrix where each row corresponds to an input field
          and each column corresponds to a output field. The rows and columns
          are assumed to correspond, by index, to the lists inputFieldPaths and 
          outputFieldPaths.
          outputs = inputs * fieldCoeffMat
        
        Each processor reads a subset of the input fields to compute as many
        outputs as a processor can have in memory at once. Each processor
        computes the "layers" from the inputs it is resonsible for, and for
        as many modes as it can fit in memory. The layers from all procs are
        then
        summed together to form the full outputs. The modes are then saved
        to file.        
        """
        if self.save_field is None:
            raise UndefinedError('save_field is undefined')
                   
        if isinstance(outputFieldPaths, str):
            outputFieldPaths = [outputFieldPaths]
        if isinstance(inputFieldPaths, str):
            inputFieldPaths = [inputFieldPaths]
        
        numInputFields = len(inputFieldPaths)
        numOutputFields = len(outputFieldPaths)
        
        if numInputFields > fieldCoeffMat.shape[0]:
            print 'numInputFields',numInputFields
            print 'rows of fieldCoeffMat',fieldCoeffMat.shape[0]
            raise ValueError('coeff mat has fewer rows than num of input paths')
        if numOutputFields > fieldCoeffMat.shape[1]:
            raise ValueError('Coeff matrix has fewer cols than num of output paths')            
        if numInputFields < self.mpi.getNumProcs():
            raise util.MPIError('Cannot find outputs when fewer inputs '+\
               'than number of processors')
               
        if numInputFields < fieldCoeffMat.shape[0]:
            print 'Warning - fewer input paths than cols in the coeff matrix'
            print '  some rows of coeff matrix will not be used'
        if numOutputFields < fieldCoeffMat.shape[1]:
            print 'Warning - fewer output paths than rows in the coeff matrix'
            print '  some cols of coeff matrix will not be used'
        
        inputProcAssignments = self.mpi.find_proc_assignments(range(len(inputFieldPaths)))
        for assignment in inputProcAssignments:
            if len(assignment) == 0:
                raise MPIError('At least one processor has no tasks'+\
                  ', currently this is unsupported, lower num of procs')
                  
        # Each processor will load only 1 processor at a time
        # and have as many (partially computed) modes in memory as possible.
        numOutputsPerProc = self.maxFieldsPerProc - 1       
        
        for startOutputIndex in range(0,numOutputFields,numOutputsPerProc):
            endOutputIndex = min(numOutputFields, startOutputIndex + numOutputsPerProc) 
            # Pass the work to individual processors    
            outputLayers = self.lin_combine_fields_chunk(
                inputFieldPaths[inputProcAssignments[self.mpi.getRank()][0] : \
                  inputProcAssignments[self.mpi.getRank()][-1]+1],
                fieldCoeffMat[inputProcAssignments[self.mpi.getRank()][0] : \
                  inputProcAssignments[self.mpi.getRank()][-1]+1,
                  startOutputIndex:endOutputIndex])       
            
            if self.mpi.isParallel():
                outputLayers = self.mpi.custom_comm.allreduce(outputLayers, 
                  op=util.sum_lists)

            saveOutputIndexAssignments = \
              self.mpi.find_proc_assignments(range(len(outputLayers)))
            if len(saveOutputIndexAssignments[self.mpi.getRank()]) != 0:
                for outputIndex in saveOutputIndexAssignments[self.mpi.getRank()]:
                    self.save_field(outputLayers[outputIndex], 
                      outputFieldPaths[startOutputIndex + outputIndex])

            if self.verbose and self.mpi.isRankZero():
                print >> sys.stderr, 'Computed and saved',\
                  round(1000.*endOutputIndex/numOutputFields)/10.,\
                  '% of output fields,',endOutputIndex,'out of',numOutputFields
            self.mpi.sync() # Not sure if necessary
            
    

    def lin_combine_fields_chunk(self,inputFieldPaths,fieldCoeffMat):
        """
        Computes a layer of the outputs for a particular processor.
        
        This method is to be called on a per-proc basis.
        inputFieldPaths is the list of input fields for which this proc 
          is responsible.
        fieldCoeffMat is a matrix containing coeffs for linearly combining
          inputFields into the layers of the outputs.
          The first index corresponds to the input, the second index the output.
          This is backwards from what one might expect from the equation
          outputs = fieldCoeffMat * inputs, where inputs and outputs
          are column vectors. It is best to think as:
          outputs = inputs * fieldCoeffMat, where inputs and outputs
          are row vectors and each element is a field object.
        """
        
        numInputs = len(inputFieldPaths)
        numOutputs = fieldCoeffMat.shape[1]
        assert fieldCoeffMat.shape[0] == numInputs
        
        numInputsPerChunk = 1

        outputLayers = []
        # Sweep through all snapshots, adding "layers", ie adding 
        # the contribution of each snapshot
        for startInputIndex in xrange(0,numInputs,numInputsPerChunk):
            endInputIndex = min(startInputIndex+numInputsPerChunk,numInputs)
            inputs=[]
            
            for inputIndex in xrange(startInputIndex,endInputIndex):
                inputs.append(self.load_field(inputFieldPaths[inputIndex]))
                # Might be able to eliminate this loop for array 
                # multiplication (after tested)
                # But this could increase memory usage, be careful
                
            for outputIndex in xrange(0,numOutputs):
                for inputIndex in xrange(startInputIndex,endInputIndex):
                    outputLayer = inputs[inputIndex-startInputIndex]*\
                      fieldCoeffMat[inputIndex,outputIndex]
                    if outputIndex>=len(outputLayers): 
                        # The mode list isn't full, must be created
                        outputLayers.append(outputLayer) 
                    else: 
                        outputLayers[outputIndex] += outputLayer
        # Return summed contributions from snapshot set to current modes            
        return outputLayers  

