
"""Collection of useful functions for modaldecomp library"""

import sys  
import copy
import numpy as N
import util
import parallel

class FieldOperations(object):
    """
    Does many useful operations on fields.

    All modal decomp classes should use the common functionality provided
    in this class as much as possible.
    
    Only advanced users should ever use this class, it is mostly a collection
    of functions used in the higher level modal decomp classes like POD,
    BPOD, and DMD.
    """
    
    def __init__(self, load_field=None, save_field=None, inner_product=None, 
        maxFieldsPerNode=None, verbose=True):
        """
        Sets the default values for data members. 
        
        BPOD, POD, DMD, other classes, should make heavy use of the low
        level functions in this class.
        """
        self.load_field = load_field
        self.save_field = save_field
        self.inner_product = inner_product
        self.verbose = verbose

        self.parallel = parallel.parallelInstance
        self.parallel.verbose = self.verbose
        if maxFieldsPerNode is None:
            self.maxFieldsPerNode = 2
            if self.parallel.isRankZero() and self.verbose:
                print 'Warning: maxFieldsPerNode was not specified. ' +\
                    'Assuming 2 fields can be loaded per node. Increase ' +\
                    'maxFieldsPerNode for a speedup.'
        else:
            self.maxFieldsPerNode = maxFieldsPerNode
        
        if self.maxFieldsPerNode < \
            2 * self.parallel.getNumProcs() / self.parallel.getNumNodes(): 
            self.maxFieldsPerProc = 2
            if self.verbose and self.parallel.isRankZero():
                print 'Warning: maxFieldsPerNode too small for given ' +\
                    'number of nodes and procs.  Assuming 2 fields can be ' +\
                    'loaded per processor. Increase maxFieldsPerNode for a ' +\
                    'speedup!'
        else:
            self.maxFieldsPerProc = self.maxFieldsPerNode * \
                self.parallel.getNumNodes()/self.parallel.getNumProcs()

    def idiot_check(self, testObj=None, testObjPath=None):
        """
        Checks that the user-supplied objects and functions work properly.
        
        The arguments are for a test object or the path to one (loaded with 
        load_field).  One of these should be supplied for thorough testing. 
        The add and mult functions are tested for the generic object.  This is 
        not a complete testing, but catches some common mistakes.
        
        Other things which could be tested:
            reading/writing doesnt effect other snaps/modes (memory problems)
            subtraction, division (currently not used for modaldecomp)
        """
        tol = 1e-10
        if testObjPath is not None:
          testObj = self.load_field(testObjPath)
        if testObj is None:
            raise RuntimeError('Supply snap (or mode) object or path to one '+\
                'for idiot check!')
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
        if abs(self.inner_product(objAddMult, objAddMult) - objCopyMag2 * (
            factor + 1 ) ** 2 ) > tol:
          raise ValueError('Multiplication and addition of snap/mode are '+\
            'inconsistent')
        
        if abs(self.inner_product(testObj,testObj)-objCopyMag2)>tol:  
          raise ValueError('Original object modified by combo of mult/add!') 
        
        #objSub = 3.5*testObj - testObj
        #N.testing.assert_array_almost_equal(objSub,2.5*testObj)
        #N.testing.assert_array_almost_equal(testObj,objCopy)
        if self.verbose:
            print 'Passed the idiot check'

    
    def _print_inner_product_progress(self, endRowIndex, numRows, numCols):
        """Prints progress if verbose is True"""
        if self.verbose:
            #if endColIndex % printAfterNumCols==0 or endColIndex==numCols: 
            numCompletedIPs = endRowIndex * numCols
            percentCompletedIPs = 100. * numCompletedIPs/(numCols*numRows)
            print >> sys.stderr, ('Processor %d completed %.1f%% of inner ' +\
                'products: IPMat[:%d, :%d] of IPMat[:%d, :%d]') % \
                (self.parallel.getRank(), percentCompletedIPs, endRowIndex, \
                numCols, numRows, numCols)

  
    def compute_inner_product_mat(self, rowFieldPaths, colFieldPaths):
        """ 
        Computes a matrix of inner products (Y'*X) and returns it.
        
        This method assigns the task of computing the a matrix of inner products
        into pieces for each processor, then passes this onto 
        self._compute_inner_product_chunk(...). After 
        _compute_inner_product_chunk returns chunks of the inner product matrix,
        they are concatenated into a completed, single, matrix on all processors 
        """      
        if isinstance(rowFieldPaths,str):
            rowFieldPaths = [rowFieldPaths]
        if isinstance(colFieldPaths,str):
            colFieldPaths = [colFieldPaths]
          
        numColFields = len(colFieldPaths)
        numRowFields = len(rowFieldPaths)

        # Enforce that there are more rows than columns for efficiency
        # Each column is read by one proc and then sent to others, so
        # processors are responsible for unique rows.
        # It might not matter which is larger since there is no repeated
        # load in any direction, only send/receives
        if numColFields > numRowFields:
            transpose = True
            temp = rowFieldPaths
            rowFieldPaths = colFieldPaths
            colFieldPaths = temp
            temp = numRowFields
            numRowFields = numColFields
            numColFields = temp
        else: 
            transpose = False

        rowFieldProcAssignments = self.parallel.find_assignments(range(
            numRowFields))

        if self.parallel.isRankZero() and rowFieldProcAssignments[0][-1] -\
            rowFieldProcAssignments[0][0] > self.maxFieldsPerProc and self.\
            verbose:
            print ('Warning: Each processor will have to read the direct ' +\
                'snapshots (%d total) multiple times. Increase number of ' +\
                'processors to avoid this and get a big speedup.') %\
                numColFields

        # Only compute if task list is nonempty
        if len(rowFieldProcAssignments[self.parallel.getRank()]) != 0:
            innerProductMatChunk = self._compute_inner_product_chunk(
                rowFieldPaths[rowFieldProcAssignments[self.parallel.getRank()][0]:
                rowFieldProcAssignments[self.parallel.getRank()][-1]+1], 
                colFieldPaths)
        else:  
            innerProductMatChunk = None

        # Gather list of chunks from each processor, ordered by rank
        if self.parallel.isDistributed():
            innerProductMatChunkList = self.parallel.comm.allgather(
                innerProductMatChunk)
            innerProductMat = N.mat(N.zeros((numRowFields, numColFields)))

            # concatenate the chunks of inner product matrix for nonempty tasks
            for rank, currentInnerProductMatChunk in enumerate(
                innerProductMatChunkList):
                if currentInnerProductMatChunk is not None:
                    innerProductMat[rowFieldProcAssignments[rank][0]:\
                        rowFieldProcAssignments[rank][-1]+1] =\
                        currentInnerProductMatChunk
        else:
            innerProductMat = innerProductMatChunk 

        if transpose:
            innerProductMat = innerProductMat.T

        return innerProductMat
  
          
    def _compute_inner_product_chunk(self, rowFieldPaths, colFieldPaths):
        """ 
        Computes inner products of snapshots in memory-efficient chunks
        
        The 'chunk' refers to the fact that within this method, the snapshots
        are read in memory-efficient ways such that they are not all in memory 
        at once. This results in finding 'chunks' of the eventual matrix that 
        is returned. It is also true that this function is meant to be used in 
        distributed - individually on each processor. In this case, each processor 
        has different lists of snapshots to take inner products of.
            rows = number of row snapshot files passed in (BPOD adjoint snaps)
            columns = number column snapshot files passed in (BPOD direct snaps)
        Currently this method only supports finding rectangular chunks.  In the 
        future this method can be expanded to find more general shapes of the 
        matrix. It returns a matrix with the above number of rows and columns.
        
        To determine how the columns will be split up, it is important to see
        how the columns are handled. Each processor is responsible for loading
        a subset of the columns. The processor which read a particular
        column field then sends it to each successive processor so it
        can be used to compute all IPs for the current row chunk on each
        processor. This process is done on each processor. It is repeated
        until all processors are done with all of their row chunks. If there
        are 2 processors:
         
          | c0 o  |
        r0| c0 o  |
          | c0 o  |
          -
          | o  c1 |
        r1| o  c1 |
          | o  c1 |
        
        Rank 0 (r0) reads column 0 (c0) and fills out IPs for all rows in a row chunk
        Here there is only one row chunk for each processor for simplicity.
        Rank 1 reads column 1 (c1) and fills out IPs for all rows in a row chunk.
        In the next step, rank 0 sends c0 to rank 1 and rank 1 sends c1 to rank 1.
        The remaining IPs are filled in.
        
          | c0 c1 |
        r0| c0 c1 |
          | c0 c1 |
          -
          | c0 c1 |
        r1| c0 c1 |
          | c0 c1 |
        This is complicated by having uneven loads in the rows and columns
        on each processor. Specifically, there are multiple row chunks,
        possibly different #s of row chunks for each processor, and each 
        processor can be responsible for a different number of columns.
        This is also generalized to allow the columns to be read in chunks.
        This could be useful, for example, in a shared memory setting where
        it is best to work in operation-units (loads, IPs, etc) of procs/node.
        """
        # Must check that these are lists, in case method is called directly
        # When called as part of compute_inner_product_matrix, paths are
        # generated by getProcAssignments, and are called such that a list is
        # always passed in
        if not isinstance(rowFieldPaths,list):
            rowFieldPaths = [rowFieldPaths]
        if not isinstance(colFieldPaths,list):
            colFieldPaths = [colFieldPaths]
        
        numRows = len(rowFieldPaths)
        numCols = len(colFieldPaths)
        
        # Enforce more columns than rows for efficiency - no longer necessary??
                
        numColsPerChunk = 1
        numRowsPerChunk = self.maxFieldsPerProc-numColsPerChunk         
        
        # Determine how the columns will be split up.
        # For now the proc assignments must all have an equal number of 
        # elements otherwise some processors will stop executing a loop
        # and a barrier/sync command will fail. The "leftover" col fields are
        # simply read by ALL processors. Inefficient, but minor for our
        # typical usage of # processors.
        numColsParallel = numCols - (numCols%self.parallel.getNumProcs())
        colFieldProcAssignments = \
            self.parallel.find_assignments(range(numColsParallel))
        # Quick check to be sure that each assignment has equal length
        lenAssignment = len(colFieldProcAssignments[0])
        if self.parallel.getNumProcs() != \
            [len(assign) for assign in colFieldProcAssignments].count(lenAssignment):
            print 'length of all colFieldProcAssignments are',\
            [len(assign) for assign in colFieldProcAssignments],'and lenAssignment is',lenAssignment,\
            'and count of them is',[len(assign) for assign in colFieldProcAssignments].count(lenAssignment),\
            'and numProcs is',self.parallel.getNumProcs() 
            raise ValueError('Assignments have different length')
        """
        # Set up the colFieldProcAssignments using all columns.
        # The loop over the assignments must be changed to account for
        # different numbers of assignments for each processor.
        colFieldProcAssignments = self.parallel.find_assignments(range(numCols))
        # Find the maximum number of assignments given to a processor
        maxAssignments = N.ceil(\
            max([len(assgn) for assgn in colFieldProcAssignments])/(1.*numColsPerChunk))
        """
        innerProductMatChunk = N.mat(N.zeros((numRows,numCols)))
        
        for startRowIndex in range(0,numRows,numRowsPerChunk):
            endRowIndex = min(numRows,startRowIndex+numRowsPerChunk)
            
            rowFields = [self.load_field(rowPath) \
                for rowPath in rowFieldPaths[startRowIndex:endRowIndex]]
            
            # New method which increases speed by factor of ~numprocs
            if lenAssignment > 0:
                for startColIndex in xrange(colFieldProcAssignments[self.parallel.getRank()][0],\
                    colFieldProcAssignments[self.parallel.getRank()][-1]+1,numColsPerChunk):
                    endColIndex = min(numCols,startColIndex+numColsPerChunk)
                    # Pass the col fields to proc with rank -> mod(rank+1,numProcs) 
                    #if self.parallel.isDistributed():
                    # Must do this for each processor, until data makes a circle
                    # should change the while loop to a for loop in the future
                    colFieldsRecv = (None, None)
                    colIndices = range(startColIndex,endColIndex)
                    for numPasses in xrange(self.parallel.getNumProcs()):
                        # If on the first pass, load the col fields, no send/recv
                        # This is all that is called when in serial, loop iterates once.
                        if numPasses == 0:
                            colFields = [self.load_field(colPath) \
                                for colPath in colFieldPaths[startColIndex:endColIndex]]
                
                        else:
                            colFieldsSend = (colFields, colIndices)
                            dest = (self.parallel.getRank()+1) % self.parallel.getNumProcs()
                            #Create unique tag based on ranks
                            sendTag = self.parallel.getRank()*(self.parallel.getNumProcs()+1) + dest
                            self.parallel.comm.send(colFieldsSend, dest=dest, tag=sendTag)
                            
                            source = (self.parallel.getRank()-1) % self.parallel.getNumProcs()
                            recvTag = source*(self.parallel.getNumProcs()+1) + self.parallel.getRank()
                            colFieldsRecv = self.parallel.comm.recv(source=source, tag=recvTag)
                            
                            colIndices = colFieldsRecv[1]
                            colFields = colFieldsRecv[0]
                        # Compute the IPs for this set of data
                        # colIndices stores the indices of the innerProductMatChunk columns
                        # to be filled in. The length of the list is <=numColsPerChunk, less 
                        # than when at the end of the columns. Currently numColsPerChunk=1.
                        for rowIndex in xrange(startRowIndex,endRowIndex):
                            for colFieldIndex,colField in enumerate(colFields):
                                innerProductMatChunk[rowIndex,colIndices[colFieldIndex]] = \
                                  self.inner_product(rowFields[rowIndex-startRowIndex],colField)
                        self.parallel.sync() # necessary?  
                    
            # Done the loop through the col fields that were parallelized,
            # now finish of the remaining ending cols the slow way, each
            # proc reading each col. This should eventually be removed, and
            # all of the columns should be done in parallel
            for startColIndex in xrange(numColsParallel,numCols,numColsPerChunk):
                endColIndex = min(numCols,startColIndex+numColsPerChunk)
                # Load a subset of the column fields this proc is responsible for
                colFields = [self.load_field(colPath) \
                    for colPath in colFieldPaths[startColIndex:endColIndex]]
                # Compute the IPs that are possible with this set of the snapshots
                for rowIndex in xrange(startRowIndex,endRowIndex):
                    for colIndex in xrange(startColIndex,endColIndex):
                        innerProductMatChunk[rowIndex,colIndex] = \
                          self.inner_product(rowFields[rowIndex-startRowIndex],
                          colFields[colIndex-startColIndex])                    
            self._print_inner_product_progress(endRowIndex, numRows, numCols)
        
        return innerProductMatChunk
      


    def _compute_upper_triangular_inner_product_chunk(self, rowFieldPaths, 
        colFieldPaths):
        """
        Computes a chunk of a symmetric matrix of inner products.  
        
        Because the matrix is symmetric, each N x M rectangular chunk will have
        a symmetric N x N square on its left side.  This part of the chunk can 
        be computed efficiently because no additional fields need to be loaded
        for the columns (due to symmetry).  In addition, only the upper-
        triangular part of this N x N subchunk will be computed.  

        The N x (M - N) remainder of the chunk is computed here rather than by 
        using the standard method _compute_inner_product_chunk because that
        would reload all the row fields.
        """
        # Must check that these are lists, in case method is called directly
        # When called as part of compute_inner_product_matrix, paths are
        # generated by getProcAssignments, and are called such that a list is
        # always passed in
        if isinstance(rowFieldPaths, str):
            rowFieldPaths = [rowFieldPaths]

        if isinstance(colFieldPaths, str):
            colFieldPaths = [colFieldPaths]
 
        numRows = len(rowFieldPaths)
        numCols = len(colFieldPaths) 

        # For a chunk of a symmetric inner product matrix, the first numRows
        # paths should be the same in rowFieldPaths and colFieldPaths
        if rowFieldPaths != colFieldPaths[:numRows]:
            raise ValueError('rowFieldPaths and colFieldPaths must share ' +\
                'same leading entries for a symmetric inner product matrix ' +\
                'chunk.')

        if self.verbose:
            # Print after this many cols are computed
            printAfterNumCols = (numCols / 5) + 1 

        # If computing a square chunk (upper triangular part) and all rows can
        # be loaded simultaneously, no need to save room for a column chunk
        if self.maxFieldsPerProc >= numRows and numRows == numCols:
            numColsPerChunk = 0
        else:
            numColsPerChunk = 1 
        numRowsPerChunk = self.maxFieldsPerProc - numColsPerChunk         

        innerProductMatChunk = N.mat(N.zeros((numRows, numCols)))
        
        for startRowIndex in range(0, numRows, numRowsPerChunk):
            endRowIndex = min(numRows, startRowIndex + numRowsPerChunk)
           
            # Load a set of row snapshots.  
            rowFields = [self.load_field(rowPath) \
                for rowPath in rowFieldPaths[startRowIndex:endRowIndex]]
           
            # On current set of rows, compute symmetric part (i.e. inner
            # products that only depend on the already loaded fields)
            for rowIndex in xrange(startRowIndex, endRowIndex):
                # Diagonal term
                innerProductMatChunk[rowIndex, rowIndex] = self.inner_product(
                    rowFields[rowIndex - startRowIndex], rowFields[rowIndex -\
                    startRowIndex])
                
                # Off diagonal terms.  This block is square, so the first index
                # is rowIndex + 1, and the last index is the last rowIndex
                for colIndex in xrange(rowIndex + 1, endRowIndex):
                    innerProductMatChunk[rowIndex, colIndex] = self.\
                        inner_product(rowFields[rowIndex - startRowIndex], 
                        rowFields[colIndex - startRowIndex])
                
            # In case this whole chunk is square and can be loaded at once,
            # define endColIndex for the progress report message.  (The
            # for loop below will not be executed in this case, so this
            # variable would not be defined.)
            endColIndex = endRowIndex
            if self.verbose:
                self._print_inner_product_progress(startRowIndex, endRowIndex,
                    endColIndex, numRows, numCols, printAfterNumCols)

            # Now compute the part that relies on snapshots that haven't been
            # loaded (ie for columns whose indices are greater than the largest
            # row index).  Not necessary if chunk is square and all row fields
            # can be loaded at same time.
            if numColsPerChunk != 0:
                for startColIndex in range(endRowIndex, numCols, 
                    numColsPerChunk):
                    endColIndex = min(numCols, startColIndex + numColsPerChunk)
                    colFields = [self.load_field(colPath) \
                        for colPath in colFieldPaths[startColIndex:endColIndex]]
                    
                    # With the chunks of the row and column matrices,
                    # find inner products
                    for rowIndex in range(startRowIndex, endRowIndex):
                        for colIndex in range(startColIndex, endColIndex):
                            innerProductMatChunk[rowIndex, colIndex] = self.\
                                inner_product(rowFields[rowIndex -\
                                startRowIndex], colFields[colIndex -\
                                startColIndex])
                    if self.verbose:
                        self._print_inner_product_progress(startRowIndex, 
                            endRowIndex, endColIndex, numRows, numCols, 
                            printAfterNumCols)
 
        return innerProductMatChunk
      

     
    def compute_symmetric_inner_product_mat(self, fieldPaths):
        """ 
        Computes a symmetric matrix of inner products and returns it.
        
        Because the inner product is symmetric, only one set of snapshots needs
        to be specified.  This method will call
        _compute_upper_triangular_inner_product_matrix_chunk and at the end
        will symemtrize the upper triangular matrix.
        """
      
        if isinstance(fieldPaths,str):
            fieldPaths = [fieldPaths]
          
        numFields = len(fieldPaths)
 
        rowFieldProcAssignments = self.parallel.find_assignments(range(
            numFields), taskWeights=range(numFields, 0, -1))

        if self.parallel.isRankZero() and rowFieldProcAssignments[0][-1] -\
            rowFieldProcAssignments[0][0] > self.maxFieldsPerProc and self.\
            verbose:
            print ('Warning: Each processor may have to read the snapshots ' +\
                '(%d total) multiple times. Increase number of processors ' +\
                'to avoid this and get a big speedup.') % numFields

        # Perform task if task assignment is not empty
        if len(rowFieldProcAssignments[self.parallel.getRank()]) != 0:
            innerProductMatChunk = \
                self._compute_upper_triangular_inner_product_chunk(\
                fieldPaths[rowFieldProcAssignments[self.parallel.getRank()][0]:\
                rowFieldProcAssignments[self.parallel.getRank()][-1] + 1], \
                fieldPaths[rowFieldProcAssignments[self.parallel.getRank()][0]:])
        else:
            innerProductMatChunk = None

        # Gather list of chunks from each processor, ordered by rank
        if self.parallel.isDistributed():
            innerProductMatChunkList = \
                self.parallel.comm.allgather(innerProductMatChunk)
            innerProductMat = N.mat(N.zeros((numFields, numFields)))

            # concatenate the chunks of inner product matrix for procs with
            # nonempty tasks
            for rank, currentInnerProductMatChunk in \
                enumerate(innerProductMatChunkList):
                if currentInnerProductMatChunk is not None:
                    innerProductMat[rowFieldProcAssignments[rank][0]:\
                        rowFieldProcAssignments[rank][-1] + 1,
                        rowFieldProcAssignments[rank][0]:] =\
                        currentInnerProductMatChunk
        else:
            innerProductMat = innerProductMatChunk 

        # Symmetrize matrix
        innerProductMat += N.triu(innerProductMat, 1).T

        return innerProductMat
    
    def _compute_modes(self, modeNumList, modePath, snapPaths, fieldCoeffMat,
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
            column contains the coefficients for computing the kth index mode, 
            ie indexFrom+k mode number. ith row contains coefficients to 
            multiply corresponding to snapshot i.

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
            raise ValueError('Cannot compute more modes than number of ' +\
                'snapshots')
                   
        for modeNum in modeNumList:
            if modeNum < indexFrom:
                raise ValueError('Cannot compute if mode number is less than '+\
                    'indexFrom')
            elif modeNum-indexFrom >= fieldCoeffMat.shape[1]:
                raise ValueError('Cannot compute if mode index is greater '+\
                    'than number of columns in the build coefficient matrix')

        if numSnaps < self.parallel.getNumProcs():
            raise util.ParallelError('Cannot find modes when fewer snapshots '+\
               'than number of processors')
        
        # Construct fieldCoeffMat and outputPaths for lin_combine_fields
        modeNumListFromZero = [modeNum-indexFrom for modeNum in modeNumList]
        fieldCoeffMatReordered = fieldCoeffMat[:,modeNumListFromZero]
        modePaths = [modePath%modeNum for modeNum in modeNumList]
        self.lin_combine(modePaths, snapPaths, fieldCoeffMatReordered)
    
    def lin_combine(self, sumFieldPaths, basisFieldPaths, fieldCoeffMat):
        """
        Linearly combines the basis fields and saves them.
        
        sumFieldPaths is a list of the files where the linear combinations
          will be saved.
        basisFieldPaths is a list of files where the basis fields will
          be read from.
        fieldCoeffMat is a matrix where each row corresponds to an basis field
          and each column corresponds to a sum (lin. comb.) field. The rows and columns
          are assumed to correspond, by index, to the lists basisFieldPaths and 
          sumFieldPaths.
          sums = basis * fieldCoeffMat
        
        Each processor reads a subset of the basis fields to compute as many
        outputs as a processor can have in memory at once. Each processor
        computes the "layers" from the basis it is resonsible for, and for
        as many modes as it can fit in memory. The layers from all procs are
        then
        summed together to form the full outputs. The modes are then saved
        to file.        
        """
        if self.save_field is None:
            raise util.UndefinedError('save_field is undefined')
                   
        if isinstance(sumFieldPaths, str):
            sumFieldPaths = [sumFieldPaths]
        if isinstance(basisFieldPaths, str):
            basisFieldPaths = [basisFieldPaths]
        
        numBasisFields = len(basisFieldPaths)
        numSumFields = len(sumFieldPaths)
        
        if numBasisFields > fieldCoeffMat.shape[0]:
            print 'numInputFields',numBasisFields
            print 'rows of fieldCoeffMat',fieldCoeffMat.shape[0]
            raise ValueError('coeff mat has fewer rows than num of basis paths')
        if numSumFields > fieldCoeffMat.shape[1]:
            raise ValueError('Coeff matrix has fewer cols than num of ' +\
                'output paths')            
        if numBasisFields < self.parallel.getNumProcs():
            raise util.ParallelError('Cannot find outputs when fewer basiss '+\
               'than number of processors')
               
        if numBasisFields < fieldCoeffMat.shape[0] and self.parallel.isRankZero():
            print 'Warning - fewer basis paths than cols in the coeff matrix'
            print '  some rows of coeff matrix will not be used'
        if numSumFields < fieldCoeffMat.shape[1] and self.parallel.isRankZero():
            print 'Warning - fewer output paths than rows in the coeff matrix'
            print '  some cols of coeff matrix will not be used'
        
        basisProcAssignments = self.parallel.find_assignments(range(len(
            basisFieldPaths)))
        for assignment in basisProcAssignments:
            if len(assignment) == 0:
                raise ParallelError('At least one processor has no tasks'+\
                  ', currently this is unsupported, lower num of procs')
                  
        # Each processor will load only 1 processor at a time
        # and have as many (partially computed) modes in memory as possible.
        numSumsPerProc = self.maxFieldsPerProc - 1       
        
        for startSumIndex in range(0,numSumFields,numSumsPerProc):
            endSumIndex = min(numSumFields, startSumIndex +\
                numSumsPerProc) 
            # Pass the work to individual processors    
            sumLayers = self.lin_combine_chunk(
                basisFieldPaths[basisProcAssignments[self.parallel.getRank()][0] : \
                  basisProcAssignments[self.parallel.getRank()][-1]+1],
                fieldCoeffMat[basisProcAssignments[self.parallel.getRank()][0] : \
                  basisProcAssignments[self.parallel.getRank()][-1]+1,
                  startSumIndex:endSumIndex])       
            
            if self.parallel.isDistributed():
                sumLayers = self.parallel.custom_comm.allreduce(sumLayers, 
                  op=util.sum_lists)

            saveSumIndexAssignments = self.parallel.find_assignments(range(
                len(sumLayers)))
            if len(saveSumIndexAssignments[self.parallel.getRank()]) != 0:
                for sumIndex in saveSumIndexAssignments[self.parallel.\
                    getRank()]:
                    self.save_field(sumLayers[sumIndex], 
                      sumFieldPaths[startSumIndex + sumIndex])
 
            if self.verbose and self.parallel.isRankZero():
                print >> sys.stderr, 'Computed and saved',\
                  round(1000.*endSumIndex/numSumFields)/10.,\
                  '% of sum fields,',endSumIndex,'out of',numSumFields
            self.parallel.sync() # Not sure if necessary
            
    

    def lin_combine_chunk(self, basisFieldPaths, fieldCoeffMat):
        """
        Computes a layer of the sums for a particular processor.
        
        This method is to be called on a per-proc basis.
        basisFieldPaths is the list of basis fields for which this proc 
          is responsible.
        fieldCoeffMat is a matrix containing coeffs for linearly combining
          basisFields into the layers of the sums.
          The first index corresponds to the basis, the second index the sum field.
          This is backwards from what one might expect from the equation
          sums = fieldCoeffMat * basis, where basis and sums
          are column vectors. It is best to think as:
          sums = basis * fieldCoeffMat, where basis and sums
          are row vectors and each element is a field object.
        """
        
        numBasisFields = len(basisFieldPaths)
        numSumFields = fieldCoeffMat.shape[1]
        assert fieldCoeffMat.shape[0] == numBasisFields
        
        numBasisPerChunk = 1

        sumLayers = []
        # Sweep through all snapshots, adding "layers", ie adding 
        # the contribution of each snapshot
        for startBasisIndex in xrange(0,numBasisFields,numBasisPerChunk):
            endBasisIndex = min(startBasisIndex+numBasisPerChunk,numBasisFields)
            
            basisFields=[self.load_field(basisFieldPaths[basisIndex])\
                for basisIndex in xrange(startBasisIndex,endBasisIndex)]
                # Might be able to eliminate this loop for array 
                # multiplication (after tested)
                # But this could increase memory usage, be careful
                
            for sumIndex in xrange(0,numSumFields):
                for basisIndex in xrange(startBasisIndex,endBasisIndex):
                    sumLayer = basisFields[basisIndex-startBasisIndex]*\
                      fieldCoeffMat[basisIndex,sumIndex]
                    if sumIndex>=len(sumLayers): 
                        # The mode list isn't full, must be created
                        sumLayers.append(sumLayer) 
                    else: 
                        sumLayers[sumIndex] += sumLayer
        # Return summed contributions from snapshot set to current modes            
        return sumLayers  
        

    def __eq__(self, other):
        #print 'comparing fieldOperations classes'
        a = (self.inner_product == other.inner_product and \
        self.load_field == other.load_field and self.save_field == other.save_field \
        and self.maxFieldsPerNode==other.maxFieldsPerNode and\
        self.verbose==other.verbose)
        return a
    def __ne__(self,other):
        return not (self.__eq__(other))

