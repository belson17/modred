
"""Collection of useful functions for modaldecomp library"""

import sys  
import copy
import numpy as N
import util
import parallel as parallel_mod

# Should this be a data member? Only reason to make it a data member
# is parallel can then take the verbose argument
#parallel = parallel_mod.parallelInstance

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

        self.parallel = parallel_mod.parallelInstance
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
        if self.verbose and self.parallel.isRankZero():
            #if endColIndex % printAfterNumCols==0 or endColIndex==numCols: 
            numCompletedIPs = endRowIndex * numCols
            percentCompletedIPs = 100. * numCompletedIPs/(numCols*numRows)           
            print >> sys.stderr, ('Completed %.1f%% of inner ' +\
                'products: IPMat[:%d, :%d] of IPMat[%d, %d]') % \
                (percentCompletedIPs, endRowIndex, \
                numCols, numRows, numCols)

  
    def compute_inner_product_mat(self, rowFieldPaths, colFieldPaths):
        """ 
        Computes a matrix of inner products (Y'*X) and returns it.
        
        It is generally best to use all available processors, however this
        depends on the computer and the nature of the load and IP functions
        supplied. In some cases, loading in parallel is slower.
        
          rowFieldPaths = row snapshot files (BPOD adjoint snaps, ~Y)
          
          colFieldPaths = column snapshot files (BPOD direct snaps, ~X)

        Within this method, the snapshots
        are read in memory-efficient ways such that they are not all in memory 
        at once. This results in finding 'chunks' of the eventual matrix that 
        is returned.
        This method only supports finding a full rectangular mat. For POD, a different
        method is used to take advantage of the symmetric matrix.
        
        Each processor is responsible for loading
        a subset of the rows and columns. The processor which reads a particular
        column field then sends it to each successive processor so it
        can be used to compute all IPs for the current row chunk on each
        processor. This is repeated
        until all processors are done with all of their row chunks. If there
        are 2 processors::
           
                | r0c0 o  |
          rank0 | r1c0 o  |
                | r2c0 o  |
            -
                | o  r3c1 |
          rank1 | o  r4c1 |
                | o  r5c1 |
        
        Rank 0 reads column 0 (c0) and fills out IPs for all rows in a row chunk (r*c*)
        Here there is only one row chunk for each processor for simplicity.
        Rank 1 reads column 1 (c1) and fills out IPs for all rows in a row chunk.
        In the next step, rank 0 sends c0 to rank 1 and rank 1 sends c1 to rank 1.
        The remaining IPs are filled in::
        
                | r0c0 r0c1 |
          rank0 | r1c0 r1c1 |
                | r2c0 r2c1 |
            -
                | r3c0 r3c1 |
          rank1 | r4c0 r4c1 |
                | r5c0 r5c1 |
          
        In reality it is more complicated since the number of cols and rows
        may not be divisible by the number of processors. This is handled
        internally, by allowing the last processor to have fewer tasks, however
        it is still part of the passing circle, and rows and cols are
        handled independently.
        This is also generalized to allow the columns to be read in chunks, 
        rather than only 1 at a time.
        This could be useful, for example, in a shared memory setting where
        it is best to work in operation-units (loads, IPs, etc) of procs/node.
        
        The scaling is:
        
          num loads / processor ~ (n_r/(max*n_p))*n_c/n_p + n_r/n_p
          
          num MPI sends / processor ~ (n_r/(max*n_p))*(n_p-1)*n_c/n_p
          
          num inner products / processor ~ n_r*n_c/n_p
        
        where n_r is number of rows, n_c number of columns, max is 
        maxFieldsPerProc-1 = maxFieldsPerNode/numNodesPerProc - 1, and
        n_p is number of processors.
        
        It is enforced that there are more columns than rows by doing an
        internal transpose and un-transpose. This improves efficiency.
        
        From these scaling laws, it can be seen that it is generally 
        good to use all available processors, even though it lowers max.
        This depends though on the particular system. Sometimes multiple 
        simultaneous loads actually makes each load very slow. 
        
        As an example, consider doing a case with len(rowFieldPaths)=8
        and len(colFieldPaths) = 12, 2 processors, 1 node, and maxFieldsPerNode=3.
        n_p=2, max=2, n_r=8, n_c=12 (n_r < n_c).
        
        num loads / proc = 16
        
        If we flip n_r and n_c, num loads / proc = 18.
        """      
        if not isinstance(rowFieldPaths,list):
            rowFieldPaths = [rowFieldPaths]
        if not isinstance(colFieldPaths,list):
            colFieldPaths = [colFieldPaths]
            
        numCols = len(colFieldPaths)
        numRows = len(rowFieldPaths)

        if numRows < numCols:
            transpose = True
            temp = rowFieldPaths
            rowFieldPaths = colFieldPaths
            colFieldPaths = temp
            temp = numRows
            numRows = numCols
            numCols = temp
        else: 
            transpose = False

        # numColsPerChunk is the number of cols each proc loads at once        
        numColsPerProcChunk = 1
        numRowsPerProcChunk = self.maxFieldsPerProc-numColsPerProcChunk         
        
        # Determine how the loading and inner products will be split up.
        numColChunks = int(N.ceil(numCols*1./(numColsPerProcChunk*self.parallel.getNumProcs())))
        numRowChunks = int(N.ceil(numRows*1./(numRowsPerProcChunk*self.parallel.getNumProcs())))
        
        numColsPerChunk = int(N.ceil(numCols*1./numColChunks))
        numRowsPerChunk = int(N.ceil(numRows*1./numRowChunks))

        if self.parallel.isRankZero() and numRowChunks > 1 and self.verbose:
            print ('Warning: The column fields (direct ' +\
                'snapshots), of which there are %d, will be read %d times each. '+\
                'Increase number of ' +\
                'nodes or maxFieldsPerNode to reduce redundant loads and get a big speedup.') %\
                (numCols,numRowChunks)
        #print 'numColChunks',numColChunks,'numRowChunks',numRowChunks
        # Currently using a little trick to finding all of the inner product mat chunks
        # Each processor has a full innerProductMat with numRows x numCols even
        # though each processor is not responsible for filling in all of these entries
        # After each proc fills in what it is responsible for, the other entries are 0's 
        # still. Then, an allreduce is done and all the chunk mats are simply summed.
        # This is simpler than trying to figure out the size of each chunk mat for allgather.
        # The efficiency is not expected to be an issue, the size of the mats are
        # small compared to the size of the fields (at least in cases where
        # the data is big and memory is a constraint).
        innerProductMatChunk = N.mat(N.zeros((numRows,numCols)))
        for startRowIndex in xrange(0, numRows, numRowsPerChunk):
            endRowIndex = min(numRows,startRowIndex+numRowsPerChunk)
            # Convenience variable, has the rows which this rank is responsible for.
            procRowAssignments = \
                self.parallel.find_assignments(range(startRowIndex,endRowIndex))\
                [self.parallel.getRank()]
            if len(procRowAssignments)!=0:
                rowFields = [self.load_field(rowPath) \
                    for rowPath in rowFieldPaths[procRowAssignments[0]:procRowAssignments[-1]+1]]
            else:
                rowFields = []
            for startColIndex in xrange(0, numCols, numColsPerChunk):
                endColIndex = min(startColIndex+numColsPerChunk, numCols)
                procColAssignments = \
                    self.parallel.find_assignments(range(startColIndex,endColIndex))\
                    [self.parallel.getRank()]
                # Pass the col fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                colFieldsRecv = (None, None)
                if len(procColAssignments) > 0:
                    colIndices = range(procColAssignments[0],procColAssignments[-1]+1)
                else:
                    colIndices = []
                    
                for numPasses in xrange(self.parallel.getNumProcs()):
                    # If on the first pass, load the col fields, no send/recv
                    # This is all that is called when in serial, loop iterates once.
                    if numPasses == 0:
                        if len(colIndices) > 0:
                            colFields = [self.load_field(colPath) \
                                for colPath in colFieldPaths[colIndices[0]:colIndices[-1]+1]]
                        else:
                            colFields = []
            
                    else:
                        colFieldsSend = (colFields, colIndices)
                        dest = (self.parallel.getRank()+1) % self.parallel.getNumProcs()
                        #Create unique tag based on ranks
                        sendTag = self.parallel.getRank()*(self.parallel.getNumProcs()+1) + dest
                        self.parallel.comm.isend(colFieldsSend, dest=dest, tag=sendTag)
                        source = (self.parallel.getRank()-1) % self.parallel.getNumProcs()
                        recvTag = source*(self.parallel.getNumProcs()+1) + self.parallel.getRank()
                        colFieldsRecv = self.parallel.comm.recv(source=source, tag=recvTag)
                        self.parallel.sync()
                        colIndices = colFieldsRecv[1]
                        colFields = colFieldsRecv[0]
                        
                    # Compute the IPs for this set of data
                    # colIndices stores the indices of the innerProductMatChunk columns
                    # to be filled in.
                        
                    if len(procRowAssignments) > 0:
                        for rowIndex in xrange(procRowAssignments[0],procRowAssignments[-1]+1):
                            for colFieldIndex,colField in enumerate(colFields):
                                innerProductMatChunk[rowIndex,colIndices[colFieldIndex]] = \
                                  self.inner_product(rowFields[rowIndex-procRowAssignments[0]],colField)
                # Completed a chunk of rows and all columns on all processors.
            self._print_inner_product_progress(endRowIndex, numRows, numCols)
        
        # Assign these chunks into innerProductMat.
        if self.parallel.isDistributed():
            innerProductMat = self.parallel.custom_comm.allreduce( \
                innerProductMatChunk)
        else:
            innerProductMat = innerProductMatChunk 

        if transpose:
            innerProductMat = innerProductMat.T

        return innerProductMat
      


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
            #self._print_inner_product_progress(startRowIndex, endRowIndex,
            #    endColIndex, numRows, numCols, printAfterNumCols)

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
                    #self._print_inner_product_progress(startRowIndex, 
                    #    endRowIndex, endColIndex, numRows, numCols, 
                    #    printAfterNumCols)
 
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
            print ('Warning: The fields (snapshots), ' +\
                'of which there are %d, will be read multiple times. If possible, '+\
                'increase number of ' +\
                'nodes or maxFieldsPerNode to avoid this and get a big speedup.') %\
                numFields

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
            elif modeNum-indexFrom > fieldCoeffMat.shape[1]:
                raise ValueError(('Mode index, %d, is greater '+\
                    'than number of columns in the build coefficient '+\
                    'matrix, %d')%(modeNum-indexFrom,fieldCoeffMat.shape[1]))
        
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
        summed together to form the full outputs. The output sumFields 
        are then saved to file.
        
        Scaling is:
        
          num loads / proc = n_s/(n_p*max) * n_b/n_p
          
          passes/proc = n_s/(n_p*max) * (n_b*(n_p-1)/n_p)
          
          scalar multiplies/proc = n_s*n_b/n_p
          
        Where n_s is number of sum fields, n_b is number of basis fields,
        n_p is number of processors, max = maxFieldsPerNode-1.
        """
        if self.save_field is None:
            raise util.UndefinedError('save_field is undefined')
                   
        if not isinstance(sumFieldPaths, list):
            sumFieldPaths = [sumFieldPaths]
        if not isinstance(basisFieldPaths, list):
            basisFieldPaths = [basisFieldPaths]
        numBases = len(basisFieldPaths)
        numSums = len(sumFieldPaths)
        if numBases > fieldCoeffMat.shape[0]:
            raise ValueError(('Coeff mat has fewer rows %d than num of basis paths %d'\
                %(fieldCoeffMat.shape[0],numBases)))
                
        if numSums > fieldCoeffMat.shape[1]:
            raise ValueError(('Coeff matrix has fewer cols %d than num of ' +\
                'output paths %d')%(fieldCoeffMat.shape[1],numSums))
                               
        if numBases < fieldCoeffMat.shape[0] and self.parallel.isRankZero():
            print 'Warning - fewer basis paths than cols in the coeff matrix'
            print '  some rows of coeff matrix will not be used'
        if numSums < fieldCoeffMat.shape[1] and self.parallel.isRankZero():
            print 'Warning - fewer output paths than rows in the coeff matrix'
            print '  some cols of coeff matrix will not be used'
        
        # numBasesPerProcChunk is the number of bases each proc loads at once        
        numBasesPerProcChunk = 1
        numSumsPerProcChunk = self.maxFieldsPerProc-numBasesPerProcChunk         
        
        # This step can be done by find_assignments as well. Really what
        # this is doing is dividing the work into num*Chunks pieces.
        # find_assignments should take an optional arg of numWorkers or numPieces.
        # Determine how the loading and scalar multiplies will be split up.
        numBasisChunks = int(N.ceil(numBases*1./(numBasesPerProcChunk*self.parallel.getNumProcs())))
        numSumChunks = int(N.ceil(numSums*1./(numSumsPerProcChunk*self.parallel.getNumProcs())))
        
        numBasesPerChunk = int(N.ceil(numBases*1./numBasisChunks))
        numSumsPerChunk = int(N.ceil(numSums*1./numSumChunks))

        if self.parallel.isRankZero() and numSumChunks > 1 and self.verbose:
            print ('Warning: The basis fields (snapshots), ' +\
                'of which there are %d, will be loaded from file %d times each. If possible, '+\
                'increase number of ' +\
                'nodes or maxFieldsPerNode to reduce redundant loads and get a big speedup.') %\
                (numBases, numSumChunks)
               
        for startSumIndex in xrange(0, numSums, numSumsPerChunk):
            endSumIndex = min(startSumIndex+numSumsPerChunk, numSums)
            sumAssignments = self.parallel.find_assignments(range(startSumIndex,endSumIndex))
            procSumAssignments = sumAssignments[self.parallel.getRank()]
            # Create empty list on each processor
            sumLayers = [None for i in xrange(len(sumAssignments[self.parallel.getRank()]))]
            
            for startBasisIndex in xrange(0, numBases, numBasesPerChunk):
                endBasisIndex = min(startBasisIndex+numBasesPerChunk, numBases)
                basisAssignments = self.parallel.find_assignments(range(startBasisIndex,endBasisIndex))
                procBasisAssignments = basisAssignments[self.parallel.getRank()]
                # Pass the basis fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                basisFieldsRecv = (None, None)
                if len(procBasisAssignments) > 0:
                    basisIndices = range(procBasisAssignments[0],procBasisAssignments[-1]+1)
                else:
                    # this proc isn't responsible for loading any basis fields
                    basisIndices = []
                    
                for numPasses in xrange(self.parallel.getNumProcs()):
                    # If on the first pass, load the basis fields, no send/recv
                    # This is all that is called when in serial, loop iterates once.
                    if numPasses == 0:
                        if len(basisIndices) > 0:
                            basisFields = [self.load_field(basisPath) \
                                for basisPath in basisFieldPaths[basisIndices[0]:basisIndices[-1]+1]]
                        else: basisFields = []
                    else:
                        basisFieldsSend = (basisFields, basisIndices)
                        dest = (self.parallel.getRank()+1) % self.parallel.getNumProcs()
                        #Create unique tag based on ranks
                        sendTag = self.parallel.getRank()*(self.parallel.getNumProcs()+1) + dest
                        self.parallel.comm.isend(basisFieldsSend, dest=dest, tag=sendTag)
                        source = (self.parallel.getRank()-1) % self.parallel.getNumProcs()
                        recvTag = source*(self.parallel.getNumProcs()+1) + self.parallel.getRank()
                        basisFieldsRecv = self.parallel.comm.recv(source=source, tag=recvTag)
                        self.parallel.sync()
                        basisIndices = basisFieldsRecv[1]
                        basisFields = basisFieldsRecv[0]
                    
                    # Compute the scalar multiplications for this set of data
                    # basisIndices stores the indices of the fieldCoeffMat to use.
                    
                    for sumIndex in xrange(len(procSumAssignments)):
                        for basisIndex,basisField in enumerate(basisFields):
                            sumLayer = basisField*\
                                fieldCoeffMat[basisIndices[basisIndex],\
                                sumIndex+procSumAssignments[0]]
                            if sumLayers[sumIndex] is None:
                                sumLayers[sumIndex] = sumLayer
                            else:
                                sumLayers[sumIndex] += sumLayer
            # Completed this set of sum fields, save to file
            for sumIndex in xrange(len(procSumAssignments)):
                self.save_field(sumLayers[sumIndex],\
                    sumFieldPaths[sumIndex+procSumAssignments[0]])
            if self.parallel.isRankZero() and self.verbose:    
                print >> sys.stderr, ('Completed %.1f%% of sum fields, %d ' +\
                    'of %d') % (endSumIndex*100./numSums,endSumIndex,numSums)

    def __eq__(self, other):
        #print 'comparing fieldOperations classes'
        a = (self.inner_product == other.inner_product and \
        self.load_field == other.load_field and self.save_field == other.save_field \
        and self.maxFieldsPerNode==other.maxFieldsPerNode and\
        self.verbose==other.verbose)
        return a
    def __ne__(self,other):
        return not (self.__eq__(other))

