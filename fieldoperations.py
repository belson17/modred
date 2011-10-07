
"""Collection of low level functions for modaldecomp library"""

import sys  
import copy
import numpy as N
import util
import parallel as parallel_mod
import time as T

# Should this be a data member? Currently it is
#parallel = parallel_mod.parallelInstance

class FieldOperations(object):
    """
    Does many useful and low level operations on fields.

    All modaldecomp classes should use the common functionality provided
    in this class as much as possible.
    
    Only advanced users should use this class, it is mostly a collection
    of functions used in the high level modaldecomp classes like POD,
    BPOD, and DMD.
    
    It is generally best to use all available processors for this class,
    however this
    depends on the computer and the nature of the load and inner_product functions
    supplied. In some cases, loading in parallel is slower.
    """
    
    def __init__(self, load_field=None, save_field=None, inner_product=None, 
        maxFieldsPerNode=None, verbose=True, printInterval=10):
        """
        Sets the default values for data members. 
        
        BPOD, POD, DMD, other classes, should make heavy use of the low
        level functions in this class.
        Arguments:
          maxFieldsPerNode: maximum number of fields that can be in memory
            simultaneously on a node.
          verbose: true/false, sets if warnings are printed or not
          printInterval: seconds, maximum of how frequently progress is printed
        """
        self.load_field = load_field
        self.save_field = save_field
        self.inner_product = inner_product
        self.verbose = verbose 
        self.printInterval = printInterval
        self.prevPrintTime = 0.
        
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


  
    def compute_inner_product_mat(self, rowFieldPaths, colFieldPaths):
        """ 
        Computes a matrix of inner products (for BPOD, Y'*X) and returns it.
        
          rowFieldPaths: row snapshot files (BPOD adjoint snaps, ~Y)
          
          colFieldPaths: column snapshot files (BPOD direct snaps, ~X)

        Within this method, the snapshots are read in memory-efficient ways
        such that they are not all in memory at once. This results in finding
        'chunks' of the eventual matrix that is returned.  This method only
        supports finding a full rectangular mat. For POD, a different method is
        used to take advantage of the symmetric matrix.
        
        Each processor is responsible for loading a subset of the rows and
        columns. The processor which reads a particular column field then sends
        it to each successive processor so it can be used to compute all IPs
        for the current row chunk on each processor. This is repeated until all
        processors are done with all of their row chunks. If there are 2
        processors::
           
                | r0c0 o  |
          rank0 | r1c0 o  |
                | r2c0 o  |
            -
                | o  r3c1 |
          rank1 | o  r4c1 |
                | o  r5c1 |
        
        Rank 0 reads column 0 (c0) and fills out IPs for all rows in a row
        chunk (r*c*) Here there is only one row chunk for each processor for
        simplicity.  Rank 1 reads column 1 (c1) and fills out IPs for all rows
        in a row chunk.  In the next step, rank 0 sends c0 to rank 1 and rank 1
        sends c1 to rank 1.  The remaining IPs are filled in::
        
                | r0c0 r0c1 |
          rank0 | r1c0 r1c1 |
                | r2c0 r2c1 |
            -
                | r3c0 r3c1 |
          rank1 | r4c0 r4c1 |
                | r5c0 r5c1 |
          
        This is more complicated when the number of cols and rows is
        not divisible by the number of processors. This is handled
        internally, by allowing the last processor to have fewer tasks, however
        it is still part of the passing circle, and rows and cols are handled
        independently.  This is also generalized to allow the columns to be
        read in chunks, rather than only 1 at a time.  This could be useful,
        for example, in a shared memory setting where it is best to work in
        operation-units (loads, IPs, etc) of multiples of procs/node.
        
        The scaling is:
        
            num loads / processor ~ (n_r/(max*n_p))*n_c/n_p + n_r/n_p
            
            num MPI sends / processor ~ (n_r/(max*n_p))*(n_p-1)*n_c/n_p
            
            num inner products / processor ~ n_r*n_c/n_p
            
        where n_r is number of rows, n_c number of columns, max is
        maxFieldsPerProc-1 = maxFieldsPerNode/numNodesPerProc - 1, and n_p is
        number of processors.
        
        It is enforced that there are more columns than rows by doing an
        internal transpose and un-transpose. This improves efficiency.
        
        From these scaling laws, it can be seen that it is generally good to
        use all available processors, even though it lowers max.  This depends
        though on the particular system and hardward.
        Sometimes multiple simultaneous loads actually makes each load very slow. 
        
        As an example, consider doing a case with len(rowFieldPaths)=8 and
        len(colFieldPaths) = 12, 2 processors, 1 node, and maxFieldsPerNode=3.
        n_p=2, max=2, n_r=8, n_c=12 (n_r < n_c).
        
            num loads / proc = 16
            
        If we flip n_r and n_c, we get
        
            num loads / proc = 18.
            
        """
             
        if not isinstance(rowFieldPaths,list):
            rowFieldPaths = [rowFieldPaths]
        if not isinstance(colFieldPaths,list):
            colFieldPaths = [colFieldPaths]
            
        numCols = len(colFieldPaths)
        numRows = len(rowFieldPaths)

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
        
        # Estimate the amount of time this will take
        if self.verbose and self.parallel.isRankZero():
            rowField = self.load_field(rowFieldPaths[0])
            colField = self.load_field(colFieldPaths[0])
            startTime = T.time()
            ip = self.inner_product(rowField, colField)
            endTime = T.time()
            duration = endTime - startTime
            print ('Computing the inner product matrix will take at least %.1f '+
                'minutes')%(numRows*numCols*duration/(60.*self.parallel.getNumProcs()))
            del rowField, colField
        # numColsPerProcChunk is the number of cols each proc loads at once        
        numColsPerProcChunk = 1
        numRowsPerProcChunk = self.maxFieldsPerProc-numColsPerProcChunk         
        
        # Determine how the loading and inner products will be split up.
        # These variables are the total number of chunks of data to be read across
        # all nodes and processors
        numColChunks = int(N.ceil(numCols * 1. / (numColsPerProcChunk * self.\
            parallel.getNumProcs())))
        numRowChunks = int(N.ceil(numRows * 1. / (numRowsPerProcChunk * self.\
            parallel.getNumProcs())))
        
        
        ### MAYBE DON'T USE THIS, IT EVENLY DISTRIBUTES CHUNKSIZE, WHICH WA
        ### ALREADY SET ABOVE TO BE NUMCOLSPERPROCCHUNK * NUMPROCS
        # These variables are the number of cols and rows in each chunk of data.
        numColsPerChunk = int(N.ceil(numCols * 1. / numColChunks))
        numRowsPerChunk = int(N.ceil(numRows * 1. / numRowChunks))

        if self.parallel.isRankZero() and numRowChunks > 1 and self.verbose:
            print ('Warning: The column fields, of which ' +\
                'there are %d, will be read %d times each. Increase number ' +\
                'of nodes or maxFieldsPerNode to reduce redundant loads and ' +\
                'get a big speedup.') % (numCols,numRowChunks)
        #print 'numColChunks',numColChunks,'numRowChunks',numRowChunks
        
        # Currently using a little trick to finding all of the inner product
        # mat chunks. Each processor has a full innerProductMat with size
        # numRows x numCols even though each processor is not responsible for
        # filling in all of these entries. After each proc fills in what it is
        # responsible for, the other entries are 0's still. Then, an allreduce
        # is done and all the chunk mats are simply summed.  This is simpler
        # than trying to figure out the size of each chunk mat for allgather.
        # The efficiency is not expected to be an issue, the size of the mats
        # are small compared to the size of the fields (at least in cases where
        # the data is big and memory is a constraint).
        innerProductMatChunk = N.mat(N.zeros((numRows,numCols)))
        for startRowIndex in xrange(0, numRows, numRowsPerChunk):
            endRowIndex = min(numRows,startRowIndex+numRowsPerChunk)
            # Convenience variable, has the rows which this rank is responsible
            # for.
            procRowAssignments = self.parallel.find_assignments(range(
                startRowIndex, endRowIndex))[self.parallel.getRank()]
            if len(procRowAssignments)!=0:
                rowFields = [self.load_field(rowPath) for rowPath in \
                    rowFieldPaths[procRowAssignments[0]:procRowAssignments[
                    -1] + 1]]
            else:
                rowFields = []
            for startColIndex in xrange(0, numCols, numColsPerChunk):
                endColIndex = min(startColIndex+numColsPerChunk, numCols)
                procColAssignments = \
                    self.parallel.find_assignments(range(startColIndex, 
                        endColIndex))[self.parallel.getRank()]
                # Pass the col fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                colFieldsRecv = (None, None)
                if len(procColAssignments) > 0:
                    colIndices = range(procColAssignments[0], 
                        procColAssignments[-1]+1)
                else:
                    colIndices = []
                    
                for numPasses in xrange(self.parallel.getNumProcs()):
                    # If on the first pass, load the col fields, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if numPasses == 0:
                        if len(colIndices) > 0:
                            colFields = [self.load_field(colPath) \
                                for colPath in colFieldPaths[colIndices[0]:\
                                    colIndices[-1] + 1]]
                        else:
                            colFields = []
                    else:
                        # Determine whom to communicate with
                        dest = (self.parallel.getRank() + 1) % self.parallel.\
                            getNumProcs()
                        source = (self.parallel.getRank() - 1) % self.parallel.\
                            getNumProcs()    
                            
                        #Create unique tag based on ranks
                        sendTag = self.parallel.getRank() * (self.parallel.\
                            getNumProcs() + 1) + dest
                        recvTag = source*(self.parallel.getNumProcs() + 1) +\
                            self.parallel.getRank()
                        
                        # Collect data and send/receive
                        colFieldsSend = (colFields, colIndices)    
                        request = self.parallel.comm.isend(colFieldsSend, dest=\
                            dest, tag=sendTag)
                        colFieldsRecv = self.parallel.comm.recv(source=source, 
                            tag=recvTag)
                        request.Wait()
                        self.parallel.sync()
                        colIndices = colFieldsRecv[1]
                        colFields = colFieldsRecv[0]
                        
                    # Compute the IPs for this set of data colIndices stores
                    # the indices of the innerProductMatChunk columns to be
                    # filled in.
                    if len(procRowAssignments) > 0:
                        for rowIndex in xrange(procRowAssignments[0],
                            procRowAssignments[-1]+1):
                            for colFieldIndex,colField in enumerate(colFields):
                                innerProductMatChunk[rowIndex, colIndices[
                                    colFieldIndex]] = self.inner_product(
                                    rowFields[rowIndex - procRowAssignments[0]],
                                    colField)
            # Completed a chunk of rows and all columns on all processors.
            if ((T.time() - self.prevPrintTime > self.printInterval) and 
                self.verbose and self.parallel.isRankZero()):
                numCompletedIPs = endRowIndex * numCols
                percentCompletedIPs = 100. * numCompletedIPs/(numCols*numRows)           
                print >> sys.stderr, ('Completed %.1f%% of inner ' +\
                    'products: IPMat[:%d, :%d] of IPMat[%d, %d]') % \
                    (percentCompletedIPs, endRowIndex, numCols, numRows, numCols)
                self.prevPrintTime = T.time()
        
        # Assign these chunks into innerProductMat.
        if self.parallel.isDistributed():
            innerProductMat = self.parallel.custom_comm.allreduce( 
                innerProductMatChunk)
        else:
            innerProductMat = innerProductMatChunk 

        if transpose:
            innerProductMat = innerProductMat.T

        return innerProductMat

        
    def compute_symmetric_inner_product_mat(self, fieldPaths):
        """
        Computes an upper-triangular chunk of a symmetric matrix of inner 
        products.  
        """
        if isinstance(fieldPaths, str):
            fieldPaths = [fieldPaths]
 
        numFields = len(fieldPaths)
        
        
        # numColsPerChunk is the number of cols each proc loads at once.  
        # Columns are loaded if the matrix must be broken up into sets of 
        # chunks.  Then symmetric upper triangular portions will be computed,
        # followed by a rectangular piece that uses columns not already loaded.
        numColsPerProcChunk = 1
        numRowsPerProcChunk = self.maxFieldsPerProc - numColsPerProcChunk
 
        # <nprocs> chunks are computed simulaneously, making up a set.
        numColsPerChunk = numColsPerProcChunk * self.parallel.getNumProcs()
        numRowsPerChunk = numRowsPerProcChunk * self.parallel.getNumProcs()

        # <numRowChunks> is the number of sets that must be computed.
        numRowChunks = int(N.ceil(numFields * 1. / numRowsPerChunk)) 
        if self.parallel.isRankZero() and numRowChunks > 1 and self.verbose:
            print ('Warning: The column fields will be read ~%d times each. ' +\
                'Increase number of nodes or maxFieldsPerNode to reduce ' +\
                'redundant loads and get a big speedup.') % numRowChunks    
        
        # Use the same trick as in compute_inner_product_mat, having each proc
        # fill in elements of a numRows x numRows sized matrix, rather than
        # assembling small chunks. This is done for the triangular portions. For
        # the rectangular portions, the inner product mat is filled in directly.
        innerProductMatChunk = N.mat(N.zeros((numFields, numFields)))
        for startRowIndex in xrange(0, numFields, numRowsPerChunk):
            endRowIndex = min(numFields, startRowIndex + numRowsPerChunk)
            procRowAssignments_all = self.parallel.find_assignments(range(
                startRowIndex, endRowIndex))
            numActiveProcs = len([assignment for assignment in \
                procRowAssignments_all if assignment != []])
            procRowAssignments = procRowAssignments_all[self.parallel.getRank()]
            if len(procRowAssignments)!=0:
                rowFields = [self.load_field(path) for path in fieldPaths[
                    procRowAssignments[0]:procRowAssignments[-1] + 1]]
            else:
                rowFields = []
            
            # Triangular chunks
            if len(procRowAssignments) > 0:
                # Test that indices are consecutive
                if procRowAssignments[0:] != range(procRowAssignments[0], 
                    procRowAssignments[-1] + 1):
                    raise ValueError('Indices are not consecutive.')
                
                # Per-processor triangles (using only loaded snapshots)
                for rowIndex in xrange(procRowAssignments[0], 
                    procRowAssignments[-1] + 1):
                    # Diagonal term
                    innerProductMatChunk[rowIndex, rowIndex] = self.\
                        inner_product(rowFields[rowIndex - procRowAssignments[
                        0]], rowFields[rowIndex - procRowAssignments[0]])
                        
                    # Off-diagonal terms
                    for colIndex in xrange(rowIndex + 1, procRowAssignments[
                        -1] + 1):
                        innerProductMatChunk[rowIndex, colIndex] = self.\
                            inner_product(rowFields[rowIndex -\
                            procRowAssignments[0]], rowFields[colIndex -\
                            procRowAssignments[0]])
               
            # Number of square chunks to fill in is n * (n-1) / 2.  At each
            # iteration we fill in n of them, so we need (n-1) / 2 
            # iterations (round up).  
            for setIndex in xrange(int(N.ceil((numActiveProcs - 1.) / 2))):
                # The current proc is "sender"
                myRank = self.parallel.getRank()
                myRowIndices = procRowAssignments
                myNumRows = len(myRowIndices)
                                       
                # The proc to send to is "destination"                         
                destRank = (myRank + setIndex + 1) % numActiveProcs
                destRowIndices = procRowAssignments_all[destRank]
                
                # The proc that data is received from is the "source"
                sourceRank = (myRank - setIndex - 1) % numActiveProcs
                
                # Find the maximum number of sends/recv to be done by any proc
                maxNumToSend = int(N.ceil(1. * max([len(assignments) for \
                    assignments in procRowAssignments_all]) /\
                    numColsPerProcChunk))
                
                # Pad assignments with nan so that everyone has the same
                # number of things to send.  Same for list of fields with None.             
                # The empty lists will not do anything when enumerated, so no 
                # inner products will be taken.  nan is inserted into the 
                # indices because then min/max of the indices can be taken.
                """
                if myNumRows != len(rowFields):
                    raise ValueError('Number of rows assigned does not ' +\
                        'match number of loaded fields.')
                if myNumRows > 0 and myNumRows < maxNumToSend:
                    myRowIndices += [N.nan] * (maxNumToSend - myNumRows) 
                    rowFields += [[]] * (maxNumToSend - myNumRows)
                """
                for sendIndex in xrange(maxNumToSend):
                    # Only processors responsible for rows communicate
                    if myNumRows > 0:  
                        # Send row fields, in groups of numColsPerProcChunk
                        # These become columns in the ensuing computation
                        startColIndex = sendIndex * numColsPerProcChunk
                        endColIndex = min(startColIndex + numColsPerProcChunk, 
                            myNumRows)   
                        colFieldsSend = (rowFields[startColIndex:endColIndex], 
                            myRowIndices[startColIndex:endColIndex])
                        
                        # Create unique tags based on ranks
                        sendTag = myRank * (self.parallel.getNumProcs() + 1) +\
                            destRank
                        recvTag = sourceRank * (self.parallel.getNumProcs() +\
                            1) + myRank
                        
                        # Send and receieve data.  It is important that we put a
                        # Wait() command after the receive.  In testing, when 
                        # this was not done, we saw a race condition.  This was a
                        # condition that could not be fixed by a sync(). It 
                        # appears that the Wait() is very important for the non-
                        # blocking send.
                        request = self.parallel.comm.isend(colFieldsSend, 
                            dest=destRank, tag=sendTag)                        
                        colFieldsRecv = self.parallel.comm.recv(source=\
                            sourceRank, tag=recvTag)
                        request.Wait()
                        colFields = colFieldsRecv[0]
                        myColIndices = colFieldsRecv[1]
                        
                        for rowIndex in xrange(myRowIndices[0], 
                            myRowIndices[-1] + 1):
                            for colFieldIndex, colField in enumerate(colFields):
                                innerProductMatChunk[rowIndex, myColIndices[
                                    colFieldIndex]] = self.inner_product(
                                    rowFields[rowIndex - myRowIndices[0]],
                                    colField)
                                   
                    # Sync after send/receive   
                    self.parallel.sync()  
                
            
            # Fill in the rectangular portion next to each triangle (if nec.).
            # Start at index after last row, continue to last column. This part
            # of the code is the same as in compute_inner_product_mat, as of 
            # revision 141.  
            for startColIndex in xrange(endRowIndex, numFields, 
                numColsPerChunk):
                endColIndex = min(startColIndex + numColsPerChunk, numFields)
                procColAssignments = self.parallel.find_assignments(range(
                    startColIndex, endColIndex))[self.parallel.getRank()]
                        
                # Pass the col fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                colFieldsRecv = (None, None)
                if len(procColAssignments) > 0:
                    colIndices = range(procColAssignments[0], 
                        procColAssignments[-1]+1)
                else:
                    colIndices = []
                    
                for numPasses in xrange(self.parallel.getNumProcs()):
                    # If on the first pass, load the col fields, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if numPasses == 0:
                        if len(colIndices) > 0:
                            colFields = [self.load_field(colPath) \
                                for colPath in fieldPaths[colIndices[0]:\
                                    colIndices[-1] + 1]]
                        else:
                            colFields = []
                    else: 
                        # Determine whom to communicate with
                        dest = (self.parallel.getRank() + 1) % self.parallel.\
                            getNumProcs()
                        source = (self.parallel.getRank() - 1) % self.parallel.\
                            getNumProcs()    
                            
                        #Create unique tag based on ranks
                        sendTag = self.parallel.getRank() * (self.parallel.\
                            getNumProcs() + 1) + dest
                        recvTag = source*(self.parallel.getNumProcs() + 1) +\
                            self.parallel.getRank()    
                        
                        # Collect data and send/receive
                        colFieldsSend = (colFields, colIndices)     
                        request = self.parallel.comm.isend(colFieldsSend, dest=\
                            dest, tag=sendTag)
                        colFieldsRecv = self.parallel.comm.recv(source=source, 
                            tag=recvTag)
                        request.Wait()
                        self.parallel.sync()
                        colIndices = colFieldsRecv[1]
                        colFields = colFieldsRecv[0]
                        
                    # Compute the IPs for this set of data colIndices stores
                    # the indices of the innerProductMatChunk columns to be
                    # filled in.
                    if len(procRowAssignments) > 0:
                        for rowIndex in xrange(procRowAssignments[0],
                            procRowAssignments[-1]+1):
                            for colFieldIndex,colField in enumerate(colFields):
                                innerProductMatChunk[rowIndex, colIndices[
                                    colFieldIndex]] = self.inner_product(
                                    rowFields[rowIndex - procRowAssignments[0]],
                                    colField)
            # Completed a chunk of rows and all columns on all processors.
            if (self.verbose and (T.time() - self.prevPrintTime > self.printInterval) and
                self.parallel.isRankZero()):
                numCompletedIPs = endRowIndex*numCols - endRowIndex**2 *.5
                percentCompletedIPs = 100. * numCompletedIPs/(.5*numCols*numRows)           
                print >> sys.stderr, ('Completed %.1f%% of inner ' +\
                    'products') % (percentCompletedIPs, endRowIndex, 
                    numCols, numRows, numCols)

                self.prevPrintTime = T.time()
                             
        # Assign the triangular portion chunks into innerProductMat.
        if self.parallel.isDistributed():
            innerProductMat = self.parallel.custom_comm.allreduce( 
                innerProductMatChunk)
        else:
            innerProductMat = innerProductMatChunk

        # Create a mask for the repeated values
        mask = innerProductMat != innerProductMat.T
        
        # Collect values below diagonal
        innerProductMat += N.multiply(N.triu(innerProductMat.T, 1), mask)
        
        # Symmetrize matrix
        innerProductMat = N.triu(innerProductMat) + N.triu(innerProductMat, 1).T
        
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
            print 'Warning: fewer basis paths than cols in the coeff matrix'
            print '  some rows of coeff matrix will not be used'
        if numSums < fieldCoeffMat.shape[1] and self.parallel.isRankZero():
            print 'Warning: fewer output paths than rows in the coeff matrix'
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
                        # Figure out whom to communicate with
                        source = (self.parallel.getRank()-1) % self.parallel.getNumProcs()
                        dest = (self.parallel.getRank()+1) % self.parallel.getNumProcs()
                        
                        #Create unique tags based on ranks
                        sendTag = self.parallel.getRank()*(self.parallel.getNumProcs()+1) + dest
                        recvTag = source*(self.parallel.getNumProcs()+1) + self.parallel.getRank()
                        
                        # Send/receive data
                        basisFieldsSend = (basisFields, basisIndices)
                        request = self.parallel.comm.isend(basisFieldsSend, dest=dest, tag=sendTag)                       
                        basisFieldsRecv = self.parallel.comm.recv(source=source, tag=recvTag)
                        request.Wait()
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
            if self.parallel.isRankZero() and self.verbose and T.time()-self.prevPrintTime>self.printInterval:    
                print >> sys.stderr, ('Completed %.1f%% of sum fields, %d ' +\
                    'of %d') % (endSumIndex*100./numSums,endSumIndex,numSums)
                self.prevPrintTime = T.time()


    def __eq__(self, other):
        #print 'comparing fieldOperations classes'
        a = (self.inner_product == other.inner_product and \
        self.load_field == other.load_field and self.save_field == other.save_field \
        and self.maxFieldsPerNode==other.maxFieldsPerNode and\
        self.verbose==other.verbose)
        return a

    def __ne__(self,other):
        return not (self.__eq__(other))

