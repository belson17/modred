
"""Collection of low level functions for modaldecomp library"""

import sys  
import copy
import time as T
import numpy as N
import util
import parallel as parallel_mod

class FieldOperations(object):
    """
    Responsible for low level (parallel) operations on fields.

    All modaldecomp classes should use the common functionality provided
    in this class as much as possible.
    
    Only advanced users should use this class; it is mostly a collection
    of functions used in the high-level modaldecomp classes like POD,
    BPOD, and DMD.
    
    It is generally best to use all available processors for this class,
    however thisa
    depends on the computer and the nature of the load and inner_product functions
    supplied. In some cases, loading in parallel is slower.
    """
    
    def __init__(self, load_field=None, save_field=None, inner_product=None, 
        max_fields_per_node=None, verbose=True, print_interval=10):
        """
        Sets the default values for data members. 
        
        Arguments:
          max_fields_per_node: maximum number of fields that can be in memory
            simultaneously on a node.
          verbose: true/false, sets if warnings are printed or not
          print_interval: seconds, maximum of how frequently progress is printed
        """
        self.load_field = load_field
        self.save_field = save_field
        self.inner_product = inner_product
        self.verbose = verbose 
        self.print_interval = print_interval
        self.prev_print_time = 0.
        self.parallel = parallel_mod.default_instance
        
        if max_fields_per_node is None:
            self.max_fields_per_node = 2
            self.print_msg('Warning: max_fields_per_node was not specified. '
                'Assuming 2 fields can be loaded per node. Increase '
                'max_fields_per_node for a speedup.')
        else:
            self.max_fields_per_node = max_fields_per_node
        
        if self.max_fields_per_node < \
            2 * self.parallel.get_num_procs() / self.parallel.get_num_nodes(): 
            self.max_fields_per_proc = 2
            self.print_msg('Warning: max_fields_per_node too small for given '
                'number of nodes and procs.  Assuming 2 fields can be '
                'loaded per processor. Increase max_fields_per_node for a '
                'speedup.')
        else:
            self.max_fields_per_proc = self.max_fields_per_node * \
                self.parallel.get_num_nodes()/self.parallel.get_num_procs()
    
    def print_msg(self, msg, output_channel=sys.stdout):
        if self.verbose and self.parallel.is_rank_zero():
            print >> sys.stdout, msg

    def idiot_check(self, test_obj=None, test_obj_path=None):
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
        if test_obj_path is not None:
          test_obj = self.load_field(test_obj_path)
        if test_obj is None:
            raise RuntimeError('Supply field object or path for idiot check!')
        obj_copy = copy.deepcopy(test_obj)
        obj_copy_mag2 = self.inner_product(obj_copy, obj_copy)
        
        factor = 2.
        objMult = test_obj * factor
        
        if abs(self.inner_product(objMult, objMult) -
                obj_copy_mag2 * factor**2) > tol:
            raise ValueError('Multiplication of snap/mode object failed')
        
        if abs(self.inner_product(test_obj, test_obj) - 
                obj_copy_mag2) > tol:  
            raise ValueError('Original object modified by multiplication!') 
        objAdd = test_obj + test_obj
        if abs(self.inner_product(objAdd, objAdd) - obj_copy_mag2 * 4) > tol:
            raise ValueError('Addition does not give correct result')
        
        if abs(self.inner_product(test_obj, test_obj) - obj_copy_mag2) > tol:  
            raise ValueError('Original object modified by addition!')       
        
        objAddMult = test_obj * factor + test_obj
        if abs(self.inner_product(objAddMult, objAddMult) - obj_copy_mag2 *
                (factor + 1) ** 2) > tol:
            raise ValueError('Multiplication and addition of snap/mode are '+\
                'inconsistent')
        
        if abs(self.inner_product(test_obj, test_obj) - obj_copy_mag2) > tol:  
            raise ValueError('Original object modified by combo of mult/add!') 
        
        #objSub = 3.5*test_obj - test_obj
        #N.testing.assert_array_almost_equal(objSub,2.5*test_obj)
        #N.testing.assert_array_almost_equal(test_obj,obj_copy)
        self.print_msg('Passed the idiot check')


    def compute_inner_product_mat(self, row_field_paths, col_field_paths):
        """ 
        Computes a matrix of inner products (for BPOD, Y'*X) and returns it.
        
          row_field_paths: row snapshot files (BPOD adjoint snaps, ~Y)
          
          col_field_paths: column snapshot files (BPOD direct snaps, ~X)

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
        max_fields_per_proc-1 = max_fields_per_node/numNodesPerProc - 1, and n_p is
        number of processors.
        
        It is enforced that there are more columns than rows by doing an
        internal transpose and un-transpose. This improves efficiency.
        
        From these scaling laws, it can be seen that it is generally good to
        use all available processors, even though it lowers max.  This depends
        though on the particular system and hardward.
        Sometimes multiple simultaneous loads actually makes each load very slow. 
        
        As an example, consider doing a case with len(row_field_paths)=8 and
        len(col_field_paths) = 12, 2 processors, 1 node, and max_fields_per_node=3.
        n_p=2, max=2, n_r=8, n_c=12 (n_r < n_c).
        
            num loads / proc = 16
            
        If we flip n_r and n_c, we get
        
            num loads / proc = 18.
            
        """
             
        if not isinstance(row_field_paths,list):
            row_field_paths = [row_field_paths]
        if not isinstance(col_field_paths,list):
            col_field_paths = [col_field_paths]
            
        num_cols = len(col_field_paths)
        num_rows = len(row_field_paths)

        if num_rows > num_cols:
            transpose = True
            temp = row_field_paths
            row_field_paths = col_field_paths
            col_field_paths = temp
            temp = num_rows
            num_rows = num_cols
            num_cols = temp
        else: 
            transpose = False
       
        # Compute a single inner product in order to determine matrix datatype
        # (real or complex) and to estimate the amount of time the IPs will take.
        row_field = self.load_field(row_field_paths[0])
        col_field = self.load_field(col_field_paths[0])
        start_time = T.time()
        IP = self.inner_product(row_field, col_field)
        IP_type = type(IP)
        end_time = T.time()

        # Estimate the amount of time this will take
        duration = end_time - start_time
        self.print_msg('Computing the inner product matrix will take at least '
                    '%.1f minutes' % (num_rows * num_cols * duration / 
                    (60. * self.parallel.get_num_procs())))
        del row_field, col_field

        # num_cols_per_proc_chunk is the number of cols each proc loads at once        
        num_cols_per_proc_chunk = 1
        num_rows_per_proc_chunk = self.max_fields_per_proc - num_cols_per_proc_chunk         
        
        # Determine how the loading and inner products will be split up.
        # These variables are the total number of chunks of data to be read 
        # across all nodes and processors
        num_col_chunks = int(N.ceil(
                (1.*num_cols) / (num_cols_per_proc_chunk *
                 self.parallel.get_num_procs())))
        num_row_chunks = int(N.ceil(
                (1.*num_rows) / (num_rows_per_proc_chunk * 
                 self.parallel.get_num_procs())))
        
        
        ### MAYBE DON'T USE THIS, IT EVENLY DISTRIBUTES CHUNKSIZE, WHICH WA
        ### ALREADY SET ABOVE TO BE num_colsPERPROCCHUNK * NUMPROCS
        # These variables are the number of cols and rows in each chunk of data.
        num_cols_per_chunk = int(N.ceil(num_cols * 1. / num_col_chunks))
        num_rows_per_chunk = int(N.ceil(num_rows * 1. / num_row_chunks))

        if num_row_chunks > 1:
            self.print_msg('Warning: The column fields, of which '
                    'there are %d, will be read %d times each. Increase '
                    ' number '
                    'of nodes or max_fields_per_node to reduce redundant '
                    'loads and get a big speedup.' % (num_cols,num_row_chunks))
        
        # Currently using a little trick to finding all of the inner product
        # mat chunks. Each processor has a full IP_mat with size
        # num_rows x num_cols even though each processor is not responsible for
        # filling in all of these entries. After each proc fills in what it is
        # responsible for, the other entries are 0's still. Then, an allreduce
        # is done and all the chunk mats are simply summed.  This is simpler
        # than trying to figure out the size of each chunk mat for allgather.
        # The efficiency is not expected to be an issue, the size of the mats
        # are small compared to the size of the fields (at least in cases where
        # the data is big and memory is a constraint).
        IP_mat_chunk = N.mat(N.zeros((num_rows, num_cols), dtype=IP_type))
        for start_row_index in xrange(0, num_rows, num_rows_per_chunk):
            end_row_index = min(num_rows, start_row_index + num_rows_per_chunk)
            # Convenience variable containing the rows which this rank is
            # responsible for.
            proc_row_tasks = self.parallel.find_assignments(range(
                   start_row_index, end_row_index))[self.parallel.get_rank()]
            if len(proc_row_tasks) != 0:
                row_fields = [self.load_field(row_path) for row_path in 
                    row_field_paths[proc_row_tasks[0]:
                    proc_row_tasks[-1] + 1]]
            else:
                row_fields = []
            for start_col_index in xrange(0, num_cols, num_cols_per_chunk):
                end_col_index = min(
                    start_col_index + num_cols_per_chunk, num_cols)
                proc_col_tasks = self.parallel.find_assignments(range(
                    start_col_index, end_col_index))[self.parallel.get_rank()]
                # Pass the col fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                col_fields_recv = (None, None)
                if len(proc_col_tasks) > 0:
                    col_indices = range(proc_col_tasks[0], proc_col_tasks[-1]+1)
                else:
                    col_indices = []
                    
                for num_passes in xrange(self.parallel.get_num_procs()):
                    # If on the first pass, load the col fields, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if num_passes == 0:
                        if len(col_indices) > 0:
                            col_fields = [self.load_field(col_path) 
                                for col_path in col_field_paths[col_indices[0]:
                                    col_indices[-1] + 1]]
                        else:
                            col_fields = []
                    else:
                        # Determine whom to communicate with
                        dest = (self.parallel.get_rank() + 1) % \
                             self.parallel.get_num_procs()
                        source = (self.parallel.get_rank() - 1) % \
                            self.parallel.get_num_procs()    
                            
                        #Create unique tag based on ranks
                        send_tag = self.parallel.get_rank() * \
                                (self.parallel.get_num_procs() + 1) + dest
                        recv_tag = source * \
                            (self.parallel.get_num_procs() + 1) + \
                            self.parallel.get_rank()
                        
                        # Collect data and send/receive
                        col_fields_send = (col_fields, col_indices)    
                        request = self.parallel.comm.isend(
                            col_fields_send, dest=dest, tag=send_tag)
                        col_fields_recv = self.parallel.comm.recv(
                            source=source, tag=recv_tag)
                        request.Wait()
                        self.parallel.sync()
                        col_indices = col_fields_recv[1]
                        col_fields = col_fields_recv[0]
                        
                    # Compute the IPs for this set of data col_indices stores
                    # the indices of the IP_mat_chunk columns to be
                    # filled in.
                    if len(proc_row_tasks) > 0:
                        for row_index in xrange(proc_row_tasks[0],
                            proc_row_tasks[-1]+1):
                            for col_field_index,col_field in enumerate(col_fields):
                                IP_mat_chunk[row_index, col_indices[
                                    col_field_index]] = self.inner_product(
                                    row_fields[row_index - proc_row_tasks[0]],
                                    col_field)
            # Completed a chunk of rows and all columns on all processors.
            if ((T.time() - self.prev_print_time > self.print_interval) and 
                self.verbose and self.parallel.is_rank_zero()):
                num_completed_IPs = end_row_index * num_cols
                percent_completed_IPs = 100. * num_completed_IPs/(num_cols*num_rows)           
                print >> sys.stderr, ('Completed %.1f%% of inner ' +\
                    'products: IPMat[:%d, :%d] of IPMat[%d, %d]') % \
                    (percent_completed_IPs, end_row_index, num_cols, num_rows, num_cols)
                self.prev_print_time = T.time()
        
        # Assign these chunks into IP_mat.
        if self.parallel.is_distributed():
            IP_mat = self.parallel.custom_comm.allreduce( 
                IP_mat_chunk)
        else:
            IP_mat = IP_mat_chunk 

        if transpose:
            IP_mat = IP_mat.T

        self.parallel.sync() # ensure that all procs leave function at same time
        return IP_mat

        
    def compute_symmetric_inner_product_mat(self, field_paths):
        """
        Computes an upper-triangular chunk of a symmetric matrix of inner 
        products.  
        """
        if isinstance(field_paths, str):
            field_paths = [field_paths]
 
        num_fields = len(field_paths)
        
        
        # num_cols_per_chunk is the number of cols each proc loads at once.  
        # Columns are loaded if the matrix must be broken up into sets of 
        # chunks.  Then symmetric upper triangular portions will be computed,
        # followed by a rectangular piece that uses columns not already loaded.
        num_cols_per_proc_chunk = 1
        num_rows_per_proc_chunk = self.max_fields_per_proc - num_cols_per_proc_chunk
 
        # <nprocs> chunks are computed simulaneously, making up a set.
        num_cols_per_chunk = num_cols_per_proc_chunk * self.parallel.get_num_procs()
        num_rows_per_chunk = num_rows_per_proc_chunk * self.parallel.get_num_procs()

        # <num_row_chunks> is the number of sets that must be computed.
        num_row_chunks = int(N.ceil(num_fields * 1. / num_rows_per_chunk)) 
        if self.parallel.is_rank_zero() and num_row_chunks > 1 and self.verbose:
            print ('Warning: The column fields will be read ~%d times each. ' +\
                'Increase number of nodes or max_fields_per_node to reduce ' +\
                'redundant loads and get a big speedup.') % num_row_chunks    
        
        # Compute a single inner product in order to determin matrix datatype
        test_field = self.load_field(field_paths[0])
        IP = self.inner_product(test_field, test_field)
        IP_type = type(IP)
        del test_field
        
        # Use the same trick as in compute_IP_mat, having each proc
        # fill in elements of a num_rows x num_rows sized matrix, rather than
        # assembling small chunks. This is done for the triangular portions. For
        # the rectangular portions, the inner product mat is filled in directly.
        IP_mat_chunk = N.mat(N.zeros((num_fields, num_fields), dtype=\
            IP_type))
        for start_row_index in xrange(0, num_fields, num_rows_per_chunk):
            end_row_index = min(num_fields, start_row_index + num_rows_per_chunk)
            proc_row_tasks_all = self.parallel.find_assignments(range(
                start_row_index, end_row_index))
            num_active_procs = len([task for task in \
                proc_row_tasks_all if task != []])
            proc_row_tasks = proc_row_tasks_all[self.parallel.get_rank()]
            if len(proc_row_tasks)!=0:
                row_fields = [self.load_field(path) for path in field_paths[
                    proc_row_tasks[0]:proc_row_tasks[-1] + 1]]
            else:
                row_fields = []
            
            # Triangular chunks
            if len(proc_row_tasks) > 0:
                # Test that indices are consecutive
                if proc_row_tasks[0:] != range(proc_row_tasks[0], 
                    proc_row_tasks[-1] + 1):
                    raise ValueError('Indices are not consecutive.')
                
                # Per-processor triangles (using only loaded snapshots)
                for row_index in xrange(proc_row_tasks[0], 
                    proc_row_tasks[-1] + 1):
                    # Diagonal term
                    IP_mat_chunk[row_index, row_index] = self.\
                        inner_product(row_fields[row_index - proc_row_tasks[
                        0]], row_fields[row_index - proc_row_tasks[0]])
                        
                    # Off-diagonal terms
                    for col_index in xrange(row_index + 1, proc_row_tasks[
                        -1] + 1):
                        IP_mat_chunk[row_index, col_index] = self.\
                            inner_product(row_fields[row_index -\
                            proc_row_tasks[0]], row_fields[col_index -\
                            proc_row_tasks[0]])
               
            # Number of square chunks to fill in is n * (n-1) / 2.  At each
            # iteration we fill in n of them, so we need (n-1) / 2 
            # iterations (round up).  
            for set_index in xrange(int(N.ceil((num_active_procs - 1.) / 2))):
                # The current proc is "sender"
                my_rank = self.parallel.get_rank()
                my_row_indices = proc_row_tasks
                mynum_rows = len(my_row_indices)
                                       
                # The proc to send to is "destination"                         
                dest_rank = (my_rank + set_index + 1) % num_active_procs
                dest_row_indices = proc_row_tasks_all[dest_rank]
                
                # The proc that data is received from is the "source"
                source_rank = (my_rank - set_index - 1) % num_active_procs
                
                # Find the maximum number of sends/recv to be done by any proc
                max_num_to_send = int(N.ceil(1. * max([len(tasks) for \
                    tasks in proc_row_tasks_all]) /\
                    num_cols_per_proc_chunk))
                
                # Pad tasks with nan so that everyone has the same
                # number of things to send.  Same for list of fields with None.             
                # The empty lists will not do anything when enumerated, so no 
                # inner products will be taken.  nan is inserted into the 
                # indices because then min/max of the indices can be taken.
                """
                if mynum_rows != len(row_fields):
                    raise ValueError('Number of rows assigned does not ' +\
                        'match number of loaded fields.')
                if mynum_rows > 0 and mynum_rows < max_num_to_send:
                    my_row_indices += [N.nan] * (max_num_to_send - mynum_rows) 
                    row_fields += [[]] * (max_num_to_send - mynum_rows)
                """
                for send_index in xrange(max_num_to_send):
                    # Only processors responsible for rows communicate
                    if mynum_rows > 0:  
                        # Send row fields, in groups of num_cols_per_proc_chunk
                        # These become columns in the ensuing computation
                        start_col_index = send_index * num_cols_per_proc_chunk
                        end_col_index = min(start_col_index + num_cols_per_proc_chunk, 
                            mynum_rows)   
                        col_fields_send = (row_fields[start_col_index:end_col_index], 
                            my_row_indices[start_col_index:end_col_index])
                        
                        # Create unique tags based on ranks
                        send_tag = my_rank * (self.parallel.get_num_procs() + 1) +\
                            dest_rank
                        recv_tag = source_rank * (self.parallel.get_num_procs() +\
                            1) + my_rank
                        
                        # Send and receieve data.  It is important that we put a
                        # Wait() command after the receive.  In testing, when 
                        # this was not done, we saw a race condition.  This was a
                        # condition that could not be fixed by a sync(). It 
                        # appears that the Wait() is very important for the non-
                        # blocking send.
                        request = self.parallel.comm.isend(col_fields_send, 
                            dest=dest_rank, tag=send_tag)                        
                        col_fields_recv = self.parallel.comm.recv(source=\
                            source_rank, tag=recv_tag)
                        request.Wait()
                        col_fields = col_fields_recv[0]
                        my_col_indices = col_fields_recv[1]
                        
                        for row_index in xrange(my_row_indices[0], 
                            my_row_indices[-1] + 1):
                            for col_field_index, col_field in enumerate(col_fields):
                                IP_mat_chunk[row_index, my_col_indices[
                                    col_field_index]] = self.inner_product(
                                    row_fields[row_index - my_row_indices[0]],
                                    col_field)
                                   
                    # Sync after send/receive   
                    self.parallel.sync()  
                
            
            # Fill in the rectangular portion next to each triangle (if nec.).
            # Start at index after last row, continue to last column. This part
            # of the code is the same as in compute_IP_mat, as of 
            # revision 141.  
            for start_col_index in xrange(end_row_index, num_fields, 
                num_cols_per_chunk):
                end_col_index = min(start_col_index + num_cols_per_chunk, num_fields)
                proc_col_tasks = self.parallel.find_assignments(range(
                    start_col_index, end_col_index))[self.parallel.get_rank()]
                        
                # Pass the col fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                col_fields_recv = (None, None)
                if len(proc_col_tasks) > 0:
                    col_indices = range(proc_col_tasks[0], 
                        proc_col_tasks[-1]+1)
                else:
                    col_indices = []
                    
                for num_passes in xrange(self.parallel.get_num_procs()):
                    # If on the first pass, load the col fields, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if num_passes == 0:
                        if len(col_indices) > 0:
                            col_fields = [self.load_field(col_path) \
                                for col_path in field_paths[col_indices[0]:\
                                    col_indices[-1] + 1]]
                        else:
                            col_fields = []
                    else: 
                        # Determine whom to communicate with
                        dest = (self.parallel.get_rank() + 1) % self.parallel.\
                            get_num_procs()
                        source = (self.parallel.get_rank() - 1) % self.parallel.\
                            get_num_procs()    
                            
                        #Create unique tag based on ranks
                        send_tag = self.parallel.get_rank() * (self.parallel.\
                            get_num_procs() + 1) + dest
                        recv_tag = source*(self.parallel.get_num_procs() + 1) +\
                            self.parallel.get_rank()    
                        
                        # Collect data and send/receive
                        col_fields_send = (col_fields, col_indices)     
                        request = self.parallel.comm.isend(col_fields_send, dest=\
                            dest, tag=send_tag)
                        col_fields_recv = self.parallel.comm.recv(source=source, 
                            tag=recv_tag)
                        request.Wait()
                        self.parallel.sync()
                        col_indices = col_fields_recv[1]
                        col_fields = col_fields_recv[0]
                        
                    # Compute the IPs for this set of data col_indices stores
                    # the indices of the IP_mat_chunk columns to be
                    # filled in.
                    if len(proc_row_tasks) > 0:
                        for row_index in xrange(proc_row_tasks[0],
                            proc_row_tasks[-1]+1):
                            for col_field_index,col_field in enumerate(col_fields):
                                IP_mat_chunk[row_index, col_indices[
                                    col_field_index]] = self.inner_product(
                                    row_fields[row_index - proc_row_tasks[0]],
                                    col_field)
            # Completed a chunk of rows and all columns on all processors.
            if T.time() - self.prev_print_time > self.print_interval:
                num_completed_IPs = end_row_index*num_fields- end_row_index**2 *.5
                percent_completed_IPs = 100. * num_completed_IPs/(.5 *\
                    num_fields **2)           
                self.print_msg('Completed %.1f%% of inner products' %
                    percent_completed_IPs, output_channel=sys.stderr)
                self.prev_print_time = T.time()
                             
        # Assign the triangular portion chunks into IP_mat.
        if self.parallel.is_distributed():
            IP_mat = self.parallel.custom_comm.allreduce(IP_mat_chunk)
        else:
            IP_mat = IP_mat_chunk

        # Create a mask for the repeated values
        mask = (IP_mat != IP_mat.T)
        
        # Collect values below diagonal
        IP_mat += N.multiply(N.triu(IP_mat.T, 1), mask)
        
        # Symmetrize matrix
        IP_mat = N.triu(IP_mat) + N.triu(IP_mat, 1).T

        self.parallel.sync() # ensure that all procs leave function at same time
        return IP_mat
        
        
    def _compute_modes(self, mode_nums, mode_path, snap_paths, field_coeff_mat,
        index_from=1):
        """
        A common method to compute and save modes from snapshots.
        
        mode_nums - mode numbers to compute on this processor. This 
          includes the index_from, so if index_from=1, examples are:
          [1,2,3,4,5] or [3,1,6,8]. The mode numbers need not be sorted,
          and sorting does not increase efficiency. 
          Repeated mode numbers is not guaranteed to work. 
        mode_path - Full path to mode location, e.g /home/user/mode_%03d.txt.
        index_from - Integer from which to index modes. E.g. from 0, 1, or other.
        snap_paths - A list paths to files from which snapshots can be loaded.
        field_coeff_mat - Matrix of coefficients for constructing modes.  The kth
            column contains the coefficients for computing the kth index mode, 
            ie index_from+k mode number. ith row contains coefficients to 
            multiply corresponding to snapshot i.

        This methods primary purpose is to recast computing modes as a simple
        linear combination of elements. To this end, it calls lin_combine_fields.
        This function mostly consists of rearranging the coeff matrix so that
        the first column corresponds to the first mode number in mode_nums.
        For more details on how the modes are formed, see docstring on
        lin_combine_fields,
        where the sum_fields are the modes and the basis_fields are the 
        snapshots.
        """        
        if self.save_field is None:
            raise UndefinedError('save_field is undefined')
                    
        if isinstance(mode_nums, int):
            mode_nums = [mode_nums]
        if isinstance(snap_paths, type('a_string')):
            snap_paths = [snap_paths]
        
        num_modes = len(mode_nums)
        num_snaps = len(snap_paths)
        
        if num_modes > num_snaps:
            raise ValueError('Cannot compute more modes than number of ' +\
                'snapshots')
                   
        for mode_num in mode_nums:
            if mode_num < index_from:
                raise ValueError('Cannot compute if mode number is less than '+\
                    'index_from')
            elif mode_num-index_from > field_coeff_mat.shape[1]:
                raise ValueError('Mode index, %d, is greater '
                    'than number of columns in the build coefficient '
                    'matrix, %d'%(mode_num-index_from,field_coeff_mat.shape[1]))
        
        # Construct field_coeff_mat and outputPaths for lin_combine_fields
        mode_numsFromZero = [mode_num-index_from for mode_num in mode_nums]
        field_coeff_matReordered = field_coeff_mat[:,mode_numsFromZero]
        mode_paths = [mode_path%mode_num for mode_num in mode_nums]
        
        self.lin_combine(mode_paths, snap_paths, field_coeff_matReordered)
        self.parallel.sync() # ensure that all procs leave function at same time
    
    
    def lin_combine(self, sum_field_paths, basis_field_paths, field_coeff_mat):
        """
        Linearly combines the basis fields and saves them.
        
          sum_field_paths is a list of the files where the linear combinations
            will be saved.
          basis_field_paths is a list of files where the basis fields will
            be read from.
          field_coeff_mat is a matrix where each row corresponds to an basis field
            and each column corresponds to a sum (lin. comb.) field. The rows and columns
            are assumed to correspond, by index, to the lists basis_field_paths and 
            sum_field_paths.
            sums = basis * field_coeff_mat
        
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
        n_p is number of processors, max = max_fields_per_node-1.
        """
        if self.save_field is None:
            raise util.UndefinedError('save_field is undefined')
                   
        if not isinstance(sum_field_paths, list):
            sum_field_paths = [sum_field_paths]
        if not isinstance(basis_field_paths, list):
            basis_field_paths = [basis_field_paths]
        num_bases = len(basis_field_paths)
        num_sums = len(sum_field_paths)
        if num_bases > field_coeff_mat.shape[0]:
            raise ValueError(('Coeff mat has fewer rows %d than num of basis paths %d'\
                %(field_coeff_mat.shape[0],num_bases)))
                
        if num_sums > field_coeff_mat.shape[1]:
            raise ValueError(('Coeff matrix has fewer cols %d than num of ' +\
                'output paths %d')%(field_coeff_mat.shape[1],num_sums))
                               
        if num_bases < field_coeff_mat.shape[0] and self.parallel.is_rank_zero():
            print 'Warning: fewer basis paths than cols in the coeff matrix'
            print '  some rows of coeff matrix will not be used'
        if num_sums < field_coeff_mat.shape[1] and self.parallel.is_rank_zero():
            print 'Warning: fewer output paths than rows in the coeff matrix'
            print '  some cols of coeff matrix will not be used'
        
        # num_bases_per_proc_chunk is the number of bases each proc loads at once        
        num_bases_per_proc_chunk = 1
        num_sums_per_proc_chunk = \
            self.max_fields_per_proc - num_bases_per_proc_chunk         
        
        # This step can be done by find_assignments as well. Really what
        # this is doing is dividing the work into num*Chunks pieces.
        # find_assignments should take an optional arg of numWorkers or numPieces.
        # Determine how the loading and scalar multiplies will be split up.
        num_basis_chunks = int(N.ceil(\
            num_bases*1./(num_bases_per_proc_chunk * 
            self.parallel.get_num_procs())))
        num_sum_chunks = int(N.ceil(
            num_sums*1./(num_sums_per_proc_chunk * 
            self.parallel.get_num_procs())))
        
        num_bases_per_chunk = int(N.ceil(num_bases*1./num_basis_chunks))
        num_sums_per_chunk = int(N.ceil(num_sums*1./num_sum_chunks))

        if num_sum_chunks > 1:
            self.print_msg('Warning: The basis fields (snapshots), ' 
                'of which there are %d, will be loaded from file %d times each. '
                'If possible, increase number of nodes or '
                'max_fields_per_node to reduce redundant loads and get a '
                'big speedup.'%(num_bases, num_sum_chunks))
               
        for start_sum_index in xrange(0, num_sums, num_sums_per_chunk):
            end_sum_index = min(start_sum_index+num_sums_per_chunk, num_sums)
            sum_assignments = self.parallel.find_assignments(
                range(start_sum_index, end_sum_index))
            proc_sum_tasks = sum_assignments[self.parallel.get_rank()]
            # Create empty list on each processor
            sum_layers = [None for i in xrange(
                len(sum_assignments[self.parallel.get_rank()]))]
            
            for start_basis_index in xrange(0, num_bases, num_bases_per_chunk):
                end_basis_index = min(
                    start_basis_index + num_bases_per_chunk, num_bases)
                basis_assignments = self.parallel.find_assignments(
                    range(start_basis_index, end_basis_index))
                proc_basis_tasks = basis_assignments[self.parallel.get_rank()]
                # Pass the basis fields to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                basis_fields_recv = (None, None)
                if len(proc_basis_tasks) > 0:
                    basis_indices = range(proc_basis_tasks[0], proc_basis_tasks[-1]+1)
                else:
                    # this proc isn't responsible for loading any basis fields
                    basis_indices = []
                    
                for num_passes in xrange(self.parallel.get_num_procs()):
                    # If on the first pass, load the basis fields, no send/recv
                    # This is all that is called when in serial, loop iterates once.
                    if num_passes == 0:
                        if len(basis_indices) > 0:
                            basis_fields = [self.load_field(basis_path) \
                                for basis_path in basis_field_paths[
                                    basis_indices[0]:basis_indices[-1]+1]]
                        else: basis_fields = []
                    else:
                        # Figure out whom to communicate with
                        source = (self.parallel.get_rank()-1) % \
                            self.parallel.get_num_procs()
                        dest = (self.parallel.get_rank()+1) % \
                            self.parallel.get_num_procs()
                        
                        #Create unique tags based on ranks
                        send_tag = self.parallel.get_rank()*(self.parallel.get_num_procs()+1) + dest
                        recv_tag = source*(self.parallel.get_num_procs()+1) + self.parallel.get_rank()
                        
                        # Send/receive data
                        basis_fields_send = (basis_fields, basis_indices)
                        request = self.parallel.comm.isend(basis_fields_send, dest=dest, tag=send_tag)                       
                        basis_fields_recv = self.parallel.comm.recv(source=source, tag=recv_tag)
                        request.Wait()
                        self.parallel.sync()
                        basis_indices = basis_fields_recv[1]
                        basis_fields = basis_fields_recv[0]
                    
                    # Compute the scalar multiplications for this set of data
                    # basis_indices stores the indices of the field_coeff_mat to use.
                    
                    for sum_index in xrange(len(proc_sum_tasks)):
                        for basis_index,basis_field in enumerate(basis_fields):
                            sum_layer = basis_field*\
                                field_coeff_mat[basis_indices[basis_index],\
                                sum_index+proc_sum_tasks[0]]
                            if sum_layers[sum_index] is None:
                                sum_layers[sum_index] = sum_layer
                            else:
                                sum_layers[sum_index] += sum_layer
            # Completed this set of sum fields, save to file
            for sum_index in xrange(len(proc_sum_tasks)):
                self.save_field(sum_layers[sum_index],\
                    sum_field_paths[sum_index+proc_sum_tasks[0]])
            if (T.time() - self.prev_print_time) > self.print_interval:    
                self.print_msg('Completed %.1f%% of sum fields, %d of %d' %
                    (end_sum_index*100./num_sums, end_sum_index, num_sums), 
                    output_channel = sys.stderr)
                self.prev_print_time = T.time()

        self.parallel.sync() # ensure that all procs leave function at same time

    def __eq__(self, other):
        #print 'comparing fieldOperations classes'
        a = (self.inner_product == other.inner_product and 
            self.load_field == other.load_field and 
            self.save_field == other.save_field and 
            self.max_fields_per_node == other.max_fields_per_node and 
            self.verbose == other.verbose)
        return a

    def __ne__(self,other):
        return not (self.__eq__(other))


