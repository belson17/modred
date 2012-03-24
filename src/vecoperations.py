
"""Collection of low level functions for modaldecomp library"""

import sys  
import copy
import time as T
import numpy as N
import util
import parallel as parallel_mod

class VecOperations(object):
    """Responsible for low level operations on vecs.

    Kwargs:
        max_vecs_per_node: max number of vecs that can be in memory
            simultaneously on a node.
        
        verbose: True or False, sets if warnings are printed.
        
        print_interval: max of how frequently progress is printed, in seconds.

    The class is a collection of methods used in the high-level 
    modred classes like POD, BPOD, and DMD. 
    It's responsible for all non-trivial parallelization.
   
    It is generally best to use all available processors, however this
    depends on the computer and the nature of the functions
    supplied. In some cases, loading from file is slower with more processors.
    """
    
    def __init__(self, get_vec=None, put_vec=None, inner_product=None, 
        max_vecs_per_node=None, verbose=True, print_interval=10):
        """Constructor. """
        self.get_vec = get_vec
        self.put_vec = put_vec
        self.inner_product = inner_product
        self.verbose = verbose 
        self.print_interval = print_interval
        self.prev_print_time = 0.
        self.parallel = parallel_mod.default_instance
        
        if max_vecs_per_node is None:
            self.max_vecs_per_node = 2
            self.print_msg('Warning: max_vecs_per_node was not specified. '
                'Assuming 2 vecs can be in memory per node. Increase '
                'max_vecs_per_node for a speedup.')
        else:
            self.max_vecs_per_node = max_vecs_per_node
        
        if self.max_vecs_per_node < \
            2 * self.parallel.get_num_procs() / self.parallel.get_num_nodes(): 
            self.max_vecs_per_proc = 2
            self.print_msg('Warning: max_vecs_per_node too small for given '
                'number of nodes and procs.  Assuming 2 vecs can be '
                'in memory per processor. Increase max_vecs_per_node for a '
                'speedup.')
        else:
            self.max_vecs_per_proc = self.max_vecs_per_node * \
                self.parallel.get_num_nodes()/self.parallel.get_num_procs()
    
    def print_msg(self, msg, output_channel=sys.stdout):
        """Print a message from rank 0 if verbose"""
        if self.verbose and self.parallel.is_rank_zero():
            print >> output_channel, msg

    def idiot_check(self, test_obj=None, test_obj_source=None):
        """Checks that the user-supplied objects and functions work.
        
        The arguments are for a test object or the source to one (retrieved with 
        ``get_vec``).  One of these should be supplied for thorough testing. 
        The add and mult functions are tested for the generic object.  This is 
        not a complete testing, but catches some common mistakes.
        
        Other things which could be tested:
            reading/writing doesnt effect other vecs/modes (memory problems)
            subtraction, division (currently not used in modred)
        """
        tol = 1e-10
        if test_obj_source is not None:
          test_obj = self.get_vec(test_obj_source)
        if test_obj is None:
            raise RuntimeError('Supply vec object or source for idiot check!')
        obj_copy = copy.deepcopy(test_obj)
        obj_copy_mag2 = self.inner_product(obj_copy, obj_copy)
        
        factor = 2.
        obj_mult = test_obj * factor
        
        if abs(self.inner_product(obj_mult, obj_mult) -
                obj_copy_mag2 * factor**2) > tol:
            raise ValueError('Multiplication of vec/mode object failed')
        
        if abs(self.inner_product(test_obj, test_obj) - 
                obj_copy_mag2) > tol:  
            raise ValueError('Original object modified by multiplication!') 
        obj_add = test_obj + test_obj
        if abs(self.inner_product(obj_add, obj_add) - obj_copy_mag2 * 4) > tol:
            raise ValueError('Addition does not give correct result')
        
        if abs(self.inner_product(test_obj, test_obj) - obj_copy_mag2) > tol:  
            raise ValueError('Original object modified by addition!')       
        
        obj_add_mult = test_obj * factor + test_obj
        if abs(self.inner_product(obj_add_mult, obj_add_mult) - obj_copy_mag2 *
                (factor + 1) ** 2) > tol:
            raise ValueError('Multiplication and addition of vec/mode are '+\
                'inconsistent')
        
        if abs(self.inner_product(test_obj, test_obj) - obj_copy_mag2) > tol:  
            raise ValueError('Original object modified by combo of mult/add!') 
        
        #objSub = 3.5*test_obj - test_obj
        #N.testing.assert_array_almost_equal(objSub,2.5*test_obj)
        #N.testing.assert_array_almost_equal(test_obj,obj_copy)
        self.print_msg('Passed the idiot check')
        return True


    def compute_inner_product_mat(self, row_vec_sources, col_vec_sources):
        """Computes a matrix of inner products and returns it.
        
        Args:
            row_vec_sources: list of row vec sources (BPOD adjoint vecs, "Y")
          
            col_vec_sources: list of column vec sources (BPOD direct vecs, "X")

        Within this method, the vecs are retrieved in a memory-efficient
        chunks so that they are not all in memory at once.
        The row vecs and col vecs are assumed to be different.        
        When they are the same (POD), a different method is used.
        
        Each processor is responsible for retrieving a subset of the rows and
        columns. The processor which retrieves a particular column vec then sends
        it to each successive processor so it can be used to compute all IPs
        for the current row chunk on each processor. This is repeated until all
        processors are done with all of their row chunks. If there are 2
        processors::
           
                | x o |
          rank0 | x o |
                | x o |
            -
                | o x |
          rank1 | o x |
                | o x |
        
        In the next step, rank 0 sends column 0 to rank 1 and rank 1
        sends column 1 to rank 0. The remaining IPs are filled in::
        
                | x x |
          rank0 | x x |
                | x x |
            -
                | x x |
          rank1 | x x |
                | x x |
          
        When the number of cols and rows is
        not divisible by the number of processors, the processors are assigned
        unequal numbers of tasks. However, all processors are always
        part of the passing circle.
        
        This is also generalized to allow the columns to be read in chunks, 
        rather than only 1 at a time.
        If we change the implementation to use hybrid distributed-shared 
        memory where it is best to work in
        operation-units (load/gets, IPs, etc) of multiples of the number
        of processors sharing memory (procs/node).
        
        The scaling is:
        
        - num gets / processor ~ (n_r*n_c/((max-1)*n_p*n_p)) + n_r/n_p
        - num MPI sends / processor ~ (n_p-1)*(n_r/((max-1)*n_p))*n_c/n_p
        - num inner products / processor ~ n_r*n_c/n_p
            
        where n_r is number of rows, n_c number of columns, max is
        max_vecs_per_proc = max_vecs_per_node/num_procs_per_node, and n_p is
        number of processors.
        
        It is enforced that there are more columns than rows by doing an
        internal transpose and un-transpose. This improves efficiency by placing
        the larger of n_c and n_r on the quadratically scaled portion.
        
        It is good to use all available processors, even if it lowers max.
        However, sometimes simultaneous loads actually makes each load slow.     
        """
             
        if not isinstance(row_vec_sources, list):
            row_vec_sources = [row_vec_sources]
        if not isinstance(col_vec_sources, list):
            col_vec_sources = [col_vec_sources]
            
        num_cols = len(col_vec_sources)
        num_rows = len(row_vec_sources)

        if num_rows > num_cols:
            transpose = True
            temp = row_vec_sources
            row_vec_sources = col_vec_sources
            col_vec_sources = temp
            temp = num_rows
            num_rows = num_cols
            num_cols = temp
        else: 
            transpose = False
       
        # Compute a single inner product in order to determine matrix datatype
        # (real or complex) and to estimate the amount of time the IPs will take.
        row_vec = self.get_vec(row_vec_sources[0])
        col_vec = self.get_vec(col_vec_sources[0])
        start_time = T.time()
        IP = self.inner_product(row_vec, col_vec)
        IP_type = type(IP)
        end_time = T.time()

        # Estimate the amount of time this will take
        duration = end_time - start_time
        self.print_msg('Computing the inner product matrix will take at least '
                    '%.1f minutes' % (num_rows * num_cols * duration / 
                    (60. * self.parallel.get_num_procs())))
        del row_vec, col_vec
        
        # convenience
        rank = self.parallel.get_rank()

        # num_cols_per_proc_chunk is the number of cols each proc gets at once        
        num_cols_per_proc_chunk = 1
        num_rows_per_proc_chunk = self.max_vecs_per_proc - num_cols_per_proc_chunk         
        
        # Determine how the retrieving and inner products will be split up.
        row_tasks = self.parallel.find_assignments(range(num_rows))
        col_tasks = self.parallel.find_assignments(range(num_cols))
           
        # Find max number of col tasks among all processors
        max_num_row_tasks = max([len(tasks) for tasks in row_tasks])
        max_num_col_tasks = max([len(tasks) for tasks in col_tasks])
        
        # These variables are the number of iters through loops that retrieve ("get")
        # row and column vecs.
        num_row_get_loops = int(N.ceil(max_num_row_tasks*1./num_rows_per_proc_chunk))
        num_col_get_loops = int(N.ceil(max_num_col_tasks*1./num_cols_per_proc_chunk))
        if num_row_get_loops > 1:
            self.print_msg('Warning: The column vecs, of which '
                    'there are %d, will be read %d times each. Increase '
                    'number of nodes or max_vecs_per_node to reduce redundant '
                    '"get_vecs"s and get a big speedup.'%(
                        num_cols,num_row_get_loops))
        
        # To find all of the inner product mat chunks, each 
        # processor has a full IP_mat with size
        # num_rows x num_cols even though each processor is not responsible for
        # filling in all of these entries. After each proc fills in what it is
        # responsible for, the other entries are 0's still. Then, an allreduce
        # is done and all the chunk mats are summed. This is simpler
        # than trying to figure out the size of each chunk mat for allgather.
        # The efficiency is not an issue, the size of the mats
        # are small compared to the size of the vecs for large data.
        IP_mat_chunk = N.mat(N.zeros((num_rows, num_cols), dtype=IP_type))
        for row_get_index in xrange(num_row_get_loops):
            if len(row_tasks[rank]) > 0:
                start_row_index = min(row_tasks[rank][0] + 
                    row_get_index*num_rows_per_proc_chunk, row_tasks[rank][-1]+1)
                end_row_index = min(row_tasks[rank][-1]+1, 
                    start_row_index + num_rows_per_proc_chunk)
                row_vecs = [self.get_vec(row_vec_source) for row_vec_source in 
                    row_vec_sources[start_row_index:end_row_index]]
            else:
                row_vecs = []

            for col_get_index in xrange(num_col_get_loops):
                if len(col_tasks[rank]) > 0:
                    start_col_index = min(col_tasks[rank][0] + 
                        col_get_index*num_cols_per_proc_chunk, 
                            col_tasks[rank][-1]+1)
                    end_col_index = min(col_tasks[rank][-1]+1, 
                        start_col_index + num_cols_per_proc_chunk)
                else:
                    start_col_index = 0
                    end_col_index = 0
                # Pass the col vecs to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                col_vecs_recv = (None, None)
                col_indices = range(start_col_index, end_col_index)
                for pass_index in xrange(self.parallel.get_num_procs()):
                    #if rank==0: print 'starting pass index=',pass_index
                    # If on the first pass, get the col vecs, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if pass_index == 0:
                        col_vecs = [self.get_vec(col_source) 
                            for col_source in col_vec_sources[start_col_index:
                            end_col_index]]
                    else:
                        # Determine with whom to communicate
                        dest = (rank + 1) % self.parallel.get_num_procs()
                        source = (rank - 1)%self.parallel.get_num_procs()    
                            
                        # Create unique tag based on send/recv ranks
                        send_tag = rank * \
                                (self.parallel.get_num_procs() + 1) + dest
                        recv_tag = source * \
                            (self.parallel.get_num_procs() + 1) + rank
                        
                        # Collect data and send/receive
                        col_vecs_send = (col_vecs, col_indices)    
                        request = self.parallel.comm.isend(
                            col_vecs_send, dest=dest, tag=send_tag)
                        col_vecs_recv = self.parallel.comm.recv(
                            source=source, tag=recv_tag)
                        request.Wait()
                        self.parallel.sync()
                        col_indices = col_vecs_recv[1]
                        col_vecs = col_vecs_recv[0]
                        
                    # Compute the IPs for this set of data col_indices stores
                    # the indices of the IP_mat_chunk columns to be
                    # filled in.
                    if len(row_vecs) > 0:
                        for row_index in xrange(start_row_index, end_row_index):
                            for col_vec_index, col_vec in enumerate(col_vecs):
                                IP_mat_chunk[row_index, col_indices[
                                    col_vec_index]] = self.inner_product(
                                    row_vecs[row_index - start_row_index],
                                    col_vec)
                    
                # Clear the retrieved column vecs after done this chunk
                del col_vecs
            # Completed a chunk of rows and all columns on all processors.
            del row_vecs
            if ((T.time() - self.prev_print_time > self.print_interval) and 
                self.verbose and self.parallel.is_rank_zero()):
                num_completed_IPs = end_row_index * num_cols
                percent_completed_IPs = 100. * num_completed_IPs/(num_cols*num_rows)           
                print >> sys.stderr, ('Completed %.1f%% of inner ' +\
                    'products: IPMat[:%d, :%d] of IPMat[%d, %d]') % \
                    (percent_completed_IPs, end_row_index, num_cols, 
                        num_rows, num_cols)
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

        
    def compute_symmetric_inner_product_mat(self, vec_sources):
        """Computes an upper-triangular chunk of a symmetric matrix of inner 
        products.
        
        See the documentation for compute_inner_product_mat for a general
        idea how this works.
        
        TODO: JON, write detailed documentation similar to 
        ``compute_inner_product_mat``'s.
        """
        if isinstance(vec_sources, str):
            vec_sources = [vec_sources]
 
        num_vecs = len(vec_sources)
        
        
        # num_cols_per_chunk is the number of cols each proc gets at once.  
        # Columns are retrieved if the matrix must be broken up into sets of 
        # chunks.  Then symmetric upper triangular portions will be computed,
        # followed by a rectangular piece that uses columns not already in memory.
        num_cols_per_proc_chunk = 1
        num_rows_per_proc_chunk = self.max_vecs_per_proc - num_cols_per_proc_chunk
 
        # <nprocs> chunks are computed simulaneously, making up a set.
        num_cols_per_chunk = num_cols_per_proc_chunk * self.parallel.get_num_procs()
        num_rows_per_chunk = num_rows_per_proc_chunk * self.parallel.get_num_procs()

        # <num_row_chunks> is the number of sets that must be computed.
        num_row_chunks = int(N.ceil(num_vecs * 1. / num_rows_per_chunk)) 
        if self.parallel.is_rank_zero() and num_row_chunks > 1 and self.verbose:
            print ('Warning: The column vecs will be read ~%d times each. ' +\
                'Increase number of nodes or max_vecs_per_node to reduce ' +\
                'redundant "get_vecs"s and get a big speedup.') % num_row_chunks    
        
        # Compute a single inner product in order to determin matrix datatype
        test_vec = self.get_vec(vec_sources[0])
        IP = self.inner_product(test_vec, test_vec)
        IP_type = type(IP)
        del test_vec
        
        # Use the same trick as in compute_IP_mat, having each proc
        # fill in elements of a num_rows x num_rows sized matrix, rather than
        # assembling small chunks. This is done for the triangular portions. For
        # the rectangular portions, the inner product mat is filled in directly.
        IP_mat_chunk = N.mat(N.zeros((num_vecs, num_vecs), dtype=\
            IP_type))
        for start_row_index in xrange(0, num_vecs, num_rows_per_chunk):
            end_row_index = min(num_vecs, start_row_index + num_rows_per_chunk)
            proc_row_tasks_all = self.parallel.find_assignments(range(
                start_row_index, end_row_index))
            num_active_procs = len([task for task in \
                proc_row_tasks_all if task != []])
            proc_row_tasks = proc_row_tasks_all[self.parallel.get_rank()]
            if len(proc_row_tasks)!=0:
                row_vecs = [self.get_vec(source) for source in vec_sources[
                    proc_row_tasks[0]:proc_row_tasks[-1] + 1]]
            else:
                row_vecs = []
            
            # Triangular chunks
            if len(proc_row_tasks) > 0:
                # Test that indices are consecutive
                if proc_row_tasks[0:] != range(proc_row_tasks[0], 
                    proc_row_tasks[-1] + 1):
                    raise ValueError('Indices are not consecutive.')
                
                # Per-processor triangles (using only vecs in memory)
                for row_index in xrange(proc_row_tasks[0], 
                    proc_row_tasks[-1] + 1):
                    # Diagonal term
                    IP_mat_chunk[row_index, row_index] = self.\
                        inner_product(row_vecs[row_index - proc_row_tasks[
                        0]], row_vecs[row_index - proc_row_tasks[0]])
                        
                    # Off-diagonal terms
                    for col_index in xrange(row_index + 1, proc_row_tasks[
                        -1] + 1):
                        IP_mat_chunk[row_index, col_index] = self.\
                            inner_product(row_vecs[row_index -\
                            proc_row_tasks[0]], row_vecs[col_index -\
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
                # This is unused?
                #dest_row_indices = proc_row_tasks_all[dest_rank]
                
                # The proc that data is received from is the "source"
                source_rank = (my_rank - set_index - 1) % num_active_procs
                
                # Find the maximum number of sends/recv to be done by any proc
                max_num_to_send = int(N.ceil(1. * max([len(tasks) for \
                    tasks in proc_row_tasks_all]) /\
                    num_cols_per_proc_chunk))
                
                # Pad tasks with nan so that everyone has the same
                # number of things to send.  Same for list of vecs with None.             
                # The empty lists will not do anything when enumerated, so no 
                # inner products will be taken.  nan is inserted into the 
                # indices because then min/max of the indices can be taken.
                """
                if mynum_rows != len(row_vecs):
                    raise ValueError('Number of rows assigned does not ' +\
                        'match number of vecs in memory.')
                if mynum_rows > 0 and mynum_rows < max_num_to_send:
                    my_row_indices += [N.nan] * (max_num_to_send - mynum_rows) 
                    row_vecs += [[]] * (max_num_to_send - mynum_rows)
                """
                for send_index in xrange(max_num_to_send):
                    # Only processors responsible for rows communicate
                    if mynum_rows > 0:  
                        # Send row vecs, in groups of num_cols_per_proc_chunk
                        # These become columns in the ensuing computation
                        start_col_index = send_index * num_cols_per_proc_chunk
                        end_col_index = min(start_col_index + num_cols_per_proc_chunk, 
                            mynum_rows)   
                        col_vecs_send = (row_vecs[start_col_index:end_col_index], 
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
                        request = self.parallel.comm.isend(col_vecs_send, 
                            dest=dest_rank, tag=send_tag)                        
                        col_vecs_recv = self.parallel.comm.recv(source=\
                            source_rank, tag=recv_tag)
                        request.Wait()
                        col_vecs = col_vecs_recv[0]
                        my_col_indices = col_vecs_recv[1]
                        
                        for row_index in xrange(my_row_indices[0], 
                            my_row_indices[-1] + 1):
                            for col_vec_index, col_vec in enumerate(col_vecs):
                                IP_mat_chunk[row_index, my_col_indices[
                                    col_vec_index]] = self.inner_product(
                                    row_vecs[row_index - my_row_indices[0]],
                                    col_vec)
                                   
                    # Sync after send/receive   
                    self.parallel.sync()  
                
            
            # Fill in the rectangular portion next to each triangle (if nec.).
            # Start at index after last row, continue to last column. This part
            # of the code is the same as in compute_IP_mat, as of 
            # revision 141.  
            for start_col_index in xrange(end_row_index, num_vecs, 
                num_cols_per_chunk):
                end_col_index = min(start_col_index + num_cols_per_chunk, num_vecs)
                proc_col_tasks = self.parallel.find_assignments(range(
                    start_col_index, end_col_index))[self.parallel.get_rank()]
                        
                # Pass the col vecs to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                col_vecs_recv = (None, None)
                if len(proc_col_tasks) > 0:
                    col_indices = range(proc_col_tasks[0], 
                        proc_col_tasks[-1]+1)
                else:
                    col_indices = []
                    
                for num_passes in xrange(self.parallel.get_num_procs()):
                    # If on the first pass, get the col vecs, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if num_passes == 0:
                        if len(col_indices) > 0:
                            col_vecs = [self.get_vec(col_source) \
                                for col_source in vec_sources[col_indices[0]:\
                                    col_indices[-1] + 1]]
                        else:
                            col_vecs = []
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
                        col_vecs_send = (col_vecs, col_indices)     
                        request = self.parallel.comm.isend(col_vecs_send, dest=\
                            dest, tag=send_tag)
                        col_vecs_recv = self.parallel.comm.recv(source=source, 
                            tag=recv_tag)
                        request.Wait()
                        self.parallel.sync()
                        col_indices = col_vecs_recv[1]
                        col_vecs = col_vecs_recv[0]
                        
                    # Compute the IPs for this set of data col_indices stores
                    # the indices of the IP_mat_chunk columns to be
                    # filled in.
                    if len(proc_row_tasks) > 0:
                        for row_index in xrange(proc_row_tasks[0],
                            proc_row_tasks[-1]+1):
                            for col_vec_index, col_vec in enumerate(col_vecs):
                                IP_mat_chunk[row_index, col_indices[
                                    col_vec_index]] = self.inner_product(
                                    row_vecs[row_index - proc_row_tasks[0]],
                                    col_vec)
            # Completed a chunk of rows and all columns on all processors.
            if T.time() - self.prev_print_time > self.print_interval:
                num_completed_IPs = end_row_index*num_vecs- end_row_index**2 *.5
                percent_completed_IPs = 100. * num_completed_IPs/(.5 *\
                    num_vecs **2)           
                self.print_msg('Completed %.1f%% of inner products' %
                    percent_completed_IPs, output_channel=sys.stderr)
                self.prev_print_time = T.time()
            # Finished row_vecs loop, delete memory used
            del row_vecs                     
        
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
        
        
    def compute_modes(self, mode_nums, mode_dests, vec_sources, vec_coeff_mat,
        index_from=1):
        """A common method to compute modes from vecs and ``put_vec`` them.
        
        This method recasts computing modes as a linear combination of elements.
        To this end, it calls lin_combine_vecs.
        It rearranges the coeff matrix so that the first column corresponds to
        the first mode number in mode_nums.
        
        Args:
          mode_nums: mode numbers to compute. 
              Examples are: [1,2,3,4,5] or [3,1,6,8]. 
              The mode numbers need not be sorted,
              and sorting does not increase efficiency. 
              
          mode_dests: destination for modes (memory or file)
          
          vec_sources: list sources from which vecs are retrieved.
          
          vec_coeff_mat: Matrix of coefficients for constructing modes. The kth
              column contains the coefficients for computing the kth index mode, 
              ie index_from+k mode number. ith row contains coefficients to 
              multiply corresponding to vec i.
              
        Kwargs:
          index_from: integer from which to index modes. E.g. from 0, 1, or other.
        
        Calls lin_combine_fiels with sum_vecs as the modes and the
        basis_vecs as the vecs.
        """        
        if self.put_vec is None:
            raise UndefinedError('put_vec is undefined')
                    
        if isinstance(mode_nums, int):
            mode_nums = [mode_nums]
        if not isinstance(vec_sources, list):
            vec_sources = [vec_sources]
        if not isinstance(mode_dests, list):
            mode_dests = [mode_dests]
        
        num_modes = len(mode_nums)
        num_vecs = len(vec_sources)
        
        if num_modes > num_vecs:
            raise ValueError(('Cannot compute more modes (%d) than number of ' +\
                'vecs(%d)')%(num_modes, num_vecs))
        
        if num_modes > len(mode_dests):
            raise ValueError('More mode numbers than mode destinations')
        elif num_modes < len(mode_dests):
            print ('Warning: Fewer mode numbers (%d) than mode destinations(%d),'+
                ' some mode destinations will not be used')%(
                    num_modes, len(mode_dests))
            mode_dests = mode_dests[:num_modes] # deepcopy?
        
        for mode_num in mode_nums:
            if mode_num < index_from:
                raise ValueError('Cannot compute if mode number is less than '+\
                    'index_from')
            elif mode_num-index_from > vec_coeff_mat.shape[1]:
                raise ValueError('Mode index, %d, is greater '
                    'than number of columns in the build coefficient '
                    'matrix, %d'%(mode_num-index_from,vec_coeff_mat.shape[1]))
        
        # Construct vec_coeff_mat and outputPaths for lin_combine_vecs
        mode_nums_from_zero = [mode_num-index_from for mode_num in mode_nums]
        vec_coeff_mat_reordered = vec_coeff_mat[:,mode_nums_from_zero]
        
        self.lin_combine(mode_dests, vec_sources, vec_coeff_mat_reordered)
        self.parallel.sync() # ensure that all procs leave function at same time
    
    
    def lin_combine(self, sum_vec_dests, basis_vec_sources, vec_coeff_mat):
        """Linearly combines the basis vecs and puts them.
        
        Args:
            sum_vec_dests: list of the files where the linear combinations
                will be ``self.put_vec`` ed.
                
            basis_vec_sources: list of files where the basis vecs will
                be ``self.get_vec`` ed from.
                
            vec_coeff_mat: matrix with each row corresponding to a basis vec
                and each column to a sum (lin. comb.) vec.
                The rows and columns correspond, by index,
                to the lists basis_vec_sources and sum_vec_dests.
                ``sums = basis * vec_coeff_mat``
          
        Each processor retrieves a subset of the basis vecs to compute as many
        outputs as a processor can have in memory at once. Each processor
        computes the "layers" from the basis it is resonsible for, and for
        as many modes as it can fit in memory. The layers from all procs are
        then
        summed together to form the full outputs. The output sum_vecs 
        are then ``self.put_vec`` ed.
        
        Scaling is:
        
          num gets / proc = n_s/(n_p*(max-1)) * n_b/n_p
          
          passes/proc = (n_p-1) * n_s/(n_p*(max-1)) * (n_b/n_p)
          
          scalar multiplies/proc = n_s*n_b/n_p
          
        Where n_s is number of sum vecs, n_b is number of basis vecs,
        n_p is number of processors, max = max_vecs_per_node.
        """
        if self.put_vec is None:
            raise util.UndefinedError('put_vec is undefined')
                   
        if not isinstance(sum_vec_dests, list):
            sum_vec_dests = [sum_vec_dests]
        if not isinstance(basis_vec_sources, list):
            basis_vec_sources = [basis_vec_sources]
        num_bases = len(basis_vec_sources)
        num_sums = len(sum_vec_dests)
        if num_bases > vec_coeff_mat.shape[0]:
            raise ValueError(('Coeff mat has fewer rows %d than num of basis sources %d'\
                %(vec_coeff_mat.shape[0],num_bases)))
                
        if num_sums > vec_coeff_mat.shape[1]:
            raise ValueError(('Coeff matrix has fewer cols %d than num of ' +\
                'output sources %d')%(vec_coeff_mat.shape[1],num_sums))
                               
        if num_bases < vec_coeff_mat.shape[0] and self.parallel.is_rank_zero():
            print 'Warning: fewer bases than cols in the coeff matrix'
            print '  some rows of coeff matrix will not be used'
        if num_sums < vec_coeff_mat.shape[1] and self.parallel.is_rank_zero():
            print 'Warning: fewer outputs than rows in the coeff matrix'
            print '  some cols of coeff matrix will not be used'

        # convencience
        rank = self.parallel.get_rank()        

        # num_bases_per_proc_chunk is the number of bases each proc gets at once        
        num_bases_per_proc_chunk = 1
        num_sums_per_proc_chunk = self.max_vecs_per_proc - \
            num_bases_per_proc_chunk
        
        basis_tasks = self.parallel.find_assignments(range(num_bases))
        sum_tasks = self.parallel.find_assignments(range(num_sums))

        # Find max number tasks among all processors
        max_num_basis_tasks = max([len(tasks) for tasks in basis_tasks])
        max_num_sum_tasks = max([len(tasks) for tasks in sum_tasks])
        
        # These variables are the number of iters through loops that retrieve ("get")
        # and "put" basis and sum vecs.
        num_basis_get_iters = int(N.ceil(max_num_basis_tasks*1./num_bases_per_proc_chunk))
        num_sum_put_iters = int(N.ceil(max_num_sum_tasks*1./num_sums_per_proc_chunk))
        if num_sum_put_iters > 1:
            self.print_msg('Warning: The basis vecs, ' 
                'of which there are %d, will be retrieved %d times each. '
                'If possible, increase number of nodes or '
                'max_vecs_per_node to reduce redundant retrieves and get a '
                'big speedup.'%(num_bases, num_sum_put_iters))
        
        for sum_put_index in xrange(num_sum_put_iters):
            if len(sum_tasks[rank]) > 0:
                start_sum_index = min(sum_tasks[rank][0] + 
                    sum_put_index*num_sums_per_proc_chunk, sum_tasks[rank][-1]+1)
                end_sum_index = min(start_sum_index+num_sums_per_proc_chunk,
                    sum_tasks[rank][-1]+1)
                # Create empty list on each processor
                sum_layers = [None]*(end_sum_index - start_sum_index)
            else:
                start_sum_index = 0
                end_sum_index = 0
                sum_layers = []

            for basis_get_index in xrange(num_basis_get_iters):
                if len(basis_tasks[rank]) > 0:    
                    start_basis_index = min(basis_tasks[rank][0] + 
                        basis_get_index*num_bases_per_proc_chunk, basis_tasks[rank][-1]+1)
                    end_basis_index = min(start_basis_index+num_bases_per_proc_chunk,
                        basis_tasks[rank][-1]+1)
                    basis_indices = range(start_basis_index, end_basis_index)
                else:
                    basis_indices = []
                
                # Pass the basis vecs to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                basis_vecs_recv = (None, None)

                for pass_index in xrange(self.parallel.get_num_procs()):
                    # If on the first pass, retrieve the basis vecs, no send/recv
                    # This is all that is called when in serial, loop iterates once.
                    if pass_index == 0:
                        if len(basis_indices) > 0:
                            basis_vecs = [self.get_vec(basis_source) \
                                for basis_source in basis_vec_sources[
                                    basis_indices[0]:basis_indices[-1]+1]]
                        else:
                            basis_vecs = []
                    else:
                        # Figure out whom to communicate with
                        source = (self.parallel.get_rank()-1) % \
                            self.parallel.get_num_procs()
                        dest = (self.parallel.get_rank()+1) % \
                            self.parallel.get_num_procs()
                        
                        #Create unique tags based on ranks
                        send_tag = self.parallel.get_rank() * \
                            (self.parallel.get_num_procs()+1) + dest
                        recv_tag = source*(self.parallel.get_num_procs()+1) + \
                            self.parallel.get_rank()
                        
                        # Send/receive data
                        basis_vecs_send = (basis_vecs, basis_indices)
                        request = self.parallel.comm.isend(basis_vecs_send,  
                            dest=dest, tag=send_tag)                       
                        basis_vecs_recv = self.parallel.comm.recv(
                            source=source, tag=recv_tag)
                        request.Wait()
                        self.parallel.sync()
                        basis_indices = basis_vecs_recv[1]
                        basis_vecs = basis_vecs_recv[0]
                    
                    # Compute the scalar multiplications for this set of data
                    # basis_indices stores the indices of the vec_coeff_mat to use.
                    for sum_index in xrange(start_sum_index, end_sum_index):
                        for basis_index, basis_vec in enumerate(basis_vecs):
                            sum_layer = basis_vec*\
                                vec_coeff_mat[basis_indices[basis_index],\
                                sum_index]
                            if sum_layers[sum_index-start_sum_index] is None:
                                sum_layers[sum_index-start_sum_index] = sum_layer
                            else:
                                sum_layers[sum_index-start_sum_index] += sum_layer

            # Completed this set of sum vecs, puts them to memory or file
            for sum_index in xrange(start_sum_index, end_sum_index):
                self.put_vec(sum_layers[sum_index-start_sum_index], \
                    sum_vec_dests[sum_index])
            del sum_layers
            if (T.time() - self.prev_print_time) > self.print_interval:    
                self.print_msg('Completed %.1f%% of sum vecs' %
                    (end_sum_index*100./max_num_sum_tasks), 
                    output_channel = sys.stderr)
                self.prev_print_time = T.time()
            

        self.parallel.sync() # ensure that all procs leave function at same time

    def __eq__(self, other):
        """Equal or not"""
        #print 'comparing vecOperations classes'
        equal = (self.inner_product == other.inner_product and 
            self.get_vec == other.get_vec and 
            self.put_vec == other.put_vec and 
            self.max_vecs_per_node == other.max_vecs_per_node and 
            self.verbose == other.verbose)
        return equal

    def __ne__(self, other):
        return not (self.__eq__(other))


