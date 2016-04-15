from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future.builtins import range
from future.builtins import object
import sys
import copy
from time import time

import numpy as np

from . import util
from .parallel import parallel_default_instance
_parallel = parallel_default_instance
from . import vectors as V


class VectorSpaceMatrices(object):
    """Implements inner products and linear combinations using data stored in
    matrices.
    
    Kwargs:
        ``inner_product_weights``: 1D array or matrix of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.
    """
    def __init__(self, weights=None):
        self.weights = weights
        if self.weights is not None:
            self.weights = np.array(self.weights).squeeze()
        if self.weights is None:
            VectorSpaceMatrices.compute_inner_product_mat = \
                VectorSpaceMatrices._IP_no_weights
        elif self.weights.ndim == 1:
            VectorSpaceMatrices.compute_inner_product_mat = \
                VectorSpaceMatrices._IP_1D_weights
        elif self.weights.ndim == 2:
            self.weights = np.mat(self.weights)
            VectorSpaceMatrices.compute_inner_product_mat = \
                VectorSpaceMatrices._IP_2D_weights
        else:
            raise ValueError('Weights must be None, 1D, or 2D')

    def _IP_no_weights(self, vecs1, vecs2):
        return np.mat(vecs1).H * np.mat(vecs2)

    def _IP_1D_weights(self, vecs1, vecs2):
        return np.mat((np.array(vecs1).conj().T * self.weights).dot(vecs2))

    def _IP_2D_weights(self, vecs1, vecs2):
        return np.mat(vecs1).H * self.weights * np.mat(vecs2)

    def __eq__(self, other):
        if type(other) == type(self):
            return smart_eq(self.weights, other.weights)
        else:
            return False
    
    def lin_combine(self, basis_vecs, coeff_mat,
        coeff_mat_col_indices=None):
        return np.mat(basis_vecs) * np.mat(coeff_mat[:,coeff_mat_col_indices])
    
    def compute_symmetric_inner_product_mat(self, vecs):
        return self.compute_inner_product_mat(vecs, vecs)
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return util.smart_eq(self.weights, other.weights)
        
    def __ne__(self, other):
        return not self.__eq__(other)


class VectorSpaceHandles(object):
    """Provides efficient, parallel implementations of vector space operations,
    using handles.

    Kwargs:
        ``inner_product``: Function that computes inner product of two vector
        objects.
        
        ``max_vecs_per_node``: Maximum number of vectors that can be stored in 
        memory, per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.
        
        ``print_interval``: Minimum time (in seconds) between printed progress 
        messages.

    This class implements low-level functions for computing large numbers of
    vector sums and inner products.  These functions are used by high-level
    classes in :py:mod:`pod`, :py:mod:`bpod`, :py:mod:`dmd` and
    :py:mod:`ltigalerkinproj`. 

    Note: Computations are often sped up by using all available processors,
    even if this lowers ``max_vecs_per_node`` proportionally. 
    However, this depends on the computer and the nature of the functions
    supplied, and sometimes loading from file is slower with more processors.
    """
    def __init__(self, inner_product=None, 
        max_vecs_per_node=None, verbosity=1, print_interval=10):
        """Constructor."""
        self.inner_product = inner_product
        self.verbosity = verbosity 
        self.print_interval = print_interval
        self.prev_print_time = 0.
        
        if max_vecs_per_node is None:
            self.max_vecs_per_node = 10000 # different default?
            self.print_msg('Warning: max_vecs_per_node was not specified. '
                'Assuming %d vecs can be in memory per node. Decrease '
                'max_vecs_per_node if memory errors.'%self.max_vecs_per_node)
        else:
            self.max_vecs_per_node = max_vecs_per_node
        
        if self.max_vecs_per_node < \
            2 * _parallel.get_num_procs() / _parallel.get_num_nodes(): 
            self.max_vecs_per_proc = 2
            self.print_msg('Warning: max_vecs_per_node too small for given '
                'number of nodes and procs. Assuming 2 vecs can be '
                'in memory per processor. If possible, increase ' 
                'max_vecs_per_node for a speedup.')
        else:
            self.max_vecs_per_proc = self.max_vecs_per_node * \
                _parallel.get_num_nodes() // _parallel.get_num_procs()
                
    def _check_inner_product(self):
        """Check that ``inner_product`` is defined"""
        if self.inner_product is None:
            raise RuntimeError('inner product function is not defined')
        
    def print_msg(self, msg, output_channel=sys.stdout):
        """Print a message from rank zero MPI worker/processor."""
        if self.verbosity > 0 and _parallel.is_rank_zero():
            print(msg, file=output_channel)

    def sanity_check(self, test_vec_handle):
        """Checks that user-supplied vector handle and vector satisfy 
        requirements.
        
        Args:
            ``test_vec_handle``: A vector handle to test.
        
        The add and multiply functions are tested for the vector object.  
        This is not a complete testing, but catches some common mistakes.
        An error is raised if a check fails.
        
        """
        # TODO: Other things which could be tested:
        # ``get``/``put`` doesn't affect other vecs (memory problems)
        self._check_inner_product()
        tol = 1e-10

        test_vec = test_vec_handle.get()
        vec_copy = copy.deepcopy(test_vec)
        vec_copy_mag2 = self.inner_product(vec_copy, vec_copy)
        
        factor = 2.
        vec_mult = test_vec * factor
        
        if abs(self.inner_product(vec_mult, vec_mult) -
                vec_copy_mag2 * factor**2) > tol:
            raise ValueError('Multiplication of vec/mode failed')
        
        if abs(self.inner_product(test_vec, test_vec) - 
                vec_copy_mag2) > tol:  
            raise ValueError('Original vec modified by multiplication!') 
        vec_add = test_vec + test_vec
        if abs(self.inner_product(vec_add, vec_add) - vec_copy_mag2 * 4) > tol:
            raise ValueError('Addition does not give correct result')
        
        if abs(self.inner_product(test_vec, test_vec) - vec_copy_mag2) > tol:  
            raise ValueError('Original vec modified by addition!')       
        
        vec_add_mult = test_vec * factor + test_vec
        if abs(self.inner_product(vec_add_mult, vec_add_mult) - vec_copy_mag2 *
                (factor + 1) ** 2) > tol:
            raise ValueError('Multiplication and addition of vec/mode are '+\
                'inconsistent')
        
        if abs(self.inner_product(test_vec, test_vec) - vec_copy_mag2) > tol:  
            raise ValueError('Original vec modified by combo of mult/add!') 
        
        #vecSub = 3.5*test_vec - test_vec
        #np.testing.assert_array_almost_equal(vecSub,2.5*test_vec)
        #np.testing.assert_array_almost_equal(test_vec,vec_copy)
        self.print_msg('Passed the sanity check')

    def compute_inner_product_mat(self, row_vec_handles, col_vec_handles):
        """Computes matrix whose elements are inner products of the vector
        objects in ``row_vec_handles`` and ``col_vec_handles``.
        
        Args:
            ``row_vec_handles``: List of handles for vector objects
            corresponding to rows of the inner product matrix.  For example, in
            BPOD this is the adjoint snapshot matrix :math:`Y`.
         
            ``col_vec_handles``: List of handles for vector objects
            corresponding to columns of the inner product matrix.  For example,
            in BPOD this is the direct snapshot matrix :math:`X`.
        
        Returns:
            ``IP_mat``: 2D array of inner products.

        The vectors are retrieved in memory-efficient chunks and are not all in
        memory at once.  The row vectors and column vectors are assumed to be
        different.  When they are the same, use
        :py:meth:`compute_symmetric_inner_product` for a 2x speedup.
        
        Each MPI worker (processor) is responsible for retrieving a subset of
        the rows and columns. The processors then send/receive columns via MPI
        so they can be used to compute all inner products for the rows on each
        MPI worker.  This is repeated until all MPI workers are done with all
        of their row chunks.  If there are 2 processors::
           
                | x o |
          rank0 | x o |
                | x o |
            -
                | o x |
          rank1 | o x |
                | o x |
        
        In the next step, rank 0 sends column 0 to rank 1 and rank 1 sends
        column 1 to rank 0. The remaining inner products are filled in::
        
                | x x |
          rank0 | x x |
                | x x |
            -
                | x x |
          rank1 | x x |
                | x x |
          
        When the number of columns and rows is not divisible by the number of
        processors, the processors are assigned unequal numbers of tasks.
        However, all processors are always part of the passing cycle.
        
        The scaling is:
        
        - num gets / processor ~ :math:`(n_r*n_c/((max-2)*n_p*n_p)) + n_r/n_p`
        - num MPI sends / processor ~ 
          :math:`(n_p-1)*(n_r/((max-2)*n_p))*n_c/n_p`
        - num inner products / processor ~ :math:`n_r*n_c/n_p`
            
        where :math:`n_r` is number of rows, :math:`n_c` number of columns,
        :math:`max` is 
        ``max_vecs_per_proc = max_vecs_per_node/num_procs_per_node``, 
        and :math:`n_p` is the number of MPI workers (processors).
        
        If there are more rows than columns, then an internal transpose and
        un-transpose is performed to improve efficiency (since :math:`n_c` only
        appears in the scaling in the quadratic term).
        """
        self._check_inner_product()
        row_vec_handles = util.make_iterable(row_vec_handles)
        col_vec_handles = util.make_iterable(col_vec_handles)
            
        num_cols = len(col_vec_handles)
        num_rows = len(row_vec_handles)

        if num_rows > num_cols:
            transpose = True
            temp = row_vec_handles
            row_vec_handles = col_vec_handles
            col_vec_handles = temp
            temp = num_rows
            num_rows = num_cols
            num_cols = temp
        else: 
            transpose = False
               
        # convenience
        rank = _parallel.get_rank()

        ## Old way that worked
        # num_cols_per_proc_chunk is the number of cols each proc gets at once
        num_cols_per_proc_chunk = 1
        num_rows_per_proc_chunk = self.max_vecs_per_proc - \
            num_cols_per_proc_chunk
               
        # Determine how the retrieving and inner products will be split up.
        row_tasks = _parallel.find_assignments(list(range(num_rows)))
        col_tasks = _parallel.find_assignments(list(range(num_cols)))
           
        # Find max number of col tasks among all processors
        max_num_row_tasks = max([len(tasks) for tasks in row_tasks])
        max_num_col_tasks = max([len(tasks) for tasks in col_tasks])
        
        ## New way
        #if self.max_vecs_per_node > max_num_row_tasks:
        #    num_cols_per_proc_chunk = 
        #num_rows_per_proc_chunk = self.max_vecs_per_proc - \
        #    num_cols_per_proc_chunk
        
        # These variables are the number of iters through loops that retrieve
        # ("get") row and column vecs.
        num_row_get_loops = \
            int(np.ceil(max_num_row_tasks*1./num_rows_per_proc_chunk))
        num_col_get_loops = \
            int(np.ceil(max_num_col_tasks*1./num_cols_per_proc_chunk))
        if num_row_get_loops > 1:
            self.print_msg('Warning: The column vecs, of which '
                    'there are %d, will be retrieved %d times each. Increase '
                    'number of nodes or max_vecs_per_node to reduce redundant '
                    '"get"s for a speedup.'%(num_cols, num_row_get_loops))
        
        
        # Estimate the time this will take and determine matrix datatype
        # (real or complex).
        row_vec = row_vec_handles[0].get()
        col_vec = col_vec_handles[0].get()
        # Burn the first, it sometimes contains slow imports
        IP_burn = self.inner_product(row_vec, col_vec)
        
        start_time = time()
        row_vec = row_vec_handles[0].get()
        get_time = time() - start_time
        
        start_time = time()
        IP = self.inner_product(row_vec, col_vec)
        IP_time = time() - start_time
        IP_type = type(IP)
        
        total_IP_time = (num_rows * num_cols * IP_time /
            _parallel.get_num_procs())
        vecs_per_proc = self.max_vecs_per_node * _parallel.get_num_nodes() / \
            _parallel.get_num_procs()
        num_gets =  (num_rows*num_cols) / ((vecs_per_proc-2) *
            _parallel.get_num_procs()**2) + num_rows/_parallel.get_num_procs()
        total_get_time = num_gets * get_time
        self.print_msg('Computing the inner product matrix will take at least '
                    '%.1f minutes' % ((total_IP_time + total_get_time) / 60.))
        del row_vec, col_vec
        
        # To find all of the inner product mat chunks, each 
        # processor has a full IP_mat with size
        # num_rows x num_cols even though each processor is not responsible for
        # filling in all of these entries. After each proc fills in what it is
        # responsible for, the other entries remain 0's. Then, an allreduce
        # is done and all the IP_mats are summed. This is simpler
        # concatenating chunks of the IPmats.
        # The efficiency is not an issue, the size of the mats
        # are small compared to the size of the vecs for large data.
        IP_mat = np.mat(np.zeros((num_rows, num_cols), dtype=IP_type))
        for row_get_index in range(num_row_get_loops):
            if len(row_tasks[rank]) > 0:
                start_row_index = min(row_tasks[rank][0] + 
                    row_get_index*num_rows_per_proc_chunk, 
                    row_tasks[rank][-1]+1)
                end_row_index = min(row_tasks[rank][-1]+1, 
                    start_row_index + num_rows_per_proc_chunk)
                row_vecs = [row_vec_handle.get() for row_vec_handle in 
                    row_vec_handles[start_row_index:end_row_index]]
            else:
                row_vecs = []

            for col_get_index in range(num_col_get_loops):
                if len(col_tasks[rank]) > 0:
                    start_col_index = min(col_tasks[rank][0] + 
                        col_get_index*num_cols_per_proc_chunk, 
                            col_tasks[rank][-1]+1)
                    end_col_index = min(col_tasks[rank][-1]+1, 
                        start_col_index + num_cols_per_proc_chunk)
                else:
                    start_col_index = 0
                    end_col_index = 0
                # Cycle the col vecs to proc with rank -> mod(rank+1,num_procs) 
                # Must do this for each processor, until data makes a circle
                col_vecs_recv = (None, None)
                col_indices = list(range(start_col_index, end_col_index))
                for pass_index in range(_parallel.get_num_procs()):
                    #if rank==0: print 'starting pass index=',pass_index
                    # If on the first pass, get the col vecs, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if pass_index == 0:
                        col_vecs = [col_handle.get() 
                            for col_handle in col_vec_handles[start_col_index:
                            end_col_index]]
                    else:
                        # Determine with whom to communicate
                        dest = (rank + 1) % _parallel.get_num_procs()
                        source = (rank - 1)%_parallel.get_num_procs()    
                            
                        # Create unique tag based on send/recv ranks
                        send_tag = rank * \
                                (_parallel.get_num_procs() + 1) + dest
                        recv_tag = source * \
                            (_parallel.get_num_procs() + 1) + rank
                        
                        # Collect data and send/receive
                        col_vecs_send = (col_vecs, col_indices)    
                        request = _parallel.comm.isend(
                            col_vecs_send, dest=dest, tag=send_tag)
                        col_vecs_recv = _parallel.comm.recv(
                            source=source, tag=recv_tag)
                        request.Wait()
                        _parallel.barrier()
                        col_indices = col_vecs_recv[1]
                        col_vecs = col_vecs_recv[0]
                        
                    # Compute the IPs for this set of data col_indices stores
                    # the indices of the IP_mat columns to be
                    # filled in.
                    if len(row_vecs) > 0:
                        for row_index in range(start_row_index, end_row_index):
                            for col_vec_index, col_vec in enumerate(col_vecs):
                                IP_mat[row_index, col_indices[
                                    col_vec_index]] = self.inner_product(
                                    row_vecs[row_index - start_row_index],
                                    col_vec)
                        if (time() - self.prev_print_time) > \
                            self.print_interval:
                            num_completed_IPs = (np.abs(IP_mat)>0).sum()
                            percent_completed_IPs = (100. * num_completed_IPs*
                                _parallel.get_num_MPI_workers()) / (
                                num_cols*num_rows)
                            self.print_msg(('Completed %.1f%% of inner ' + 
                                'products')%percent_completed_IPs, sys.stderr)
                            self.prev_print_time = time()
                        
                # Clear the retrieved column vecs after done this pass cycle
                del col_vecs
            # Completed a chunk of rows and all columns on all processors.
            del row_vecs

        # Assign these chunks into IP_mat.
        if _parallel.is_distributed():
            IP_mat = _parallel.custom_comm.allreduce(IP_mat)

        if transpose:
            IP_mat = IP_mat.T
        
        percent_completed_IPs = 100.
        self.print_msg(('Completed %.1f%% of inner ' + 
            'products')%percent_completed_IPs, sys.stderr)
        self.prev_print_time = time()

        _parallel.barrier() 
        return IP_mat
        
    def compute_symmetric_inner_product_mat(self, vec_handles):
        """Computes symmetric matrix whose elements are inner products of the
        vector objects in ``vec_handles`` with each other.  

        Args:
            ``vec_handles``: List of handles for vector objects corresponding
            to both rows and columns.  For example, in POD this is the snapshot
            matrix :math:`X`.
        
        Returns:
            ``IP_mat``: 2D array of inner products.

        See the documentation for :py:meth:`compute_inner_product_mat` for an
        idea how this works.  Efficiency is achieved by only computing the
        upper-triangular elements, since the matrix is symmetric.  Within the
        upper-triangular portion, there are rectangular chunks and triangular
        chunks.  The rectangular chunks are divided up among MPI workers
        (processors) as weighted tasks.  Once those have been computed, the
        triangular chunks are dealt with.  
        """
        # TODO: JON, write detailed documentation similar to 
        # :py:meth:`compute_inner_product_mat`.
        self._check_inner_product()
        vec_handles = util.make_iterable(vec_handles)
 
        num_vecs = len(vec_handles)        
        
        # num_cols_per_chunk is the number of cols each proc gets at once.
        # Columns are retrieved if the matrix must be broken up into sets of
        # chunks.  Then symmetric upper triangular portions will be computed,
        # followed by a rectangular piece that uses columns not already in
        # memory.
        num_cols_per_proc_chunk = 1
        num_rows_per_proc_chunk = self.max_vecs_per_proc -\
            num_cols_per_proc_chunk
 
        # <nprocs> chunks are computed simulaneously, making up a set.
        num_cols_per_chunk = num_cols_per_proc_chunk * _parallel.get_num_procs()
        num_rows_per_chunk = num_rows_per_proc_chunk * _parallel.get_num_procs()

        # <num_row_chunks> is the number of sets that must be computed.
        num_row_chunks = int(np.ceil(num_vecs * 1. / num_rows_per_chunk)) 
        if num_row_chunks > 1:
            self.print_msg('Warning: The vecs, of which '
                'there are %d, will be retrieved %d times each. Increase '
                'number of nodes or max_vecs_per_node to reduce redundant '
                '"get"s for a speedup.'%(num_vecs,num_row_chunks))
        
        # Estimate the time this will take and determine matrix datatype
        # (real or complex).
        test_vec = vec_handles[0].get()
        # Burn the first, it sometimes contains slow imports
        IP_burn = self.inner_product(test_vec, test_vec)
        
        start_time = time()
        test_vec = vec_handles[0].get()
        get_time = time() - start_time
        
        start_time = time()
        IP = self.inner_product(test_vec, test_vec)
        IP_time = time() - start_time
        IP_type = type(IP)
        
        total_IP_time = (num_vecs**2 * IP_time / 2. /
            _parallel.get_num_procs())
        vecs_per_proc = self.max_vecs_per_node * _parallel.get_num_nodes() / \
            _parallel.get_num_procs()
        num_gets =  (num_vecs**2 /2.) / ((vecs_per_proc-2) *
            _parallel.get_num_procs()**2) + \
            num_vecs/_parallel.get_num_procs()/2.
        total_get_time = num_gets * get_time
        self.print_msg('Computing the inner product matrix will take at least '
                    '%.1f minutes' % ((total_IP_time + total_get_time) / 60.))
        del test_vec

        
        # Use the same trick as in compute_IP_mat, having each proc
        # fill in elements of a num_rows x num_rows sized matrix, rather than
        # assembling small chunks. This is done for the triangular portions. 
        # For the rectangular portions, the inner product mat is filled 
        # in directly.
        IP_mat = np.mat(np.zeros((num_vecs, num_vecs), dtype=IP_type))
        for start_row_index in range(0, num_vecs, num_rows_per_chunk):
            end_row_index = min(num_vecs, start_row_index + num_rows_per_chunk)
            proc_row_tasks_all = _parallel.find_assignments(list(range(
                start_row_index, end_row_index)))
            num_active_procs = len([task for task in \
                proc_row_tasks_all if task != []])
            proc_row_tasks = proc_row_tasks_all[_parallel.get_rank()]
            if len(proc_row_tasks)!=0:
                row_vecs = [vec_handle.get() for vec_handle in vec_handles[
                    proc_row_tasks[0]:proc_row_tasks[-1] + 1]]
            else:
                row_vecs = []
            
            # Triangular chunks
            if len(proc_row_tasks) > 0:
                # Test that indices are consecutive
                if proc_row_tasks[0:] != list(range(proc_row_tasks[0], 
                    proc_row_tasks[-1] + 1)):
                    raise ValueError('Indices are not consecutive.')
                
                # Per-processor triangles (using only vecs in memory)
                for row_index in range(proc_row_tasks[0], 
                    proc_row_tasks[-1] + 1):
                    # Diagonal term
                    IP_mat[row_index, row_index] = self.\
                        inner_product(row_vecs[row_index - proc_row_tasks[
                        0]], row_vecs[row_index - proc_row_tasks[0]])
                        
                    # Off-diagonal terms
                    for col_index in range(row_index + 1, proc_row_tasks[
                        -1] + 1):
                        IP_mat[row_index, col_index] = self.\
                            inner_product(row_vecs[row_index -\
                            proc_row_tasks[0]], row_vecs[col_index -\
                            proc_row_tasks[0]])
               
            # Number of square chunks to fill in is n * (n-1) / 2.  At each
            # iteration we fill in n of them, so we need (n-1) / 2 
            # iterations (round up).  
            for set_index in range(int(np.ceil((num_active_procs - 1.) / 2))):
                # The current proc is "sender"
                my_rank = _parallel.get_rank()
                my_row_indices = proc_row_tasks
                my_num_rows = len(my_row_indices)
                                       
                # The proc to send to is "destination"                         
                dest_rank = (my_rank + set_index + 1) % num_active_procs
                # This is unused?
                #dest_row_indices = proc_row_tasks_all[dest_rank]
                
                # The proc that data is received from is the "source"
                source_rank = (my_rank - set_index - 1) % num_active_procs
                
                # Find the maximum number of sends/recv to be done by any proc
                max_num_to_send = int(np.ceil(1. * max([len(tasks) for \
                    tasks in proc_row_tasks_all]) /\
                    num_cols_per_proc_chunk))
                """
                # Pad tasks with nan so that everyone has the same
                # number of things to send.  Same for list of vecs with None.
                # The empty lists will not do anything when enumerated, so no 
                # inner products will be taken.  nan is inserted into the 
                # indices because then min/max of the indices can be taken.
                
                if my_num_rows != len(row_vecs):
                    raise ValueError('Number of rows assigned does not ' +\
                        'match number of vecs in memory.')
                if my_num_rows > 0 and my_num_rows < max_num_to_send:
                    my_row_indices += [np.nan] * (max_num_to_send - my_num_rows)
                    row_vecs += [[]] * (max_num_to_send - my_num_rows)
                """
                for send_index in range(max_num_to_send):
                    # Only processors responsible for rows communicate
                    if my_num_rows > 0:  
                        # Send row vecs, in groups of num_cols_per_proc_chunk
                        # These become columns in the ensuing computation
                        start_col_index = send_index * num_cols_per_proc_chunk
                        end_col_index = min(start_col_index + 
                            num_cols_per_proc_chunk, my_num_rows)   
                        col_vecs_send = (
                            row_vecs[start_col_index:end_col_index], 
                            my_row_indices[start_col_index:end_col_index])
                        
                        # Create unique tags based on ranks
                        send_tag = my_rank * (
                            _parallel.get_num_procs() + 1) + dest_rank
                        recv_tag = source_rank * (
                            _parallel.get_num_procs() + 1) + my_rank
                        
                        # Send and receieve data.  The Wait() command after the
                        # receive prevents a race condition not fixed by sync().
                        # The Wait() is very important for the non-
                        # blocking send (though we are unsure why).
                        request = _parallel.comm.isend(col_vecs_send, 
                            dest=dest_rank, tag=send_tag)                       
                        col_vecs_recv = _parallel.comm.recv(source = 
                            source_rank, tag=recv_tag)
                        request.Wait()
                        col_vecs = col_vecs_recv[0]
                        my_col_indices = col_vecs_recv[1]
                        
                        for row_index in range(my_row_indices[0], 
                            my_row_indices[-1] + 1):
                            for col_vec_index, col_vec in enumerate(col_vecs):
                                IP_mat[row_index, my_col_indices[
                                    col_vec_index]] = self.inner_product(
                                    row_vecs[row_index - my_row_indices[0]],
                                    col_vec)
                            if (time() - self.prev_print_time) > \
                                self.print_interval:
                                num_completed_IPs = (np.abs(IP_mat)>0).sum()
                                percent_completed_IPs = \
                                    (100.*2*num_completed_IPs * \
                                    _parallel.get_num_MPI_workers())/\
                                    (num_vecs**2)
                                self.print_msg(
                                    ('Completed %.1f%% of inner products') % 
                                    percent_completed_IPs, sys.stderr)
                                self.prev_print_time = time()
                    
                    # Sync after send/receive   
                    _parallel.barrier()  
            
            # Fill in the rectangular portion next to each triangle (if nec.).
            # Start at index after last row, continue to last column. This part
            # of the code is the same as in compute_IP_mat, as of 
            # revision 141.  
            for start_col_index in range(end_row_index, num_vecs, 
                num_cols_per_chunk):
                end_col_index = min(start_col_index + num_cols_per_chunk, 
                    num_vecs)
                proc_col_tasks = _parallel.find_assignments(list(range(
                    start_col_index, end_col_index)))[_parallel.get_rank()]
                        
                # Pass the col vecs to proc with rank -> mod(rank+1,numProcs) 
                # Must do this for each processor, until data makes a circle
                col_vecs_recv = (None, None)
                if len(proc_col_tasks) > 0:
                    col_indices = list(range(proc_col_tasks[0], 
                        proc_col_tasks[-1]+1))
                else:
                    col_indices = []
                    
                for num_passes in range(_parallel.get_num_procs()):
                    # If on the first pass, get the col vecs, no send/recv
                    # This is all that is called when in serial, loop iterates
                    # once.
                    if num_passes == 0:
                        if len(col_indices) > 0:
                            col_vecs = [col_handle.get() \
                                for col_handle in vec_handles[col_indices[0]:\
                                    col_indices[-1] + 1]]
                        else:
                            col_vecs = []
                    else: 
                        # Determine whom to communicate with
                        dest = (_parallel.get_rank() + 1) % _parallel.\
                            get_num_procs()
                        source = (_parallel.get_rank() - 1) % _parallel.\
                            get_num_procs()    
                            
                        # Create unique tag based on ranks
                        send_tag = _parallel.get_rank() * (_parallel.\
                            get_num_procs() + 1) + dest
                        recv_tag = source*(_parallel.get_num_procs() + 1) +\
                            _parallel.get_rank()    
                        
                        # Collect data and send/receive
                        col_vecs_send = (col_vecs, col_indices)     
                        request = _parallel.comm.isend(col_vecs_send, dest=\
                            dest, tag=send_tag)
                        col_vecs_recv = _parallel.comm.recv(source=source, 
                            tag=recv_tag)
                        request.Wait()
                        _parallel.barrier()
                        col_indices = col_vecs_recv[1]
                        col_vecs = col_vecs_recv[0]
                        
                    # Compute the IPs for this set of data col_indices stores
                    # the indices of the IP_mat columns to be
                    # filled in.
                    if len(proc_row_tasks) > 0:
                        for row_index in range(proc_row_tasks[0],
                            proc_row_tasks[-1]+1):
                            for col_vec_index, col_vec in enumerate(col_vecs):
                                IP_mat[row_index, col_indices[
                                    col_vec_index]] = self.inner_product(
                                    row_vecs[row_index - proc_row_tasks[0]],
                                    col_vec)
                        if (
                            (time() - self.prev_print_time) > 
                            self.print_interval):
                            num_completed_IPs = (np.abs(IP_mat)>0).sum()
                            percent_completed_IPs = (100.*2*num_completed_IPs *
                                _parallel.get_num_MPI_workers())/(num_vecs**2)
                            self.print_msg(('Completed %.1f%% of inner ' + 
                                'products')%percent_completed_IPs, sys.stderr)
                            self.prev_print_time = time()

            # Completed a chunk of rows and all columns on all processors.
            # Finished row_vecs loop, delete memory used
            del row_vecs                     
        
        # Assign the triangular portion chunks into IP_mat.
        if _parallel.is_distributed():
            IP_mat = _parallel.custom_comm.allreduce(IP_mat)
       
        # Create a mask for the repeated values.  Select values that are zero
        # in the upper triangular portion (not computed there) but nonzero in
        # the lower triangular portion (computed there).  For the case where
        # the inner product is not perfectly symmetric, this will select the
        # computation done in the upper triangular portion.
        mask = np.multiply(IP_mat == 0, IP_mat.T != 0)
        
        # Collect values below diagonal
        IP_mat += np.multiply(np.triu(IP_mat.T, 1), mask)
        
        # Symmetrize matrix
        IP_mat = np.triu(IP_mat) + np.triu(IP_mat, 1).T
        
        percent_completed_IPs = 100.
        self.print_msg(('Completed %.1f%% of inner ' + 
            'products')%percent_completed_IPs, sys.stderr)
        self.prev_print_time = time()
        
        _parallel.barrier()
        return IP_mat
    
    def lin_combine(self, sum_vec_handles, basis_vec_handles, coeff_mat,
        coeff_mat_col_indices=None):
        """Computes linear combination(s) of basis vector objects and calls
        ``put`` on result(s), using handles.
        
        Args:
            ``sum_vec_handles``: List of handles for the sum vector objects.
                
            ``basis_vec_handles``: List of handles for the basis vector objects.
                
            ``coeff_mat``: Matrix whose rows correspond to basis vectors and
            whose columns correspond to sum vectors.  The rows and columns
            correspond, by index, to the lists ``basis_vec_handles`` and
            ``sum_vec_handles``.  In matrix notation, we can write ``sums =
            basis * coeff_mat``
        
        Kwargs:
            ``coeff_mat_col_indices``: List of column indices.  Only the
            ``sum_vecs`` corresponding to these columns of the coefficient
            matrix are computed.
            
        Each MPI worker (processor) retrieves a subset of the basis vectors to
        compute as many outputs as an MPI worker (processor) can have in memory
        at once. Each MPI worker (processor) computes the "layers" from the
        basis it is responsible for, and for as many modes as it can fit in
        memory. The layers from all MPI workers (processors) are summed
        together to form the ``sum_vecs`` and ``put`` is called on each.
        
        Scaling is:
        
          num gets/worker = :math:`n_s/(n_p*(max-2)) * n_b/n_p`
          
          passes/worker = :math:`(n_p-1) * n_s/(n_p*(max-2)) * (n_b/n_p)`
          
          scalar multiplies/worker = :math:`n_s*n_b/n_p`
          
        where :math:`n_s` is number of sum vecs, :math:`n_b` is 
        number of basis vecs,
        :math:`n_p` is number of processors, 
        :math:`max` = ``max_vecs_per_node``.
        """
        sum_vec_handles = util.make_iterable(sum_vec_handles)
        basis_vec_handles = util.make_iterable(basis_vec_handles)
        num_bases = len(basis_vec_handles)
        num_sums = len(sum_vec_handles)
        if coeff_mat_col_indices is not None:
            coeff_mat = coeff_mat[:, coeff_mat_col_indices]
        if num_bases != coeff_mat.shape[0]:
            raise ValueError(('Number of coeff_mat rows (%d) does not equal '
                'number of basis handles (%d)'%(coeff_mat.shape[0],num_bases)))
        if num_sums != coeff_mat.shape[1]:
            raise ValueError(('Number of coeff_mat cols (%d) does not equal '
                'number of output handles (%d)')%(coeff_mat.shape[1],num_sums))
        
        # Estimate time it will take
        # Burn the first one for slow imports
        test_vec_burn = basis_vec_handles[0].get()
        test_vec_burn_3 = test_vec_burn + 2.*test_vec_burn
        del test_vec_burn, test_vec_burn_3
        start_time = time()
        test_vec = basis_vec_handles[0].get()
        get_time = time() - start_time
        start_time = time()
        test_vec_3 = test_vec + 2.*test_vec
        add_scale_time = time() - start_time
        del test_vec, test_vec_3
        
        vecs_per_worker = self.max_vecs_per_node * _parallel.get_num_nodes() / \
            _parallel.get_num_MPI_workers()
        num_gets = num_sums/(_parallel.get_num_MPI_workers()*(\
            vecs_per_worker-2)) + \
            num_bases/_parallel.get_num_MPI_workers()
        num_add_scales = num_sums*num_bases/_parallel.get_num_MPI_workers()
        self.print_msg('Linear combinations will take at least %.1f minutes'%
            (num_gets*get_time/60. + num_add_scales*add_scale_time/60.))

        # convenience
        rank = _parallel.get_rank()

        # num_bases_per_proc_chunk is the num of bases each proc gets at once.
        num_bases_per_proc_chunk = 1
        num_sums_per_proc_chunk = self.max_vecs_per_proc - \
            num_bases_per_proc_chunk
        
        basis_tasks = _parallel.find_assignments(list(range(num_bases)))
        sum_tasks = _parallel.find_assignments(list(range(num_sums)))

        # Find max number tasks among all processors
        max_num_basis_tasks = max([len(tasks) for tasks in basis_tasks])
        max_num_sum_tasks = max([len(tasks) for tasks in sum_tasks])
        
        # These variables are the number of iters through loops that retrieve 
        # ("get")
        # and "put" basis and sum vecs.
        num_basis_get_iters = int(np.ceil(
            max_num_basis_tasks*1./num_bases_per_proc_chunk))
        num_sum_put_iters = int(np.ceil(
            max_num_sum_tasks*1./num_sums_per_proc_chunk))
        if num_sum_put_iters > 1:
            self.print_msg('Warning: The basis vecs, ' 
                'of which there are %d, will be retrieved %d times each. '
                'If possible, increase number of nodes or '
                'max_vecs_per_node to reduce redundant retrieves and get a '
                'big speedup.'%(num_bases, num_sum_put_iters))
               
        for sum_put_index in range(num_sum_put_iters):
            if len(sum_tasks[rank]) > 0:
                start_sum_index = min(sum_tasks[rank][0] + 
                    sum_put_index*num_sums_per_proc_chunk, 
                    sum_tasks[rank][-1]+1)
                end_sum_index = min(start_sum_index+num_sums_per_proc_chunk,
                    sum_tasks[rank][-1]+1)

                # Create empty list on each processor
                sum_layers = [None]*(end_sum_index - start_sum_index)
            else:
                start_sum_index = 0
                end_sum_index = 0
                sum_layers = []

            for basis_get_index in range(num_basis_get_iters):
                if len(basis_tasks[rank]) > 0:    
                    start_basis_index = min(basis_tasks[rank][0] + 
                        basis_get_index*num_bases_per_proc_chunk, 
                        basis_tasks[rank][-1]+1)
                    end_basis_index = min(start_basis_index + 
                        num_bases_per_proc_chunk, basis_tasks[rank][-1]+1)
                    basis_indices = list(
                        range(start_basis_index, end_basis_index))
                else:
                    basis_indices = []
                
                # Pass the basis vecs to proc with rank -> mod(rank+1,num_procs)
                # Must do this for each processor, until data makes a circle
                basis_vecs_recv = (None, None)

                for pass_index in range(_parallel.get_num_procs()):
                    # If on the first pass, retrieve the basis vecs, 
                    # no send/recv.
                    # This is all that is called when in serial, 
                    # loop iterates once.
                    if pass_index == 0:
                        if len(basis_indices) > 0:
                            basis_vecs = [basis_handle.get() \
                                for basis_handle in basis_vec_handles[
                                    basis_indices[0]:basis_indices[-1]+1]]
                        else:
                            basis_vecs = []
                    else:
                        # Figure out with whom to communicate
                        source = (_parallel.get_rank()-1) % \
                            _parallel.get_num_procs()
                        dest = (_parallel.get_rank()+1) % \
                            _parallel.get_num_procs()
                        
                        #Create unique tags based on ranks
                        send_tag = _parallel.get_rank() * \
                            (_parallel.get_num_procs()+1) + dest
                        recv_tag = source*(_parallel.get_num_procs()+1) + \
                            _parallel.get_rank()
                        
                        # Send/receive data
                        basis_vecs_send = (basis_vecs, basis_indices)
                        request = _parallel.comm.isend(basis_vecs_send,  
                            dest=dest, tag=send_tag)                       
                        basis_vecs_recv = _parallel.comm.recv(
                            source=source, tag=recv_tag)
                        request.Wait()
                        _parallel.barrier()
                        basis_indices = basis_vecs_recv[1]
                        basis_vecs = basis_vecs_recv[0]
                    
                    # Compute the scalar multiplications for this set of data.
                    # basis_indices stores the indices of the coeff_mat to
                    # use.
                    for sum_index in range(start_sum_index, end_sum_index):
                        for basis_index, basis_vec in enumerate(basis_vecs):
                            sum_layer = basis_vec * \
                                coeff_mat[basis_indices[basis_index],\
                                sum_index]
                            if sum_layers[sum_index-start_sum_index] is None:
                                sum_layers[sum_index-start_sum_index] = \
                                    sum_layer
                            else:
                                sum_layers[sum_index-start_sum_index] += \
                                    sum_layer
                        if (
                            (time() - self.prev_print_time) > 
                            self.print_interval):
                            self.print_msg(
                                'Completed %.1f%% of linear combinations' %
                                (sum_index*100./len(sum_tasks[rank])))
                            self.prev_print_time = time()

            # Completed this set of sum vecs, puts them to memory or file
            for sum_index in range(start_sum_index, end_sum_index):
                sum_vec_handles[sum_index].put(
                    sum_layers[sum_index-start_sum_index])
            del sum_layers
        
        self.print_msg('Completed %.1f%% of linear combinations' % 100.)
        self.prev_print_time = time()
        _parallel.barrier() 
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.inner_product == other.inner_product and 
            self.verbosity == other.verbosity)
        
    def __ne__(self, other):
        return not self.__eq__(other)


