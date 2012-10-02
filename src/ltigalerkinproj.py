"""Module for Galerkin projection of LTI systems."""

import numpy as N

import util
from vectors import InMemoryVecHandle
from vectorspace import VectorSpace
from parallel import parallel_default_instance
_parallel = parallel_default_instance

def standard_basis(num_dims):
    """Returns list of standard basis vecs of space R^n.
    
    Args:
        ``num_dims``: Integer number of dimensions.
    
    Returns:
        ``basis``: The standard basis as a list of 1D arrays.
            ``[array([1,0,...]), array([0,1,...]), ...]``.
    """
    return list(N.identity(num_dims))


def compute_derivs(vec_handles, adv_vec_handles, deriv_vec_handles, dt):
    """Computes 1st-order time derivatives of vectors. 
    
    Args:
        ``vec_handles``: List of vec handles.
        
        ``adv_vec_handles``: List of vec handles for vecs advanced ``dt`` in time.
        
        ``deriv_vec_handles``: List of vec handles for time derivatives of vecs.
        
        ``dt``: Time step of ``adv_vec_handles``
        
    Computes d(``vec``)/dt = (``vec``(t=dt) - ``vec``(t=0)) / dt.
    """
    num_vecs = len(vec_handles)
    if num_vecs != len(adv_vec_handles) or \
        num_vecs != len(deriv_vec_handles):
        raise RuntimeError('Number of vectors not equal')
    
    vec_index_tasks = _parallel.find_assignments(range(num_vecs))[
        _parallel.get_rank()]
    
    for i in vec_index_tasks:
        vec = vec_handles[i].get()
        vec_dt = adv_vec_handles[i].get()
        deriv_vec_handles[i].put((vec_dt - vec)*(1./dt))
    _parallel.barrier()


def compute_derivs_in_memory(vecs, adv_vecs, dt):
    """Computes 1st-order time derivatives of vectors. 
    
    Args:
        ``vecs``: List of vecs.
        
        ``adv_vecs``: List of vecs advanced ``dt`` in time.
        
        ``dt``: Time step of ``adv_vecs``.
                
    Returns:
        ``deriv_vecs``: List of time-derivs of vectors.
        
    In parallel, each processor returns all derivatives.
    """
    vec_handles = map(InMemoryVecHandle, vecs)
    adv_vec_handles = map(InMemoryVecHandle, adv_vecs)
    deriv_vec_handles = [InMemoryVecHandle() for i in xrange(len(adv_vecs))]
    compute_derivs(vec_handles, adv_vec_handles, deriv_vec_handles, dt)
    deriv_vecs = [v.get() for v in deriv_vec_handles]
    if _parallel.is_distributed():
        # Remove empty entries
        for i in range(deriv_vecs.count(None)):
            deriv_vecs.remove(None)
        all_deriv_vecs = util.flatten_list(
            _parallel.comm.allgather(deriv_vecs))
        return all_deriv_vecs
    else:
        return deriv_vecs


class LookUpOperator(object):
    """Looks up precomputed operation on vectors or vector handles.
    
    Args:
        ``vecs``: List of vecs or handles on which operator acted.
        
        ``operated_on_vecs``: List of vecs or handles resulting from operation.
    
    Useful when the action of an operator on a set of vectors (e.g. A on the
    direct modes) is computed outside of python and modred.
    
    Usage::
      
      # Compute action of operator A on basis_vecs outside of python.
      A = LookUpOperator(basis_vecs, A_on_modes)
      A_reduced = LTIGalerkinProjection(inner_product).reduce_A(
        A, basis_vecs, adjoint_basis_vecs)
    
    """
    def __init__(self, vecs, operated_on_vecs):
        self.vecs = vecs
        self.operated_on_vecs = operated_on_vecs
    def __call__(self, vec):
        """Given vec, finds and returns corresponding operated_on_vec."""
        for v, ov in zip(self.vecs, self.operated_on_vecs):
            if util.smart_eq(v, vec):
                return ov
        raise RuntimeError('Vector was not previously operated on')

       
class MatrixOperator(object):
    """Callable class that operates on 1D arrays via matrix multiplication.
    
    Args:
        ``mat``: A numpy matrix/array.
    
    Usage::
    
      my_mat_op = MatrixOperator(N.random.random((3, 3)))
      product = my_mat_op(N.random.random(3))
      
    """
    def __init__(self, mat):
        self.mat = mat
    def __call__(self, vec):
        """Does matrix multiplication on vec, a numpy array or matrix"""
        return N.dot(self.mat, vec)
   

class LTIGalerkinProjection(object):
    """Computes the reduced-order model matrices from modes for an LTI system.
    
    Args:
        ``inner_product``: Function to take inner product of vectors.
        
        ``basis_vecs``: List of basis vectors (vecs or vec handles). 
    
    Kwargs:
        ``adjoint_basis_vecs``: List of adjoint vectors (vecs or vec handles).
            If not given, then ``basis_vecs`` are used.
    
        ``is_basis_orthonormal``: Bool for bi-orthonormality of the basis vecs.
            ``True`` if the modes are orthonormal. Default is ``False``.
            
        ``put_mat``: Function to put a matrix elsewhere (memory or file).
        
        ``verbosity``: 1 prints progress and warnings, 0 almost nothing.
        
        ``max_vecs_per_node``: Max number of vectors in memory per node.
    
    This class projects discrete or continuous time dynamics onto a set of basis
    vectors, resulting in models. Often the basis vecs are modes from 
    :py:class:`POD` or :py:class:`BPOD`.
    
    If the ``basis_vecs`` (and optionally the ``adjoint_basis_vecs``) are given
    as vectors rather than vector handles, then use the ``*_in_memory`` 
    member functions when there is a distinction. 
    For vector handles, use the regular functions.
    
    Usage::
        
      LTI_proj = LTIGalerkinProjection(inner_product, basis_vecs,
        adjoint_basis_vecs=adjoint_basis_vecs, is_basis_orthonormal=True)
      A, B, C = LTI_proj.compute_model(A, B, C, num_inputs)
        
    """
    def __init__(self, inner_product, basis_vecs, adjoint_basis_vecs=None, 
        is_basis_orthonormal=False, put_mat=util.save_array_text, verbosity=1, 
        max_vecs_per_node=10000):
        """Constructor"""
        self.inner_product = inner_product
        self.basis_vecs = basis_vecs
        if adjoint_basis_vecs is None:
            self.adjoint_basis_vecs = self.basis_vecs
        else:
            self.adjoint_basis_vecs = adjoint_basis_vecs
            if len(self.adjoint_basis_vecs) != len(self.basis_vecs):
                raise ValueError('Number of basis vecs is not equal to the '
                    'number of adjoint basis vecs')
        self.is_basis_orthonormal = is_basis_orthonormal
        self.put_mat = put_mat
        self.vec_space = VectorSpace(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.verbosity = verbosity
        self._proj_mat = None
        self.A_reduced = None
        self.B_reduced = None
        self.C_reduced = None

        
    def put_A_reduced(self, dest):
        """Put reduced A matrix to ``dest``."""
        _parallel.call_from_rank_zero(self.put_mat, self.A_reduced, dest)
        _parallel.barrier()
        
    def put_B_reduced(self, dest):
        """Put reduced B matrix to ``dest``."""
        _parallel.call_from_rank_zero(self.put_mat, self.B_reduced, dest)
        _parallel.barrier()
        
    def put_C_reduced(self, dest):
        """Put reduced C matrix to ``dest``."""
        _parallel.call_from_rank_zero(self.put_mat, self.C_reduced, dest)
        _parallel.barrier()
        
    def put_model(self, A_reduced_dest, B_reduced_dest, C_reduced_dest):
        """Put reduced A, B, and C matrices to destinations.
        
        Args:
            ``A_reduced_dest``: Destination for ``A_reduced``.
            
            ``B_reduced_dest``: Destination for ``B_reduced``.
            
            ``C_reduced_dest``: Destination for ``C_reduced``.
        """
        self.put_A_reduced(A_reduced_dest)
        self.put_B_reduced(B_reduced_dest)
        self.put_C_reduced(C_reduced_dest)
        
        
    def compute_model(self, A, B, C, num_inputs):
        """Computes and returns the reduced matrices. For use with vec handles.
        
        Args:
            ``A``: Callable which takes a mode handle, returns "A*mode".
                There are two primary flavors of ``A``.
                First, it can compute the action of A on the mode.
                Second, if the action of A on all of the modes is computed 
                outside of python, one can use :py:class:`LookUpOperator`.
                
                ``A`` can correpond to discrete or continuous time.
                For continuous time systems, see also 
                :py:meth:`compute_derivs`.
                
            ``B``: Callable which takes basis vec, e_j, returns "B*e_j".
            
            ``C``: Callable which takes a direct mode handle, returns "C*modes".

            ``num_inputs``: Number of inputs to the system.
        """
        self.reduce_A(A)
        self.reduce_B(B, num_inputs)
        self.reduce_C(C)
        return self.A_reduced, self.B_reduced, self.C_reduced
        
    def compute_model_in_memory(self, A, B, C, num_inputs):
        """See :py:meth:`compute_model`, but takes vecs instead of handles."""
        self.reduce_A_in_memory(A)
        self.reduce_B_in_memory(B, num_inputs)
        self.reduce_C_in_memory(C)
        return self.A_reduced, self.B_reduced, self.C_reduced

    
    def reduce_A_in_memory(self, A):
        """Computes and returns the continous or discrete time A matrix.
        
        See :py:meth:`reduce_A`, but use when modes are vecs not vec handles."""
        A_on_basis_vecs = map(A, self.basis_vecs)
        self.A_reduced = self.vec_space.compute_inner_product_mat_in_memory(
            self.adjoint_basis_vecs, A_on_basis_vecs)
        if not self.is_basis_orthonormal:
            self.A_reduced = N.dot(self._get_proj_mat_in_memory(), 
                self.A_reduced)
        return self.A_reduced

    def reduce_A(self, A):
        """Computes and returns the continous or discrete time A matrix.
        
        Args:
            ``A``: Callable which takes a mode handle, returns handle "A*mode".
                    
        Returns:
            ``A_reduced``: Reduced A matrix.
        """
        A_on_basis_vecs = map(A, self.basis_vecs)
        self.A_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_basis_vecs, A_on_basis_vecs)
        if not self.is_basis_orthonormal:
            self.A_reduced = N.dot(self._get_proj_mat(), self.A_reduced)
        return self.A_reduced
        
    
    def reduce_B(self, B, num_inputs):
        """Computes and returns the reduced B matrix.
        
        Args:		
            ``B``: Callable which takes a standard basis element (array),
            returns handle "B*e_j".
            
            ``num_inputs``: Number of inputs to the system.
        
        Returns:
		    ``B_reduced``: Reduced B matrix.
		    
        Tip: If you found the modes via sampling IC responses to B and C,
        then your impulse responses may be missing a factor of 1/dt
        (where dt is the first simulation time step).
        This can be remedied by multiplying the reduced B by dt.
        """
        # TODO: Check this description, then move to docstring
        #To see this dt effect, consider:
        #
        #dx/dt = Ax+Bu, approximate as (x^(k+1)-x^k)/dt = Ax^k + Bu^k.
        #Rearranging terms, x^(k+1) = (dt*I+A)x^k + dt*Bu^k.
        #The impulse response is: x^0=0, u^0=1, and u^k=0 for k>=1.
        #Thus x^1 = dt*B, x^2 = dt*(I+dt*A)*B, ...
        #and y^1 = dt*C*B, y^2 = dt*C*(I+dt*A)*B, ...
        #However, the impulse response to the true discrete-time system is
        #x^1 = B, x^2 = A_d*B, ...
        #and y^1 = CB, y^2 = CA_d*B, ...
        #(where I+dt*A ~ A_d)
        #The important thing to see is the factor of dt difference.

        B_on_basis_vecs = map(B, standard_basis(num_inputs))
        self.B_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_basis_vecs, B_on_basis_vecs)
        if not self.is_basis_orthonormal:
            self.B_reduced = N.dot(self._get_proj_mat(), self.B_reduced)
        return self.B_reduced
        

    def reduce_B_in_memory(self, B, num_inputs):
        """Computes and returns the reduced B matrix.
        
		See :py:meth:`reduce_B`, but use when modes are vecs not vec handles."""
        B_on_basis_vecs = map(B, standard_basis(num_inputs))
        self.B_reduced = self.vec_space.compute_inner_product_mat_in_memory(
            self.adjoint_basis_vecs, B_on_basis_vecs)
        if not self.is_basis_orthonormal:
            self.B_reduced = N.dot(self._get_proj_mat_in_memory(), 
                self.B_reduced)
        return self.B_reduced
                
        
    def reduce_C(self, C):
        """Computes and returns the reduced C matrix.
               
        Args: 
            ``C``: Callable that takes mode handle, returns 1D array "C*mode".
        
        Returns:
            ``C_reduced``: Reduced C matrix.
        """
        C_on_basis_vecs = map(C, self.basis_vecs)
        # Force each output from C to be a 1D array
        C_on_basis_vecs = [N.array(m.squeeze(), ndmin=1) 
            for m in C_on_basis_vecs]
        self.C_reduced = N.array(C_on_basis_vecs, ndmin=2).T
        return self.C_reduced
    
    
    def reduce_C_in_memory(self, C):
        """Same as :py:meth:`reduce_C`, but modes are vecs not vec handles.
        
        Only for convenience and consistency, since it doesn't matter if 
        the modes are vecs or vec handles for :py:meth:`reduce_C`."""
        return self.reduce_C(C)
        
        
    def _get_proj_mat(self, basis_vecs=None, adjoint_basis_vecs=None):
        """Gets the projection mat, i.e. inv(Psi^* Phi).
        
        Kwargs:
           ``basis_vecs``: Direct mode handles, default ``self.basis_vecs``.

           ``adjoint_basis_vecs``: Adjoint mode handles, default
               ``self.adjoint_basis_vecs``.
                
        There are only arguments to avoid repeating code for an "in_memory" 
        version. Otherwise one could just use the class self variables.
        """            
        if self._proj_mat is None:
            if basis_vecs is None: 
                basis_vecs = self.basis_vecs
            if adjoint_basis_vecs is None: 
                adjoint_basis_vecs = self.adjoint_basis_vecs
            
            # Check if direct and adjoint modes are equal
            symmetric = True
            basis_vec_index = 0
            while symmetric and basis_vec_index < len(basis_vecs):
                if not util.smart_eq(basis_vecs[basis_vec_index], 
                    adjoint_basis_vecs[basis_vec_index]):
                    symmetric = False
                basis_vec_index += 1
                
            if symmetric:
                IP_mat = self.vec_space.compute_symmetric_inner_product_mat(
                    basis_vecs)
            else:
                IP_mat = self.vec_space.compute_inner_product_mat(
                    adjoint_basis_vecs, basis_vecs)
            self._proj_mat = N.linalg.inv(IP_mat)
        return self._proj_mat
    
    
    def _get_proj_mat_in_memory(self):
        """Gets the projection mat, i.e. inv(Psi^* Phi).
        
        See :py:meth:`_get_proj_mat`, but modes are vecs not handles."""
        basis_vec_handles = map(InMemoryVecHandle, self.basis_vecs)
        adjoint_basis_vec_handles = map(InMemoryVecHandle, 
            self.adjoint_basis_vecs)
        return self._get_proj_mat(basis_vec_handles, adjoint_basis_vec_handles)
            
