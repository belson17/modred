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
      
      # Compute action of operator A on direct_modes outside of python.
      A = LookUpOperator(direct_modes, A_on_modes)
      A_reduced = LTIGalerkinProjection(inner_product).reduce_A(
        A, direct_modes, adjoint_modes)
    
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
        
        ``direct_modes``: List of direct modes (vecs or vec handles). 
    
    Kwargs:
        ``adjoint_modes``: List of adjoint modes (vecs or vec handles).
            If not given, then ``direct_modes`` are used.
    
        ``are_modes_orthonormal``: Bool for the bi-orthonormality of the modes.
            ``True`` if the modes are orthonormal.
            
        ``put_mat``: Function to put a matrix elsewhere (memory or file).
        
        ``verbosity``: 1 prints progress and warnings, 0 almost nothing.
        
        ``max_vecs_per_node``: Max number of vectors in memory per node.
    
    This class creates either discrete or continuous time models from
    modes (could obtain modes from :py:class:`POD` and :py:class:`BPOD`).
    
    If the ``direct_modes`` (and optionally the ``adjoint_modes``) are given
    as vectors rather than vector handles, then use the ``*_in_memory`` 
    member functions when there is a distinction. 
    For vector handles, use the regular functions.
    
    Usage::
        
      LTI_proj = LTIGalerkinProjection(inner_product, direct_modes,
        adjoint_modes=adjoint_modes, are_modes_orthonormal=True)
      A, B, C = LTI_proj.compute_model(A, B, C, num_inputs)
        
    """
    def __init__(self, inner_product, direct_modes, adjoint_modes=None, 
        are_modes_orthonormal=False, put_mat=util.save_array_text, verbosity=1, 
        max_vecs_per_node=10000):
        """Constructor"""
        self.inner_product = inner_product
        self.direct_modes = direct_modes
        if adjoint_modes is None:
            self.adjoint_modes = self.direct_modes
        else:
            self.adjoint_modes = adjoint_modes
        self.are_modes_orthonormal = are_modes_orthonormal
        self.put_mat = put_mat
        self.vec_space = VectorSpace(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        self.model_dim = None
        self.verbosity = verbosity
        self._proj_mat = None
        self.A_reduced = None
        self.B_reduced = None
        self.C_reduced = None

    def put_A_reduced(self, A_reduced_dest):
        """Put reduced A matrix to ``A_reduced_dest``"""
        if _parallel.is_rank_zero():
            self.put_mat(self.A_reduced, A_reduced_dest)
        _parallel.barrier()
        
    def put_B_reduced(self, B_reduced_dest):
        """Put reduced B matrix to ``B_reduced_dest``"""
        if _parallel.is_rank_zero():
            self.put_mat(self.B_reduced, B_reduced_dest)
        _parallel.barrier()
        
    def put_C_reduced(self, C_reduced_dest):
        """Put reduced C matrix to ``C_reduced_dest``"""
        if _parallel.is_rank_zero():
            self.put_mat(self.C_reduced, C_reduced_dest)
        _parallel.barrier()
        
    def put_model(self, A_reduced_dest, B_reduced_dest, C_reduced_dest):
        """Put reduced A, B, and C mats (numpy arrays) to destinations.
        
        Args:
            ``A_reduced_dest``: Destination for ``A_reduced``.
            
            ``B_reduced_dest``: Destination for ``B_reduced``.
            
            ``C_reduced_dest``: Destination for ``C_reduced``.
        """
        self.put_A_reduced(A_reduced_dest)
        self.put_B_reduced(B_reduced_dest)
        self.put_C_reduced(C_reduced_dest)
        
        
    def compute_model(self, A, B, C, num_inputs, model_dim=None):
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
                
            ``B``: callable which takes basis vec, e_j, returns "B*e_j".
            
            ``C``: callable which takes a direct mode handle, returns "C*modes".

            ``num_inputs``: number of inputs to system
            
        Kwargs:
            ``model_dim``: number of states to keep in the model. 
                Can omit if already given. Default is maximum possible.
        """
        self.reduce_A(A, model_dim=model_dim)
        self.reduce_B(B, num_inputs)
        self.reduce_C(C)
        return self.A_reduced, self.B_reduced, self.C_reduced
        
    def compute_model_in_memory(self, A, B, C, num_inputs, model_dim=None):
        """See :py:meth:`compute_model`, but takes vecs instead of handles."""
        self.reduce_A_in_memory(A, model_dim=model_dim)
        self.reduce_B_in_memory(B, num_inputs)
        self.reduce_C_in_memory(C)
        return self.A_reduced, self.B_reduced, self.C_reduced
        
    
    def reduce_A_in_memory(self, A, model_dim=None):
        """Computes and returns the continous or discrete time A matrix.
        
        See :py:meth:`reduce_A`, but use when modes are vecs not vec handles."""
        if model_dim is not None:
            self.model_dim = model_dim
        if self.model_dim is None:
            self.model_dim = min(len(self.direct_modes), 
                len(self.adjoint_modes))
        A_on_direct_modes = map(A, self.direct_modes[:self.model_dim])
        self.A_reduced = self.vec_space.compute_inner_product_mat(
            map(InMemoryVecHandle, self.adjoint_modes[:self.model_dim]),
            map(InMemoryVecHandle, A_on_direct_modes))
        if not self.are_modes_orthonormal:
            self.A_reduced = N.dot(self._get_proj_mat_in_memory(),
                self.A_reduced)
        return self.A_reduced
        
    
    def reduce_A(self, A, model_dim=None):
        """Computes and returns the continous or discrete time A matrix.
        
        Args:
            ``A``: Callable which takes a mode handle, returns handle "A*mode".
            
        Kwargs:
            ``model_dim``: Number of modes/states to keep in the model. 
                Can omit if already given. Default is maximum possible.
        
        Returns:
            ``A_reduced``: reduced A matrix (2D numpy array).
        """
        if model_dim is not None:
            self.model_dim = model_dim
        if self.model_dim is None:
            self.model_dim = min(len(self.direct_modes), 
                len(self.adjoint_modes))
        A_on_direct_modes = map(A, self.direct_modes[:self.model_dim])
        self.A_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_modes[:self.model_dim], A_on_direct_modes)
        if not self.are_modes_orthonormal:
            self.A_reduced = N.dot(self._get_proj_mat(), self.A_reduced)
        return self.A_reduced
                
    
    def reduce_B(self, B, num_inputs, model_dim=None):
        """Computes and returns the reduced B matrix.
        
        Args:		
            ``B``: Callable which takes a standard basis element (array),
            returns handle "B*e_j".
            
            ``num_inputs``: Number of inputs to the system. 
                
        Kwargs:
            ``model_dim``: Number of modes/states to keep in the model. 
                Can omit if already given.
		
		Returns:
		    ``B_reduced``: Reduced B matrix (2D numpy array).
		    
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

        if model_dim is not None:
            self.model_dim = model_dim
        if self.model_dim is None:
            self.model_dim = len(adjoint_mode_handles)
        B_on_basis = map(B, standard_basis(num_inputs))
        self.B_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_modes[:self.model_dim], B_on_basis)
        if not self.are_modes_orthonormal:
            self.B_reduced = N.dot(self._get_proj_mat(), self.B_reduced)
        return self.B_reduced

    def reduce_B_in_memory(self, B, num_inputs, model_dim=None):
        """Computes and returns the reduced B matrix.
        
		See :py:meth:`reduce_B`, but use when modes are vecs not vec handles."""
        if model_dim is not None:
            self.model_dim = model_dim
        if self.model_dim is None:
            self.model_dim = len(self.adjoint_modes)
        B_on_basis = map(B, standard_basis(num_inputs))
        self.B_reduced = self.vec_space.compute_inner_product_mat(
            map(InMemoryVecHandle, self.adjoint_modes[:self.model_dim]), 
            map(InMemoryVecHandle, B_on_basis))
        if not self.are_modes_orthonormal:
            self.B_reduced = N.dot(self._get_proj_mat_in_memory(),
                self.B_reduced)
        return self.B_reduced

        
        
    def reduce_C(self, C, model_dim=None):
        """Computes and returns the reduced C matrix.
               
        Args: 
            ``C``: Callable that takes mode handle, returns 1D array "C*mode".
        		
        Kwargs:
            ``model_dim``: Number of modes/states to keep in the model. 
                Can omit if already given.
        
        Returns:
            ``C_reduced``: Reduced C matrix (2D numpy array).
        """
        if model_dim is not None:
            self.model_dim = model_dim
        if self.model_dim is None:
            self.model_dim = len(self.direct_modes)
        C_on_direct_modes = map(C, self.direct_modes[:self.model_dim])
        # Force each output from C to be a 1D array
        C_on_direct_modes = [N.array(m.squeeze(), ndmin=1) 
            for m in C_on_direct_modes]
        self.C_reduced = N.array(C_on_direct_modes, ndmin=2).T
        return self.C_reduced
    
    
    def reduce_C_in_memory(self, C, model_dim=None):
        """Same as :py:meth:`reduce_C`, but modes are vecs not vec handles.
        
        Only for convenience and consistency, since it doesn't matter if 
        the modes are vecs or vec handles for :py:meth:`reduce_C`."""
        return self.reduce_C(C, model_dim=model_dim)
        
        
    def _get_proj_mat(self, direct_modes=None, adjoint_modes=None,
        are_modes_orthonormal=None):
        """Gets the projection mat, i.e. inv(Psi^* Phi).
        
        Kwargs:
           ``direct_modes``: Direct mode handles, default ``self.direct_modes``.

           ``adjoint_modes``: Adjoint mode handles, default ``self.adjoint_modes``.

           ``are_modes_orthonormal``: Bool, default ``self.are_modes_orthonormal``.
                
        There are only arguments to avoid repeating code for an "in_memory" 
        version. Otherwise one could just use the class self variables.
        """            
        if self._proj_mat is None:
            if direct_modes is None: 
                direct_modes = self.direct_modes
            if adjoint_modes is None: 
                adjoint_modes = self.adjoint_modes
            if are_modes_orthonormal is None: 
                are_modes_orthonormal = self.are_modes_orthonormal
            
            # Check if direct and adjoint modes are equal
            symmetric = True
            for d_mode, a_mode in zip(self.direct_modes[:self.model_dim], 
                self.adjoint_modes[:self.model_dim]):
                if not util.smart_eq(d_mode, a_mode):
                    symmetric = False
            
            if symmetric:
                IP_mat = self.vec_space.compute_symmetric_inner_product_mat(
                    self.direct_modes[:self.model_dim])
            else:
                IP_mat = self.vec_space.compute_inner_product_mat(
                    self.adjoint_modes[:self.model_dim], 
                    self.direct_modes[:self.model_dim])
            
            self._proj_mat = N.linalg.inv(IP_mat)
        return self._proj_mat
    
    
    def _get_proj_mat_in_memory(self):
        """Gets the projection mat, i.e. inv(Psi^* Phi).
        
        See :py:meth:`_get_proj_mat`, but modes are vecs not handles."""
        direct_mode_handles = map(InMemoryVecHandle, self.direct_modes)
        adjoint_mode_handles = map(InMemoryVecHandle, self.adjoint_modes)
        return self._get_proj_mat(direct_mode_handles, adjoint_mode_handles)
            
