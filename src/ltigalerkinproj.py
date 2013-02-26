"""Module for Galerkin projection of LTI systems."""

import numpy as N

import util
from vectors import InMemoryVecHandle
from vectorspace import *
from parallel import parallel_default_instance
_parallel = parallel_default_instance

def standard_basis(num_dims):
    """Returns list of standard basis vecs of space :math:`R^n`.
    
    Args:
        ``num_dims``: Integer number of dimensions.
    
    Returns:
        ``basis``: The standard basis as a list of 1D arrays.
            ``[array([1,0,...]), array([0,1,...]), ...]``.
    """
    return list(N.identity(num_dims))



def compute_derivs_handles(vec_handles, adv_vec_handles, deriv_vec_handles, dt):
    """Computes 1st-order time derivatives of vectors using handles. 
    
    Args:
        ``vec_handles``: List of vec handles.
        
        ``adv_vec_handles``: List of vec handles for vecs advanced ``dt`` in
        time.
        
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
        deriv_vec_handles[i].put((1./dt)*(vec_dt - vec))
    _parallel.barrier()


def compute_derivs_arrays(vec_array, adv_vec_array, dt):
    """Computes 1st-order time derivatives of vectors using arrays. 
    
    Args:
        ``vec_array``: 2D array with vectors as columns.
        
        ``adv_vec_array``: 2D array with time-advanced vectors as columns.
        
        ``dt``: Time step between ``vec_array`` and ``adv_vec_array``.
                
    Returns:
        ``deriv_vec_array``: 2D array of with time-derivs of vectors as cols.
    """
    return (adv_vec_array - vec_array)/(1.*dt)



class LTIGalerkinProjectionBase(object):
    def __init__(self, is_basis_orthonormal=False, put_mat=util.save_array_text, 
        verbosity=1):
        self.is_basis_orthonormal = is_basis_orthonormal
        self.verbosity = verbosity
        self.put_mat = put_mat
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
       
        
       
        

class LTIGalerkinProjectionArrays(LTIGalerkinProjectionBase):
    """Computes the reduced-order model matrices from modes for an LTI system.
    
    Args:
        ``basis_vec_array``: 2D array with basis vectors as columns. 
    
    Kwargs:
        ``adjoint_basis_vec_handles``: 2D array with adjoint vectors as columns.
            If not given, then ``basis_vec_handles`` are used.
    
        ``is_basis_orthonormal``: Bool for bi-orthonormality of the basis vecs.
            ``True`` if the basis and adjoint vectors are bi-orthonormal.
            Default is ``False``.
            
        ``put_mat``: Function to put a matrix elsewhere (memory or file).
        
        ``inner_product_weights``: 1D or 2D array, ``Y* weights X``.

    This class projects discrete or continuous time dynamics onto a set of basis
    vectors, resulting in models. Often the basis vecs are modes from 
    POD or BPOD.
    It uses :py:class:`vectorspace.VectorSpaceArrays` for low level functions.
    
    Usage::
        
      LTI_proj = LTIGalerkinProjectionArray(basis_vec_array,
        adjoint_basis_vec_array=adjoint_basis_vec_array, 
        is_basis_orthonormal=True)
      A, B, C = LTI_proj.compute_model(A_on_basis_vec_array, 
        B_on_standard_basis_array, C_on_basis_vecs)
        
    """
    def __init__(self, basis_vec_array, adjoint_basis_vec_array=None,  
        is_basis_orthonormal=False, inner_product_weights=None,
        put_mat=util.save_array_text, verbosity=1):
        """Constructor"""
        LTIGalerkinProjectionBase.__init__(self, is_basis_orthonormal,
            put_mat=put_mat, verbosity=verbosity)
        if _parallel.is_distributed():
            raise RuntimeError('Not for parallel use.')
        self.basis_vec_array = basis_vec_array
        if adjoint_basis_vec_array is None:
            self.adjoint_basis_vec_array = self.basis_vec_array
            self.symmetric = True
        else:
            self.symmetric = False
            self.adjoint_basis_vec_array = adjoint_basis_vec_array
            if self.adjoint_basis_vec_array.shape != self.basis_vec_array.shape:
                raise ValueError('Basis vec and adjoint basis vec arrays '+\
                    'are different shapes')
        self.vec_space = VectorSpaceArrays(weights=inner_product_weights)
        

    def reduce_A(self, A_on_basis_vec_array):
        """Computes the continous or discrete time reduced A matrix.
        
        Args:
            ``A_on_basis_vec_array``: 2D array with ``A*vec_array``.
                Columns are ``A`` acting on individual basis vectors.
                    
        Returns:
            ``A_reduced``: Reduced A matrix.
        """
        self.A_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_basis_vec_array, A_on_basis_vec_array)
        if not self.is_basis_orthonormal:
            self.A_reduced = self._get_proj_mat().dot(self.A_reduced)
        return self.A_reduced
        
    
    def reduce_B(self, B_on_standard_basis):
        """Computes the reduced B matrix.
        
        Args:
            ``B_on_standard_basis``: 2D array with column j ``B*e_j``.
                ``e_j`` is the jth standard basis. 
            
        Returns:
		    ``B_reduced``: Reduced B matrix.
		    
        Tip: If you found the modes via sampling initial condition responses 
        to B (and C),
        then your snapshots may be missing a factor of 1/dt
        (where dt is the first simulation time step).
        This can be remedied by multiplying ``B_reduced`` by dt.
        """
        # TODO: Check this description, then move to docstring
        #To see this dt effect, consider:
        #
        #dx/dt = Ax+Bu, approximate as (x^(k+1)-x^k)/dt = Ax^k + Bu^k.
        #Rearranging terms, x^(k+1) = (I+dt*A)x^k + dt*Bu^k.
        #The impulse response is: x^0=0, u^0=1, and u^k=0 for k>=1.
        #Thus x^1 = dt*B, x^2 = dt*(I+dt*A)*B, ...
        #and y^1 = dt*C*B, y^2 = dt*C*(I+dt*A)*B, ...
        #However, the impulse response to the true discrete-time system is
        #x^1 = B, x^2 = A_d*B, ...
        #and y^1 = CB, y^2 = CA_d*B, ...
        #(where I+dt*A ~ A_d)
        #The important thing to see is the factor of dt difference.

        self.B_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_basis_vec_array, B_on_standard_basis)
        if not self.is_basis_orthonormal:
            self.B_reduced = self._get_proj_mat().dot(self.B_reduced)
        return self.B_reduced
        

                
        
    def reduce_C(self, C_on_basis_vecs):
        """Computes the reduced C matrix.
               
        Args: 
            ``C_on_basis_vecs``: 2D array with ``C*basis_vec`` as columns.
        
        Returns:
            ``C_reduced``: Reduced C matrix.
        """
        self.C_reduced = N.array(C_on_basis_vecs, ndmin=2)
        return self.C_reduced


    def _get_proj_mat(self):
        """Gets the projection mat, i.e. inv(Psi^* Phi)."""            
        if self._proj_mat is None:
            if self.symmetric:
                IP_mat = self.vec_space.compute_symmetric_inner_product_mat(
                    self.basis_vec_array)
            else:
                IP_mat = self.vec_space.compute_inner_product_mat(
                    self.adjoint_basis_vec_array, self.basis_vec_array)
            self._proj_mat = N.linalg.inv(IP_mat)
        return self._proj_mat
        
        
    def compute_model(self, A_on_basis_vec_array, B_on_standard_basis_array,
        C_on_basis_vec_array):
        """Computes and returns the reduced matrices.
        
        Args:
            ``A_on_basis_vec_array``: 2D array with columns ``A*basis_vec``.
                ``A`` can correpond to discrete or continuous time.
                For continuous time systems, see also 
                :py:meth:`compute_derivs_arrays`.
                
            ``B_on_standard_basis_array``: 2D array with columns ``B*e_j``.
                ``e_j`` is the jth standard basis.
            
            ``C_on_basis_vec_array``: 2D array with columns ``C*basis_vec``.
        """
        self.reduce_A(A_on_basis_vec_array)
        self.reduce_B(B_on_standard_basis_array)
        self.reduce_C(C_on_basis_vec_array)
        return self.A_reduced, self.B_reduced, self.C_reduced

        
        

class LTIGalerkinProjectionHandles(LTIGalerkinProjectionBase):
    """Computes the reduced-order model matrices from modes for an LTI system.
    
    Args:
        ``inner_product``: Function to take inner product of vectors.
        
        ``basis_vec_handles``: List of basis vector handles. 
    
    Kwargs:
        ``adjoint_basis_vec_handles``: List of adjoint vector handles.
            If not given, then ``basis_vec_handles`` are used.
    
        ``is_basis_orthonormal``: Bool for bi-orthonormality of the basis vecs.
            ``True`` if the basis and adjoint vectors are bi-orthonormal.
            Default is ``False``.
            
        ``put_mat``: Function to put a matrix elsewhere (memory or file).
        
        ``verbosity``: 1 prints progress and warnings, 0 almost nothing.
        
        ``max_vecs_per_node``: Max number of vectors in memory per node.
    
    This class projects discrete or continuous time dynamics onto a set of basis
    vectors, resulting in models. Often the basis vecs are modes from 
    POD or BPOD.
    
    Usage::
        
      LTI_proj = LTIGalerkinProjectionHandles(inner_product, basis_vec_handles,
        adjoint_basis_vec_handles=adjoint_basis_vec_handles, 
        is_basis_orthonormal=True)
      A, B, C = LTI_proj.compute_model(A_on_basis_vec_handles, 
        B_on_standard_basis_handles, C_on_basis_vecs)
        
    """
    def __init__(self, inner_product, basis_vec_handles, 
        adjoint_basis_vec_handles=None, 
        is_basis_orthonormal=False, put_mat=util.save_array_text, verbosity=1, 
        max_vecs_per_node=10000):
        """Constructor"""
        LTIGalerkinProjectionBase.__init__(self, is_basis_orthonormal,
            put_mat=put_mat, verbosity=verbosity)
        self.basis_vec_handles = basis_vec_handles
        if adjoint_basis_vec_handles is None:
            self.symmetric = True
            self.adjoint_basis_vec_handles = self.basis_vec_handles
        else:
            self.symmetric = False
            self.adjoint_basis_vec_handles = adjoint_basis_vec_handles
            if len(self.adjoint_basis_vec_handles) != len(
                self.basis_vec_handles):
                raise ValueError('Number of basis vecs is not equal to the '
                    'number of adjoint basis vecs')
        self.vec_space = VectorSpaceHandles(inner_product=inner_product,
            max_vecs_per_node=max_vecs_per_node, verbosity=verbosity)
        

    def reduce_A(self, A_on_basis_vec_handles):
        """Computes and returns the continous or discrete time A matrix.
        
        Args:
            ``A_on_basis_vec_handles``: List of handles for ``A*basis_vec``.
                    
        Returns:
            ``A_reduced``: Reduced A matrix.
        """
        self.A_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_basis_vec_handles, A_on_basis_vec_handles)
        if not self.is_basis_orthonormal:
            self.A_reduced = N.dot(self._get_proj_mat(), self.A_reduced)
        return self.A_reduced
        
    
    def reduce_B(self, B_on_standard_basis_handles):
        """Computes and returns the reduced B matrix.
        
        Args:		
            ``B_on_standard_basis_handles``: List of handles for :math:`B*e_j`.
                :math:`e_j` are the standard basis elements.
        
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
        #Rearranging terms, x^(k+1) = (I+dt*A)x^k + dt*Bu^k.
        #The impulse response is: x^0=0, u^0=1, and u^k=0 for k>=1.
        #Thus x^1 = dt*B, x^2 = dt*(I+dt*A)*B, ...
        #and y^1 = dt*C*B, y^2 = dt*C*(I+dt*A)*B, ...
        #However, the impulse response to the true discrete-time system is
        #x^1 = B, x^2 = A_d*B, ...
        #and y^1 = CB, y^2 = CA_d*B, ...
        #(where I+dt*A ~ A_d)
        #The important thing to see is the factor of dt difference.
        
        self.B_reduced = self.vec_space.compute_inner_product_mat(
            self.adjoint_basis_vec_handles, B_on_standard_basis_handles)
        if not self.is_basis_orthonormal:
            self.B_reduced = N.dot(self._get_proj_mat(), self.B_reduced)
        return self.B_reduced
        

                
        
    def reduce_C(self, C_on_basis_vecs):
        """Computes and returns the reduced C matrix.
               
        Args: 
            ``C_on_basis_vecs``: list with elements of 1D arrays ``C*basis_vec``
        
        Returns:
            ``C_reduced``: Reduced C matrix.
        """
        self.C_reduced = N.array(C_on_basis_vecs, ndmin=2).T
        return self.C_reduced
    
    
    def compute_model(self, A_on_basis_vec_handles, B_on_standard_basis_handles,
        C_on_basis_vecs):
        """Computes and returns the reduced matrices.
        
        Args:
            ``A_on_basis_vec_handles``: List of handles for ``A*basis_vec``.
                ``A`` can correpond to discrete or continuous time.
                For continuous time systems, see also 
                :py:meth:`compute_derivs_handles`.
                
            ``B_on_standard_basis_handles``: List of handles for :math:`B*e_j`.
                :math:`e_j` is the jth standard basis.
            
            ``C_on_basis_vecs``: list with elements of 1D arrays 
            ``C*basis_vec``.
        """
        self.reduce_A(A_on_basis_vec_handles)
        self.reduce_B(B_on_standard_basis_handles)
        self.reduce_C(C_on_basis_vecs)
        return self.A_reduced, self.B_reduced, self.C_reduced

    
    def _get_proj_mat(self):
        """Gets the projection mat, i.e. ``inv(adjoint_vecs^* direct_vecs)``."""
        if self._proj_mat is None:
            if self.symmetric:
                IP_mat = self.vec_space.compute_symmetric_inner_product_mat(
                    self.basis_vec_handles)
            else:
                IP_mat = self.vec_space.compute_inner_product_mat(
                    self.adjoint_basis_vec_handles, self.basis_vec_handles)
            self._proj_mat = N.linalg.inv(IP_mat)
        return self._proj_mat
