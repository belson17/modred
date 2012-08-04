.. _sec_model_reduction:
-------------------------------------------------
Model reduction
-------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BPOD linear reduced-order models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a class to find BPOD linear models (continuous and discrete time)
from BPOD modes.
The governing equation of the full system is assumed to be either continuous
time

.. math::

  \partial x(t)/ \partial t &= A x(t) + B u(t) \\
  y(t) &= C x(t) 

where t >= 0 , or discrete time 

.. math::

  x(k+1) &= A x(k) + B u(k) \\
  y(k) &= C x(k) 

where k = 0,1,2,...
Here x is the state, and is a vector. 
A, B, and C are, in general, linear operators (often matrices).
A acts on x and returns a vector that lives in the same vector space as x.
B acts on elements of the input space (R^p, where p is the number of inputs) 
and returns elements of the vector space in which x lives. 
C acts on x and returns elements of the output space, R^q, where q is the 
number of outputs.

Projecting this equation onto the BPOD modes gives, loosely speaking:
 
  Ar = inner_product(adjoint_modes, A*direct_modes)
  
  Br[:,j] = inner_product(adjoint_modes, B*e_j), where e_j is the jth standard
  basis of R^p.
  
  Cr = C*direct_modes

For a precise description, see Rowley 2005, International Journal on Bifurcation
and Chaos.
For discrete time systems, ``A*direct_modes`` are the modes advanced one time 
step, and the resulting model is in discrete time.
For continuous time systems, ``A*direct_modes`` are the time-derivatives of the
modes and the resulting model is in continuous time.

The A, B, and C operators may or may not be available within python.
For example, if you have a large solver written in another language then 
it might be hard to access A, B, and C from python, even with tools like 
Cython, SWIG, and f2py. 
Therefore, modred lets you compute the action of A, B, and C either inside of 
python or externally.


Here's an example that uses matrix representations of the linear operators.

.. literalinclude:: ../examples/rom_ex1.py

The three objects ``A_op``, ``B_op``, and ``C_op`` are callable and
perform matrix multiplication (:py:class:`ltigalerkinproj.MatrixOperator`).
The ``LTI_proj`` instance uses these to operate on the appropriate vectors,
including ``basis_vecs``, and computes the reduced matrices. 

The list ``basis_vecs`` contains the vectors that define the basis onto 
which the dynamics are projected.
This example works in parallel with no modifications.


Here's another example, this time using vector handles for large data.

.. literalinclude:: ../examples/rom_ex2.py

This example works in parallel with no modifications.

If you do not have direct access to the time-derivatives of the direct modes
but want a continuous time model, modred provides 
:py:func:`ltigalerkinproj.compute_derivs` and 
:py:func:`ltigalerkinproj.compute_derivs_in_memory`.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Eigensystem Realization Algorithm (ERA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The documentation and examples provided in :py:func:`era.compute_ERA_model` 
and :py:class:`era.ERA` should be sufficient.
