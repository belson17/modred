.. _sec_model_reduction:

-------------------------------------------------
Model reduction
-------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Linear reduced-order models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a class to find reduced-order models (continuous and discrete time)
by projecting linear dynamics onto modes (Galerkin projection).
The governing equation of the full system is assumed to be either continuous
time

.. math::

  \partial x(t)/ \partial t &= A x(t) + B u(t) \\
  y(t) &= C x(t) 

or discrete time 

.. math::

  x(k+1) &= A x(k) + B u(k) \\
  y(k) &= C x(k) 

where :math:`k` is the time step.
Here :math:`x` is the state, and is a vector. 
:math:`A`, :math:`B`, and :math:`C` are, in general, linear operators (often matrices).
:math:`A` acts on :math:`x` and returns a vector that lives in the same vector space as :math:`x`.
:math:`B` acts on elements of the input space (:math:`R^p`, where :math:`p` is the number of inputs) 
and returns elements of the vector space in which :math:`x` lives. 
:math:`C` acts on :math:`x` and returns elements of the output space, :math:`\mathcal{R}^q`, where :math:`q`
is the number of outputs.

Projecting this equation onto the BPOD modes gives, loosely speaking:
 
  ``Ar[i, j] = inner_product(adjoint_modes[i], A*direct_modes[j])``
  
  ``Br[i, j] = inner_product(adjoint_modes[i], B*e_j)``, where ``e_j`` is the jth standard
  basis of :math:`\mathcal{R}^p`.
  
  ``Cr[:, j] = C*direct_modes[j]``

For a precise description, see Rowley 2005, International Journal on Bifurcation
and Chaos.
For discrete time systems, ``A*direct_modes`` are the modes advanced one time 
step, and the resulting model is in discrete time.
For continuous time systems, ``A*direct_modes`` are the time-derivatives of the
modes and the resulting model is in continuous time.

The A, B, and C operators may or may not be available within python.
For example, if you have a large solver written in another language then 
it might be hard to access A, B, and C from python. 
Modred *only* requires the action of A, B, and C on the appropriate vectors,
not the operators.


Here's an example that uses matrix representations of the linear operators.

.. literalinclude:: ../examples/rom_ex1.py

The list ``basis_vecs`` contains the vectors that define the basis onto 
which the dynamics are projected.
This example works in parallel with no modifications.


Here's another example, this time using vector handles for large data.

.. literalinclude:: ../examples/rom_ex2.py

This example works in parallel with no modifications.

If you do not have direct access to the time-derivatives of the direct modes
but want a continuous time model, see
:py:func:`ltigalerkinproj.compute_derivs_arrays` and 
:py:func:`ltigalerkinproj.compute_derivs_handles`.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Eigensystem Realization Algorithm (ERA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See documentation and examples provided in :py:func:`era.compute_ERA_model` 
and :py:class:`era.ERA`.
