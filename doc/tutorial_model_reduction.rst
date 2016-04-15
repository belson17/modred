.. _sec_model_reduction:

-------------------------------------------------
Model reduction
-------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Linear reduced-order models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a class to find reduced-order models (continuous and discrete time)
by projecting linear dynamics onto modes (Petrov-Galerkin projection).
The governing equation of the full system is assumed to be either continuous
time:

.. math::

  \partial x(t)/ \partial t &= A x(t) + B u(t) 
  \\
  y(t) &= C x(t) 

or discrete time:

.. math::

  x(k+1) &= A x(k) + B u(k) 
  \\
  y(k) &= C x(k) 
  
where :math:`k` is the time step.
Here :math:`x` is the state vector. 
:math:`A`, :math:`B`, and :math:`C` are, in general, linear operators (often
matrices).  
In cases where there are no inputs and outputs, :math:`B` and :math:`C` are
zero.
:math:`A` acts on :math:`x` and returns a vector that lives in the same vector 
space as :math:`x`.
:math:`B` acts on elements of the input space, :math:`\mathbb{R}^p`, where :math:`p` is
the number of inputs and returns elements of the vector space in which :math:`x`
lives. 
:math:`C` acts on :math:`x` and returns elements of the output space,
:math:`\mathbb{R}^q`, where :math:`q` is the number of outputs.

These dynamical equations can be projected onto a set of modes.
First, approximate the state vector as a linear combination of :math:`r` modes,
stacked as columns of matrix :math:`\Phi`, and time-varying coefficients
:math:`q(k)`:

.. math::

  x(k) \approx \Phi q(k) .

Then substitute into the governing equations and take the inner product with a
set of adjoint modes, columns of matrix :math:`\Psi`.
The result is a reduced system for :math:`q`, which has as many elements as
:math:`\Phi` has columns, :math:`r`.
The adjoint, :math:`(\,\,)^+`, is defined with respect to inner product weight
:math:`W`.

.. math::

  q(k+1) &= A_r q(k) + B_r u(k) \\
  y(k) &= C_r q(k) \\
  \text{where} \\
  A_r &= (\Psi^+ \Phi)^{-1} \Psi^+ A \Phi \\
  B_r &= (\Psi^+ \Phi)^{-1} \Psi^+ B \\
  C_r &= C \Phi

An analagous result exists for continuous time.  

If the modes are not stacked into matrices, then the following equations are
used, where :math:`[\,\,]_{i,j}` denotes row :math:`i` and column :math:`j`.

.. math::

  [\Psi^+  \Phi]_{i,j} &= \langle\psi_i,\, \phi_j \rangle_W \\
  [\Psi^+ A \Phi]_{i,j} &= \langle\psi_i,\, A \phi_j\rangle_W \\
  [\Psi^+ B] &= \langle \psi_i,\, B e_j\rangle_W \\
  [C \Phi]_{:,j} &= C \phi_j

:math:`e_j` is the jth standard basis (intuitively :math:`B e_j` is
:math:`[B]_{:,j}` if :math:`B` is a matrix.).

The :math:`A`, :math:`B`, and :math:`C` operators may or may not be available
within Python.
For example, you may do simulations using code written in another language. 
For this reason, modred requires only the *action* of the operators on the
vectors, i.e., the products :math:`A \phi_j`, :math:`Be_j`, and :math:`C
\phi_j`, and *not* the operators :math:`A`, :math:`B`, and :math:`C` themselves.

*****************************************
Example 1: Smaller data and matrices
*****************************************
Here's an example that uses matrices.

.. literalinclude:: ../modred/examples/rom_ex1.py

The array ``basis_vecs`` contains the vectors that define the basis onto which
the dynamics are projected.

A few variations of this are illustrative. 
First, if no inputs or outputs exist, then there is only :math:`A_r` and no
:math:`B_r` or :math:`C_r`. 
The last two lines would then be replaced with::

  A_reduced = LTI_proj.reduce_A(A.dot(basis_vec_array))
  LTI_proj.put_A_reduced('A_reduced.txt')

Another variant is if the basis vecs are known to be orthonormal, as is always
the case with POD modes.
Then, the :math:`\Psi^* W \Phi` matrix and its inverse are identity, and
computing it is wasteful.
Specifying the constructor keyword argument ``is_basis_orthonormal=True`` tells
modred this matrix is identity and to not compute it.


**********************************************
Example 2: Larger data and vector handles
**********************************************
Here's an example similar to what might arise when doing large simulations in
another language or program.

.. literalinclude:: ../modred/examples/rom_ex2.py

This example works in parallel with no modifications.

If you do not have the time-derivatives of the direct modes but want a
continuous time model, see
:py:func:`ltigalerkinproj.compute_derivs_arrays` and 
:py:func:`ltigalerkinproj.compute_derivs_handles`.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Eigensystem Realization Algorithm (ERA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See the documentation and examples provided in :py:func:`era.compute_ERA_model` 
and :py:class:`era.ERA`.
