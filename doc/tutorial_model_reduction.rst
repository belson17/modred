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
In this sense, we consider the columns of B and the rows of C as vectors.
Then, y is the inner product of rows of C with x.

Projecting this equation onto the BPOD modes gives, very loosely speaking::
 
  Ar = inner_product(adjoint_modes, A*direct_modes)
  Br = inner_product(adjoint_modes, B)
  Cr = inner_product(C, direct_modes)

For a precise description, see Rowley 2005, International Journal on Bifurcation
and Chaos.
The important thing to notice is ``Ar`` requires ``A`` to operate on the 
direct modes. 
This necessarily must be done outside of modred, since modred
doesn't have access to the full ``A`` matrix (or solver that approximates
``A``.)
For discrete time systems, ``A*direct_modes`` is the modes advanced one time 
step, and the resulting model is in discrete time.
For continuous time systems, ``A*direct_modes`` is the time-derivatives of the
modes and the resulting model is in continuous time.

Here's an example::

  # Given these lists of numpy arrays:
  # direct_modes
  # adjoint_modes
  # A_times_direct_modes
  # B_vecs
  # C_vecs
  
  import modred as MR
  my_BPODROM = MR.BPODROM(N.vdot)
  A = my_BPODROM.compute_A_in_memory(A_times_direct_modes, adjoint_modes)
  B = my_BPODROM.compute_B_in_memory(B_vecs, adjoint_modes)
  C = my_BPODROM.compute_C_in_memory(C_vecs, direct_modes)
  my_BPODROM.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')
  
The list ``A_times_direct_modes`` are modes that have been operated on by the
full matrix A, which can be either discrete or continuous time.
The list ``B_vecs`` contains the vectors that comprise the columns
of the full system's B matrix.
Similarly, the list ``C_vecs`` contains the vectors that compromise the rows
of the full system's C matrix.
This example works in parallel with no modifications.

Here's an example that uses vector handles for large data. It also
uses the function ``compute_model`` to find the A, B, and C matrices
all at once::

  # Given these lists of numpy arrays:
  # direct_mode_handles
  # adjoint_mode_handles
  # A_times_direct_mode_handles
  # B_vec_handles
  # C_vec_handles
  
  import modred as MR
  my_BPODROM = MR.BPODROM(N.vdot)
  A, B, C = my_BPODROM.compute_model(A_times_direct_mode_handles, B_vec_handles,
    C_vec_handles, direct_mode_handles, adjoint_mode_handles)
  my_BPODROM.put_model('A_reduced.txt', 'B_reduced.txt', 'C_reduced.txt')

This example works in parallel with no modifications.
Note also that if you don't need the A, B, and C matrices, you can simply
omit them as return arguments and still "put" them with ``put_model``.

If you do not have direct access to the time-derivatives of the direct modes
but want a continuous time model, modred provides a first-order time-derivative
operation which takes the modes and the modes at a later time ``dt``,
and approximates the derivative with ``(mode(t=dt) - mode(t=0))/dt``.
For small ``dt``, this is often satisfactory.

Here is an example::

  # Given these lists:
  # direct_modes
  # A_times_direct_modes
  import modred as MR
  deriv_modes = MR.compute_derivs_in_memory(A_times_direct_modes,
      direct_modes)

Where ``A_times_direct_modes`` are assumed to be the modes advanced ``dt``
in time.
As usual, we also provide ``compute_derivs`` which takes handles as arguments
instead for large vectors.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Eigensystem Realization Algorithm (ERA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The documentation and examples provided in :py:func:`era.compute_ERA_model` 
and :py:class:`era.ERA` should be sufficient.
