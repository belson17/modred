.. _sec_details:

==================================================
Interfacing with your data
==================================================

The simplest way to use modred is with the Matlab-like functions and 2D arrays.
However, sometimes your data is too large for this.
In these cases, there is a high-level object-oriented interface that works with
data in any format and never needs the data stacked into a 2D array.
Of course, you'll need to tell modred how to interact with your data.
This section explains how to do this and provides some mathematical background.

-------------------
Vector objects
-------------------

The building block of the modal decompositions is the vector object.
Sets of these vector objects are decomposed into modes by POD, BPOD, and DMD.
Others call these vector objects snapshots, planes of spatial data, fields, time
histories, and many other names.
Within modred, "vector" refers to the object you, the user, use to represent
your data.  
**By "vector", we do not mean a 1D array**. 
We do mean an element of a vector space (technically an inner product space).
You are free to choose *any* object, from numpy arrays to your own class, so
long as it satisfies a few simple requirements.

The vector object must:

1. Support scalar multiplication, i.e. ``vector2 = 2.0*vector1``. 
2. Support addition with other vectors, i.e. ``vector3 = vector1 + vector2``.
3. Be compatible with a user-supplied ``inner_product(vector1, vector2)`` 
   function.

Numpy arrays already meet requirements 1 and 2. 
For your own classes, define the special methods ``__mul__`` and ``__add__`` for
1 and 2.

You also need an inner product function that takes two vectors and returns a 
single number. 
This number can be real or complex, but may not switch from real to complex 
depending on the input, i.e., it must be real for all inputs or complex for all
inputs.
Your inner product must satisfy the mathematical definition for an inner
product:

- Conjugate symmetry: 
  ``inner_product(vec1, vec2) == numpy.conj(inner_product(vec2, vec1))``.
- Linearity: 
  ``inner_product(vec1, scalar*vec2) == scalar*inner_product(vec1, vec2)``.
- Implied norm: ``inner_product(vec, vec) >= 0`` with equality if and only if
  ``vec`` is the zero vector.

The two examples we show are numpy's ``vdot`` and the trapezoidal rule in
:py:class:`vectors.InnerProductTrapz`.
It's often a good idea to define an inner product function as a member function
of the vector class, and write a simple wrapper. 
There is an example of this in the tutorial.

The resulting modes are also vectors.
We mean this in both the programming sense that modes are vector objects and the
mathematical sense that modes live in the same vector space as vectors.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a useful base class for all user-defined vectors to inherit from,
``mr.Vector``.
It isn't required to inherit from it, but encouraged because it defines a few
useful special functions and has some error checking.
If you're curious, take a look at it in the :mod:`vectors` module (click on the
[source] link on the right side).

----------------------------
Vector handles
----------------------------

When the vectors are large, it can be inefficient or impossible to have all of
them in memory simultaneously.
Thus, modred only needs a subset of vectors in memory, loading and saving them
as necessary.
Therefore, you can provide it with a list of *vector handles*. 
These are *lightweight* objects that in some sense point to a vector's location,
like the filename where it's saved.
In general, vector handles get a vector from a location and return it, and also
put a vector in a location.
That is, they implement this interface:

 - Constructor with interface ``VectorHandle(location)``.
 - A get function with interface ``vec = vec_handle.get()``.
 - A put function with interface ``vec_handle.put(vec)``.

An example would be a constructor that takes a file name as an argument, a
``get`` that loads and returns the vector, and a ``put`` that saves the vector
to the file name.

One can think of ``get`` as loading, but it is more general because ``get`` can
retrieve the vector from anywhere (though most often from file).
Similarly, one can think of ``put`` as saving, but it is more general because 
``put`` can send the vector anywhere (though most often to file).

It's natural to think of a vector handle's ``get`` and ``put`` as inverses, but
they don't have to be.
For example, it's acceptable to load an input vector from one file format and
save modes to another file format.
However, it does mean that if one wanted to load the modes, one couldn't with
this vector handle because ``get`` assumes a different file format.

Another way to handle the case of different input vector and mode (or any output
vector) file formats is to define a different vector handle class for each.
In this case, technically one wouldn't need a ``put`` for the input vector
handle since one never saves to this format.
Similarly, one only needs to write a ``get`` for the mode vector handle if one
wants to load the modes (for example to plot them).

It's very important that the vector handles actually be lightweight (use little
memory). 
modred is most efficient when it uses all of the memory available to have as
many vectors in memory as possible.
So if vector handles contain vectors or other large data, then modred could run
slowly or stop with "out of memory" errors.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide a useful base class for all user-defined vector handles to inherit
from.
An example of a user-defined vector handle that inherits from ``mr.VecHandle``
is provided in the tutorial.
This isn't required, but strongly encouraged because it contains extra
functionality.
The ``mr.VecHandle`` constructor accepts two additional arguments, a base vector
handle ``base_handle`` and a scaling factor ``scale``. 
This allows the ``get`` function to retrieve a vector, subtract from it a base
vector (for example an equilibrium or mean state), scale it (for example by a
quadrature weight), and return the modified vector.
The base class achieves this via a ``get`` that calls the derived class's member
function ``_get`` and performs the additional operations for base vectors and/or
scaling.
The base class's ``put`` simply calls ``_put`` of the derived class.
Examples are shown in the tutorial.

One might be concerned that the base class is reloading the base vector at each
call of ``get``, but this is avoidable. 
As long as the ``base_handle`` you give each vector handle instance is equal
(with respect to ``==``), then the base vector is loaded on the first call of
``get`` and stored as ``mr.VecHandle.cached_base_vec``, which is used by all
instances of classes derived from ``mr.VecHandle``. 

If you're curious, feel free to take a look at it in the :mod:`vectors` module
(click on the [source] link on the right side).

--------------------------------------------------------
Checking requirements automatically
--------------------------------------------------------

First, we encourage you to write your own tests (see module ``unittest``) to be
sure your vector object and vector handle work as you expect.
Many classes provide a function ``sanity_check`` that checks a few common
mistakes in your vector object addition, scalar multiplication, and inner
products.
*We encourage you to run* ``sanity_check`` *every time you use modred.*
We used to call this the ``idiot_check`` as motivation to use it; keep that in
mind!

--------------------------------------------------
How vector objects and handles are used in modred
--------------------------------------------------

The classes ``POD``, ``BPOD``, and ``DMD`` have similar interfaces which
interact with vectors and vector handles.
First, each has ``compute_decomp`` functions that take lists of vector handles,
``vec_handles``, as arguments.
Within the ``compute_decomp`` functions, ``vec = vec_handle.get()`` is called
repeatedly to retrieve vectors as needed. 
In fact, ``compute_decomp`` does not "know" or "care" what's inside the vector
handles and vectors; only that they satisfy the requirements.

More information about these methods is provided in the documentation for each
class.

----------
Example
----------

An example of a custom class for vectors and vector handles is shown below:

.. literalinclude:: ../modred/examples/customvector.py

For an example using this class, see the tutorial in :ref:`sec_modaldecomp`.

---------------------------------------
Summary and next steps
---------------------------------------

Summarizing, to use modred on arbitrary data, define

1. A vector object that has:

   1. Vector addition ("+", ``__add__``),

   2. Scalar multiplication ("*", ``__mul__``),

   3. Optional: inherits from :py:class:`vectors.Vector`.


2. A function ``inner_product(vec1, vec2)``.


3. A vector handle class that has:

   1. Member function ``get()`` which returns a vector handle.
   2. Member function ``put(vec)`` where ``vec`` is a vector handle.
   3. Optionally inherits from :py:class:`vectors.VecHandle`. If so, 
      member function names in 1 and 2 change to ``_get`` and ``_put``.

Then you can get started using any of the modal decomposition classes!
Before writing your own classes, check out :py:mod:`vectors`, which has several
common vector and vector handles classes.

For large data, Python's speed limitations can be bypassed by implementing
functions in compiled languages such as C/C++ and Fortran and accessing them
within python with Cython, SWIG, f2py, etc. 
