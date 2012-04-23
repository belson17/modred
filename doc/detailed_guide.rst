.. _sec_details:

==================================================
Interfacing with your data
==================================================

As stated before, modred can work with any data in any format.
Of course, you'll need to tell modred how to do this.
This section fully explains the process and the relevant mathematical background.

-------------------
Vector objects
-------------------

The building block of the modal decompositions is the vector object.
Sets of these vector objects are modally decomposed by POD, BPOD, and DMD.
Others call these snapshots, planes of spatial data, fields, time histories,
and many other names.
Within modred, "vector" refers to the object you, the user, use to represent your data.
You are free to choose *any* object, from numpy arrays to your own class, so long as
it satisfies a few simple requirements.
**By "vector", we do not mean a 1D array**. 
We do mean an element of a vector space (technically a Hilbert space).

The requirements of the vector object are that it must:

1. Support scalar multiplication, i.e. ``vector2 = 2.0*vector1``. 
2. Support addition with other vectors, i.e. ``vector3 = vector1 + vector2``.
3. Be compatible with the ``inner_product(vector1, vector2)`` function.

Numpy arrays already meet requirements 1 and 2. 
For your own classes, define ``__mul__`` and ``__add__`` special methods for 1 and 2.

You also need an inner product function that takes two vectors returns a single number.
This number can be real or complex, but must always be the same type.
Your inner product must satisfy the mathematical definition for an inner product:

- Conjugate symmetry: 
  ``inner_product(vec1, vec2) == numpy.conj(inner_product(vec2, vec1))``.
- Linearity: ``inner_product(scalar*vec1, vec2) == scalar*inner_product(vec1, vec2)``.
- Implied norm: ``inner_product(vec, vec) >= 0`` with equality if and only if
  ``vec == 0``.

The two examples we show are numpy's ``vdot`` and the trapezoidal rule in
``vectors.InnerProductTrapz``.
It's often a good idea to define an inner product function as a member 
function of the vector class, and write a simple wrapper. 
There is an example of this in the quickstart.

The resulting modes are also vectors.
We mean this in both the programming sense that modes are also vector objects
and the mathematical sense that modes live in the same vector space as vectors.



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a useful base class for all user-defined vectors to inherit from,
``MR.Vector``.
It isn't required to make your vector class inherit from it, but we encourage
you to because it defines a few useful special functions and has some
error checking.
If you're curious, take a look at it in the :mod:`vectors` module
(click on the [source] link on the right side).

----------------------------
Vector handles
----------------------------

When the vectors are large, it can be inefficient or impossible to have all 
of them in memory simultaneously.
Instead, modred is designed to have only a subset of vectors in memory, loading
and saving them as necessary.
Therefore, rather than providing modred with a list of vectors, you can 
provide it with a list of "vector handles". 
These are **lightweight** class instances that in some sense point to a vector's
location, like the filename of where it's saved.
In general, vector handles must be able to get a vector from some location and
return it, and also take a vector and put it to some location.
That is, they generally must have these two member functions:

 - A get function with interface ``vec = vec_handle.get()``.
 - A put function with interface ``vec_handle.put(vec)``.

A simple vector handle would have a constructor that takes a path name as
an argument, a ``get`` that loads and returns the vector, and a ``put``
that saves the vector to the path name.

One can think of ``get`` as loading, but more general because ``get`` can
retrieve the vector from anywhere (though most often from file).
Similarly, one can think of ``put`` as saving, but more general because ``put``
can send the vector anywhere (though most often to file).

It's natural to think of a vector handle's ``get`` and ``put`` as
"inverses", but they don't have to be.
For example, it's acceptable to load an input vector from one file format
and save modes to another file format.
However, it does mean that one can't load the modes with this vector handle 
since ``get`` assumes a different file format.

Another way to handle the case of different input vector and mode (or any output
vector) file formats is to define a different vector handle class for each.
In this case, one wouldn't need a ``put`` for the input vector handle
since one never saves to this format.
Similarly, one only needs to write a ``get`` for the mode vector handle if 
one wants to load the modes (for example to plot them).

It's very important that the vector handles actually be lightweight (use
little memory). 
Modred is most efficient when it uses all of the memory available to have
as many vectors in memory as possible.
So if vector handles contain vectors or other large data, then modred 
could run slowly or Python could give otherwise inexplicable out of memory
errors.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We provide a useful base class for all user-defined vector handles
to inherit from.
An example of a user-defined vector handle that inherits from ``MR.VecHandle``
is provided in the quickstart.
This isn't required, but strongly encouraged because it contains extra
functionality.
The ``MR.VecHandle`` constructor accepts two additional arguments, a 
base vector handle ``base_handle`` and a scaling ``scale``. 
This allows the ``get`` function to retrieve a vector, subtract a base vector
(for example an equilibrium or mean), scale (for example by a quadrature weight),
and return a modified vector.
The base class achieves this via a ``get`` that calls the derived
class's member function ``_get`` and performs the additional operations
for base vectors and/or scaling (or neither, if neither ``base_handle`` or ``scale``
are given).
The base class's ``put`` simply calls ``_put`` of the derived class.
Examples are shown in the quickstart.

One might be concerned that the base class is reloading the base vector
at each call of ``get``, but this is avoidable. 
As long as the ``base_handle`` you give each vector handle instance is equal
(with respect to ``==``), then the base vector is loaded on the first 
call of ``get`` and stored as ``MR.VecHandle.cached_base_vec``, which is used
by all instances of classes derived from ``MR.VecHandle``. 

If you're curious, feel free to take a look at it in the :mod:`vectors` module
(click on the [source] link on the right side).


--------------------------------------------------------
Checking requirements automatically
--------------------------------------------------------

First off, we encourage you to write your own tests (see module unittest) to
be sure
your vector object and vector handle work as you expect.
Classes ``BPOD, POD, DMD`` (and ``VectorSpace``) provide a member function 
``sanity_check`` 
that checks a few common mistakes in your vector object addition,
scalar multiplication, and inner products.
**We encourage you to run** ``sanity_check`` **every time you use modred.**
We used to call this the ``idiot_check`` as motivation to use it... 
keep that in mind!


--------------------------------------------------
Vector objects and handles in classes
--------------------------------------------------

The classes POD, BPOD, and DMD have similar interfaces which interact
with vectors and vector handles.
First, each has ``compute_decomp`` 
functions that take lists of vector handles, ``vec_handles``, as arguments.
Within the ``compute_decomp`` functions, ``vec = vec_handle.get()``
is called repeatedly to retrieve vectors as needed. 
They also have member functions ``compute_decomp_in_memory`` that directly take
lists of vectors as arguments since the use of vector handles is
somewhat unnecessary.
In fact, ``compute_decomp`` and ``compute_decomp_in_memory`` do not "know"
or "care" what's inside the vector handles and vectors; only
that they satisfy the requirements.

Similarly, POD, BPOD, and DMD all have member functions resembling 
``compute_modes`` and ``compute_modes_in_memory``.
Function ``compute_modes`` takes a list of vector handles for the modes and
calls ``put(mode)``, and returns nothing.
Function ``compute_modes_in_memory`` returns a list of modes directly, which
is a simple option for small data.

More information about these methods is provided in the documentation
for each class.

----------
Example
----------

An example of a custom class for vectors and vector handles is shown
below:

.. literalinclude:: ../examples/custom_vector.py

For an example using this class, see the tutorial in :ref:`sec_modaldecomp`.

---------------------------------------
Summary and next steps
---------------------------------------

Summarizing, to use modred on arbitrary data, define

1. A vector object that has:
  1. Vector addition ("+", ``__add__``)
  2. Scalar multiplication ("*", ``__mul__``)
  3. Inherits from ``MR.Vector`` (recommended but not required)
2. A vector handle class that has:
  1. Member function ``get()``
  2. Member function ``put(vec)``
  3. Inherits from ``MR.VecHandle``, if so, requirements 1 and 2 change to:
    1. Member function ``_get()``
    2. Member function ``_put(vec)``
3. Function ``inner_product(vec1, vec2)``

Then you can get started using any of the modal decomposition classes 
(``POD``, ``BPOD``, ``DMD``, ``VectorSpace``, and ``BPODROM``)!

For large data, Python's speed limitations can be 
bypassed by implementing functions in compiled languages such as C/C++ and 
Fortran and accessing them within python with Cython, SWIG, f2py, etc. 
