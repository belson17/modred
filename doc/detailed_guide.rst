================================
Working with arbitrary data
================================

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
- Linearity: ``inner_product(a*vec1, vec2) == a*inner_product(vec1, vec2)`` 
  for a scalar ``a``.
- Implied norm: ``inner_product(vec1, vec1) >= 0``, with equality iff ``vec1 == 0``.

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
We provide a useful base class for all user-defined vectors to inherit from::
  
  import modred as MR
  class CustomVector(MR.Vector):
      pass

This isn't required, but encouraged. 
The base class defines a few useful special functions, and has some simple
error checking.
If you're curious, feel free to take a look at it in the :mod:`vectors` module
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
These are lightweight class instances that in some sense point to a vector's
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
Classes ``BPOD, POD, DMD`` (and ``VecOperations``) provide a member function 
``sanity_check`` 
that checks a few common mistakes in your vector object addition,
scalar multiplication, and inner products.
**We encourage you to run ``sanity_check`` every time you use modred.**
We used to call this the ``idiot_check`` as motivation to use it... 
keep that in mind!


-----------------------------------
Use in classes
-----------------------------------

!!!!NOT UPDATED PAST THIS POINT!!!!


The classes POD, BPOD, and DMD have very similar interfaces.
First, they all have ``compute_decomp`` and ``compute_decomp_in_memory``
functions that take as arguments ``vec_handles`` and ``vecs``, lists of vector
handles and vectors, respectively.
Within each class's ``compute_decomp`` functions, ``vec_handle.get()``
is called repeatedly. 
In fact, ``compute_decomp`` and ``compute_decomp_in_memory`` do not "know"
what's inside  they just pass its elements along to ``get_vec``.

The difference between ``compute_decomp`` and ``compute_decomp_and_return`` is
that ``compute_decomp`` doesn't return the SVD matrices. 
It calls ``put_mat`` on the SVD matrices, which by default saves them to text
files (``compute_decomp`` requires arguments that specify the destinations
for the matrices, usually path names).
Function ``compute_decomp_and_return`` simply returns these matrices::
  
  sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vec_sources)
  # Or 
  my_POD.compute_decomp(vec_sources, 'sing_vecs.txt', 'sing_vals.txt')
  
Similarly, POD, BPOD, and DMD all have functions resembling 
``compute_modes`` and ``compute_modes_and_return``.
Both call ``put_vec`` on the modes.
The difference between ``compute_modes`` and ``compute_modes_and_return`` is
that ``compute_modes`` doesn't return the modes. 
When using ``compute_modes``, ``put_vec`` instead puts the modes to some destination.
Thus, ``compute_modes`` requires a list ``vec_dests`` which contains elements
of type ``vec_dest``, which are in turn given to ``put_vec``. 
In many cases, ``vec_dests`` is a list of paths where the modes are to be saved.
When using ``compute_modes_and_return``, the ``put_vec`` function must return
a vector (a mode). The ``vec_dest`` argument to ``put_vec`` is generally left
unused in this case.

The usage difference is::

  mode_nums = range(10)
  sing_vecs, sing_vals = my_POD.compute_modes_and_return(mode_nums)
  # Or
  vec_dests = ['mode%02d.txt'%i for i in mode_nums]
  my_POD.compute_modes(mode_nums, vec_dests)
  
Here's a simple outline of what the ``get_vec`` and ``put_vec`` functions
do for these two cases::

  # For doing everything in memory with "compute_modes_and_return".
  def get_vec(vec_object):
      return vec_object
  def put_vec(vec_object, dummy_dest):
      return vec_object
      
  # For saving/loading with "compute_modes".
  def get_vec(vec_path):
      return my_load(vec_path)
  def put_vec(vec_object, vec_path):
      my_save(vec_object, vec_path)

This can all come off as a bit abstract; the following use-cases are helpful 
(also see the Quickstart).


-------------------------------
Examples of vector functions
-------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Loading and saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a good choice when your data is large or comes from some an independent
simulation.
The ``get_vec`` function simply takes a path as its ``vec_source`` argument,
loads the data from that path, and returns a vector object. 
Similarly, ``put_vec`` saves the vector to the given path (as argument
``vec_dest``). 
For parellelization, using files to store vectors is **strongly recommended**
for efficiency. 

Here we reproduce a brief example that's provided in the ``vecdef`` module 
(as ``ArrayTextUniform``)::

  import modred as MR
  class ArrayText(object):
      def get_vec(self, path):
          vec = MR.load_mat_text(path)
          return vec
      def put_vec(self, vec, path):
          MR.save_mat_text(vec, path)
      def inner_product(self, vec1, vec2):
          return N.vdot(vec1, vec2)

Then we use this class with the following::
          
  num_vecs = 30
  vec_paths = ['vec%02d.txt'%i for i in range(num_vecs)]
  my_DMD = MR.DMD(ArrayText())
  ritz_vals, mode_norms, build_coeffs = \
      my_DMD.compute_decomp_and_return(vec_paths)
  num_modes = 10
  mode_paths = ['dmd_mode%02d.txt'%i for i in range(num_modes)]
  my_DMD.compute_modes(range(num_modes), mode_paths)
     

The vectors are saved/loaded to/from text files, and the vectors are numpy 
arrays. 
The elements of the list given to ``compute_decomp_and_return`` are given
to ``get_vec``. This is true in general. 
A case is shown later that doesn't use arrays.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Returning, in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This case was summarized in the quickstart, and now you can see how the vector
definition class (shown below) enables the usage. 
(This class is supplied as ``vecdefs.ArrayInMemoryUniform``, with only slight 
differences.)::

  import numpy as N
  import modred as MR
  
  class ArrayInMemory():
      def get_vec(self, vec):
          return vec
      def put_vec(self, vec, dummy_dest):
          return vec
      def inner_product(self, vec1, vec2):
          return N.vdot(vec1, vec2)
  
  num_vecs = 30
  my_POD = MR.POD(ArrayInMemory())
  num_modes = 10
  sing_vecs, sing_vals = my_POD.compute_decomp_and_return(
      [N.random.random(num_modes) for i in range(num_vecs)])
  modes = my_POD.compute_modes_and_return(range(num_modes))          

This case is a bit special/degenerate; ``get_vec`` just returns its argument, and 
``put_vec`` returns its first argument while ignoring its second!
In the previous save/load example, ``get_vec`` loaded from the ``vec_source``
argument.
In this example, the ``vec_source`` argument is the vector object itself, so it is simply
returned.
In the previous save/load example, ``put_vec`` saved to the ``vec_dest``
argument and returned nothing.
In this example, ``put_vec`` simply returns the vector object, and doesn't use
the ``vec_dest`` argument.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
User-defined vector object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your data is more complicated, don't use the simple stuff in ``vecdefs``. 
Instead, write your own vector object, e.g. ``VecObject``::

  import modred as MR
  class VecObject(object):
      def load(self, path):
          # Load data from disk in any format
          pass
      def save(self, path):
          # Save data to disk in any format
          pass
      def inner_product(self, other_vec):
          # Take inner product of self with other_vec
          pass
      def __add__(self, other):
          # Return a new object that is the sum of self and other
          pass
      def __mul__(self, scalar):
          # Return a new object that is "self * scalar"
          pass
  
  
  class VecDefs(object):
      def get_vec(path):
          vec = VecObject()
          vec.load(path)
          return vec
      def put_vec(vec, path):
          vec.save(path)
      def inner_product(vec1, vec2):
          return vec1.inner_product(vec2)
  
  my_DMD = MR.DMD(VecDefs())
  # Generate vectors and save them to vec_paths.
  my_DMD.compute_decomp(vec_paths, 'ritz_vals.txt', 'mode_norms.txt', 
      'build_coeffs.txt')
  mode_nums = [1, 4, 0, 2, 10]
  mode_paths = ['mode%02d'%i for i in mode_nums]
  my_DMD.compute_modes(mode_nums, mode_paths)
  
  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data class, in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are the beginnings of another way to bypass loading and saving by using
a ``DataClass``::
  
  class DataClass(object):
      def make_data(self, stuff):
          # Create the vecs to decompose into modes.
          pass
          
  class VecDefsDataClass(object):
      @staticmethod
      def get_vec(my_data_class_and_vec_attr):
          my_data_class = my_data_class_and_attr[0]
          attr = my_data_class_and_attr[1]
          return getattr(my_data_class, attr)
          
      @staticmethod
      def put_vec(vec, my_data_class_and_vec_attr):
          my_data_class = my_data_class_and_attr[0]
          attr = my_data_class_and_attr[1]
          setattr(my_data_class, attr, vec)
      
      @staticmethod
      def inner_product(vec1, vec2):
          # Some inner product
          pass

(The use of static methods isn't necessary, but is often appropriate.)

There are of course many other choices, these are just a 
few examples to help your understanding and inspire your own choices.



---------------------------------------
Summary and next steps
---------------------------------------

Summarizing, define

1. A vector object that has:
  1. vector addition ("+", ``__add__``)
  2. scalar multiplication ("*", ``__mul__``)
2. A vector defintion class or module that has:
  1. ``get_vec`` function
  2. ``put_vec`` function
  3. ``inner_product`` function

Then you can get started using any of the modal decomposition classes 
(POD, BPOD, and DMD)!
See the examples directory for more examples of how everything works 
together. 
The rest of this documentation details how to use each individual class and
method.


There has been essentially no discussion of ERA and OKID.
The documentation for the individual classes and functions should be sufficient.
