================================
Working with arbitrary data
================================

As stated before, modred can work with any data in any format.
Of course, you'll need to tell modred how to do this.
This section explains the steps, along with the relevant mathematical background.

-------------------
The vector object
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
3. Be compatible with supplied ``get_vec`` function.
4. Be compatible with supplied ``put_vec`` function.
5. Be compatible with supplied ``inner_product`` function.

Numpy arrays already meet requirements 1 and 2. 
For your own classes, define ``__mul__`` and ``__add__`` special methods for 1 and 2.

Requirements 3--5 are discussed next.

----------------------------
Functions of vectors
----------------------------


^^^^^^^^^^^^^^^^^^
Definitions
^^^^^^^^^^^^^^^^^^

The modal decomposition classes (POD, BPOD, and DMD) interact with vectors
with three functions which you supply:

1. ``get_vec(vec_source)``: Gets a vector specified by ``vec_source`` and returns it.
2. ``put_vec(vec, vec_dest)``: Puts ``vec`` in the destination specified by ``vec_dest``.
3. ``inner_product(vec1, vec2)``: Returns the inner product between two vectors.

The first, ``get_vec``, gets a vector from the source specified by its argument
(which can be anything), 
possibly does some operations to that vector, then returns that vector for use in modred.
It is like loading, but more general. 
The ``vec_source`` can be a file name, a set of indices for accessing data in an array,
a tuple with several entries, anything.
Since you choose ``get_vec`` and the arguments it will take, you just have to be consistent.

Similarly, ``put_vec`` takes a vector, possibly does some operations on that vector, 
then puts that vector into the destination pointed to by its second argument.
It is like saving, but more general.
Just like ``vec_source``, ``vec_dest`` can be anything, you just have to be consistent.

You also need an inner product function that takes two vectors returns a single number.
This number can be real or complex, but must always be the same type.
Your inner product must satisfy the mathematical definition for an inner product:

- Conjugate symmetry: 
  ``inner_product(vec1, vec2) == numpy.conj(inner_product(vec2, vec1))``.
- Linearity: ``inner_product(a*vec1, vec2) == a*inner_product(vec1, vec2)`` 
  for a scalar ``a``.
- Implied norm: ``inner_product(vec1, vec1) >= 0``, with equality iff ``vec1 == 0``.

To see an example for a non-uniform grid/sampling, see main_bpod_disk.py.

The modes that are produced are also vectors.
We mean this in both the programming sense that modes are also vector objects
and the mathematical sense that modes live in the same vector space as vectors.
After computing the modes, modred calls ``put_vec`` on them.

The three functions ``get_vec``, ``put_vec``, and  ``inner_product`` must be
defined as member functions of a module or class.
We provided a few common ones in ``src/vecdefs``, named as such because
these three functions (along with the "+" and "*" operators) define the
way modred handles the vector objects.


**Checking requirements automatically**

Classes ``BPOD, POD, DMD`` (and ``VecOperations``) include a method ``idiot_check``
that checks common mistakes in your vector object addition, scalar multiplication,
and inner products. 
Still, we encourage you to write your own tests and not risk being exposed
by the ``idiot_check``!




^^^^^^^^^^^^^^^^^^^^^
Use in classes
^^^^^^^^^^^^^^^^^^^^^

The classes POD, BPOD, and DMD have very similar interfaces.
First, they all have ``compute_decomp`` and ``compute_decomp_and_return``
functions that take as an argument ``vec_sources``, which is a list with 
``vec_source`` type elements.
Within each class's ``compute_decomp`` and ``compute_decomp_and_return``
functions, ``get_vec`` is called repeatedly with an argument that is an element of 
``vec_sources``.
In fact, ``compute_decomp`` and ``compute_decomp_and_return`` do not "know"
what's inside ``vec_sources``, they just pass its elements along to ``get_vec``.

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
