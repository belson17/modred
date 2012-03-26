=============
Customization
=============

-------------------
The vector object
-------------------

The building block of the modal decompositions is the vector object.
Sets of these vector objects are modally decomposed by POD, BPOD, and DMD.
Others call these snapshots, planes of spatial data, fields, time histories,
and many other names.
Within modred, vector refers to the object you, the user, use to represent your data.
You are free to choose *any* object, from numpy arrays to your own class, so long as it satisfies
a few simple requirements.
It is intentionally extremely flexible so it can fill many different needs.
**By "vector", we do not mean a 1D array**. 
We do mean an element of a vector space (technically a Hilbert space).

The requirements of the vector object are that it must:

1. Be compatible with supplied ``inner_product`` function (described later).
2. Support scalar multiplication, i.e. ``vector2 = 2.0*vector1``. 
3. Support addition with other vectors, i.e. ``vector3 = vector1 + vector2``.
4. Be compatible with supplied ``get_vector`` and ``put_vector`` functions (described later).

Numpy arrays already meet requirements 2 and 3. 
For your own classes, define ``__mul__`` and ``__add__`` special methods for 2 and 3 (see
examples/main_bpod_disk.py).

----------------------------
Functions of vector objects
----------------------------

The modal decomposition classes (POD, BPOD, and DMD) manipulate the vectors
with functions you supply. 
There are three such functions:

1. ``get_vec(vec_source)``: Gets a vector specified by ``vec_source`` and returns it.
2. ``put_vec(vec, vec_dest)``: Puts ``vec`` in the destination specified by ``vec_dest``.
3. ``inner_product(vec1, vec2)``: Returns the inner product between two vectors.

First consider ``get_vec``. 
It gets a vector from the source specified by its argument (which can be anything), 
possibly does some operations to that vector, then returns that vector for use in modred.
In a sense it is like loading, but more general. 
The ``vector_source`` can be a file name, a set of indices for accessing data in an array,
a tuple with several entries, anything.
Since you choose ``get_vec`` and the arguments it will take, you just have to be consistent.

Similarly, ``put_vec`` takes a vector, possibly does some operations on that vector, 
then puts that vector into the destination pointed to by its second argument.
In a sense it is like saving, but more general.
Just like ``vec_source``, ``vec_dest`` can be anything, you just have to be consistent.

You also need an inner product function that takes two vectors returns a single number.
This number can be real or complex, but must always be the same type.
Your inner product must satisfy the mathematical definition for an inner product, namely:

1. Conjugate symmetry: ``inner_product(vec1, vec2) == numpy.conj(inner_product(vec2, vec1))``
2. Linearity: ``inner_product(a*vec1, vec2) == a*inner_product(vec1, vec2)`` 
   for a scalar ``a``.
3. Implied norm: ``inner_product(vec1, vec1) >= 0``, with equality iff ``vec1 == 0``.


The modes that are produced are also vectors.
We mean this in both the programming sense that modes are also vector objects and the mathematical
sense that modes live in the same vector space as vectors.
After computing them, modred calls ``put_vec`` on them.

This is all a bit abstract; the following use-cases are helpful.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Examples of ``get_vec`` and ``put_vec``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Case 1: Loading and saving vectors**

This is a good choice when your data is large and cannot all be in memory simultaneously.
Even if your data isn't that large, this is a fine choice.
You can define your ``get_vec`` function to simply take a path as its argument,
load the data in that path, and return a vector object. Similarly, ``put_vec`` can
save the vector to a path. 
For parellelization, using files to store vectors is **strongly recommended**
for efficiency. 

Many common functions are provided in the ``vecdefs`` module.
Here we reproduce a brief example::

  import modred
  def my_get_vec(path):
      vec = modred.load_mat_text(path)
      return vec
  
  def my_put_vec(vec, path):
      modred.save_mat_text(vec, path)
      
Here the ``vec`` is actually a numpy array, which is perfectly fine.

You could also make your own ``VectorClass`` class::

  class VectorClass(object):
      def load(self, path):
          # Load data from disk
      def save(self, path):
          # Save data to disk
      #...
      
  def my_get_vec(path):
      vec = VectorClass()
      vec.load(path)
      return vec
  
  def my_put_vec(vec, path):
      vec.save(path)


**Case 2: Putting vectors into an external array**

If your data is not very big and you do not plan on using modred in parallel,
saving and loading your data can be unnecessary. 
Instead, you might just want modred to return the modes.
**We provide you with convenience classes which make this very easy**, 
see classes ``modred.SimpleUsePOD``, ``modred.SimpleUseBPOD``, and ``modred.SimpleUseDMD``.

Below we show you the basic way these convenience classes interact with the rest of the
modred library solely for educational purposes. 
You should never have to write this, use the "SimpleUse" classes we give you!::

  def my_get_vec(array_and_index):
      my_array = array_and_index[0]
      index = array_and_index[1]
      return my_array[index]
  
  def my_put_vec(vec, array_and_index):
      my_array = array_and_index[0]
      index = array_and_index[1]
      my_array[index] = vec


As another demonstration of how to bypass loading and saving, you can use
a ``DataClass``::
  
  class DataClass(object):
      def __init__(self):
          # Create the vecs to decompose into modes.
      #...
      
  def my_get_vec(data_class_and_attr):
      my_data_class = data_class_and_attr[0]
      attr = data_class_and_attr[1]
      return getattr(my_data_class, attr)
  
  def my_put_vec(vec, data_class_and_attr):
      my_data_class = data_class_and_attr[0]
      attr = data_class_and_attr[1]
      setattr(my_data_class, attr, vec)

There are of course many other implementation choices, these are just a few simple examples
to help your understanding and inspire your own choices.




-----------------------------------
The ``inner_product`` function
-----------------------------------

A default inner product is provided as ``modred.inner_product``, which assumes
the vectors are numpy arrays and does ``(vec1*vec2.conj()).sum()``.
Things tend not to be so simple in the real world. 
First of all, you may not be using numpy arrays as your vector object.
Secondly, your data might be more complicated and require several steps
to find the inner product accurately and efficiently.

Therefore we allow you to supply your own inner product.
To see an example for a non-uniform grid/sampling, see ``examples/main_bpod_disk.py``.


---------------------------------------
Checking requirements automatically
---------------------------------------

Classes ``BPOD, POD, DMD`` (and ``VecOperations``) include a method ``idiot_check``
that checks common mistakes in your vector object addition, scalar multiplication,
and inner products. 
Still, we encourage you to write your own tests and not risk being exposed
by the ``idiot_check``!


---------------------------------------
Functions of matrices
---------------------------------------

You can also define ``put_mat`` and ``get_mat``. 
They are exactly analagous to the vector cases. 
However, modred supplies a default to save and load matrices (real and imaginary)
to text files.



---------------------------------------
Summary and getting started
---------------------------------------

Summarizing, define suitable

1. ``vec`` object
2. ``get_vec`` function
3. ``put_vec`` function
4. ``inner_product`` function

then you can get started using any of the modal decomposition classes (POD, BPOD, and DMD)!

The rest of this sphinx documentation has details on how to use each individual class
and method, including common usages.

The examples directory is a good place to see how everything works together.

---------------------------------------
ERA and OKID
---------------------------------------

ERA and OKID are standardized among different disciplines, and the 
documentation of those classes should be sufficient.
