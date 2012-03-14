=============
Tutorial
=============

-------------------
The field object
-------------------

The building block of the modal decompositions is the field object, where 
a set of these field objects are modally decomposed by POD, BPOD, and DMD.
Others call these snapshots, vectors (as in elements of a vector space), planes of spatial data,
time histories, and many other names.
Within modred, field refers to the object you, the user, use to represent your data.
You are free to choose *any* object, from numpy arrays to your own class, so long as it satisfies
a few simple requirements.
It is intentionally extremely flexible so it can fill many different needs.

These requirements are:

1. Must be compatible with supplied ``inner_product`` function (described later).
2. Must support scalar multiplication, i.e. ``field2 = 2.0*field1``. 
3. Must support addition with other fields, i.e. ``field3 = field1 + field2``.
4. Must be compatible with supplied ``get_field`` and ``put_field`` functions (described later).

Requirements 1--3 essentially require that your field object is an element of a
Hilbert vector space.
Numpy arrays already meet requirements 2 and 3. 
For your own classes, define ``__mul__`` and ``__add__`` special methods for 2 and 3 (see
examples/main_bpod_disk.py).

----------------------------
Functions of field objects
----------------------------

The modal decomposition classes (POD, BPOD, and DMD) manipulate the fields
with functions you supply. 
There are three such functions:

1. ``get_field(field_source)``: Gets a field specified by ``field_source`` and returns it.
2. ``put_field(field, field_dest)``: Puts ``field`` in the destination specified by ``field_dest``.
3. ``inner_product(field1, field2)``: Returns the inner product between two fields.

First consider ``get_field``. 
It gets a field from the source specified by its argument (which can be anything), 
possibly does some operations to that field, then returns that field for use in modred.
In a sense it is like loading, but more general. 
The ``field_source`` can be a file name, a set of indices for accessing data in an array,
a tuple with several entries, anything.
Since you choose ``get_field`` and the arguments it will take, you just have to be consistent.

Similarly, ``put_field`` takes a field, possibly does some operations on that field, 
then puts that field into the destination pointed to by its second argument.
In a sense it is like saving, but more general.
Just like ``field_source``, ``field_dest`` can be anything, you just have to be consistent.

You also need an inner product function that takes two fields returns a single number.
This number can be real or complex, but must always be the same type.
Your inner product must satisfy the mathematical definition for an inner product, namely:

1. Conjugate symmetry: ``inner_product(field1, field2) == numpy.conj(inner_product(field2, field1))``
2. Linearity: ``inner_product(a*field1, field2) == a*inner_product(field1, field12)`` 
   for a scalar ``a``.
3. Implied norm: ``inner_product(field1, field1) >= 0``, with equality iff ``field1 == 0``.


The modes that are produced are also fields.
We mean this in both the programming sense that modes are also field objects and the mathematical
sense that modes live in the same vector space as fields.
modred ``put_field``s them.


This is all a bit abstract; the following use-cases are helpful.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Examples of ``get_field`` and ``put_field``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Case 1: Loading and saving fields**

This is a good choice when your data is large and cannot all be in memory simultaneously.
You could define your ``get_field`` function to simply take a path as its argument,
load the data in that path, and return a field object. Similarly, ``put_field`` can
save the field to a path. For parellelization, using files to store fields is **strongly 
recommended**.

A typical case is provided in examples/main_bpod_disk.py.
Here we show a brief example::

  import modred
  def my_get_field(path):
      field = modred.load_mat_text(path)
      return field
  
  def my_put_field(field, path):
      modred.save_mat_text(field, path)
      
Here the ``field`` is actually a numpy array, which is perfectly fine.

Another simple case is defining wrappers for your own ``FieldClass`` class::

  class FieldClass(object):
      def load(self, path):
          # Load data from disk
      def save(self, path):
          # Save data to disk
      #...
      
  def my_get_field(path):
      field = FieldClass()
      field.load(path)
      return field
  
  def my_put_field(field, path):
      field.save(path)


**Case 2: Putting fields into an external array**

If your data is not very big and you do not plan on using modred in parallel, saving and loading
your data can be unnecessary. 
Instead, you might just want modred to return the modes.
*We provide you with convenience classes which make this very easy*, 
see classes ``modred.SimpleUsePOD``, ``modred.SimpleUseBPOD``, and ``modred.SimpleUseDMD``.

Below we show you the basic way these convenience classes interact with the rest of the
modred library::

  def my_get_field(array_and_index):
      my_array = array_and_index[0]
      index = array_and_index[1]
      return my_array[index]
  
  def my_put_field(field, array_and_index):
      my_array = array_and_index[0]
      index = array_and_index[1]
      my_array[index] = field


As another demonstrationg of how to bypass loading and saving, you can use a ``DataClass``::
  
  class DataClass(object):
      def __init__(self):
          # Create the fields to decompose into modes.
      #...
      
  def my_get_field(data_class_and_attr):
      my_data_class = data_class_and_attr[0]
      attr = data_class_and_attr[1]
      return getattr(my_data_class, attr)
  
  def my_put_field(field, data_class_and_attr):
      my_data_class = data_class_and_attr[0]
      attr = data_class_and_attr[1]
      setattr(my_data_class, attr, field)

There are of course many other implementation choices, these are just a few simple examples
to help your understanding and inspire your own choices.




-----------------------------------
The ``inner_product`` function
-----------------------------------

A default inner product is provided as ``modred.inner_product``, which assumes
the fields are numpy arrays and does ``(field1*field2.conj()).sum()``.
Things tend not to be so simple in the real world. 
First of all, you may not be using numpy arrays as your field object.
Secondly, your data might be more complicated and require several steps
to find the inner product accurately and efficiently.

Therefore we allow you to supply your own inner product.
To see an example for a non-uniform grid/sampling, see ``examples/main_bpod_disk.py``.


---------------------------------------
Checking requirements automatically
---------------------------------------

Classes ``BPOD, POD, DMD`` (and ``FieldOperations``) include a method ``idiot_check``
that checks common mistakes. 
Still, we encourage you to write your own tests and not risk being exposed
by the ``idiot_check``!


---------------------------------------
Functions of matrices
---------------------------------------

You can also define ``put_mat`` and ``get_mat``. 
They are exactly analagous to the field
cases. 
However, in this case modred supplies a default to save and load matrices to text files.




---------------------------------------
Summary and getting started
---------------------------------------

Summarizing, define suitable

1. ``field`` object
2. ``get_field`` function
3. ``put_field`` function
4. ``inner_product`` function

then you can get started using any of the modal decomposition classes (POD, BPOD, and DMD)!

The rest of this sphinx documentation has details on how to use each individual class
and method, including common usages.

The examples directory is a good place to see how everything works together.

---------------------------------------
ERA and OKID
---------------------------------------

The terminology in ERA and OKID is more standardized among different disciplines, and so the
naming schemes should be sufficiently explained by the documentation of those classes.
