============
Introduction
============

Welcome to the modred project!

This is a easy-to-use, multi-purpose, and parallelized library for finding modal
decompositions and reduced-order models.

Parallel implementations of POD, BPOD, and DMD are provided, as well as serial implementations
of OKID and ERA.

====================
Naming conventions
====================

We use the word "field" to describe the pieces of data that are modally decomposed by POD,
BPOD, and DMD.
Others call these snapshots, vectors (as in elements of a vector space), planes of spatial data,
time histories, and many other names.
Within modred, "field" also refers to the object you, the user, use to represent your data.
You are free to choose *any* object, from numpy arrays to your own class, so long as it satisfies
a few simple requirements.

The classes have method names beginning with "get" and "put".
You, the user, must define ``get_field`` and ``put_field``, and optionally you can define
``get_mat`` and ``save_mat`` (for matrices).
First consider ``get_field``. It takes one argument and must return a ``field`` object, which can be
*any* object that supports addition and scalar multiplication (via ``__add__`` and ``__mul__``).
It can be understood as "``get_field`` gets a field from the source pointed to by its argument, 
possibly does some operations to that field, then returns that field for use in modred", like 
loading, but more general. 
The source can be a file name, a set of indices for accessing data in an array, a tuple with
several entries, anything.
Since you choose ``get_field`` and the arguments it will take, you just have to be consistent.

Similarly, ``put_field`` takes a ``field`` object and "puts" it somewhere else, like into another
variable outside of modred or even to a file.
It can be understood as "``put_field`` takes a field, possibly does some operations on that field, 
then puts that field into the destination pointed to by its second argument", like 
saving, but more general.
Just like the source can be anything, so can the destination, you just have to be consistent.


The modes that are produced are also "fields".
We mean this in both the programming sense that modes are also field objects and the mathematical
sense that modes live in the same vector space as fields.

This is all a bit abstract; the following use-cases are helpful.

===========================================
Examples of ``get`` and ``put``
===========================================

**Case 1: Loading and saving fields**

This is a good choice when your data is large and cannot all be in memory simultaneously.
You could define your ``get_field`` function to simply take a path as its argument,
load the data in that path, and return a field object. Similarly, ``put_field`` can
save the field to a path. For parellelization, using files to store fields is **strongly 
recommended**.

A simple case is::

  import modred.util
  def my_get_field(path):
      field = modred.util.load_mat_text(path)
      return field
  
  def my_put_field(field, path):
      modred.util.save_mat_text(field, path)
      
Here the ``field`` is actually a numpy array, which is perfectly fine.

Another simple case is defining wrappers for your own ``FieldClass`` class::

  def FieldClass(object):
      def load(self, path):
          # Load data from disk
      def save(self, path):
          # Save data to disk
      ...
      
  def my_get_field(path):
      field = FieldClass()
      field.load(path)
  
  def my_put_field(field, path):
      field.save(path)


**Case 2: Putting fields into an external array**

If your data is not very big and you do not plan on using modred in parallel, saving and loading
your data can be unnecessary. Instead, you might just want modred to return the modes.
You can do almost this via::

  def my_get_field(array_and_index):
      my_array = array_and_index[0]
      index = array_and_index[1]
      return my_array[index]
  
  def my_put_field(field, array_and_index):
      my_array = array_and_index[0]
      index = array_and_index[1]
      my_array[index] = field

You can't directly return the modes because modred's interface has to properly
deal with large data, and returning all of the modes is impossible due to memory limitations.
Still, you can achieve this in a less direct way as shown above.

As another demonstrationg of how to bypass loading and saving, you can use a ``DataClass``::
  
  def DataClass(object):
      __init__(self):
          # Create the fields to decompose into modes.
      ...
      
  def my_get_field(data_class_and_attr):
      my_data_class = data_class_and_attr[0]
      attr = data_class_and_attr[1]
      return getattr(my_data_class, attr)
  
  def my_put_field(field, data_class_and_attr):
      my_data_class = data_class_and_attr[0]
      attr = data_class_and_attr[1]
      setattr(my_data_class, attr, field)

There are of course many other implementation choices, these are just a few simple examples
to help understanding and inspire your own choices.


**Matrices***

You can also define ``put_mat`` and ``get_mat``. 
They are exactly analagous to the field
cases. 
However, in this case modred supplies a default to save and load matrices to text files.

===============================
The ``inner_product`` function
===============================

In addition to ``get_field`` and ``put_field``, you must define an inner product
function that takes two fields returns a single number.
It can return real or complex, but must always return the same type.
Your inner product must satisfy the mathematical definition for an inner products, namely:

1. Conjugate symmetry: ``inner_product(field1, field2) == numpy.conj(inner_product(field2, field1))``
2. Linearity: ``inner_product(a*field1, field2) == a*inner_product(field1, field12)`` 
   for a scalar ``a``.
3. Implied norm: ``inner_product(field1, field1) >= 0``, with equality iff ``field1 == 0``.


==================================
Requirements for ``field`` object
==================================

As briefly mentioned before, the ``field`` object that is manipulated by ``get_field`` and 
``put_field`` can be *any* object that satisfies a few simple requirements:

1. Must be compatible with supplied ``get_field`` and ``put_field`` functions.
2. Must be compatible with supplied ``inner_product`` function.
3. Must support scalar multiplication, i.e. ``field2 = 2.0*field1``. 
4. Must support addition with other fields, i.e. ``field3 = field1 + field2``.


Numpy arrays already meet requirements 3 and 4. 
For your own classes, define ``__mul__`` and ``__add__`` special methods for 3 and 4.



=====================================
Checking requirements automatically
=====================================

We include a ``FieldOperations`` method ``idiot_check`` that checks common mistakes. 
Still, we encourage you to write your own tests and not risk being exposed by the ``idiot_check``!


============
Get started
============
Summarizing, after you've defined suitable

1. ``get_field`` function
2. ``put_field`` function
3. ``inner_product`` function
4. ``field`` object

you can get started using any of the modal decomposition classes (POD, BPOD, and DMD)!

The rest of this sphinx documentation has details on how to use each individual class
and method, including common usages.

More mathematical information is available in the user's guide.

TODO: Add full examples?


=================
ERA and OKID
=================

The terminology in ERA and OKID is more standardized among different disciplines, and so the
naming schemes should be sufficiently explained by the documentation of those classes.


