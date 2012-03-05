============
Introduction
============

Welcome to the modred project!

This is a easy-to-use, multi-purpose, and parallelized library for finding modal
decompositions and reduced-order models.

Parallel implementations of POD, BPOD, and DMD are provided, as well as serial implementations
of OKID and ERA.

=============
Naming conventions
=============

We use the word "field" to describe the pieces of data that are modally decomposed by POD,
BPOD, and DMD.
Others call these snapshots, vectors (as in elements of a vector space), planes of spatial data,
time histories, and many other names.
Within modred, "field" also refers to the object you, the user, use to represent your data.
You are free to choose *any* object, from numpy arrays to your own class, so long as it satisfies
a few simple requirements.

The classes make use of method names beginning with "get" and "put".
You, the user, must define ``get_field`` and ``put_field``, and optionally you can define
``get_mat`` and ``save_mat`` (for matrices).
First consider ``get_field``. It takes one argument and must return a ``field`` object, which can be
*any* object that supports addition and scalar multiplication (via ``__add__`` and ``__mul__``).
It can be understood as "``get_field`` grabs a field from the source pointed to by its argument, 
possibly does some operations to that field, then returns that field for use in modred", like 
loading, but more general. 
The source can be a file name, a set of indices for accessing data in an array, anything.
Since you define ``get_field``, you just have to be consistent.

Similarly, ``put_field`` takes a ``field`` object and "puts" it somewhere else, like into another
variable outside of modred or even saved to a file.
It can be understood as "``put_field`` takes a field and a destination pointed to by its argument,
possibly does some operations on that field, then puts that field in the destination", like 
saving, but more general.
Just like the source can be anything, so can the destination, like a file name, array indices,
or anything else, you just have to be consistent.


The modes that are produced are also "fields".
We mean this in both the programming sense that modes are also field objects and the mathematical
sense that modes live in the same vector space as fields.

This is all a bit abstract; the following use-cases are helpful.

===============
Examples
===============

**Case 1: Loading and saving fields**

This is a good choice when your data is large and cannot all be in memory simultaneously.

The elements of ``field_sources`` are passed into your ``get_field`` function,
which then must return the correct field. 
Since you, the user, define ``get_field`` it can do anything you want it to do.
A common case for large data is ``get_field`` loads a field from file, and the argument to it
is a path to a file, e.g. ``/user/me/data/velocity_time008.txt``.




The terminology in ERA and OKID is more standardized among different disciplines, and so the
naming schemes should be sufficiently explained by the documentation of those classes.
