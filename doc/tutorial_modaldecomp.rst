.. _sec_modaldecomp:

-------------------------------------------------
Modal decompositions -- POD, BPOD, and DMD
-------------------------------------------------

This tutorial discusses computing modes from data, using Proper
Orthogonal Decomposition (POD), Balanced Proper Orthogonal
Decomposition (BPOD), and Dynamic Mode Decomposition (DMD).
For details of these algorithms, see [HLBR]_ for POD and BPOD, and
[RMBSH]_ or [Schmid]_ for DMD.

The first step is to collect your data. 
We call each piece of data a vector.
**By vector, we don't mean a 1D array.**  Rather, we mean an element
of a vector space.
This could be a 1D, 2D, or 3D array, or any other
object that satisfies the properties of a vector space.
The examples below build on one another, each introducing one or more
new functions or features.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 1 -- Data in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A simple way to use modred to find POD modes is as follows:

.. literalinclude:: ../examples/tutorial_ex1.py

Let's walk through the important steps.
First, we created a list of arbitrary arrays; these are the vectors.
(The ``vecs`` argument must be a list.)
Then we created an instance of ``POD`` called ``pod``.
The constructor took the argument
``inner_product``, a function that takes two vectors (numpy arrays in this case),
and returns their inner product. 
The ``inner_product`` must satisfy the properties of inner products, and is
explained more in :ref:`sec_details`.

The next line, ``compute_decomp_in_memory`` computes the correlation matrix 
(often written "X* X", if the data vectors are columns of the matrix
X), and returns its eigenvectors and eigenvalues.
The correlation matrix is computed by taking the inner products of all 
combinations of vectors.
This procedure is called the "method of snapshots", as described in Section 3.4 of [HLBR]_.
We stress that modred's approach is to never form the "X" matrix, which is
why we pass a list of vectors rather than a single large array or matrix.
This is explained further in later sections.

The last line, ``compute_modes_in_memory`` takes a list of mode numbers as an
argument and returns the list of modes, ``modes``. 

The above example can be run in parallel with *no modifications*.
At the end, each processor (MPI worker more generally) will have all of the
modes in its list ``modes``.
To do this, assuming the above script is saved as ``main_pod.py``, one
would execute the following:: 
  
  mpiexec -n 8 python main_pod.py

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 2 -- Inner product functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You are free to use any inner product function with interface 
``value = inner_product(vec1, vec2)`` and that satisfies the mathematical
definition (see :ref:`sec_details`).
This example uses a provided trapezoidal rule for inner products on 
an arbitrary n-dimensional cartesian grid 
(:py:class:`vectors.InnerProductTrapz`).
The vector objects are again numpy arrays.

.. literalinclude:: ../examples/tutorial_ex2.py

The object ``weighted_ip`` is a callable (that is, it has a special
method ``__call__``) so it acts as the inner product the usual
way: ``value = weighted_ip(v1, v2)``.

This code can be executed in parallel without any modifications.



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 3 -- Vector handles for loading and saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example again uses numpy arrays as the vectors, but this time
we load and save them from and to text files with vector handles
``ArrayTextVecHandle``.
This is a very important difference from the previous example because vector
handles allow modred to efficiently interact with large data.
The following example computes direct and adjoint modes using Balanced
POD (see Chapter 5 of [HLBR]_):

.. literalinclude:: ../examples/tutorial_ex3.py

First, we create lists of snapshots, ``direct_snapshots`` and
``adjoint_snapshots``.
Each element in these lists is a vector handle, which is a lighweight
pointer to a vector.
They are necessary for large data when memory is limited, i.e., in cases
where it is impossible or inefficient to have a list of all vectors.
The vector handles know where to retrieve the data for a particular
vector (e.g., the filename), but the actual data is not loaded until
it is needed.
For this reason, one may create large lists of vector handles without
worrying about filling up the machine's memory.
If one has a vector handle ``vec_handle``, one may load or save the
corresponding vector with ``vec = vec_handle.get()`` or
``vec_handle.put(vec)``, respectively.

Ordinarily, the snapshots would already exist, for instance as files
on disk.
In the example above, we generate some artificial data and write it to
the corresponding files, using the ``put()`` method just described.

Next, we compute the BPOD modes.
The ``BPOD`` constructor takes an optional argument
``max_vecs_per_node=10``, ensuring that no more than 10
vectors (snapshots + modes) are loaded at once on one node.
The function ``compute_decomp`` takes lists of vector *handles* as
arguments---by contrast, ``compute_decomp_in_memory`` used in the
previous examples takes lists of *vectors* as arguments, and is thus unsuitable for large datasets.
The modred library calls ``vec = handle.get()`` internally only when the 
vector is needed.
In this example, note that we we couldn't pass all 30 direct and 30 adjoint 
snapshots to modred
without violating ``max_vecs_per_node``, so handles are essential.

Similarly, ``compute_direct_modes`` and ``compute_adjoint_modes`` take
lists of handles, and save all of the modes internally via ``put()``,
rather than returning a list of modes.

Replacing ``ArrayTextVecHandle`` with ``PickleVecHandle`` would load/save  
all vectors (snapshots and/or modes) to pickle files.
Pickling works with *any* type of vector, including user-defined ones, not
only numpy arrays.

To run this example in parallel is easy.
The only complication is the data must be saved by only one processor 
(MPI worker).
Moving a few lines inside the following if block solves this::
  
  parallel = MR.parallel_default_instance
  if parallel.is_rank_zero():
      # Loops that call handles.put
      pass
  parallel.barrier()

After this change, the code will still work in serial, even if mpi4py is not
installed.
It is rare to need to handle parallelization yourself, but if you do, 
you should use the provided ``parallel`` class instance
as in this example.
Also provided are member functions ``parallel.get_rank()`` and 
``parallel.get_num_procs()`` (see docs for details).

If you're curious, the text files are saved with whitespace after each
column entry and
line breaks after each row, so the 2x3 array::
  
  1 2 3
  4 5 6

looks just like this in the text file. See docs for ``util.load_array_text`` 
and ``util.save_array_text`` for more information. 


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 4 -- Subtracting a base vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often vectors are saved with an offset (also called a "shift" or "translation") 
such as a mean or equilibrium state, but we want to do model 
reduction with this known offset removed.
We call this offset the "base vector", and it can be subtracted off by the
vector handle class as shown below.  The following example computes
DMD modes from a given set of snapshots:

.. literalinclude:: ../examples/tutorial_ex4.py
  
Note that the ``handle.put`` function does not use the base vector; the base
vector is only subtracted from the loaded vector when ``handle.get``
is called.
To run this example in parallel, the ``put`` loop must be done only on
one processor, as in the previous example.
The function ``dmd.put_decomp``, by default, saves the three decomposition
matrices to text files.
This behavior can be changed by passing the ``DMD`` constructor
the optional argument ``put_mat=`` as a different function to "put" the matrices
in a different way, for instance to a different file format.
See :ref:`sec_matrices`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 5 -- Scaling vectors and using ``VectorSpace``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might want to scale all of your vectors by factors, and this can be done
by the vector handle ``get`` function, just like the base vector.
For example, below we show the use of quadrature weights, where each vector
is weighted.
This example also shows how to load vectors in one format (pickle) 
and save modes in another (text).
At the end of this example, we use the lower-level 
:class:`vectorspace.VectorSpace` class to check that the POD modes are 
orthonormal.

.. literalinclude:: ../examples/tutorial_ex5.py
      
When using both base vector subtraction and scaling, the default order
is first subtraction, then multiplication: ``(vec - base_vec)*scale``.

The input vectors are saved in pickle format (``MR.PickleVecHandle``) 
and the modes are saved in text format (``MR.ArrayTextVecHandle``).

The last section uses the ``VectorSpace`` class, 
which contains most of the parallelization and "heavy lifting" and is
used by ``POD``, ``BPOD``, and ``DMD``.
It is a good idea to use this class whenever possible since it is tested
and parallelized (see :mod:`vectorspace`).


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 6 -- User-defined vectors and handles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far all of the vectors have been arrays, but you may want to apply modred 
to data saved in your own custom format with more complicated inner 
products and other operations.
This is no problem at all; modred works with any data in any format!
That's worth saying again: **modred works with any data in any format!**
Of course, you'll have to tell modred how to interact with your data, but 
that's pretty easy.
You just need to define and use your own vector handle and vector objects.

There are two important new features of this example: a custom vector class
``CustomVector`` and a custom vector handle class
``CustomVecHandle``.  These definitions may be collected together in a
file, for instance called ``custom_vector.py``:

.. literalinclude:: ../examples/custom_vector.py

A ``CustomVector`` meets the requirements for a vector object: vector addition
``__add__`` and scalar multiplication ``__mul__`` are defined, and 
the objects are compatible with an inner product function such as
``inner_product(v1, v2)``.
Note that ``CustomVector`` inherits from a base class ``MR.Vector``.
This is not required, but is recommended, as the base class provides
some useful additional methods.
The member function ``inner_product`` is useful, but not required.
This example also uses the trapezoidal rule for inner products to account for 
an 3D arbitrary cartesian grid
(:py:class:`vectors.InnerProductTrapz`).

The class ``CustomVecHandle`` inherits from a base class
``MR.VecHandle``, and defines methods ``_get`` and ``_put``, which
load and save vectors from/to Pickle files.  Note the leading
underscore: the functions ``get`` and ``put`` (without leading
underscore) are defined in the ``VecHandle`` base class, and take care
of things like scaling or subtracting a base vector.  The methods
``_get`` and ``_put`` defined here are in turn called by the base
class (the "template method" design pattern).
While you don't need to understand the guts of these base classes, more 
is covered in :ref:`sec_details`.

The above file may then be imported whenever the ``CustomVector``
class is needed, as in the following example:

.. literalinclude:: ../examples/tutorial_ex6.py

This example is similar to previous ones, but some aspects are worth
pointing out.
The call to ``bpod.sanity_check()`` runs some basic tests to verify
that the vector handles and vectors behave as expected, so this can be
useful for debugging.
After execution, the modes are saved to ``direct_mode0.pkl``, 
``direct_mode1.pkl`` ... and ``adjoint_mode0.pkl``, 
``adjoint_mode1.pkl``.
Also, note that the only time the ``CustomVector`` class is used is in
generating the "fake" random data: most scripts will only need
to deal with the vector handle classes, not the vectors themselves.

When you're ready to start using modred, take a look at what types of 
vectors, file formats, and inner_products are supplied in :mod:`vectors`.
If you don't find what you need, you can define your own vectors and vector handles following examples like this.
Built-in classes like ``ArrayTextVecHandle`` and ``PickleVecHandle`` are
good examples, and your own custom vector handle classes will probably
closely resemble these.
We provide them with modred since they are common cases.

As usual, this example can be executed in parallel without any 
modifications.

 

.. _sec_matrices:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Matrix input and output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``put_mat`` and ``get_mat`` save and load to text files.
This tends to be a versatile option because the files are
easy to load into other programs (such as Matlab), and are human-readable, portable, and
rarely large enough to be problematic.
However, if you prefer a different format, you can define your own
functions ``put_mat`` and ``get_mat``, and pass them as optional
arguments to the constructors for ``POD``, ``BPOD``, etc..
For example, you can save in a different file format, or ``get`` and ``put``
the matrices to another class's data member. The only requirement is that the functions
match the required interfaces: ``put_mat(mat, mat_dest)`` and
``mat = get_mat(mat_source)``.

While these functions' names suggest that they work for numpy matrices, they 
must also accept 1D and 2D arrays as arguments. 

^^^^^^^^^^^^^^
References
^^^^^^^^^^^^^^

.. [HLBR] P. Holmes, J. L. Lumley, G. Berkooz, and C. W. Rowley.
   *Turbulence, Coherent Structures, Dynamical Systems and Symmetry*,
   2nd edition, Cambridge University Press, 2012.

.. [RMBSH] C. W. Rowley, I. Mezic, S. Bagheri, P. Schlatter, and D. S. Henningson.
   Spectral analysis of nonlinear flows.
   *Journal of Fluid Mechanics*, 641:115-127, Dec 2009.

.. [Schmid] P. J. Schmid.  Dynamic mode decomposition of numerical and
   experimental data.
   *Journal of FLuid Mechanics*, 656:5-28, Aug 2010.
