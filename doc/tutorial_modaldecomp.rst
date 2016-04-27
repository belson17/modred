.. _sec_modaldecomp:

-------------------------------------------------
Modal decompositions -- POD, BPOD, and DMD
-------------------------------------------------

This tutorial discusses computing modes from data, using the Proper Orthogonal
Decomposition (POD), Balanced Proper Orthogonal Decomposition (BPOD), and
Dynamic Mode Decomposition (DMD).  
For details of these algorithms, see [HLBR]_ for POD and BPOD and [TRLBK]_ for 
DMD.

The first step is to collect your data.  
We call each piece of data a "vector" or "vector object".  
**By vector, we don't mean a 1D array.**  
Rather, we mean an element of a vector space.  
This could be a 1D, 2D, or 3D array, or any other object that satisfies the 
properties of a vector space.  
The examples below build on one another, each introducing new aspects.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 1 -- All data in a matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A simple way to find POD modes is:

.. literalinclude:: ../modred/examples/tutorial_ex1.py

Let's walk through the important steps.  
First, we create an array of random data.  
Each column is a vector represented as a 1D array.  
Then we call the function ``compute_POD_matrices_direct_method``, which returns
the first ``num_modes`` modes as columns of the matrix ``modes``, and all of the
non-zero eigenvalues, sorted from largest to smallest.

This function implements the "method of snapshots", as described in Section 3.4
of [HLBR]_.  
In short, it computes the correlation matrix :math:`X^* X`, where :math:`X` is `
`vecs``, then finds its eigenvectors and eigenvalues, which are related to the 
modes.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 2 -- Inner products with all data in a matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can use a weighted inner product, specified here by a 1D array of weights,
so that the correlation matrix is :math:`X^* W X`, where :math:`X` is ``vecs``
and :math:`W` contains the inner product weights.  
The weights also can, more generally, be a matrix.
The vectors are again represented as columns of a matrix.

.. literalinclude:: ../modred/examples/tutorial_ex2.py

This function computes the singular value decomposition (SVD) of :math:`W^(1/2)
X`, and we refer to this as the "direct method" to distinguish it from the
method of snapshots in the previous example.  
The differences between the two are insignificant in most cases.  
For details, see :py:func:`pod.compute_POD_arrays_direct_method`.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 3 -- Vector handles for loading and saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example demonstrates *vector handles*, which are very important because
they allow modred to interact with large and complicated datasets without
requiring that all of the vectors be stacked into a single matrix.  
This is necessary, for example, if the data is too large to all fit in memory
simultaneously.

The following example computes direct and adjoint modes using Balanced
POD (see Chapter 5 of [HLBR]_):

.. literalinclude:: ../modred/examples/tutorial_ex3.py

First, we create lists ``direct_snapshots`` and ``adjoint_snapshots``.
("Snapshots" is just a common word to describe vectors that come from 
time-sampling a system as it evolves.)
Each element in these lists is a vector handle (in particular, an instance of
``VecHandleArrayText``).
All vector handles have member functions to load, ``vec = vec_handle.get()``,
and save, ``vec_handle.put(vec)``, but themselves use very little memory because
they do *not* internally contain a vector.
modred uses these vector handles to load and save individual vectors only as it 
needs them.

All the snapshots are vectors and each is represented as an array, but they are
*not* stacked into a single 2D array. 
Ordinarily the snapshots would already exist, for instance as files from a
simulation or experiment.
Here, we artificially generate snapshots and write them to file using the
``put()`` method.

Next, we compute the BPOD modes.
The ``BPODHandles`` constructor takes an optional argument
``max_vecs_per_node=10``, specifying that only 10 vectors (snapshots + modes)
can be in one node's memory at one time.
The function ``compute_decomp`` takes lists of *vector handles* as arguments.
In this example, note that there are 30 direct and 30 adjoint snapshots, so
handles are necessary to avoid violating ``max_vecs_per_node=10``.

Similarly, ``compute_direct_modes`` and ``compute_adjoint_modes`` take lists of
handles.
The modes are saved via ``handle.put()`` rather than returned, which might
require too much memory.

Replacing ``VecHandleArrayText`` with ``VecHandlePickle`` would load/save  all
vectors (snapshots and/or modes) to Python's binary pickle files.
Note that pickling works with *any* type of vector, including user-defined ones,
whereas saving to text is only written for 1D and 2D arrays.

To run this example in parallel is easy.
The only complication is the data must be saved by only one processor, and
moving these lines inside an ``if`` block solves this::
  
  parallel = mr.parallel_default_instance
  if parallel.is_rank_zero():
      # Loops that call handles.put
      pass
  parallel.barrier()

After this change, the code will still work in serial, even if mpi4py is not
installed.
To run this, where the above script is saved as ``main_bpod.py``, execute:: 
  
  mpiexec -n 8 python main_bpod.py

It is rare to need to handle parallelization yourself, but if you do, 
you should use the provided ``mr.parallel_default_instance`` instance
as above.
Also provided are member functions ``parallel.get_rank()`` and 
``parallel.get_num_procs()`` (see :py:mod:`parallel` for details).

If you're curious, the text files are saved in a format defined in
:py:func:`util.load_array_text` and :py:func:`util.save_array_text`. 


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 4 -- Inner product function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You are free to use any inner product function, as long as it has the interface 
``value = inner_product(vec1, vec2)`` and satisfies the mathematical
definition of an inner product (see :ref:`sec_details`).
This example uses the trapezoidal rule for inner products on an arbitrary 
n-dimensional cartesian grid (see :py:class:`vectors.InnerProductTrapz`).
The object ``weighted_IP`` is callable (it has a special method ``__call__``) so
it acts as the inner product the usual way: ``value = weighted_IP(vec1, vec2)``.

.. literalinclude:: ../modred/examples/tutorial_ex4.py

Also shown in this example is the useful ``put_decomp``, which, by default 
saves the arrays associated with the decomposition to text files.
(See :ref:`sec_matrices`.)

Again, this code can be executed in parallel without any modifications.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 5 -- Shifting and scaling vectors using vector handles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Often vectors contain an offset (also called a shift or translation) such as a
mean or equilibrium state, and one might want to do model reduction with this
known offset removed.
We call this offset the base vector, and it can be subtracted off by the vector
handle class as shown in this example. 
  
Note that ``handle.put`` does *not* use the base vector; the base vector is only
subtracted by ``handle.get``.

You might also want to scale all of your vectors by factors as you retrieve them
for use in modred, and this can also be done by the ``handle.get`` function.
When using both base vector shifting and scaling, the default order is first
shifting then scaling: ``(vec - base_vec)*scale``.

This examples uses quadrature weights, where each vector is weighted.
It also shows how to load vectors in one format (pickle, via
``mr.VecHandlePickle``) and save modes in another (text, via
``mr.VecHandleArrayText``).

.. literalinclude:: ../modred/examples/tutorial_ex5.py

At the end of this example, we use an instance of the low-level class
:class:`vectorspace.VectorSpaceHandles` to check that the POD modes are
orthonormal.
It's generally not necessary to use this class (or do this check), but if the
need arises, it should be used (instead of writing new code) since it is tested 
and parallelized.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 6 -- User-defined vectors and handles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
So far, all of the vectors have been arrays, but you may want to apply modred to
data saved in your own custom format with more complicated inner products and
other operations.
This is no problem at all; modred works with data in any format!
That's worth saying again: **modred works with data in any format!**
Of course, you'll have to tell modred how to interact with your data, but that's
pretty easy.
You just need to define and use your own vector handle and vector objects.

There are two important new features of this example: a custom vector class
``CustomVector`` and a custom vector handle class ``CustomVecHandle``.  
These definitions may be collected together in a file, for instance called
``customvector.py``:

.. literalinclude:: ../modred/examples/customvector.py

Instances of ``CustomVector`` meet the requirements for a vector object: vector
addition ``__add__`` and scalar multiplication ``__mul__`` are defined, and the
objects are compatible with an inner product function such as
``inner_product(v1, v2)``.
Note that ``CustomVector`` inherits from a base class ``mr.Vector``.
This is not required, but is recommended, as the base class provides some useful
additional methods.
The member function ``inner_product`` is useful, but not required.
This example also uses the trapezoidal rule for inner products to account for a
3D arbitrary cartesian grid (:py:class:`vectors.InnerProductTrapz`).

The class ``CustomVecHandle`` inherits from a base class ``mr.VecHandle``, and
defines methods ``_get`` and ``_put``, which load and save vectors from/to
Pickle files.  
Note the leading underscore: the functions ``get`` and ``put`` (without leading
underscore) are defined in the ``VecHandle`` base class, and take care of
scaling or subtracting a base vector.  
The methods ``_get`` and ``_put`` defined here are in turn called by the base
class (the "template method" design pattern).
While you don't need to understand the guts of these base classes, more is
covered in :ref:`sec_details`.

Here's an example using these classes:

.. literalinclude:: ../modred/examples/tutorial_ex6.py

This example is similar to previous ones, but some aspects are worth pointing
out.
The call to ``my_BPOD.sanity_check()`` runs some basic tests to verify that the
vector handles and vectors behave as expected, so this can be useful for
debugging.
After execution, the modes are saved to ``direct_mode0.pkl``,
``direct_mode1.pkl`` ... and ``adjoint_mode0.pkl``, ``adjoint_mode1.pkl``.
Also, note that the only time the ``CustomVector`` class is used is in
generating the "fake" random data: most scripts will only need to deal with the
vector handle classes, not the vectors themselves.

When you're ready to start using modred, take a look at what types of vectors,
file formats, and inner products are supplied in :mod:`vectors`.
If you don't find what you need, you can define your own vectors and vector
handles following examples like this.
Built-in classes like ``VecHandleArrayText`` and ``VecHandlePickle`` are common
cases and serve as good examples since your own custom vector handle classes
will probably resemble them.

As usual, this example can be executed in parallel without any modifications.


.. _sec_matrices:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Matrix input and output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``put_mat`` and ``get_mat`` save and load to text files.
If you prefer a different format, you can pass your own functions as keyword
arguments ``put_mat`` and ``get_mat`` to the constructors.
The functions should have the following interfaces: ``put_mat(mat, mat_dest)`` 
and ``mat = get_mat(mat_source)``.
The ``mat`` argument could be a numpy matrix, 1D array, or 2D array. 

^^^^^^^^^^^^^^
References
^^^^^^^^^^^^^^

.. [HLBR] P. Holmes, J. L. Lumley, G. Berkooz, and C. W. Rowley.
   *Turbulence, Coherent Structures, Dynamical Systems and Symmetry*,
   2nd edition, Cambridge University Press, 2012.

.. [TRLBK] J. H. Tu, C. W. Rowle, D. M. Luchtenburg, S. L. Brunton, J. N. Kutz.
   On Dynamic Mode Decmposition: Theory and Applications.
   *Journal of Computational Dynamics*, 1:391-421, Dec. 2014.
