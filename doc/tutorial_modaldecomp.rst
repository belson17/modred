.. _sec_modaldecomp:

-------------------------------------------------
Modal decompositions -- POD, BPOD, and DMD
-------------------------------------------------

First, collect your data. 
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
A simple way to use modred to find POD modes is.

.. literalinclude:: ../examples/tutorial_ex1.py

Let's walk through the important steps.
First, we created a list of arbitrary arrays; these are the vectors.
(The ``vecs`` argument must be a list.)
Then we created an instance of ``POD`` called ``my_POD``.
The constructor took the argument
``inner_product``, a function that takes two vectors (numpy arrays in this case),
and returns their inner product. 
The ``inner_product`` must satisfy the properties of inner products, and is
explained more in :ref:`sec_details`.

The next line, ``compute_decomp_in_memory`` computes the correlation matrix 
(often written "X* X"), takes its SVD, and returns the SVD matrices 
(the left and right singular vectors are equal for POD).
The correlation matrix is computed by taking the inner products of all 
combinations of vectors.
We stress that modred's approach is to never form the "X" matrix, which is
why we pass a list of vectors rather than a single large array or matrix.
This is explained in later sections.

The last line, ``compute_modes_in_memory`` takes a list of mode numbers as an
argument and returns the list of modes, ``modes``. 

The above example can be run in parallel with *no modifications*.
At the end, each processor (MPI worker more generally) will have all of the
modes in its list ``modes``.
To do this, assuming the above script is saved as ``main_pod.py`, one
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

The instance ``my_IP_trapz`` is a callable (that is, it has a special
method ``__call__``) so it acts as the inner product the usual
way: ``value = my_IP_trapz(v1, v2)``.

This code can be executed in parallel without any modifications.



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 3 -- Vector handles for loading and saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example again uses numpy arrays as the vectors, but this time
we load and save them from and to text files with vector handles
``ArrayTextVecHandle``.
This is a very important difference from the previous example because vector
handles allow modred to efficiently interact with large data.

.. literalinclude:: ../examples/tutorial_ex3.py

First, arrays are filled with arbitrary data to serve as the vectors.
Then, we create lists of instances of vector handles, in particular 
the class ``ArrayTextVecHandle``.
The vector handles are lightweight pointers to a vector. 
In this case, each handle contains a path where a vector is saved. 
They are necessary for large data when memory is limited, i.e. cases
where it is impossible or inefficient to have a list of all vectors.
Instead, we work with these lightweight handles which save and/or load
vectors when requested via ``vec_handle.put(vec)`` and 
``vec = vec_handle.get()``, respectively.

Returning to the example, the ``BPOD`` constructor takes optional argument
``max_vecs_per_node=10``, ensuring that no more than 10
vectors (snapshots + modes) are loaded at once on one node.
Function ``compute_decomp`` takes *handles* to vectors as arguments instead of
vectors.
Modred calls ``vec = handle.get()`` internally only when the 
vector is needed. 
In this example, note that we we couldn't pass all 30 direct and 30 adjoint 
snapshots to modred
without violating ``max_vecs_per_node``, so handles are essential.

Similarly, ``compute_direct_modes`` and ``compute_adjoint_modes`` take handles
and save all of the modes via ``mode_handle.put(mode)`` internally, rather than
returning a list of modes.

Replacing ``ArrayTextVecHandle`` with ``PickleVecHandle`` would load/save  
all vectors (snapshots and modes) to pickle files.
Pickling works with *any* type of vector, including user-defined ones, not
only numpy arrays.

To run this in parallel is easy.
The only complication is the data must be saved by only one processor 
(MPI worker).
Moving a few lines inside the following if block solves this::
  
  parallel = MR.parallel.default_instance
  if parallel.is_rank_zero():
      # Loops that call handles.put
      pass

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
vector handle class as shown below.

.. literalinclude:: ../examples/tutorial_ex4.py
  
Note that the ``handle.put`` function does not use the base vector; the base
vector is only subtracted from the loaded vector with ``handle.get``. 
To run this in parallel, the ``put`` loop must be done only on one processor 
as in the previous example. 
The function ``my_DMD.put_decomp``, by default, saves the three decomposition
matrices to text files.
This behavior can be changed by passing the ``DMD`` constructor
the optional argument ``put_mat=`` as a different function to "put" the matrices
in a different way, like to a different file format, or anything else.
See :ref:`sec_matrices`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 5 -- Scaling vectors and using ``VecOperations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You might want to scale all of your vectors by factors, and this can be done
by the vector handle ``get`` function, just like the base vector.
For example, below we show the use of quadrature weights, where each vector
is weighted.
This example also shows how to load vectors in one format (pickle) 
and save modes in another (text).
At the end of this example, we use the lower-level 
:class:`vecoperations.VecOperations` class to check the POD modes are 
orthonormal.

.. literalinclude:: ../examples/tutorial_ex5.py
      
When using both base vector subtraction and scaling, the default order
is first subtraction, then mulitplication: ``(vec - base_vec)*scale``.

The input vectors are saved in pickle format (``MR.PickleVecHandle``) 
and the modes are saved in text format (``MR.ArrayTextVecHandle``).

The last section uses the ``VecOperations`` class, 
which contains most of the parallelization and "heavy-lifting" and is
heavily used by ``POD``, ``BPOD``, and ``DMD``.
It is a good idea to use this class whenever possible since it is tested
and parallelized (see :mod:`vecoperations`).


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 6 -- User-defined vectors and handles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far all of the vectors have been arrays, but you may want to apply modred 
to data saved in your own custom format with more complicated inner 
products and other operations.
This is no problem at all; modred works with any data in any format!
That's worth saying again, **modred works with any data in any format!**
Of course, you'll have to tell modred how to interact with your data, but 
that's pretty easy.
You just need to define and use your own vector handle and vector objects.

There are two important new features of this example: a custom vector class
``CustomVector`` and a custom vector handle class ``CustomVecHandle``.
``CustomVector`` meets the requirements for a vector object: vector addition
``__add__``, scalar multiplication ``__mul__``, and 
compatibility with an inner product function such as
``inner_product(v1, v2)``.
The member function ``inner_product`` is useful, but not required.
``CustomVecHandle`` meets the requirements for a vector handle, defining 
``vec = get()`` and ``put(vec)`` (through inheritance of 
``MR.VecHandle``)

The vector and the grid are all saved to a single pickle file by the 
custom vector class's method, ``CustomVector.save``, which is called by 
``CustomVecHandle.put``.

This example also uses the trapezoidal rule for inner products to account for 
an 3D arbitrary cartesian grid (:py:class:`vectors.InnerProductTrapz`).

.. literalinclude:: ../examples/tutorial_ex6.py

After execution, the modes are saved to ``direct_mode0.pkl``, 
``direct_mode1.pkl`` ... and ``adjoint_mode0.pkl``, 
``adjoint_mode1.pkl``.
The ``CustomVector`` class inherits from ``MR.Vector``, which is recommended
since it provides useful methods.
Similarly, the ``CustomVecHandle`` class inherits from ``MR.VecHandle``.
This is *strongly* recommended since it adds the additional functionality
for subtracting base vectors and scaling in an efficient way. 
The base class ``MR.VecHandle`` defines ``get`` and ``put`` methods which call
the derived class's ``_get`` and ``_put`` methods, as defined in the example.
While you don't need to understand the guts of these base classes, more 
is covered in :ref:sec_details.

When you're ready to start using modred, take a look at what types of 
vectors, file formats, and inner_products we supply in :mod:`vectors`.
If you don't find what you need, we can't stress enough that this is 
no problem at all.
You can define your own vectors and vector handles following examples like this.
Also, classes like ``ArrayTextVecHandle`` and ``PickleVecHandle`` are
good examples because they, in fact, are just like your own 
``CustomVecHandle`` in that they are derived from ``VecHandle``. 
We provide them with modred since they are common cases. 
You're welcome!

As usual, this example can be executed in parallel without any 
modifications.

 

.. _sec_matrices:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Matrix input and output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``put_mat`` and ``get_mat`` save and load to text files.
This tends to be a versatile option because the files are
easy to load into Matlab and other programs, human-readable, portable, and
rarely large enough to be problematic.
However, you can define your own functions ``put_mat`` and ``get_mat`` with 
interfaces, and pass them as optional arguments to the constructors.
For example, you can save in a different file format, or ``get`` and ``put``
the matrices to another class's data member. As long as they
meet the required interfaces: ``put_mat(mat, mat_dest)`` and
``mat = get_mat(mat_source)``, it's all the same to modred.

While these functions' names suggest that they work for numpy matrices, they 
must also accept 1D and 2D arrays as arguments. 
