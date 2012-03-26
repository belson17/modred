Quickstart
=============

-----------------------------------------
Modal decompositions (POD, BPOD, DMD)
-----------------------------------------
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 1 (data in memory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, collect your data. We call each piece of data a vector.
**By vector, we don't mean a 1D array**, we mean an element of a vector space.

A simple way to do this is to put each vector into its own numpy array, and
to then make a list of numpy arrays.
Then you could find POD modes with::

  import numpy as N
  import modred
  # Collect your data into a list of numpy arrays called "vecs".
  # We use random data as a placeholder
  num_vecs = 30
  vecs = [N.random.random(50) for i in range(num_vecs)]
  
  my_POD = modred.POD(modred.vecdefs.VecDefsArrayInMemory(), 
      max_vecs_per_node=1000)
  my_POD.compute_decomp(vecs)
  num_modes = 10
  # The POD modes are the elements of the "modes" list.
  modes = my_POD.compute_modes(range(1, 1+num_modes), [None]*num_modes)

Let's walk through the important steps.
First, we create an instance of ``POD`` called ``my_POD``.
The first point to see is the call to 
``modred.vecdefs.VecDefsArrayInMemory()``.
This is a predefined class that contains the functions for interacting with
the vectors (which are stored in arrays here). 
One of those functions is the inner product, which in this case is
taken to be ``(vec1*vec2.conj()).sum()``.
(If that's not the right inner product for your vectors, you can write your own, 
but that's covered in a later section.)
Another argument to the POD constructor is ``max_vecs_per_node``.
This is the maximum number of vectors that can be in memory at once.
For larger data this is important, but in this case it isn't, so just set it
to some large number.

The next line calls ``compute_decomp``. 
This computes all of the inner products between each vector in ``vecs`` and
all of the others.
That is, it's finding the correlation matrix often written as "X* X", where
X is a matrix with vectors as columns.
We stress though that it never forms the X matrix. 
This approach, inner products and not matrix multiplication, 
is explained in later sections.
``compute_decomp`` also finds the SVD of the correlation matrix.

The last line returns a list of the modes, ``modes``. 
The first argument to ``compute_modes`` is the mode numbers.
Modes are indexed starting with 1, hence the "+1"s.
The second argument is the mode destinations -- a list whose elements
are the destination where each mode should be stored.
In this case the modes are returned, the argument is ignored and we
fill it with ``None`` s.

The above example can be run in parallel with *no modifications*. 
At the end, each MPI worker (processor) will have all of the modes in the
``modes`` list.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 2 (loading/saving data)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Maybe your data is all saved to disk.
For example maybe it is large and it's impractical to load it all at once. 
Here's an example that uses the ``cPickle`` module to save/load vectors to
compute the BPOD modes (still with the inner product from before)::
  
  import numpy as N
  import modred
  
  # Save your data into pickle files.
  # We use random data as a placeholder
  num_vecs = 30
  direct_vec_paths = ['direct_vec%d.pkl'%i for i in range(num_vecs)]
  adjoint_vec_paths = ['adjoint_vec%d.pkl'%i for i in range(num_vecs)]
  
  for direct_vec_path in direct_vec_paths:
      modred.vecdefs.put_pickle(N.random.random(50), direct_vec_path)
  for adjoint_vec_path in adjoint_vec_paths:
      modred.vecdefs.put_pickle(N.random.random(50), adjoint_vec_path)
  
  my_BPOD = modred.BPOD(modred.vecdefs.VecDefsArrayPickle(), 
      max_vecs_per_node=10)
  my_BPOD.compute_decomp(direct_vec_paths, adjoint_vec_paths)

  # The BPOD modes are saved to disk.
  num_modes = 15
  mode_nums = range(1, 1+num_modes)  
  my_BPOD.compute_direct_modes(mode_nums, ['direct_mode%d'%i for i in mode_nums])
  my_BPOD.compute_adjoint_modes(mode_nums, ['adjoint_mode%d'%i for i in mode_nums])

Let's walk through the important steps.
First some random data is made and saved with ``modred.vecdefs.put_pickle``
to pickle files.
Then an instance of ``BPOD`` is created.
We set ``max_vecs_per_node`` to the maximum number of
vectors that can be in memory simultaneously, 10.
An important difference from the previous example is the
``modred.vecdefs.VecDefsAarrayPickle()``.
This class instance provides the pickle function to read the vectors.
``my_BPOD.compute_modes`` doesn't return the modes, instead they're saved
to file directly.
Since we can only have 10 vectors in memory at once, it would actually be
impossible to have all 15 modes returned!

To run this in parallel is very easy.
The only complication is the random data must be saved by only one MPI worker.
Moving those lines inside the following if block solves this::
  
  parallel = modred.parallel.default_instance
  if parallel.is_rank_zero():
      # Loops that call put_pickle
      pass
       
  
It is rare to need to handle parallelization yourself. 
However, if you do, you should use the provided ``parallel`` class instance
as in this example.
Also provided are member functions ``parallel.get_rank()`` and 
``parallel.get_num_MPI_workers()`` (see docs for details).

^^^^^^^^^^^^^^^^^^
Other formats
^^^^^^^^^^^^^^^^^^
You can save in text format too, and there are other ``vecdef`` classes to
choose from. 
When you're ready to start using modred, take a look at what is available in
the ``vecdefs`` module.
There's a good chance it has something useful for you already written. 


If it doesn't though, that's fine. 
A strength of modred's is that it works with **any** data.
More on that is covered in the next sections. 


---------------------------------------
System identification (ERA and OKID)
---------------------------------------
These are fairly straight-forward and the documenation of these algorithms
should be enough to get started quickly.
