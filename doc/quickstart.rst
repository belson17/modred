=============
Quickstart
=============


-----------------------------------------
Requirements
-----------------------------------------

1. Python 2.x (tested for 2.6 and 2.7) 
2. Relatively new version of Numpy (tested for 1.6), http://numpy.scipy.org/

Optional:

1. For parallel execution, an MPI implementation and mpi4py, http://mpi4py.scipy.org/
2. For plotting within Python, matplotlib, http://matplotlib.sourceforge.net/

Below are some short examples demonstrating basic usage.

-----------------------------------------
Modal decompositions (POD, BPOD, DMD)
-----------------------------------------

First, collect your data. 
We call each piece of data a vector.
**By vector, we don't mean a 1D array**, we mean an element of a vector space.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 1 -- POD with data in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A simple way to use modred to find POD modes is::

  import numpy as N
  import modred as MR
  # We use random data as a placeholder
  num_vecs = 30
  vecs = [N.random.random(50) for i in range(num_vecs)]
  
  my_POD = MR.POD(MR.vecdefs.ArrayInMemory())
  sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vecs)
  num_modes = 10
  modes = my_POD.compute_modes_and_return(range(num_modes))

Let's walk through the important steps.
First, we put our vector data into a list of numpy arrays.
Then we created an instance of ``POD`` called ``my_POD``.
The constructor took the argument
``MR.vecdefs.ArrayInMemory()``, which
is a class instance that contains functions for interacting with
the vectors (numpy arrays). 

The next line, ``compute_decomp_and_return`` computes the correlation matrix 
(often written "X* X"), takes its SVD, and returns the SVD matrices 
(note that the left and right singular vectors
are equal for POD).
The correlation matrix is computed by taking the inner products of all 
combinations of vectors.
We stress that modred's approach is to never form the "X" matrix.
This is explained in later sections.

The last line returns a list of the modes, ``modes``. 
The argument is a list of the mode numbers.

The above example can be run in parallel with *no modifications*.
At the end, each MPI worker (processor) will have all of the modes in its
list, ``modes``.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 2 -- loading/saving data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here's an example that uses the ``cPickle`` module to save/load vectors to
compute the BPOD modes::
  
  import numpy as N
  import modred as MR
  
  # Save your data into pickle files
  # We use random data as a placeholder
  num_vecs = 30
  my_vec_defs = MR.vecdefs.ArrayPickle()
  direct_vec_paths = ['direct_vec%d.pkl'%i for i in range(num_vecs)]
  adjoint_vec_paths = ['adjoint_vec%d.pkl'%i for i in range(num_vecs)]
  
  for direct_vec_path in direct_vec_paths:
      my_vec_defs.put_vec(N.random.random(50), direct_vec_path)
  for adjoint_vec_path in adjoint_vec_paths:
      my_vec_defs.put_vec(N.random.random(50), adjoint_vec_path)
  
  my_BPOD = MR.BPOD(my_vec_defs, max_vecs_per_node=10)
  L_sing_vecs, sing_vals, R_sing_vecs = \
      my_BPOD.compute_decomp_and_return(direct_vec_paths, adjoint_vec_paths)

  # The BPOD modes are saved to disk.
  num_modes = 15
  mode_nums = range(num_modes)  
  my_BPOD.compute_direct_modes(mode_nums, ['direct_mode%d'%i for i in mode_nums])
  my_BPOD.compute_adjoint_modes(mode_nums, ['adjoint_mode%d'%i for i in mode_nums])

Let's walk through the important steps.
Random data is created and saved to pickle files via ``my_vec_defs.put_vec``.
An important difference from the previous example is the use of class
``MR.vecdefs.ArrayPickle()``.
This class contains a function to load vectors from pickle files.
Also, ``my_BPOD.compute_modes`` doesn't return the modes like before, instead
they're saved to pickle files directly.
Simply replacing ``ArrayPickle()`` with ``ArrayText()`` would load/save  
vectors (including the modes) to text files.

Then an instance of ``BPOD`` is created.
We set ``max_vecs_per_node`` to the maximum number of
vectors that can be in memory simultaneously, 10.
Since we can only have 10 vectors in memory at once (per node), it would 
actually be impossible to have all 15 modes returned.

To run this in parallel is very easy.
The only complication is the random data must be saved by only one MPI worker.
Moving those lines inside the following if block solves this::
  
  parallel = MR.parallel.default_instance
  if parallel.is_rank_zero():
      # Loops that call put_pickle
      pass

After this change, the code works in serial too.
It is rare to need to handle parallelization yourself. 
However, if you do, you should use the provided ``parallel`` class instance
as in this example.
Also provided are member functions ``parallel.get_rank()`` and 
``parallel.get_num_MPI_workers()`` (see docs for details).


^^^^^^^^^^^^^^^^^^^^^^^^^
Several more examples
^^^^^^^^^^^^^^^^^^^^^^^^^

All of these work in serial and in parallel (parallel requires mpi4py to be
installed).

Text files and arrays, with a base vector to subtract from each saved vector::

  import modred as MR
  # A base vector to be subtracted off from each vector as it is loaded.
  base_vec = MR.vecdefs.get_vec_text('base_vec.txt')
  
  my_DMD = MR.DMD(MR.vecdefs.ArrayText(base_vec=base_vec))
  # Generate vectors and save them to vec_paths.
  my_DMD.compute_decomp(vec_paths, 'ritz_vals.txt', 'mode_norms.txt', 
      'build_coeffs.txt')
  mode_nums = [1, 4, 0, 2, 10]
  mode_paths = ['mode%02d'%i for i in mode_nums]
  my_DMD.compute_modes(mode_nums, mode_paths)
  

Defining your own vector defintion module::

  #-------- newvecdefs.py ---------
  def get_vec(vec_path):
      # load from your format, return a vec
      pass
  def put_vec(vec, vec_path):
      # save to your format, return nothing
      pass
  def inner_product(vec1, vec2):
      # inner product for your type of data, return a scalar
      pass

  #--------- main_script.py ---------
  import newvecdefs as VD
  # fill this in...
  vec_paths = ['a', 'b'] 
  my_POD = MR.POD(VD)
  sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vec_paths)
  num_modes = 15
  mode_nums = range(num_modes)  
  my_POD.compute_modes(mode_nums, ['mode%d'%i for i in mode_nums])
  

Defining your own vector object and vector definition class::
  
  import modred as MR
  class VectorClass(object):
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
          vec = VectorClass()
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

  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other formats and data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's quite possible you need to define your own inner product or maybe your 
data is already saved in some format.
Modred is designed for arbitrary data and functions, and most of the
examples above are actually just special, common, cases.
When you're ready to start using modred, take a look at what types of 
vectors, file formats, and functions we supply in the ``vecdef`` module.
If you don't find what you need, that's fine; modred works with **any** data
in any format!
That's worth saying again, **modred works with any data in any format!**

The last example is an outline of how you'd write your own functions to
interact with modred.

Of course, you'll have to tell modred how to interact with your data, but 
that's pretty easy and covered in the following sections. 

Extended examples are provided in the examples directory.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions of matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can define ``put_mat(mat, mat_dest)`` and ``mat = get_mat(mat_source)``, 
and pass them as optional arguments to the constructors.
The default is to save and load matrices (real and imaginary) to text files.
This tends to be a good option even for advanced use (e.g. easy loading into
Matlab and other programs).


---------------------------------------
System identification (ERA and OKID)
---------------------------------------
These are fairly straight-forward and the documenation of these algorithms
should be enough to get started quickly.
