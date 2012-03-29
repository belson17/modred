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
  
  my_POD = MR.POD(MR.vecdefs.ArrayInMemoryUniform())
  sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vecs)
  num_modes = 10
  modes = my_POD.compute_modes_and_return(range(num_modes))

Let's walk through the important steps.
First, we created random vector data and put it into a list of numpy arrays.
Then we created an instance of ``POD`` called ``my_POD``.
The constructor took the argument
``MR.vecdefs.ArrayInMemoryUniform()``, which
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
At the end, each MPI worker (process) will have all of the modes in its
list, ``modes``.
To do this, assuming the above script is saved as ``find_POD.py`:: 
  
  mpiexec -n 8 python find_POD.py


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 2 -- loading/saving data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here's an example that uses the ``cPickle`` module to save/load vectors 
(snapshots) to compute the BPOD modes::
  
  import numpy as N
  import modred as MR
  
  # Save your data into pickle files
  # We use random data as a placeholder
  num_vecs = 30
  my_vec_defs = MR.vecdefs.ArrayPickleUniform()
  direct_snap_paths = ['direct_vec%d.pkl'%i for i in range(num_vecs)]
  adjoint_snap_paths = ['adjoint_vec%d.pkl'%i for i in range(num_vecs)]
  
  for direct_snap_path in direct_snap_paths:
      my_vec_defs.put_vec(N.random.random(50), direct_snap_path)
  for adjoint_snap_path in adjoint_snap_paths:
      my_vec_defs.put_vec(N.random.random(50), adjoint_snap_path)
  
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
``MR.vecdefs.ArrayPickleUniform()``.
This class contains a function to load vectors from pickle files.
Also, ``my_BPOD.compute_modes`` doesn't return the modes like before, instead
they're saved to pickle files directly.
Simply replacing ``ArrayPickleUniform()`` with ``ArrayTextUniform()`` would load/save  
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
      # Loops that call my_vec_defs.put_vec
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
  base_vec_source = 'base_vec.txt'
  
  my_DMD = MR.DMD(MR.vecdefs.ArrayTextUniform(base_vec_source=base_vec_source))
  
  # Generate your own vectors and save them to vec_paths.
  
  my_DMD.compute_decomp(vec_paths, 'ritz_vals.txt', 'mode_norms.txt', 
      'build_coeffs.txt')
  mode_nums = [1, 4, 0, 2, 10]
  mode_paths = ['mode%02d'%i for i in mode_nums]
  my_DMD.compute_modes(mode_nums, mode_paths)

The text files are saved with whitespace after each column entry and
line breaks after each row, so the 2x3 array::
  
  1 2 3
  4 5 6

looks just like this in the text file. See docs for ``util.load_mat_text`` 
and ``util.save_mat_text`` for more functionality. 


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
  import newvecdefs
  # fill this in...
  vec_paths = ['a', 'b'] 
  my_POD = MR.POD(newvecdefs)
  sing_vecs, sing_vals = my_POD.compute_decomp_and_return(vec_paths)
  num_modes = 15
  mode_nums = range(num_modes)  
  my_POD.compute_modes(mode_nums, ['mode%d'%i for i in mode_nums])
  

Defining your own vector object and vector definition class::
  
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

  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other formats and data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's quite possible your data requires a more complicated inner product or is 
already saved in another format.
Modred is designed for arbitrary data and functions, and most of the
examples above are actually just special, common, cases.
When you're ready to start using modred, take a look at what types of 
vectors, file formats, and functions we supply in the ``vecdef`` module.
If you don't find what you need, that's fine; modred works with **any** data
in any format!
That's worth saying again, **modred works with any data in any format!**
Of course, you'll have to tell modred how to interact with your data, but 
that's pretty easy and covered in the following sections. 
(The last example is an outline of how you'd do this.)


A few extended examples are provided in the examples directory.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions of matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can define ``put_mat(mat, mat_dest)`` and ``mat = get_mat(mat_source)``, 
and pass them as optional arguments to the constructors.
By default, ``put_mat`` and ``get_mat`` save and load to text files.
This tends to be a good option even for advanced use (e.g. easy loading into
Matlab and other programs).


---------------------------------------
System identification (ERA and OKID)
---------------------------------------
These are fairly straight-forward and the documentation of these algorithms
should be enough to get started quickly.
