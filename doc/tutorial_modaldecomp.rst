.. _sec_modaldecomp:

-------------------------------------------------
Modal decompositions -- POD, BPOD, and DMD
-------------------------------------------------

First, collect your data. 
We call each piece of data a vector.
**By vector, we don't mean a 1D array**, we mean an element of a vector space.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 1 -- POD with data in memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A simple way to use modred to find POD modes is::

  import numpy as N
  import modred as MR
  num_vecs = 30
  
  # We use arbitrary fake data as a placeholder
  x = N.arange(0, N.pi, 100)
  vecs = [N.sin(x*0.1*i) for i in range(num_vecs)]
  
  my_POD = MR.POD(inner_product=N.vdot)
  sing_vecs, sing_vals = my_POD.compute_decomp_in_memory(vecs)
  num_modes = 10
  modes = my_POD.compute_modes_in_memory(range(num_modes))

Let's walk through the important steps.
First, we created a list of arbitrary arrays; these are the vectors.
Then we created an instance of ``POD`` called ``my_POD``.
The constructor took the argument
``inner_product``, a function that takes two vectors (numpy arrays in this case), and returns
their inner product. 

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
To do this, assuming the above script is saved as ``main_pod.py`:: 
  
  mpiexec -n 8 python main_pod.py



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 2 -- Loading/saving files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here's an example that again uses numpy arrays as the vectors, but this time
loads/saves them to text files::

  import numpy as N
  import modred as MR
  
  num_vecs = 30
  
  direct_snap_handles = [MR.ArrayTextVecHandle('direct_vec%d.txt'%i) 
  for i in range(num_vecs)]
  adjoint_snap_handles = [MR.ArrayTextVecHandle('adjoint_vec%d.txt'%i)
  for i in range(num_vecs)]
  
  # Save arrays in text files
  # We use arbitrary fake data as a placeholder
  x = N.arange(0, N.pi, 10000)
  for i,direct_snap_handle in enumerate(direct_snap_handles):
      direct_snap_handle.put(N.sin(x*0.1*i) for i in range(num_vecs))
  for i,adjoint_snap_handle in enumerate(adjoint_snap_handles):
      adjoint_snap_handle.put(N.cos(x*0.1*i) for i in range(num_vecs))
  
  my_BPOD = MR.BPOD(inner_product=N.vdot, max_vecs_per_node=10)
  L_sing_vecs, sing_vals, R_sing_vecs = \
      my_BPOD.compute_decomp(direct_vec_handles, adjoint_vec_handles)

  # The BPOD modes are saved to disk.
  num_modes = 15
  mode_nums = range(num_modes)  
  direct_mode_handles = [MR.ArrayTextVecHandle('direct_mode%d'%i) for i in mode_nums]
  adjoint_mode_handles = [MR.ArrayTextVecHandle('adjoint_mode%d'%i) for i in mode_nums]
  my_BPOD.compute_direct_modes(mode_nums, direct_mode_handles )
  my_BPOD.compute_adjoint_modes(mode_nums, adjoint_mode_handles)

First, arrays are filled with arbitrary data to serve as the vectors.
Then, we create lists of instances of vector handles, in particular 
the class ``ArrayTextVecHandle``.
The use vector handles is an important difference between the two examples.
Handles are lightweight pointers to a vector. 
In this case, each handle contains a path where a vector is saved. 
They are necessary for large data when we are memory-limited, i.e. cases
where it is impossible or inefficient to have a list of all vectors 
like in the previous example.
Instead, we work with these lightweight handles which save and/or load
vectors when requested via ``vec_handle.put(vec)`` and 
``vec = vec_handle.get()``, respectively.

Returning to the example, we create an instance of ``BPOD`` and specify
``max_vecs_per_node=10``, informing modred we can never have more than 10
vectors (snapshots + modes) loaded at once on one node.
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

To run this in parallel is very easy.
The only complication is the data must be saved by only one MPI worker.
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
Example 3 -- Subtracting a base vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Text files and arrays, with a base vector to subtract from each saved vector::

  import modred as MR
  parallel = MR.parallel.default_instance
  
  num_elements = 2000  
  num_vecs = 100
  # Save fake data. Typically the data already exists from a previous
  # simulation or experiment.
  if parallel.is_rank_zero():
      # A base vector to be subtracted off from each vector as it is loaded.
      base_vec = N.random.random(num_elements)
      base_vec_handle = MR.PickleVecHandle('base_vec.pkl')
      for i in range(num_vecs):
          MR.PickleVecHandle('vec%d.pkl'%i).put(N.random.random(num_elements))
  
  vec_handles = [MR.PickleVecHandle('vec%d.pkl'%i, base_handle=base_vec_handle)
      for i in range(num_vecs)]

  my_DMD = MR.DMD(inner_product=N.vdot)  
  my_DMD.compute_decomp(vec_paths)
  my_DMD.put_decomp('ritz_vals.txt', 'mode_norms.txt', 'build_coeffs.txt')
  mode_nums = [1, 4, 5, 2, 10]
  mode_handles = [MR.PickleVecHandle('mode%d.pkl'%i) for i in mode_nums]
  my_DMD.compute_modes(mode_nums, mode_handles)
  
To run this in parallel, the ``put`` must be done only on one processor,
see the previous example. 




^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 4 -- Scaling the vectors by a constant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The vector handles can also automatically scale the vectors as they ``get`` 
them::

  import numpy as N
  import modred as MR
  num_elements = 2000
  
  # A scaling
  scale = N.pi
  
  num_vecs = 100
  # Save fake data. Typically the data already exists from a previous
  # simulation or experiment.
  if parallel.is_rank_zero():
      for i in range(num_vecs):
          MR.PickleVecHandle('vec%d.pkl'%i).put(N.random.random(num_elements))
  
  vec_handles = [MR.PickleVecHandle('vec%d.pkl'%i, scale=scale)
      for i in range(num_vecs)]

  my_POD = MR.POD(inner_product=N.vdot)  
  my_POD.compute_decomp(vec_handles)
  my_POD.put_decomp('ritz_vals.txt', 'mode_norms.txt', 'build_coeffs.txt')
  mode_nums = [1, 4, 5, 2, 10]
  mode_handles = [MR.ArrayTextVecHandle('mode%d.txt'%i) for i in mode_nums]
  my_POD.compute_modes(mode_nums, mode_handles)
  
  # Check that modes are orthonormal
  my_vec_ops = MR.VecOperations(inner_product=N.vdot)
  IP_mat = my_vec_ops.compute_symmetric_inner_product_mat(mode_handles)
  if not N.allclose(IP_mat, N.eye(len(mode_nums))):
      print 'Modes are not orthonormal'
      

When using both base vector subtraction and scaling, note that the default order
is subtraction, then mulitplication: ``(vec - base_vec)*scale``.
The example demonstrates that the vector handles can be different for the
input vectors and output modes.
Here, the input vectors are saved in pickle format (``MR.PickleVecHandle``) 
and the modes are saved
in text format (``MR.ArrayTextVecHandle``).

The last section uses the ``VecOperations`` class, 
which is a heavy-lifting, parallelized class that ``POD, BPOD,`` and ``DMD``
mostly call.
It is a good idea to use this class whenever possible since it is tested
and parallelized (see :mod:`vecoperations`).


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 5 -- User-defined vector and non-uniform grids
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far all of the vectors have been arrays, but this is not required or even
suggested for some cases.
In this example, the grid is allowed to be a 3D arbitrary cartesian grid and
the inner products are computed via the 2nd-order accurate trapezoidal rule.
The vector and the grid are all saved to a single pickle file by the 
custom vector class, ``CustomVector``::

  import modred as MR
  import numpy as N
  import cPickle
  class CustomVector(MR.Vector):
      def __init__(self, path=None):
          if path is not None:
              self.load(path)
          self.my_trapz_IP = None
      def load(self, path):
          file_id = open(path, 'rb')
          self.x, self.y, self.z = cPickle.load(file_id)
          self.data_array = cPickle.load(file_id)
          file_id.close()
      def save(self, path):
          file_id = open(path, 'wb')
          cPickle.dump((self.x, self.y, self.z), file_id)
          cPickle.dump(self.data_array, file_id)
          file_id.close()
      def copy(self):
          """Returns a copy of self"""
          from copy import deepcopy
          return deepcopy(self)
      def __add__(self, other):
          # Return a new object that is the sum of self and other
          sum_vec = self.copy()
          sum_vec.data_array = self.data_array + other.data_array
          return sum_vec
      def __mul__(self, scalar):
          # Return a new object that is "self * scalar"
          mult_vec = self.copy()
          mult_vec.data_array = mult_vec.data_array*scalar
      def inner_product(self, other):
          if self.my_trapz_IP is None:
              self.my_trapz_IP = MR.InnerProductTrapz(self.x, self.y, self.z)
          return self.my_trapz_IP(self.data_array, other.data_array)
          
  def inner_product(v1, v2):
      return v1.inner_product(v2)
      
  # Set vec handles
  vec_handles = [CustomVecHandle(vec_path='existing_vec%d.pkl'%i,
      scale=2.5) for i in range(10)]
  
  my_POD = MR.POD(inner_product=inner_product)
  sing_vecs, sing_vals = my_POD.compute_decomp(vec_handles)
  num_modes = 5
  mode_nums = range(num_modes)  
  mode_handles = [CustomVecHandle('mode%d.pkl'%i) for i in mode_nums] 
  my_POD.compute_modes(mode_nums, mode_handles)

After execution, the modes are saved to ``mode0.pkl, mode1.pkl`` ...
The imporant part of this example is the ``CustomVector`` class, which
inherits from ``MR.Vector`` (strongly recommended).
``CustomVector`` meets the requirements for a vector object: addition,
``__add__``, 
multiplication, ``__mul__``, and compatibility with the inner product function
``inner_product(v1, v2)``.
The other member functions of ``CustomVector`` (``save``, ``load``, etc.)
are useful, but not required.
(This vector object could be modified to work for arbitrary numbers of
dimensions by replacing the tuple ``(self.x, self.y, self.z)`` with 
``self.grids`` and ``*self.grids`` in constructor ``MR.InnerProductTrapz``.)

This code can be executed in parallel without any modifications.
 


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example 6 -- Working with arbitrary data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to apply modred to data 
which is saved in your own custom format and has more complicated inner 
products and other operations.
This is no problem at all; modred works with **any** data in any format!
That's worth saying again, **modred works with any data in any format!**
Of course, you'll have to tell modred how to interact with your data, but 
that's pretty easy.
You just need to define and use your own vector handle and vector objects.
Here's an example::

  import modred as MR
  class CustomVector(MR.Vector):
      def __init__(self, path=None):
          if path is not None:
              self.load(path)
      def load(self, path):
          pass # Load data from disk
      def save(self, path):
          pass # Save data to disk
      def inner_product(self, other_vec):
          pass # Take inner product of self with other_vec
      def __add__(self, other):
          pass # Return a new object that is the sum of self and other
      def __mul__(self, scalar):
          pass # Return a new object that is "self * scalar"
          
  def inner_product(v1, v2):
      return v1.inner_product(v2)

  class CustomVecHandle(MR.VecHandle):
      def __init__(self, vec_path, base_handle=None, scale=None):
          VecHandle.__init__(self, base_handle, scale)
          self.vec_path = vec_path
      def _get(self):
          return CustomVector(self.vec_path)
      def _put(self, vec):
          vec.save(self.vec_path)
  
  # Set vec handles
  base_handle = CustomVecHandle(vec_path='existing_base_vec.ext')
  vec_handles = [CustomVecHandle(vec_path='existing_vec%d.ext'%i,
      base_handle=base_handle) for i in range(10)]
  
  my_POD = MR.POD(inner_product=inner_product)
  sing_vecs, sing_vals = my_POD.compute_decomp(vec_handles)
  num_modes = 5
  mode_nums = range(num_modes)  
  mode_handles = [CustomVecHandle('mode%d.ext'%i) for i in mode_nums] 
  my_POD.compute_modes(mode_nums, mode_handles)

After execution, the modes are saved to ``mode0.ext, mode1.ext`` ...
The important part of this example is the ``CustomVecHandle`` class, 
which
inherits from ``MR.VecHandle`` (*strongly* recommended), and the implementation
of the ``_get`` and ``_put`` member functions. 
All vector handles that inherit from ``MR.VecHandle``
must have member functions ``_get`` and ``_put`` with interfaces:
``vec = _get()`` and ``_put(vec)``. 
This code can be executed in parallel without any modifications.

When you're ready to start using modred, take a look at what types of 
vectors, file formats, and inner_products we supply in the ``vectors`` module.
If you don't find what you need, we can't stress enough that this is 
no problem at all.
You can define your own vectors and vector handles following this example, or
the others in the examples directory.
For a more thorough discussion of the details, read this section: 
:ref:`sec_details`.



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions of matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can define ``put_mat(mat, mat_dest)`` and ``mat = get_mat(mat_source)``, 
and pass them as optional arguments to the constructors.
By default, ``put_mat`` and ``get_mat`` save and load to text files.
This tends to be a versatile option even for advanced use because the files are
easy to load into Matlab and other programs, human-readable, portable, etc.
The matrices are rarely large enough that the inefficiency of text format
is problematic.


