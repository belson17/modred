=============
Release notes
=============

------------
modred 1.0.0
------------
Many interface changes, including new classes for different sized data.

**New features and improvements**

* New classes for data that fits entirely on one node's memory. These are
  :py:class:`pod.PODArrays`, :py:class:`bpod.BPODArrays`, 
  :py:class:`dmd.DMDArrays`, 
  :py:class:`ltigalerkinproj.LTIGalerkinProjectionArrays`, and
  :py:class:`vectorspace.VectorSpaceArrays`.
  These replace the ``in_memory`` member functions and improve 
  computational efficiency for small data.

* Balance model provided in :py:mod:`vectorspace`.


**Bug fixes**

* None


**Interface changes**

* The ``POD``, ``BPOD``, ``DMD``, ``LTIGalerkinProjection``, and
  ``VectorSpace`` classes have 
  each been split in two for different size data. 
  The class names are appended with "``Arrays``" and "``Handles``".

* All ``in_memory`` member functions have been removed, replaced by 
  classes above.

* Removed the ``index_from`` optional argument in ``compute_modes`` functions. 
  Mode numbers are now always indexed from zero and are renamed mode indices.

* The `VectorSpace` member function ``compute_modes`` has
  been removed and its functionality moved to ``lin_combine``.

* Function ``compute_model`` requires the result of an operator on a vector, 
  rather than the operator itself. For example, see 
  :py:meth:`ltigalerkinproj.LTIGalerkinProjectionHandles.compute_model`.
  The operator classes have been removed.


**Internal changes**

* OKID now uses least squares instead of a pseudo-inverse for improved numerical
  stability. 

* Base classes :py:class:`pod.PODBase`, :py:class:`bpod.BPODBase`, 
  :py:class:`dmd.DMDBase`, and :py:class:`ltigalerkinproj.LTIGalerkinProjBase`
  contain common functionality for the arrays and handles derived classes. 

* Added :py:class:`util.InnerProductBlock` for testing.



------------
modred 0.3.2
------------
The main change is a bug fix in :py:meth:`util.lsim`.

**New features and improvements**

None

**Bug fixes**

* Function :py:meth:`util.lsim`, which is only provided for the user's convenience, 
  is simplified and corrected.

**Interface changes**

* :py:meth:`util.lsim`.


**Internal changes**

None


------------
modred 0.3.1
------------
The main change is a bug fix in the ``numpy.eigh`` wrapper, :py:meth:`util.eigh`.

**New features and improvements**

None

**Bug fixes**

* The POD and DMD classes now use :py:meth:`util.eigh` with the 
  ``is_positive_definite`` flag 
  set to ``True``.  This eliminates the possibility of small negative eigenvalues
  that sometimes appear due to numerical precision which led to errors.

**Interface changes**

None


**Internal changes**

* Function :py:meth:`util.eigh` now has a flag for positive definite matrices.  
  When
  ``True``, the function will automatically adjust the tolerance such that only
  positive eigenvalues are returned.


------------
modred 0.3.0
------------

**New features and improvements**

* New class :py:class:`ltigalerkinproj.LTIGalerkinProjection`
  for LTI Galerkin projections. Replaces and generalizes old class ``BPODLTIROM``.

* Improved print messages to print every 10 seconds and be more informative.

**Bug fixes**

* Corrected small error in symmetric inner product matrix calculation (used
  by POD and DMD) where some very small matrix entries were double the true 
  value. 

* Fixed race condition in :py:meth:`vectorspace.VectorSpace.lin_combine` by adding
  a barrier.
  
**Interface changes**

* Removed class ``BPODLTIROM``.

* Changed order of indices in Markov parameters returned by :py:meth:`okid.OKID`.

* Changed all uses of ``hankel`` to ``Hankel`` to be consistent with naming 
  convention.
  
**Internal changes**

* Added :py:meth:`parallel.Parallel.call_and_bcast` method to ``Parallel`` class.

* Changed interface of :py:meth:`helper.add_to_path`.

* :py:class:`dmd.DMD` no longer uses an instance of :py:class:`pod.POD`. 

* The equals operator of vector handles now better deals with vectors which
  are numpy array objects.


------------
modred 0.2.1
------------

No noteworthy changes from v0.2.0, figuring out pypi website.

------------
modred 0.2.0
------------

First publicly available version.
