=============
Release notes
=============

------------
modred 0.3.2
------------
The main change is a bug fix in :py:meth:`util.lsim`.

**New features and improvements**

None

**Bug fixes**

* Function :py:meth:`util.lsim`, which is only provided for convenience (not used) 
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
