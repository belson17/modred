=============
Release notes
=============


------------
modred 2.0.4
------------

Fixed bug for Windows environments.  Updated documentation to use Read the Docs.

**Bug fixes**

* ``parallel`` modules now uses ``socket`` module instead of ``os`` module to
  find hostname, which is portable to Windows environments.


------------
modred 2.0.3
------------

Minor bug fix to OKID tests.

**Bug fixes**

* OKID files needed for tests are included in the source distribution. All
  doc files are also included.


------------
modred 2.0.2
------------

Minor bug fix.

**Interface changes**

* Order of returned values for :py:func:`dmd.compute_DMD_matrices_snaps_method`
  and :py:func:`dmd.compute_DMD_matrices_direct_method` is now consistent for
  both values of ``return_all``.


------------
modred 2.0.1
------------

Minor bug fix.

**Bug fixes**

* :py:func:`dmd.DMDHandles.compute_spectrum` now returns real numbers, as it
  should have before, instead of complex values.


------------
modred 2.0.0
------------

Main changes are an updated interface for DMD that matches the latest theory
and support for Python 3.  Python 3 support was primarily implemented by Pierre
Augier (pa371 [-at-] damtp [-dot-] cam [-dot-] ac [-dot-] uk). Thanks, Pierre!

**New features and improvements**

* Python 3 is now supported!

* Documentation has been updated for clarity and consistency, and example code
  works with the latest interface.

* DMD implementation now matches newest theory, laid out in a 2014 paper by Tu
  et al. in the Journal of Computational Dynamics.  Features were only added,
  i.e., none were removed.  Any DMD computations previously done using modred
  can be reproduced, though the names of some function calls have changed.
  Namely, :py:func:`dmd.DMDHandles.compute_proj_modes` replaces
  :py:func:`dmd.DMDHandles.compute_modes`, and
  :py:func:`dmd.DMDHandles.put_eigvals` replaces
  :py:func:`dmd.DMDHandles.put_ritz_vals`.  Generally, the term "projected
  modes" has replaced "modes," and similarly "eigenvalues" has replaced "Ritz
  values."  "Exact modes" are now availble in addition to the projected modes.

  A full list of the new functions consists of:
  :py:func:`dmd.DMDHandles.compute_exact_modes`,
  :py:func:`dmd.DMDHandles.compute_proj_modes`,
  :py:func:`dmd.DMDHandles.compute_spectrum`,
  :py:func:`dmd.DMDHandles.compute_proj_coeffs`,
  :py:func:`dmd.DMDHandles.compute_eigendecomp`,
  :py:func:`dmd.DMDHandles.put_spectral_coeffs`,
  and :py:func:`dmd.DMDHandles.put_eigvals`.

* The ``compute_decomp`` step in DMD has been refactored, resulting in the new
  method :py:func:`dmd.DMDHandles.compute_eigendecomp`. This method can be used
  to restart DMD computations from saved correlation and cross-correlation
  matrices, or to compute a DMD using a truncated basis.

* Absolute and relative tolerances can now be passed in using the keyword
  arguments ``atol`` and ``rtol``, respectively, when calling
  ``compute_decomp`` in either POD, BPOD, or DMD.  These are then passed on into
  internal computations of singular value decompositions or eigendecompositions
  of positive definite matrices.  They allow the user to filter out singular
  values or eigenvalues that should be considered numerical artifacts.  They can
  also be used to truncate the computations and limit the number of modes making
  up the decompositions.

* In DMD, truncation can also be achieved by setting the keyword
  argument ``max_num_eigvals`` in  either
  :py:func:`dmd.DMDHandles.compute_decomp` or
  :py:func:`dmd.DMDHandles.compute_eigendecomp`.

* Added new methods that compute the projection of the original data vectors
  onto the modes, for POD, BPOD, and DMD, respectively:
  :py:func:`pod.PODHandles.compute_proj_coeffs`,
  :py:func:`bpod.BPODHandles.compute_proj_coeffs`,
  :py:func:`bpod.BPODHandles.compute_adj_proj_coeffs`,
  :py:func:`dmd.DMDHandles.compute_proj_coeffs`.

**Bug fixes**

* Fixed minor bug in the function ``util.impulse``.

* Fixed minor bug in ``testvectorspace.py``

* Fixed minor bugs in loading/saving test files, some related to delimiters.

* Fixed bug in ``testutil`` where ``eig_biorthog`` was assuming the wrong number
  of return values.

* Fixed minor bugs in DMD tests related to casting of matrices/arrays.

**Interface changes**

* Changed the returned values in
  :py:func:`dmd.compute_DMD_matrices_snaps_method`,
  :py:func:`dmd.compute_DMD_matrices_direct_method`,
  :py:func:`dmd.DMDHandles.compute_decomp`.

* Changed the order of the returned values in
  :py:func:`pod.PODHandles.compute_decomp`.
  :py:func:`bpod.BPODHandles.compute_decomp`.

* Changed the order of the arguments in
  :py:func:`pod.PODHandles.get_decomp`,
  :py:func:`pod.PODHandles.put_decomp`,
  :py:func:`bpod.BPODHandles.get_decomp`,
  :py:func:`bpod.BPODHandles.put_decomp`, and
  :py:func:`era.ERA.put_decomp`.

* Changed the arguments to
  :py:func:`dmd.DMDHandles.get_decomp` and
  :py:func:`dmd.DMDHandles.put_decomp`.

* Added the following new methods that compute projections onto modes:
  :py:func:`pod.PODHandles.compute_proj_coeffs`,
  :py:func:`bpod.BPODHandles.compute_proj_coeffs`,
  :py:func:`bpod.BPODHandles.compute_adj_proj_coeffs`, and
  :py:func:`dmd.DMDHandles.compute_proj_coeffs`.

* Added the following new methods that save projection coefficients:
  :py:func:`pod.PODHandles.put_proj_coeffs`,
  :py:func:`bpod.BPODHandles.put_direct_proj_coeffs`,
  :py:func:`bpod.BPODHandles.put_adjoint_proj_coeffs`, and
  :py:func:`dmd.DMDHandles.put_proj_coeffs`.

* Added the following new methods in the updated ``DMDHandles`` class:
  :py:func:`dmd.DMDHandles.compute_exact_modes`,
  :py:func:`dmd.DMDHandles.compute_spectrum`,
  :py:func:`dmd.DMDHandles.compute_eigendecomp`,
  :py:func:`dmd.DMDHandles.put_R_low_order_eigvecs`,
  :py:func:`dmd.DMDHandles.put_L_low_order_eigvecs`,
  :py:func:`dmd.DMDHandles.put_correlation_mat_eigvals`,
  :py:func:`dmd.DMDHandles.put_correlation_mat_eigvecs`,
  :py:func:`dmd.DMDHandles.put_cross_correlation_mat`, and
  :py:func:`dmd.DMDHandles.put_spectral_coeffs`.

* :py:func:`dmd.DMDHandles.compute_proj_modes` replaces
  :py:func:`dmd.DMDHandles.compute_modes`.

* :py:func:`dmd.DMDHandles.put_eigvals` replaces
  :py:func:`dmd.DMDHandles.put_ritz_vals`.

* :py:func:`dmd.DMDHandles.put_build_coeffs` and
  :py:func:`dmd.DMDHandles.put_mode_norms` are now deprecated.

* Optional ``atol`` and ``rtol`` arguments were added to
  :py:func:`pod.PODHandles.compute_decomp`,
  :py:func:`bpod.BPODHandles.compute_decomp`,
  :py:func:`dmd.DMDHandles.compute_decomp`.

* Optional ``max_num_eigvals`` argument added to
  :py:func:`dmd.DMDHandles.compute_decomp`.

* ``util.svd``, ``util.eigh``, and ``util.eig_biorthog`` now consistently return
  numpy matrices.  Previously, the SVD method returned matrices but the
  eigendecompositions returned arrays.

**Internal changes**

* In DMD, the build coefficients are no longer considered part of the
  decomposition and are no longer saved as internal attributes.  Instead, its
  constituent parts define the decomposition (and are saved as internal
  attributes).  Thus computation of the build coefficients in DMD has been moved
  from the ``compute_decomp`` method to the ``compute_exact_modes`` and
  ``compute_proj_modes`` methods, respectively, which makes more sense
  mathematically.

* Added :py:func:`util.eig_biorthog` method to compute both left and right
  eigenvectors of a matrix, scaled to yield a biorthogonal set.

* Added optional ``atol`` and ``rtol`` arguments to :py:func:`util.svd` and
  :py:func:`util.eigh`.

* Updated tests for ``util.svd`` and ``util.eigh``.  Properties of the
  decompositions are now checked, rather than simply duplicating the
  computations using built-in numpy methods.  This allows for better testing of
  truncated decompositions.  Truncation levels are determined during testing, to
  ensure that truncation actually occurs and is tested.

* Updated tests for ``util.biorthog`` to reduce number of failures.  Some
  failures are to be expected due to the fact that we test on random data, but
  these are much less frequent now.

* Changed how positive definite matrices are generated for use as inner product
  weight matrices.  Previous implementation led to failed tests.

* Changed default delimiter when loading test arrays to ``None``.

* Improved type checking to allow for any iterable container, not just lists.

* Removed dependencies on ``util.make_list`` where possible.

* Removed some duplicate code in ``util`` module, where ``eig_biorthog`` had
  been implemented twice.

* The packaging has been improved.

* Ported to python >= 3.3 using `python-future <http://python-future.org/>`_.

* Replaced instances of ``xrange`` with ``range`` for compatability with Python
  3.  (In Python 3, ``xrange`` has been renamed as ``range``.) This is not as
  efficient in Python 2, but only occurs in a few places and with small enough
  loops that the impact should be negligible.

* Added a few more checks for ``None`` values, as Python 3 doesn't allow
  comparisons of floats to ``None``.


------------
modred 1.0.2
------------
We increased the speed of the BPOD implementations.

**New features and improvements**

* None

**Bug fixes**

* None

**Interface changes**

* None

**Internal changes**

* BPOD classes now compute fewer inner products. The number of inner products
  is now the sum of the number of direct vectors and the number of adjoint
  vectors, whereas previously it was the product. This is achieved by taking
  advantage of a property of the adjoint.


------------
modred 1.0.1
------------
Small changes mostly related to examples.

**New features and improvements**

* None

**Bug fixes**

* Changed a tutorial example.

**Interface changes**

* None

**Internal changes**

* None


------------
modred 1.0.0
------------
Many interface changes including new classes and functions for different
sized data.

**New features and improvements**

* New functions and classes for data that fits entirely on one node's memory.
  These are
  :py:func:`pod.compute_POD_matrices_snaps_method`,
  :py:func:`pod.compute_POD_matrices_direct_method`,
  :py:func:`bpod.compute_BPOD_matrices`,
  :py:func:`dmd.compute_DMD_matrices_snaps_method`,
  :py:func:`dmd.compute_DMD_matrices_direct_method`,
  :py:class:`ltigalerkinproj.LTIGalerkinProjectionMatrices`, and
  :py:class:`vectorspace.VectorSpaceMatrices`.
  These replace the ``in_memory`` member functions and improve
  computational efficiency for small data.

* Added balanced truncation :py:meth:`util.balanced_truncation`.

**Bug fixes**

* None

**Interface changes**

* The old classes ``POD``, ``BPOD``, ``DMD``,
  are now only for large data and have their names appended with "``Handles``".

* Old classes ``LTIGalerkinProjection``, and ``VectorSpace``
  have been split into two, and names appended with "``Matrices``" and
  "``Handles``".

* All ``in_memory`` member functions have been removed, replaced by
  the functions and classes above.

* Removed the ``index_from`` optional argument in ``compute_modes`` functions.
  Mode numbers are now always indexed from zero and are renamed mode indices.

* The ``VectorSpace`` member function ``compute_modes`` has
  been removed and its functionality moved to ``lin_combine``.

* ``LTIGalerkinProjection`` member function ``compute_model`` uses the
  result of an operator on a vector,
  rather than the operator itself. See
  :py:meth:`ltigalerkinproj.LTIGalerkinProjectionHandles.compute_model`.
  The operator classes have been removed.

**Internal changes**

* OKID now uses least squares instead of a pseudo-inverse for improved numerical
  stability.

* Added :py:class:`util.InnerProductBlock` for testing.


------------
modred 0.3.2
------------
The main change is a bug fix in :py:meth:`util.lsim`.

**New features and improvements**

None

**Bug fixes**

* Function :py:meth:`util.lsim`, which is only provided for the user's
  convenience, is simplified and corrected.

**Interface changes**

* :py:meth:`util.lsim`.

**Internal changes**

None


------------
modred 0.3.1
------------
The main change is a bug fix in the ``numpy.eigh`` wrapper,
:py:meth:`util.eigh`.

**New features and improvements**

None

**Bug fixes**

* The POD and DMD classes now use :py:meth:`util.eigh` with the
  ``is_positive_definite`` flag set to ``True``.  This eliminates the
  possibility of small negative eigenvalues that sometimes appear due to
  numerical precision which led to errors.

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
  for LTI Galerkin projections. Replaces and generalizes old class
  ``BPODLTIROM``.

* Improved print messages to print every 10 seconds and be more informative.

**Bug fixes**

* Corrected small error in symmetric inner product matrix calculation (used
  by POD and DMD) where some very small matrix entries were double the true
  value.

* Fixed race condition in :py:meth:`vectorspace.VectorSpace.lin_combine` by
  adding a barrier.

**Interface changes**

* Removed class ``BPODLTIROM``.

* Changed order of indices in Markov parameters returned by
  :py:meth:`okid.OKID`.

* Changed all uses of ``hankel`` to ``Hankel`` to be consistent with naming
  convention.

**Internal changes**

* Added :py:meth:`parallel.Parallel.call_and_bcast` method to ``Parallel``
  class.

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
