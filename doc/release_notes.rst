=============
Release notes
=============

These release notes will tell you what is new about each new modred release.
This includes new features, bug fixes, and interface changes that may affect
your scripts.  It will also include some details about internal changes that
were made, which will be useful to anyone interested in modred's development.


------------
modred 0.3.1
------------
Version 0.3.1 is a minor release.
The main change is a bug fix in the eigh wrapper.

**New features**

(None)

**Bug fixes**

* The POD routine now calls the eigh wrapper with the is_positive_definite flag \
  set to True.  This eliminates the possibility of negative eigenvalues, which
  lead to errors later in the algorithm.

* Similar to above, the DMD routine now calls the eigh wrapper with the
  is_positive_definite flag set to True.

**Interface changes**

(None)


**Internal changes**

* The eigh wrapper routine now has a flag for positive definite matrices.  When
  True, the eigh routine will automatically adjust the tolerance such that only
  positive eigenvalues are returned.


------------
modred 0.3.0
------------
Some stuff about 0.3.0.

------------
modred 0.2.1
------------
Some stuff about 0.2.1.

------------
modred 0.2.0
------------
Some stuff about 0.2.0.
