
============================
Introduction 
============================

Welcome to the modred project!

This is a easy-to-use, multi-purpose, and parallelized library for finding modal
decompositions and reduced-order models.

Parallel implementations of the proper orthogonal decomposition (POD),
balanced POD (BPOD), and dynamic mode decomposition (DMD) are provided, 
as well as serial implementations of OKID and ERA.
Modred is applicable to a wide range of problems and any type of data 
because you supply the way (via functions and objects) the library handles your data.

For POD, BPOD, and DMD, the library itself is light-weight; the majority of
the computation time is spent calling the functions you provide.
Python plays well with other languages (C/C++ via Cython and SWIG, Fortran via f2py, etc), 
so for large data, Python's speed limitations can be bypassed.
The library can be thought of as a smart wrapper that calls the 
functions in an efficient manner. 

For OKID and ERA, efficiency is not much of a concern. 
These methods are implemented in a general way to handle nearly any data.

