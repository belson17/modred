============================
Introduction 
============================

Welcome to the modred project!

This is a easy-to-use and parallelized library for finding modal
decompositions and reduced-order models.

Parallel implementations of the proper orthogonal decomposition (POD),
balanced POD (BPOD), and dynamic mode decomposition (DMD) are provided, 
as well as serial implementations of the Observer Kalman filter Identification
method (OKID) and the Eigensystem Realization Algorithm (ERA).
Modred is applicable to a wide range of problems and nearly
any type of data.

For POD, BPOD, and DMD, the library itself is lightweight; the majority of
the computation time is spent calling functions you provide.
The library is essentially a wrapper that calls the functions in an 
efficient way. 

Python's speed limitations can be bypassed by calling compiled code
via tools like Cython, SWIG, and f2py. 
