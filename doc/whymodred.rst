================
Why modred?
================

Modred's a good choice for beginners, experts, 
experimentalists, and computationalists from many fields.
The main advantages of modred are summarized below.
If you don't know Python, it's a terrific programming language with 
similarities to Matlab.


**Ease of use**

For smaller and simpler data, often only a single function call with
a Matlab-like interface is needed.
For larger and more complicated data, there is a high-level 
object-oriented interface.
The code is written-to-be-read, open source, and well documented.
In almost all cases, modred can be run in parallel (MPI) with *no changes* to
the code. 


**Several algorithms included**

Parallel implementations of the proper orthogonal decomposition (POD),
balanced POD (BPOD), dynamic mode decomposition (DMD), and 
Petrov-Galerkin projection are provided, 
as well as serial implementations of the Observer Kalman Filter Identification
method (OKID) and the Eigensystem Realization Algorithm (ERA).
It is easy to switch between methods.

Modred can be easily extended to other methods you might like to use.


**Applicable to your data**

For the common case of data stacked in arrays, there are simple, Matlab-like, 
functions to use.
For larger and more complicated data, you can provide the functions 
which interface with your data format.
These functions should be written with no parallel consideration; modred
does the parallelization for you.
It is also possible to call existing functions in
other languages such as C/C++, Fortran, Java, and Matlab with tools like Cython, 
SWIG, f2py, and mlabwrap, thus eliminating the need
to translate existing code into Python.


**Computational speed**

The library essentially efficiently orchestrates calls to numpy functions 
and/or functions you provide with little added overhead.
If Python's speed limitations become problematic, they
can be bypassed by calling compiled code via tools like Cython, SWIG, and f2py. 

Further, it is parallelized for a distributed memory architecture 
via MPI and the mpi4py module.
The scaling of speedup/processors is better-than-linear up to several
hundred processors, if not more. 

The other methods, ERA and OKID, are typically not computationally demanding. 


**Reliable**

Each individual function is unit-tested independently, and the results 
are trustworthy.
Modred has been used for a few different real-world, complicated, datasets
already with success.


**Limitations**

The biggest limitation is for datasets so large that it is impossible to
have three elements in one node's memory at once. 
The parallelization doesn't break up individual pieces of
data for you, i.e. it doesn't do domain decomposition for you (by design).
However, modred could be extended so you can 
provide parallelized functions, allowing arbitrarily large
data.
If you're curious about this, contact Brandt Belson at
bbelson@princeton.edu.
For now, a work-around is to use "fat" nodes with large amounts
of memory (RAM). 



