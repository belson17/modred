================
Why modred?
================

Modred's easy-to-use interface, reliability, and parallelization
make it a good choice for beginners, experts, 
experimentalists, and computationalists from many fields.
If you don't know Python (it's like Matlab), learning
enough to use modred will save you time in the long run.
And even if you're an expert in model reduction and programming, 
writing software takes significant time that could be spent elsewhere.
The main advantages of modred are summarized below.

**Ease of use**

The interface is high-level and object-oriented, making
it easy to use.
The code is written-to-be-read, open source, and well documented.
For custom data formats, you provide functions to interface
with the data, which is also simple. 
In almost all cases, modred can be run in parallel (MPI) with *no changes* to
the code. 
The functions which interface with your particular data format should be 
written with no parallel consideration; modred
does it for you.



**Several algorithms included**

Parallel implementations of the proper orthogonal decomposition (POD),
balanced POD (BPOD), and dynamic mode decomposition (DMD) are provided, 
as well as serial implementations of the Observer Kalman Filter Identification
method (OKID) and the Eigensystem Realization Algorithm (ERA).
Once you create simple functions 
to interface with your data, you can easily use all of these methods.

Modred can be easily extended to other methods you might like to use.


**Applicable to your data**

Since you provide the functions which interface with your data,
modred can handle any data format.
For the common case of arrays, there are very simple classes to use.
It is also possible to call existing functions in
other languages such as C/C++, Fortran, Java, and Matlab with tools like Cython, 
SWIG, f2py, and mlabwrap, thus eliminating the need
to translate existing code into Python.


**Computational speed**

For modal decompositions POD, BPOD, and DMD, the library itself is lightweight;
the majority of the computation time is spent calling functions you provide.
The library calls the functions in an efficient way. 
If Python's speed limitations become problematic (large data), they
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
If you're curious about this, please contact Brandt Belson at
bbelson@princeton.edu.
For now, one work-around is to use "fat" nodes with large amounts
of memory (RAM). 



