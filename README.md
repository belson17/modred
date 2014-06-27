Welcome to the modred library!

This is an easy-to-use and parallelized library for finding modal
decompositions and reduced-order models.

Parallel implementations of the proper orthogonal decomposition (POD),
balanced POD (BPOD), dynamic mode decomposition (DMD), and
Petrov-Galerkin projection are provided, as well as serial
implementations of the Observer Kalman filter Identification method
(OKID) and the Eigensystem Realization Algorithm (ERA). Modred is
applicable to a wide range of problems and nearly any type of data.

For smaller and simpler datasets, there is a Matlab-like
interface. For larger and more complicated datasets, you can provide
modred classes with functions to interact with your data.

This work was supported by a grant from the the National Science
Foundation.

Installation
------------

To install::

  [sudo] python setup.py install

You may need to change the permissions on the modred folder, on Posix systems,
use the bash command::

  chmod 777 modred-*

To check the installation, you can run the unit tests 
(parallel requires mpi4py)::

  python -c 'import modred.tests; modred.tests.run()'
  mpiexec -n 3 python -c 'import modred.tests; modred.tests.run()'

Please report failures and installation problems to bbelson@princeton.edu with
the following information:

1. Copy of the entire output of the tests or installation
2. Python version (``python -V``)
3. Numpy version (``python -c 'import numpy; print numpy.__version__'``)
4. Your operating system

The documentation is available at: http://packages.python.org/modred
