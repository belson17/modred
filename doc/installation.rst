====================================
Installation and requirements
====================================

These are mandatory requirements for modred:

1. Python 2.x (tested for 2.6 and 2.7), http://python.org/
2. Relatively new version of Numpy (tested for 1.6), http://numpy.scipy.org/

These are optional:

1. For parallel execution, an MPI implementation and mpi4py, http://mpi4py.scipy.org/
2. For plotting within Python, matplotlib, http://matplotlib.sourceforge.net/

To install::

  [sudo] python setup.py install

To run the unit tests and be sure it's working, run the following from
a directory to which you have read and write permissions::

  python -c 'import modred.tests; modred.tests.run()'

To test the parallel components (requires mpi4py), do::
  
  mpiexec -n 3 python -c 'import modred.tests; modred.tests.run()'

Please report test failures to bbelson@princeton.edu with the following 
information:

1. Copy of the entire output of the tests
2. Python version (use: python -V)
3. Numpy version (use: python -c 'import numpy; print numpy.__version__')
4. Your operating system

