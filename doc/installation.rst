====================================
Installation and requirements
====================================

Mandatory:

1. Python 2 (>2.6, tested for 2.7) or Python 3 (>=3.3), http://python.org.

2. Relatively new version of Numpy (>1.5, tested for 1.10),
   http://numpy.scipy.org.

Optional:

1. For parallel execution, an MPI implementation and mpi4py,
   http://mpi4py.scipy.org.


To install::

  [sudo] python setup.py install

To be sure it's working, run the unit tests. 
The parallel tests require mpi4py to be installed::

  python -c 'import modred.tests; modred.tests.run()'
  mpiexec -n 3 python -c 'import modred.tests; modred.tests.run()'
  
Please report problems to modred-discuss@googlegroups.com with the following:

1. Copy of the entire output of the installation and tests
2. Python version (``python -V``)
3. Numpy version (``python -c 'import numpy; print numpy.__version__'``)
4. Your operating system

The documentation is available at http://modred.readthedocs.io.

While unnecessary, you can build the documentation with Sphinx 
(http://sphinx.pocoo.org). 
From the modred directory, run ``sphinx-build doc doc/build`` and then open 
``doc/build/index.html`` in a web browser.

