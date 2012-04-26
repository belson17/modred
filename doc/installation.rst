====================================
Installation and requirements
====================================

These are mandatory requirements for modred:

1. Python 2.x (tested for 2.7), http://python.org/
2. Relatively new version of Numpy (tested for 1.5 and 1.6), http://numpy.scipy.org/

These are optional:

1. For parallel execution, an MPI implementation and mpi4py, http://mpi4py.scipy.org/
2. For plotting within Python, matplotlib, http://matplotlib.sourceforge.net/

To install::

  [sudo] python setup.py install

If you get permissions errors, you may need to change the permissions of the
modred directory. On Posix, the bash command::
 
  chmod 777 modred-*.* 

solves the problem.

To run the unit tests and be sure it's working, run the following from
a directory to which you have read and write permissions. The
parallel tests require mpi4py to be installed::

  python -c 'import modred.tests; modred.tests.run()'
  mpiexec -n 3 python -c 'import modred.tests; modred.tests.run()'
  
Please report test failures or installation problems to bbelson@princeton.edu 
with the following information:

1. Copy of the entire output of the installation/tests
2. Python version (use: python -V)
3. Numpy version (use: python -c 'import numpy; print numpy.__version__')
4. Your operating system


The documentation is available at: http://packages.python.org/modred

You can also build it yourself with Sphinx.

-  Get Sphinx via ``easy_install sphinx`` or from http://sphinx.pocoo.org/    
 
-  From the modred directory, run ``sphinx-build doc doc/_build``. Then
   open doc/_build/index.html in a browser to view the HTML documentation

