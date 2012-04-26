Welcome to the modred library!


Installation
------------

To install::

  [sudo] python setup.py install

You may need to change the permissions on the modred folder, on Posix systems,
use the bash command::

  chmod 777 modred-*.*

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
