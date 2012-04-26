Welcome to the modred library!


Installation
------------

To install:
>> [sudo] python setup.py install

You may need to change the permissions on the modred folder, on Posix systems,
use the bash command:
>> chmod 777 modred-*.*

To be sure it's working, run the unit tests from a directory to which you 
have read/write access:
>> python -c 'import modred.tests; modred.tests.run()'

To test the parallel components (requires mpi4py), do:
>> mpiexec -n 3 python -c 'import modred.tests; modred.tests.run()'

Please report test failures to bbelson@princeton.edu with the following 
information:

1. Copy of the entire output of the tests
2. Python version (use >> python -V)
3. Numpy version (use >> python -c 'import numpy; print numpy.__version__')
4. Your operating system



Sphinx Documentation
--------------------

The sphinx-generated HTML documentation is available at
http://packages.python.org/modred

You can also build it yourself.
This is usually quite simple.
The code is primarily documented with reStructuredText docstrings which
Sphinx reads and makes pretty.


-  Get Sphinx 
    Easier way:
      easy_install sphinx
    
    Harder way:
      http://sphinx.pocoo.org/    
      python setup.py build/install
    
-  Build the documentation. From the modred directory, run 
   >> sphinx-build doc doc/_build
   Open doc/_build/index.html in a browser to view the HTML documentation

