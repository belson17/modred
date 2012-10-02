#!/usr/bin/env python

from sys import version
# Check out unittest2, maybe use older pythons. Not 3 though.
if version < '2.6' or version > '3':
		raise ImportError('modred requires python version 2.7.x')
# Soon we will need to require numpy 1.7 for a change Jon will make
# in POD with new squeeze function.
from distutils.core import setup
from __init__ import __version__
setup(name='modred',
      version=__version__,
      author='Brandt Belson, Jonathan Tu, and Clarence W. Rowley',
      author_email='bbelson@princeton.edu, jhtu@princeton.edu, cwrowley@princeton.edu',
      maintainer='Brandt Belson',
      maintainer_email='bbelson@princeton.edu',
      description='Compute modal decompositions and reduced-order models'+\
      		' easily, efficiently, and in parallel.',
      classifiers=['Programming Language :: Python', 
        ],
      license='Free BSD',
      packages=['modred', 'modred.src', 'modred.examples', 'modred.tests'],
      package_dir={'modred':'', 'modred.src': 'src', 'modred.examples': 'examples',
          'modred.test': 'tests'},
      package_data={'modred':['tests/files_okid/SISO/*', 'tests/files_okid/SIMO/*', 
          'tests/files_okid/MISO/*', 'tests/files_okid/MIMO/*']},
      )
