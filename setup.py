
from sys import version
# Check out unittest2, maybe use older pythons. Not 3 though.
if version < '2.6' or version > '3':
    raise ImportError('modred requires python version 2.7.x')

from setuptools import setup, find_packages

from version import __version__
setup(
    name='modred',
    version=__version__,
    author='Brandt Belson, Jonathan Tu, and Clarence W. Rowley',
    author_email=(
        'bbelson@princeton.edu, jhtu@princeton.edu, cwrowley@princeton.edu'
        ),
    maintainer='Brandt Belson, Jonathan Tu, and Clarence W. Rowley',
    maintainer_email='bbelson@princeton.edu',
    description=(
        'Compute modal decompositions and reduced-order models'
        ' easily, efficiently, and in parallel.'
        ),
    license='Free BSD',
    classifiers=[
        'Programming Language :: Python', 
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        ],
    packages=find_packages(exclude=['docs', 'matlab']),
    package_data={'modred':[
            'modred/tests/files_okid/SISO/*', 
            'modred/tests/files_okid/SIMO/*', 
            'modred/tests/files_okid/MISO/*', 
            'modred/tests/files_okid/MIMO/*']},
    )
