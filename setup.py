
from setuptools import setup, find_packages

import os
here = os.path.abspath(os.path.dirname(__file__))

# # Get the long description from the relevant file
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='modred',
    version='1.0.2',
    description=(
        'Compute modal decompositions and reduced-order models, '
        'easily, efficiently, and in parallel.'
        ),
    # long_description=long_description,
    # keywords='',
    author='Brandt Belson, Jonathan Tu, and Clarence W. Rowley',
    author_email=(
        'bbelson@princeton.edu, jhtu@princeton.edu, cwrowley@princeton.edu'
        ),
    url='https://pythonhosted.org/modred',
    maintainer='Brandt Belson, Jonathan Tu, and Clarence W. Rowley',
    maintainer_email='bbelson@princeton.edu',
    license='Free BSD',
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'
        ],
    packages=find_packages(exclude=['docs', 'matlab']),
    package_data={'modred':[
            'modred/tests/files_okid/SISO/*', 
            'modred/tests/files_okid/SIMO/*', 
            'modred/tests/files_okid/MISO/*', 
            'modred/tests/files_okid/MIMO/*']},
    )
