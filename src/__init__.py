# This file makes the modaldecomp directory a python package.

# Modules to load when using "from modaldecomp import *"
#__all__ = ['bpod', 'bpodltirom', 'dmd', 'era', 'fieldoperations',
#		'okid', 'parallel', 'pod', 'reductions', 'util']

# Modules whose internal contents are available through the 
# modaldecomp namespace as "modaldecomp.foo()" are imported below.
# For example, this allows "myPOD = modaldecomp.POD()" rather than
# "myPOD = modaldecomp.POD.POD()".
# Since we have a small library with few classes and functions,
# it's easiest to make many modules available from the top level.
# There are no naming conflicts and there is no room for confusion.

from bpod import *
from pod import *
from dmd import *
from era import *
from fieldoperations import *
from okid import *
from parallel import *
from util import *
import reductions

