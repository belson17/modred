#!/usr/bin/env python
from past.builtins import execfile
from future.builtins import range

import os

import modred as mr     # modred must be installed.


parallel = mr.parallel_default_instance

for i in range(1, 7):
    if not parallel.is_distributed():
        execfile('tutorial_ex%d.py'%i)
    if parallel.is_distributed() and i >= 3:
        parallel.barrier()
        execfile('tutorial_ex%d.py'%i)
        parallel.barrier()

if not parallel.is_distributed():
    execfile('main_CGL.py')

for i in range(1, 3):
    if not parallel.is_distributed():
        execfile('rom_ex%d.py'%i)
    if parallel.is_distributed() and i > 1:
        execfile('rom_ex%d.py'%i)
        parallel.barrier()
