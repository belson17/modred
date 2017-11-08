#!/usr/bin/env python
import os

from modred import parallel
import modred as mr     # modred must be installed.


# Run tutorial scripts
for i in mr.range(1, 7):
    if not parallel.is_distributed():
        mr.run_script('tutorial_ex%d.py' % i)
    if parallel.is_distributed() and i >= 3:
        parallel.barrier()
        mr.run_script('tutorial_ex%d.py' % i)
        parallel.barrier()

# Run reduced-order model scripts
for i in mr.range(1, 3):
    if not parallel.is_distributed():
        mr.run_script('rom_ex%d.py' % i)
    if parallel.is_distributed() and i > 1:
        mr.run_script('rom_ex%d.py' % i)
        parallel.barrier()

# Run CGL scripts
if not parallel.is_distributed():
    mr.run_script('main_CGL.py')
