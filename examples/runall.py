#!/usr/bin/env python

# Assumes modred has been installed.

import modred as MR
parallel = MR.parallel_default_instance

for i in range(1, 7):
    if not (parallel.is_distributed() and i==3):
        execfile('tutorial_ex%d.py'%i)
        parallel.barrier()

for i in range(1, 3):
    execfile('rom_ex%d.py'%i)
    parallel.barrier()

