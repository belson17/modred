#!/usr/bin/env python
"""Helper script for looking at the scaling and profiling in parallel.

Give this script a number of processors and optionally a number of lines of
stats to print, e.g. ``python load_prof_parallel.py 24 40``
"""
from __future__ import print_function
from future.builtins import range
import sys
import pstats as P


prof_path = 'lincomb_r%d.prof'

num_procs = 1
num_stats = 30

if len(sys.argv) == 2:
    num_procs = int(sys.argv[1])
    
if len(sys.argv) == 3:
    num_procs = int(sys.argv[1])
    num_stats = int(sys.argv[2])

stats = P.Stats(prof_path%0)
for rank in range(1, num_procs):
    stats.add(prof_path%rank)   
    
print('\n----- Sum of all processors stats -----')
stats.strip_dirs().sort_stats('cumulative').print_stats(num_stats)



