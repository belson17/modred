"""This file makes the modred directory a python package."""
from __future__ import absolute_import

from ._version import __version__


# Modules whose internal contents are available through the modred
# namespace as "modred.foo" are imported below.  For example, this
# allows "myPOD = modred.POD()" rather than "myPOD =
# modred.POD.POD()".  Since we have a small library with few classes
# and functions, it's easiest to make many modules available from the
# top level.  There are no naming conflicts and there is no room for
# confusion.

from .pod import (
    PODHandles,
    compute_POD_matrices_direct_method, compute_POD_matrices_snaps_method
)

from .dmd import (
    DMDHandles,
    compute_DMD_matrices_direct_method, compute_DMD_matrices_snaps_method
)

from .bpod import BPODHandles, compute_BPOD_matrices

from .era import compute_ERA_model, make_sampled_format, ERA

from .okid import OKID

from .ltigalerkinproj import (
    LTIGalerkinProjectionBase,
    LTIGalerkinProjectionHandles, 
    LTIGalerkinProjectionMatrices,
    compute_derivs_handles, compute_derivs_matrices, standard_basis
)

from .vectorspace import VectorSpaceHandles, VectorSpaceMatrices

from .vectors import (
    VecHandlePickle, VecHandleInMemory, 
    Vector, VecHandle, VecHandleArrayText,
    InnerProductTrapz, inner_product_array_uniform
)

from .parallel import Parallel, ParallelError, parallel_default_instance

from .util import (
    UndefinedError, make_mat, make_iterable, flatten_list, save_array_text,
    load_array_text, get_file_list, get_data_members, sum_arrays, sum_lists,
    smart_eq, InnerProductBlock, svd, eigh, eig_biorthog,
    solve_Lyapunov_iterative, solve_Lyapunov_direct, balanced_truncation, drss,
    rss, lsim, impulse, load_multiple_signals, load_signals, Hankel
)

from modred import tests

del absolute_import


