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

from .bpod import BPODHandles, compute_BPOD_matrices

from .pod import (
    PODHandles,
    compute_POD_matrices_direct_method,
    compute_POD_matrices_snaps_method
)

from .dmd import (
    DMDHandles,
    compute_DMD_matrices_direct_method,
    compute_DMD_matrices_snaps_method
)

from .era import compute_ERA_model, make_sampled_format, ERA
from .vectorspace import VectorSpaceHandles, VectorSpaceMatrices
from .okid import OKID

from .ltigalerkinproj import (
    LTIGalerkinProjectionHandles,
    compute_derivs_handles,
    LTIGalerkinProjectionBase,
    standard_basis,
    compute_derivs_matrices,
    LTIGalerkinProjectionMatrices
)
from .parallel import Parallel, ParallelError, parallel_default_instance

from .util import (
    save_array_text, impulse, solve_Lyapunov_iterative,
    load_array_text, balanced_truncation, get_data_members,
    flatten_list, make_mat, load_multiple_signals,
    make_list, UndefinedError, InnerProductBlock,
    svd, lsim, load_signals, eigh, drss, Hankel, rss, sum_lists,
    get_file_list, solve_Lyapunov_direct, smart_eq, sum_arrays
)

from .vectors import (
    VecHandlePickle, VecHandleInMemory,
    Vector, VecHandle, VecHandleArrayText,
    InnerProductTrapz, inner_product_array_uniform
)

from modred import tests

del absolute_import


