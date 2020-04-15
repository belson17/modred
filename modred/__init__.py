"""This file makes the modred directory a python package."""
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
    compute_POD_arrays_direct_method, compute_POD_arrays_snaps_method
)

from .dmd import (
    DMDHandles,
    compute_DMD_arrays_direct_method,
    compute_DMD_arrays_snaps_method,
    TLSqrDMDHandles,
    compute_TLSqrDMD_arrays_direct_method,
    compute_TLSqrDMD_arrays_snaps_method,
)

from .bpod import BPODHandles, compute_BPOD_arrays

from .era import compute_ERA_model, make_sampled_format, ERA

from .okid import OKID

from .ltigalerkinproj import (
    LTIGalerkinProjectionBase,
    LTIGalerkinProjectionHandles,
    LTIGalerkinProjectionArrays,
    compute_derivs_handles, compute_derivs_arrays, standard_basis
)

from .vectorspace import VectorSpaceHandles, VectorSpaceArrays

from .vectors import (
    Vector, VecHandle,
    VecHandlePickle, VecHandleInMemory, VecHandleArrayText,
    InnerProductTrapz, inner_product_array_uniform
)

from . import parallel

from .util import (
    UndefinedError,
    atleast_2d_row, atleast_2d_col,
    make_iterable, flatten_list,
    save_array_text, load_array_text, get_file_list, get_data_members,
    sum_arrays, sum_lists,
    smart_eq,
    InnerProductBlock,
    svd, eigh, eig_biorthog,
    balanced_truncation,
    drss, rss, lsim, impulse, load_signals, load_multiple_signals,
    Hankel, Hankel_chunks
)

from .py2to3 import (
    run_script, print_msg, print_stdout, print_stderr, range
)

from modred import tests
