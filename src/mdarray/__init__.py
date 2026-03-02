"""mdarray: Pure-Python N-dimensional array library with mixed-radix FFT.

A pedagogical and pragmatically optimized exploration of N-dimensional array
computation, built from scratch with no compiled dependencies for core
functionality. Implements strided memory layout, broadcasting, mixed-radix FFT,
and basic linear algebra entirely in Python.
"""

from .array import (
    broadcast,
    broadcast_iter,
    broadcast_nary,
    broadcast_toshape,
    concatenate,
    dense_meshgrid,
    dstack,
    expand_indicies,
    flatten,
    full,
    generate_broadcast_shape,
    hstack,
    indicies,
    irange,
    ix_meshgrid,
    linear_range,
    log_range,
    make_mdim,
    make_nested_list,
    mdarray,
    meshgrid,
    ones,
    pad_array_fmt,
    print_array,
    ravel,
    repeat,
    reshape,
    roll_axis,
    slice_array,
    swap_axis,
    tile,
    tomdarray,
    transpose,
    trim_string,
    unravel,
    unravel_dense,
    vstack,
    zeros,
)
from .core.exceptions import IncompatibleDimensions
from .core.padding import pad_array
from .core.reduction import inner_product, reductor
from .core.types import inf, nan
from .fft import BACKEND as FFT_BACKEND
from .fft import cfft, fftn, ifft, ifftn, rfft

__version__ = "0.1.0"

__all__ = [
    "FFT_BACKEND",
    "IncompatibleDimensions",
    # Broadcasting
    "broadcast",
    "broadcast_iter",
    "broadcast_nary",
    "broadcast_toshape",
    # FFT
    "cfft",
    "concatenate",
    "dense_meshgrid",
    "dstack",
    "expand_indicies",
    "fftn",
    "flatten",
    "full",
    "generate_broadcast_shape",
    "hstack",
    "ifft",
    "ifftn",
    "indicies",
    # Types
    "inf",
    "inner_product",
    "irange",
    "ix_meshgrid",
    "linear_range",
    "log_range",
    "make_mdim",
    "make_nested_list",
    # Array class
    "mdarray",
    # Grid
    "meshgrid",
    "nan",
    "ones",
    # Padding
    "pad_array",
    "pad_array_fmt",
    # Formatting
    "print_array",
    # Indexing
    "ravel",
    # Reduction
    "reductor",
    "repeat",
    # Manipulation
    "reshape",
    "rfft",
    "roll_axis",
    "slice_array",
    "swap_axis",
    "tile",
    "tomdarray",
    "transpose",
    "trim_string",
    "unravel",
    "unravel_dense",
    "vstack",
    # Creation
    "zeros",
]
