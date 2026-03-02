from .exceptions import IncompatibleDimensions
from .helper import (
    flatten_list,
    get_strides,
    make_mdim_shape,
    pair_wise,
    remove_extraneous_dims,
    roll_array,
    swap,
)
from .reduction import inner_product, reductor
from .types import inf, nan

__all__ = [
    # exceptions
    "IncompatibleDimensions",
    # helper
    "flatten_list",
    "get_strides",
    # types
    "inf",
    # reduction
    "inner_product",
    "make_mdim_shape",
    "nan",
    "pair_wise",
    "reductor",
    "remove_extraneous_dims",
    "roll_array",
    "swap",
]
