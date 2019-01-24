from functools import reduce

import numpy as np

from mdarray import arange, mdarray, tomdarray
from mdarray_formatting import pad_array_fmt
from mdarray_helper import get_strides, pair_wise_accumulate, update_dict, swap_item
from mdarray_indexing import gslice, iter_gslice, make_nested
from mdarray_types import inf, mdarray_inquery, nan, IncompatibleDimensions



