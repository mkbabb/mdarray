import math
import re
from functools import reduce

from core.helper import pair_wise_accumulate
from core.types import inf, nan

__all__ = ["print_array", "pad_array_fmt"]

MAX_CHAR_LINE = 50
SAVED_CHAR = 3


def trim_string(s, sep):
    sep_len = len(sep)

    if len(s) >= MAX_CHAR_LINE:
        s_re = list(re.finditer(sep, s))
        N = len(s_re)

        if N >= (2*SAVED_CHAR + 1):
            start = s_re[SAVED_CHAR - 1].start()
            stop = s_re[N - (SAVED_CHAR)].start()

            s = "{0} ... {1}".format(s[:start], s[stop + sep_len:])

    return s


def print_array(arr, sep='', formatter=None):
    mdim = arr.mdim
    axis_counter = [0]*mdim

    if not formatter:
        def formatter(x): return '{0}'.format(x)

    def recurse(ix):
        axis = arr.shape[ix]
        remaining_axes = mdim - ix

        s = ''
        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i
                ix_i = pair_wise_accumulate(axis_counter, arr.strides)

                a_val = arr.data[ix_i]

                val = formatter(a_val)
                s += (val + sep) if i < axis - 1 else val
        else:
            new_line = '\n'*(ix)
            for i in range(axis):
                axis_counter[ix] = i

                val = recurse(ix - 1)

                s += ' '*(remaining_axes) + val if i > 0 else val
                s += sep.strip() if i < axis - 1 else ''
                s += new_line if i != axis - 1 else ''

        if ix == 0:
            s = trim_string(s, sep)

        s = '[{0}]'.format(s)
        
        return s

    return recurse(mdim - 1)


def pad_array_fmt(arr):
    max_len = len(str(max(arr.data, key=lambda x: len(str(x)))))

    def fmmter(x): return '{0}{1}{2}'.format(' '*int(math.ceil((max_len - len(str(x)))/2)), x,
                                             ' '*int(math.floor((max_len - len(str(x)))/2)))
    return fmmter
