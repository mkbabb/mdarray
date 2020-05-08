import math
import re
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.types import inf, nan
from MultiArray import MultiArray

__all__ = ["print_array", "pad_array_fmt", "trim_string"]

MAX_CHARS_LINE = 50
SAVED_CHAR = 3


def trim_string(s: str, sep: str) -> str:
    sep_len = len(sep)

    if len(s) >= MAX_CHARS_LINE:
        s_re = list(re.finditer(sep, s))
        N = len(s_re)

        if N >= (2 * SAVED_CHAR + 1):
            start = s_re[SAVED_CHAR - 1].start()
            stop = s_re[N - (SAVED_CHAR)].start()

            s = "{0} ... {1}".format(s[:start], s[stop + sep_len:])

    return s


def print_array(arr: MultiArray,
                sep: Optional[str] = ", ",
                formatter: Optional[Callable[[str], str]] = None
                ) -> str:
    if not formatter:
        def formatter(x, y): return f"{x[y]}"
    if not sep:
        sep = ", "

    t_ix = arr.index

    s = "[" * (arr.mdim)

    for i in range(arr.size):
        s += formatter(arr.data, arr.index)
        next(arr)
        ix = 0
        for j in range(arr.mdim):
            if (arr.was_advanced[j]):
                ix += 1

        if (i < arr.size - 1):
            if (ix > 0):
                s += "]" * ix
                s += "\n" * ix
                s += " " * (arr.mdim - ix)
                s += "[" * ix
            else:
                s += ", "
        else:
            s += "]" * arr.mdim
    arr.at(t_ix)
    return s


def pad_array_fmt(arr: MultiArray
                  ) -> Callable[[int, float, Any], str]:
    max_len = len(str(max(arr.data, key=lambda x: len(str(x)))))

    def fmter(x): return '{0}{1}{2}'.format(' ' * int(math.ceil((max_len - len(str(x))) / 2)), x,
                                            ' ' * int(math.floor((max_len - len(str(x))) / 2)))
    return fmter


# def make_nested_list(arr: MultiArray) -> list:
#     tmp = "[" * arr.mdim
#     nests = [""] * (arr.mdim - 1)

#     for i in range(arr.size):
#         tmp += str(arr.data[arr.index])
#         next(arr)

#         ix = 0
#         for j in range(1, arr.mdim):
#             if arr.was_advanced[j]:
#                 if j == 1:
#                     nests[0] += tmp
#                     tmp = ""
#                 else:
#                     nests[j - 1] += nests[j - 2]
#                     nests[j - 2] = ""
#                 ix += 1

#         if (i < arr.size - 1):
#             if (ix > 0):
#                 tmp += "]" * ix
#                 tmp += "\n" * ix
#                 tmp += " " * (arr.mdim - ix)
#                 tmp += "[" * ix
#             else:
#                 tmp += ", "
#         else:
#             nests[-1] += "]" * arr.mdim

#     arr.at(0)
#     return nests[-1]
