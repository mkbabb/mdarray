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
        formatter = lambda x: f"{x}"
    if not sep:
        sep = ", "

    mdim = arr.mdim
    size = arr.size
    data = arr.data
    mditer = arr.iterator

    s = ""
    strings = [""] * (mdim - 1)
    for i in range(size):
        s += formatter(data[mditer.index])
        next(mditer)
        s += sep if not mditer.was_advanced[1] else ''

        ix = 0
        for j in range(1, mdim):
            if mditer.was_advanced[j]:
                if j == 1:
                    strings[0] += f"[{s}]"
                    s = ""
                else:
                    strings[j - 1] += f"[{strings[j - 2]}]"
                    strings[j - 2] = ""
                ix += 1
        if ix > 0 and i != size - 1:
            new_line = "\n" * ix
            hanging_indent = " " * (mdim - ix)
            strings[ix - 1] += (sep.strip() + new_line + hanging_indent)

    s = f"[{strings[-1]}]"
    mditer.at(0)
    return s


def pad_array_fmt(arr: MultiArray
                  ) -> Callable[[int, float, Any], str]:
    max_len = len(str(max(arr.data, key=lambda x: len(str(x)))))

    fmmter = lambda x: '{0}{1}{2}'.format(' ' * int(math.ceil((max_len - len(str(x))) / 2)), x,
                                          ' ' * int(math.floor((max_len - len(str(x))) / 2)))
    return fmmter
