from __future__ import annotations

import math
from typing import Any

from ..array import broadcast_nary, mdarray, tomdarray

__all__ = [
    "apply_binary_function",
    "apply_unary_function",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "cos",
    "cosh",
    "nroot",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
]


def apply_unary_function(arr1: mdarray, func: Any) -> None:
    size = arr1.size
    for i in range(size):
        arr1.data[i] = func(arr1.data[i])


def apply_binary_function(arr1: Any, arr2: Any, func: Any) -> mdarray:
    arr1 = tomdarray(arr1)
    arr2 = tomdarray(arr2)
    return broadcast_nary(arr1, arr2, func=func)


def sqrt(x: Any) -> Any:
    return math.sqrt(x)


def nroot(x: Any, root: Any) -> Any:
    return math.pow(x, root)


def sin(x: Any) -> Any:
    return math.sin(x)


def cos(x: Any) -> Any:
    return math.cos(x)


def tan(x: Any) -> Any:
    return math.tan(x)


def arcsin(x: Any) -> Any:
    return math.asin(x)


def arccos(x: Any) -> Any:
    return math.acos(x)


def arctan(x: Any) -> Any:
    return math.atan(x)


def arctan2(y: Any, x: Any) -> Any:
    return math.atan2(y, x)


def sinh(x: Any) -> Any:
    return math.sinh(x)


def cosh(x: Any) -> Any:
    return math.cosh(x)


def tanh(x: Any) -> Any:
    return math.tanh(x)


def arcsinh(x: Any) -> Any:
    return math.asinh(x)


def arccosh(x: Any) -> Any:
    return math.acosh(x)


def arctanh(x: Any) -> Any:
    return math.atanh(x)
