import math
import operator

import mdarray as md
from mdarray_core.exceptions import IncompatibleDimensions

__all__ = ["apply_unary_function", "apply_binary_function"]


def apply_unary_function(arr1, func):
    size = arr1.size
    for i in range(size):
        arr1.data[i] = func(arr1.data[i])


def apply_binary_function(arr1, arr2, func):
    if isinstance(arr1, md.mdarray) and not isinstance(arr2, md.mdarray):
        for i in range(arr1.size):
            arr1.data[i] = func(arr1.data[i], arr2)
    elif not isinstance(arr1, md.mdarray) and isinstance(arr2, md.mdarray):
        for i in range(arr2.size):
            arr2.data[i] = func(arr1, arr2.data[i])
    elif isinstance(arr1, md.mdarray) and isinstance(arr2, md.mdarray):
        if arr1.size != arr2.size:
            raise IncompatibleDimensions
        else:
            for i in range(arr1.size):
                arr1.data[i] = func(arr1.data[i], arr2.data[i])


def sin(x):
    try:
        return x.__sin__()
    except AttributeError:
        return math.sin(x)


def cos(x):
    try:
        return x.__cos__()
    except AttributeError:
        return math.cos(x)


def tan(x):
    try:
        return x.__tan__()
    except AttributeError:
        return math.tan(x)


def arcsin(x):
    try:
        return x.__arcsin__()
    except AttributeError:
        return math.asin(x)


def arccos(x):
    try:
        return x.__arccos__()
    except AttributeError:
        return math.acos(x)


def arctan(x):
    try:
        return x.__arctan__()
    except AttributeError:
        return math.atan(x)


def arctan2(y, x):
    try:
        return x.__arctan2__(y)
    except AttributeError:
        return math.atan2(x)


def sinh(x):
    try:
        return x.__sinh__()
    except AttributeError:
        return math.sin(x)


def cosh(x):
    try:
        return x.__cosh__()
    except AttributeError:
        return math.cos(x)


def tanh(x):
    try:
        return x.__tanh__()
    except AttributeError:
        return math.tanh(x)


def arcsinh(x):
    try:
        return x.__arcsinh__()
    except AttributeError:
        return math.asinh(x)


def arccosh(x):
    try:
        return x.__arccosh__()
    except AttributeError:
        return math.acosh(x)


def arctanh(x):
    try:
        return x.__arctanh__()
    except AttributeError:
        return math.atanh(x)
