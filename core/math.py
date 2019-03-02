import math
import operator

import MultiArray as ma
from core.creation import broadcast_bnry, tomdarray
from core.exceptions import IncompatibleDimensions


def apply_unary_function(arr1, func):
    size = arr1.size
    for i in range(size):
        arr1.data[i] = func(arr1.data[i])


def apply_binary_function(arr1, arr2, func):
    arr1 = tomdarray(arr1)
    arr2 = tomdarray(arr2)
    return broadcast_bnry(arr1, arr2, func=func)


def sqrt(x):
    try:
        return x.__sqrt__()
    except AttributeError:
        return math.sqrt(x)


def nroot(x, root):
    try:
        return x.__nroot__(root)
    except AttributeError:
        return math.pow(x, root)


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
