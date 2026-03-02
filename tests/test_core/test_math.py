"""Tests for math operations and operator overloading."""

from __future__ import annotations


def test_add_scalar():
    from mdarray import irange
    arr = irange(5)
    result = arr + 10
    assert result.data == [10, 11, 12, 13, 14]


def test_mul_scalar():
    from mdarray import irange
    arr = irange(5)
    result = arr * 2
    assert result.data == [0, 2, 4, 6, 8]


def test_sub():
    from mdarray import irange
    arr = irange(5)
    result = arr - 1
    assert result.data == [-1, 0, 1, 2, 3]


def test_pow():
    from mdarray import tomdarray
    arr = tomdarray([1, 2, 3])
    result = arr ** 2
    assert result.data == [1, 4, 9]


def test_neg():
    from mdarray import tomdarray
    arr = tomdarray([1, -2, 3])
    result = -arr
    assert result.data == [-1, 2, -3]


def test_eq():
    from mdarray import tomdarray
    arr1 = tomdarray([1, 2, 3])
    arr2 = tomdarray([1, 0, 3])
    result = arr1 == arr2
    assert result.data == [True, False, True]


def test_radd():
    from mdarray import irange
    arr = irange(3)
    result = 10 + arr
    assert result.data == [10, 11, 12]


def test_rmul():
    from mdarray import irange
    arr = irange(3)
    result = 5 * arr
    assert result.data == [0, 5, 10]
