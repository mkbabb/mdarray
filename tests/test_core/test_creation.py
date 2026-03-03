"""Tests for array creation functions."""

from __future__ import annotations


def test_zeros_1d():
    from mdarray import zeros

    arr = zeros([20])
    assert arr.data == [0] * 20
    assert arr.shape == [20]
    assert arr.size == 20


def test_ones_1d():
    from mdarray import ones

    arr = ones([20])
    assert arr.data == [1] * 20


def test_full_1d():
    from mdarray import full

    arr = full([20], fill_value=99.9)
    assert arr.data == [99.9] * 20


def test_irange_flat():
    from mdarray import irange

    arr = irange(100)
    assert arr.data == list(range(100))
    assert arr.shape == [100]


def test_irange_shaped():
    from mdarray import irange

    arr = irange([3, 4])
    assert arr.data == list(range(12))
    assert arr.shape == [3, 4]


def test_linear_range_int():
    from mdarray import linear_range

    arr = linear_range(0, 100, 100)
    assert arr.data == list(range(100))


def test_linear_range_float():
    from mdarray import linear_range

    denom = 100
    expected = [round(i / denom, 8) for i in range(denom)]
    arr = linear_range(start=0.0, stop=1, size=denom)
    result = [round(i, 8) for i in arr.data]
    assert result == expected


def test_tomdarray_2d():
    from mdarray import irange, tomdarray

    lst = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    arr1 = tomdarray(lst)
    arr2 = irange([3, 3])
    assert arr1.data == arr2.data


def test_tomdarray_3d():
    from mdarray import irange, tomdarray

    lst = [
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
    ]
    arr1 = tomdarray(lst)
    arr2 = irange([3, 3, 2])
    assert arr1.data == arr2.data


def test_tomdarray_scalar():
    from mdarray import tomdarray

    arr = tomdarray(42)
    assert arr.data == [42]
    assert arr.size == 1


def test_zeros_multidim():
    from mdarray import zeros

    arr = zeros([2, 3, 4])
    assert arr.size == 24
    assert arr.shape == [2, 3, 4]
    assert all(x == 0 for x in arr.data)
