"""Tests for broadcasting."""

from __future__ import annotations

import pytest


def test_generate_broadcast_shape_6d():
    from mdarray import generate_broadcast_shape, zeros
    arr1 = zeros([1, 5, 4, 3, 2, 2])
    arr2 = zeros([6, 1, 4, 3, 2, 2])
    arr3 = zeros([6, 5, 1, 3, 2, 2])
    arr4 = zeros([6, 5, 4, 1, 2, 2])
    arr5 = zeros([6, 5, 4, 3, 1, 2])
    arr6 = zeros([6, 5, 4, 3, 2, 1])

    shape, _repts = generate_broadcast_shape(arr1, arr2, arr3, arr4, arr5, arr6)
    assert shape == [6, 5, 4, 3, 2, 2]


def test_broadcast_scalar():
    from mdarray import broadcast_nary, irange, tomdarray
    arr1 = irange([2, 2])
    arr2 = tomdarray(10)
    result = broadcast_nary(arr1, arr2, func=lambda args: args[0] + args[1])
    assert result.data == [10, 11, 12, 13]


def test_broadcast_toshape():
    from mdarray import broadcast_toshape, irange
    arr = irange(3)
    # Broadcasting [3] to [3, 1, 1, 3] — arr gets padded to [3, 1, 1, 1]
    # then broadcast against [3, 1, 1, 3], giving shape [3, 1, 1, 3]
    result = broadcast_toshape(arr, [3, 1, 1, 3])
    assert result.shape == [3, 1, 1, 3]
    assert result.size == 9  # 3*1*1*3 = 9


def test_incompatible_broadcast():
    from mdarray import generate_broadcast_shape, zeros
    from mdarray.core.exceptions import IncompatibleDimensions
    arr1 = zeros([3, 4])
    arr2 = zeros([5, 4])
    with pytest.raises(IncompatibleDimensions):
        generate_broadcast_shape(arr1, arr2)
