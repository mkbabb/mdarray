"""Tests for array manipulation: reshape, transpose, concatenate, etc."""

from __future__ import annotations

import pytest


def test_reshape():
    from mdarray import irange
    arr = irange(24)
    arr.reshape([4, 3, 2])
    assert arr.shape == [4, 3, 2]
    assert arr.size == 24


def test_reshape_incompatible():
    from mdarray import irange
    from mdarray.core.exceptions import IncompatibleDimensions
    arr = irange(24)
    with pytest.raises(IncompatibleDimensions):
        arr.reshape([5, 5])


def test_transpose():
    from mdarray import irange
    arr = irange([3, 4])
    arr.T(0, 1)
    assert arr.shape == [4, 3]


def test_make_nested_list():
    from mdarray import irange
    arr = irange([3, 2])
    lst = arr.to_list()
    assert len(lst) == 2
    assert len(lst[0]) == 3


def test_concatenate_axis0():
    from mdarray import concatenate, irange
    arr1 = irange([3, 2])
    arr2 = irange([3, 2])
    result = concatenate(arr1, arr2, caxis=0)
    assert result.shape[0] == 6
    assert result.shape[1] == 2


def test_flatten():
    from mdarray import irange
    arr = irange([3, 4, 2])
    # flatten(-1) with mdim=3: order = -1 + 3 = 2, new_mdim = 3-2 = 1
    # This collapses the last 2 dims into 1
    arr.flatten(-1)
    # With order=-1 -> order=2, new_mdim=1, last dim = 3*4*2=24... wrong.
    # Actually: order=2, new_mdim = 3-2 = 1, new_shape[0] = shape[0] = 3,
    # then last dim = product(shape[0:]) = 24. But with new_mdim=1 the shape is [24].
    # The flatten(-1) semantics: completely flatten.
    assert arr.mdim == 1
    assert arr.size == 24


def test_print_array():
    from mdarray import irange
    arr = irange([3, 2])
    s = str(arr)
    assert "[" in s
    assert "]" in s
