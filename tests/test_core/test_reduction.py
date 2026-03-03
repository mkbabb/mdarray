"""Tests for reduction operations."""

from __future__ import annotations


def test_reductor_add():
    from mdarray.core.reduction import reductor

    r = reductor.add()
    assert r.reduce([1, 2, 3, 4, 5]) == 15


def test_reductor_mul():
    from mdarray.core.reduction import reductor

    r = reductor.mul()
    assert r.reduce([1, 2, 3, 4, 5]) == 120


def test_inner_product():
    from mdarray.core.reduction import inner_product

    assert inner_product([1, 2, 3], [4, 5, 6]) == 32


def test_reductor_accumulate():
    from mdarray.core.reduction import reductor

    r = reductor.add()
    result = r.accumulate([1, 2, 3, 4, 5])
    assert result == [1, 3, 6, 10, 15]
