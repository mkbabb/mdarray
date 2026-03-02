"""Shared fixtures for mdarray test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def MA():
    """Provide the mdarray class."""
    from mdarray import mdarray
    return mdarray


@pytest.fixture
def md():
    """Provide the mdarray module."""
    import mdarray
    return mdarray
