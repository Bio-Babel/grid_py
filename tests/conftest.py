"""Shared pytest fixtures for the grid_py test suite."""

from __future__ import annotations

import sys
import os

import pytest

# Ensure the grid_py package is importable from the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_REPO, "grid_py")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import grid_py


# ---------------------------------------------------------------------------
# Fixture: reset grid state before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_grid_state():
    """Reset the global GridState before every test so tests are isolated."""
    state = grid_py.get_state()
    state.reset()
    yield
    # Optionally reset again after the test to leave a clean slate.
    state.reset()


# ---------------------------------------------------------------------------
# Basic object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_unit():
    """A simple 1 cm Unit."""
    return grid_py.Unit(1, "cm")


@pytest.fixture
def basic_gpar():
    """A Gpar with col='red'."""
    return grid_py.Gpar(col="red")


@pytest.fixture
def basic_viewport():
    """A Viewport with width=0.5 npc and height=0.5 npc."""
    return grid_py.Viewport(
        width=grid_py.Unit(0.5, "npc"),
        height=grid_py.Unit(0.5, "npc"),
    )


@pytest.fixture
def basic_grob():
    """A rect_grob with default parameters."""
    return grid_py.rect_grob()
