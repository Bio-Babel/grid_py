"""Tests for grid_py Viewport module."""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

# Ensure grid_py is importable from the sibling directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grid_py"))

from grid_py._gpar import Gpar
from grid_py._units import Unit
from grid_py._path import VpPath
from grid_py._viewport import (
    Viewport,
    VpList,
    VpStack,
    VpTree,
    depth,
    data_viewport,
    plot_viewport,
    edit_viewport,
    _reset_vp_auto_name,
    push_viewport,
    pop_viewport,
    up_viewport,
    current_viewport,
)
from grid_py._state import GridState, get_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_auto_name():
    """Reset the viewport auto-name counter before each test."""
    _reset_vp_auto_name()
    yield


@pytest.fixture()
def fresh_state():
    """Reset the global GridState singleton and return it."""
    state = get_state()
    state.reset()
    return state


# ---------------------------------------------------------------------------
# Viewport construction -- defaults
# ---------------------------------------------------------------------------


class TestViewportDefaults:
    """Viewport constructed with default arguments."""

    def test_default_name_auto(self):
        vp = Viewport()
        assert vp.name == "GRID.VP.1"

    def test_default_xscale(self):
        vp = Viewport()
        assert vp.xscale == [0.0, 1.0]

    def test_default_yscale(self):
        vp = Viewport()
        assert vp.yscale == [0.0, 1.0]

    def test_default_angle(self):
        vp = Viewport()
        assert vp.angle == 0.0

    def test_default_gp_is_empty(self):
        vp = Viewport()
        assert len(vp.gp) == 0  # no params set

    def test_default_clip_inherit(self):
        vp = Viewport()
        # "inherit" maps to False internally
        assert vp.clip is False

    def test_default_layout_none(self):
        vp = Viewport()
        assert vp.layout is None

    def test_default_layout_pos(self):
        vp = Viewport()
        assert vp.layout_pos_row is None
        assert vp.layout_pos_col is None

    def test_str_contains_name(self):
        vp = Viewport(name="myvp")
        assert str(vp) == "viewport[myvp]"


# ---------------------------------------------------------------------------
# Viewport construction -- custom parameters
# ---------------------------------------------------------------------------


class TestViewportCustom:
    """Viewport constructed with custom arguments."""

    def test_custom_name(self):
        vp = Viewport(name="panel")
        assert vp.name == "panel"

    def test_custom_xscale(self):
        vp = Viewport(xscale=[0, 100])
        assert vp.xscale == [0.0, 100.0]

    def test_custom_yscale(self):
        vp = Viewport(yscale=[-5, 5])
        assert vp.yscale == [-5.0, 5.0]

    def test_custom_angle(self):
        vp = Viewport(angle=45)
        assert vp.angle == 45.0

    def test_custom_gp(self):
        gp = Gpar(col="red")
        vp = Viewport(gp=gp)
        assert vp.gp.get("col") == "red"

    def test_clip_on(self):
        vp = Viewport(clip="on")
        assert vp.clip is True

    def test_clip_off(self):
        vp = Viewport(clip="off")
        assert vp.clip is None

    def test_numeric_x_coerced_to_unit(self):
        vp = Viewport(x=0.3, default_units="npc")
        assert isinstance(vp.x, Unit)

    def test_unit_x_stays_unit(self):
        u = Unit(0.5, "npc")
        vp = Viewport(x=u)
        assert vp.x is u


# ---------------------------------------------------------------------------
# Auto-naming
# ---------------------------------------------------------------------------


class TestViewportAutoName:
    """Auto-generated viewport names increment correctly."""

    def test_sequential_names(self):
        vp1 = Viewport()
        vp2 = Viewport()
        assert vp1.name == "GRID.VP.1"
        assert vp2.name == "GRID.VP.2"

    def test_reset_counter(self):
        _ = Viewport()
        _reset_vp_auto_name()
        vp = Viewport()
        assert vp.name == "GRID.VP.1"


# ---------------------------------------------------------------------------
# Scale validation
# ---------------------------------------------------------------------------


class TestViewportScaleValidation:
    """xscale / yscale validation."""

    def test_xscale_must_be_length_2(self):
        with pytest.raises(ValueError, match="invalid 'xscale'"):
            Viewport(xscale=[1, 2, 3])

    def test_xscale_must_be_distinct(self):
        with pytest.raises(ValueError, match="range must be non-zero"):
            Viewport(xscale=[5, 5])

    def test_xscale_must_be_finite(self):
        with pytest.raises(ValueError, match="invalid 'xscale'"):
            Viewport(xscale=[0, float("inf")])

    def test_yscale_must_be_length_2(self):
        with pytest.raises(ValueError, match="invalid 'yscale'"):
            Viewport(yscale=[1])

    def test_yscale_must_be_distinct(self):
        with pytest.raises(ValueError, match="range must be non-zero"):
            Viewport(yscale=[3, 3])

    def test_yscale_must_be_finite(self):
        with pytest.raises(ValueError, match="invalid 'yscale'"):
            Viewport(yscale=[float("nan"), 1])

    def test_angle_must_be_finite(self):
        with pytest.raises(ValueError, match="invalid 'angle'"):
            Viewport(angle=float("inf"))


# ---------------------------------------------------------------------------
# VpList, VpStack, VpTree
# ---------------------------------------------------------------------------


class TestVpContainers:
    """VpList, VpStack, and VpTree construction and basic behaviour."""

    def test_vplist_len(self):
        vl = VpList(Viewport(name="a"), Viewport(name="b"))
        assert len(vl) == 2

    def test_vplist_getitem(self):
        vp_a = Viewport(name="a")
        vl = VpList(vp_a, Viewport(name="b"))
        assert vl[0] is vp_a

    def test_vplist_str(self):
        vl = VpList(Viewport(name="a"), Viewport(name="b"))
        assert "viewport[a]" in str(vl)
        assert "viewport[b]" in str(vl)

    def test_vplist_rejects_non_viewport(self):
        with pytest.raises(TypeError, match="only viewports allowed"):
            VpList("not_a_viewport")

    def test_vpstack_len(self):
        vs = VpStack(Viewport(name="outer"), Viewport(name="inner"))
        assert len(vs) == 2

    def test_vpstack_str_arrow(self):
        vs = VpStack(Viewport(name="outer"), Viewport(name="inner"))
        assert str(vs) == "viewport[outer]->viewport[inner]"

    def test_vpstack_rejects_non_viewport(self):
        with pytest.raises(TypeError, match="only viewports allowed"):
            VpStack(42)

    def test_vptree_parent_children(self):
        p = Viewport(name="p")
        c = VpList(Viewport(name="c1"))
        tree = VpTree(p, c)
        assert tree.parent is p
        assert tree.children is c

    def test_vptree_str(self):
        tree = VpTree(
            Viewport(name="p"),
            VpList(Viewport(name="c1")),
        )
        s = str(tree)
        assert "viewport[p]" in s
        assert "viewport[c1]" in s

    def test_vptree_rejects_bad_parent(self):
        with pytest.raises(TypeError, match="must be a Viewport"):
            VpTree("bad", VpList(Viewport(name="c")))

    def test_vptree_rejects_bad_children(self):
        with pytest.raises(TypeError, match="must be a VpList"):
            VpTree(Viewport(name="p"), Viewport(name="c"))


# ---------------------------------------------------------------------------
# VpPath
# ---------------------------------------------------------------------------


class TestVpPath:
    """VpPath construction and properties."""

    def test_simple_path(self):
        p = VpPath("root", "panel")
        assert str(p) == "root::panel"

    def test_name_property(self):
        p = VpPath("root", "panel", "strip")
        assert p.name == "strip"

    def test_n_property(self):
        p = VpPath("a", "b", "c")
        assert p.n == 3

    def test_split_separator(self):
        p = VpPath("a::b", "c")
        assert p.n == 3
        assert p.name == "c"


# ---------------------------------------------------------------------------
# data_viewport
# ---------------------------------------------------------------------------


class TestDataViewport:
    """data_viewport computes xscale/yscale from data."""

    def test_basic_data(self):
        vp = data_viewport(xData=[1, 2, 3], yData=[10, 20, 30])
        # range 1..3 with 4% extension on each side => 0.92 .. 3.08
        assert vp.xscale[0] < 1.0
        assert vp.xscale[1] > 3.0
        assert vp.yscale[0] < 10.0
        assert vp.yscale[1] > 30.0

    def test_explicit_xscale_overrides_data(self):
        vp = data_viewport(xData=[1, 2], yData=[10, 20], xscale=[0, 100])
        assert vp.xscale == [0.0, 100.0]

    def test_missing_xdata_and_xscale_raises(self):
        with pytest.raises(ValueError, match="must specify at least one"):
            data_viewport(yData=[1, 2])

    def test_missing_ydata_and_yscale_raises(self):
        with pytest.raises(ValueError, match="must specify at least one"):
            data_viewport(xData=[1, 2])

    def test_extension_zero(self):
        vp = data_viewport(xData=[0, 10], yData=[0, 10], extension=0.0)
        assert vp.xscale == [0.0, 10.0]
        assert vp.yscale == [0.0, 10.0]

    def test_name_passthrough(self):
        vp = data_viewport(xData=[1, 2], yData=[3, 4], name="datavp")
        assert vp.name == "datavp"


# ---------------------------------------------------------------------------
# plot_viewport
# ---------------------------------------------------------------------------


class TestPlotViewport:
    """plot_viewport with margin specification."""

    def test_default_margins(self):
        vp = plot_viewport()
        # just should be (left, bottom) => (0.0, 0.0) for left-bottom just
        assert vp.just == (0.0, 0.0)

    def test_custom_margins(self):
        vp = plot_viewport(margins=[1, 2, 3, 4])
        # Should succeed without error; the position should use line units
        assert isinstance(vp.x, Unit)
        assert isinstance(vp.y, Unit)

    def test_name_passthrough(self):
        vp = plot_viewport(name="plotvp")
        assert vp.name == "plotvp"


# ---------------------------------------------------------------------------
# edit_viewport
# ---------------------------------------------------------------------------


class TestEditViewport:
    """edit_viewport returns modified copy."""

    def test_edit_xscale(self):
        vp = Viewport(name="orig", xscale=[0, 10])
        edited = edit_viewport(vp, xscale=[0, 50])
        assert edited.xscale == [0.0, 50.0]
        # Original unchanged.
        assert vp.xscale == [0.0, 10.0]

    def test_edit_preserves_name(self):
        vp = Viewport(name="keep")
        edited = edit_viewport(vp, angle=30)
        assert edited.name == "keep"
        assert edited.angle == 30.0

    def test_edit_change_name(self):
        vp = Viewport(name="old")
        edited = edit_viewport(vp, name="new")
        assert edited.name == "new"


# ---------------------------------------------------------------------------
# depth()
# ---------------------------------------------------------------------------


class TestViewportDepth:
    """depth() function for various viewport types."""

    def test_viewport_depth_is_1(self):
        assert depth(Viewport()) == 1

    def test_vpstack_depth(self):
        vs = VpStack(Viewport(name="a"), Viewport(name="b"))
        assert depth(vs) == 2

    def test_vplist_depth(self):
        vl = VpList(Viewport(name="a"), Viewport(name="b"))
        # depth of last element
        assert depth(vl) == 1

    def test_vptree_depth(self):
        tree = VpTree(
            Viewport(name="p"),
            VpList(Viewport(name="c1")),
        )
        # parent depth (1) + last child depth (1) = 2
        assert depth(tree) == 2

    def test_vppath_depth(self):
        p = VpPath("a", "b", "c")
        assert depth(p) == 3

    def test_depth_bad_type(self):
        with pytest.raises(TypeError, match="does not support"):
            depth("string")


# ---------------------------------------------------------------------------
# Navigation: push_viewport / pop_viewport / up_viewport / current_viewport
# ---------------------------------------------------------------------------


class TestViewportNavigation:
    """Basic viewport navigation using the global state."""

    def test_push_and_current(self, fresh_state):
        """After pushing a viewport, current_viewport should return it."""
        vp = Viewport(name="pushed")
        fresh_state.push_viewport(vp)
        cur = fresh_state.current_viewport()
        assert cur is vp

    def test_push_multiple_nested(self, fresh_state):
        vp1 = Viewport(name="level1")
        vp1.children = []  # Viewport.children defaults to None; initialise for push
        vp2 = Viewport(name="level2")
        fresh_state.push_viewport(vp1)
        fresh_state.push_viewport(vp2)
        assert fresh_state.current_viewport() is vp2

    def test_pop_returns_to_parent(self, fresh_state):
        vp1 = Viewport(name="outer")
        vp1.children = []  # initialise for nested push
        vp2 = Viewport(name="inner")
        fresh_state.push_viewport(vp1)
        fresh_state.push_viewport(vp2)
        fresh_state.pop_viewport(1)
        assert fresh_state.current_viewport() is vp1

    def test_pop_zero_returns_to_root(self, fresh_state):
        vp = Viewport(name="child")
        fresh_state.push_viewport(vp)
        fresh_state.pop_viewport(0)
        root = fresh_state.current_viewport()
        # Root is a dict with name "ROOT"
        assert root["name"] == "ROOT"

    def test_up_does_not_remove(self, fresh_state):
        vp = Viewport(name="child")
        fresh_state.push_viewport(vp)
        fresh_state.up_viewport(1)
        # Child should still be findable in the tree (not removed).
        root = fresh_state.current_viewport()
        children = root.get("children", []) if isinstance(root, dict) else getattr(root, "children", [])
        assert len(children) > 0

    def test_up_zero_to_root(self, fresh_state):
        vp = Viewport(name="child")
        fresh_state.push_viewport(vp)
        fresh_state.up_viewport(0)
        root = fresh_state.current_viewport()
        assert (root["name"] if isinstance(root, dict) else root.name) == "ROOT"

    def test_pop_negative_raises(self, fresh_state):
        with pytest.raises(ValueError):
            fresh_state.pop_viewport(-1)

    def test_up_negative_raises(self, fresh_state):
        with pytest.raises(ValueError):
            fresh_state.up_viewport(-1)

    def test_current_vp_path_at_root(self, fresh_state):
        path = fresh_state.current_vp_path()
        assert "ROOT" in path

    def test_current_vp_path_after_push(self, fresh_state):
        vp = Viewport(name="myvp")
        fresh_state.push_viewport(vp)
        path = fresh_state.current_vp_path()
        assert "myvp" in path

    def test_gp_type_validation(self):
        with pytest.raises(TypeError, match="expected Gpar"):
            Viewport(gp="not_a_gpar")
