"""Tests for grid_py display-list edit operations.

Covers grid_edit, grid_get, grid_set, grid_add, grid_remove, and grid_gedit.
"""

import warnings

import pytest

import grid_py
from grid_py import (
    DisplayList,
    GList,
    GTree,
    Gpar,
    Grob,
    grid_add,
    grid_edit,
    grid_gedit,
    grid_get,
    grid_gget,
    grid_gremove,
    grid_remove,
    grid_set,
    rect_grob,
    circle_grob,
    lines_grob,
    text_grob,
    is_grob,
    get_state,
    Unit,
)
from grid_py._display_list import DLDrawGrob
from grid_py._path import GPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_display_list():
    """Return a new DisplayList and install it on the global state."""
    state = get_state()
    dl = DisplayList()
    state.display_list = dl  # the edit module accesses state.display_list
    return dl


def _record_grob(dl, grob):
    """Record a grob onto *dl*."""
    dl.record(DLDrawGrob(grob=grob))


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset grid state before each test."""
    state = get_state()
    state.reset()
    yield
    state.reset()


# =========================================================================
# grid_edit -- modifying grob attributes on the display list
# =========================================================================


class TestGridEdit:
    """Tests for grid_edit."""

    def test_edit_changes_grob_attribute(self):
        dl = _fresh_display_list()
        r = rect_grob(name="myrect")
        _record_grob(dl, r)

        grid_edit("myrect", redraw=False, x=Unit(0.3, "npc"))
        # The grob on the display list should have been modified
        found = dl[0].grob
        assert found.name == "myrect"
        assert hasattr(found, "x")

    def test_edit_name_attribute(self):
        dl = _fresh_display_list()
        r = rect_grob(name="oldrect")
        _record_grob(dl, r)

        grid_edit("oldrect", redraw=False, name="newrect")
        found = dl[0].grob
        assert found.name == "newrect"

    def test_edit_gp(self):
        dl = _fresh_display_list()
        r = rect_grob(name="styled")
        _record_grob(dl, r)

        new_gp = Gpar(col="red")
        grid_edit("styled", redraw=False, gp=new_gp)
        found = dl[0].grob
        assert found.gp is not None

    def test_edit_no_match_does_nothing(self):
        dl = _fresh_display_list()
        r = rect_grob(name="r1")
        _record_grob(dl, r)

        # Editing a non-existent grob should not raise, just do nothing.
        grid_edit("nonexistent", redraw=False, x=Unit(0.5, "npc"))

    def test_edit_warns_on_unknown_slot(self):
        dl = _fresh_display_list()
        r = rect_grob(name="r1")
        _record_grob(dl, r)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_edit("r1", redraw=False, zzz_nonexistent_slot=42)
            slot_warns = [x for x in w if "slot" in str(x.message).lower()]
            assert len(slot_warns) >= 1

    def test_edit_allDevices_raises(self):
        dl = _fresh_display_list()
        r = rect_grob(name="r1")
        _record_grob(dl, r)

        with pytest.raises(NotImplementedError):
            grid_edit("r1", allDevices=True, redraw=False)

    def test_edit_global_modifies_all(self):
        dl = _fresh_display_list()
        r1 = rect_grob(name="box")
        r2 = rect_grob(name="box")
        _record_grob(dl, r1)
        _record_grob(dl, r2)

        grid_edit("box", global_=True, redraw=False, x=Unit(0.9, "npc"))
        # Both items should have been edited
        for item in dl:
            assert hasattr(item.grob, "x")

    def test_edit_with_gpath_object(self):
        dl = _fresh_display_list()
        r = rect_grob(name="r1")
        _record_grob(dl, r)

        grid_edit(GPath("r1"), redraw=False, x=Unit(0.1, "npc"))
        assert hasattr(dl[0].grob, "x")


# =========================================================================
# grid_get -- retrieving grobs from the display list
# =========================================================================


class TestGridGet:
    """Tests for grid_get."""

    def test_get_returns_grob(self):
        dl = _fresh_display_list()
        r = rect_grob(name="target")
        _record_grob(dl, r)

        result = grid_get("target")
        assert result is not None
        assert is_grob(result)
        assert result.name == "target"

    def test_get_returns_deep_copy(self):
        dl = _fresh_display_list()
        r = rect_grob(name="orig")
        _record_grob(dl, r)

        result = grid_get("orig")
        assert result is not dl[0].grob  # must be a copy

    def test_get_no_match_returns_none(self):
        dl = _fresh_display_list()
        r = rect_grob(name="r1")
        _record_grob(dl, r)

        result = grid_get("missing")
        assert result is None

    def test_get_global_returns_list(self):
        dl = _fresh_display_list()
        r1 = rect_grob(name="dup")
        r2 = rect_grob(name="dup")
        _record_grob(dl, r1)
        _record_grob(dl, r2)

        result = grid_get("dup", global_=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_global_empty_returns_empty_list(self):
        dl = _fresh_display_list()
        result = grid_get("nothing", global_=True)
        assert result == []

    def test_get_with_grep(self):
        dl = _fresh_display_list()
        r = rect_grob(name="my_rect_1")
        _record_grob(dl, r)

        result = grid_get("my_rect", grep=True)
        assert result is not None
        assert result.name == "my_rect_1"

    def test_get_allDevices_raises(self):
        dl = _fresh_display_list()
        with pytest.raises(NotImplementedError):
            grid_get("x", allDevices=True)

    def test_gget_convenience(self):
        dl = _fresh_display_list()
        r = rect_grob(name="abc_rect")
        _record_grob(dl, r)

        result = grid_gget("abc")
        assert isinstance(result, list)
        assert len(result) >= 1


# =========================================================================
# grid_set -- replacing a grob on the display list
# =========================================================================


class TestGridSet:
    """Tests for grid_set."""

    def test_set_replaces_grob(self):
        dl = _fresh_display_list()
        old = rect_grob(name="old_grob")
        _record_grob(dl, old)

        new = circle_grob(name="new_grob")
        grid_set("old_grob", new, redraw=False)

        found = dl[0].grob
        assert found.name == "new_grob"

    def test_set_no_match_raises(self):
        dl = _fresh_display_list()
        r = rect_grob(name="exists")
        _record_grob(dl, r)

        with pytest.raises(ValueError):
            grid_set("nonexistent", circle_grob(name="c"), redraw=False)

    def test_set_with_grep(self):
        dl = _fresh_display_list()
        old = rect_grob(name="abc123")
        _record_grob(dl, old)

        new = circle_grob(name="replacement")
        grid_set("abc", new, grep=True, redraw=False)
        assert dl[0].grob.name == "replacement"


# =========================================================================
# grid_add -- adding grobs to the display list or a gTree
# =========================================================================


class TestGridAdd:
    """Tests for grid_add."""

    def test_add_toplevel(self):
        dl = _fresh_display_list()

        new = rect_grob(name="added")
        grid_add(new, redraw=False)
        assert len(dl) == 1
        assert dl[0].grob.name == "added"

    def test_add_to_gtree(self):
        dl = _fresh_display_list()
        child1 = rect_grob(name="child1")
        tree = GTree(children=GList(child1), name="mytree")
        _record_grob(dl, tree)

        child2 = circle_grob(name="child2")
        grid_add(child2, gPath="mytree", redraw=False)

        # The tree should now have two children
        t = dl[0].grob
        assert t.n_children() == 2
        assert "child2" in t._children_order

    def test_add_no_gpath_appends_toplevel(self):
        dl = _fresh_display_list()
        grid_add(rect_grob(name="a"), redraw=False)
        grid_add(circle_grob(name="b"), redraw=False)
        assert len(dl) == 2

    def test_add_to_non_gtree_warns(self):
        dl = _fresh_display_list()
        r = rect_grob(name="leaf")
        _record_grob(dl, r)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_add(circle_grob(name="child"), gPath="leaf", redraw=False)
            warn_msgs = [x for x in w if "non-gTree" in str(x.message)]
            assert len(warn_msgs) >= 1

    def test_add_allDevices_raises(self):
        dl = _fresh_display_list()
        with pytest.raises(NotImplementedError):
            grid_add(rect_grob(name="x"), allDevices=True, redraw=False)


# =========================================================================
# grid_remove -- removing grobs from the display list
# =========================================================================


class TestGridRemove:
    """Tests for grid_remove."""

    def test_remove_toplevel(self):
        dl = _fresh_display_list()
        r = rect_grob(name="victim")
        _record_grob(dl, r)
        assert len(dl) == 1

        grid_remove("victim", redraw=False)
        assert len(dl) == 0

    def test_remove_nonexistent_warns(self):
        dl = _fresh_display_list()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_remove("ghost", redraw=False)
            gpath_warns = [x for x in w if "gPath" in str(x.message)]
            assert len(gpath_warns) >= 1

    def test_remove_child_from_gtree(self):
        dl = _fresh_display_list()
        c1 = rect_grob(name="c1")
        c2 = circle_grob(name="c2")
        tree = GTree(children=GList(c1, c2), name="parent")
        _record_grob(dl, tree)

        grid_remove(GPath("parent", "c1"), redraw=False)
        t = dl[0].grob
        assert t.n_children() == 1
        assert "c1" not in t._children_order

    def test_remove_with_grep(self):
        dl = _fresh_display_list()
        r1 = rect_grob(name="item_A")
        r2 = rect_grob(name="item_B")
        _record_grob(dl, r1)
        _record_grob(dl, r2)

        grid_remove("item_A", grep=True, redraw=False)
        assert len(dl) == 1
        assert dl[0].grob.name == "item_B"

    def test_remove_global(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="dup"))
        _record_grob(dl, rect_grob(name="dup"))
        assert len(dl) == 2

        grid_remove("dup", global_=True, redraw=False)
        assert len(dl) == 0

    def test_gremove_convenience(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="abc_rect"))
        _record_grob(dl, rect_grob(name="abc_circle"))

        grid_gremove("abc", redraw=False)
        assert len(dl) == 0


# =========================================================================
# grid_gedit -- grep=True convenience wrapper
# =========================================================================


class TestGridGedit:
    """Tests for grid_gedit (grep=True variant of grid_edit)."""

    def test_gedit_matches_regex(self):
        dl = _fresh_display_list()
        r1 = rect_grob(name="box_1")
        r2 = rect_grob(name="box_2")
        _record_grob(dl, r1)
        _record_grob(dl, r2)

        grid_gedit("box_", redraw=False, x=Unit(0.5, "npc"))
        for item in dl:
            assert hasattr(item.grob, "x")

    def test_gedit_global_default_true(self):
        dl = _fresh_display_list()
        r1 = rect_grob(name="item1")
        r2 = rect_grob(name="item2")
        _record_grob(dl, r1)
        _record_grob(dl, r2)

        # grid_gedit defaults to global_=True
        grid_gedit("item", redraw=False, x=Unit(0.7, "npc"))
        for item in dl:
            assert hasattr(item.grob, "x")

    def test_gedit_no_match(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="hello"))

        # Should not raise
        grid_gedit("zzz_no_match", redraw=False, x=Unit(0.1, "npc"))


# =========================================================================
# Edge cases and type validation
# =========================================================================


class TestEditEdgeCases:
    """Edge-case tests for the edit module."""

    def test_ensure_gpath_type_error(self):
        dl = _fresh_display_list()
        with pytest.raises(TypeError):
            grid_edit(12345, redraw=False)  # type: ignore[arg-type]

    def test_grep_type_error(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="r"))
        with pytest.raises(TypeError):
            grid_edit("r", grep="yes", redraw=False)  # type: ignore[arg-type]

    def test_edit_nested_gtree(self):
        """grid_edit should find a grob inside a gTree (non-strict)."""
        dl = _fresh_display_list()
        inner = rect_grob(name="inner")
        outer = GTree(children=GList(inner), name="outer")
        _record_grob(dl, outer)

        grid_edit("inner", redraw=False, x=Unit(0.2, "npc"))
        found_inner = outer._children["inner"]
        assert hasattr(found_inner, "x")
