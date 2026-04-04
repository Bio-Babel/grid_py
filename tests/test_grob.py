"""Tests for grid_py._grob -- Grob, GTree, GList, GEdit, and related functions."""

from __future__ import annotations

import sys
import os
import copy

import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grid_py"))

from grid_py._grob import (
    Grob,
    GTree,
    GList,
    GEdit,
    GEditList,
    grob_tree,
    grob_name,
    is_grob,
    edit_grob,
    force_grob,
    set_children,
    reorder_grob,
    apply_edit,
    apply_edits,
    _reset_auto_name,
)
from grid_py._gpar import Gpar
from grid_py._path import GPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_counter():
    """Reset the auto-name counter before every test for deterministic names."""
    _reset_auto_name()
    yield
    _reset_auto_name()


# ---------------------------------------------------------------------------
# Grob construction
# ---------------------------------------------------------------------------

class TestGrobConstruction:

    def test_auto_name_default(self):
        g = Grob()
        assert g.name == "GRID.grob.1"

    def test_auto_name_increments(self):
        g1 = Grob()
        g2 = Grob()
        assert g1.name == "GRID.grob.1"
        assert g2.name == "GRID.grob.2"

    def test_explicit_name(self):
        g = Grob(name="mygrob")
        assert g.name == "mygrob"

    def test_grid_class_default(self):
        g = Grob()
        assert g._grid_class == "grob"

    def test_grid_class_custom(self):
        g = Grob(_grid_class="rect")
        assert g._grid_class == "rect"
        # Auto-name uses the grid class as suffix
        assert "rect" in g.name

    def test_gp_slot(self):
        gp = Gpar(col="red")
        g = Grob(gp=gp)
        assert g.gp is gp

    def test_gp_invalid_raises(self):
        with pytest.raises(TypeError, match="invalid 'gp' slot"):
            Grob(gp="not a Gpar")

    def test_vp_slot(self):
        g = Grob(vp="my_vp")
        # String vp should be wrapped into a VpPath
        assert g.vp is not None

    def test_custom_kwargs(self):
        g = Grob(x=1, y=2, colour="blue")
        assert g.x == 1
        assert g.y == 2
        assert g.colour == "blue"

    def test_repr(self):
        g = Grob(name="foo", _grid_class="rect")
        assert repr(g) == "rect[foo]"


# ---------------------------------------------------------------------------
# is_grob
# ---------------------------------------------------------------------------

class TestIsGrob:

    def test_grob_instance(self):
        assert is_grob(Grob()) is True

    def test_gtree_instance(self):
        assert is_grob(GTree()) is True

    def test_non_grob(self):
        assert is_grob("hello") is False
        assert is_grob(42) is False
        assert is_grob(None) is False


# ---------------------------------------------------------------------------
# GTree construction and child management
# ---------------------------------------------------------------------------

class TestGTree:

    def test_construction_empty(self):
        t = GTree(name="tree1")
        assert t.name == "tree1"
        assert t.n_children() == 0
        assert t._grid_class == "gTree"

    def test_construction_with_children(self):
        c1 = Grob(name="child1")
        c2 = Grob(name="child2")
        t = GTree(children=GList(c1, c2), name="tree1")
        assert t.n_children() == 2
        assert t.get_child("child1") is c1

    def test_add_child(self):
        t = GTree(name="tree1")
        c = Grob(name="added")
        t.add_child(c)
        assert t.n_children() == 1
        assert t.get_child("added") is c

    def test_remove_child(self):
        c1 = Grob(name="child1")
        c2 = Grob(name="child2")
        t = GTree(children=GList(c1, c2), name="tree1")
        t.remove_child("child1")
        assert t.n_children() == 1
        with pytest.raises(KeyError):
            t.get_child("child1")

    def test_remove_child_not_found(self):
        t = GTree(name="tree1")
        with pytest.raises(KeyError, match="not found"):
            t.remove_child("nonexistent")

    def test_get_child_not_found(self):
        t = GTree(name="tree1")
        with pytest.raises(KeyError, match="not found"):
            t.get_child("nonexistent")

    def test_get_children_returns_glist(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        t = GTree(children=GList(c1, c2), name="tree1")
        gl = t.get_children()
        assert isinstance(gl, GList)
        assert len(gl) == 2

    def test_set_child_replaces(self):
        c1 = Grob(name="child1")
        t = GTree(children=GList(c1), name="tree1")
        replacement = Grob(name="child1", _grid_class="rect")
        t.set_child("child1", replacement)
        assert t.get_child("child1")._grid_class == "rect"

    def test_set_child_name_mismatch(self):
        c1 = Grob(name="child1")
        t = GTree(children=GList(c1), name="tree1")
        wrong = Grob(name="wrong_name")
        with pytest.raises(ValueError, match="does not match"):
            t.set_child("child1", wrong)

    def test_add_child_non_grob_raises(self):
        t = GTree(name="tree1")
        with pytest.raises(TypeError, match="Grob"):
            t.add_child("not_a_grob")

    def test_children_order_preserved(self):
        c1 = Grob(name="first")
        c2 = Grob(name="second")
        c3 = Grob(name="third")
        t = GTree(children=GList(c1, c2, c3), name="tree1")
        assert t._children_order == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# GList
# ---------------------------------------------------------------------------

class TestGList:

    def test_construction_empty(self):
        gl = GList()
        assert len(gl) == 0

    def test_construction_with_grobs(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        assert len(gl) == 2

    def test_iteration(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        names = [g.name for g in gl]
        assert names == ["a", "b"]

    def test_indexing(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        assert gl[0].name == "a"
        assert gl[1].name == "b"

    def test_slicing_returns_glist(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        g3 = Grob(name="c")
        gl = GList(g1, g2, g3)
        sliced = gl[0:2]
        assert isinstance(sliced, GList)
        assert len(sliced) == 2

    def test_append(self):
        gl = GList()
        g = Grob(name="new")
        gl.append(g)
        assert len(gl) == 1

    def test_append_non_grob_raises(self):
        gl = GList()
        with pytest.raises(TypeError):
            gl.append("not a grob")

    def test_none_filtered(self):
        g1 = Grob(name="a")
        gl = GList(g1, None)
        assert len(gl) == 1

    def test_flatten_nested_glist(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        inner = GList(g1, g2)
        outer = GList(inner)
        assert len(outer) == 2

    def test_non_grob_raises(self):
        with pytest.raises(TypeError, match="only Grob"):
            GList(42)


# ---------------------------------------------------------------------------
# GPath
# ---------------------------------------------------------------------------

class TestGPath:

    def test_single_component(self):
        p = GPath("foo")
        assert p.name == "foo"
        assert p.path is None
        assert p.n == 1

    def test_multi_component(self):
        p = GPath("a", "b", "c")
        assert p.name == "c"
        assert p.path == "a::b"
        assert p.n == 3

    def test_separator_splitting(self):
        p = GPath("a::b::c")
        assert p.n == 3
        assert p.components == ("a", "b", "c")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            GPath("")

    def test_str(self):
        p = GPath("a", "b")
        assert str(p) == "a::b"

    def test_equality(self):
        assert GPath("a", "b") == GPath("a::b")

    def test_hash(self):
        assert hash(GPath("a", "b")) == hash(GPath("a::b"))


# ---------------------------------------------------------------------------
# GEdit and GEditList
# ---------------------------------------------------------------------------

class TestGEdit:

    def test_construction(self):
        e = GEdit(col="red", lwd=2)
        assert e.specs == {"col": "red", "lwd": 2}

    def test_repr(self):
        e = GEdit(x=1)
        assert "GEdit" in repr(e)

    def test_edit_list_construction(self):
        e1 = GEdit(col="red")
        e2 = GEdit(lwd=3)
        el = GEditList(e1, e2)
        assert len(el) == 2

    def test_edit_list_iteration(self):
        e1 = GEdit(col="red")
        e2 = GEdit(lwd=3)
        el = GEditList(e1, e2)
        items = list(el)
        assert len(items) == 2

    def test_edit_list_non_gedit_raises(self):
        with pytest.raises(TypeError, match="GEdit"):
            GEditList("not an edit")


# ---------------------------------------------------------------------------
# apply_edit / apply_edits
# ---------------------------------------------------------------------------

class TestApplyEdit:

    def test_apply_edit_none_is_noop(self):
        g = Grob(name="original")
        result = apply_edit(g, None)
        assert result is g

    def test_apply_edit_changes_attribute(self):
        g = Grob(name="original", x=1)
        e = GEdit(x=42)
        result = apply_edit(g, e)
        assert result.x == 42
        # Original should be unchanged (deep copy)
        assert g.x == 1

    def test_apply_edits_gedit(self):
        g = Grob(name="original", x=0)
        e = GEdit(x=10)
        result = apply_edits(g, e)
        assert result.x == 10

    def test_apply_edits_gedit_list(self):
        g = Grob(name="original", x=0, y=0)
        e1 = GEdit(x=10)
        e2 = GEdit(y=20)
        el = GEditList(e1, e2)
        result = apply_edits(g, el)
        assert result.x == 10
        assert result.y == 20

    def test_apply_edits_none_is_noop(self):
        g = Grob(name="original")
        result = apply_edits(g, None)
        assert result is g

    def test_apply_edit_invalid_raises(self):
        g = Grob(name="original")
        with pytest.raises(TypeError, match="invalid edit"):
            apply_edit(g, "not an edit")


# ---------------------------------------------------------------------------
# edit_grob
# ---------------------------------------------------------------------------

class TestEditGrob:

    def test_edit_returns_copy(self):
        g = Grob(name="orig", x=1)
        edited = edit_grob(g, x=99)
        assert edited.x == 99
        assert g.x == 1

    def test_edit_name(self):
        g = Grob(name="orig")
        edited = edit_grob(g, name="new_name")
        assert edited.name == "new_name"
        assert g.name == "orig"

    def test_edit_gp(self):
        g = Grob(name="orig")
        new_gp = Gpar(col="blue")
        edited = edit_grob(g, gp=new_gp)
        assert edited.gp is not None


# ---------------------------------------------------------------------------
# grob_tree
# ---------------------------------------------------------------------------

class TestGrobTree:

    def test_grob_tree_wraps_children(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        t = grob_tree(c1, c2, name="my_tree")
        assert isinstance(t, GTree)
        assert t.name == "my_tree"
        assert t.n_children() == 2

    def test_grob_tree_auto_name(self):
        t = grob_tree()
        assert t.name is not None
        assert isinstance(t.name, str)

    def test_grob_tree_with_gp(self):
        gp = Gpar(col="green")
        t = grob_tree(name="styled", gp=gp)
        assert t.gp is gp


# ---------------------------------------------------------------------------
# grob_name
# ---------------------------------------------------------------------------

class TestGrobName:

    def test_grob_name_none(self):
        name = grob_name()
        assert name.startswith("GRID.")

    def test_grob_name_from_grob(self):
        g = Grob(_grid_class="circle")
        name = grob_name(g, prefix="TEST")
        assert name.startswith("TEST.circle.")

    def test_grob_name_invalid_raises(self):
        with pytest.raises(TypeError, match="invalid 'grob'"):
            grob_name("not a grob")


# ---------------------------------------------------------------------------
# force_grob
# ---------------------------------------------------------------------------

class TestForceGrob:

    def test_force_grob_returns_grob(self):
        g = Grob(name="forced")
        result = force_grob(g)
        assert isinstance(result, Grob)

    def test_force_grob_stores_original(self):
        g = Grob(name="forced")
        result = force_grob(g)
        assert hasattr(result, "_original")
        assert result._original is g

    def test_force_grob_gtree_forces_children(self):
        c = Grob(name="child")
        t = GTree(children=GList(c), name="tree")
        result = force_grob(t)
        assert isinstance(result, GTree)
        assert result.n_children() == 1


# ---------------------------------------------------------------------------
# set_children
# ---------------------------------------------------------------------------

class TestSetChildren:

    def test_set_children_replaces(self):
        c1 = Grob(name="old")
        t = GTree(children=GList(c1), name="tree")
        c2 = Grob(name="new")
        result = set_children(t, GList(c2))
        assert result.n_children() == 1
        assert result.get_child("new").name == "new"
        # Original unchanged
        assert t.n_children() == 1
        assert t.get_child("old").name == "old"

    def test_set_children_non_gtree_raises(self):
        g = Grob(name="not_a_tree")
        with pytest.raises(TypeError, match="GTree"):
            set_children(g, GList())


# ---------------------------------------------------------------------------
# reorder_grob
# ---------------------------------------------------------------------------

class TestReorderGrob:

    def test_reorder_by_index_back(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        c3 = Grob(name="c")
        t = GTree(children=GList(c1, c2, c3), name="tree")
        result = reorder_grob(t, [2], back=True)
        assert result._children_order[0] == "c"

    def test_reorder_by_name_front(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        c3 = Grob(name="c")
        t = GTree(children=GList(c1, c2, c3), name="tree")
        result = reorder_grob(t, ["c"], back=False)
        assert result._children_order[-1] == "c"

    def test_reorder_invalid_name_raises(self):
        c1 = Grob(name="a")
        t = GTree(children=GList(c1), name="tree")
        with pytest.raises(ValueError, match="not found"):
            reorder_grob(t, ["nonexistent"])

    def test_reorder_invalid_index_raises(self):
        c1 = Grob(name="a")
        t = GTree(children=GList(c1), name="tree")
        with pytest.raises(ValueError, match="out of range"):
            reorder_grob(t, [99])

    def test_reorder_non_gtree_raises(self):
        g = Grob(name="not_a_tree")
        with pytest.raises(TypeError, match="GTree"):
            reorder_grob(g, [0])

    def test_reorder_preserves_original(self):
        c1 = Grob(name="a")
        c2 = Grob(name="b")
        t = GTree(children=GList(c1, c2), name="tree")
        result = reorder_grob(t, [1], back=True)
        assert t._children_order == ["a", "b"]
        assert result._children_order == ["b", "a"]
