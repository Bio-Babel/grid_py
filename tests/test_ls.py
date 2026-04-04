"""Tests for grid_py listing and search operations.

Covers grid_ls, grid_grep, nested_listing, path_listing,
show_grob, get_names, and child_names.
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
    grid_ls,
    grid_grep,
    nested_listing,
    path_listing,
    grob_path_listing,
    show_grob,
    get_names,
    child_names,
    rect_grob,
    circle_grob,
    lines_grob,
    text_grob,
    is_grob,
    get_state,
    Unit,
)
from grid_py._display_list import DLDrawGrob
from grid_py._ls import FlatGridListing
from grid_py._path import GPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_display_list():
    """Return a new DisplayList and install it on the global state."""
    state = get_state()
    dl = DisplayList()
    state.display_list = dl
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
# grid_ls -- listing grobs
# =========================================================================


class TestGridLsBasicGrobs:
    """grid_ls on basic (non-tree) grobs."""

    def test_ls_single_grob(self):
        r = rect_grob(name="r1")
        listing = grid_ls(r, print_=False)
        assert isinstance(listing, FlatGridListing)
        assert len(listing) == 1
        assert listing.names[0] == "r1"

    def test_ls_returns_flat_listing(self):
        r = rect_grob(name="r1")
        listing = grid_ls(r, print_=False)
        assert hasattr(listing, "names")
        assert hasattr(listing, "g_depths")
        assert hasattr(listing, "types")

    def test_ls_grob_type_tag(self):
        r = rect_grob(name="r1")
        listing = grid_ls(r, print_=False)
        assert listing.types[0] == "grobListing"

    def test_ls_no_grobs_flag(self):
        r = rect_grob(name="r1")
        listing = grid_ls(r, grobs=False, print_=False)
        assert len(listing) == 0

    def test_ls_display_list(self):
        """grid_ls(None) should list grobs on the display list."""
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="dlrect"))
        listing = grid_ls(print_=False)
        assert "dlrect" in listing.names

    def test_ls_print_true(self, capsys):
        """grid_ls with print_=True should produce stdout output."""
        r = rect_grob(name="visible")
        grid_ls(r, print_=True)
        captured = capsys.readouterr()
        assert "visible" in captured.out

    def test_ls_print_callable(self):
        """grid_ls with a callable print_ should call it."""
        collected = []
        r = rect_grob(name="cb")
        grid_ls(r, print_=collected.append)
        assert len(collected) == 1

    def test_ls_print_invalid_raises(self):
        r = rect_grob(name="r")
        with pytest.raises(TypeError):
            grid_ls(r, print_=42)  # type: ignore[arg-type]


class TestGridLsGTree:
    """grid_ls on gTree with children."""

    def test_ls_gtree_lists_tree_and_children(self):
        c1 = rect_grob(name="c1")
        c2 = circle_grob(name="c2")
        tree = GTree(children=GList(c1, c2), name="tree1")
        listing = grid_ls(tree, print_=False)
        assert "tree1" in listing.names
        assert "c1" in listing.names
        assert "c2" in listing.names

    def test_ls_gtree_type_tag(self):
        tree = GTree(children=GList(rect_grob(name="ch")), name="tree")
        listing = grid_ls(tree, print_=False)
        assert "gTreeListing" in listing.types

    def test_ls_gtree_non_recursive(self):
        """Non-recursive listing should only include the tree itself."""
        c1 = rect_grob(name="c1")
        tree = GTree(children=GList(c1), name="tree")
        listing = grid_ls(tree, recursive=False, print_=False)
        assert "tree" in listing.names
        assert "c1" not in listing.names

    def test_ls_nested_gtree(self):
        inner = GTree(
            children=GList(rect_grob(name="leaf")),
            name="inner",
        )
        outer = GTree(children=GList(inner), name="outer")
        listing = grid_ls(outer, print_=False)
        assert "outer" in listing.names
        assert "inner" in listing.names
        assert "leaf" in listing.names

    def test_ls_gtree_depth(self):
        c1 = rect_grob(name="c1")
        tree = GTree(children=GList(c1), name="tree")
        listing = grid_ls(tree, print_=False)
        tree_idx = listing.names.index("tree")
        child_idx = listing.names.index("c1")
        assert listing.g_depths[tree_idx] == 0
        assert listing.g_depths[child_idx] == 1


# =========================================================================
# grid_grep -- pattern matching
# =========================================================================


class TestGridGrep:
    """Tests for grid_grep."""

    def test_grep_finds_by_regex(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="alpha_rect"))
        _record_grob(dl, circle_grob(name="beta_circle"))

        result = grid_grep("alpha")
        assert len(result) >= 1

    def test_grep_returns_gpath(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="mygrob"))

        result = grid_grep("mygrob", global_=False)
        assert isinstance(result, GPath)

    def test_grep_global_returns_list(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="dup_1"))
        _record_grob(dl, rect_grob(name="dup_2"))

        result = grid_grep("dup_", global_=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_grep_no_match_returns_empty(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="hello"))

        result = grid_grep("zzz_nothing")
        assert result == []

    def test_grep_on_grob_tree(self):
        inner = rect_grob(name="leaf_node")
        tree = GTree(children=GList(inner), name="thetree")
        result = grid_grep("leaf", x=tree, global_=True)
        assert len(result) >= 1

    def test_grep_exact_match(self):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="exact"))
        result = grid_grep("exact", grep=False, global_=False)
        assert isinstance(result, GPath)

    def test_grep_allDevices_raises(self):
        with pytest.raises(NotImplementedError):
            grid_grep("x", allDevices=True)

    def test_grep_invalid_path_type(self):
        with pytest.raises(TypeError):
            grid_grep(12345)  # type: ignore[arg-type]


# =========================================================================
# Formatting helpers
# =========================================================================


class TestNestedListing:
    """Tests for nested_listing output."""

    def test_nested_listing_output(self, capsys):
        c1 = rect_grob(name="child1")
        tree = GTree(children=GList(c1), name="parent")
        listing = grid_ls(tree, print_=False)
        nested_listing(listing)
        captured = capsys.readouterr()
        assert "parent" in captured.out
        assert "child1" in captured.out

    def test_nested_listing_indentation(self, capsys):
        c1 = rect_grob(name="child1")
        tree = GTree(children=GList(c1), name="parent")
        listing = grid_ls(tree, print_=False)
        nested_listing(listing, gindent=">>")
        captured = capsys.readouterr()
        # The child should be indented
        lines = captured.out.strip().split("\n")
        child_line = [l for l in lines if "child1" in l][0]
        assert child_line.startswith(">>")

    def test_nested_listing_type_error(self):
        with pytest.raises(TypeError):
            nested_listing("not a listing")  # type: ignore[arg-type]


class TestPathListing:
    """Tests for path_listing output."""

    def test_path_listing_output(self, capsys):
        c1 = rect_grob(name="leaf")
        tree = GTree(children=GList(c1), name="root")
        listing = grid_ls(tree, print_=False)
        path_listing(listing)
        captured = capsys.readouterr()
        assert "leaf" in captured.out

    def test_path_listing_empty(self, capsys):
        listing = FlatGridListing()
        path_listing(listing)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_path_listing_type_error(self):
        with pytest.raises(TypeError):
            path_listing(42)  # type: ignore[arg-type]

    def test_grob_path_listing(self, capsys):
        c1 = rect_grob(name="x")
        tree = GTree(children=GList(c1), name="t")
        listing = grid_ls(tree, print_=False)
        grob_path_listing(listing)
        captured = capsys.readouterr()
        # Should contain grob entries
        assert len(captured.out) > 0


# =========================================================================
# show_grob
# =========================================================================


class TestShowGrob:
    """Tests for show_grob."""

    def test_show_grob_direct(self, capsys):
        r = rect_grob(name="shown")
        result = show_grob(r)
        assert result is r
        captured = capsys.readouterr()
        assert "shown" in captured.out

    def test_show_grob_none_lists_dl(self, capsys):
        dl = _fresh_display_list()
        _record_grob(dl, rect_grob(name="dlitem"))
        result = show_grob()
        # Should list the display list -- returns None
        assert result is None

    def test_show_grob_with_gpath(self):
        dl = _fresh_display_list()
        r = rect_grob(name="findme")
        _record_grob(dl, r)
        result = show_grob(gPath="findme")
        assert result is not None
        assert result.name == "findme"

    def test_show_grob_in_gtree(self, capsys):
        child = rect_grob(name="inner")
        tree = GTree(children=GList(child), name="outer")
        result = show_grob(tree, gPath="inner")
        assert result is not None


# =========================================================================
# get_names / child_names (deprecated)
# =========================================================================


class TestGetNamesDeprecated:
    """Tests for deprecated get_names and child_names."""

    def test_get_names_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_names(rect_grob(name="r"))
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_get_names_on_grob(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = get_names(rect_grob(name="solo"))
        assert names == ["solo"]

    def test_get_names_on_gtree(self):
        c1 = rect_grob(name="a")
        c2 = circle_grob(name="b")
        tree = GTree(children=GList(c1, c2), name="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = get_names(tree)
        assert "a" in names
        assert "b" in names

    def test_child_names_warns(self):
        tree = GTree(children=GList(rect_grob(name="x")), name="t")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            child_names(tree)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_child_names_on_gtree(self):
        c1 = rect_grob(name="a")
        c2 = rect_grob(name="b")
        tree = GTree(children=GList(c1, c2), name="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = child_names(tree)
        assert names == ["a", "b"]

    def test_child_names_non_gtree(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            names = child_names(rect_grob(name="leaf"))
        assert names == []


# =========================================================================
# FlatGridListing data structure
# =========================================================================


class TestFlatGridListing:
    """Tests for FlatGridListing operations."""

    def test_extend(self):
        a = FlatGridListing(names=["a"], g_depths=[0], vp_depths=[0],
                            g_paths=[""], vp_paths=[""], types=["grobListing"])
        b = FlatGridListing(names=["b"], g_depths=[1], vp_depths=[0],
                            g_paths=["p"], vp_paths=[""], types=["grobListing"])
        a.extend(b)
        assert len(a) == 2
        assert a.names == ["a", "b"]

    def test_subset(self):
        listing = FlatGridListing(
            names=["a", "b", "c"],
            g_depths=[0, 1, 2],
            vp_depths=[0, 0, 0],
            g_paths=["", "p", "p"],
            vp_paths=["", "", ""],
            types=["grobListing", "gTreeListing", "grobListing"],
        )
        sub = listing.subset([0, 2])
        assert len(sub) == 2
        assert sub.names == ["a", "c"]

    def test_repr(self):
        listing = FlatGridListing(names=["x"])
        assert "1" in repr(listing)
