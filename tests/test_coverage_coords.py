"""Coverage tests for grid_py._coords -- GridCoords, GridGrobCoords,
GridGTreeCoords, grob_coords, grob_points, etc.
"""
from __future__ import annotations

import pytest
import numpy as np

from grid_py._coords import (
    GridCoords,
    GridGrobCoords,
    GridGTreeCoords,
    grob_coords,
    grob_points,
    grid_coords,
    grid_grob_coords,
    grid_gtree_coords,
    empty_coords,
    empty_grob_coords,
    empty_gtree_coords,
    is_empty_coords,
    coords_bbox,
    is_closed,
)
from grid_py._grob import Grob, GTree, GList


# ---------------------------------------------------------------------------
# GridCoords
# ---------------------------------------------------------------------------


class TestGridCoords:
    def test_basic(self):
        c = GridCoords([1, 2, 3], [4, 5, 6])
        assert len(c) == 3
        np.testing.assert_array_equal(c.x, [1, 2, 3])
        np.testing.assert_array_equal(c.y, [4, 5, 6])

    def test_name(self):
        c = GridCoords([1], [2], name="test")
        assert c.name == "test"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            GridCoords([1, 2], [1])

    def test_get_x_y(self):
        c = GridCoords([1, 2], [3, 4])
        np.testing.assert_array_equal(c.get_x(), [1, 2])
        np.testing.assert_array_equal(c.get_y(), [3, 4])

    def test_is_empty_true(self):
        c = GridCoords([0.0], [0.0])
        assert c.is_empty()

    def test_is_empty_false(self):
        c = GridCoords([1.0], [0.0])
        assert not c.is_empty()

    def test_to_device_none(self):
        c = GridCoords([1], [2])
        result = c.to_device(None)
        assert result is c

    def test_to_device_empty(self):
        c = GridCoords([0.0], [0.0])
        result = c.to_device("state")
        # Empty coords return self
        assert result is c

    def test_to_device_with_state(self):
        class MockState:
            def device_loc(self, x, y):
                return x * 2, y * 2
        c = GridCoords([1, 2], [3, 4])
        result = c.to_device(MockState())
        np.testing.assert_array_equal(result.x, [2, 4])
        np.testing.assert_array_equal(result.y, [6, 8])

    def test_from_device_none(self):
        c = GridCoords([1], [2])
        result = c.from_device(None)
        assert result is c

    def test_from_device_matrix(self):
        c = GridCoords([1.0, 2.0], [3.0, 4.0])
        tm = np.eye(3, dtype=float)
        result = c.from_device(tm)
        np.testing.assert_allclose(result.x, c.x, atol=1e-10)

    def test_from_device_with_transform_attr(self):
        class MockTrans:
            transform = np.eye(3, dtype=float)
        c = GridCoords([1.0], [2.0])
        result = c.from_device(MockTrans())
        np.testing.assert_allclose(result.x, c.x, atol=1e-10)

    def test_from_device_bad_shape_raises(self):
        c = GridCoords([1.0], [2.0])
        with pytest.raises(ValueError, match="3x3"):
            c.from_device(np.eye(2))

    def test_transform_coords(self):
        c = GridCoords([1.0, 0.0], [0.0, 1.0])
        tm = np.eye(3, dtype=float)
        result = c.transform_coords(tm)
        np.testing.assert_allclose(result.x, c.x, atol=1e-10)

    def test_flatten(self):
        c = GridCoords([1, 2], [3, 4])
        flat = c.flatten()
        np.testing.assert_array_equal(flat.x, c.x)
        assert flat is not c

    def test_repr(self):
        c = GridCoords([1, 2, 3], [4, 5, 6])
        r = repr(c)
        assert "x:" in r
        assert "y:" in r

    def test_repr_long(self):
        c = GridCoords(np.arange(10), np.arange(10))
        r = repr(c)
        assert "..." in r


# ---------------------------------------------------------------------------
# GridGrobCoords
# ---------------------------------------------------------------------------


class TestGridGrobCoords:
    def test_basic(self):
        c1 = GridCoords([1, 2], [3, 4])
        c2 = GridCoords([5, 6], [7, 8])
        gc = GridGrobCoords([c1, c2], name="test")
        assert len(gc) == 2
        assert gc.name == "test"

    def test_empty(self):
        gc = GridGrobCoords(None, name="empty")
        assert gc.is_empty() is True

    def test_get_x_y(self):
        c1 = GridCoords([1, 2], [3, 4])
        gc = GridGrobCoords([c1], name="test")
        np.testing.assert_array_equal(gc.get_x(), [1, 2])
        np.testing.assert_array_equal(gc.get_y(), [3, 4])

    def test_get_x_y_empty(self):
        gc = GridGrobCoords(None, name="empty")
        assert len(gc.get_x()) == 0
        assert len(gc.get_y()) == 0

    def test_get_x_y_subset(self):
        c1 = GridCoords([1], [2])
        c2 = GridCoords([3], [4])
        gc = GridGrobCoords([c1, c2], name="test")
        np.testing.assert_array_equal(gc.get_x([1]), [3])
        np.testing.assert_array_equal(gc.get_y([1]), [4])

    def test_rule(self):
        gc = GridGrobCoords(None, name="test", rule="winding")
        assert gc.rule == "winding"

    def test_to_device(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="test")
        result = gc.to_device(None)
        assert result.name == "test"

    def test_from_device(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="test")
        result = gc.from_device(None)
        assert result.name == "test"

    def test_is_empty_all_empty(self):
        c = GridCoords([0.0], [0.0])
        gc = GridGrobCoords([c], name="test")
        assert gc.is_empty() is True

    def test_is_empty_not_all_empty(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="test")
        assert gc.is_empty() is False

    def test_transform_coords(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="test")
        tm = np.eye(3, dtype=float)
        result = gc.transform_coords(tm)
        assert len(result) == 1

    def test_flatten(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="test")
        flat = gc.flatten()
        assert len(flat) == 1

    def test_iter(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="test")
        items = list(gc)
        assert len(items) == 1

    def test_getitem(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="test")
        assert isinstance(gc[0], GridCoords)

    def test_repr(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="test", rule="winding")
        r = repr(gc)
        assert "test" in r
        assert "winding" in r


# ---------------------------------------------------------------------------
# GridGTreeCoords
# ---------------------------------------------------------------------------


class TestGridGTreeCoords:
    def test_basic_dict(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        assert len(tc) == 1
        assert tc.name == "tree"

    def test_basic_list(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords([gc], name="tree")
        assert len(tc) == 1

    def test_none(self):
        tc = GridGTreeCoords(None, name="tree")
        assert tc.is_empty() is True

    def test_get_x_y(self):
        c = GridCoords([1, 2], [3, 4])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        np.testing.assert_array_equal(tc.get_x(), [1, 2])
        np.testing.assert_array_equal(tc.get_y(), [3, 4])

    def test_get_x_y_empty(self):
        tc = GridGTreeCoords(None, name="tree")
        assert len(tc.get_x()) == 0
        assert len(tc.get_y()) == 0

    def test_to_device(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        result = tc.to_device(None)
        assert result.name == "tree"

    def test_from_device(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        result = tc.from_device(None)
        assert result.name == "tree"

    def test_transform_coords(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        result = tc.transform_coords(np.eye(3))
        assert len(result) == 1

    def test_flatten(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        flat = tc.flatten()
        assert len(flat) == 1

    def test_is_empty_all(self):
        c = GridCoords([0.0], [0.0])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        assert tc.is_empty() is True

    def test_iter(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        keys = list(tc)
        assert keys == ["child"]

    def test_getitem(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        assert isinstance(tc["child"], GridGrobCoords)

    def test_repr(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="child")
        tc = GridGTreeCoords({"child": gc}, name="tree")
        r = repr(tc)
        assert "tree" in r


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactories:
    def test_grid_coords(self):
        c = grid_coords([1, 2], [3, 4])
        assert isinstance(c, GridCoords)

    def test_grid_grob_coords(self):
        c = grid_coords([1], [2])
        gc = grid_grob_coords([c], "test", rule="winding")
        assert isinstance(gc, GridGrobCoords)

    def test_grid_gtree_coords(self):
        c = grid_coords([1], [2])
        gc = grid_grob_coords([c], "child")
        tc = grid_gtree_coords({"child": gc}, "tree")
        assert isinstance(tc, GridGTreeCoords)


# ---------------------------------------------------------------------------
# Empty / canonical
# ---------------------------------------------------------------------------


class TestEmpty:
    def test_empty_coords(self):
        c = empty_coords()
        assert c.is_empty()

    def test_empty_grob_coords(self):
        gc = empty_grob_coords("test")
        assert gc.is_empty() is True

    def test_empty_gtree_coords(self):
        tc = empty_gtree_coords("test")
        assert tc.is_empty() is True

    def test_is_empty_coords(self):
        c = empty_coords()
        assert is_empty_coords(c)


# ---------------------------------------------------------------------------
# coords_bbox
# ---------------------------------------------------------------------------


class TestCoordsBbox:
    def test_basic(self):
        c = GridCoords([1, 3], [2, 6])
        bbox = coords_bbox(c)
        assert bbox["left"] == 1.0
        assert bbox["bottom"] == 2.0
        assert bbox["width"] == 2.0
        assert bbox["height"] == 4.0

    def test_grob_coords_with_subset(self):
        c1 = GridCoords([0, 10], [0, 10])
        c2 = GridCoords([20, 30], [20, 30])
        gc = GridGrobCoords([c1, c2], name="test")
        bbox = coords_bbox(gc, subset=[0])
        assert bbox["left"] == 0.0
        assert bbox["width"] == 10.0


# ---------------------------------------------------------------------------
# is_closed
# ---------------------------------------------------------------------------


class TestIsClosed:
    def test_rect_closed(self):
        g = Grob(name="r", _grid_class="rect")
        assert is_closed(g) is True

    def test_lines_open(self):
        g = Grob(name="l", _grid_class="lines")
        assert is_closed(g) is False

    def test_polyline_open(self):
        g = Grob(name="p", _grid_class="polyline")
        assert is_closed(g) is False

    def test_segments_open(self):
        g = Grob(name="s", _grid_class="segments")
        assert is_closed(g) is False

    def test_beziergrob_open(self):
        g = Grob(name="b", _grid_class="beziergrob")
        assert is_closed(g) is False

    def test_xspline_open(self):
        # xspline grobs store the flag as ``open_`` (see xspline_grob)
        g = Grob(name="x", _grid_class="xspline", open_=True)
        assert is_closed(g) is False

    def test_xspline_closed(self):
        g = Grob(name="x", _grid_class="xspline", open_=False)
        assert is_closed(g) is True

    def test_points_closed_default(self):
        g = Grob(name="p", _grid_class="points")
        assert is_closed(g) is True

    def test_points_open_pch(self):
        g = Grob(name="p", _grid_class="points", pch=3)
        assert is_closed(g) is False

    def test_points_closed_pch(self):
        g = Grob(name="p", _grid_class="points", pch=1)
        assert is_closed(g) is True

    def test_unknown_class_closed(self):
        g = Grob(name="u", _grid_class="custom")
        assert is_closed(g) is True

    def test_move_to_open(self):
        g = Grob(name="m", _grid_class="move.to")
        assert is_closed(g) is False

    def test_line_to_open(self):
        g = Grob(name="l", _grid_class="line.to")
        assert is_closed(g) is False


# ---------------------------------------------------------------------------
# grob_coords / grob_points dispatchers
# ---------------------------------------------------------------------------


class TestGrobCoordsDispatch:
    def test_grob_coords_plain_grob(self):
        # Plain Grob (no _grid_class) returns empty coords
        g = Grob(name="test")
        result = grob_coords(g)
        assert isinstance(result, GridGrobCoords)
        assert result.is_empty()

    def test_grob_coords_glist(self):
        g1 = Grob(name="a")
        g2 = Grob(name="b")
        gl = GList(g1, g2)
        result = grob_coords(gl)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_coords_unsupported_raises(self):
        with pytest.raises(TypeError, match="does not support"):
            grob_coords("not_a_grob")

    def test_grob_coords_with_closed(self):
        g = Grob(name="test")
        result = grob_coords(g, closed=True)
        assert isinstance(result, GridGrobCoords)
        assert result.is_empty()

    def test_grob_points_plain(self):
        # Plain Grob returns empty coords
        g = Grob(name="test")
        result = grob_points(g)
        assert isinstance(result, GridGrobCoords)
        assert result.is_empty()

    def test_grob_points_glist(self):
        gl = GList(Grob(name="a"))
        result = grob_points(gl)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_points_empty_glist(self):
        gl = GList()
        result = grob_points(gl)
        assert isinstance(result, GridGTreeCoords)
        assert result.is_empty()

    def test_grob_coords_empty_gtree(self):
        tree = GTree(name="tree")
        result = grob_coords(tree)
        assert isinstance(result, GridGTreeCoords)
        assert result.is_empty()

    def test_grob_points_empty_gtree(self):
        tree = GTree(name="tree")
        result = grob_points(tree)
        assert isinstance(result, GridGTreeCoords)
        assert result.is_empty()

    def test_grob_points_unknown_type(self):
        # Something that's not Grob/GTree/GList and has no grob_points method
        class Fake:
            name = "fake"
        result = grob_points(Fake())
        assert isinstance(result, GridGrobCoords)
        assert result.is_empty()


class TestInternalGrobCoordFunctions:
    """Test internal _grob_coords_gtree / _grob_points_gtree directly."""

    def test_grob_coords_gtree_with_children(self):
        from grid_py._coords import _grob_coords_gtree
        child = Grob(name="c")  # plain grob, no _grid_class
        tree = GTree(children=GList(child), name="tree")
        result = _grob_coords_gtree(tree, closed=True)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_coords_gtree_empty(self):
        from grid_py._coords import _grob_coords_gtree
        tree = GTree(name="tree")
        result = _grob_coords_gtree(tree, closed=True)
        assert result.is_empty()

    def test_grob_coords_gtree_no_order(self):
        from grid_py._coords import _grob_coords_gtree
        child = Grob(name="c")  # plain grob
        tree = GTree(children=GList(child), name="tree")
        result = _grob_coords_gtree(tree, closed=True)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_points_gtree_with_children(self):
        from grid_py._coords import _grob_points_gtree
        child = Grob(name="c")  # plain grob
        tree = GTree(children=GList(child), name="tree")
        result = _grob_points_gtree(tree, closed=True)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_points_gtree_empty(self):
        from grid_py._coords import _grob_points_gtree
        tree = GTree(name="tree")
        result = _grob_points_gtree(tree, closed=True)
        assert result.is_empty()

    def test_grob_points_gtree_no_order(self):
        from grid_py._coords import _grob_points_gtree
        child = Grob(name="c")  # plain grob
        tree = GTree(children=GList(child), name="tree")
        result = _grob_points_gtree(tree, closed=True)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_coords_grob_direct(self):
        from grid_py._coords import _grob_coords_grob
        g = Grob(name="test")  # plain grob, no _grid_class
        result = _grob_coords_grob(g, closed=True)
        # Without renderer, returns grob_points which is empty for base Grob
        assert result.is_empty()
