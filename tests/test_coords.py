"""Tests for grid_py._coords -- coordinate query system."""

import numpy as np
import pytest

from grid_py._coords import (
    GridCoords,
    GridGrobCoords,
    GridGTreeCoords,
    coords_bbox,
    empty_coords,
    empty_grob_coords,
    empty_gtree_coords,
    grid_coords,
    grid_grob_coords,
    grid_gtree_coords,
    grob_coords,
    grob_points,
    is_closed,
    is_empty_coords,
)
from grid_py._grob import GList, GTree, Grob
from grid_py._primitives import rect_grob, null_grob


# ------------------------------------------------------------------ #
# GridCoords                                                         #
# ------------------------------------------------------------------ #


class TestGridCoords:
    """GridCoords construction and methods."""

    def test_basic_construction(self):
        c = GridCoords([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_array_equal(c.x, [1.0, 2.0])
        np.testing.assert_array_equal(c.y, [3.0, 4.0])

    def test_scalar_input(self):
        c = GridCoords(5.0, 10.0)
        assert len(c) == 1
        assert c.x[0] == 5.0
        assert c.y[0] == 10.0

    def test_name_default(self):
        c = GridCoords([0], [0])
        assert c.name == "coords"

    def test_name_custom(self):
        c = GridCoords([0], [0], name="mycoords")
        assert c.name == "mycoords"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            GridCoords([1, 2], [1, 2, 3])

    def test_len(self):
        c = GridCoords([1, 2, 3], [4, 5, 6])
        assert len(c) == 3

    def test_get_x(self):
        c = GridCoords([10, 20], [30, 40])
        np.testing.assert_array_equal(c.get_x(), [10.0, 20.0])

    def test_get_y(self):
        c = GridCoords([10, 20], [30, 40])
        np.testing.assert_array_equal(c.get_y(), [30.0, 40.0])

    def test_repr(self):
        c = GridCoords([1.0, 2.0], [3.0, 4.0])
        r = repr(c)
        assert "x:" in r
        assert "y:" in r

    def test_flatten_returns_copy(self):
        c = GridCoords([1, 2], [3, 4])
        f = c.flatten()
        assert f is not c
        np.testing.assert_array_equal(f.x, c.x)

    def test_is_empty_true(self):
        """A single (0,0) point is considered empty."""
        c = GridCoords([0.0], [0.0])
        assert bool(c.is_empty()) is True

    def test_is_empty_false(self):
        c = GridCoords([1.0], [0.0])
        assert bool(c.is_empty()) is False


# ------------------------------------------------------------------ #
# GridCoords -- transform operations                                 #
# ------------------------------------------------------------------ #


class TestGridCoordsTransform:
    """Coordinate transformation methods on GridCoords."""

    def test_to_device_no_state(self):
        """to_device with state=None returns self."""
        c = GridCoords([1, 2], [3, 4])
        result = c.to_device(None)
        assert result is c

    def test_to_device_empty(self):
        c = GridCoords([0.0], [0.0])
        result = c.to_device(state=None)
        assert result is c

    def test_from_device_no_state(self):
        c = GridCoords([1, 2], [3, 4])
        result = c.from_device(None)
        assert result is c

    def test_from_device_identity_matrix(self):
        """Identity transform should return same coordinates."""
        c = GridCoords([1.0, 2.0], [3.0, 4.0])
        identity = np.eye(3)
        result = c.from_device(identity)
        np.testing.assert_array_almost_equal(result.x, c.x)
        np.testing.assert_array_almost_equal(result.y, c.y)

    def test_from_device_bad_shape_raises(self):
        c = GridCoords([1], [2])
        with pytest.raises(ValueError, match="3x3"):
            c.from_device(np.eye(2))

    def test_transform_coords_identity(self):
        c = GridCoords([1.0, 2.0], [3.0, 4.0])
        result = c.transform_coords(np.eye(3))
        np.testing.assert_array_almost_equal(result.x, [1.0, 2.0])
        np.testing.assert_array_almost_equal(result.y, [3.0, 4.0])

    def test_transform_coords_translation(self):
        """A translation matrix should shift coordinates."""
        c = GridCoords([0.0, 1.0], [0.0, 1.0])
        tm = np.eye(3)
        tm[0, 2] = 10.0  # shift x by 10
        tm[1, 2] = 20.0  # shift y by 20
        # Note: transform_coords does pts @ tm, so we need the transpose form
        tm_t = np.eye(3)
        tm_t[2, 0] = 10.0
        tm_t[2, 1] = 20.0
        result = c.transform_coords(tm_t)
        np.testing.assert_array_almost_equal(result.x, [10.0, 11.0])
        np.testing.assert_array_almost_equal(result.y, [20.0, 21.0])


# ------------------------------------------------------------------ #
# GridGrobCoords                                                     #
# ------------------------------------------------------------------ #


class TestGridGrobCoords:
    """GridGrobCoords construction and methods."""

    def test_construction_empty(self):
        gc = GridGrobCoords()
        assert len(gc) == 0

    def test_construction_with_coords(self):
        c1 = GridCoords([1, 2], [3, 4])
        c2 = GridCoords([5, 6], [7, 8])
        gc = GridGrobCoords([c1, c2], name="test")
        assert len(gc) == 2
        assert gc.name == "test"

    def test_rule(self):
        gc = GridGrobCoords([], name="test", rule="winding")
        assert gc.rule == "winding"

    def test_get_x_all(self):
        c1 = GridCoords([1, 2], [0, 0])
        c2 = GridCoords([3, 4], [0, 0])
        gc = GridGrobCoords([c1, c2], name="test")
        np.testing.assert_array_equal(gc.get_x(), [1, 2, 3, 4])

    def test_get_y_all(self):
        c1 = GridCoords([0, 0], [10, 20])
        c2 = GridCoords([0, 0], [30, 40])
        gc = GridGrobCoords([c1, c2], name="test")
        np.testing.assert_array_equal(gc.get_y(), [10, 20, 30, 40])

    def test_get_x_subset(self):
        c1 = GridCoords([1], [0])
        c2 = GridCoords([2], [0])
        c3 = GridCoords([3], [0])
        gc = GridGrobCoords([c1, c2, c3], name="test")
        np.testing.assert_array_equal(gc.get_x(subset=[0, 2]), [1, 3])

    def test_get_x_empty(self):
        gc = GridGrobCoords([], name="test")
        result = gc.get_x()
        assert len(result) == 0

    def test_is_empty_true(self):
        gc = GridGrobCoords([], name="test")
        assert gc.is_empty() is True

    def test_is_empty_with_empty_coords(self):
        ec = GridCoords([0.0], [0.0])
        gc = GridGrobCoords([ec], name="test")
        assert gc.is_empty() is True

    def test_is_empty_false(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="test")
        assert gc.is_empty() is False

    def test_iter(self):
        c1 = GridCoords([1], [2])
        c2 = GridCoords([3], [4])
        gc = GridGrobCoords([c1, c2], name="test")
        items = list(gc)
        assert len(items) == 2

    def test_getitem(self):
        c1 = GridCoords([1], [2])
        c2 = GridCoords([3], [4])
        gc = GridGrobCoords([c1, c2], name="test")
        np.testing.assert_array_equal(gc[0].x, [1])
        np.testing.assert_array_equal(gc[1].x, [3])

    def test_transform_coords(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="test")
        result = gc.transform_coords(np.eye(3))
        assert isinstance(result, GridGrobCoords)
        np.testing.assert_array_almost_equal(result[0].x, [1.0])

    def test_flatten(self):
        c = GridCoords([1.0], [2.0])
        gc = GridGrobCoords([c], name="test")
        result = gc.flatten()
        assert isinstance(result, GridGrobCoords)
        assert result is not gc

    def test_repr(self):
        c = GridCoords([1], [2])
        gc = GridGrobCoords([c], name="mygrob")
        r = repr(gc)
        assert "mygrob" in r


# ------------------------------------------------------------------ #
# GridGTreeCoords                                                    #
# ------------------------------------------------------------------ #


class TestGridGTreeCoords:
    """GridGTreeCoords construction and methods."""

    def test_construction_empty(self):
        gt = GridGTreeCoords()
        assert len(gt) == 0

    def test_construction_with_dict(self):
        gc = GridGrobCoords([GridCoords([1], [2])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        assert len(gt) == 1
        assert gt.name == "tree"
        assert "child" in gt

    def test_construction_with_list(self):
        gc = GridGrobCoords([GridCoords([1], [2])], name="child")
        gt = GridGTreeCoords([gc], name="tree")
        assert len(gt) == 1

    def test_get_x(self):
        gc = GridGrobCoords([GridCoords([10, 20], [0, 0])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        np.testing.assert_array_equal(gt.get_x(), [10, 20])

    def test_get_y(self):
        gc = GridGrobCoords([GridCoords([0, 0], [30, 40])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        np.testing.assert_array_equal(gt.get_y(), [30, 40])

    def test_get_x_empty(self):
        gt = GridGTreeCoords(None, name="empty")
        assert len(gt.get_x()) == 0

    def test_is_empty_true(self):
        gt = GridGTreeCoords(None, name="empty")
        assert gt.is_empty() is True

    def test_is_empty_false(self):
        gc = GridGrobCoords([GridCoords([1.0], [2.0])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        assert gt.is_empty() is False

    def test_getitem(self):
        gc = GridGrobCoords([GridCoords([5], [6])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        assert gt["child"] is gc

    def test_iter(self):
        gc = GridGrobCoords([], name="a")
        gt = GridGTreeCoords({"a": gc}, name="tree")
        assert list(gt) == ["a"]

    def test_transform_coords(self):
        gc = GridGrobCoords([GridCoords([1.0], [2.0])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        result = gt.transform_coords(np.eye(3))
        assert isinstance(result, GridGTreeCoords)

    def test_flatten(self):
        gc = GridGrobCoords([GridCoords([1], [2])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="tree")
        result = gt.flatten()
        assert isinstance(result, GridGTreeCoords)
        assert result is not gt

    def test_repr(self):
        gc = GridGrobCoords([GridCoords([1], [2])], name="child")
        gt = GridGTreeCoords({"child": gc}, name="mytree")
        r = repr(gt)
        assert "mytree" in r


# ------------------------------------------------------------------ #
# Factory functions                                                  #
# ------------------------------------------------------------------ #


class TestFactoryFunctions:
    """grid_coords, grid_grob_coords, grid_gtree_coords."""

    def test_grid_coords(self):
        c = grid_coords([1, 2], [3, 4])
        assert isinstance(c, GridCoords)

    def test_grid_grob_coords(self):
        c = grid_coords([1], [2])
        gc = grid_grob_coords([c], name="test")
        assert isinstance(gc, GridGrobCoords)

    def test_grid_gtree_coords(self):
        gc = grid_grob_coords([grid_coords([1], [2])], name="child")
        gt = grid_gtree_coords({"child": gc}, name="tree")
        assert isinstance(gt, GridGTreeCoords)


# ------------------------------------------------------------------ #
# Empty coords                                                       #
# ------------------------------------------------------------------ #


class TestEmptyCoords:
    """empty_coords, empty_grob_coords, empty_gtree_coords."""

    def test_empty_coords(self):
        ec = empty_coords()
        assert isinstance(ec, GridCoords)
        assert bool(ec.is_empty()) is True
        assert len(ec) == 1
        assert ec.x[0] == 0.0
        assert ec.y[0] == 0.0

    def test_empty_grob_coords(self):
        egc = empty_grob_coords()
        assert isinstance(egc, GridGrobCoords)
        assert egc.is_empty() is True

    def test_empty_grob_coords_custom_name(self):
        egc = empty_grob_coords(name="custom")
        assert egc.name == "custom"

    def test_empty_gtree_coords(self):
        egt = empty_gtree_coords()
        assert isinstance(egt, GridGTreeCoords)
        assert egt.is_empty() is True

    def test_empty_gtree_coords_custom_name(self):
        egt = empty_gtree_coords(name="custom")
        assert egt.name == "custom"


# ------------------------------------------------------------------ #
# is_empty_coords                                                    #
# ------------------------------------------------------------------ #


class TestIsEmptyCoords:
    """is_empty_coords dispatches to .is_empty()."""

    def test_empty_gridcoords(self):
        assert bool(is_empty_coords(empty_coords())) is True

    def test_nonempty_gridcoords(self):
        c = GridCoords([1.0], [2.0])
        assert bool(is_empty_coords(c)) is False

    def test_empty_grob_coords(self):
        assert bool(is_empty_coords(empty_grob_coords())) is True

    def test_empty_gtree_coords(self):
        assert bool(is_empty_coords(empty_gtree_coords())) is True


# ------------------------------------------------------------------ #
# coords_bbox                                                        #
# ------------------------------------------------------------------ #


class TestCoordsBbox:
    """coords_bbox computes axis-aligned bounding boxes."""

    def test_basic_bbox(self):
        c = GridCoords([0, 10, 5], [2, 8, 5])
        bbox = coords_bbox(c)
        assert bbox["left"] == 0.0
        assert bbox["bottom"] == 2.0
        assert bbox["width"] == 10.0
        assert bbox["height"] == 6.0

    def test_grob_coords_bbox(self):
        c1 = GridCoords([0, 5], [0, 5])
        c2 = GridCoords([10, 15], [10, 15])
        gc = GridGrobCoords([c1, c2], name="test")
        bbox = coords_bbox(gc)
        assert bbox["left"] == 0.0
        assert bbox["bottom"] == 0.0
        assert bbox["width"] == 15.0
        assert bbox["height"] == 15.0

    def test_grob_coords_bbox_subset(self):
        c1 = GridCoords([0, 5], [0, 5])
        c2 = GridCoords([100, 200], [100, 200])
        gc = GridGrobCoords([c1, c2], name="test")
        bbox = coords_bbox(gc, subset=[0])
        assert bbox["left"] == 0.0
        assert bbox["width"] == 5.0


# ------------------------------------------------------------------ #
# grob_coords / grob_points on basic grobs                          #
# ------------------------------------------------------------------ #


class TestGrobCoordsDispatch:
    """grob_coords and grob_points dispatch on grob type."""

    def test_grob_coords_plain_grob(self):
        """A plain Grob has a grob_coords method that returns None by default."""
        g = Grob(name="plain")
        result = grob_coords(g)
        # Grob.grob_coords() returns None (no coordinate implementation)
        assert result is None

    def test_grob_coords_null_grob(self):
        """null_grob has a grob_coords method that returns None by default."""
        ng = null_grob()
        result = grob_coords(ng)
        assert result is None

    def test_grob_coords_gtree(self):
        """Calling grob_coords on a GTree dispatches through grob_coords method."""
        child = Grob(name="child1")
        tree = GTree(children=GList(child), name="tree1")
        result = grob_coords(tree)
        # GTree.grob_coords returns None or a GridGTreeCoords
        # The hasattr path is taken, returning whatever the method produces
        assert result is None or isinstance(result, GridGTreeCoords)

    def test_grob_coords_glist(self):
        """Calling grob_coords on a GList returns GridGTreeCoords."""
        g1 = Grob(name="g1")
        g2 = Grob(name="g2")
        gl = GList(g1, g2)
        # GList does not have grob_coords method, so the isinstance branch fires
        result = grob_coords(gl)
        assert isinstance(result, GridGTreeCoords)

    def test_grob_points_plain_grob(self):
        """grob_points on a plain Grob: method returns None by default."""
        g = Grob(name="plain")
        result = grob_points(g)
        # Grob.grob_points() returns None
        assert result is None

    def test_grob_points_gtree_via_method(self):
        """grob_points on a GTree dispatches through grob_points method."""
        child = Grob(name="child")
        tree = GTree(children=GList(child), name="tree")
        result = grob_points(tree)
        assert result is None or isinstance(result, GridGTreeCoords)

    def test_grob_coords_unsupported_type(self):
        with pytest.raises(TypeError, match="does not support"):
            grob_coords("not_a_grob")


# ------------------------------------------------------------------ #
# is_closed                                                          #
# ------------------------------------------------------------------ #


class TestIsClosed:
    """is_closed defaults based on grob class."""

    def test_rect_is_closed(self):
        g = Grob(name="r", _grid_class="rect")
        assert is_closed(g) is True

    def test_lines_is_open(self):
        g = Grob(name="l", _grid_class="lines")
        assert is_closed(g) is False

    def test_polyline_is_open(self):
        g = Grob(name="p", _grid_class="polyline")
        assert is_closed(g) is False

    def test_segments_is_open(self):
        g = Grob(name="s", _grid_class="segments")
        assert is_closed(g) is False

    def test_unknown_class_defaults_closed(self):
        g = Grob(name="u", _grid_class="custom_thing")
        assert is_closed(g) is True
