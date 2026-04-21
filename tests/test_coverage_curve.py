"""Tests for grid_py._curve -- curve, xspline, and bezier grobs."""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._curve import (
    _calc_bezier_points,
    _calc_control_points,
    _calc_curve_points,
    _calc_origin,
    _calc_square_control_points,
    _calc_xspline_points,
    _ensure_unit,
    _interleave,
    arc_curvature,
    bezier_grob,
    bezier_points,
    curve_grob,
    grid_bezier,
    grid_curve,
    grid_xspline,
    xspline_grob,
    xspline_points,
)
from grid_py._grob import GTree, Grob
from grid_py._units import Unit


# ---------------------------------------------------------------------------
# _ensure_unit
# ---------------------------------------------------------------------------

class TestEnsureUnit:

    def test_already_unit(self):
        u = Unit(1, "cm")
        result = _ensure_unit(u, "npc")
        assert result is u

    def test_numeric(self):
        result = _ensure_unit(0.5, "npc")
        assert isinstance(result, Unit)


# ---------------------------------------------------------------------------
# arc_curvature
# ---------------------------------------------------------------------------

class TestArcCurvature:

    def test_collinear_points(self):
        c = arc_curvature(0, 0, 1, 1, 2, 2)
        assert c == 0.0

    def test_coincident_points(self):
        c = arc_curvature(0, 0, 0, 0, 0, 0)
        assert c == 0.0

    def test_right_angle(self):
        c = arc_curvature(0, 0, 1, 0, 1, 1)
        assert c != 0.0

    def test_sign(self):
        c1 = arc_curvature(0, 0, 1, 0, 1, 1)
        c2 = arc_curvature(0, 0, 1, 0, 1, -1)
        # Opposite curvatures
        assert c1 * c2 < 0


# ---------------------------------------------------------------------------
# _calc_origin
# ---------------------------------------------------------------------------

class TestCalcOrigin:

    def test_basic(self):
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([1.0])
        ox, oy = _calc_origin(x1, y1, x2, y2, 0.5, "right")
        assert len(ox) == 1
        assert len(oy) == 1

    def test_vertical_line(self):
        x1 = np.array([1.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([2.0])
        ox, oy = _calc_origin(x1, y1, x2, y2, 0.5, "left")
        assert len(ox) == 1

    def test_horizontal_line(self):
        x1 = np.array([0.0])
        y1 = np.array([1.0])
        x2 = np.array([2.0])
        y2 = np.array([1.0])
        ox, oy = _calc_origin(x1, y1, x2, y2, 0.5, "right")
        assert len(ox) == 1


# ---------------------------------------------------------------------------
# _calc_control_points
# ---------------------------------------------------------------------------

class TestCalcControlPoints:

    def test_basic(self):
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([1.0])
        cpx, cpy = _calc_control_points(x1, y1, x2, y2, 1.0, 90.0, 2)
        assert len(cpx) == 2
        assert len(cpy) == 2

    def test_negative_curvature(self):
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([0.5])
        cpx, cpy = _calc_control_points(x1, y1, x2, y2, -1.0, 90.0, 1)
        assert len(cpx) == 1

    def test_auto_angle(self):
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([1.0])
        cpx, cpy = _calc_control_points(x1, y1, x2, y2, 1.0, None, 1)
        assert len(cpx) == 1


# ---------------------------------------------------------------------------
# _interleave
# ---------------------------------------------------------------------------

class TestInterleave:

    def test_basic(self):
        val = np.array([10.0, 20.0])
        sval = np.array([0.0])
        eval_ = np.array([30.0])
        end = np.array([True])
        result = _interleave(2, 1, val, sval, eval_, end)
        assert len(result) == 3

    def test_prepend(self):
        val = np.array([10.0])
        sval = np.array([0.0])
        eval_ = np.array([30.0])
        end = np.array([False])
        result = _interleave(1, 1, val, sval, eval_, end)
        assert len(result) == 2
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# _calc_square_control_points
# ---------------------------------------------------------------------------

class TestCalcSquareControlPoints:

    def test_basic(self):
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([1.0])
        cpx, cpy, end = _calc_square_control_points(x1, y1, x2, y2, 1.0, 90.0, 1)
        assert len(end) == 1

    def test_negative_curvature(self):
        x1 = np.array([0.0])
        y1 = np.array([0.0])
        x2 = np.array([1.0])
        y2 = np.array([0.5])
        cpx, cpy, end = _calc_square_control_points(x1, y1, x2, y2, -1.0, 90.0, 1)
        assert len(end) == 1


# ---------------------------------------------------------------------------
# _calc_curve_points
# ---------------------------------------------------------------------------

class TestCalcCurvePoints:

    def test_straight_line(self):
        xs, ys = _calc_curve_points(0, 0, 1, 1, curvature=0)
        np.testing.assert_array_equal(xs, [0, 1])
        np.testing.assert_array_equal(ys, [0, 1])

    def test_extreme_angle_straight(self):
        xs, ys = _calc_curve_points(0, 0, 1, 1, angle=0)
        np.testing.assert_array_equal(xs, [0, 1])

    def test_identical_endpoints_raises(self):
        with pytest.raises(ValueError, match="identical"):
            _calc_curve_points(1, 1, 1, 1)

    def test_curved(self):
        xs, ys = _calc_curve_points(0, 0, 1, 1, curvature=1.0, angle=90, ncp=2)
        assert len(xs) > 2

    def test_inflect(self):
        xs, ys = _calc_curve_points(
            0, 0, 2, 2, curvature=1.0, angle=90, ncp=1, inflect=True
        )
        assert len(xs) > 2

    def test_inflect_non_square(self):
        xs, ys = _calc_curve_points(
            0, 0, 2, 2, curvature=1.0, angle=90, ncp=1,
            inflect=True, square=False,
        )
        assert len(xs) > 2

    def test_non_square(self):
        xs, ys = _calc_curve_points(
            0, 0, 1, 1, curvature=1.0, angle=90, ncp=2, square=False
        )
        assert len(xs) > 2


# ---------------------------------------------------------------------------
# _calc_xspline_points
# ---------------------------------------------------------------------------

class TestCalcXsplinePoints:

    def test_single_point(self):
        x = np.array([1.0])
        y = np.array([2.0])
        rx, ry = _calc_xspline_points(x, y)
        np.testing.assert_array_equal(rx, x)
        np.testing.assert_array_equal(ry, y)

    def test_two_points(self):
        # Default shape=0 (sharp corners) with 2 CPs: Blanc-Schlick degenerates
        # to a straight line through the control points, yielding exactly the
        # 2 endpoints (matches R's ``xsplinePoints``).
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y)
        assert len(rx) == 2
        np.testing.assert_allclose(rx, x)
        np.testing.assert_allclose(ry, y)

    def test_two_points_with_shape(self):
        # With shape=0.5 both CPs duplicated, still returns 2 points (degenerate
        # segment — matches R's GEXspline).
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y, shape=0.5)
        assert len(rx) == 2

    def test_closed(self):
        # Closed 3-point spline with shape=0: Blanc-Schlick outputs 3
        # polygon vertices (matches R).
        x = np.array([0.0, 1.0, 0.5])
        y = np.array([0.0, 0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y, open_=False)
        assert len(rx) == 3

    def test_closed_with_shape(self):
        # Closed 3-point spline with shape=0.5 interpolates; expect more
        # than 3 output points.
        x = np.array([0.0, 1.0, 0.5])
        y = np.array([0.0, 0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y, shape=0.5, open_=False)
        assert len(rx) > 3

    def test_no_rep_ends_requires_four_cps(self):
        # Matches R's error: "there must be at least four control points".
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="four control points"):
            _calc_xspline_points(x, y, repEnds=False)

    def test_no_rep_ends_with_four_cps(self):
        # repEnds=False with 4 CPs interpolates without endpoint replication.
        x = np.array([0.0, 0.33, 0.67, 1.0])
        y = np.array([0.0, 1.0, 1.0, 0.0])
        rx, ry = _calc_xspline_points(x, y, shape=0.5, repEnds=False)
        assert len(rx) >= 2


# ---------------------------------------------------------------------------
# _calc_bezier_points
# ---------------------------------------------------------------------------

class TestCalcBezierPoints:

    def test_single_point(self):
        x = np.array([1.0])
        y = np.array([2.0])
        rx, ry = _calc_bezier_points(x, y)
        np.testing.assert_array_equal(rx, x)

    def test_cubic_bezier(self):
        x = np.array([0.0, 0.25, 0.75, 1.0])
        y = np.array([0.0, 1.0, 1.0, 0.0])
        rx, ry = _calc_bezier_points(x, y, n=20)
        assert len(rx) == 20
        # Starts at first control point, ends at last
        assert rx[0] == pytest.approx(0.0)
        assert rx[-1] == pytest.approx(1.0)

    def test_linear(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        rx, ry = _calc_bezier_points(x, y, n=10)
        assert len(rx) == 10


# ---------------------------------------------------------------------------
# curve_grob
# ---------------------------------------------------------------------------

class TestCurveGrob:

    def test_basic(self):
        g = curve_grob(0, 0, 1, 1)
        assert isinstance(g, GTree)
        assert g._grid_class == "curve"

    def test_unit_coords(self):
        g = curve_grob(
            x1=Unit(0, "npc"), y1=Unit(0, "npc"),
            x2=Unit(1, "npc"), y2=Unit(1, "npc"),
        )
        assert isinstance(g, GTree)

    def test_custom_params(self):
        g = curve_grob(
            0, 0, 1, 1,
            curvature=0.5, angle=45, ncp=3,
            shape=0.2, squareShape=0.8,
        )
        assert g.curvature == 0.5
        assert g.ncp == 3

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            curve_grob(0, 0, 1, 1, shape=2.0)

    def test_invalid_square_shape_raises(self):
        with pytest.raises(ValueError, match="squareShape"):
            curve_grob(0, 0, 1, 1, squareShape=-2.0)

    def test_name(self):
        g = curve_grob(0, 0, 1, 1, name="mycurve")
        assert g.name == "mycurve"

    def test_inflect(self):
        g = curve_grob(0, 0, 1, 1, inflect=True)
        assert g.inflect is True


# ---------------------------------------------------------------------------
# grid_curve
# ---------------------------------------------------------------------------

class TestGridCurve:

    def test_draw_true(self):
        g = grid_curve(0, 0, 1, 1, draw=True)
        assert isinstance(g, GTree)

    def test_draw_false(self):
        g = grid_curve(0, 0, 1, 1, draw=False)
        assert isinstance(g, GTree)


# ---------------------------------------------------------------------------
# xspline_grob
# ---------------------------------------------------------------------------

class TestXsplineGrob:

    def test_defaults(self):
        g = xspline_grob()
        assert g._grid_class == "xspline"

    def test_with_coords(self):
        g = xspline_grob(x=[0, 0.5, 1], y=[0, 1, 0])
        assert isinstance(g, Grob)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            xspline_grob(shape=2.0)

    def test_shape_sequence(self):
        g = xspline_grob(x=[0, 0.5, 1], y=[0, 1, 0], shape=[0.0, 0.5, -0.5])
        assert isinstance(g, Grob)

    def test_with_unit_coords(self):
        g = xspline_grob(x=Unit([0, 1], "npc"), y=Unit([0, 1], "npc"))
        assert isinstance(g, Grob)

    def test_closed(self):
        g = xspline_grob(x=[0, 1, 0.5], y=[0, 0, 1], open_=False)
        assert g.open_ is False


# ---------------------------------------------------------------------------
# grid_xspline
# ---------------------------------------------------------------------------

class TestGridXspline:

    def test_draw_true(self):
        g = grid_xspline(draw=True)
        assert isinstance(g, Grob)

    def test_draw_false(self):
        g = grid_xspline(draw=False)
        assert isinstance(g, Grob)


# ---------------------------------------------------------------------------
# xspline_points
# ---------------------------------------------------------------------------

class TestXsplinePoints:

    def test_basic(self):
        # Default shape=0 returns the CPs directly (Blanc-Schlick: corners).
        g = xspline_grob(x=[0, 0.5, 1], y=[0, 1, 0])
        pts = xspline_points(g)
        assert "x" in pts
        assert "y" in pts
        # shape=0 ⇒ exactly the 3 control points
        assert len(pts["x"]) == 3

    def test_interpolating_shape(self):
        # shape=0.5 (interpolating) produces more sampled points.
        g = xspline_grob(x=[0, 0.5, 1], y=[0, 1, 0], shape=0.5)
        pts = xspline_points(g)
        assert len(pts["x"]) > 3

    def test_invalid_grob_raises(self):
        g = Grob(name="not_xspline")
        with pytest.raises(TypeError, match="xspline"):
            xspline_points(g)


# ---------------------------------------------------------------------------
# bezier_grob
# ---------------------------------------------------------------------------

class TestBezierGrob:

    def test_basic(self):
        g = bezier_grob(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0])
        assert isinstance(g, GTree)
        assert g._grid_class == "beziergrob"

    def test_with_units(self):
        g = bezier_grob(
            x=Unit([0, 0.25, 0.75, 1], "npc"),
            y=Unit([0, 1, 1, 0], "npc"),
        )
        assert isinstance(g, GTree)


# ---------------------------------------------------------------------------
# grid_bezier
# ---------------------------------------------------------------------------

class TestGridBezier:

    def test_draw_true(self):
        g = grid_bezier(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0], draw=True)
        assert isinstance(g, GTree)

    def test_draw_false(self):
        g = grid_bezier(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0], draw=False)
        assert isinstance(g, GTree)


# ---------------------------------------------------------------------------
# bezier_points
# ---------------------------------------------------------------------------

class TestBezierPoints:

    def test_basic(self):
        g = bezier_grob(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0])
        pts = bezier_points(g, n=20)
        assert "x" in pts
        assert "y" in pts
        assert len(pts["x"]) == 20

    def test_invalid_grob_raises(self):
        g = Grob(name="not_bezier")
        with pytest.raises(TypeError, match="beziergrob"):
            bezier_points(g)
