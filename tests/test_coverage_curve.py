"""Tests for grid_py._curve -- curve, xspline, and bezier grobs."""

from __future__ import annotations

import numpy as np
import pytest

from grid_py._curve import (
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

    def test_single_point_raises(self):
        # R compute_open_spline: "there must be at least two control points"
        x = np.array([1.0])
        y = np.array([2.0])
        with pytest.raises(ValueError, match="two control points"):
            _calc_xspline_points(x, y)

    def test_two_points(self):
        # Raw engine output (GEXspline analogue): the duplicated first
        # control point of the repEnds windows is emitted twice; R trims
        # such runs only at the grid.c layer (see xspline_points).
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y)
        assert rx[0] == pytest.approx(0.0)
        assert rx[-1] == pytest.approx(1.0)
        from grid_py._curve import _trim_identical_ends
        tx, ty = _trim_identical_ends(rx, ry)
        assert len(tx) == 2
        np.testing.assert_allclose(tx, x)
        np.testing.assert_allclose(ty, y)

    def test_two_points_with_shape(self):
        # The engine uses shapes exactly as given (like R's C code): with
        # nonzero shapes on both CPs the degenerate segment is sampled
        # smoothly.  End-shape zeroing is a grob-validation concern
        # (validDetails.xspline) — see TestXsplineGrob.
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y, shape=0.5)
        assert len(rx) > 2

    def test_closed(self):
        # Closed 3-point spline with shape=0: Blanc-Schlick outputs the 3
        # polygon vertices IN ORDER (matches R windows (n-1,0,1,2), (0,1,2,3),
        # ..., i.e. segment k emits control point k at t=0).
        x = np.array([0.0, 1.0, 0.5])
        y = np.array([0.0, 0.0, 1.0])
        rx, ry = _calc_xspline_points(x, y, open_=False)
        np.testing.assert_allclose(rx, x)
        np.testing.assert_allclose(ry, y)

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
# Bezier -> X-spline control point transform (R: Msinv %*% Mb)
# ---------------------------------------------------------------------------

class TestSplineFromBezier:

    def test_transform_matrix(self):
        # R: Ms <- 1/6*rbind(...); Msinv <- solve(Ms); Mb <- rbind(...)
        # For control points (0.5, 1.5, 3.5, 4.5)in the spline control
        # points are (-0.5, -0.5, 5.5, 5.5)in (verified against
        # grid:::splinegrob in R 4.4.3).
        from grid_py._curve import _SPLINE_FROM_BEZIER
        ctrl = np.array([0.5, 1.5, 3.5, 4.5])
        np.testing.assert_allclose(
            _SPLINE_FROM_BEZIER @ ctrl, [-0.5, -0.5, 5.5, 5.5], atol=1e-12)
        ctrl_y = np.array([1.0, 4.0, 4.0, 1.0])
        np.testing.assert_allclose(
            _SPLINE_FROM_BEZIER @ ctrl_y, [-14.0, 4.0, 4.0, -14.0],
            atol=1e-12)

    def test_splinegrob_fields(self):
        # R splinegrob(): shape=1 (zeroed at the ends by validDetails),
        # repEnds=FALSE, same name as the bezier grob.
        from grid_py._curve import _splinegrob
        g = bezier_grob(x=[0.1, 0.3, 0.7, 0.9], y=[0.2, 0.8, 0.8, 0.2])
        sg = _splinegrob(g)
        assert sg._grid_class == "xspline"
        assert sg.repEnds is False
        assert sg.open_ is True
        assert sg.name == g.name
        np.testing.assert_allclose(sg.shape, [0.0, 1.0, 1.0, 0.0])


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
        # R bezierGrob() creates a plain grob (not a gTree); the X-spline
        # content is generated by make_content at draw time.
        g = bezier_grob(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0])
        assert isinstance(g, Grob)
        assert not isinstance(g, GTree)
        assert g._grid_class == "beziergrob"

    def test_with_units(self):
        g = bezier_grob(
            x=Unit([0, 0.25, 0.75, 1], "npc"),
            y=Unit([0, 1, 1, 0], "npc"),
        )
        assert isinstance(g, Grob)

    def test_defaults_match_r(self):
        # R: bezierGrob()$x == c(0, 0.5, 1, 0.5), $y == c(0.5, 1, 0.5, 0)
        g = bezier_grob()
        np.testing.assert_allclose(np.asarray(g.x.values, float),
                                   [0, 0.5, 1, 0.5])
        np.testing.assert_allclose(np.asarray(g.y.values, float),
                                   [0.5, 1, 0.5, 0])

    def test_make_content_is_xspline(self):
        # R makeContent.beziergrob -> splinegrob (xspline shape=1,
        # repEnds=FALSE); this is what used to be missing and made
        # bezier rendering blank.
        g = bezier_grob(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0])
        content = g.make_content()
        assert content._grid_class == "xspline"
        assert content.repEnds is False

    def test_exactly_four_points_required(self):
        with pytest.raises(ValueError, match="exactly 4 control points"):
            bezier_grob(x=[0, 0.2, 0.4, 0.6, 1], y=[0, 1, 0, 1, 0])

    def test_id_lengths_multi_curve(self):
        g = bezier_grob(
            x=[0.1, 0.3, 0.7, 0.9, 0.9, 0.7, 0.3, 0.1],
            y=[0.2, 0.8, 0.8, 0.2, 0.2, 0.4, 0.4, 0.2],
            id_lengths=[4, 4],
        )
        assert g._grid_class == "beziergrob"

    def test_id_lengths_non_multiple_of_four(self):
        with pytest.raises(ValueError,
                           match="4 control points per Bezier curve"):
            bezier_grob(x=list(np.arange(10) / 10.0),
                        y=list(np.arange(10) / 10.0),
                        id_lengths=[5, 5])

    def test_id_and_id_lengths_conflict(self):
        with pytest.raises(ValueError, match="both 'id' and 'id.lengths'"):
            bezier_grob(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0],
                        id=[1, 1, 1, 1], id_lengths=[4])


# ---------------------------------------------------------------------------
# grid_bezier
# ---------------------------------------------------------------------------

class TestGridBezier:

    def test_draw_true(self):
        g = grid_bezier(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0], draw=True)
        assert isinstance(g, Grob)

    def test_draw_false(self):
        g = grid_bezier(x=[0, 0.25, 0.75, 1], y=[0, 1, 1, 0], draw=False)
        assert isinstance(g, Grob)


# ---------------------------------------------------------------------------
# bezier_points
# ---------------------------------------------------------------------------

class TestBezierPoints:

    def test_basic(self):
        # R bezierPoints(): points come from the X-spline approximation,
        # in INCHES, with a device-size-dependent count (no ``n`` arg).
        g = bezier_grob(x=[0.0, 0.25, 0.75, 1.0], y=[0.0, 1.0, 1.0, 0.0])
        pts = bezier_points(g)
        assert "x" in pts
        assert "y" in pts
        assert len(pts["x"]) > 4

    def test_endpoints_interpolated(self):
        # The traced curve starts/ends at the first/last control point
        # (converted to inches in the current context).
        from grid_py._units import convert_x, convert_y
        g = bezier_grob(x=[0.1, 0.3, 0.7, 0.9], y=[0.2, 0.8, 0.8, 0.2])
        pts = bezier_points(g)
        x_in = np.asarray(convert_x(g.x, "inches", valueOnly=True), float)
        y_in = np.asarray(convert_y(g.y, "inches", valueOnly=True), float)
        assert pts["x"][0] == pytest.approx(x_in[0], abs=1e-9)
        assert pts["y"][0] == pytest.approx(y_in[0], abs=1e-9)
        assert pts["x"][-1] == pytest.approx(x_in[-1], abs=1e-9)
        assert pts["y"][-1] == pytest.approx(y_in[-1], abs=1e-9)

    def test_multi_curve_returns_list(self):
        # R returns a list of point sets for id.lengths=c(4, 4)
        g = bezier_grob(
            x=[0.1, 0.3, 0.7, 0.9, 0.9, 0.7, 0.3, 0.1],
            y=[0.2, 0.8, 0.8, 0.2, 0.2, 0.4, 0.4, 0.2],
            id_lengths=[4, 4],
        )
        pts = bezier_points(g)
        assert isinstance(pts, list)
        assert len(pts) == 2
        assert all(len(p["x"]) > 4 for p in pts)

    def test_invalid_grob_raises(self):
        g = Grob(name="not_bezier")
        with pytest.raises(TypeError, match="beziergrob"):
            bezier_points(g)
