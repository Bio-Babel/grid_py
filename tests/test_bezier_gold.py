"""R-gold regression tests for bezier / xspline curve evaluation.

Every expected number in this file was produced by R 4.4.3 ``grid`` on a
5x5-inch (or where noted 7x7-inch) pdf device and verified to match the
Python implementation to < 1e-11 inches:

    bg  <- bezierGrob(x=c(0.1, 0.3, 0.7, 0.9), y=c(0.2, 0.8, 0.8, 0.2))
    sg  <- grid:::splinegrob(bg)         # control-point transform
    bezierPoints(bg)                     # traced curve (inches)
    xsplinePoints(xsplineGrob(..., shape=1))
    convertWidth(grobWidth(bg), "inches", valueOnly=TRUE)   # etc.

The traced-point COUNT is part of the contract: R's engine samples the
curve adaptively from the PHYSICAL segment length (xfig 1200ppi space,
xspline.c ``step_computing``), so the same geometry on the same device
size must yield the same number of points regardless of dpi.
"""

from __future__ import annotations

import numpy as np
import pytest

from grid_py import (
    Gpar,
    Viewport,
    bezier_grob,
    bezier_points,
    get_state,
    grob_coords,
    push_viewport,
    xspline_grob,
    xspline_points,
)
from grid_py._draw import grid_newpage
from grid_py._size import height_details, width_details

BEZ_X = [0.1, 0.3, 0.7, 0.9]
BEZ_Y = [0.2, 0.8, 0.8, 0.2]


@pytest.fixture
def page5():
    grid_newpage(width=5.0, height=5.0, dpi=100)
    yield
    get_state().reset()


class TestSplinegrobTransform:
    """grid:::splinegrob — Bezier -> X-spline control points."""

    def test_control_points_5x5(self, page5):
        from grid_py._curve import _splinegrob
        sg = _splinegrob(bezier_grob(x=BEZ_X, y=BEZ_Y))
        # R: as.numeric(sg$x) / as.numeric(sg$y) on a 5x5in device
        np.testing.assert_allclose(
            np.asarray(sg.x.values, float), [-0.5, -0.5, 5.5, 5.5],
            atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(sg.y.values, float), [-14.0, 4.0, 4.0, -14.0],
            atol=1e-12)
        np.testing.assert_allclose(sg.shape, [0.0, 1.0, 1.0, 0.0])
        assert sg.repEnds is False

    def test_control_points_nested_viewport(self, page5):
        from grid_py._curve import _splinegrob
        push_viewport(Viewport(x=0.25, y=0.25, width=0.5, height=0.4,
                               just=("left", "bottom")))
        sg = _splinegrob(bezier_grob(x=BEZ_X, y=BEZ_Y))
        np.testing.assert_allclose(
            np.asarray(sg.x.values, float), [-0.25, -0.25, 2.75, 2.75],
            atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(sg.y.values, float), [-5.6, 1.6, 1.6, -5.6],
            atol=1e-12)


class TestBezierPointsGold:
    """bezierPoints() traced curve, R 4.4.3 gold values (inches)."""

    def test_single_curve_5x5(self, page5):
        pts = bezier_points(bezier_grob(x=BEZ_X, y=BEZ_Y))
        assert len(pts["x"]) == 46
        gold = {
            0: (0.5, 1.0),
            1: (0.562951412729569, 1.18084547148852),
            23: (2.57294855953263, 3.16627904552386),
            44: (4.45995905789827, 1.11682927346178),
            45: (4.5, 1.0),
        }
        for i, (gx, gy) in gold.items():
            assert pts["x"][i] == pytest.approx(gx, abs=1e-11)
            assert pts["y"][i] == pytest.approx(gy, abs=1e-11)

    def test_count_scales_with_device_size(self):
        # R: 46 points on 5x5in, 52 on 7x7in (physical-size dependence)
        grid_newpage(width=7.0, height=7.0, dpi=100)
        pts = bezier_points(bezier_grob(x=BEZ_X, y=BEZ_Y))
        assert len(pts["x"]) == 52

    def test_count_independent_of_dpi(self):
        # R: 46 points at 5x5in whether the raster is 100 or 300 dpi
        grid_newpage(width=5.0, height=5.0, dpi=300)
        pts = bezier_points(bezier_grob(x=BEZ_X, y=BEZ_Y))
        assert len(pts["x"]) == 46

    def test_multi_curve_counts(self, page5):
        g = bezier_grob(
            x=[0.1, 0.3, 0.7, 0.9, 0.9, 0.7, 0.3, 0.1],
            y=[0.2, 0.8, 0.8, 0.2, 0.2, 0.4, 0.4, 0.2],
            id_lengths=[4, 4],
        )
        pts = bezier_points(g)
        assert isinstance(pts, list)
        assert [len(p["x"]) for p in pts] == [46, 38]
        # second curve, gold samples
        assert pts[1]["x"][0] == pytest.approx(4.5, abs=1e-11)
        assert pts[1]["y"][1] == pytest.approx(1.07309725593663, abs=1e-11)
        assert pts[1]["x"][19] == pytest.approx(2.4111325994254, abs=1e-11)
        assert pts[1]["x"][-1] == pytest.approx(0.5, abs=1e-11)


class TestXsplinePointsGold:
    """xsplinePoints() gold values (inches, 5x5in device)."""

    def test_shape1_repends_true(self, page5):
        g = xspline_grob(x=BEZ_X, y=BEZ_Y, shape=1)
        pts = xspline_points(g)
        assert len(pts["x"]) == 85
        gold = {
            0: (0.5, 1.0),
            1: (0.500352604044259, 1.00099415807303),
            42: (2.52363127951005, 3.86116168817659),
            83: (4.49983823812532, 1.00045643073908),
            84: (4.5, 1.0),
        }
        for i, (gx, gy) in gold.items():
            assert pts["x"][i] == pytest.approx(gx, abs=1e-11)
            assert pts["y"][i] == pytest.approx(gy, abs=1e-11)

    def test_closed_positive_shape(self, page5):
        # Exercises the closed-spline control-point windows
        # ((n-1,0,1,2), (0,1,2,3), ... — R NEXT_CONTROL_POINTS)
        g = xspline_grob(x=[0.2, 0.5, 0.8, 0.7, 0.3],
                         y=[0.3, 0.8, 0.55, 0.2, 0.15],
                         shape=0.7, open_=False)
        pts = xspline_points(g)
        assert len(pts["x"]) == 117
        gold = {
            0: (1.28400812278783, 1.74850710743935),
            58: (3.77598088009718, 2.10560033817345),
            116: (1.26374539734991, 1.67793269134963),
        }
        for i, (gx, gy) in gold.items():
            assert pts["x"][i] == pytest.approx(gx, abs=1e-11)
            assert pts["y"][i] == pytest.approx(gy, abs=1e-11)


class TestCurveMetricsGold:
    """grobWidth / grobHeight, R gold values (inches, 5x5in device)."""

    def test_bezier_width_height(self, page5):
        g = bezier_grob(x=BEZ_X, y=BEZ_Y)
        w = width_details(g)
        h = height_details(g)
        assert w._units[0] == "inches"
        # R: convertWidth/convertHeight(grobWidth/grobHeight(bg), "inches")
        assert w._values[0] == pytest.approx(4.0, abs=1e-9)
        assert h._values[0] == pytest.approx(2.1679889116637279, abs=1e-9)

    def test_xspline_width_height(self, page5):
        g = xspline_grob(x=BEZ_X, y=BEZ_Y, shape=1)
        w = width_details(g)
        h = height_details(g)
        assert w._values[0] == pytest.approx(4.0, abs=1e-9)
        assert h._values[0] == pytest.approx(2.8611616881765873, abs=1e-9)


class TestBezierCoordsGold:
    """grobCoords(beziergrob) goes through make_content -> xspline trace."""

    def test_coords_match_trace(self, page5):
        g = bezier_grob(x=BEZ_X, y=BEZ_Y)
        coords = grob_coords(g, closed=False)
        shape0 = coords[0]
        cx = np.asarray(shape0.x, float)
        cy = np.asarray(shape0.y, float)
        # R: 46 trace points; first at (0.5, 1)in
        assert len(cx) == 46
        assert cx[0] == pytest.approx(0.5, abs=1e-11)
        assert cy[0] == pytest.approx(1.0, abs=1e-11)


class TestBezierRenders:
    """The original bug: grid_bezier produced a blank page."""

    def test_bezier_produces_ink(self, page5):
        import grid_py as g

        g.grid_bezier(x=BEZ_X, y=BEZ_Y, gp=Gpar(col="black", lwd=2))
        renderer = get_state().get_renderer()
        buf = renderer.to_png_bytes()
        import cairo
        import io
        surf = cairo.ImageSurface.create_from_png(io.BytesIO(buf))
        arr = np.ndarray(
            shape=(surf.get_height(), surf.get_width(), 4),
            dtype=np.uint8, buffer=surf.get_data())
        nonwhite = int(np.sum(~np.all(arr[:, :, :3] == 255, axis=2)))
        assert nonwhite > 500
