"""Tests for curve/xspline/bezier grob metrics.

Validates width_details/height_details for xspline, bezier, and curve grobs,
mirroring R's C_xsplineBounds, splinegrob(), and calcCurveGrob() semantics.
"""

import pytest

from grid_py._size import width_details, height_details
from grid_py._units import Unit
from grid_py._draw import grid_newpage


class TestXsplineMetrics:
    """xspline grob: evaluated spline point bounding box."""

    def test_width_positive(self):
        from grid_py._curve import xspline_grob
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = xspline_grob(
            x=Unit([0.2, 0.5, 0.8], "npc"),
            y=Unit([0.3, 0.7, 0.3], "npc"),
            shape=0.5,
        )
        w = width_details(g)
        assert w._units[0] == "inches"
        assert w._values[0] > 0

    def test_height_positive(self):
        from grid_py._curve import xspline_grob
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = xspline_grob(
            x=Unit([0.2, 0.5, 0.8], "npc"),
            y=Unit([0.3, 0.7, 0.3], "npc"),
            shape=0.5,
        )
        h = height_details(g)
        assert h._units[0] == "inches"
        assert h._values[0] > 0

    def test_straight_line_shape_zero(self):
        from grid_py._curve import xspline_grob
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = xspline_grob(
            x=Unit([0.0, 1.0], "npc"),
            y=Unit([0.0, 1.0], "npc"),
            shape=0,
        )
        w = width_details(g)
        assert w._values[0] > 0


class TestBezierMetrics:
    """bezier grob: evaluated Bezier point bounding box."""

    def test_width_positive(self):
        from grid_py._curve import bezier_grob
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = bezier_grob(
            x=Unit([0.1, 0.3, 0.7, 0.9], "npc"),
            y=Unit([0.2, 0.8, 0.8, 0.2], "npc"),
        )
        w = width_details(g)
        assert w._units[0] == "inches"
        assert w._values[0] > 0

    def test_height_positive(self):
        from grid_py._curve import bezier_grob
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = bezier_grob(
            x=Unit([0.1, 0.3, 0.7, 0.9], "npc"),
            y=Unit([0.2, 0.8, 0.8, 0.2], "npc"),
        )
        h = height_details(g)
        assert h._units[0] == "inches"
        assert h._values[0] > 0


class TestCurveMetrics:
    """curve grob: expanded control point bounding box."""

    def test_width_positive(self):
        from grid_py._curve import curve_grob
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = curve_grob(
            x1=Unit(0.2, "npc"), y1=Unit(0.2, "npc"),
            x2=Unit(0.8, "npc"), y2=Unit(0.8, "npc"),
            curvature=1.0,
        )
        w = width_details(g)
        assert w._units[0] == "inches"
        assert w._values[0] > 0

    def test_height_positive(self):
        from grid_py._curve import curve_grob
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = curve_grob(
            x1=Unit(0.2, "npc"), y1=Unit(0.2, "npc"),
            x2=Unit(0.8, "npc"), y2=Unit(0.8, "npc"),
        )
        h = height_details(g)
        assert h._units[0] == "inches"
        assert h._values[0] > 0

    def test_straight_line_zero_curvature(self):
        from grid_py._curve import curve_grob
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = curve_grob(
            x1=Unit(0.0, "npc"), y1=Unit(0.0, "npc"),
            x2=Unit(1.0, "npc"), y2=Unit(0.0, "npc"),
            curvature=0,
        )
        w = width_details(g)
        # Straight line from 0 to 1 NPC = 4 inches
        assert w._values[0] == pytest.approx(4.0, abs=0.5)
        h = height_details(g)
        # No vertical extent
        assert h._values[0] == pytest.approx(0.0, abs=0.1)
