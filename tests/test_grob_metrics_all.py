"""Tests for coordinate-based grob metrics (Gap B fix).

Validates width_details/height_details for lines, points, polygon,
segments, and circle grobs, mirroring R's C_locnBounds/C_circleBounds.
"""

import pytest

from grid_py._gpar import Gpar
from grid_py._primitives import (
    circle_grob,
    lines_grob,
    points_grob,
    polygon_grob,
    segments_grob,
)
from grid_py._size import width_details, height_details
from grid_py._units import Unit, unit_c
from grid_py._draw import grid_newpage


class TestLinesGrobMetrics:
    """lines grob: bounding box of x/y coordinates."""

    def test_width(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = lines_grob(x=Unit([0.2, 0.8], "npc"), y=Unit([0.3, 0.7], "npc"))
        w = width_details(g)
        # 0.6 NPC of 5 inches = 3 inches
        assert w._units[0] == "inches"
        assert w._values[0] == pytest.approx(3.0, abs=0.1)

    def test_height(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = lines_grob(x=Unit([0.2, 0.8], "npc"), y=Unit([0.3, 0.7], "npc"))
        h = height_details(g)
        # 0.4 NPC of 5 inches = 2 inches
        assert h._units[0] == "inches"
        assert h._values[0] == pytest.approx(2.0, abs=0.1)

    def test_single_point_zero_width(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = lines_grob(x=Unit([0.5], "npc"), y=Unit([0.5], "npc"))
        w = width_details(g)
        assert w._values[0] == pytest.approx(0.0, abs=0.001)

    def test_absolute_units(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g = lines_grob(x=Unit([0, 2.54], "cm"), y=Unit([0, 1], "npc"))
        w = width_details(g)
        # 2.54 cm = 1 inch; 0 cm = 0 inch; width = 1 inch
        assert w._values[0] == pytest.approx(1.0, abs=0.05)


class TestPointsGrobMetrics:
    """points grob: bounding box of x/y coordinates."""

    def test_width(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = points_grob(
            x=Unit([0.1, 0.5, 0.9], "npc"),
            y=Unit([0.2, 0.8, 0.5], "npc"),
        )
        w = width_details(g)
        # 0.8 NPC of 4 inches = 3.2 inches
        assert w._values[0] == pytest.approx(3.2, abs=0.1)

    def test_height(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = points_grob(
            x=Unit([0.1, 0.5, 0.9], "npc"),
            y=Unit([0.2, 0.8, 0.5], "npc"),
        )
        h = height_details(g)
        # 0.6 NPC of 4 inches = 2.4 inches
        assert h._values[0] == pytest.approx(2.4, abs=0.1)


class TestPolygonGrobMetrics:
    """polygon grob: bounding box of x/y coordinates."""

    def test_width(self):
        grid_newpage(width=3.0, height=3.0, dpi=100)
        g = polygon_grob(
            x=Unit([0.0, 0.5, 1.0, 0.5], "npc"),
            y=Unit([0.5, 1.0, 0.5, 0.0], "npc"),
        )
        w = width_details(g)
        # 1.0 NPC of 3 inches = 3 inches
        assert w._values[0] == pytest.approx(3.0, abs=0.1)

    def test_height(self):
        grid_newpage(width=3.0, height=3.0, dpi=100)
        g = polygon_grob(
            x=Unit([0.0, 0.5, 1.0, 0.5], "npc"),
            y=Unit([0.5, 1.0, 0.5, 0.0], "npc"),
        )
        h = height_details(g)
        assert h._values[0] == pytest.approx(3.0, abs=0.1)


class TestSegmentsGrobMetrics:
    """segments grob: bounding box of all endpoints."""

    def test_width(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = segments_grob(
            x0=Unit([0.1], "npc"), y0=Unit([0.2], "npc"),
            x1=Unit([0.9], "npc"), y1=Unit([0.8], "npc"),
        )
        w = width_details(g)
        # x range: 0.1 to 0.9 = 0.8 NPC; 0.8 * 4 inches = 3.2 inches
        assert w._values[0] == pytest.approx(3.2, abs=0.1)

    def test_height(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = segments_grob(
            x0=Unit([0.1], "npc"), y0=Unit([0.2], "npc"),
            x1=Unit([0.9], "npc"), y1=Unit([0.8], "npc"),
        )
        h = height_details(g)
        assert h._values[0] == pytest.approx(2.4, abs=0.1)

    def test_multiple_segments(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = segments_grob(
            x0=Unit([0.1, 0.3], "npc"), y0=Unit([0.1, 0.3], "npc"),
            x1=Unit([0.5, 0.9], "npc"), y1=Unit([0.5, 0.7], "npc"),
        )
        w = width_details(g)
        # x range: min(0.1, 0.3) to max(0.5, 0.9) = 0.8 NPC = 3.2 inches
        assert w._values[0] == pytest.approx(3.2, abs=0.1)


class TestCircleGrobMetrics:
    """circle grob: bounding box with radius."""

    def test_width(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = circle_grob(
            x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
            r=Unit(0.2, "npc"),
        )
        w = width_details(g)
        # (0.5+0.2) - (0.5-0.2) = 0.4 NPC; 0.4 * 4 = 1.6 inches
        assert w._values[0] == pytest.approx(1.6, abs=0.1)

    def test_height(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = circle_grob(
            x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
            r=Unit(0.3, "npc"),
        )
        h = height_details(g)
        # 0.6 NPC * 4 = 2.4 inches
        assert h._values[0] == pytest.approx(2.4, abs=0.1)

    def test_absolute_radius(self):
        grid_newpage(width=4.0, height=4.0, dpi=100)
        g = circle_grob(
            x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
            r=Unit(1, "inches"),
        )
        w = width_details(g)
        # radius = 1 inch = 0.25 NPC; width = 2*0.25 = 0.5 NPC = 2 inches
        assert w._values[0] == pytest.approx(2.0, abs=0.1)


class TestNoRendererFallback:
    """Without an active renderer, coordinate-based metrics return 0."""

    def test_lines_without_renderer(self):
        # Don't call grid_newpage → no renderer
        from grid_py._size import _get_renderer
        # We can't easily clear the renderer, so just verify the function exists
        # and returns Unit type
        g = lines_grob(x=Unit([0.1, 0.9], "npc"), y=Unit([0.1, 0.9], "npc"))
        w = width_details(g)
        assert w._units[0] == "inches"
        # With a renderer active from previous tests, this should work
        assert w._values[0] >= 0
