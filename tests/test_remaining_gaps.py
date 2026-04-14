"""Tests for the remaining gap fixes: A (convert_unit context), C (line path
collect), D (viewport rotation), E (up_viewport return), F (exports)."""

import math

import numpy as np
import pytest

from grid_py._gpar import Gpar
from grid_py._primitives import (
    circle_grob,
    lines_grob,
    segments_grob,
    stroke_grob,
)
from grid_py._units import Unit, convert_unit, unit_c
from grid_py._viewport import Viewport, push_viewport, pop_viewport, up_viewport
from grid_py._draw import grid_draw, grid_newpage
from grid_py.renderer import CairoRenderer


# ===================================================================== #
#  Gap A: convert_unit with renderer context                             #
# ===================================================================== #


class TestConvertUnitContext:
    """convert_unit() uses active renderer for context-dependent units."""

    def test_npc_to_inches_with_renderer(self):
        grid_newpage(width=3.0, height=2.0, dpi=100)
        result = convert_unit(Unit(0.5, "npc"), "inches", axisFrom="x",
                              typeFrom="dimension")
        # 0.5 NPC of 3 inches = 1.5 inches
        assert result._values[0] == pytest.approx(1.5, abs=0.01)
        assert result._units[0] == "inches"

    def test_npc_to_cm_with_renderer(self):
        grid_newpage(width=3.0, height=2.0, dpi=100)
        result = convert_unit(Unit(1.0, "npc"), "cm", axisFrom="y",
                              typeFrom="dimension")
        # 1.0 NPC of 2 inches = 2 inches = 5.08 cm
        assert result._values[0] == pytest.approx(5.08, abs=0.05)

    def test_lines_to_inches_with_renderer(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        result = convert_unit(Unit(1, "lines"), "inches", axisFrom="x",
                              typeFrom="dimension")
        # 1 line = 12 * 1 * 1.2 / 72 = 0.2 inches
        assert result._values[0] == pytest.approx(0.2, abs=0.01)

    def test_char_to_cm_with_renderer(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        result = convert_unit(Unit(1, "char"), "cm", axisFrom="x",
                              typeFrom="dimension")
        # 1 char = 12/72 inches = 1/6 inch = 0.4233 cm
        assert result._values[0] == pytest.approx(12.0 / 72.0 * 2.54, abs=0.02)

    def test_absolute_works_without_renderer(self):
        # Absolute-to-absolute doesn't need renderer
        result = convert_unit(Unit(2.54, "cm"), "inches")
        assert result._values[0] == pytest.approx(1.0, abs=0.001)


# ===================================================================== #
#  Gap C: path collection mode for lines/polyline/segments               #
# ===================================================================== #


class TestPathCollectionLines:
    """stroke_grob with line-based grobs uses path collection correctly."""

    def test_stroke_lines_grob(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        lg = lines_grob(
            x=Unit([0.1, 0.9], "npc"),
            y=Unit([0.1, 0.9], "npc"),
        )
        sg = stroke_grob(lg, gp=Gpar(col="red", lwd=2))
        grid_draw(sg)  # should not raise

    def test_stroke_segments_grob(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        seg = segments_grob(
            x0=Unit([0.1], "npc"), y0=Unit([0.1], "npc"),
            x1=Unit([0.9], "npc"), y1=Unit([0.9], "npc"),
        )
        sg = stroke_grob(seg, gp=Gpar(col="blue", lwd=3))
        grid_draw(sg)


# ===================================================================== #
#  Gap D: viewport rotation in unit resolution                           #
# ===================================================================== #


class TestViewportRotation:
    """Unit resolution accounts for viewport rotation."""

    def test_no_rotation(self):
        r = CairoRenderer(width=3.0, height=2.0, dpi=100)
        # 1 inch at 100 dpi = 100 device units
        assert r.resolve_w(Unit(1, "inches")) == pytest.approx(100.0)

    def test_90_degree_rotation(self):
        r = CairoRenderer(width=3.0, height=2.0, dpi=100)
        vp = Viewport(angle=90)
        r.push_viewport(vp)
        # 1 inch = 100 device units (inches are absolute, not viewport-relative)
        assert r.resolve_w(Unit(1, "inches")) == pytest.approx(100.0)

    def test_0_degree_unchanged(self):
        r = CairoRenderer(width=3.0, height=2.0, dpi=100)
        vp = Viewport(angle=0)
        r.push_viewport(vp)
        assert r.resolve_w(Unit(1, "inches")) == pytest.approx(100.0)


# ===================================================================== #
#  Gap E: up_viewport returns VpPath                                     #
# ===================================================================== #


class TestUpViewportReturn:
    """up_viewport() returns the navigated path segment."""

    def test_returns_none_at_root(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        result = up_viewport(0)
        # At root, no VP parts beyond ROOT → return None
        assert result is None

    def test_returns_path_after_push(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        push_viewport(Viewport(name="A"))
        push_viewport(Viewport(name="B"))
        result = up_viewport(1)
        # Should return path segment for "B"
        assert result is not None
        assert "B" in str(result)

    def test_returns_full_path_up_zero(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        push_viewport(Viewport(name="X"))
        push_viewport(Viewport(name="Y"))
        result = up_viewport(0)
        # Navigate to root → returns entire path "X/Y"
        assert result is not None
        assert "X" in str(result)
        assert "Y" in str(result)


# ===================================================================== #
#  Gap F: exports                                                        #
# ===================================================================== #


class TestExports:
    """Verify key functions are properly exported."""

    def test_calc_string_metric_exported(self):
        from grid_py import calc_string_metric
        m = calc_string_metric("Hello")
        assert "width" in m

    def test_width_details_exported(self):
        from grid_py import width_details, text_grob
        w = width_details(text_grob("X"))
        assert w._values[0] > 0

    def test_fill_stroke_exported(self):
        from grid_py import (
            as_path, stroke_grob, grid_stroke,
            fill_grob, grid_fill,
            fill_stroke_grob, grid_fill_stroke,
        )
        assert callable(as_path)
        assert callable(stroke_grob)
        assert callable(fill_grob)
        assert callable(fill_stroke_grob)
