"""Tests for unit resolution in the rendering pipeline.

Validates that CairoRenderer._resolve_to_npc() correctly converts all
unit types to NPC values, mirroring R's unit.c:transformLocation/transform.
"""

import numpy as np
import pytest

from grid_py._gpar import Gpar
from grid_py._units import Unit
from grid_py._viewport import Viewport
from grid_py.renderer import CairoRenderer


# ===================================================================== #
#  Fixtures                                                              #
# ===================================================================== #


@pytest.fixture
def renderer_3x2():
    """A 300x200 px renderer at 100 DPI → 3 in x 2 in viewport."""
    return CairoRenderer(width=3.0, height=2.0, dpi=100)


# ===================================================================== #
#  Absolute unit resolution                                              #
# ===================================================================== #


class TestAbsoluteUnits:
    """Absolute units (cm, inches, mm, points) → NPC."""

    def test_npc_passthrough(self, renderer_3x2):
        assert renderer_3x2.resolve_x(Unit(0.5, "npc")) == pytest.approx(0.5)

    def test_npc_dimension(self, renderer_3x2):
        assert renderer_3x2.resolve_w(Unit(0.25, "npc")) == pytest.approx(0.25)

    def test_inches_x_dim(self, renderer_3x2):
        # 1 inch / 3 inch viewport = 1/3
        assert renderer_3x2.resolve_w(Unit(1.0, "inches")) == pytest.approx(1.0 / 3.0)

    def test_inches_y_dim(self, renderer_3x2):
        # 1 inch / 2 inch viewport = 1/2
        assert renderer_3x2.resolve_h(Unit(1.0, "inches")) == pytest.approx(1.0 / 2.0)

    def test_cm_dim(self, renderer_3x2):
        # 2.54 cm = 1 inch; 1 inch / 3 inch viewport = 1/3
        assert renderer_3x2.resolve_w(Unit(2.54, "cm")) == pytest.approx(1.0 / 3.0)

    def test_mm_dim(self, renderer_3x2):
        # 25.4 mm = 1 inch; 1 inch / 3 inch viewport = 1/3
        assert renderer_3x2.resolve_w(Unit(25.4, "mm")) == pytest.approx(1.0 / 3.0)

    def test_points_dim(self, renderer_3x2):
        # 72.27 points = 1 inch (TeX); 1 inch / 3 inch = 1/3
        assert renderer_3x2.resolve_w(
            Unit(72.27, "points")
        ) == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_bigpts_dim(self, renderer_3x2):
        # 72 bigpts = 1 inch (PS); 1 inch / 3 inch = 1/3
        assert renderer_3x2.resolve_w(
            Unit(72.0, "bigpts")
        ) == pytest.approx(1.0 / 3.0)

    def test_bare_float_passthrough(self, renderer_3x2):
        # Plain floats are treated as NPC
        assert renderer_3x2.resolve_x(0.75) == pytest.approx(0.75)

    def test_bare_int_passthrough(self, renderer_3x2):
        assert renderer_3x2.resolve_x(0) == pytest.approx(0.0)


# ===================================================================== #
#  Native unit resolution                                                #
# ===================================================================== #


class TestNativeUnits:
    """Native (data coordinate) units → NPC."""

    def test_native_location(self, renderer_3x2):
        vp = Viewport(xscale=[0, 10], yscale=[0, 100])
        renderer_3x2.push_viewport(vp)
        # x=5 in [0,10] → NPC 0.5
        assert renderer_3x2.resolve_x(Unit(5.0, "native")) == pytest.approx(0.5)

    def test_native_dimension(self, renderer_3x2):
        vp = Viewport(xscale=[0, 10], yscale=[0, 100])
        renderer_3x2.push_viewport(vp)
        # width=3 in range 10 → NPC 0.3
        assert renderer_3x2.resolve_w(Unit(3.0, "native")) == pytest.approx(0.3)

    def test_native_y(self, renderer_3x2):
        vp = Viewport(xscale=[0, 10], yscale=[0, 100])
        renderer_3x2.push_viewport(vp)
        # y=25 in [0,100] → NPC 0.25
        assert renderer_3x2.resolve_y(Unit(25.0, "native")) == pytest.approx(0.25)

    def test_native_offset_scale(self, renderer_3x2):
        vp = Viewport(xscale=[100, 200])
        renderer_3x2.push_viewport(vp)
        # x=150 in [100,200] → NPC 0.5
        assert renderer_3x2.resolve_x(Unit(150.0, "native")) == pytest.approx(0.5)


# ===================================================================== #
#  Font-relative units                                                   #
# ===================================================================== #


class TestFontUnits:
    """Char and lines units → NPC."""

    def test_char_unit(self, renderer_3x2):
        gp = Gpar(fontsize=12, cex=1)
        # 1 char = 12pt/72 inches = 1/6 inch; viewport = 3 inches
        result = renderer_3x2.resolve_w(Unit(1, "char"), gp=gp)
        assert result == pytest.approx((12.0 / 72.0) / 3.0)

    def test_lines_unit(self, renderer_3x2):
        gp = Gpar(fontsize=12, cex=1, lineheight=1.2)
        # 1 line = 12*1.2pt/72 inches = 0.2 inch; viewport = 3 inches
        result = renderer_3x2.resolve_w(Unit(1, "lines"), gp=gp)
        assert result == pytest.approx((12.0 * 1.2 / 72.0) / 3.0)

    def test_lines_larger_font(self, renderer_3x2):
        gp = Gpar(fontsize=24, cex=2)
        # 1 line = 24*2*1.2/72 inches; viewport = 3 inches
        result = renderer_3x2.resolve_w(Unit(1, "lines"), gp=gp)
        assert result == pytest.approx((24.0 * 2.0 * 1.2 / 72.0) / 3.0)


# ===================================================================== #
#  Special units                                                         #
# ===================================================================== #


class TestSpecialUnits:
    """Null, snpc units."""

    def test_null_is_zero(self, renderer_3x2):
        assert renderer_3x2.resolve_w(Unit(1, "null")) == pytest.approx(0.0)

    def test_snpc_x_axis(self, renderer_3x2):
        # 3x2 viewport: min dim = 2 inches (height)
        # snpc on x: value * min_dim_inches / vp_x_inches = 0.5 * 2/3
        result = renderer_3x2.resolve_w(Unit(0.5, "snpc"))
        assert result == pytest.approx(0.5 * 2.0 / 3.0)

    def test_snpc_y_axis(self, renderer_3x2):
        # snpc on y: value * min_dim_inches / vp_y_inches = 0.5 * 2/2 = 0.5
        result = renderer_3x2.resolve_h(Unit(0.5, "snpc"))
        assert result == pytest.approx(0.5 * 2.0 / 2.0)


# ===================================================================== #
#  String metric units                                                   #
# ===================================================================== #


class TestStringMetricUnits:
    """strwidth/strheight resolution."""

    def test_strwidth(self, renderer_3x2):
        u = Unit(1, "strwidth", data="Hello")
        result = renderer_3x2.resolve_w(u)
        assert result > 0

    def test_strheight(self, renderer_3x2):
        u = Unit(1, "strheight", data="X")
        result = renderer_3x2.resolve_h(u)
        assert result > 0


# ===================================================================== #
#  Grob metric units                                                     #
# ===================================================================== #


class TestGrobMetricUnits:
    """grobwidth/grobheight resolution."""

    def test_grobwidth_text(self, renderer_3x2):
        from grid_py._primitives import text_grob
        tg = text_grob("Hello")
        u = Unit(1, "grobwidth", data=tg)
        result = renderer_3x2.resolve_w(u)
        assert result > 0

    def test_grobheight_null(self, renderer_3x2):
        from grid_py._primitives import null_grob
        ng = null_grob()
        u = Unit(1, "grobheight", data=ng)
        result = renderer_3x2.resolve_h(u)
        assert result == pytest.approx(0.0)


# ===================================================================== #
#  Array resolution                                                      #
# ===================================================================== #


class TestArrayResolution:
    """Vectorized resolve_x_array / resolve_y_array."""

    def test_unit_array(self, renderer_3x2):
        from grid_py._units import unit_c
        u = unit_c(Unit(0.25, "npc"), Unit(0.5, "npc"), Unit(0.75, "npc"))
        arr = renderer_3x2.resolve_x_array(u)
        np.testing.assert_array_almost_equal(arr, [0.25, 0.5, 0.75])

    def test_mixed_unit_array(self, renderer_3x2):
        from grid_py._units import unit_c
        # 1 inch = 1/3 NPC on x-axis (3in viewport)
        u = unit_c(Unit(0.5, "npc"), Unit(1.0, "inches"))
        arr = renderer_3x2.resolve_x_array(u)
        assert arr[0] == pytest.approx(0.5)
        assert arr[1] == pytest.approx(1.0 / 3.0)

    def test_float_list(self, renderer_3x2):
        arr = renderer_3x2.resolve_x_array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(arr, [0.1, 0.2, 0.3])


# ===================================================================== #
#  push_viewport with absolute units                                     #
# ===================================================================== #


class TestViewportResolution:
    """push_viewport resolves absolute units correctly."""

    def test_viewport_absolute_width(self, renderer_3x2):
        vp = Viewport(width=Unit(1.0, "inches"), height=Unit(1.0, "inches"))
        renderer_3x2.push_viewport(vp)
        # Child viewport should be 100px wide (1 inch at 100 DPI)
        x0, y0, pw, ph, vp_obj = renderer_3x2._vp_stack[-1]
        assert pw == pytest.approx(100.0, abs=1.0)

    def test_viewport_cm_width(self, renderer_3x2):
        vp = Viewport(width=Unit(2.54, "cm"), height=Unit(2.54, "cm"))
        renderer_3x2.push_viewport(vp)
        x0, y0, pw, ph, vp_obj = renderer_3x2._vp_stack[-1]
        # 2.54 cm = 1 inch = 100 px at 100 DPI
        assert pw == pytest.approx(100.0, abs=1.0)

    def test_viewport_npc_unchanged(self, renderer_3x2):
        vp = Viewport(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        renderer_3x2.push_viewport(vp)
        x0, y0, pw, ph, vp_obj = renderer_3x2._vp_stack[-1]
        # 0.5 * 300px = 150px
        assert pw == pytest.approx(150.0, abs=1.0)
        # 0.5 * 200px = 100px
        assert ph == pytest.approx(100.0, abs=1.0)


# ===================================================================== #
#  Rendering integration                                                 #
# ===================================================================== #


class TestRenderingIntegration:
    """End-to-end: draw grobs with absolute units."""

    def test_draw_rect_cm(self):
        """A 2.54cm-wide rect on a 3in-wide canvas produces correct output."""
        from grid_py._draw import grid_draw, grid_newpage
        from grid_py._primitives import rect_grob

        grid_newpage(width=3.0, height=2.0, dpi=100)
        g = rect_grob(
            x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
            width=Unit(2.54, "cm"), height=Unit(1.0, "inches"),
        )
        # Should not raise
        grid_draw(g)

    def test_draw_text_cm_position(self):
        """Text positioned with cm units."""
        from grid_py._draw import grid_draw, grid_newpage
        from grid_py._primitives import text_grob

        grid_newpage(width=5.0, height=5.0, dpi=72)
        g = text_grob("Test", x=Unit(2.54, "cm"), y=Unit(2.54, "cm"))
        grid_draw(g)
