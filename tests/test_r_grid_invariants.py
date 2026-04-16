"""Tests derived from R's grid package test suite.

R source: R/src/library/grid/tests/units.R, bugs.R, coords.R
Plus grid_py-specific backend tests (Cairo renderer).
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import grid_py
from grid_py._primitives import text_grob, rect_grob, circle_grob
from grid_py._size import width_details, height_details, descent_details


# ---------------------------------------------------------------------------
# R units.R: Unit arithmetic invariants
# ---------------------------------------------------------------------------

class TestUnitArithmetic:
    """R grid/tests/units.R — unit operations must be consistent."""

    def test_unit_addition_same_type(self):
        """R: unit(1,'cm') + unit(2,'cm') should equal 3cm."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        a = grid_py.Unit(1, "cm")
        b = grid_py.Unit(2, "cm")
        c = a + b
        mm = float(np.squeeze(grid_py.convert_width(c, "mm", valueOnly=True)))
        assert abs(mm - 30.0) < 0.1

    def test_unit_addition_mixed_types(self):
        """R: unit(5,'mm') + unit(1,'cm') = 15mm."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        c = grid_py.Unit(5, "mm") + grid_py.Unit(1, "cm")
        mm = float(np.squeeze(grid_py.convert_width(c, "mm", valueOnly=True)))
        assert abs(mm - 15.0) < 0.1

    def test_unit_multiplication(self):
        """R: unit(5,'mm') * 3 = 15mm."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        u = grid_py.Unit(5, "mm") * 3
        mm = float(np.squeeze(grid_py.convert_width(u, "mm", valueOnly=True)))
        assert abs(mm - 15.0) < 0.1

    def test_unit_subtraction(self):
        """R: unit(1,'npc') - unit(10,'mm') in mm."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        u = grid_py.Unit(1, "npc") - grid_py.Unit(10, "mm")
        mm = float(np.squeeze(grid_py.convert_width(u, "mm", valueOnly=True)))
        # 1 npc = full width = 4 inches = 101.6mm; 101.6 - 10 = 91.6
        assert mm > 80

    def test_unit_c(self):
        """R: unit.c(unit(5,'mm'), unit(1,'cm'))."""
        uc = grid_py.unit_c(grid_py.Unit(5, "mm"), grid_py.Unit(1, "cm"))
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        mm = grid_py.convert_width(uc, "mm", valueOnly=True)
        assert len(mm) == 2
        assert abs(mm[0] - 5.0) < 0.1
        assert abs(mm[1] - 10.0) < 0.1

    def test_unit_zero(self):
        u = grid_py.Unit(0, "mm")
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        mm = float(np.squeeze(grid_py.convert_width(u, "mm", valueOnly=True)))
        assert mm == 0.0


# ---------------------------------------------------------------------------
# R bugs.R: known bug fixes
# ---------------------------------------------------------------------------

class TestRBugFixes:
    """R grid/tests/bugs.R — regression tests for fixed bugs."""

    def test_physical_unit_in_zero_height_viewport(self):
        """R bugs.R:4-6: unit(72,'bigpts') in viewport(height=0)."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(height=grid_py.Unit(0, "mm")))
        result = grid_py.convert_height(
            grid_py.Unit(72, "points"), "inches", valueOnly=True)
        assert np.isfinite(float(np.squeeze(result)))
        grid_py.up_viewport()


# ---------------------------------------------------------------------------
# Viewport coordinate conversion — core grid semantics
# ---------------------------------------------------------------------------

class TestViewportCoordinates:
    """Test viewport xscale/yscale and native unit conversions."""

    def test_native_x_conversion(self):
        """50 out of 0-100 native = 0.5 npc."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(xscale=(0, 100)))
        x = grid_py.convert_width(grid_py.Unit(50, "native"), "npc", valueOnly=True)
        assert abs(float(np.squeeze(x)) - 0.5) < 0.01
        grid_py.up_viewport()

    def test_native_y_conversion(self):
        """25 out of 0-50 native = 0.5 npc."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(yscale=(0, 50)))
        y = grid_py.convert_height(grid_py.Unit(25, "native"), "npc", valueOnly=True)
        assert abs(float(np.squeeze(y)) - 0.5) < 0.01
        grid_py.up_viewport()

    def test_npc_to_mm_roundtrip(self):
        """convert npc → mm → npc should roundtrip."""
        grid_py.grid_newpage(width=6, height=4, dpi=72)
        orig = grid_py.Unit(0.3, "npc")
        mm = grid_py.convert_width(orig, "mm", valueOnly=True)
        back = grid_py.convert_width(
            grid_py.Unit(float(np.squeeze(mm)), "mm"), "npc", valueOnly=True)
        assert abs(float(np.squeeze(back)) - 0.3) < 0.001

    def test_nested_viewport_scales(self):
        """Inner viewport scale should be independent of outer."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(xscale=(0, 10)))
        grid_py.push_viewport(grid_py.Viewport(xscale=(0, 100),
            width=grid_py.Unit(0.5, "npc")))
        # 50 native in inner = 0.5 npc of inner viewport
        x = grid_py.convert_width(grid_py.Unit(50, "native"), "npc", valueOnly=True)
        assert abs(float(np.squeeze(x)) - 0.5) < 0.01
        grid_py.up_viewport()
        grid_py.up_viewport()

    def test_mm_independent_of_viewport(self):
        """Absolute units (mm) should not depend on viewport."""
        grid_py.grid_newpage(width=8, height=6, dpi=72)
        mm1 = grid_py.convert_width(grid_py.Unit(10, "mm"), "mm", valueOnly=True)
        grid_py.push_viewport(grid_py.Viewport(
            width=grid_py.Unit(0.2, "npc")))
        mm2 = grid_py.convert_width(grid_py.Unit(10, "mm"), "mm", valueOnly=True)
        grid_py.up_viewport()
        assert abs(float(np.squeeze(mm1)) - float(np.squeeze(mm2))) < 0.01


# ---------------------------------------------------------------------------
# GridLayout — null unit distribution
# ---------------------------------------------------------------------------

class TestGridLayout:
    """R: GridLayout with null units should distribute remaining space."""

    def test_null_unit_fills_remainder(self):
        """Layout: 10mm fixed + 1null + 5mm fixed → null gets rest."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        layout = grid_py.GridLayout(
            nrow=3, ncol=1,
            heights=grid_py.unit_c(
                grid_py.Unit(10, "mm"),
                grid_py.Unit(1, "null"),
                grid_py.Unit(5, "mm"),
            ),
        )
        vp = grid_py.Viewport(layout=layout)
        grid_py.push_viewport(vp)

        # Row 1: 10mm
        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=1, layout_pos_col=1))
        h1 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()
        assert abs(h1 - 10.0) < 0.5

        # Row 3: 5mm
        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=3, layout_pos_col=1))
        h3 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()
        assert abs(h3 - 5.0) < 0.5

        # Row 2: remaining
        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=2, layout_pos_col=1))
        h2 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()
        total = 4 * 25.4  # 4 inches in mm
        assert abs(h2 - (total - 15.0)) < 1.0

        grid_py.up_viewport()

    def test_two_null_units_equal(self):
        """Two equal null units should get equal share."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        layout = grid_py.GridLayout(
            nrow=2, ncol=1,
            heights=grid_py.unit_c(
                grid_py.Unit(1, "null"),
                grid_py.Unit(1, "null"),
            ),
        )
        vp = grid_py.Viewport(layout=layout)
        grid_py.push_viewport(vp)

        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=1, layout_pos_col=1))
        h1 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()

        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=2, layout_pos_col=1))
        h2 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()

        assert abs(h1 - h2) < 0.5  # should be equal
        grid_py.up_viewport()

    def test_weighted_null_units(self):
        """null(2) : null(1) should be 2:1 ratio."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        layout = grid_py.GridLayout(
            nrow=2, ncol=1,
            heights=grid_py.unit_c(
                grid_py.Unit(2, "null"),
                grid_py.Unit(1, "null"),
            ),
        )
        vp = grid_py.Viewport(layout=layout)
        grid_py.push_viewport(vp)

        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=1, layout_pos_col=1))
        h1 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()

        grid_py.push_viewport(grid_py.Viewport(layout_pos_row=2, layout_pos_col=1))
        h2 = float(np.squeeze(
            grid_py.convert_height(grid_py.Unit(1, "npc"), "mm", valueOnly=True)))
        grid_py.up_viewport()

        ratio = h1 / h2
        assert abs(ratio - 2.0) < 0.1
        grid_py.up_viewport()


# ---------------------------------------------------------------------------
# Text measurement — grobWidth/grobHeight with rotation
# ---------------------------------------------------------------------------

class TestTextMeasurement:
    """Test text_grob measurement matches R's grobWidth/grobHeight."""

    def test_width_details_positive(self):
        g = text_grob(label="hello", x=0.5, y=0.5)
        w = width_details(g)
        w_mm = float(np.squeeze(grid_py.convert_width(w, "mm", valueOnly=True)))
        assert w_mm > 0

    def test_height_details_positive(self):
        g = text_grob(label="hello", x=0.5, y=0.5)
        h = height_details(g)
        h_mm = float(np.squeeze(grid_py.convert_height(h, "mm", valueOnly=True)))
        assert h_mm > 0

    def test_rot90_swaps_width_height(self):
        """R: grobWidth(textGrob('X', rot=90)) ≈ grobHeight(textGrob('X', rot=0))"""
        g0 = text_grob(label="X", x=0.5, y=0.5, rot=0)
        g90 = text_grob(label="X", x=0.5, y=0.5, rot=90)
        w0 = float(np.squeeze(grid_py.convert_width(width_details(g0), "mm", valueOnly=True)))
        h0 = float(np.squeeze(grid_py.convert_height(height_details(g0), "mm", valueOnly=True)))
        w90 = float(np.squeeze(grid_py.convert_width(width_details(g90), "mm", valueOnly=True)))
        h90 = float(np.squeeze(grid_py.convert_height(height_details(g90), "mm", valueOnly=True)))
        # After 90° rotation: width becomes height and vice versa
        assert abs(w90 - h0) < 0.5
        assert abs(h90 - w0) < 0.5

    def test_longer_text_wider(self):
        g1 = text_grob(label="a", x=0.5, y=0.5)
        g2 = text_grob(label="hello world this is long", x=0.5, y=0.5)
        w1 = float(np.squeeze(grid_py.convert_width(width_details(g1), "mm", valueOnly=True)))
        w2 = float(np.squeeze(grid_py.convert_width(width_details(g2), "mm", valueOnly=True)))
        assert w2 > w1 * 3

    def test_larger_fontsize_wider(self):
        g1 = text_grob(label="X", x=0.5, y=0.5, gp=grid_py.Gpar(fontsize=10))
        g2 = text_grob(label="X", x=0.5, y=0.5, gp=grid_py.Gpar(fontsize=20))
        w1 = float(np.squeeze(grid_py.convert_width(width_details(g1), "mm", valueOnly=True)))
        w2 = float(np.squeeze(grid_py.convert_width(width_details(g2), "mm", valueOnly=True)))
        ratio = w2 / w1
        assert 1.8 < ratio < 2.2  # ~2x

    def test_descent_details(self):
        """R: grobDescent(textGrob('jA')) — should be positive."""
        g = text_grob(label="jA", x=0.5, y=0.5)
        d = descent_details(g)
        d_mm = float(np.squeeze(grid_py.convert_height(d, "mm", valueOnly=True)))
        assert d_mm > 0  # 'j' has descender


# ---------------------------------------------------------------------------
# Cairo backend specifics
# ---------------------------------------------------------------------------

class TestCairoBackend:
    """Tests specific to grid_py's Cairo rendering backend."""

    def test_newpage_creates_surface(self):
        grid_py.grid_newpage(width=4, height=3, dpi=72)
        state = grid_py.get_state()
        renderer = state.get_renderer()
        assert renderer is not None

    def test_newpage_dimensions(self):
        """Width/height in inches should produce correct pixel dimensions."""
        grid_py.grid_newpage(width=4, height=3, dpi=100)
        state = grid_py.get_state()
        renderer = state.get_renderer()
        # 4in * 100dpi = 400px, 3in * 100dpi = 300px
        assert renderer._width_px == 400
        assert renderer._height_px == 300

    def test_save_png(self, tmp_path):
        """Should produce a valid PNG file."""
        grid_py.grid_newpage(width=2, height=2, dpi=72)
        grid_py.grid_rect(gp=grid_py.Gpar(fill="red"))
        outfile = str(tmp_path / "test.png")
        grid_py.get_state().get_renderer().write_to_png(outfile)
        import os
        assert os.path.exists(outfile)
        assert os.path.getsize(outfile) > 100  # not empty

    def test_multiple_pages(self, tmp_path):
        """Multiple grid_newpage calls should not leak state."""
        for i in range(5):
            grid_py.grid_newpage(width=2, height=2, dpi=72)
            grid_py.grid_rect(gp=grid_py.Gpar(fill="blue"))

    def test_viewport_stack_cleanup(self):
        """up_viewport should properly pop from stack."""
        grid_py.grid_newpage(width=2, height=2, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(name="v1"))
        grid_py.push_viewport(grid_py.Viewport(name="v2"))
        grid_py.up_viewport()  # pop v2
        grid_py.up_viewport()  # pop v1
        # Should not error — we're back at root


# ---------------------------------------------------------------------------
# Drawing primitives — smoke tests
# ---------------------------------------------------------------------------

class TestDrawPrimitives:
    """Test that all drawing primitives execute without error."""

    @pytest.fixture(autouse=True)
    def _newpage(self):
        grid_py.grid_newpage(width=4, height=4, dpi=72)

    def test_grid_rect(self):
        grid_py.grid_rect(
            x=grid_py.Unit(0.5, "npc"),
            y=grid_py.Unit(0.5, "npc"),
            width=grid_py.Unit(0.8, "npc"),
            height=grid_py.Unit(0.8, "npc"),
            gp=grid_py.Gpar(fill="lightblue", col="black"),
        )

    def test_grid_circle(self):
        grid_py.grid_circle(
            x=grid_py.Unit(0.5, "npc"),
            y=grid_py.Unit(0.5, "npc"),
            r=grid_py.Unit(0.3, "npc"),
            gp=grid_py.Gpar(fill="red"),
        )

    def test_grid_text(self):
        grid_py.grid_text(
            label="Hello Grid",
            x=grid_py.Unit(0.5, "npc"),
            y=grid_py.Unit(0.5, "npc"),
            gp=grid_py.Gpar(fontsize=14),
        )

    def test_grid_lines(self):
        grid_py.grid_lines(
            x=grid_py.Unit([0.1, 0.5, 0.9], "npc"),
            y=grid_py.Unit([0.1, 0.9, 0.1], "npc"),
        )

    def test_grid_segments(self):
        grid_py.grid_segments(
            x0=grid_py.Unit(0.1, "npc"),
            y0=grid_py.Unit(0.1, "npc"),
            x1=grid_py.Unit(0.9, "npc"),
            y1=grid_py.Unit(0.9, "npc"),
        )

    def test_grid_points(self):
        grid_py.grid_points(
            x=grid_py.Unit([0.2, 0.5, 0.8], "npc"),
            y=grid_py.Unit([0.3, 0.7, 0.4], "npc"),
            pch=16,
        )

    def test_grid_polygon(self):
        grid_py.grid_polygon(
            x=grid_py.Unit([0.2, 0.5, 0.8], "npc"),
            y=grid_py.Unit([0.2, 0.8, 0.2], "npc"),
            gp=grid_py.Gpar(fill="green"),
        )

    def test_grid_xaxis(self):
        grid_py.push_viewport(grid_py.Viewport(xscale=(0, 10)))
        grid_py.grid_xaxis()
        grid_py.up_viewport()

    def test_grid_yaxis(self):
        grid_py.push_viewport(grid_py.Viewport(yscale=(0, 10)))
        grid_py.grid_yaxis()
        grid_py.up_viewport()


# ---------------------------------------------------------------------------
# Gpar inheritance — R: grid.get.gpar in nested viewports
# ---------------------------------------------------------------------------

class TestGparInheritance:
    """R: gpar settings cascade through the viewport tree into grob measurement."""

    def test_fontsize_via_explicit_gp(self):
        """Explicit gp on text_grob controls measurement."""
        g10 = text_grob(label="X", x=0.5, y=0.5, gp=grid_py.Gpar(fontsize=10))
        g20 = text_grob(label="X", x=0.5, y=0.5, gp=grid_py.Gpar(fontsize=20))
        w10 = float(np.squeeze(grid_py.convert_width(
            width_details(g10), "mm", valueOnly=True)))
        w20 = float(np.squeeze(grid_py.convert_width(
            width_details(g20), "mm", valueOnly=True)))
        assert w20 > w10 * 1.5  # ~2x

    def test_fontsize_inherited_from_viewport(self):
        """R: grobWidth(textGrob('X')) in viewport(gp=gpar(fontsize=20))
        should use fontsize=20, not the default."""
        grid_py.grid_newpage(width=4, height=4, dpi=72)
        grid_py.push_viewport(grid_py.Viewport(gp=grid_py.Gpar(fontsize=20)))
        g = text_grob(label="X", x=0.5, y=0.5)  # no explicit gp
        w = float(np.squeeze(grid_py.convert_width(
            width_details(g), "mm", valueOnly=True)))
        grid_py.up_viewport()

        grid_py.push_viewport(grid_py.Viewport(gp=grid_py.Gpar(fontsize=10)))
        g2 = text_grob(label="X", x=0.5, y=0.5)
        w2 = float(np.squeeze(grid_py.convert_width(
            width_details(g2), "mm", valueOnly=True)))
        grid_py.up_viewport()

        # fontsize=20 viewport should produce wider measurement
        assert w > w2 * 1.5


# ---------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------

class TestColour:
    """Test grid_py colour parsing (R: col2rgb)."""

    def test_named_colour(self):
        """R named colours should be resolved."""
        from grid_py._colour import parse_r_colour
        r, g, b, a = parse_r_colour("cornflowerblue")
        assert 0.3 < r < 0.5
        assert 0.5 < g < 0.7
        assert 0.8 < b < 1.0

    def test_hex_colour(self):
        from grid_py._colour import parse_r_colour
        r, g, b, a = parse_r_colour("#FF8040")
        assert abs(r - 1.0) < 0.01
        assert abs(g - 0.502) < 0.01
        assert abs(b - 0.251) < 0.01

    def test_transparent(self):
        from grid_py._colour import parse_r_colour
        r, g, b, a = parse_r_colour("transparent")
        assert a == 0.0


# ---------------------------------------------------------------------------
# Unit type checking
# ---------------------------------------------------------------------------

class TestUnitTypes:
    def test_is_unit(self):
        assert grid_py.is_unit(grid_py.Unit(1, "mm")) is True
        assert grid_py.is_unit(5) is False
        assert grid_py.is_unit("npc") is False

    def test_unit_type(self):
        assert grid_py.unit_type(grid_py.Unit(1, "mm")) == "mm"
        assert grid_py.unit_type(grid_py.Unit(1, "npc")) == "npc"

    def test_unit_length(self):
        u = grid_py.unit_c(grid_py.Unit(1, "mm"), grid_py.Unit(2, "cm"))
        assert grid_py.unit_length(u) == 2
