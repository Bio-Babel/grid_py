"""Tests for grid_py.renderer — CairoRenderer drawing primitives and output."""

from __future__ import annotations

import io
import math
import os
import tempfile

import numpy as np
import pytest

from grid_py.renderer import CairoRenderer, _parse_colour
from grid_py._gpar import Gpar
from grid_py._state import get_state
from grid_py._draw import grid_newpage, grid_draw
from grid_py._grob import Grob
from grid_py._size import calc_string_metric


# ------------------------------------------------------------------ #
# Fixtures                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global grid state between tests."""
    state = get_state()
    state.reset()
    yield


@pytest.fixture
def renderer():
    """Create a small CairoRenderer for testing."""
    return CairoRenderer(width=4, height=3, dpi=100, bg="white")


# ------------------------------------------------------------------ #
# CairoRenderer construction                                        #
# ------------------------------------------------------------------ #


class TestRendererConstruction:

    def test_image_surface(self):
        r = CairoRenderer(width=4, height=3, dpi=100)
        assert r.width_in == 4.0
        assert r.height_in == 3.0
        assert r._width_px == 400
        assert r._height_px == 300

    def test_pdf_surface(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            r = CairoRenderer(width=4, height=3, surface_type="pdf", filename=path)
            r.draw_rect(0.5, 0.5, 0.5, 0.5, gp=None)
            r.finish()
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_svg_surface(self):
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            path = f.name
        try:
            r = CairoRenderer(width=4, height=3, surface_type="svg", filename=path)
            r.draw_rect(0.5, 0.5, 0.5, 0.5, gp=None)
            r.finish()
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_ps_surface(self):
        with tempfile.NamedTemporaryFile(suffix=".ps", delete=False) as f:
            path = f.name
        try:
            r = CairoRenderer(width=4, height=3, surface_type="ps", filename=path)
            r.draw_rect(0.5, 0.5, 0.5, 0.5, gp=None)
            r.finish()
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_unknown_surface_raises(self):
        with pytest.raises(ValueError, match="Unknown surface_type"):
            CairoRenderer(surface_type="notreal")

    def test_vector_without_filename_raises(self):
        with pytest.raises(ValueError, match="filename is required"):
            CairoRenderer(surface_type="pdf")


# ------------------------------------------------------------------ #
# PNG output                                                         #
# ------------------------------------------------------------------ #


class TestPNGOutput:

    def test_to_png_bytes(self, renderer):
        renderer.draw_rect(0.5, 0.5, 0.5, 0.5, gp=None)
        data = renderer.to_png_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 100
        # PNG magic bytes
        assert data[:4] == b"\x89PNG"

    def test_write_to_png(self, renderer):
        renderer.draw_circle(0.5, 0.5, 0.1, gp=None)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            renderer.write_to_png(path)
            assert os.path.getsize(path) > 100
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
# Drawing primitives                                                 #
# ------------------------------------------------------------------ #


class TestDrawRect:

    def test_no_error(self, renderer):
        renderer.draw_rect(0.5, 0.5, 0.8, 0.6, gp=None)

    def test_with_gpar(self, renderer):
        gp = Gpar(fill="red", col="blue", lwd=2)
        renderer.draw_rect(0.5, 0.5, 0.4, 0.3, gp=gp)

    def test_justify(self, renderer):
        renderer.draw_rect(0.0, 0.0, 0.5, 0.5, hjust=0.0, vjust=0.0, gp=None)

    def test_produces_output(self, renderer):
        renderer.draw_rect(0.5, 0.5, 0.8, 0.6,
                           gp=Gpar(fill="red", col="black"))
        data = renderer.to_png_bytes()
        # Should be larger than a blank image
        blank = CairoRenderer(width=4, height=3, dpi=100).to_png_bytes()
        assert len(data) != len(blank) or data != blank


class TestDrawCircle:

    def test_no_error(self, renderer):
        renderer.draw_circle(0.5, 0.5, 0.2, gp=None)

    def test_with_fill(self, renderer):
        renderer.draw_circle(0.3, 0.7, 0.1, gp=Gpar(fill="green"))


class TestDrawLine:

    def test_simple_line(self, renderer):
        x = np.array([0.1, 0.9])
        y = np.array([0.1, 0.9])
        renderer.draw_line(x, y, gp=None)

    def test_multipoint(self, renderer):
        x = np.linspace(0, 1, 20)
        y = np.sin(x * math.pi)
        renderer.draw_line(x, y, gp=Gpar(col="blue", lwd=2))

    def test_short_array_skipped(self, renderer):
        """Lines with < 2 points should be silently skipped."""
        renderer.draw_line(np.array([0.5]), np.array([0.5]), gp=None)

    def test_recycling(self, renderer):
        """Mismatched x/y lengths should be recycled (R behaviour)."""
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.5])
        renderer.draw_line(x, y, gp=None)


class TestDrawPolyline:

    def test_single_group(self, renderer):
        x = np.array([0.1, 0.5, 0.9])
        y = np.array([0.2, 0.8, 0.2])
        renderer.draw_polyline(x, y, id_=None, gp=None)

    def test_multiple_groups(self, renderer):
        x = np.array([0.1, 0.3, 0.5, 0.7])
        y = np.array([0.2, 0.8, 0.2, 0.8])
        id_ = np.array([1, 1, 2, 2])
        renderer.draw_polyline(x, y, id_=id_, gp=Gpar(col="red"))


class TestDrawSegments:

    def test_basic(self, renderer):
        x0 = np.array([0.1, 0.5])
        y0 = np.array([0.1, 0.5])
        x1 = np.array([0.9, 0.5])
        y1 = np.array([0.9, 0.1])
        renderer.draw_segments(x0, y0, x1, y1, gp=None)

    def test_mismatched_lengths(self, renderer):
        """Shorter arrays should not crash — take minimum."""
        x0 = np.array([0.1, 0.5, 0.9])
        y0 = np.array([0.1])
        x1 = np.array([0.9, 0.5])
        y1 = np.array([0.9, 0.1])
        renderer.draw_segments(x0, y0, x1, y1, gp=None)


class TestDrawPolygon:

    def test_triangle(self, renderer):
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.2, 0.8, 0.2])
        renderer.draw_polygon(x, y, gp=Gpar(fill="yellow", col="black"))

    def test_too_few_points(self, renderer):
        """< 3 points should be silently skipped."""
        renderer.draw_polygon(np.array([0.5, 0.6]), np.array([0.5, 0.6]), gp=None)


class TestDrawText:

    def test_basic(self, renderer):
        renderer.draw_text(0.5, 0.5, "Hello", gp=None)

    def test_rotated(self, renderer):
        renderer.draw_text(0.5, 0.5, "Rotated", rot=45.0, gp=None)

    def test_with_font(self, renderer):
        gp = Gpar(fontsize=18, fontfamily="serif")
        renderer.draw_text(0.5, 0.5, "Serif 18pt", gp=gp)

    def test_justify(self, renderer):
        renderer.draw_text(0.0, 0.0, "BL", hjust=0.0, vjust=0.0, gp=None)
        renderer.draw_text(1.0, 1.0, "TR", hjust=1.0, vjust=1.0, gp=None)


class TestDrawPoints:

    def test_basic(self, renderer):
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.3, 0.6, 0.9])
        renderer.draw_points(x, y, size=5.0, gp=Gpar(col="red"))

    def test_per_point_colour(self, renderer):
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.3, 0.6, 0.9])
        gp = Gpar(col=["red", "green", "blue"])
        renderer.draw_points(x, y, gp=gp)


class TestDrawPath:

    def test_single_path(self, renderer):
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.2, 0.8, 0.2])
        path_id = np.array([1, 1, 1])
        renderer.draw_path(x, y, path_id, rule="winding", gp=Gpar(fill="cyan"))

    def test_evenodd_rule(self, renderer):
        x = np.array([0.2, 0.5, 0.8, 0.3, 0.5, 0.7])
        y = np.array([0.2, 0.8, 0.2, 0.3, 0.7, 0.3])
        path_id = np.array([1, 1, 1, 2, 2, 2])
        renderer.draw_path(x, y, path_id, rule="evenodd", gp=None)


class TestDrawRoundRect:

    def test_basic(self, renderer):
        renderer.draw_roundrect(0.5, 0.5, 0.6, 0.4, r=0.05, gp=Gpar(fill="lightblue"))

    def test_zero_radius(self, renderer):
        """r=0 should draw a normal rectangle."""
        renderer.draw_roundrect(0.5, 0.5, 0.6, 0.4, r=0.0, gp=None)


class TestDrawRaster:

    def test_rgb_image(self, renderer):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # red
        renderer.draw_raster(img, 0.2, 0.2, 0.6, 0.6)

    def test_rgba_image(self, renderer):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        img[:, :, :3] = 128
        img[:, :, 3] = 200
        renderer.draw_raster(img, 0.1, 0.1, 0.8, 0.8, interpolate=False)

    def test_greyscale_image(self, renderer):
        img = np.linspace(0, 255, 100, dtype=np.uint8).reshape(10, 10)
        renderer.draw_raster(img, 0.0, 0.0, 1.0, 1.0)


# ------------------------------------------------------------------ #
# Pen-position drawing (move.to / line.to)                           #
# ------------------------------------------------------------------ #


class TestMoveTo:

    def test_move_then_line(self, renderer):
        renderer.move_to(0.1, 0.1)
        renderer.line_to(0.9, 0.9, gp=Gpar(col="black"))


# ------------------------------------------------------------------ #
# Clipping                                                           #
# ------------------------------------------------------------------ #


class TestClipping:

    def test_push_pop_clip(self, renderer):
        renderer.push_clip(0.2, 0.2, 0.8, 0.8)
        renderer.draw_rect(0.0, 0.0, 1.0, 1.0, gp=Gpar(fill="red"))
        renderer.pop_clip()


# ------------------------------------------------------------------ #
# new_page                                                           #
# ------------------------------------------------------------------ #


class TestNewPage:

    def test_clears_canvas(self, renderer):
        renderer.draw_rect(0.5, 0.5, 0.8, 0.8, gp=Gpar(fill="red"))
        before = renderer.to_png_bytes()
        renderer.new_page(bg="white")
        after = renderer.to_png_bytes()
        # After clearing, should be a blank white image (different from before)
        assert before != after


# ------------------------------------------------------------------ #
# _parse_colour                                                      #
# ------------------------------------------------------------------ #


class TestParseColour:

    def test_none(self):
        assert _parse_colour(None) == (0.0, 0.0, 0.0, 1.0)

    def test_named_colours(self):
        r, g, b, a = _parse_colour("red")
        assert (r, g, b) == (1.0, 0.0, 0.0)
        assert a == 1.0

    def test_hex6(self):
        r, g, b, a = _parse_colour("#00FF00")
        assert (r, g, b, a) == pytest.approx((0.0, 1.0, 0.0, 1.0))

    def test_hex8(self):
        r, g, b, a = _parse_colour("#FF000080")
        assert r == pytest.approx(1.0)
        assert a == pytest.approx(128 / 255.0)

    def test_grey_number(self):
        r, g, b, a = _parse_colour("grey50")
        # R's grey50 = #7F7F7F = 127/255 ≈ 0.498
        assert r == pytest.approx(0x7F / 255.0)
        assert g == pytest.approx(0x7F / 255.0)

    def test_gray_variant(self):
        assert _parse_colour("gray0") == (0.0, 0.0, 0.0, 1.0)
        r, _, _, _ = _parse_colour("gray100")
        assert r == pytest.approx(1.0)

    def test_transparent(self):
        _, _, _, a = _parse_colour("transparent")
        assert a == 0.0

    def test_na_string(self):
        _, _, _, a = _parse_colour("NA")
        assert a == 0.0

    def test_rgb_tuple(self):
        assert _parse_colour((0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5, 1.0)

    def test_rgba_tuple(self):
        assert _parse_colour((1.0, 0.0, 0.0, 0.5)) == (1.0, 0.0, 0.0, 0.5)

    def test_single_element_list(self):
        r, g, b, a = _parse_colour(["red"])
        assert (r, g, b) == (1.0, 0.0, 0.0)

    def test_numeric_float(self):
        r, g, b, a = _parse_colour(0.7)
        assert r == pytest.approx(0.7)


# ------------------------------------------------------------------ #
# Gpar application on renderer                                      #
# ------------------------------------------------------------------ #


class TestGparRendering:

    def test_line_type_dashed(self, renderer):
        """Dashed line type should not error."""
        gp = Gpar(col="black", lty="dashed", lwd=2)
        x = np.array([0.1, 0.9])
        y = np.array([0.5, 0.5])
        renderer.draw_line(x, y, gp=gp)

    def test_line_type_dotted(self, renderer):
        gp = Gpar(col="black", lty="dotted")
        renderer.draw_line(np.array([0.1, 0.9]), np.array([0.3, 0.3]), gp=gp)

    def test_alpha(self, renderer):
        gp = Gpar(fill="blue", alpha=0.3)
        renderer.draw_rect(0.5, 0.5, 0.5, 0.5, gp=gp)

    def test_font_bold(self, renderer):
        gp = Gpar(fontsize=14)
        gp._params["fontface"] = 2  # bold
        renderer.draw_text(0.5, 0.5, "Bold", gp=gp)

    def test_font_italic(self, renderer):
        gp = Gpar(fontsize=14)
        gp._params["fontface"] = 3  # italic
        renderer.draw_text(0.5, 0.5, "Italic", gp=gp)

    def test_lineend_cap(self, renderer):
        gp = Gpar(col="black", lineend="round", lwd=5)
        renderer.draw_line(np.array([0.2, 0.8]), np.array([0.5, 0.5]), gp=gp)

    def test_linejoin(self, renderer):
        gp = Gpar(col="black", linejoin="bevel", lwd=5)
        renderer.draw_polygon(
            np.array([0.2, 0.5, 0.8]),
            np.array([0.2, 0.8, 0.2]),
            gp=gp,
        )


# ------------------------------------------------------------------ #
# Text metrics (calc_string_metric with Cairo)                       #
# ------------------------------------------------------------------ #


class TestCalcStringMetric:

    def test_returns_dict(self):
        m = calc_string_metric("Hello")
        assert "ascent" in m
        assert "descent" in m
        assert "width" in m

    def test_positive_values(self):
        m = calc_string_metric("Hello")
        assert m["ascent"] > 0
        assert m["descent"] >= 0
        assert m["width"] > 0

    def test_wider_for_longer_text(self):
        m1 = calc_string_metric("Hi")
        m2 = calc_string_metric("Hello World")
        assert m2["width"] > m1["width"]

    def test_larger_font(self):
        m1 = calc_string_metric("X", gp=Gpar(fontsize=10))
        m2 = calc_string_metric("X", gp=Gpar(fontsize=20))
        assert m2["width"] > m1["width"]
        assert m2["ascent"] > m1["ascent"]

    def test_empty_string(self):
        m = calc_string_metric("")
        assert m["width"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ #
# Integration: grid_newpage + grid_draw + renderer output            #
# ------------------------------------------------------------------ #


class TestIntegration:

    def test_newpage_creates_renderer(self):
        grid_newpage()
        renderer = get_state().get_renderer()
        assert renderer is not None
        assert isinstance(renderer, CairoRenderer)

    def test_draw_rect_produces_png(self):
        grid_newpage(width=3, height=2, dpi=72)
        g = Grob(name="r", _grid_class="rect")
        g.x = 0.5
        g.y = 0.5
        g.width = 0.8
        g.height = 0.6
        g.hjust = 0.5
        g.vjust = 0.5
        grid_draw(g)
        data = get_state().get_renderer().to_png_bytes()
        assert data[:4] == b"\x89PNG"

    def test_multiple_grobs(self):
        grid_newpage(width=4, height=3, dpi=100)
        for cls, kwargs in [
            ("rect", {"x": 0.5, "y": 0.5, "width": 0.3, "height": 0.3,
                       "hjust": 0.5, "vjust": 0.5}),
            ("circle", {"x": 0.2, "y": 0.8, "r": 0.05}),
            ("text", {"x": 0.5, "y": 0.2, "label": "test", "rot": 0.0,
                       "hjust": 0.5, "vjust": 0.5}),
            ("null", {}),
        ]:
            g = Grob(name=f"g_{cls}", _grid_class=cls)
            for k, v in kwargs.items():
                setattr(g, k, v)
            grid_draw(g)
        data = get_state().get_renderer().to_png_bytes()
        assert len(data) > 200

    def test_write_to_file(self):
        grid_newpage(width=3, height=2, dpi=72)
        g = Grob(name="r", _grid_class="rect")
        g.x = 0.5
        g.y = 0.5
        g.width = 0.6
        g.height = 0.4
        g.hjust = 0.5
        g.vjust = 0.5
        grid_draw(g)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            get_state().get_renderer().write_to_png(path)
            assert os.path.getsize(path) > 100
        finally:
            os.unlink(path)

    def test_renderer_text_extents(self):
        grid_newpage()
        renderer = get_state().get_renderer()
        m = renderer.text_extents("Hello World")
        assert m["width"] > 0
        assert m["ascent"] > 0


# ------------------------------------------------------------------ #
# Non-finite ("pen up") coordinate semantics                        #
#                                                                    #
# Gold standard: grid/src/grid.c — gridLines / L_segments /         #
# L_polygon / L_path / gridPoints. A non-finite (NA/NaN/Inf) device #
# coordinate breaks lines/polygons into finite runs, is skipped for #
# segments/points, and is an error for a graphics path.             #
# ------------------------------------------------------------------ #

class TestNonFiniteCoords:
    def test_split_finite_runs_matches_gridlines(self):
        from grid_py._renderer_base import split_finite_runs
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        runs = split_finite_runs(x, y)
        # grid.c: emit each run with length >= 2 (i - start > 1)
        assert len(runs) == 2
        assert list(runs[0][0]) == [1.0, 2.0]
        assert list(runs[1][0]) == [4.0, 5.0]

    def test_lone_point_between_breaks_drops(self):
        from grid_py._renderer_base import split_finite_runs
        # a single finite point flanked by NAs draws no line (run len 1)
        runs = split_finite_runs(
            np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0]), np.ones(6),
        )
        assert len(runs) == 1
        assert list(runs[0][0]) == [5.0, 6.0]

    def test_na_in_either_axis_breaks(self):
        from grid_py._renderer_base import split_finite_runs
        # y-only NA still breaks the line (grid.c checks both axes)
        runs = split_finite_runs(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]),
        )
        assert len(runs) == 0

    def test_draw_line_with_nan_does_not_raise(self, renderer):
        # Regression: NaN fed to cairo line_to used to collapse to the
        # device origin, drawing a spurious stroke to (0,0).
        renderer.draw_line(
            np.array([0.1, 0.5, np.nan, 0.7, 0.9]),
            np.array([0.1, 0.5, np.nan, 0.7, 0.9]),
            gp=Gpar(col="black"),
        )

    def test_draw_polyline_ids_with_nan(self, renderer):
        renderer.draw_polyline(
            np.array([0.1, 0.5, np.nan, 0.6, 0.9]),
            np.array([0.1, 0.5, np.nan, 0.6, 0.9]),
            id_=np.array([1, 1, 1, 2, 2]),
            gp=Gpar(col="red"),
        )

    def test_draw_segments_skips_nonfinite_segment(self, renderer):
        # middle segment has a NaN endpoint → skipped, others drawn
        renderer.draw_segments(
            x0=np.array([0.1, np.nan, 0.5]),
            y0=np.array([0.1, 0.2, 0.5]),
            x1=np.array([0.2, 0.4, 0.9]),
            y1=np.array([0.4, 0.4, 0.9]),
            gp=Gpar(col="black"),
        )

    def test_draw_points_skips_nonfinite(self, renderer):
        renderer.draw_points(
            np.array([0.1, np.nan, 0.5]),
            np.array([0.1, 0.2, np.nan]),
            size=0.02, pch=19, gp=Gpar(col="black", fill="black"),
        )

    def test_draw_polygon_splits_into_subpolygons(self, renderer):
        # grid.c L_polygon: a NaN vertex splits into separate closed
        # sub-polygons (does not connect across the break)
        renderer.draw_polygon(
            np.array([0.1, 0.3, 0.2, np.nan, 0.6, 0.9, 0.75]),
            np.array([0.1, 0.1, 0.3, np.nan, 0.6, 0.6, 0.9]),
            gp=Gpar(fill="grey", col="black"),
        )

    def test_draw_path_raises_on_nonfinite(self, renderer):
        # grid.c L_path: "non-finite x or y in graphics path" is an error
        with pytest.raises(ValueError, match="non-finite"):
            renderer.draw_path(
                np.array([0.1, 0.5, np.nan, 0.9]),
                np.array([0.1, 0.5, 0.5, 0.9]),
                path_id=np.array([1, 1, 1, 1]),
                gp=Gpar(fill="grey"),
            )
