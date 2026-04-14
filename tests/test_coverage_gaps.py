"""Tests targeting uncovered code paths to push coverage from 93% to 95%.

Covers:
  - renderer.py: PDF/SVG surface types, push_clip/pop_clip, draw_path with
    path_id, draw_raster, _set_font edge cases, new_page for vector surfaces
  - _grab.py: grid_grab, grid_grab_expr, grid_force, grid_revert, grid_cap,
    grid_reorder
  - _highlevel.py: grid_legend, grid_plot_and_legend, grid_grill,
    grid_show_layout, grid_show_viewport, grid_abline, layout_torture,
    grid_strip, grid_panel
  - _draw.py: _render_grob dispatches for circle, pathgrob, rastergrob,
    move.to/line.to, unknown class warning
  - _primitives.py: circle_grob, raster_grob, path_grob, clip_grob,
    null_grob, function_grob edge cases
  - _edit.py: grid_edit, grid_remove, grid_get, grid_set, grid_add,
    grid_gedit, grid_gget, grid_gremove
  - _grob.py: force_grob, reorder_grob, GTree edge cases
"""

from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

sys.path.insert(0, '/scratch/groups/xiaojie/nianping_scratch/Test_grid/grid_py')

import grid_py
from grid_py import (
    CairoRenderer,
    Gpar,
    GPath,
    GridLayout,
    GridState,
    Unit,
    Viewport,
    get_state,
)
from grid_py._grob import (
    GEdit,
    GEditList,
    GList,
    GTree,
    Grob,
    add_grob,
    apply_edit,
    apply_edits,
    edit_grob,
    force_grob,
    get_grob,
    grob_name,
    is_grob,
    remove_grob,
    reorder_grob,
    set_children,
    set_grob,
)
from grid_py._primitives import (
    circle_grob,
    clip_grob,
    function_grob,
    grid_circle,
    grid_clip,
    grid_function,
    grid_null,
    grid_path,
    grid_polygon,
    grid_raster,
    grid_rect,
    grid_text,
    null_grob,
    path_grob,
    polygon_grob,
    raster_grob,
    rect_grob,
    text_grob,
    lines_grob,
    points_grob,
    segments_grob,
    grid_lines,
    grid_points,
    grid_segments,
    move_to_grob,
    grid_move_to,
    line_to_grob,
    grid_line_to,
    roundrect_grob,
    grid_roundrect,
    polyline_grob,
    grid_polyline,
)
from grid_py._draw import (
    _merge_gpar,
    _render_grob,
    _resolve_just,
    _subset_gpar,
    _unit_to_array,
    _unit_to_float,
    grid_draw,
    grid_newpage,
    grid_pretty,
)
from grid_py._edit import (
    grid_add,
    grid_edit,
    grid_gedit,
    grid_get,
    grid_gget,
    grid_gremove,
    grid_remove,
    grid_set,
)
from grid_py._grab import (
    grid_cap,
    grid_force,
    grid_grab,
    grid_grab_expr,
    grid_reorder,
    grid_revert,
)
from grid_py._highlevel import (
    frame_grob,
    grid_abline,
    grid_frame,
    grid_grill,
    grid_legend,
    grid_pack,
    grid_place,
    grid_plot_and_legend,
    grid_show_layout,
    grid_show_viewport,
    grid_strip,
    grid_panel,
    layout_torture,
    legend_grob,
    pack_grob,
    place_grob,
    xaxis_grob,
    yaxis_grob,
    grid_xaxis,
    grid_yaxis,
)
from grid_py._display_list import DLDrawGrob, DisplayList
from grid_py.renderer import _parse_colour


# =========================================================================
# renderer.py -- surface types
# =========================================================================


class TestRendererSurfaceTypes:
    """Test CairoRenderer with different surface types."""

    def test_pdf_surface(self, tmp_path):
        """Create a PDF surface and draw a rect."""
        pdf_path = str(tmp_path / "test.pdf")
        r = CairoRenderer(width=5, height=4, surface_type="pdf", filename=pdf_path)
        r.draw_rect(0.5, 0.5, 0.5, 0.5, gp=Gpar(col="red", fill="blue"))
        r.finish()
        assert os.path.exists(pdf_path)

    def test_svg_surface(self, tmp_path):
        """Create an SVG surface and draw a rect."""
        svg_path = str(tmp_path / "test.svg")
        r = CairoRenderer(width=5, height=4, surface_type="svg", filename=svg_path)
        r.draw_rect(0.5, 0.5, 0.5, 0.5, gp=Gpar(col="blue", fill="green"))
        r.finish()
        assert os.path.exists(svg_path)

    def test_ps_surface(self, tmp_path):
        """Create a PS surface and draw a rect."""
        ps_path = str(tmp_path / "test.ps")
        r = CairoRenderer(width=5, height=4, surface_type="ps", filename=ps_path)
        r.draw_rect(0.5, 0.5, 0.5, 0.5, gp=Gpar(col="black"))
        r.finish()
        assert os.path.exists(ps_path)

    def test_pdf_requires_filename(self):
        """PDF surface without filename should raise ValueError."""
        with pytest.raises(ValueError, match="filename is required"):
            CairoRenderer(surface_type="pdf")

    def test_svg_requires_filename(self):
        """SVG surface without filename should raise ValueError."""
        with pytest.raises(ValueError, match="filename is required"):
            CairoRenderer(surface_type="svg")

    def test_ps_requires_filename(self):
        """PS surface without filename should raise ValueError."""
        with pytest.raises(ValueError, match="filename is required"):
            CairoRenderer(surface_type="ps")

    def test_unknown_surface_type(self):
        """Unknown surface type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown surface_type"):
            CairoRenderer(surface_type="xyzzy")

    def test_vector_viewport_stack(self, tmp_path):
        """Vector surfaces should use pt-based viewport stack."""
        pdf_path = str(tmp_path / "vp_test.pdf")
        r = CairoRenderer(width=5, height=4, surface_type="pdf", filename=pdf_path)
        # Initial viewport stack for vector surfaces is in pt
        x0, y0, w, h, *_ = r._vp_stack[-1]
        assert w == pytest.approx(5 * 72.0)
        assert h == pytest.approx(4 * 72.0)
        r.finish()


# =========================================================================
# renderer.py -- push_clip / pop_clip
# =========================================================================


class TestRendererClipping:
    """Test push_clip / pop_clip on CairoRenderer."""

    def test_push_pop_clip(self):
        r = CairoRenderer(width=5, height=4)
        r.push_clip(0.1, 0.1, 0.9, 0.9)
        # Drawing should be clipped
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="red"))
        r.pop_clip()
        # After pop_clip, clipping restored
        r.draw_rect(0.5, 0.5, 0.2, 0.2, gp=Gpar(fill="blue"))

    def test_push_clip_inverted_coords(self):
        """push_clip should handle x0 > x1 or y0 > y1."""
        r = CairoRenderer(width=5, height=4)
        r.push_clip(0.9, 0.9, 0.1, 0.1)
        r.pop_clip()


# =========================================================================
# renderer.py -- draw_path with multiple path_ids
# =========================================================================


class TestRendererDrawPath:
    """Test draw_path with path_id arrays and fill rules."""

    def test_draw_path_multiple_subpaths(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.1, 0.3, 0.2, 0.6, 0.8, 0.7])
        y = np.array([0.1, 0.3, 0.5, 0.1, 0.3, 0.5])
        path_id = np.array([1, 1, 1, 2, 2, 2])
        r.draw_path(x, y, path_id, rule="winding", gp=Gpar(fill="green", col="black"))

    def test_draw_path_evenodd(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.5])
        y = np.array([0.1, 0.1, 0.9, 0.3, 0.3, 0.7])
        path_id = np.array([1, 1, 1, 2, 2, 2])
        r.draw_path(x, y, path_id, rule="evenodd",
                    gp=Gpar(fill="blue", col="red"))


# =========================================================================
# renderer.py -- draw_raster
# =========================================================================


class TestRendererDrawRaster:
    """Test draw_raster with different image formats."""

    def test_draw_raster_rgba(self):
        r = CairoRenderer(width=5, height=4)
        img = np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
        r.draw_raster(img, x=0.0, y=0.0, w=1.0, h=1.0, interpolate=True)

    def test_draw_raster_rgb(self):
        r = CairoRenderer(width=5, height=4)
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        r.draw_raster(img, x=0.0, y=0.0, w=1.0, h=1.0, interpolate=False)

    def test_draw_raster_greyscale(self):
        r = CairoRenderer(width=5, height=4)
        img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        r.draw_raster(img, x=0.1, y=0.1, w=0.8, h=0.8)

    def test_draw_raster_invalid_shape(self):
        """An image with unexpected shape should be silently skipped."""
        r = CairoRenderer(width=5, height=4)
        img = np.random.randint(0, 255, (10, 10, 5), dtype=np.uint8)
        r.draw_raster(img, x=0.0, y=0.0, w=1.0, h=1.0)


# =========================================================================
# renderer.py -- _set_font edge cases
# =========================================================================


class TestRendererSetFont:
    """Test _set_font with various Gpar font specifications."""

    def test_set_font_bold(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fontface="bold", fontsize=16)
        r._set_font(gp)

    def test_set_font_italic(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fontface="italic", fontsize=12)
        r._set_font(gp)

    def test_set_font_bold_italic(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fontface="bold.italic", fontsize=14)
        r._set_font(gp)

    def test_set_font_numeric_face(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fontface=2, fontsize=12)  # 2 = bold
        r._set_font(gp)

    def test_set_font_with_cex(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fontsize=12, cex=2.0)
        r._set_font(gp)

    def test_set_font_with_family(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fontfamily="monospace", fontsize=10)
        r._set_font(gp)

    def test_set_font_none_gp(self):
        r = CairoRenderer(width=5, height=4)
        result = r._set_font(None)
        assert result > 0

    def test_set_font_vector_surface(self, tmp_path):
        """Font size computation differs for vector surfaces."""
        pdf_path = str(tmp_path / "font_test.pdf")
        r = CairoRenderer(width=5, height=4, surface_type="pdf", filename=pdf_path)
        gp = Gpar(fontsize=12)
        device_fs = r._set_font(gp)
        # For vector surfaces, device_fs should be the raw point size
        assert device_fs == pytest.approx(12.0)
        r.finish()


# =========================================================================
# renderer.py -- new_page for vector vs image surfaces
# =========================================================================


class TestRendererNewPage:
    """Test new_page for image and vector surfaces."""

    def test_new_page_image(self):
        r = CairoRenderer(width=5, height=4, surface_type="image")
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="red"))
        r.new_page(bg="white")

    def test_new_page_pdf(self, tmp_path):
        pdf_path = str(tmp_path / "multipage.pdf")
        r = CairoRenderer(width=5, height=4, surface_type="pdf", filename=pdf_path)
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="red"))
        r.new_page()  # calls show_page on vector surfaces
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="blue"))
        r.finish()

    def test_write_to_png_and_to_png_bytes(self, tmp_path):
        r = CairoRenderer(width=3, height=3)
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="green"))
        png_path = str(tmp_path / "out.png")
        r.write_to_png(png_path)
        assert os.path.exists(png_path)
        data = r.to_png_bytes()
        assert len(data) > 0
        assert data[:4] == b'\x89PNG'


# =========================================================================
# renderer.py -- draw_circle, draw_roundrect, draw_text, draw_points
# =========================================================================


class TestRendererDrawPrimitives:
    """Test renderer drawing primitives that may lack coverage."""

    def test_draw_circle(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_circle(0.5, 0.5, 0.2, gp=Gpar(fill="red", col="black"))

    def test_draw_circle_no_fill(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_circle(0.5, 0.5, 0.2, gp=Gpar(fill="transparent", col="blue"))

    def test_draw_circle_no_stroke(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_circle(0.5, 0.5, 0.2, gp=Gpar(fill="red", col="transparent"))

    def test_draw_roundrect_with_radius(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_roundrect(0.5, 0.5, 0.8, 0.6, r=0.05,
                         gp=Gpar(fill="yellow", col="black"))

    def test_draw_roundrect_zero_radius(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_roundrect(0.5, 0.5, 0.8, 0.6, r=0.0,
                         gp=Gpar(fill="yellow", col="black"))

    def test_draw_text_with_rotation(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_text(0.5, 0.5, "Hello", rot=45.0, gp=Gpar(col="black"))

    def test_draw_text_no_rotation(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_text(0.5, 0.5, "World", rot=0.0, gp=Gpar(col="black"))

    def test_draw_points_with_per_point_col(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.2, 0.5, 0.8])
        gp = Gpar(col=["red", "green", "blue"])
        r.draw_points(x, y, size=2.0, gp=gp)

    def test_draw_points_uniform_col(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.2, 0.5, 0.8])
        y = np.array([0.2, 0.5, 0.8])
        r.draw_points(x, y, size=2.0, gp=Gpar(col="red", fill="red"))

    def test_draw_points_vector_surface(self, tmp_path):
        pdf_path = str(tmp_path / "points.pdf")
        r = CairoRenderer(width=5, height=4, surface_type="pdf", filename=pdf_path)
        x = np.array([0.3, 0.7])
        y = np.array([0.3, 0.7])
        r.draw_points(x, y, size=3.0, gp=Gpar(col="black", fill="black"))
        r.finish()

    def test_move_to_line_to(self):
        r = CairoRenderer(width=5, height=4)
        r.move_to(0.1, 0.1)
        r.line_to(0.9, 0.9, gp=Gpar(col="black"))
        r.line_to(0.5, 0.1, gp=Gpar(col="red"))

    def test_draw_polygon(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.2, 0.8, 0.5])
        y = np.array([0.2, 0.2, 0.8])
        r.draw_polygon(x, y, gp=Gpar(fill="blue", col="black"))

    def test_draw_segments(self):
        r = CairoRenderer(width=5, height=4)
        r.draw_segments(
            np.array([0.1, 0.5]),
            np.array([0.1, 0.5]),
            np.array([0.9, 0.5]),
            np.array([0.9, 0.5]),
            gp=Gpar(col="black"),
        )

    def test_draw_polyline_with_id(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y = np.array([0.2, 0.8, 0.2, 0.8, 0.2])
        id_ = np.array([1, 1, 1, 2, 2])
        r.draw_polyline(x, y, id_=id_, gp=Gpar(col="red"))

    def test_draw_polyline_no_id(self):
        r = CairoRenderer(width=5, height=4)
        x = np.array([0.1, 0.5, 0.9])
        y = np.array([0.2, 0.8, 0.2])
        r.draw_polyline(x, y, gp=Gpar(col="red"))

    def test_draw_line_single_point(self):
        """draw_line with < 2 points should be a no-op."""
        r = CairoRenderer(width=5, height=4)
        r.draw_line(np.array([0.5]), np.array([0.5]), gp=Gpar(col="black"))


# =========================================================================
# renderer.py -- _parse_colour edge cases
# =========================================================================


class TestParseColour:
    """Test _parse_colour with various inputs."""

    def test_transparent(self):
        assert _parse_colour("transparent") == (0.0, 0.0, 0.0, 0.0)

    def test_na(self):
        assert _parse_colour("NA") == (0.0, 0.0, 0.0, 0.0)

    def test_none_str(self):
        assert _parse_colour("none") == (0.0, 0.0, 0.0, 0.0)

    def test_empty_str(self):
        assert _parse_colour("") == (0.0, 0.0, 0.0, 0.0)

    def test_hex_6(self):
        r, g, b, a = _parse_colour("#FF0000")
        assert r == pytest.approx(1.0)
        assert a == pytest.approx(1.0)

    def test_hex_8(self):
        r, g, b, a = _parse_colour("#FF000080")
        assert r == pytest.approx(1.0)
        assert a == pytest.approx(128 / 255.0)

    def test_grey_number(self):
        r, g, b, a = _parse_colour("grey50")
        assert r == pytest.approx(0.5)

    def test_gray_number(self):
        r, g, b, a = _parse_colour("gray25")
        assert r == pytest.approx(0.25)

    def test_named_colour(self):
        r, g, b, a = _parse_colour("red")
        assert r == pytest.approx(1.0) and g == pytest.approx(0.0)

    def test_none_input(self):
        assert _parse_colour(None) == (0.0, 0.0, 0.0, 1.0)

    def test_numeric_input(self):
        r, g, b, a = _parse_colour(0.5)
        assert r == pytest.approx(0.5)

    def test_tuple_3(self):
        assert _parse_colour((0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3, 1.0)

    def test_tuple_4(self):
        assert _parse_colour((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4)

    def test_single_element_list(self):
        # Single-element list wrapping a string
        r, g, b, a = _parse_colour(["red"])
        assert r == pytest.approx(1.0)

    def test_unknown_string(self):
        # Unknown colour string falls through to black
        assert _parse_colour("not_a_colour") == (0.0, 0.0, 0.0, 1.0)


# =========================================================================
# renderer.py -- _apply_stroke edge cases
# =========================================================================


class TestRendererApplyStroke:
    """Test _apply_stroke edge cases."""

    def test_stroke_with_lty(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col="black", lty="dashed", lwd=2)
        rgba = r._apply_stroke(gp)
        assert rgba[3] > 0

    def test_stroke_with_alpha(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col="red", alpha=0.5)
        rgba = r._apply_stroke(gp)
        assert rgba[3] == pytest.approx(0.5)

    def test_stroke_with_lineend_butt(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col="black", lineend="butt")
        r._apply_stroke(gp)

    def test_stroke_with_linejoin_mitre(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col="black", linejoin="mitre")
        r._apply_stroke(gp)

    def test_stroke_lwd_zero(self):
        """lwd=0 means invisible line."""
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col="black", lwd=0)
        rgba = r._apply_stroke(gp)
        assert rgba[3] == 0.0

    def test_stroke_col_none(self):
        """col=None (unset) defaults to black, matching R's default."""
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col=None)
        rgba = r._apply_stroke(gp)
        # col=None means unset → default black (opaque)
        assert rgba == (0.0, 0.0, 0.0, 1.0)

    def test_stroke_col_na(self):
        """col='NA' means no border (transparent)."""
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(col="NA")
        rgba = r._apply_stroke(gp)
        assert rgba[3] == 0.0

    def test_stroke_none_gp(self):
        r = CairoRenderer(width=5, height=4)
        rgba = r._apply_stroke(None)
        assert rgba == (0.0, 0.0, 0.0, 1.0)

    def test_stroke_various_lty(self):
        """Test various known lty values."""
        r = CairoRenderer(width=5, height=4)
        for lty in ("solid", "dashed", "dotted", "dotdash", "longdash", "twodash"):
            gp = Gpar(col="black", lty=lty)
            r._apply_stroke(gp)

    def test_fill_rgba_none_gp(self):
        r = CairoRenderer(width=5, height=4)
        result = r._fill_rgba(None)
        assert result == (1.0, 1.0, 1.0, 1.0)

    def test_fill_rgba_none_fill(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fill=None)
        result = r._fill_rgba(gp)
        assert result[3] == 0.0

    def test_fill_rgba_with_alpha(self):
        r = CairoRenderer(width=5, height=4)
        gp = Gpar(fill="red", alpha=0.5)
        result = r._fill_rgba(gp)
        assert result[3] == pytest.approx(0.5)


# =========================================================================
# renderer.py -- text_extents
# =========================================================================


class TestRendererTextExtents:
    """Test text_extents method."""

    def test_text_extents_basic(self):
        r = CairoRenderer(width=5, height=4)
        ext = r.text_extents("Hello World")
        assert "width" in ext
        assert "ascent" in ext
        assert ext["width"] > 0

    def test_text_extents_with_gpar(self):
        r = CairoRenderer(width=5, height=4)
        ext = r.text_extents("Test", gp=Gpar(fontsize=24))
        assert ext["width"] > 0
        assert ext["ascent"] > 0


# =========================================================================
# _grab.py -- grid_grab
# =========================================================================


class TestGridGrab:
    """Test grid_grab and related functions."""

    def test_grab_empty_dl(self):
        """Grab from empty display list returns None."""
        result = grid_grab()
        assert result is None

    def test_grab_with_grobs(self):
        """Grab after drawing returns a GTree."""
        grid_newpage()
        grid_draw(rect_grob(name="r1"))
        result = grid_grab(warn=0)
        assert result is not None
        assert isinstance(result, GTree)

    def test_grab_with_wrap(self):
        """Wrap mode deep-copies children."""
        grid_newpage()
        grid_draw(rect_grob(name="r1"))
        result = grid_grab(warn=0, wrap=True)
        assert result is not None

    def test_grab_duplicate_names_warning(self):
        """Duplicate grob names should emit a warning at warn=1."""
        grid_newpage()
        g1 = rect_grob(name="dup")
        g2 = rect_grob(name="dup")
        grid_draw(g1)
        grid_draw(g2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_grab(warn=1)
            found = any("overwritten" in str(msg.message) for msg in w)
            assert found


# =========================================================================
# _grab.py -- grid_grab_expr
# =========================================================================


class TestGridGrabExpr:
    """Test grid_grab_expr."""

    def test_grab_expr_basic(self):
        def draw_some():
            grid_draw(rect_grob(name="expr_r1"))
            grid_draw(text_grob("hello", name="expr_t1"))

        grid_newpage()
        result = grid_grab_expr(draw_some, warn=0)
        assert result is not None
        assert isinstance(result, GTree)

    def test_grab_expr_restores_dl(self):
        """After grab_expr, the original display list is restored."""
        grid_newpage()
        grid_draw(rect_grob(name="original"))
        state = get_state()
        before_len = len(state.display_list)

        def draw_tmp():
            grid_draw(rect_grob(name="tmp"))

        grid_grab_expr(draw_tmp, warn=0)
        after_len = len(state.display_list)
        assert after_len == before_len

    def test_grab_expr_empty(self):
        """Grab from an expr that draws nothing returns None."""
        grid_newpage()
        result = grid_grab_expr(lambda: None, warn=0)
        assert result is None


# =========================================================================
# _grab.py -- grid_force / grid_revert
# =========================================================================


class TestGridForceRevert:
    """Test grid_force and grid_revert."""

    def test_force_grob_directly(self):
        g = rect_grob(name="fg1")
        forced = grid_force(g)
        assert forced is not None
        assert hasattr(forced, "_original")

    def test_force_gtree(self):
        child = rect_grob(name="child1")
        tree = GTree(children=GList(child), name="tree1")
        forced = grid_force(tree)
        assert isinstance(forced, GTree)

    def test_force_display_list(self):
        """Force the entire display list (x=None)."""
        grid_newpage()
        grid_draw(rect_grob(name="dl_g1"))
        result = grid_force(redraw=False)
        assert result is None

    def test_revert_grob_with_original(self):
        g = rect_grob(name="rev1")
        forced = force_grob(g)
        reverted = grid_revert(forced)
        assert reverted is g  # original is returned

    def test_revert_grob_without_original(self):
        g = rect_grob(name="nofrc")
        result = grid_revert(g)
        assert result is g  # unchanged

    def test_revert_gtree(self):
        child = rect_grob(name="c1")
        tree = GTree(children=GList(child), name="t1")
        forced = force_grob(tree)
        reverted = grid_revert(forced)
        assert isinstance(reverted, GTree)

    def test_revert_display_list(self):
        """Revert the entire display list (x=None)."""
        grid_newpage()
        grid_draw(rect_grob(name="rvdl1"))
        grid_force(redraw=False)
        result = grid_revert(redraw=False)
        assert result is None


# =========================================================================
# _grab.py -- grid_cap
# =========================================================================


class TestGridCap:
    """Test grid_cap for raster capture."""

    def test_cap_with_renderer(self):
        grid_newpage()
        grid_draw(rect_grob(gp=Gpar(fill="red")))
        arr = grid_cap(native=True)
        if arr is not None:
            assert arr.ndim == 3
            assert arr.shape[2] == 4  # RGBA

    def test_cap_no_renderer(self):
        """grid_cap without a renderer returns None with warning."""
        state = get_state()
        state.reset()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grid_cap()
            assert result is None

    def test_cap_float_output(self):
        grid_newpage()
        grid_draw(rect_grob(gp=Gpar(fill="blue")))
        arr = grid_cap(native=False)
        if arr is not None:
            assert arr.dtype == np.float64


# =========================================================================
# _grab.py -- grid_reorder
# =========================================================================


class TestGridReorder:
    """Test grid_reorder."""

    def test_reorder_basic(self):
        grid_newpage()
        c1 = rect_grob(name="ch1")
        c2 = rect_grob(name="ch2")
        c3 = rect_grob(name="ch3")
        tree = GTree(children=GList(c1, c2, c3), name="mytree")
        grid_draw(tree)
        grid_reorder("mytree", order=[2, 0], redraw=False)

    def test_reorder_by_name(self):
        grid_newpage()
        c1 = rect_grob(name="a")
        c2 = rect_grob(name="b")
        tree = GTree(children=GList(c1, c2), name="nametree")
        grid_draw(tree)
        grid_reorder("nametree", order=["b", "a"], redraw=False)

    def test_reorder_back_false(self):
        grid_newpage()
        c1 = rect_grob(name="x")
        c2 = rect_grob(name="y")
        tree = GTree(children=GList(c1, c2), name="backtree")
        grid_draw(tree)
        grid_reorder("backtree", order=["x"], back=False, redraw=False)

    def test_reorder_nonexistent_raises(self):
        grid_newpage()
        grid_draw(rect_grob(name="somegrob"))
        with pytest.raises(ValueError, match="does not match"):
            grid_reorder("nosuchtree", order=[0])

    def test_reorder_non_gtree_raises(self):
        grid_newpage()
        grid_draw(rect_grob(name="plain"))
        with pytest.raises(TypeError, match="not a gTree"):
            grid_reorder("plain", order=[0])

    def test_reorder_invalid_gpath_type(self):
        with pytest.raises(TypeError):
            grid_reorder(12345, order=[0])


# =========================================================================
# _highlevel.py -- grid_legend
# =========================================================================


class TestGridLegend:
    """Test grid_legend and legend_grob."""

    def test_legend_grob_basic(self):
        g = legend_grob(labels=["A", "B", "C"])
        assert isinstance(g, GTree)

    def test_legend_grob_with_pch(self):
        g = legend_grob(labels=["X", "Y"], pch=[1, 2], do_lines=True,
                        do_points=True)
        assert isinstance(g, GTree)

    def test_legend_grob_no_lines(self):
        g = legend_grob(labels=["A"], do_lines=False, do_points=True, pch=[19])
        assert isinstance(g, GTree)

    def test_legend_grob_no_points_or_pch(self):
        g = legend_grob(labels=["A"], do_lines=True, do_points=False, pch=None)
        assert isinstance(g, GTree)

    def test_legend_grob_neither_lines_nor_points(self):
        g = legend_grob(labels=["A"], do_lines=False, do_points=False, pch=None)
        assert isinstance(g, GTree)

    def test_legend_empty_labels(self):
        """Empty labels should return a null grob."""
        g = legend_grob(labels=[])
        assert g._grid_class == "null"

    def test_legend_grob_byrow(self):
        g = legend_grob(labels=["A", "B", "C", "D"], nrow=2, ncol=2, byrow=True)
        assert isinstance(g, GTree)

    def test_legend_grob_lines_first_false(self):
        g = legend_grob(labels=["A"], pch=[1], lines_first=False)
        assert isinstance(g, GTree)

    def test_legend_grob_with_gp(self):
        g = legend_grob(labels=["A", "B"], gp=Gpar(col=["red", "blue"]))
        assert isinstance(g, GTree)

    def test_grid_legend_draw(self):
        grid_newpage()
        g = grid_legend(labels=["A", "B"], draw=True)
        assert isinstance(g, GTree)


# =========================================================================
# _highlevel.py -- grid_plot_and_legend
# =========================================================================


class TestGridPlotAndLegend:
    """Test grid_plot_and_legend."""

    def test_plot_and_legend(self):
        grid_newpage()
        plot_g = rect_grob(gp=Gpar(fill="lightblue"))
        legend_g = legend_grob(labels=["A", "B"])
        grid_plot_and_legend(plot_expr=plot_g, legend_expr=legend_g)

    def test_plot_and_legend_none_parts(self):
        grid_plot_and_legend(plot_expr=None, legend_expr=None)


# =========================================================================
# _highlevel.py -- grid_grill
# =========================================================================


class TestGridGrill:
    """Test grid_grill."""

    def test_grid_grill_defaults(self):
        grid_newpage()
        grid_grill()

    def test_grid_grill_custom(self):
        grid_newpage()
        grid_grill(
            h=Unit([0.3, 0.6], "npc"),
            v=Unit([0.3, 0.6], "npc"),
            gp=Gpar(col="red"),
        )

    def test_grid_grill_with_vp(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.8, "npc"), height=Unit(0.8, "npc"))
        grid_grill(vp=vp)


# =========================================================================
# _highlevel.py -- grid_show_layout, layout_torture
# =========================================================================


class TestGridShowLayout:
    """Test grid_show_layout and layout_torture."""

    def test_show_layout(self):
        grid_newpage()
        lay = GridLayout(nrow=2, ncol=2)
        grid_show_layout(lay, newpage=True)

    def test_show_layout_no_newpage(self):
        grid_newpage()
        lay = GridLayout(nrow=1, ncol=2)
        grid_show_layout(lay, newpage=False)

    def test_layout_torture(self):
        layout_torture(n_row=2, n_col=2)


# =========================================================================
# _highlevel.py -- grid_show_viewport
# =========================================================================


class TestGridShowViewport:
    """Test grid_show_viewport."""

    def test_show_viewport_default(self):
        grid_newpage()
        grid_show_viewport()

    def test_show_viewport_with_vp(self):
        grid_newpage()
        v = Viewport(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_show_viewport(v=v, newpage=True)


# =========================================================================
# _highlevel.py -- grid_abline
# =========================================================================


class TestGridAbline:
    """Test grid_abline."""

    def test_abline_default(self):
        grid_newpage()
        g = grid_abline()
        assert g._grid_class == "lines"

    def test_abline_custom(self):
        grid_newpage()
        g = grid_abline(intercept=0.2, slope=0.5, gp=Gpar(col="red"), draw=True)
        assert g._grid_class == "lines"

    def test_abline_no_draw(self):
        g = grid_abline(draw=False)
        assert g._grid_class == "lines"


# =========================================================================
# _highlevel.py -- grid_strip, grid_panel
# =========================================================================


class TestGridStripPanel:
    """Test grid_strip and grid_panel."""

    def test_grid_strip(self):
        grid_newpage()
        grid_strip(label="test", range_full=(0, 1), range_thumb=(0.2, 0.6))

    def test_grid_strip_with_vp(self):
        grid_newpage()
        vp = Viewport()
        grid_strip(label="vp_test", vp=vp)

    def test_grid_panel_defaults(self):
        grid_newpage()
        np.random.seed(42)
        result = grid_panel()
        assert isinstance(result, dict)

    def test_grid_panel_custom(self):
        grid_newpage()
        result = grid_panel(
            x=[0.1, 0.3, 0.5, 0.7],
            y=[0.2, 0.4, 0.6, 0.8],
            xscale=[0, 1],
            yscale=[0, 1],
        )
        assert isinstance(result, dict)


# =========================================================================
# _highlevel.py -- axis constructors
# =========================================================================


class TestAxisConstructors:
    """Test xaxis_grob, yaxis_grob, grid_xaxis, grid_yaxis."""

    def test_xaxis_grob_with_at(self):
        g = xaxis_grob(at=[0.0, 0.5, 1.0])
        assert isinstance(g, GTree)

    def test_xaxis_grob_no_labels(self):
        g = xaxis_grob(at=[0.0, 0.5, 1.0], label=False)
        assert isinstance(g, GTree)

    def test_xaxis_grob_no_at(self):
        g = xaxis_grob()
        assert isinstance(g, GTree)

    def test_xaxis_not_main(self):
        g = xaxis_grob(at=[0.0, 1.0], main=False)
        assert isinstance(g, GTree)

    def test_yaxis_grob_with_at(self):
        g = yaxis_grob(at=[0.0, 0.5, 1.0])
        assert isinstance(g, GTree)

    def test_yaxis_grob_no_at(self):
        g = yaxis_grob()
        assert isinstance(g, GTree)

    def test_yaxis_not_main(self):
        g = yaxis_grob(at=[0.0, 1.0], main=False)
        assert isinstance(g, GTree)

    def test_yaxis_no_labels(self):
        g = yaxis_grob(at=[0.0, 1.0], label=False)
        assert isinstance(g, GTree)

    def test_grid_xaxis_draw(self):
        grid_newpage()
        g = grid_xaxis(at=[0.2, 0.5, 0.8], draw=True)
        assert isinstance(g, GTree)

    def test_grid_yaxis_draw(self):
        grid_newpage()
        g = grid_yaxis(at=[0.2, 0.5, 0.8], draw=True)
        assert isinstance(g, GTree)


# =========================================================================
# _draw.py -- _render_grob dispatches
# =========================================================================


class TestRenderGrobDispatch:
    """Test _render_grob dispatches for less common grob types."""

    def _make_renderer(self):
        return CairoRenderer(width=5, height=4)

    def test_render_circle(self):
        r = self._make_renderer()
        g = circle_grob(x=0.5, y=0.5, r=0.3)
        _render_grob(g, r, gp=Gpar(fill="red", col="black"))

    def test_render_pathgrob(self):
        r = self._make_renderer()
        g = path_grob(
            x=[0.1, 0.5, 0.9, 0.3, 0.7, 0.5],
            y=[0.1, 0.1, 0.5, 0.3, 0.3, 0.7],
            path_id=[1, 1, 1, 2, 2, 2],
            rule="evenodd",
        )
        _render_grob(g, r, gp=Gpar(fill="blue", col="black"))

    def test_render_pathgrob_no_pathid(self):
        """pathgrob without pathId should default to all-ones."""
        r = self._make_renderer()
        g = path_grob(x=[0.1, 0.5, 0.9], y=[0.1, 0.5, 0.9])
        # Manually remove pathId to test default
        if hasattr(g, 'pathId'):
            delattr(g, 'pathId')
        _render_grob(g, r, gp=Gpar(fill="green"))

    def test_render_rastergrob(self):
        r = self._make_renderer()
        img = np.random.randint(0, 255, (8, 8, 4), dtype=np.uint8)
        g = raster_grob(image=img, x=0.0, y=0.0, width=1.0, height=1.0)
        # The rastergrob stores image as 'raster' not 'image'
        # Set the attribute used by _render_grob
        g.image = img
        _render_grob(g, r, gp=Gpar())

    def test_render_roundrect(self):
        r = self._make_renderer()
        g = roundrect_grob(x=0.5, y=0.5, width=0.8, height=0.6, r=0.05)
        _render_grob(g, r, gp=Gpar(fill="yellow", col="black"))

    def test_render_move_to(self):
        r = self._make_renderer()
        g = move_to_grob(x=0.1, y=0.1)
        _render_grob(g, r)

    def test_render_line_to(self):
        r = self._make_renderer()
        r.move_to(0.1, 0.1)
        g = line_to_grob(x=0.9, y=0.9)
        _render_grob(g, r, gp=Gpar(col="black"))

    def test_render_null(self):
        r = self._make_renderer()
        g = null_grob()
        _render_grob(g, r)

    def test_render_unknown_class(self):
        r = self._make_renderer()
        g = Grob(_grid_class="xyzzy_unknown")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _render_grob(g, r)
            found = any("unknown grob class" in str(msg.message) for msg in w)
            assert found

    def test_render_none_renderer(self):
        """_render_grob with None renderer should be a no-op."""
        g = rect_grob()
        _render_grob(g, None)

    def test_render_polygon_with_id(self):
        r = self._make_renderer()
        g = polygon_grob(
            x=[0.1, 0.3, 0.2, 0.6, 0.8, 0.7],
            y=[0.1, 0.3, 0.5, 0.1, 0.3, 0.5],
            id=[1, 1, 1, 2, 2, 2],
        )
        _render_grob(g, r, gp=Gpar(fill=["red", "blue"], col="black"))

    def test_render_segments(self):
        r = self._make_renderer()
        g = segments_grob(x0=[0.1, 0.3], y0=[0.1, 0.3],
                          x1=[0.9, 0.7], y1=[0.9, 0.7])
        _render_grob(g, r, gp=Gpar(col="black"))

    def test_render_text(self):
        r = self._make_renderer()
        g = text_grob("hello", x=0.5, y=0.5)
        _render_grob(g, r, gp=Gpar(col="black"))

    def test_render_points(self):
        r = self._make_renderer()
        g = points_grob(x=[0.2, 0.5, 0.8], y=[0.2, 0.5, 0.8])
        _render_grob(g, r, gp=Gpar(col="black", fill="red"))


# =========================================================================
# _draw.py -- helper functions
# =========================================================================


class TestDrawHelpers:
    """Test _draw.py helper functions."""

    def test_resolve_just_string(self):
        g = Grob(just="left")
        h, v = _resolve_just(g)
        assert h == 0.0

    def test_resolve_just_tuple(self):
        g = Grob(just=["right", "top"])
        h, v = _resolve_just(g)
        assert h == 1.0 and v == 1.0

    def test_resolve_just_single_element_list(self):
        g = Grob(just=["centre"])
        h, v = _resolve_just(g)
        assert h == 0.5

    def test_resolve_just_numeric(self):
        g = Grob(just=[0.3, 0.7])
        h, v = _resolve_just(g)
        assert h == pytest.approx(0.3) and v == pytest.approx(0.7)

    def test_resolve_just_top(self):
        g = Grob(just="top")
        h, v = _resolve_just(g)
        assert v == 1.0

    def test_resolve_just_bottom(self):
        g = Grob(just="bottom")
        h, v = _resolve_just(g)
        assert v == 0.0

    def test_resolve_just_explicit_hjust_vjust(self):
        g = Grob(hjust=0.3, vjust=0.7)
        h, v = _resolve_just(g)
        assert h == pytest.approx(0.3) and v == pytest.approx(0.7)

    def test_subset_gpar_vectorised(self):
        gp = Gpar(col=np.array(["red", "blue", "green"]))
        g0 = _subset_gpar(gp, 0)
        assert g0 is not None

    def test_subset_gpar_none(self):
        assert _subset_gpar(None, 0) is None

    def test_merge_gpar(self):
        ctx = Gpar(col="red", lwd=2)
        grob_gp = Gpar(col="blue")
        merged = _merge_gpar(ctx, grob_gp)
        assert merged.get("col") == "blue"
        assert merged.get("lwd") == 2

    def test_merge_gpar_both_none(self):
        result = _merge_gpar(None, None)
        assert isinstance(result, Gpar)

    def test_unit_to_float_unit(self):
        u = Unit(0.5, "npc")
        assert _unit_to_float(u) == pytest.approx(0.5)

    def test_unit_to_float_plain(self):
        assert _unit_to_float(0.5) == pytest.approx(0.5)

    def test_unit_to_array_unit(self):
        u = Unit([0.1, 0.5, 0.9], "npc")
        arr = _unit_to_array(u)
        assert len(arr) == 3

    def test_unit_to_array_list(self):
        arr = _unit_to_array([0.1, 0.5, 0.9])
        assert len(arr) == 3

    def test_grid_pretty(self):
        result = grid_pretty([0, 10])
        assert len(result) > 0


# =========================================================================
# _draw.py -- grid_draw dispatch
# =========================================================================


class TestGridDrawDispatch:
    """Test grid_draw with different object types."""

    def test_draw_none(self):
        grid_draw(None)

    def test_draw_grob(self):
        grid_newpage()
        grid_draw(rect_grob())

    def test_draw_glist(self):
        grid_newpage()
        gl = GList(rect_grob(name="gl1"), rect_grob(name="gl2"))
        grid_draw(gl)

    def test_draw_viewport(self):
        grid_newpage()
        vp = Viewport(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_draw(vp)

    def test_draw_numeric(self):
        """Drawing a numeric is a no-op (R compatibility)."""
        grid_newpage()
        grid_draw(1)

    def test_draw_unknown_type(self):
        """Drawing an unknown type emits a warning."""
        grid_newpage()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_draw("not_a_grob")
            found = any("don't know how to draw" in str(msg.message) for msg in w)
            assert found

    def test_grid_newpage_creates_renderer(self):
        state = get_state()
        state.reset()
        grid_newpage()
        assert state.get_renderer() is not None

    def test_grid_newpage_existing_renderer(self):
        grid_newpage()
        grid_newpage()  # second call reuses renderer


# =========================================================================
# _primitives.py -- edge cases
# =========================================================================


class TestPrimitivesEdgeCases:
    """Test edge cases in grob constructors."""

    def test_circle_grob(self):
        g = circle_grob(x=0.3, y=0.7, r=0.2)
        assert g._grid_class == "circle"

    def test_raster_grob_rgba(self):
        img = np.zeros((5, 5, 4), dtype=np.uint8)
        g = raster_grob(image=img, width=0.5, height=0.5)
        assert g._grid_class == "rastergrob"

    def test_raster_grob_no_width_height(self):
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        g = raster_grob(image=img)
        assert g._grid_class == "rastergrob"

    def test_path_grob_with_pathid(self):
        g = path_grob(
            x=[0.1, 0.5, 0.9],
            y=[0.1, 0.5, 0.9],
            path_id=[1, 1, 1],
        )
        assert g._grid_class == "pathgrob"

    def test_path_grob_evenodd(self):
        g = path_grob(x=[0.1, 0.5, 0.9], y=[0.1, 0.5, 0.9], rule="evenodd")
        assert g._grid_class == "pathgrob"

    def test_path_grob_invalid_rule(self):
        with pytest.raises(ValueError, match="'rule' must be"):
            path_grob(x=[0.1], y=[0.1], rule="invalid")

    def test_path_grob_both_id_and_id_lengths(self):
        with pytest.raises(ValueError, match="invalid to specify both"):
            path_grob(x=[0.1], y=[0.1], id=[1], id_lengths=[1])

    def test_clip_grob(self):
        g = clip_grob(x=0.5, y=0.5, width=0.8, height=0.8)
        assert g._grid_class == "clip"

    def test_null_grob(self):
        g = null_grob()
        assert g._grid_class == "null"

    def test_function_grob(self):
        g = function_grob(fn=lambda: None)
        assert g._grid_class == "functiongrob"

    def test_polyline_both_id_error(self):
        with pytest.raises(ValueError, match="invalid to specify both"):
            polyline_grob(id=[1], id_lengths=[1])

    def test_polygon_both_id_error(self):
        with pytest.raises(ValueError, match="invalid to specify both"):
            polygon_grob(id=[1], id_lengths=[1])

    def test_roundrect_default_r(self):
        g = roundrect_grob()
        assert g._grid_class == "roundrect"

    def test_roundrect_numeric_r(self):
        g = roundrect_grob(r=0.05)
        assert g._grid_class == "roundrect"


# =========================================================================
# _edit.py -- grid_edit, grid_remove
# =========================================================================


class TestGridEdit:
    """Test grid_edit and grid_remove operations."""

    def test_grid_edit_basic(self):
        grid_newpage()
        g = rect_grob(name="editme", gp=Gpar(fill="red"))
        grid_draw(g)
        grid_edit("editme", gp=Gpar(fill="blue"), redraw=False)

    def test_grid_edit_with_kwargs(self):
        grid_newpage()
        g = rect_grob(name="editkw")
        grid_draw(g)
        grid_edit("editkw", gp=Gpar(col="green"), redraw=False)

    def test_grid_edit_nonexistent(self):
        """Editing a non-existent grob should just not match."""
        grid_newpage()
        grid_draw(rect_grob(name="exists"))
        # No error, just no matches
        grid_edit("nonexistent", gp=Gpar(col="red"), redraw=False)

    def test_grid_edit_all_devices_raises(self):
        with pytest.raises(NotImplementedError):
            grid_edit("x", allDevices=True, redraw=False)

    def test_grid_gedit(self):
        grid_newpage()
        g = rect_grob(name="gedit_target")
        grid_draw(g)
        grid_gedit("gedit_target", gp=Gpar(fill="yellow"), redraw=False)

    def test_grid_remove_top_level(self):
        grid_newpage()
        g = rect_grob(name="removeme")
        grid_draw(g)
        grid_remove("removeme", redraw=False)
        result = grid_get("removeme")
        assert result is None

    def test_grid_remove_with_grep(self):
        grid_newpage()
        grid_draw(rect_grob(name="rm_item_1"))
        grid_draw(rect_grob(name="rm_item_2"))
        grid_remove("rm_item_", grep=True, global_=True, redraw=False)

    def test_grid_remove_nonexistent_warns(self):
        grid_newpage()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_remove("nosuchgrob", warn=True, redraw=False)
            found = any("not found" in str(msg.message) for msg in w)
            assert found

    def test_grid_remove_from_gtree(self):
        """Remove a leaf from a parent gTree using multi-level path."""
        grid_newpage()
        child = rect_grob(name="leaf")
        tree = GTree(children=GList(child), name="parent")
        grid_draw(tree)
        grid_remove("parent::leaf", redraw=False)

    def test_grid_gremove(self):
        grid_newpage()
        grid_draw(rect_grob(name="gremove_target"))
        grid_gremove("gremove_target", redraw=False)


# =========================================================================
# _edit.py -- grid_get, grid_set, grid_add
# =========================================================================


class TestGridGetSetAdd:
    """Test grid_get, grid_set, grid_add."""

    def test_grid_get_found(self):
        grid_newpage()
        grid_draw(rect_grob(name="findme"))
        result = grid_get("findme")
        assert result is not None

    def test_grid_get_not_found(self):
        grid_newpage()
        result = grid_get("nope")
        assert result is None

    def test_grid_get_global(self):
        grid_newpage()
        grid_draw(rect_grob(name="multi1"))
        grid_draw(rect_grob(name="multi2"))
        result = grid_get("multi", grep=True, global_=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_grid_gget(self):
        grid_newpage()
        grid_draw(rect_grob(name="gget_target"))
        result = grid_gget("gget_target")
        assert isinstance(result, list)

    def test_grid_set(self):
        grid_newpage()
        g = rect_grob(name="setme")
        grid_draw(g)
        new_g = rect_grob(name="setme", gp=Gpar(fill="green"))
        grid_set("setme", newGrob=new_g, redraw=False)

    def test_grid_set_not_found(self):
        grid_newpage()
        with pytest.raises(ValueError, match="does not specify"):
            grid_set("nosuchgrob", newGrob=rect_grob(), redraw=False)

    def test_grid_add_top_level(self):
        grid_newpage()
        new_g = rect_grob(name="added")
        grid_add(new_g, redraw=False)

    def test_grid_add_to_gtree(self):
        grid_newpage()
        tree = GTree(children=GList(rect_grob(name="ch")), name="target_tree")
        grid_draw(tree)
        new_child = rect_grob(name="new_ch")
        grid_add(new_child, gPath="target_tree", redraw=False)

    def test_grid_add_to_non_gtree_warns(self):
        grid_newpage()
        grid_draw(rect_grob(name="plain_grob"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_add(rect_grob(name="x"), gPath="plain_grob", redraw=False)
            found = any("non-gTree" in str(msg.message) for msg in w)
            assert found

    def test_grid_add_all_devices_raises(self):
        with pytest.raises(NotImplementedError):
            grid_add(rect_grob(), allDevices=True, redraw=False)

    def test_grid_remove_all_devices_raises(self):
        with pytest.raises(NotImplementedError):
            grid_remove("x", allDevices=True, redraw=False)


# =========================================================================
# _grob.py -- force_grob, reorder_grob, GTree edge cases
# =========================================================================


class TestGrobEdgeCases:
    """Test grob helper functions and GTree edge cases."""

    def test_force_grob_simple(self):
        g = rect_grob(name="fg")
        result = force_grob(g)
        assert hasattr(result, "_original")

    def test_force_grob_gtree(self):
        child = rect_grob(name="c1")
        tree = GTree(children=GList(child), name="t1")
        result = force_grob(tree)
        assert isinstance(result, GTree)

    def test_reorder_grob_by_index(self):
        c1 = rect_grob(name="a")
        c2 = rect_grob(name="b")
        c3 = rect_grob(name="c")
        tree = GTree(children=GList(c1, c2, c3), name="t")
        result = reorder_grob(tree, order=[2, 0])
        assert result._children_order[0] == "c"
        assert result._children_order[1] == "a"

    def test_reorder_grob_by_name(self):
        c1 = rect_grob(name="alpha")
        c2 = rect_grob(name="beta")
        tree = GTree(children=GList(c1, c2), name="t")
        result = reorder_grob(tree, order=["beta", "alpha"])
        assert result._children_order[0] == "beta"

    def test_reorder_grob_back_false(self):
        c1 = rect_grob(name="a")
        c2 = rect_grob(name="b")
        tree = GTree(children=GList(c1, c2), name="t")
        result = reorder_grob(tree, order=["a"], back=False)
        assert result._children_order[-1] == "a"

    def test_reorder_grob_invalid_name(self):
        tree = GTree(children=GList(rect_grob(name="a")), name="t")
        with pytest.raises(ValueError, match="not found"):
            reorder_grob(tree, order=["nonexistent"])

    def test_reorder_grob_invalid_index(self):
        tree = GTree(children=GList(rect_grob(name="a")), name="t")
        with pytest.raises(ValueError, match="out of range"):
            reorder_grob(tree, order=[99])

    def test_reorder_grob_not_gtree(self):
        with pytest.raises(TypeError, match="GTree"):
            reorder_grob(rect_grob(), order=[0])

    def test_set_children(self):
        c1 = rect_grob(name="old")
        tree = GTree(children=GList(c1), name="t")
        new_children = GList(rect_grob(name="new"))
        result = set_children(tree, new_children)
        assert "new" in result._children_order

    def test_set_children_not_gtree(self):
        with pytest.raises(TypeError, match="GTree"):
            set_children(rect_grob(), GList())

    def test_grob_name_auto(self):
        name = grob_name()
        assert "GRID" in name

    def test_grob_name_from_grob(self):
        g = rect_grob()
        name = grob_name(g)
        assert "rect" in name

    def test_grob_name_invalid(self):
        with pytest.raises(TypeError):
            grob_name("not_a_grob")

    def test_grob_vp_string(self):
        """Passing a string as vp should wrap it as VpPath."""
        g = Grob(vp="some::path")
        assert g.vp is not None


# =========================================================================
# _draw.py -- grid_newpage edge cases
# =========================================================================


class TestGridNewpage:
    """Test grid_newpage edge cases."""

    def test_newpage_custom_dims(self):
        grid_newpage(width=10, height=8, dpi=72)
        state = get_state()
        r = state.get_renderer()
        assert r is not None

    def test_newpage_recording_false(self):
        grid_newpage(recording=False)

    def test_newpage_clear_dl_false(self):
        grid_newpage(clear_dl=False)


# =========================================================================
# _highlevel.py -- frame / pack / place
# =========================================================================


class TestFramePackPlace:
    """Test frame_grob, pack_grob, place_grob."""

    def test_frame_grob_default(self):
        f = frame_grob()
        assert f._grid_class == "frame"

    def test_frame_grob_with_layout(self):
        lay = GridLayout(nrow=2, ncol=2)
        f = frame_grob(layout=lay)
        assert f._grid_class == "frame"

    def test_pack_grob_side_left(self):
        f = frame_grob()
        g = rect_grob()
        result = pack_grob(f, g, side="left")
        assert isinstance(result, GTree)

    def test_pack_grob_side_right(self):
        f = frame_grob()
        g = rect_grob()
        result = pack_grob(f, g, side="right")
        assert isinstance(result, GTree)

    def test_pack_grob_side_top(self):
        f = frame_grob()
        g = rect_grob()
        result = pack_grob(f, g, side="top")
        assert isinstance(result, GTree)

    def test_pack_grob_side_bottom(self):
        f = frame_grob()
        g = rect_grob()
        result = pack_grob(f, g, side="bottom")
        assert isinstance(result, GTree)

    def test_pack_grob_invalid_frame(self):
        with pytest.raises(TypeError, match="invalid 'frame'"):
            pack_grob(rect_grob(), rect_grob())

    def test_pack_grob_invalid_grob(self):
        with pytest.raises(TypeError, match="invalid 'grob'"):
            pack_grob(frame_grob(), "not_a_grob")

    def test_place_grob_basic(self):
        """Place into a frame that already has a layout."""
        f = frame_grob()
        g1 = rect_grob()
        f = pack_grob(f, g1, side="left")
        f = pack_grob(f, rect_grob(), side="right")
        # Now place into existing cell
        g2 = text_grob("placed")
        result = place_grob(f, g2, row=1, col=1)
        assert isinstance(result, GTree)

    def test_place_grob_invalid_frame(self):
        with pytest.raises(TypeError, match="invalid 'frame'"):
            place_grob(rect_grob(), rect_grob())


# =========================================================================
# Renderer viewport management
# =========================================================================


class TestRendererViewportManagement:
    """Test push_viewport / pop_viewport on CairoRenderer."""

    def test_push_pop_simple_viewport(self):
        r = CairoRenderer(width=5, height=4)
        vp = Viewport(
            x=Unit(0.25, "npc"), y=Unit(0.25, "npc"),
            width=Unit(0.5, "npc"), height=Unit(0.5, "npc"),
        )
        r.push_viewport(vp)
        assert len(r._vp_stack) == 2
        r.pop_viewport()
        assert len(r._vp_stack) == 1

    def test_push_viewport_with_clip(self):
        r = CairoRenderer(width=5, height=4)
        vp = Viewport(
            width=Unit(0.5, "npc"), height=Unit(0.5, "npc"),
            clip=True,
        )
        r.push_viewport(vp)
        assert r._clip_stack[-1] is True
        r.pop_viewport()

    def test_pop_viewport_at_root(self):
        """Popping at root level should be a no-op."""
        r = CairoRenderer(width=5, height=4)
        r.pop_viewport()  # Should not crash
        assert len(r._vp_stack) == 1


# =========================================================================
# _draw.py -- grid_draw with full rendering pipeline
# =========================================================================


class TestFullRenderPipeline:
    """Test grid_draw with full rendering including renderer."""

    def test_draw_circle_grob_via_pipeline(self):
        grid_newpage()
        g = circle_grob(x=0.5, y=0.5, r=0.2, gp=Gpar(fill="red", col="black"))
        grid_draw(g)

    def test_draw_path_grob_via_pipeline(self):
        grid_newpage()
        g = path_grob(
            x=[0.1, 0.5, 0.9], y=[0.1, 0.9, 0.1],
            rule="winding",
        )
        grid_draw(g)

    def test_draw_raster_grob_via_pipeline(self):
        grid_newpage()
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        g = raster_grob(image=img, width=0.5, height=0.5)
        # Ensure the draw path can handle the raster attribute name
        g.image = img
        grid_draw(g)

    def test_draw_move_to_line_to_via_pipeline(self):
        grid_newpage()
        grid_draw(move_to_grob(x=0.1, y=0.1))
        grid_draw(line_to_grob(x=0.9, y=0.9, gp=Gpar(col="black")))

    def test_draw_roundrect_via_pipeline(self):
        grid_newpage()
        g = roundrect_grob(r=0.05, gp=Gpar(fill="yellow", col="black"))
        grid_draw(g)

    def test_draw_points_via_pipeline(self):
        grid_newpage()
        g = points_grob(x=[0.2, 0.5, 0.8], y=[0.3, 0.6, 0.9],
                        gp=Gpar(col="red"))
        grid_draw(g)

    def test_draw_multiple_rects(self):
        """Test vectorised rect drawing (multiple rects)."""
        grid_newpage()
        g = rect_grob(
            x=[0.25, 0.75], y=[0.25, 0.75],
            width=[0.3, 0.3], height=[0.3, 0.3],
            gp=Gpar(fill=["red", "blue"], col="black"),
        )
        grid_draw(g)

    def test_draw_gtree_with_children(self):
        grid_newpage()
        c1 = rect_grob(name="bg", gp=Gpar(fill="grey"))
        c2 = text_grob("Hello", name="label", gp=Gpar(col="black"))
        tree = GTree(children=GList(c1, c2), name="composite")
        grid_draw(tree)
