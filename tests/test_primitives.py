"""Tests for grid_py._primitives -- primitive grob constructors and grid_* wrappers."""

from __future__ import annotations

import sys
import os

import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grid_py"))

from grid_py._grob import Grob, _reset_auto_name, is_grob
from grid_py._gpar import Gpar
from grid_py._units import Unit, is_unit
from grid_py._primitives import (
    _display_list,
    rect_grob,
    grid_rect,
    circle_grob,
    grid_circle,
    lines_grob,
    grid_lines,
    polyline_grob,
    grid_polyline,
    segments_grob,
    grid_segments,
    points_grob,
    grid_points,
    text_grob,
    grid_text,
    polygon_grob,
    grid_polygon,
    path_grob,
    grid_path,
    raster_grob,
    grid_raster,
    null_grob,
    grid_null,
    move_to_grob,
    grid_move_to,
    line_to_grob,
    grid_line_to,
    clip_grob,
    grid_clip,
    roundrect_grob,
    grid_roundrect,
    function_grob,
    grid_function,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Reset auto-name counter and display list before every test."""
    _reset_auto_name()
    _display_list.clear()
    yield
    _display_list.clear()
    _reset_auto_name()


# ---------------------------------------------------------------------------
# rect_grob
# ---------------------------------------------------------------------------

class TestRectGrob:

    def test_construction_defaults(self):
        g = rect_grob()
        assert g._grid_class == "rect"
        assert is_grob(g)
        assert is_unit(g.x)
        assert is_unit(g.y)
        assert is_unit(g.width)
        assert is_unit(g.height)

    def test_custom_values(self):
        g = rect_grob(x=0.1, y=0.2, width=0.5, height=0.3, name="myrect")
        assert g.name == "myrect"

    def test_just_stored(self):
        g = rect_grob(just="left")
        assert g.just == "left"

    def test_gp_attached(self):
        gp = Gpar(col="red", lwd=2)
        g = rect_grob(gp=gp)
        assert g.gp is gp

    def test_numeric_args_coerced_to_unit(self):
        g = rect_grob(x=0.5, y=0.5, width=1, height=1)
        assert is_unit(g.x)
        assert is_unit(g.width)

    def test_unit_args_pass_through(self):
        u = Unit(0.3, "inches")
        g = rect_grob(x=u)
        assert g.x is u


# ---------------------------------------------------------------------------
# circle_grob
# ---------------------------------------------------------------------------

class TestCircleGrob:

    def test_construction(self):
        g = circle_grob()
        assert g._grid_class == "circle"
        assert is_unit(g.r)

    def test_custom_radius(self):
        g = circle_grob(r=0.25, name="c1")
        assert g.name == "c1"
        assert is_unit(g.r)

    def test_gp(self):
        gp = Gpar(fill="blue")
        g = circle_grob(gp=gp)
        assert g.gp is gp


# ---------------------------------------------------------------------------
# lines_grob / polyline_grob
# ---------------------------------------------------------------------------

class TestLinesGrob:

    def test_lines_defaults(self):
        g = lines_grob()
        assert g._grid_class == "lines"
        assert is_unit(g.x)
        assert is_unit(g.y)

    def test_lines_custom_coords(self):
        g = lines_grob(x=[0, 0.5, 1], y=[0, 1, 0])
        assert is_unit(g.x)

    def test_polyline_defaults(self):
        g = polyline_grob()
        assert g._grid_class == "polyline"

    def test_polyline_with_id(self):
        g = polyline_grob(
            x=[0, 1, 0, 1], y=[0, 1, 1, 0],
            id=[1, 1, 2, 2],
        )
        assert g.id == [1, 1, 2, 2]

    def test_polyline_id_and_id_lengths_raises(self):
        with pytest.raises(ValueError, match="both 'id' and 'id_lengths'"):
            polyline_grob(
                x=[0, 1, 0, 1], y=[0, 1, 1, 0],
                id=[1, 1, 2, 2], id_lengths=[2, 2],
            )


# ---------------------------------------------------------------------------
# segments_grob
# ---------------------------------------------------------------------------

class TestSegmentsGrob:

    def test_construction(self):
        g = segments_grob()
        assert g._grid_class == "segments"
        assert is_unit(g.x0)
        assert is_unit(g.y1)

    def test_custom_endpoints(self):
        g = segments_grob(x0=0, y0=0, x1=1, y1=1, name="seg")
        assert g.name == "seg"


# ---------------------------------------------------------------------------
# points_grob
# ---------------------------------------------------------------------------

class TestPointsGrob:

    def test_construction(self):
        g = points_grob()
        assert g._grid_class == "points"
        assert is_unit(g.x)
        assert is_unit(g.size)

    def test_custom_pch(self):
        g = points_grob(pch=16)
        assert g.pch == 16

    def test_named(self):
        g = points_grob(name="pts")
        assert g.name == "pts"


# ---------------------------------------------------------------------------
# text_grob
# ---------------------------------------------------------------------------

class TestTextGrob:

    def test_construction(self):
        g = text_grob("hello")
        assert g._grid_class == "text"
        assert g.label == "hello"
        assert is_unit(g.x)

    def test_rotation(self):
        g = text_grob("rotated", rot=45)
        assert g.rot == 45.0

    def test_check_overlap(self):
        g = text_grob("a", check_overlap=True)
        assert g.check_overlap is True


# ---------------------------------------------------------------------------
# polygon_grob
# ---------------------------------------------------------------------------

class TestPolygonGrob:

    def test_defaults(self):
        g = polygon_grob()
        assert g._grid_class == "polygon"
        assert is_unit(g.x)

    def test_with_id_lengths(self):
        g = polygon_grob(
            x=[0, 1, 0.5, 0, 1, 0.5],
            y=[0, 0, 1, 0, 0, 1],
            id_lengths=[3, 3],
        )
        assert g.id_lengths == [3, 3]

    def test_id_and_id_lengths_raises(self):
        with pytest.raises(ValueError, match="both 'id' and 'id_lengths'"):
            polygon_grob(x=[0, 1], y=[0, 1], id=[1, 1], id_lengths=[2])


# ---------------------------------------------------------------------------
# path_grob
# ---------------------------------------------------------------------------

class TestPathGrob:

    def test_construction(self):
        g = path_grob(x=[0, 1, 0.5], y=[0, 0, 1])
        assert g._grid_class == "pathgrob"
        assert g.rule == "winding"

    def test_evenodd_rule(self):
        g = path_grob(x=[0, 1, 0.5], y=[0, 0, 1], rule="evenodd")
        assert g.rule == "evenodd"

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError, match="'rule'"):
            path_grob(x=[0, 1], y=[0, 1], rule="invalid")


# ---------------------------------------------------------------------------
# raster_grob
# ---------------------------------------------------------------------------

class TestRasterGrob:

    def test_construction(self):
        image = [[1, 2], [3, 4]]
        g = raster_grob(image)
        assert g._grid_class == "rastergrob"
        assert g.raster is image
        assert g.interpolate is True

    def test_no_interpolate(self):
        g = raster_grob([[0]], interpolate=False)
        assert g.interpolate is False

    def test_optional_width_height(self):
        g = raster_grob([[0]])
        assert g.width is None
        assert g.height is None

    def test_explicit_width_height(self):
        g = raster_grob([[0]], width=0.8, height=0.6)
        assert is_unit(g.width)
        assert is_unit(g.height)


# ---------------------------------------------------------------------------
# null_grob
# ---------------------------------------------------------------------------

class TestNullGrob:

    def test_construction(self):
        g = null_grob()
        assert g._grid_class == "null"
        assert is_unit(g.x)

    def test_named(self):
        g = null_grob(name="placeholder")
        assert g.name == "placeholder"


# ---------------------------------------------------------------------------
# move_to_grob / line_to_grob
# ---------------------------------------------------------------------------

class TestMoveToLineTo:

    def test_move_to(self):
        g = move_to_grob(x=0.1, y=0.2)
        assert g._grid_class == "move.to"
        assert is_unit(g.x)

    def test_line_to(self):
        g = line_to_grob(x=0.9, y=0.8)
        assert g._grid_class == "line.to"
        assert is_unit(g.x)

    def test_line_to_with_gp(self):
        gp = Gpar(col="black")
        g = line_to_grob(gp=gp)
        assert g.gp is gp


# ---------------------------------------------------------------------------
# clip_grob
# ---------------------------------------------------------------------------

class TestClipGrob:

    def test_construction(self):
        g = clip_grob()
        assert g._grid_class == "clip"
        assert is_unit(g.width)

    def test_custom_region(self):
        g = clip_grob(x=0.25, y=0.25, width=0.5, height=0.5, name="clip1")
        assert g.name == "clip1"


# ---------------------------------------------------------------------------
# roundrect_grob
# ---------------------------------------------------------------------------

class TestRoundrectGrob:

    def test_construction(self):
        g = roundrect_grob()
        assert g._grid_class == "roundrect"
        assert is_unit(g.r)

    def test_custom_radius(self):
        g = roundrect_grob(r=Unit(5, "mm"))
        assert is_unit(g.r)

    def test_numeric_radius_coerced(self):
        g = roundrect_grob(r=0.2)
        assert is_unit(g.r)


# ---------------------------------------------------------------------------
# function_grob
# ---------------------------------------------------------------------------

class TestFunctionGrob:

    def test_construction(self):
        fn = lambda t: {"x": t, "y": t ** 2}
        g = function_grob(fn, n=50)
        assert g._grid_class == "functiongrob"
        assert g.f is fn
        assert g.n == 50

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="'n' must be >= 1"):
            function_grob(lambda t: t, n=0)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="'fn' must be callable"):
            function_grob("not_callable")


# ---------------------------------------------------------------------------
# Default units coercion (non-Unit args converted)
# ---------------------------------------------------------------------------

class TestDefaultUnitsCoercion:

    def test_rect_numeric_becomes_unit(self):
        g = rect_grob(x=0.3, y=0.4, width=0.6, height=0.7)
        for attr in ("x", "y", "width", "height"):
            assert is_unit(getattr(g, attr))

    def test_circle_numeric_becomes_unit(self):
        g = circle_grob(x=0, y=0, r=1)
        assert is_unit(g.x)
        assert is_unit(g.r)

    def test_segments_numeric_becomes_unit(self):
        g = segments_grob(x0=0, y0=0, x1=1, y1=1)
        for attr in ("x0", "y0", "x1", "y1"):
            assert is_unit(getattr(g, attr))

    def test_custom_default_units(self):
        g = rect_grob(x=2, default_units="cm")
        # The Unit should have been created with "cm"
        assert is_unit(g.x)

    def test_unit_passthrough_not_rewrapped(self):
        u = Unit(0.5, "npc")
        g = rect_grob(x=u)
        assert g.x is u


# ---------------------------------------------------------------------------
# grid_* functions with draw=False
# ---------------------------------------------------------------------------

class TestGridDrawFalse:

    def test_grid_rect_no_draw(self):
        g = grid_rect(draw=False)
        assert is_grob(g)
        assert g._grid_class == "rect"
        assert len(_display_list) == 0

    def test_grid_rect_draw(self):
        from grid_py._state import get_state
        from grid_py._draw import grid_newpage
        grid_newpage()
        g = grid_rect(draw=True)
        dl = get_state().get_display_list()
        assert len(dl) >= 1

    def test_grid_circle_no_draw(self):
        g = grid_circle(draw=False)
        assert len(_display_list) == 0
        assert g._grid_class == "circle"

    def test_grid_lines_no_draw(self):
        g = grid_lines(draw=False)
        assert len(_display_list) == 0

    def test_grid_polyline_no_draw(self):
        g = grid_polyline(draw=False)
        assert len(_display_list) == 0

    def test_grid_segments_no_draw(self):
        g = grid_segments(draw=False)
        assert len(_display_list) == 0

    def test_grid_points_no_draw(self):
        g = grid_points(draw=False)
        assert len(_display_list) == 0

    def test_grid_text_no_draw(self):
        g = grid_text("hello", draw=False)
        assert len(_display_list) == 0

    def test_grid_polygon_no_draw(self):
        g = grid_polygon(draw=False)
        assert len(_display_list) == 0

    def test_grid_path_no_draw(self):
        g = grid_path(draw=False)
        assert len(_display_list) == 0

    def test_grid_raster_no_draw(self):
        g = grid_raster([[0]], draw=False)
        assert len(_display_list) == 0

    def test_grid_null_no_draw(self):
        g = grid_null(draw=False)
        assert len(_display_list) == 0

    def test_grid_move_to_no_draw(self):
        g = grid_move_to(draw=False)
        assert len(_display_list) == 0

    def test_grid_line_to_no_draw(self):
        g = grid_line_to(draw=False)
        assert len(_display_list) == 0

    def test_grid_clip_no_draw(self):
        g = grid_clip(draw=False)
        assert len(_display_list) == 0

    def test_grid_roundrect_no_draw(self):
        g = grid_roundrect(draw=False)
        assert len(_display_list) == 0

    def test_grid_function_no_draw(self):
        g = grid_function(lambda t: t, draw=False)
        assert len(_display_list) == 0

    def test_grid_function_draw(self):
        from grid_py._state import get_state
        from grid_py._draw import grid_newpage
        grid_newpage()
        g = grid_function(lambda t: t, draw=True)
        dl = get_state().get_display_list()
        assert len(dl) >= 1
