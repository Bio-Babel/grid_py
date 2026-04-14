"""Coverage tests for grid_py._primitives -- missing primitive grob constructors.
"""
from __future__ import annotations

import pytest
import numpy as np

import grid_py
from grid_py._primitives import (
    move_to_grob,
    grid_move_to,
    line_to_grob,
    grid_line_to,
    lines_grob,
    grid_lines,
    polyline_grob,
    grid_polyline,
    segments_grob,
    grid_segments,
    arrows_grob,
    grid_arrows,
    points_grob,
    grid_points,
    rect_grob,
    grid_rect,
    roundrect_grob,
    grid_roundrect,
    circle_grob,
    grid_circle,
    polygon_grob,
    grid_polygon,
    path_grob,
    grid_path,
    text_grob,
    grid_text,
    raster_grob,
    grid_raster,
    clip_grob,
    grid_clip,
    null_grob,
    grid_null,
    function_grob,
    grid_function,
)
from grid_py._units import Unit
from grid_py._gpar import Gpar
from grid_py._grob import Grob


# ---------------------------------------------------------------------------
# move_to / line_to
# ---------------------------------------------------------------------------


class TestMoveToLineTo:
    def test_move_to_grob(self):
        g = move_to_grob(0.5, 0.5)
        assert g._grid_class == "move.to"

    def test_grid_move_to_no_draw(self):
        g = grid_move_to(0.5, 0.5, draw=False)
        assert g._grid_class == "move.to"

    def test_grid_move_to_draw(self):
        g = grid_move_to(0.5, 0.5, draw=True)
        assert g._grid_class == "move.to"

    def test_line_to_grob(self):
        g = line_to_grob(1, 1)
        assert g._grid_class == "line.to"

    def test_grid_line_to_no_draw(self):
        g = grid_line_to(1, 1, draw=False)
        assert g._grid_class == "line.to"

    def test_grid_line_to_draw(self):
        g = grid_line_to(1, 1, draw=True)
        assert g._grid_class == "line.to"

    def test_move_to_with_unit(self):
        g = move_to_grob(Unit(0.5, "npc"), Unit(0.5, "npc"))
        assert g._grid_class == "move.to"

    def test_line_to_with_gp(self):
        g = line_to_grob(1, 1, gp=Gpar(col="red"))
        assert g.gp is not None

    def test_line_to_with_name(self):
        g = line_to_grob(1, 1, name="myline")
        assert g.name == "myline"


# ---------------------------------------------------------------------------
# lines / polyline
# ---------------------------------------------------------------------------


class TestLinesPolyline:
    def test_lines_grob_defaults(self):
        g = lines_grob()
        assert g._grid_class == "lines"

    def test_lines_grob_with_data(self):
        g = lines_grob(x=[0, 0.5, 1], y=[0, 1, 0])
        assert g._grid_class == "lines"

    def test_grid_lines_no_draw(self):
        g = grid_lines(draw=False)
        assert g._grid_class == "lines"

    def test_polyline_grob_defaults(self):
        g = polyline_grob()
        assert g._grid_class == "polyline"

    def test_polyline_grob_with_id(self):
        g = polyline_grob(x=[0, 1, 0, 1], y=[0, 1, 0, 1], id=[1, 1, 2, 2])
        assert g._grid_class == "polyline"

    def test_polyline_grob_with_id_lengths(self):
        g = polyline_grob(x=[0, 1, 0, 1], y=[0, 1, 0, 1], id_lengths=[2, 2])
        assert g._grid_class == "polyline"

    def test_polyline_both_id_raises(self):
        with pytest.raises(ValueError, match="both 'id' and 'id_lengths'"):
            polyline_grob(id=[1], id_lengths=[1])

    def test_grid_polyline_no_draw(self):
        g = grid_polyline(draw=False)
        assert g._grid_class == "polyline"


# ---------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------


class TestSegments:
    def test_segments_grob(self):
        g = segments_grob(x0=0, y0=0, x1=1, y1=1)
        assert g._grid_class == "segments"

    def test_grid_segments_no_draw(self):
        g = grid_segments(draw=False)
        assert g._grid_class == "segments"


# ---------------------------------------------------------------------------
# arrows (defunct)
# ---------------------------------------------------------------------------


class TestArrows:
    def test_arrows_grob_defunct(self):
        """R primitives.R:539: arrowsGrob calls .Defunct() — must raise."""
        with pytest.raises(NotImplementedError, match="defunct"):
            arrows_grob()

    def test_grid_arrows_defunct(self):
        """R primitives.R:543: grid.arrows calls .Defunct() — must raise."""
        with pytest.raises(NotImplementedError, match="defunct"):
            grid_arrows()


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------


class TestPoints:
    def test_points_grob(self):
        g = points_grob(x=[0.5], y=[0.5])
        assert g._grid_class == "points"

    def test_grid_points_no_draw(self):
        g = grid_points(draw=False)
        assert g._grid_class == "points"

    def test_points_with_pch(self):
        g = points_grob(x=[0.5], y=[0.5], pch=3)
        assert g.pch == 3


# ---------------------------------------------------------------------------
# rect / roundrect / circle
# ---------------------------------------------------------------------------


class TestRectRoundrectCircle:
    def test_rect_grob(self):
        g = rect_grob()
        assert g._grid_class == "rect"

    def test_grid_rect_no_draw(self):
        g = grid_rect(draw=False)
        assert g._grid_class == "rect"

    def test_rect_with_params(self):
        g = rect_grob(x=0.3, y=0.7, width=0.4, height=0.2,
                       name="myrect", gp=Gpar(col="blue"))
        assert g.name == "myrect"

    def test_roundrect_grob(self):
        g = roundrect_grob()
        assert g._grid_class == "roundrect"

    def test_grid_roundrect_no_draw(self):
        g = grid_roundrect(draw=False)
        assert g._grid_class == "roundrect"

    def test_circle_grob(self):
        g = circle_grob()
        assert g._grid_class == "circle"

    def test_grid_circle_no_draw(self):
        g = grid_circle(draw=False)
        assert g._grid_class == "circle"

    def test_circle_with_params(self):
        g = circle_grob(x=0.5, y=0.5, r=0.3)
        assert g._grid_class == "circle"


# ---------------------------------------------------------------------------
# polygon / path
# ---------------------------------------------------------------------------


class TestPolygonPath:
    def test_polygon_grob(self):
        g = polygon_grob(x=[0, 1, 0.5], y=[0, 0, 1])
        assert g._grid_class == "polygon"

    def test_grid_polygon_no_draw(self):
        g = grid_polygon(draw=False)
        assert g._grid_class == "polygon"

    def test_path_grob(self):
        g = path_grob(x=[0, 1, 0.5, 0], y=[0, 0, 1, 0])
        assert g._grid_class == "pathgrob"

    def test_grid_path_no_draw(self):
        g = grid_path(draw=False)
        assert g._grid_class == "pathgrob"


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


class TestText:
    def test_text_grob(self):
        g = text_grob(label="hello")
        assert g._grid_class == "text"
        assert g.label == "hello"

    def test_grid_text_no_draw(self):
        g = grid_text(label="hi", draw=False)
        assert g._grid_class == "text"

    def test_text_with_rotation(self):
        g = text_grob(label="rotated", rot=45)
        assert g.rot == 45


# ---------------------------------------------------------------------------
# raster
# ---------------------------------------------------------------------------


class TestRaster:
    def test_raster_grob(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        g = raster_grob(image=img)
        assert g._grid_class == "rastergrob"

    def test_grid_raster_no_draw(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        g = grid_raster(image=img, draw=False)
        assert g._grid_class == "rastergrob"


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------


class TestClip:
    def test_clip_grob(self):
        g = clip_grob()
        assert g is not None

    def test_grid_clip_no_draw(self):
        g = grid_clip(draw=False)
        assert g is not None


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------


class TestNull:
    def test_null_grob(self):
        g = null_grob()
        assert g._grid_class == "null"

    def test_grid_null_no_draw(self):
        g = grid_null(draw=False)
        assert g._grid_class == "null"


# ---------------------------------------------------------------------------
# function
# ---------------------------------------------------------------------------


class TestFunction:
    def test_function_grob(self):
        g = function_grob(fn=lambda: None)
        assert g is not None

    def test_grid_function_no_draw(self):
        g = grid_function(fn=lambda: None, draw=False)
        assert g is not None
