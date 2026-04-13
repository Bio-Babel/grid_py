"""Tests for fill/stroke/fillStroke path grobs (R 4.2+, grid/R/path.R)."""

import os
import tempfile

import pytest

from grid_py._gpar import Gpar
from grid_py._grob import Grob
from grid_py._primitives import (
    as_path,
    circle_grob,
    fill_grob,
    fill_stroke_grob,
    rect_grob,
    stroke_grob,
)
from grid_py._draw import grid_draw, grid_newpage
from grid_py._units import Unit


class TestGrobConstruction:
    """Factory functions create correct grob types."""

    def test_stroke_grob_from_grob(self):
        g = circle_grob()
        sg = stroke_grob(g, name="s1")
        assert sg._grid_class == "GridStroke"
        assert getattr(sg, "path", None) is g

    def test_fill_grob_from_grob(self):
        g = rect_grob()
        fg = fill_grob(g, rule="evenodd", name="f1")
        assert fg._grid_class == "GridFill"
        assert getattr(fg, "rule", None) == "evenodd"

    def test_fill_stroke_grob_from_grob(self):
        g = circle_grob()
        fsg = fill_stroke_grob(g, name="fs1")
        assert fsg._grid_class == "GridFillStroke"

    def test_as_path(self):
        g = rect_grob()
        p = as_path(g, rule="evenodd")
        assert p["_class"] == "GridPath"
        assert p["rule"] == "evenodd"

    def test_stroke_grob_from_path(self):
        g = circle_grob()
        p = as_path(g, gp=Gpar(col="red"))
        sg = stroke_grob(p, name="s2")
        assert sg._grid_class == "GridStroke"
        assert getattr(sg, "path", None) is g

    def test_fill_grob_from_path(self):
        g = rect_grob()
        p = as_path(g, rule="evenodd")
        fg = fill_grob(p, name="f2")
        assert fg._grid_class == "GridFill"
        assert getattr(fg, "rule", None) == "evenodd"

    def test_fill_stroke_grob_from_path(self):
        g = rect_grob()
        p = as_path(g, rule="evenodd")
        fsg = fill_stroke_grob(p, name="fs2")
        assert fsg._grid_class == "GridFillStroke"
        assert getattr(fsg, "rule", None) == "evenodd"


class TestRendering:
    """Rendering fill/stroke grobs produces valid output."""

    def test_stroke_renders(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        g = circle_grob(x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                         r=Unit(0.3, "npc"))
        sg = stroke_grob(g, gp=Gpar(col="red", lwd=2))
        grid_draw(sg)  # should not raise

    def test_fill_renders(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        g = rect_grob(width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        fg = fill_grob(g, gp=Gpar(fill="blue"))
        grid_draw(fg)

    def test_fill_stroke_renders(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        g = circle_grob(r=Unit(0.2, "npc"))
        fsg = fill_stroke_grob(g, gp=Gpar(fill="green", col="black"))
        grid_draw(fsg)

    def test_stroke_text_outline(self):
        """stroke_grob(text_grob(...)) renders text as outline path."""
        from grid_py._primitives import text_grob as _text_grob
        grid_newpage(width=3.0, height=3.0, dpi=72)
        tg = _text_grob("ABC", x=Unit(0.5, "npc"), y=Unit(0.5, "npc"))
        sg = stroke_grob(tg, gp=Gpar(col="black", lwd=2))
        grid_draw(sg)  # should not raise

    def test_fill_text_outline(self):
        """fill_grob(text_grob(...)) fills text outlines."""
        from grid_py._primitives import text_grob as _text_grob
        grid_newpage(width=3.0, height=3.0, dpi=72)
        tg = _text_grob("XYZ", x=Unit(0.5, "npc"), y=Unit(0.5, "npc"))
        fg = fill_grob(tg, gp=Gpar(fill="red"))
        grid_draw(fg)

    def test_fill_evenodd_rule(self):
        grid_newpage(width=3.0, height=3.0, dpi=72)
        g = rect_grob()
        fg = fill_grob(g, rule="evenodd", gp=Gpar(fill="grey"))
        grid_draw(fg)

    def test_output_not_empty(self):
        from grid_py._state import get_state
        grid_newpage(width=3.0, height=3.0, dpi=72)
        g = circle_grob(r=Unit(0.3, "npc"))
        fsg = fill_stroke_grob(g, gp=Gpar(fill="red", col="black", lwd=3))
        grid_draw(fsg)
        renderer = get_state()._renderer
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            renderer.write_to_png(f.name)
            size = os.path.getsize(f.name)
            assert size > 100
            os.unlink(f.name)
