"""Additional coverage tests for grid_py._draw -- render paths, edge cases.
"""
from __future__ import annotations

import warnings
import pytest
import numpy as np

import grid_py
from grid_py._draw import (
    _unit_to_float,
    _unit_to_array,
    _render_grob,
    _merge_gpar,
    _vp_depth,
    grid_draw,
    grid_newpage,
    grid_refresh,
    grid_record,
    record_grob,
    grid_delay,
    delay_grob,
    grid_dl_apply,
    grid_locator,
    grid_pretty,
)
from grid_py._grob import Grob, GTree, GList
from grid_py._gpar import Gpar
from grid_py._units import Unit
from grid_py._state import get_state
from grid_py._display_list import DLDrawGrob


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_unit_to_float_unit(self):
        assert _unit_to_float(Unit(2.5, "cm")) == 2.5

    def test_unit_to_float_scalar(self):
        assert _unit_to_float(3.14) == 3.14

    def test_unit_to_array_unit(self):
        arr = _unit_to_array(Unit([1, 2, 3], "cm"))
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_unit_to_array_list(self):
        arr = _unit_to_array([1, 2, 3])
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_unit_to_array_scalar(self):
        arr = _unit_to_array(5)
        assert arr[0] == 5.0

    def test_unit_to_array_tuple(self):
        arr = _unit_to_array((1, 2))
        np.testing.assert_array_equal(arr, [1, 2])

    def test_unit_to_array_list_of_units(self):
        # This triggers the ValueError/TypeError fallback
        arr = _unit_to_array([Unit(1, "cm"), Unit(2, "cm")])
        np.testing.assert_array_equal(arr, [1, 2])

    def test_merge_gpar_both_none(self):
        result = _merge_gpar(None, None)
        assert isinstance(result, Gpar)

    def test_merge_gpar_context_none(self):
        gp = Gpar(col="red")
        result = _merge_gpar(None, gp)
        assert result is gp

    def test_merge_gpar_grob_none(self):
        gp = Gpar(col="blue")
        result = _merge_gpar(gp, None)
        assert result is gp

    def test_merge_gpar_both(self):
        context = Gpar(col="red", lwd=2)
        grob_gp = Gpar(col="blue")
        result = _merge_gpar(context, grob_gp)
        assert result.get("col") == "blue"

    def test_vp_depth_vppath(self):
        from grid_py._path import VpPath
        p = VpPath("a", "b")
        assert _vp_depth(p) == 2

    def test_vp_depth_default(self):
        assert _vp_depth(object()) == 1

    def test_vp_depth_with_depth_method(self):
        class MockVp:
            def depth(self):
                return 3
        assert _vp_depth(MockVp()) == 3


# ---------------------------------------------------------------------------
# _render_grob
# ---------------------------------------------------------------------------


class TestRenderGrob:
    def test_render_none_renderer(self):
        g = Grob(name="test", _grid_class="rect")
        _render_grob(g, None)  # Should not crash

    def test_render_unknown_class_warns(self):
        g = Grob(name="test", _grid_class="totally_unknown_class")

        class MockRenderer:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _render_grob(g, MockRenderer())
            assert any("unknown grob class" in str(x.message) for x in w)

    def test_render_null_noop(self):
        g = Grob(name="test", _grid_class="null")

        class MockRenderer:
            pass

        _render_grob(g, MockRenderer())  # Should not crash


# ---------------------------------------------------------------------------
# grid_draw
# ---------------------------------------------------------------------------


class TestGridDraw:
    def test_draw_none(self):
        grid_draw(None)  # Should not crash

    def test_draw_grob(self):
        grid_newpage()
        g = Grob(name="test", _grid_class="rect", x=Unit(0.5, "npc"),
                 y=Unit(0.5, "npc"), width=Unit(0.5, "npc"),
                 height=Unit(0.5, "npc"))
        grid_draw(g)

    def test_draw_gtree(self):
        grid_newpage()
        child = Grob(name="c", _grid_class="rect",
                     x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                     width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        tree = GTree(children=GList(child), name="tree")
        grid_draw(tree)

    def test_draw_glist(self):
        grid_newpage()
        g = Grob(name="g", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        gl = GList(g)
        grid_draw(gl)

    def test_draw_numeric(self):
        grid_draw(42)  # numeric dispatch should silently pass

    def test_draw_unknown_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid_draw("unknown_type")
            assert any("don't know how to draw" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# grid_newpage
# ---------------------------------------------------------------------------


class TestGridNewpage:
    def test_newpage_creates_renderer(self):
        state = get_state()
        state.reset()
        grid_newpage()
        assert state.get_renderer() is not None

    def test_newpage_reuses_renderer(self):
        grid_newpage()
        r1 = get_state().get_renderer()
        grid_newpage()
        # After second newpage, renderer may be different or same
        r2 = get_state().get_renderer()
        assert r2 is not None

    def test_newpage_custom_size(self):
        grid_newpage(width=10, height=8, dpi=72)
        assert get_state().get_renderer() is not None


# ---------------------------------------------------------------------------
# grid_refresh
# ---------------------------------------------------------------------------


class TestGridRefresh:
    def test_refresh_empty(self):
        grid_newpage()
        grid_refresh()  # Should not crash

    def test_refresh_with_grobs(self):
        grid_newpage()
        g = Grob(name="test", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_draw(g)
        grid_refresh()


# ---------------------------------------------------------------------------
# record_grob / grid_record
# ---------------------------------------------------------------------------


class TestRecordGrob:
    def test_record_grob(self):
        g = record_grob(lambda: None)
        assert g._grid_class == "recordedGrob"

    def test_record_grob_with_env(self):
        g = record_grob(lambda x=1: None, list_={"x": 1})
        assert g._grid_class == "recordedGrob"

    def test_grid_record(self):
        grid_newpage()
        grid_record(lambda: None)


# ---------------------------------------------------------------------------
# delay_grob / grid_delay
# ---------------------------------------------------------------------------


class TestDelayGrob:
    def test_delay_grob(self):
        def make_rect():
            return Grob(name="delayed_rect", _grid_class="rect")
        g = delay_grob(make_rect)
        assert g._grid_class == "delayedgrob"

    def test_delay_grob_make_content(self):
        def make_rect():
            return Grob(name="delayed_rect", _grid_class="rect",
                       x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                       width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        g = delay_grob(make_rect)
        g = g.make_content()
        assert g.n_children() == 1

    def test_delay_grob_make_content_glist(self):
        def make_list():
            return GList(Grob(name="a", _grid_class="rect"))
        g = delay_grob(make_list)
        g = g.make_content()
        assert g.n_children() == 1

    def test_delay_grob_make_content_invalid_raises(self):
        def bad():
            return "not_a_grob"
        g = delay_grob(bad)
        with pytest.raises(TypeError, match="must return"):
            g.make_content()

    def test_grid_delay(self):
        grid_newpage()
        grid_delay(lambda: Grob(name="d", _grid_class="rect",
                                x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                                width=Unit(0.5, "npc"), height=Unit(0.5, "npc")))


# ---------------------------------------------------------------------------
# grid_dl_apply
# ---------------------------------------------------------------------------


class TestGridDlApply:
    def test_dl_apply_identity(self):
        grid_newpage()
        g = Grob(name="test", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_draw(g)
        grid_dl_apply(lambda item: item)

    def test_dl_apply_type_mismatch_raises(self):
        grid_newpage()
        g = Grob(name="test", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        grid_draw(g)
        with pytest.raises(TypeError, match="invalid modification"):
            grid_dl_apply(lambda item: "wrong_type")


# ---------------------------------------------------------------------------
# grid_locator
# ---------------------------------------------------------------------------


class TestGridLocator:
    def test_locator_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = grid_locator()
            assert result is None
            assert any("not supported" in str(x.message) for x in w)


# ---------------------------------------------------------------------------
# grid_pretty
# ---------------------------------------------------------------------------


class TestGridPretty:
    def test_basic_range(self):
        result = grid_pretty([0, 10])
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert result[0] >= 0
        assert result[-1] <= 10


# ---------------------------------------------------------------------------
# Rendering dispatch for various grob classes
# ---------------------------------------------------------------------------


class TestRenderDispatch:
    """Test _render_grob with a mock renderer for each grob class."""

    class CapturingRenderer:
        def __init__(self):
            self.calls = []
            self.dpi = 100.0
            self._vp_stack = [(0.0, 0.0, 300.0, 200.0, None)]

        def __getattr__(self, name):
            # Resolve methods must return proper values
            if name in ("resolve_x", "resolve_y", "resolve_w", "resolve_h"):
                def resolve_scalar(val, gp=None):
                    from grid_py._units import Unit
                    if isinstance(val, Unit):
                        return float(val._values[0])
                    return float(val)
                return resolve_scalar
            if name.startswith("resolve_") and name.endswith("_array"):
                def resolve_array(val, gp=None):
                    import numpy as _np
                    from grid_py._units import Unit
                    if isinstance(val, Unit):
                        return _np.asarray(val._values, dtype=float)
                    return _np.atleast_1d(_np.asarray(val, dtype=float))
                return resolve_array
            def method(*args, **kwargs):
                self.calls.append(name)
            return method

    def test_render_rect(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        _render_grob(g, r, gp=Gpar())
        assert "draw_rect" in r.calls

    def test_render_roundrect(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="roundrect",
                 x=0.5, y=0.5, width=0.5, height=0.5, r=0.1)
        _render_grob(g, r, gp=Gpar())
        assert "draw_roundrect" in r.calls

    def test_render_circle(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="circle", x=0.5, y=0.5, r=0.3)
        _render_grob(g, r, gp=Gpar())
        assert "draw_circle" in r.calls

    def test_render_lines(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="lines",
                 x=[0, 1], y=[0, 1])
        _render_grob(g, r, gp=Gpar())
        assert "draw_polyline" in r.calls

    def test_render_polyline_with_id(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="polyline",
                 x=[0, 1], y=[0, 1], id=[1, 1])
        _render_grob(g, r, gp=Gpar())
        assert "draw_polyline" in r.calls

    def test_render_segments(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="segments",
                 x0=[0], y0=[0], x1=[1], y1=[1])
        _render_grob(g, r, gp=Gpar())
        assert "draw_segments" in r.calls

    def test_render_polygon(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="polygon",
                 x=[0, 1, 0.5], y=[0, 0, 1])
        _render_grob(g, r, gp=Gpar())
        assert "draw_polygon" in r.calls

    def test_render_text(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="text",
                 x=0.5, y=0.5, label="hello")
        _render_grob(g, r, gp=Gpar())
        assert "draw_text" in r.calls

    def test_render_points(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="points",
                 x=[0.5], y=[0.5], pch=19, size=1.0)
        _render_grob(g, r, gp=Gpar())
        assert "draw_points" in r.calls

    def test_render_points_list_pch(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="points",
                 x=[0.5], y=[0.5], pch=[19], size=1.0)
        _render_grob(g, r, gp=Gpar())
        assert "draw_points" in r.calls

    def test_render_pathgrob(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="pathgrob",
                 x=[0, 1, 0.5, 0], y=[0, 0, 1, 0])
        _render_grob(g, r, gp=Gpar())
        assert "draw_path" in r.calls

    def test_render_pathgrob_with_pathid(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="pathgrob",
                 x=[0, 1], y=[0, 1], pathId=[1, 1])
        _render_grob(g, r, gp=Gpar())
        assert "draw_path" in r.calls

    def test_render_rastergrob(self):
        r = self.CapturingRenderer()
        img = np.zeros((5, 5, 4), dtype=np.uint8)
        g = Grob(name="t", _grid_class="rastergrob",
                 image=img, x=0, y=0, width=1, height=1)
        _render_grob(g, r, gp=Gpar())
        assert "draw_raster" in r.calls

    def test_render_rastergrob_no_image(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="rastergrob")
        _render_grob(g, r, gp=Gpar())
        assert "draw_raster" not in r.calls

    def test_render_move_to(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="move.to", x=0.5, y=0.5)
        _render_grob(g, r, gp=Gpar())
        assert "move_to" in r.calls

    def test_render_line_to(self):
        r = self.CapturingRenderer()
        g = Grob(name="t", _grid_class="line.to", x=0.5, y=0.5)
        _render_grob(g, r, gp=Gpar())
        assert "line_to" in r.calls


# ---------------------------------------------------------------------------
# Drawing with viewports
# ---------------------------------------------------------------------------


class TestDrawViewport:
    def test_draw_viewport(self):
        from grid_py._viewport import Viewport
        grid_newpage()
        vp = Viewport(name="drawvp")
        grid_draw(vp)

    def test_draw_vppath(self):
        from grid_py._viewport import Viewport, push_viewport
        from grid_py._path import VpPath
        grid_newpage()
        push_viewport(Viewport(name="target_vp"))
        # Navigate up first
        from grid_py._viewport import up_viewport
        up_viewport(1)
        grid_draw(VpPath("target_vp"))

    def test_draw_grob_with_vp(self):
        from grid_py._viewport import Viewport
        grid_newpage()
        vp = Viewport(name="grob_vp")
        g = Grob(name="test", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"),
                 vp=vp)
        grid_draw(g)

    def test_draw_grob_with_gp(self):
        grid_newpage()
        g = Grob(name="test", _grid_class="rect",
                 x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                 width=Unit(0.5, "npc"), height=Unit(0.5, "npc"),
                 gp=Gpar(col="red"))
        grid_draw(g)

    def test_draw_gtree_with_children_vp(self):
        from grid_py._viewport import Viewport
        grid_newpage()
        child_vp = Viewport(name="children_vp")
        child = Grob(name="c", _grid_class="rect",
                     x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                     width=Unit(0.5, "npc"), height=Unit(0.5, "npc"))
        tree = GTree(children=GList(child), name="tree")
        tree.childrenvp = child_vp
        grid_draw(tree)
