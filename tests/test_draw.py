"""Tests for grid_py._draw -- drawing engine, display list, and helpers."""

import numpy as np
import pytest

from grid_py._draw import (
    delay_grob,
    grid_dl_apply,
    grid_draw,
    grid_newpage,
    grid_pretty,
    grid_refresh,
    record_grob,
)
from grid_py._gpar import Gpar
from grid_py._grob import GList, GTree, Grob
from grid_py.renderer import CairoRenderer, _parse_colour
from grid_py._state import GridState, get_state


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #


def _make_simple_rect(name="testrect"):
    """Create a rect Grob with plain float attributes (no Units).

    This avoids the Unit-to-float conversion issue that occurs when
    rect_grob() stores Unit objects and _render_grob tries float().
    """
    g = Grob(name=name, _grid_class="rect")
    g.x = 0.5
    g.y = 0.5
    g.width = 0.4
    g.height = 0.4
    g.hjust = 0.5
    g.vjust = 0.5
    return g


def _make_null_grob(name="testnull"):
    """Create a null grob that is a no-op when rendered."""
    return Grob(name=name, _grid_class="null")


@pytest.fixture(autouse=True)
def _reset_state():
    """Ensure each test starts with a clean grid state."""
    state = get_state()
    state.reset()
    yield


# ------------------------------------------------------------------ #
# grid_newpage                                                       #
# ------------------------------------------------------------------ #


class TestGridNewpage:
    """grid_newpage creates (or resets) a CairoRenderer."""

    def test_creates_renderer(self):
        grid_newpage()
        state = get_state()
        renderer = state.get_renderer()
        assert renderer is not None
        assert isinstance(renderer, CairoRenderer)

    def test_renderer_dimensions(self):
        """After newpage, renderer has the expected dimensions."""
        grid_newpage(width=5.0, height=4.0, dpi=100)
        state = get_state()
        renderer = state.get_renderer()
        assert renderer.width_in == 5.0
        assert renderer.height_in == 4.0

    def test_clears_display_list(self):
        """Display list should be empty after newpage."""
        grid_newpage()
        state = get_state()
        dl = state.get_display_list()
        assert len(list(dl)) == 0


# ------------------------------------------------------------------ #
# grid_draw                                                          #
# ------------------------------------------------------------------ #


class TestGridDraw:
    """grid_draw dispatches on grob type."""

    def test_draw_none_is_noop(self):
        """Passing None should not raise."""
        grid_newpage()
        grid_draw(None)

    def test_draw_rect_grob(self):
        """Drawing a rect grob should not raise."""
        grid_newpage()
        rg = _make_simple_rect("testrect")
        grid_draw(rg)
        # The display list should now contain at least one item
        state = get_state()
        dl = list(state.get_display_list())
        assert len(dl) >= 1

    def test_draw_glist(self):
        """Drawing a GList draws each child."""
        grid_newpage()
        g1 = _make_null_grob("r1")
        g2 = _make_null_grob("r2")
        gl = GList(g1, g2)
        grid_draw(gl)
        state = get_state()
        dl = list(state.get_display_list())
        # Each child in a GList is drawn with recording=True
        assert len(dl) >= 2

    def test_draw_gtree(self):
        """Drawing a GTree draws the tree and its children."""
        grid_newpage()
        child = _make_null_grob("child_rect")
        tree = GTree(children=GList(child), name="mytree")
        grid_draw(tree)
        state = get_state()
        dl = list(state.get_display_list())
        assert len(dl) >= 1

    def test_draw_with_recording_false(self):
        """recording=False should not add to the display list."""
        grid_newpage()
        rg = _make_null_grob("norec")
        grid_draw(rg, recording=False)
        state = get_state()
        dl = list(state.get_display_list())
        assert len(dl) == 0


# ------------------------------------------------------------------ #
# grid_refresh                                                       #
# ------------------------------------------------------------------ #


class TestGridRefresh:
    """grid_refresh replays the display list."""

    def test_refresh_after_draw(self):
        """Refresh should not raise after drawing a grob."""
        grid_newpage()
        rg = _make_null_grob("refreshtest")
        grid_draw(rg)
        # This should replay the display list
        grid_refresh()
        state = get_state()
        renderer = state.get_renderer()
        assert renderer is not None


# ------------------------------------------------------------------ #
# record_grob / delay_grob                                          #
# ------------------------------------------------------------------ #


class TestRecordGrob:
    """record_grob wraps a callable in a grob."""

    def test_returns_grob(self):
        def my_expr():
            pass

        rg = record_grob(my_expr, name="rec1")
        assert isinstance(rg, Grob)
        assert rg._grid_class == "recordedGrob"

    def test_draw_details_calls_expr(self):
        called = {"flag": False}

        def my_expr():
            called["flag"] = True

        rg = record_grob(my_expr, name="rec2")
        rg.draw_details()
        assert called["flag"] is True

    def test_env_passed_to_expr(self):
        results = {}

        def my_expr(x=0, y=0):
            results["x"] = x
            results["y"] = y

        rg = record_grob(my_expr, list_={"x": 42, "y": 99}, name="rec3")
        rg.draw_details()
        assert results == {"x": 42, "y": 99}


class TestDelayGrob:
    """delay_grob wraps a callable that returns a grob at make_content time."""

    def test_returns_gtree(self):
        def expr():
            return _make_null_grob("delayed_rect")

        dg = delay_grob(expr, name="delay1")
        assert isinstance(dg, GTree)
        assert dg._grid_class == "delayedgrob"

    def test_make_content_evaluates_expr(self):
        def expr():
            return _make_null_grob("delayed_rect")

        dg = delay_grob(expr, name="delay2")
        result = dg.make_content()
        # After make_content, the tree should have children
        assert result is dg
        assert len(list(dg._children)) > 0

    def test_make_content_with_env(self):
        def expr(w=0.5):
            return _make_null_grob("delayed_rect_w")

        dg = delay_grob(expr, list_={"w": 0.75}, name="delay3")
        dg.make_content()
        assert len(list(dg._children)) > 0


# ------------------------------------------------------------------ #
# grid_dl_apply                                                      #
# ------------------------------------------------------------------ #


class TestGridDlApply:
    """grid_dl_apply applies a function to each display-list item."""

    def test_identity_function(self):
        """Applying an identity function should keep the list unchanged."""
        grid_newpage()
        rg = _make_null_grob("dlapply_test")
        grid_draw(rg)

        state = get_state()
        dl_before = list(state.get_display_list())
        count_before = len(dl_before)

        grid_dl_apply(lambda item: item)

        dl_after = list(state.get_display_list())
        assert len(dl_after) == count_before

    def test_type_mismatch_raises(self):
        """Returning a different type should raise TypeError."""
        grid_newpage()
        rg = _make_null_grob("dlapply_bad")
        grid_draw(rg)

        with pytest.raises(TypeError, match="invalid modification"):
            grid_dl_apply(lambda item: "wrong_type")


# ------------------------------------------------------------------ #
# grid_pretty                                                        #
# ------------------------------------------------------------------ #


class TestGridPretty:
    """grid_pretty returns nice tick positions."""

    def test_basic_range(self):
        ticks = grid_pretty([0.0, 1.0])
        assert isinstance(ticks, np.ndarray)
        assert len(ticks) > 0
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 1.0

    def test_negative_range(self):
        ticks = grid_pretty([-10.0, 10.0])
        assert ticks[0] <= -10.0
        assert ticks[-1] >= 10.0

    def test_single_point_range(self):
        ticks = grid_pretty([5.0, 5.0])
        assert len(ticks) >= 1
        assert 5.0 in ticks


# ------------------------------------------------------------------ #
# _parse_colour (renderer colour helper)                             #
# ------------------------------------------------------------------ #


class TestParseColour:
    """_parse_colour converts R colour specs to (r, g, b, a)."""

    def test_none_returns_black(self):
        assert _parse_colour(None) == (0.0, 0.0, 0.0, 1.0)

    def test_named_red(self):
        r, g, b, a = _parse_colour("red")
        assert r == 1.0 and g == 0.0 and b == 0.0 and a == 1.0

    def test_hex_colour(self):
        r, g, b, a = _parse_colour("#FF0000")
        assert r == pytest.approx(1.0) and g == 0.0 and b == 0.0

    def test_hex_with_alpha(self):
        r, g, b, a = _parse_colour("#FF000080")
        assert a == pytest.approx(128 / 255.0)

    def test_grey_scale(self):
        r, g, b, a = _parse_colour("grey50")
        # R's grey50 = #7F7F7F = 127/255 ≈ 0.498
        assert r == pytest.approx(0x7F / 255.0)

    def test_transparent(self):
        _, _, _, a = _parse_colour("transparent")
        assert a == 0.0


class TestVectorisedGparDispatch:
    """Per-sub-polyline / per-segment gpar recycling.

    Gold standard: grid/src/grid.c gridLines / L_segments call
    ``updateGContext(gp, j)`` per sub-line / segment, so a vectorised
    colour/lwd/lty applies element-wise.  A scalar dash-pattern lty
    ("44") must NOT be mistaken for a vectorised value.
    """

    def test_scalar_dash_lty_not_vectorised(self):
        from grid_py._draw import _gpar_is_vectorised
        assert _gpar_is_vectorised(Gpar(col="red", lwd=2, lty="44")) is False
        assert _gpar_is_vectorised(Gpar(col="red", lty="dashed")) is False

    def test_vectorised_colour_detected(self):
        from grid_py._draw import _gpar_is_vectorised, _subset_gpar
        gp = Gpar(col=["red", "blue"], lwd=3, lty="44")
        assert _gpar_is_vectorised(gp) is True
        # per-line colour, but the scalar dash pattern survives intact
        assert _subset_gpar(gp, 0).get("col") == "red"
        assert _subset_gpar(gp, 1).get("col") == "blue"
        assert _subset_gpar(gp, 0).get("lty") == "44"
        assert _subset_gpar(gp, 1).get("lty") == "44"

    def test_vectorised_lty_types_indexed(self):
        from grid_py._draw import _subset_gpar
        gp = Gpar(col="black", lty=["solid", "dashed"])
        assert _subset_gpar(gp, 0).get("lty") == "solid"
        assert _subset_gpar(gp, 1).get("lty") == "dashed"

    def test_polyline_vector_gp_renders(self):
        from grid_py._draw import _render_grob
        from grid_py import polyline_grob
        r = CairoRenderer(width=4, height=3, dpi=100)
        gp = Gpar(col=["red", "blue"], lwd=[2, 5])
        g = polyline_grob(
            x=np.array([0.1, 0.5, 0.6, 0.9]),
            y=np.array([0.1, 0.5, 0.6, 0.2]),
            id=[1, 1, 2, 2], gp=gp,
        )
        _render_grob(g, r, gp=gp)  # must not raise

    def test_segments_vector_gp_renders(self):
        from grid_py._draw import _render_grob
        from grid_py import segments_grob
        r = CairoRenderer(width=4, height=3, dpi=100)
        gp = Gpar(col=["green", "purple"], lwd=[2, 5])
        g = segments_grob(
            x0=np.array([0.2, 0.4]), y0=np.array([0.1, 0.1]),
            x1=np.array([0.2, 0.4]), y1=np.array([0.4, 0.4]), gp=gp,
        )
        _render_grob(g, r, gp=gp)  # must not raise
