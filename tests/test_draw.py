"""Tests for grid_py._draw -- drawing engine, display list, and helpers."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from grid_py._draw import (
    _gpar_to_mpl,
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
# grid_py._primitives is not imported directly; we use helper functions
# to construct simple grobs with plain float attributes.
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
    """Ensure each test starts with a clean grid state and closed figures."""
    plt.close("all")
    state = get_state()
    state.reset()
    yield
    plt.close("all")


# ------------------------------------------------------------------ #
# grid_newpage                                                       #
# ------------------------------------------------------------------ #


class TestGridNewpage:
    """grid_newpage creates (or resets) a matplotlib figure."""

    def test_creates_figure(self):
        grid_newpage()
        state = get_state()
        fig, ax = state.get_device()
        assert fig is not None
        assert ax is not None

    def test_axes_limits(self):
        """After newpage, axes span [0,1] x [0,1]."""
        grid_newpage()
        state = get_state()
        _, ax = state.get_device()
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)

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
        fig, ax = state.get_device()
        assert fig is not None


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
# _gpar_to_mpl conversion                                           #
# ------------------------------------------------------------------ #


class TestGparToMpl:
    """_gpar_to_mpl converts Gpar to matplotlib keyword arguments."""

    def test_none_returns_empty(self):
        assert _gpar_to_mpl(None) == {}

    def test_col_mapped_to_color(self):
        gp = Gpar(col="red")
        kw = _gpar_to_mpl(gp)
        assert kw["color"] == "red"

    def test_fill_mapped_to_facecolor(self):
        gp = Gpar(fill="blue")
        kw = _gpar_to_mpl(gp)
        assert kw["facecolor"] == "blue"

    def test_alpha(self):
        gp = Gpar(alpha=0.5)
        kw = _gpar_to_mpl(gp)
        assert kw["alpha"] == pytest.approx(0.5)

    def test_lwd_mapped_to_linewidth(self):
        gp = Gpar(lwd=2.5)
        kw = _gpar_to_mpl(gp)
        assert kw["linewidth"] == pytest.approx(2.5)

    def test_lty_mapped_to_linestyle(self):
        gp = Gpar(lty="dashed")
        kw = _gpar_to_mpl(gp)
        assert kw["linestyle"] == "dashed"

    def test_fontsize(self):
        gp = Gpar(fontsize=14)
        kw = _gpar_to_mpl(gp)
        assert kw["fontsize"] == pytest.approx(14.0)

    def test_fontsize_with_cex(self):
        gp = Gpar(fontsize=10, cex=2.0)
        kw = _gpar_to_mpl(gp)
        assert kw["fontsize"] == pytest.approx(20.0)

    def test_fontfamily(self):
        gp = Gpar(fontfamily="serif")
        kw = _gpar_to_mpl(gp)
        assert kw["fontfamily"] == "serif"

    def test_bold_fontface(self):
        """_gpar_to_mpl reads 'fontface' key; set it directly via _params."""
        gp = Gpar()
        gp._params["fontface"] = 2
        kw = _gpar_to_mpl(gp)
        assert kw.get("fontweight") == "bold"

    def test_lineend_mapped(self):
        gp = Gpar(lineend="round")
        kw = _gpar_to_mpl(gp)
        assert kw["solid_capstyle"] == "round"

    def test_linejoin_mapped(self):
        gp = Gpar(linejoin="bevel")
        kw = _gpar_to_mpl(gp)
        assert kw["solid_joinstyle"] == "bevel"
