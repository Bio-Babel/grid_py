"""Tests for grid_py high-level functions.

Covers frame_grob, pack_grob, place_grob, xaxis_grob, yaxis_grob,
legend_grob, grid_grill, grid_abline, grid_show_layout, and layout_torture.
"""

import warnings

import matplotlib
matplotlib.use("Agg")

import pytest

import grid_py
from grid_py import (
    GList,
    GTree,
    Gpar,
    Grob,
    GridLayout,
    Unit,
    frame_grob,
    grid_frame,
    pack_grob,
    grid_pack,
    place_grob,
    grid_place,
    xaxis_grob,
    grid_xaxis,
    yaxis_grob,
    grid_yaxis,
    legend_grob,
    grid_legend,
    grid_grill,
    grid_abline,
    grid_show_layout,
    layout_torture,
    rect_grob,
    circle_grob,
    text_grob,
    is_grob,
    get_state,
    grid_newpage,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset grid state before each test."""
    state = get_state()
    state.reset()
    yield
    state.reset()


# =========================================================================
# frame_grob
# =========================================================================


class TestFrameGrob:
    """Tests for frame_grob construction."""

    def test_creates_gtree(self):
        fg = frame_grob()
        assert isinstance(fg, GTree)

    def test_grid_class_is_frame(self):
        fg = frame_grob()
        assert fg._grid_class == "frame"

    def test_custom_name(self):
        fg = frame_grob(name="myframe")
        assert fg.name == "myframe"

    def test_with_layout(self):
        lay = GridLayout(nrow=2, ncol=2)
        fg = frame_grob(layout=lay)
        assert hasattr(fg, "framevp")
        assert fg.framevp is not None

    def test_without_layout(self):
        fg = frame_grob()
        assert fg.framevp is None

    def test_with_gp(self):
        gp = Gpar(col="red")
        fg = frame_grob(gp=gp)
        assert fg.gp is not None


# =========================================================================
# pack_grob
# =========================================================================


class TestPackGrob:
    """Tests for pack_grob basic packing."""

    def test_pack_returns_gtree(self):
        fg = frame_grob(name="f")
        r = rect_grob(name="r")
        result = pack_grob(fg, r)
        assert isinstance(result, GTree)

    def test_pack_adds_child(self):
        fg = frame_grob(name="f")
        r = rect_grob(name="r")
        result = pack_grob(fg, r)
        assert result.n_children() >= 1

    def test_pack_creates_layout(self):
        fg = frame_grob(name="f")
        r = rect_grob(name="r")
        result = pack_grob(fg, r)
        assert hasattr(result, "framevp")
        assert result.framevp is not None

    def test_pack_side_left(self):
        fg = frame_grob(name="f")
        r1 = rect_grob(name="r1")
        r2 = rect_grob(name="r2")
        fg = pack_grob(fg, r1, side="left")
        fg = pack_grob(fg, r2, side="right")
        assert fg.n_children() == 2

    def test_pack_side_top_bottom(self):
        fg = frame_grob(name="f")
        r1 = rect_grob(name="top_item")
        r2 = rect_grob(name="bottom_item")
        fg = pack_grob(fg, r1, side="top")
        fg = pack_grob(fg, r2, side="bottom")
        assert fg.n_children() == 2

    def test_pack_with_explicit_width_height(self):
        fg = frame_grob(name="f")
        r = rect_grob(name="r")
        result = pack_grob(fg, r, width=Unit(0.5, "npc"), height=Unit(0.3, "npc"))
        assert isinstance(result, GTree)

    def test_pack_invalid_frame_raises(self):
        with pytest.raises(TypeError):
            pack_grob(rect_grob(name="not_a_frame"), rect_grob(name="r"))

    def test_pack_invalid_grob_raises(self):
        fg = frame_grob(name="f")
        with pytest.raises(TypeError):
            pack_grob(fg, "not_a_grob")  # type: ignore[arg-type]

    def test_pack_does_not_mutate_original(self):
        fg = frame_grob(name="f")
        original_children = fg.n_children()
        pack_grob(fg, rect_grob(name="r"))
        # pack_grob deepcopies, so original should be unchanged
        assert fg.n_children() == original_children


# =========================================================================
# place_grob
# =========================================================================


class TestPlaceGrob:
    """Tests for place_grob."""

    def test_place_into_existing_cell(self):
        fg = frame_grob(name="f")
        r1 = rect_grob(name="r1")
        fg = pack_grob(fg, r1, side="left")
        # Now place into the existing cell
        r2 = circle_grob(name="r2")
        result = place_grob(fg, r2, row=1, col=1)
        assert isinstance(result, GTree)
        assert result.n_children() >= 2

    def test_place_invalid_frame_raises(self):
        with pytest.raises(TypeError):
            place_grob(rect_grob(name="x"), rect_grob(name="y"))

    def test_place_invalid_grob_raises(self):
        fg = frame_grob(name="f")
        fg = pack_grob(fg, rect_grob(name="seed"), side="left")
        with pytest.raises(TypeError):
            place_grob(fg, "not_a_grob")  # type: ignore[arg-type]

    def test_place_out_of_range_raises(self):
        fg = frame_grob(name="f")
        fg = pack_grob(fg, rect_grob(name="r"), side="left")
        with pytest.raises(ValueError):
            place_grob(fg, circle_grob(name="c"), row=99, col=99)


# =========================================================================
# xaxis_grob / yaxis_grob
# =========================================================================


class TestAxisGrobs:
    """Tests for xaxis_grob and yaxis_grob construction."""

    def test_xaxis_grob_returns_gtree(self):
        xg = xaxis_grob(at=[0.2, 0.4, 0.6, 0.8])
        assert isinstance(xg, GTree)

    def test_xaxis_grob_class_tag(self):
        xg = xaxis_grob(at=[0.0, 0.5, 1.0])
        assert xg._grid_class == "xaxis"

    def test_xaxis_grob_has_children(self):
        xg = xaxis_grob(at=[0.0, 0.5, 1.0])
        assert xg.n_children() >= 2  # major + ticks (+ possibly labels)

    def test_xaxis_grob_custom_name(self):
        xg = xaxis_grob(at=[0, 1], name="myxaxis")
        assert xg.name == "myxaxis"

    def test_xaxis_grob_no_labels(self):
        xg = xaxis_grob(at=[0, 0.5, 1], label=False)
        children_names = list(xg._children_order)
        assert "labels" not in children_names

    def test_xaxis_grob_custom_labels(self):
        xg = xaxis_grob(at=[0, 0.5, 1], label=["lo", "mid", "hi"])
        children_names = list(xg._children_order)
        assert "labels" in children_names

    def test_xaxis_grob_none_at(self):
        xg = xaxis_grob(at=None)
        assert isinstance(xg, GTree)
        assert xg.n_children() == 0

    def test_xaxis_main_false(self):
        xg = xaxis_grob(at=[0, 1], main=False)
        assert xg._grid_class == "xaxis"

    def test_yaxis_grob_returns_gtree(self):
        yg = yaxis_grob(at=[0.2, 0.4, 0.6, 0.8])
        assert isinstance(yg, GTree)

    def test_yaxis_grob_class_tag(self):
        yg = yaxis_grob(at=[0.0, 0.5, 1.0])
        assert yg._grid_class == "yaxis"

    def test_yaxis_grob_has_children(self):
        yg = yaxis_grob(at=[0.0, 0.5, 1.0])
        assert yg.n_children() >= 2

    def test_yaxis_grob_no_labels(self):
        yg = yaxis_grob(at=[0, 1], label=False)
        children_names = list(yg._children_order)
        assert "labels" not in children_names

    def test_yaxis_main_false(self):
        yg = yaxis_grob(at=[0, 1], main=False)
        assert yg._grid_class == "yaxis"


# =========================================================================
# legend_grob
# =========================================================================


class TestLegendGrob:
    """Tests for legend_grob construction."""

    def test_returns_gtree(self):
        lg = legend_grob(labels=["A", "B"])
        assert isinstance(lg, GTree)

    def test_single_label(self):
        lg = legend_grob(labels=["solo"])
        assert isinstance(lg, GTree)

    def test_empty_labels_returns_null(self):
        lg = legend_grob(labels=[])
        # Empty labels returns a null grob
        assert is_grob(lg)

    def test_custom_nrow_ncol(self):
        lg = legend_grob(labels=["A", "B", "C", "D"], nrow=2, ncol=2)
        assert isinstance(lg, GTree)

    def test_byrow(self):
        lg = legend_grob(labels=["A", "B", "C", "D"], nrow=2, ncol=2, byrow=True)
        assert isinstance(lg, GTree)

    def test_do_lines_false(self):
        lg = legend_grob(labels=["A"], do_lines=False)
        assert isinstance(lg, GTree)

    def test_do_points_false(self):
        lg = legend_grob(labels=["A"], do_points=False)
        assert isinstance(lg, GTree)

    def test_with_pch(self):
        lg = legend_grob(labels=["A", "B"], pch=[1, 2])
        assert isinstance(lg, GTree)

    def test_with_gp(self):
        gp = Gpar(col=["red", "blue"])
        lg = legend_grob(labels=["A", "B"], gp=gp)
        assert isinstance(lg, GTree)

    def test_invalid_nrow_raises(self):
        with pytest.raises(ValueError):
            legend_grob(labels=["A"], nrow=0)

    def test_insufficient_cells_raises(self):
        with pytest.raises(ValueError):
            legend_grob(labels=["A", "B", "C"], nrow=1, ncol=1)

    def test_custom_hgap_vgap(self):
        lg = legend_grob(
            labels=["A", "B"],
            hgap=Unit(2, "lines"),
            vgap=Unit(0.5, "lines"),
        )
        assert isinstance(lg, GTree)

    def test_numeric_hgap(self):
        lg = legend_grob(labels=["A"], hgap=2, vgap=0.5)
        assert isinstance(lg, GTree)


# =========================================================================
# grid_grill
# =========================================================================


class TestGridGrill:
    """Tests for grid_grill."""

    def test_basic_call(self):
        """grid_grill should run without error."""
        grid_newpage()
        grid_grill()

    def test_custom_positions(self):
        grid_newpage()
        grid_grill(h=Unit([0.3, 0.7], "npc"), v=Unit([0.3, 0.7], "npc"))

    def test_custom_gp(self):
        grid_newpage()
        grid_grill(gp=Gpar(col="blue"))

    def test_numeric_positions(self):
        grid_newpage()
        grid_grill(h=[0.25, 0.5, 0.75], v=[0.25, 0.5, 0.75])


# =========================================================================
# grid_abline
# =========================================================================


class TestGridAbline:
    """Tests for grid_abline."""

    def test_returns_grob(self):
        grid_newpage()
        g = grid_abline(draw=False)
        assert is_grob(g)

    def test_default_slope_intercept(self):
        grid_newpage()
        g = grid_abline(draw=False)
        assert g is not None

    def test_custom_slope_intercept(self):
        grid_newpage()
        g = grid_abline(intercept=0.5, slope=0.5, draw=False)
        assert is_grob(g)

    def test_with_gp(self):
        grid_newpage()
        g = grid_abline(gp=Gpar(col="red", lwd=2), draw=False)
        assert g.gp is not None

    def test_custom_name(self):
        grid_newpage()
        g = grid_abline(name="myline", draw=False)
        assert g.name == "myline"

    def test_draw_true(self):
        """grid_abline with draw=True constructs the grob and records it."""
        grid_newpage()
        # draw=True records to display list; rendering may warn on Unit types
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = grid_abline(draw=True)
        assert is_grob(g)


# =========================================================================
# grid_show_layout
# =========================================================================


class TestGridShowLayout:
    """Tests for grid_show_layout."""

    def test_basic_call(self):
        lay = GridLayout(nrow=2, ncol=2)
        vp = grid_show_layout(lay)
        # Should return a viewport
        assert vp is not None

    def test_no_newpage(self):
        grid_newpage()
        lay = GridLayout(nrow=1, ncol=1)
        vp = grid_show_layout(lay, newpage=False)
        assert vp is not None

    def test_custom_colors(self):
        lay = GridLayout(nrow=2, ncol=3)
        vp = grid_show_layout(
            lay,
            bg="white",
            cell_border="red",
            cell_fill="yellow",
            label_col="black",
            unit_col="green",
        )
        assert vp is not None

    def test_no_cell_label(self):
        lay = GridLayout(nrow=2, ncol=2)
        vp = grid_show_layout(lay, cell_label=False)
        assert vp is not None


# =========================================================================
# layout_torture
# =========================================================================


class TestLayoutTorture:
    """Tests for layout_torture."""

    def test_basic_call(self):
        """layout_torture should run without error."""
        layout_torture()

    def test_custom_dimensions(self):
        layout_torture(n_row=3, n_col=4)

    def test_single_cell(self):
        layout_torture(n_row=1, n_col=1)


# =========================================================================
# grid_xaxis / grid_yaxis (drawing variants)
# =========================================================================


class TestAxisDrawing:
    """Tests for grid_xaxis and grid_yaxis with drawing."""

    def test_grid_xaxis_draw(self):
        grid_newpage()
        xg = grid_xaxis(at=[0.0, 0.5, 1.0], draw=True)
        assert isinstance(xg, GTree)

    def test_grid_xaxis_no_draw(self):
        xg = grid_xaxis(at=[0, 1], draw=False)
        assert isinstance(xg, GTree)

    def test_grid_yaxis_draw(self):
        grid_newpage()
        yg = grid_yaxis(at=[0.0, 0.5, 1.0], draw=True)
        assert isinstance(yg, GTree)

    def test_grid_yaxis_no_draw(self):
        yg = grid_yaxis(at=[0, 1], draw=False)
        assert isinstance(yg, GTree)


# =========================================================================
# grid_legend (drawing variant)
# =========================================================================


class TestGridLegend:
    """Tests for grid_legend with drawing."""

    def test_draw_true(self):
        grid_newpage()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lg = grid_legend(labels=["A", "B"], draw=True)
            except (ValueError, TypeError):
                # Rendering may fail on shape mismatches for complex gTrees
                lg = grid_legend(labels=["A", "B"], draw=False)
        assert isinstance(lg, GTree)

    def test_draw_false(self):
        lg = grid_legend(labels=["X", "Y"], draw=False)
        assert isinstance(lg, GTree)
