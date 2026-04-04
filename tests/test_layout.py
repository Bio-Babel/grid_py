"""Tests for grid_py._layout -- GridLayout and accessor functions."""

import numpy as np
import pytest

from grid_py._layout import (
    GridLayout,
    layout_heights,
    layout_ncol,
    layout_nrow,
    layout_region,
    layout_respect,
    layout_widths,
)
from grid_py._units import Unit


# ------------------------------------------------------------------ #
# Construction                                                       #
# ------------------------------------------------------------------ #


class TestGridLayoutConstruction:
    """GridLayout.__init__ with various argument combinations."""

    def test_defaults(self):
        """Default 1x1 layout with null widths/heights."""
        gl = GridLayout()
        assert gl.nrow == 1
        assert gl.ncol == 1
        assert gl.dim == (1, 1)
        assert gl.respect is False

    def test_custom_nrow_ncol(self):
        """Explicit nrow/ncol creates the right shape."""
        gl = GridLayout(nrow=3, ncol=4)
        assert gl.nrow == 3
        assert gl.ncol == 4
        assert gl.dim == (3, 4)

    def test_default_widths_are_null_units(self):
        """When widths is None, each column gets Unit(1, 'null')."""
        gl = GridLayout(nrow=2, ncol=3)
        w = gl.widths
        assert isinstance(w, Unit)
        # Should have 3 values, all 1.0
        vals = np.asarray(w._values, dtype=float)
        np.testing.assert_array_equal(vals, [1.0, 1.0, 1.0])

    def test_default_heights_are_null_units(self):
        """When heights is None, each row gets Unit(1, 'null')."""
        gl = GridLayout(nrow=4, ncol=2)
        h = gl.heights
        vals = np.asarray(h._values, dtype=float)
        np.testing.assert_array_equal(vals, [1.0, 1.0, 1.0, 1.0])

    def test_custom_widths_unit(self):
        """Explicit Unit widths are stored unchanged."""
        w = Unit([2.0, 3.0, 5.0], "cm")
        gl = GridLayout(nrow=1, ncol=3, widths=w)
        assert gl.widths is w

    def test_custom_heights_unit(self):
        """Explicit Unit heights are stored unchanged."""
        h = Unit([1.0, 2.0], "inches")
        gl = GridLayout(nrow=2, ncol=1, heights=h)
        assert gl.heights is h

    def test_numeric_widths_coerced_with_default_units(self):
        """Plain numeric widths are wrapped in Unit(default_units)."""
        gl = GridLayout(nrow=1, ncol=2, widths=[3.0, 7.0], default_units="cm")
        vals = np.asarray(gl.widths._values, dtype=float)
        np.testing.assert_array_almost_equal(vals, [3.0, 7.0])


# ------------------------------------------------------------------ #
# Respect matrix                                                     #
# ------------------------------------------------------------------ #


class TestGridLayoutRespect:
    """Respect specification: False, True, and matrix modes."""

    def test_respect_false(self):
        gl = GridLayout(nrow=2, ncol=2, respect=False)
        assert gl.respect is False
        assert gl._valid_respect == 0
        np.testing.assert_array_equal(
            gl.respect_mat, np.zeros((2, 2), dtype=np.int32)
        )

    def test_respect_true(self):
        gl = GridLayout(nrow=2, ncol=2, respect=True)
        assert gl.respect is True
        assert gl._valid_respect == 1

    def test_respect_matrix(self):
        mat = np.array([[1, 0], [0, 1]], dtype=np.int32)
        gl = GridLayout(nrow=2, ncol=2, respect=mat)
        assert gl._valid_respect == 2
        np.testing.assert_array_equal(gl.respect, mat)

    def test_respect_matrix_wrong_shape_raises(self):
        mat = np.array([[1, 0, 1]], dtype=np.int32)
        with pytest.raises(ValueError, match="nrow.*ncol"):
            GridLayout(nrow=2, ncol=2, respect=mat)


# ------------------------------------------------------------------ #
# Properties                                                         #
# ------------------------------------------------------------------ #


class TestGridLayoutProperties:
    """Property accessors on GridLayout."""

    def test_nrow(self):
        assert GridLayout(nrow=5, ncol=3).nrow == 5

    def test_ncol(self):
        assert GridLayout(nrow=5, ncol=3).ncol == 3

    def test_dim(self):
        assert GridLayout(nrow=5, ncol=3).dim == (5, 3)

    def test_repr_contains_nrow_ncol(self):
        r = repr(GridLayout(nrow=2, ncol=3))
        assert "nrow=2" in r
        assert "ncol=3" in r


# ------------------------------------------------------------------ #
# Module-level accessor functions                                    #
# ------------------------------------------------------------------ #


class TestLayoutAccessors:
    """layout_nrow, layout_ncol, layout_widths, layout_heights, layout_respect."""

    @pytest.fixture()
    def layout(self):
        return GridLayout(nrow=3, ncol=4)

    def test_layout_nrow(self, layout):
        assert layout_nrow(layout) == 3

    def test_layout_ncol(self, layout):
        assert layout_ncol(layout) == 4

    def test_layout_widths(self, layout):
        w = layout_widths(layout)
        assert isinstance(w, Unit)
        assert len(np.asarray(w._values)) == 4

    def test_layout_heights(self, layout):
        h = layout_heights(layout)
        assert isinstance(h, Unit)
        assert len(np.asarray(h._values)) == 3

    def test_layout_respect_false(self, layout):
        assert layout_respect(layout) is False


# ------------------------------------------------------------------ #
# layout_region                                                      #
# ------------------------------------------------------------------ #


class TestLayoutRegion:
    """layout_region returns correct NPC fractions for cell ranges."""

    @pytest.fixture()
    def equal_layout(self):
        """A 3x3 layout with equal null widths and heights."""
        return GridLayout(nrow=3, ncol=3)

    def test_single_cell_center(self, equal_layout):
        """Cell (2, 2) in a 3x3 equal layout occupies the centre third."""
        region = layout_region(equal_layout, row=2, col=2)
        assert isinstance(region["left"], Unit)
        np.testing.assert_almost_equal(float(region["left"]._values[0]), 1.0 / 3)
        np.testing.assert_almost_equal(float(region["width"]._values[0]), 1.0 / 3)
        np.testing.assert_almost_equal(float(region["height"]._values[0]), 1.0 / 3)

    def test_single_cell_top_left(self, equal_layout):
        """Cell (1, 1) is the top-left cell."""
        region = layout_region(equal_layout, row=1, col=1)
        np.testing.assert_almost_equal(float(region["left"]._values[0]), 0.0)
        np.testing.assert_almost_equal(float(region["width"]._values[0]), 1.0 / 3)
        # Bottom should be 2/3 (rows go top-to-bottom, NPC y=0 is bottom)
        np.testing.assert_almost_equal(float(region["bottom"]._values[0]), 2.0 / 3)

    def test_range_of_cells(self, equal_layout):
        """Rows 1-2, columns 2-3 should span 2/3 of width and 2/3 of height."""
        region = layout_region(equal_layout, row=[1, 2], col=[2, 3])
        np.testing.assert_almost_equal(float(region["width"]._values[0]), 2.0 / 3)
        np.testing.assert_almost_equal(float(region["height"]._values[0]), 2.0 / 3)

    def test_full_layout_span(self, equal_layout):
        """Spanning all rows/cols gives width=1 and height=1."""
        region = layout_region(equal_layout, row=[1, 3], col=[1, 3])
        np.testing.assert_almost_equal(float(region["width"]._values[0]), 1.0)
        np.testing.assert_almost_equal(float(region["height"]._values[0]), 1.0)
        np.testing.assert_almost_equal(float(region["left"]._values[0]), 0.0)
        np.testing.assert_almost_equal(float(region["bottom"]._values[0]), 0.0)

    def test_unequal_widths(self):
        """Non-uniform widths produce correct fractional regions."""
        gl = GridLayout(
            nrow=1,
            ncol=3,
            widths=Unit([1.0, 2.0, 1.0], "null"),
        )
        region = layout_region(gl, row=1, col=2)
        # Total width = 4, col 2 width = 2 -> fraction 0.5
        np.testing.assert_almost_equal(float(region["width"]._values[0]), 0.5)
        # Left offset = 1/4 = 0.25
        np.testing.assert_almost_equal(float(region["left"]._values[0]), 0.25)
