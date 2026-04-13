"""Tests for the three-phase layout negotiation algorithm.

Validates _calc_layout_sizes() and layout_region() against R's
layout.c:calcViewportLayout semantics.
"""

import numpy as np
import pytest

from grid_py._layout import (
    GridLayout,
    _calc_layout_sizes,
    _col_respected,
    _row_respected,
    layout_region,
)
from grid_py._units import Unit, unit_c


# ===================================================================== #
#  Helper: 300x200 px at 100 DPI (3 in x 2 in)                         #
# ===================================================================== #
PW, PH, DPI = 300.0, 200.0, 100.0


# ===================================================================== #
#  Phase 1: Absolute unit allocation                                     #
# ===================================================================== #


class TestPhase1Absolute:
    """Absolute (non-null) units are allocated first."""

    def test_all_null_proportional(self):
        layout = GridLayout(nrow=1, ncol=3,
                            widths=Unit([1, 2, 1], "null"))
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        assert cw[0] == pytest.approx(75.0)   # 1/4 * 300
        assert cw[1] == pytest.approx(150.0)  # 2/4 * 300
        assert cw[2] == pytest.approx(75.0)

    def test_mixed_cm_and_null(self):
        # 2.54cm = 1 inch = 100px at 100 DPI; remaining 200px to null
        widths = unit_c(Unit(2.54, "cm"), Unit(1, "null"))
        layout = GridLayout(nrow=1, ncol=2, widths=widths)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        assert cw[0] == pytest.approx(100.0, abs=1.0)  # 1 inch
        assert cw[1] == pytest.approx(200.0, abs=1.0)  # remaining

    def test_mixed_npc_and_null(self):
        widths = unit_c(Unit(0.3, "npc"), Unit(1, "null"))
        layout = GridLayout(nrow=1, ncol=2, widths=widths)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        assert cw[0] == pytest.approx(90.0)   # 0.3 * 300
        assert cw[1] == pytest.approx(210.0)  # remaining

    def test_mixed_inches_and_null(self):
        widths = unit_c(Unit(1, "inches"), Unit(1, "inches"), Unit(1, "null"))
        layout = GridLayout(nrow=1, ncol=3, widths=widths)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        assert cw[0] == pytest.approx(100.0)  # 1 inch
        assert cw[1] == pytest.approx(100.0)  # 1 inch
        assert cw[2] == pytest.approx(100.0)  # remaining

    def test_absolute_overflow_null_gets_zero(self):
        # 5 inches > 3 inches total → null gets 0
        widths = unit_c(Unit(5, "inches"), Unit(1, "null"))
        layout = GridLayout(nrow=1, ncol=2, widths=widths)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        assert cw[0] == pytest.approx(500.0)  # 5 inches regardless
        assert cw[1] == pytest.approx(0.0)    # no space left

    def test_heights_cm_and_null(self):
        heights = unit_c(Unit(2.54, "cm"), Unit(1, "null"))
        layout = GridLayout(nrow=2, ncol=1, heights=heights)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        assert rh[0] == pytest.approx(100.0, abs=1.0)  # 1 inch
        assert rh[1] == pytest.approx(100.0, abs=1.0)  # remaining


# ===================================================================== #
#  Phase 2: Respected null units (aspect ratio)                          #
# ===================================================================== #


class TestPhase2Respect:
    """Respected null units maintain aspect ratio."""

    def test_respect_true_square_layout(self):
        # 2 cols [1,1] null, 2 rows [1,1] null, respect=True
        # Available: 300x200. Limiting dim = height (200).
        # Scale = min(300/2, 200/2) = 100
        layout = GridLayout(nrow=2, ncol=2,
                            widths=Unit([1, 1], "null"),
                            heights=Unit([1, 1], "null"),
                            respect=True)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        # All nulls respected → Phase 3 has nothing
        # scale = min(300/2, 200/2) = 100
        assert cw[0] == pytest.approx(100.0)
        assert cw[1] == pytest.approx(100.0)
        assert rh[0] == pytest.approx(100.0)
        assert rh[1] == pytest.approx(100.0)

    def test_respect_true_unequal(self):
        # widths [1,2] null, heights [1,2] null, respect=True
        # sumW = 3, sumH = 3
        # Available: 300x200
        # tempH * sumW (200*3=600) > sumH * tempW (3*300=900)? 600 < 900
        # So denom=sumH=3, mult=tempH=200
        layout = GridLayout(nrow=2, ncol=2,
                            widths=Unit([1, 2], "null"),
                            heights=Unit([1, 2], "null"),
                            respect=True)
        cw, rh = _calc_layout_sizes(layout, PW, PH, DPI)
        # scale = 200/3 ≈ 66.67
        scale = 200.0 / 3.0
        assert cw[0] == pytest.approx(1.0 * scale, abs=0.1)
        assert cw[1] == pytest.approx(2.0 * scale, abs=0.1)
        assert rh[0] == pytest.approx(1.0 * scale, abs=0.1)
        assert rh[1] == pytest.approx(2.0 * scale, abs=0.1)


# ===================================================================== #
#  Respect matrix (per-cell respect)                                     #
# ===================================================================== #


class TestRespectMatrix:
    """Per-cell respect using a matrix."""

    def test_col_respected(self):
        mat = np.array([[1, 0], [0, 0]])  # Only cell (0,0) respected
        layout = GridLayout(nrow=2, ncol=2,
                            widths=Unit([1, 1], "null"),
                            heights=Unit([1, 1], "null"),
                            respect=mat)
        assert _col_respected(0, layout) is True   # col 0 has respected cell
        assert _col_respected(1, layout) is False   # col 1 has no respected cell

    def test_row_respected(self):
        mat = np.array([[1, 0], [0, 0]])
        layout = GridLayout(nrow=2, ncol=2,
                            widths=Unit([1, 1], "null"),
                            heights=Unit([1, 1], "null"),
                            respect=mat)
        assert _row_respected(0, layout) is True
        assert _row_respected(1, layout) is False

    def test_no_respect(self):
        layout = GridLayout(nrow=2, ncol=2, respect=False)
        assert _col_respected(0, layout) is False
        assert _row_respected(0, layout) is False


# ===================================================================== #
#  layout_region() with full algorithm                                   #
# ===================================================================== #


class TestLayoutRegionFull:
    """layout_region() with parent dimensions uses three-phase algorithm."""

    def test_mixed_units_cell_1_1(self):
        # col 1: 1 inch, col 2: 1 null (gets remaining)
        widths = unit_c(Unit(1, "inches"), Unit(1, "null"))
        layout = GridLayout(nrow=1, ncol=2, widths=widths)
        region = layout_region(layout, 1, 1,
                               parent_w_px=PW, parent_h_px=PH, dpi=DPI)
        # col 1 = 100px / 300px total = 1/3 NPC
        assert region["width"]._values[0] == pytest.approx(100.0 / 300.0, abs=0.01)
        assert region["left"]._values[0] == pytest.approx(0.0)

    def test_mixed_units_cell_1_2(self):
        widths = unit_c(Unit(1, "inches"), Unit(1, "null"))
        layout = GridLayout(nrow=1, ncol=2, widths=widths)
        region = layout_region(layout, 1, 2,
                               parent_w_px=PW, parent_h_px=PH, dpi=DPI)
        # col 2 = 200px / 300px = 2/3 NPC
        assert region["width"]._values[0] == pytest.approx(200.0 / 300.0, abs=0.01)
        assert region["left"]._values[0] == pytest.approx(100.0 / 300.0, abs=0.01)

    def test_fallback_proportional(self):
        # Without parent dims, proportional allocation
        layout = GridLayout(nrow=1, ncol=3,
                            widths=Unit([1, 2, 1], "null"))
        region = layout_region(layout, 1, 2)
        assert region["width"]._values[0] == pytest.approx(0.5)
        assert region["left"]._values[0] == pytest.approx(0.25)
