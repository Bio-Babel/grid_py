"""Tests for grob metrics: text, null, rect width/height/ascent/descent.

Validates the _grid_class-based dispatch in _size.py and the grob/string
metric evaluation in convert_unit().
"""

import numpy as np
import pytest

from grid_py._gpar import Gpar
from grid_py._grob import Grob
from grid_py._primitives import null_grob, rect_grob, text_grob
from grid_py._size import (
    ascent_details,
    descent_details,
    grob_height,
    grob_width,
    height_details,
    width_details,
)
from grid_py._units import (
    Unit,
    convert_unit,
    string_ascent,
    string_descent,
    string_height,
    string_width,
)


# ===================================================================== #
#  Text grob metrics                                                     #
# ===================================================================== #


class TestTextGrobMetrics:
    """Text grob width/height/ascent/descent (R: primitives.R:1430-1470)."""

    def test_width_positive(self):
        tg = text_grob("Hello")
        w = width_details(tg)
        assert isinstance(w, Unit)
        assert w._units[0] == "inches"
        assert w._values[0] > 0

    def test_width_longer_text_wider(self):
        w_short = width_details(text_grob("X"))
        w_long = width_details(text_grob("XXXXX"))
        assert w_long._values[0] > w_short._values[0]

    def test_width_larger_font_wider(self):
        w_small = width_details(text_grob("X"))
        w_large = width_details(text_grob("X", gp=Gpar(fontsize=24)))
        assert w_large._values[0] > w_small._values[0]

    def test_width_multiple_labels_takes_max(self):
        tg = text_grob(["Hi", "Hello World"])
        w = width_details(tg)
        assert w._units[0] == "inches"
        # Max should be width of "Hello World"
        w_single = width_details(text_grob("Hello World"))
        assert w._values[0] == pytest.approx(w_single._values[0], abs=1e-6)

    def test_width_empty_string(self):
        w = width_details(text_grob(""))
        assert w._values[0] == pytest.approx(0, abs=0.01)

    def test_height_positive(self):
        h = height_details(text_grob("X"))
        assert h._units[0] == "inches"
        assert h._values[0] > 0

    def test_height_equals_ascent_plus_descent(self):
        tg = text_grob("X")
        h = height_details(tg)
        a = ascent_details(tg)
        d = descent_details(tg)
        assert h._values[0] == pytest.approx(
            a._values[0] + d._values[0], abs=1e-6
        )

    def test_ascent_single_label(self):
        a = ascent_details(text_grob("X"))
        assert a._units[0] == "inches"
        assert a._values[0] > 0

    def test_ascent_multi_label_falls_back_to_height(self):
        tg = text_grob(["A", "B"])
        a = ascent_details(tg)
        h = height_details(tg)
        assert a._values[0] == pytest.approx(h._values[0], abs=1e-6)

    def test_descent_single_label(self):
        d = descent_details(text_grob("y"))  # 'y' has a descender
        assert d._units[0] == "inches"
        assert d._values[0] >= 0

    def test_descent_multi_label_zero(self):
        d = descent_details(text_grob(["A", "B"]))
        assert d._values[0] == pytest.approx(0, abs=1e-10)
        assert d._units[0] == "inches"

    def test_with_bold_gpar(self):
        w = width_details(text_grob("X", gp=Gpar(fontface="bold")))
        assert w._values[0] > 0

    def test_with_serif_gpar(self):
        w = width_details(text_grob("X", gp=Gpar(fontfamily="serif")))
        assert w._values[0] > 0


# ===================================================================== #
#  Null grob metrics                                                     #
# ===================================================================== #


class TestNullGrobMetrics:
    """Null grob: always zero (R: primitives.R:1676-1682)."""

    def test_width_zero(self):
        w = width_details(null_grob())
        assert w._values[0] == 0
        assert w._units[0] == "inches"

    def test_height_zero(self):
        h = height_details(null_grob())
        assert h._values[0] == 0
        assert h._units[0] == "inches"


# ===================================================================== #
#  Rect grob metrics                                                     #
# ===================================================================== #


class TestRectGrobMetrics:
    """Rect grob: returns its own width/height (R: primitives.R:1146-1166)."""

    def test_width_absolute(self):
        rg = rect_grob(width=Unit(3, "cm"))
        w = width_details(rg)
        assert w._values[0] == 3
        assert w._units[0] == "cm"

    def test_height_absolute(self):
        rg = rect_grob(height=Unit(2, "inches"))
        h = height_details(rg)
        assert h._values[0] == 2
        assert h._units[0] == "inches"

    def test_default_width_npc(self):
        rg = rect_grob()
        w = width_details(rg)
        # Default rect width is Unit(1, "npc")
        assert w._units[0] == "npc"

    def test_default_height_npc(self):
        rg = rect_grob()
        h = height_details(rg)
        assert h._units[0] == "npc"


# ===================================================================== #
#  Default / fallback                                                    #
# ===================================================================== #


class TestDefaultFallback:
    """Plain Grob() without _grid_class returns defaults."""

    def test_width_null_unit(self):
        g = Grob()
        w = width_details(g)
        assert w._units[0] == "null"
        assert w._values[0] == 1

    def test_height_null_unit(self):
        g = Grob()
        h = height_details(g)
        assert h._units[0] == "null"
        assert h._values[0] == 1

    def test_ascent_falls_back_to_height(self):
        g = Grob()
        a = ascent_details(g)
        assert a._units[0] == "null"

    def test_descent_zero_inches(self):
        g = Grob()
        d = descent_details(g)
        assert d._values[0] == 0
        assert d._units[0] == "inches"

    def test_custom_subclass_dispatch(self):
        class MyGrob(Grob):
            def width_details(self):
                return Unit(42, "mm")

        w = width_details(MyGrob())
        assert w._values[0] == 42
        assert w._units[0] == "mm"


# ===================================================================== #
#  convert_unit with string metrics                                      #
# ===================================================================== #


class TestConvertUnitStringMetrics:
    """convert_unit() evaluating strwidth/strheight/strascent/strdescent."""

    def test_strwidth_to_inches(self):
        u = string_width("Hello")
        result = convert_unit(u, "inches")
        assert result._units[0] == "inches"
        assert result._values[0] > 0

    def test_strwidth_to_cm(self):
        u = string_width("Test")
        result = convert_unit(u, "cm")
        assert result._units[0] == "cm"
        assert result._values[0] > 0

    def test_strheight_to_inches(self):
        u = string_height("X")
        result = convert_unit(u, "inches")
        assert result._units[0] == "inches"
        assert result._values[0] > 0

    def test_strascent_to_inches(self):
        u = string_ascent("X")
        result = convert_unit(u, "inches")
        assert result._units[0] == "inches"
        assert result._values[0] > 0

    def test_strdescent_to_inches(self):
        u = string_descent("y")
        result = convert_unit(u, "inches")
        assert result._units[0] == "inches"
        assert result._values[0] >= 0

    def test_strwidth_value_only(self):
        u = string_width("X")
        result = convert_unit(u, "cm", valueOnly=True)
        assert isinstance(result, np.ndarray)
        assert result[0] > 0

    def test_strwidth_longer_text_wider(self):
        u1 = convert_unit(string_width("X"), "inches")
        u2 = convert_unit(string_width("XXXXX"), "inches")
        assert u2._values[0] > u1._values[0]


# ===================================================================== #
#  convert_unit with grob metrics                                        #
# ===================================================================== #


class TestConvertUnitGrobMetrics:
    """convert_unit() evaluating grobwidth/grobheight/grobascent/grobdescent."""

    def test_grobwidth_text_to_inches(self):
        gw = grob_width(text_grob("Hello"))
        result = convert_unit(gw, "inches")
        assert result._units[0] == "inches"
        assert result._values[0] > 0

    def test_grobheight_text_to_cm(self):
        gh = grob_height(text_grob("X"))
        result = convert_unit(gh, "cm")
        assert result._units[0] == "cm"
        assert result._values[0] > 0

    def test_grobwidth_null_is_zero(self):
        gw = grob_width(null_grob())
        result = convert_unit(gw, "inches")
        assert result._values[0] == pytest.approx(0, abs=1e-10)

    def test_grobheight_null_is_zero(self):
        gh = grob_height(null_grob())
        result = convert_unit(gh, "inches")
        assert result._values[0] == pytest.approx(0, abs=1e-10)

    def test_grobwidth_rect_absolute(self):
        rg = rect_grob(width=Unit(5, "cm"))
        gw = grob_width(rg)
        result = convert_unit(gw, "inches")
        assert result._units[0] == "inches"
        assert result._values[0] == pytest.approx(5 / 2.54, abs=1e-4)

    def test_grobwidth_value_only(self):
        gw = grob_width(text_grob("X"))
        result = convert_unit(gw, "inches", valueOnly=True)
        assert isinstance(result, np.ndarray)
        assert result[0] > 0

    def test_grobwidth_text_consistency(self):
        """grob_width → convert should match direct width_details."""
        tg = text_grob("Test")
        direct = width_details(tg)
        via_convert = convert_unit(grob_width(tg), "inches")
        assert direct._values[0] == pytest.approx(
            via_convert._values[0], abs=1e-6
        )
