"""Tests to improve coverage of grid_py._size."""

import pytest

from grid_py._gpar import Gpar
from grid_py._size import (
    calc_string_metric,
    width_details,
    height_details,
    ascent_details,
    descent_details,
    x_details,
    y_details,
    grob_width,
    grob_height,
    grob_x,
    grob_y,
    grob_ascent,
    grob_descent,
    absolute_size,
    _apply_font_from_gpar,
    _get_measure_ctx,
)
from grid_py._units import Unit


class TestCalcStringMetric:
    """Tests for calc_string_metric."""

    def test_basic(self):
        m = calc_string_metric("Hello")
        assert "ascent" in m
        assert "descent" in m
        assert "width" in m
        assert m["width"] > 0

    def test_with_gpar_fontsize(self):
        gp = Gpar(fontsize=24)
        m = calc_string_metric("Test", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontfamily(self):
        gp = Gpar(fontfamily="serif")
        m = calc_string_metric("Test", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_cex(self):
        gp = Gpar(cex=2.0)
        m = calc_string_metric("X", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontface_bold(self):
        gp = Gpar(fontface="bold")
        m = calc_string_metric("B", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontface_italic(self):
        gp = Gpar(fontface="italic")
        m = calc_string_metric("I", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontface_bold_italic(self):
        gp = Gpar(fontface="bold.italic")
        m = calc_string_metric("BI", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontface_numeric_2(self):
        gp = Gpar(fontface=2)  # bold
        m = calc_string_metric("B", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontface_numeric_3(self):
        gp = Gpar(fontface=3)  # italic
        m = calc_string_metric("I", gp=gp)
        assert m["width"] > 0

    def test_with_gpar_fontface_numeric_4(self):
        gp = Gpar(fontface=4)  # bold italic
        m = calc_string_metric("BI", gp=gp)
        assert m["width"] > 0

    def test_empty_string(self):
        m = calc_string_metric("")
        assert m["width"] == pytest.approx(0, abs=0.01)

    def test_none_gpar(self):
        m = calc_string_metric("X", gp=None)
        assert m["width"] > 0

    def test_with_list_fontsize(self):
        gp = Gpar(fontsize=[18])
        m = calc_string_metric("X", gp=gp)
        assert m["width"] > 0

    def test_with_list_fontfamily(self):
        gp = Gpar(fontfamily=["monospace"])
        m = calc_string_metric("X", gp=gp)
        assert m["width"] > 0

    def test_with_list_cex(self):
        gp = Gpar(cex=[1.5])
        m = calc_string_metric("X", gp=gp)
        assert m["width"] > 0

    def test_with_list_fontface(self):
        gp = Gpar(fontface=["bold"])
        m = calc_string_metric("X", gp=gp)
        assert m["width"] > 0


class TestDetailDispatchers:
    """Tests for width_details, height_details, etc."""

    def test_width_details_with_method(self):
        class FakeGrob:
            def width_details(self):
                return Unit(5, "cm")
        result = width_details(FakeGrob())
        assert result._values[0] == 5

    def test_width_details_fallback(self):
        result = width_details("no_method")
        assert result._units[0] == "null"

    def test_height_details_with_method(self):
        class FakeGrob:
            def height_details(self):
                return Unit(3, "cm")
        result = height_details(FakeGrob())
        assert result._values[0] == 3

    def test_height_details_fallback(self):
        result = height_details("no_method")
        assert result._units[0] == "null"

    def test_ascent_details_with_method(self):
        class FakeGrob:
            def ascent_details(self):
                return Unit(2, "cm")
        result = ascent_details(FakeGrob())
        assert result._values[0] == 2

    def test_ascent_details_fallback_to_height(self):
        class FakeGrob:
            def height_details(self):
                return Unit(4, "cm")
        result = ascent_details(FakeGrob())
        assert result._values[0] == 4

    def test_ascent_details_full_fallback(self):
        result = ascent_details("no_method")
        assert result._units[0] == "null"

    def test_descent_details_with_method(self):
        class FakeGrob:
            def descent_details(self):
                return Unit(1, "cm")
        result = descent_details(FakeGrob())
        assert result._values[0] == 1

    def test_descent_details_fallback(self):
        result = descent_details("no_method")
        assert result._values[0] == 0

    def test_x_details_with_method(self):
        class FakeGrob:
            def x_details(self, theta):
                return Unit(theta, "cm")
        result = x_details(FakeGrob(), theta=45)
        assert result._values[0] == 45

    def test_x_details_fallback(self):
        result = x_details("no_method")
        assert result._values[0] == 0.5

    def test_y_details_with_method(self):
        class FakeGrob:
            def y_details(self, theta):
                return Unit(theta, "cm")
        result = y_details(FakeGrob(), theta=90)
        assert result._values[0] == 90

    def test_y_details_fallback(self):
        result = y_details("no_method")
        assert result._values[0] == 0.5


class TestGrobConvenience:
    """Tests for grob_width, grob_height, etc."""

    def test_grob_width(self):
        u = grob_width("fake_grob")
        assert u._units[0] == "grobwidth"

    def test_grob_height(self):
        u = grob_height("fake_grob")
        assert u._units[0] == "grobheight"

    def test_grob_x(self):
        u = grob_x("fake_grob", theta=45)
        assert u._units[0] == "grobx"

    def test_grob_y(self):
        u = grob_y("fake_grob", theta=90)
        assert u._units[0] == "groby"

    def test_grob_ascent(self):
        u = grob_ascent("fake_grob")
        assert u._units[0] == "grobascent"

    def test_grob_descent(self):
        u = grob_descent("fake_grob")
        assert u._units[0] == "grobdescent"


class TestDetailDispatchersReal:
    """Tests using real grobs (not FakeGrob) for _grid_class dispatch."""

    def test_text_grob_width(self):
        from grid_py._primitives import text_grob
        result = width_details(text_grob("Hello"))
        assert result._units[0] == "inches"
        assert result._values[0] > 0

    def test_null_grob_height(self):
        from grid_py._primitives import null_grob
        result = height_details(null_grob())
        assert result._values[0] == 0
        assert result._units[0] == "inches"

    def test_grob_returning_none_falls_back(self):
        """A grob whose width_details() returns None should get the default."""
        from grid_py._grob import Grob
        g = Grob()  # base Grob: width_details returns None
        result = width_details(g)
        assert result._units[0] == "null"

    def test_text_ascent_dispatch(self):
        from grid_py._primitives import text_grob
        result = ascent_details(text_grob("X"))
        assert result._units[0] == "inches"
        assert result._values[0] > 0

    def test_text_descent_dispatch(self):
        from grid_py._primitives import text_grob
        result = descent_details(text_grob("X"))
        assert result._units[0] == "inches"
        assert result._values[0] >= 0


class TestAbsoluteSize:
    """Tests for absolute_size."""

    def test_absolute_unit(self):
        u = Unit(2, "inches")
        result = absolute_size(u)
        assert isinstance(result, Unit)

    def test_relative_unit(self):
        u = Unit(0.5, "npc")
        result = absolute_size(u)
        assert isinstance(result, Unit)
