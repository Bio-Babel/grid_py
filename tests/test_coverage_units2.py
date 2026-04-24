"""Coverage tests for grid_py._units -- Unit arithmetic, conversion functions,
string constructors, etc.
"""
from __future__ import annotations

import warnings
import pytest
import numpy as np

from grid_py._units import (
    Unit,
    is_unit,
    unit_type,
    unit_c,
    unit_length,
    unit_pmax,
    unit_pmin,
    unit_psum,
    unit_rep,
    string_width,
    string_height,
    string_ascent,
    string_descent,
    absolute_size,
    convert_unit,
    convert_x,
    convert_y,
    convert_width,
    convert_height,
    _resolve_alias,
    _make_compound,
    _parallel_op,
    _ABSOLUTE_UNIT_TYPES,
)


# ---------------------------------------------------------------------------
# Unit construction
# ---------------------------------------------------------------------------


class TestUnitConstruction:
    def test_scalar(self):
        u = Unit(1, "cm")
        assert len(u) == 1
        assert u.units_list == ["cm"]
        assert float(u.values[0]) == 1.0

    def test_list(self):
        u = Unit([1, 2, 3], "cm")
        assert len(u) == 3

    def test_numpy_array(self):
        u = Unit(np.array([1.0, 2.0]), "inches")
        assert len(u) == 2

    def test_unit_from_unit(self):
        u1 = Unit(1, "cm")
        u2 = Unit(u1, "cm")  # unit from unit copies
        assert u2 == u1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="length > 0"):
            Unit([], "cm")

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit type"):
            Unit(1, "bogus_unit")

    def test_alias_resolution(self):
        u = Unit(1, "in")
        assert u.units_list[0] == "inches"

    def test_alias_pt(self):
        u = Unit(1, "pt")
        assert u.units_list[0] == "points"

    def test_alias_centimeters(self):
        u = Unit(1, "centimeters")
        assert u.units_list[0] == "cm"

    def test_multiple_units(self):
        u = Unit([1, 2], ["cm", "inches"])
        assert u.units_list == ["cm", "inches"]

    def test_units_recycled(self):
        u = Unit([1, 2, 3], ["cm", "inches"])
        assert len(u) == 3
        assert u.units_list[2] == "cm"  # recycled

    def test_values_recycled_to_match_units(self):
        u = Unit(1, ["cm", "inches"])
        assert len(u) == 2

    def test_empty_units_raises(self):
        with pytest.raises(ValueError, match="length > 0"):
            Unit(1, [])

    def test_data_default_none(self):
        u = Unit(1, "cm")
        assert u.data == [None]

    def test_data_scalar(self):
        u = Unit(1, "strwidth", data="hello")
        assert u.data == ["hello"]

    def test_data_list(self):
        u = Unit([1, 2], "strwidth", data=["a", "b"])
        assert u.data == ["a", "b"]

    def test_data_list_recycled(self):
        u = Unit([1, 2, 3], "strwidth", data=["a"])
        assert len(u.data) == 3


# ---------------------------------------------------------------------------
# Unit repr / str
# ---------------------------------------------------------------------------


class TestUnitRepr:
    def test_repr(self):
        u = Unit(2.5, "cm")
        r = repr(u)
        assert "2.5" in r
        assert "cm" in r

    def test_str(self):
        u = Unit(2.5, "cm")
        s = str(u)
        assert "2.5" in s
        assert "cm" in s

    def test_as_character_compound(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "inches")
        compound = u1 + u2
        s = compound.as_character()
        assert "sum" in s

    def test_desc_element_strwidth(self):
        u = Unit(1, "strwidth", data="hello")
        s = u._desc_element(0)
        assert "strwidth" in s
        assert "hello" in s

    def test_desc_element_grobwidth(self):
        u = Unit(1, "grobwidth", data="some_grob")
        s = u._desc_element(0)
        assert "grobwidth" in s


# ---------------------------------------------------------------------------
# Unit indexing
# ---------------------------------------------------------------------------


class TestUnitIndexing:
    def test_int_index(self):
        u = Unit([1, 2, 3], "cm")
        sub = u[1]
        assert len(sub) == 1
        assert float(sub.values[0]) == 2.0

    def test_negative_index(self):
        u = Unit([1, 2, 3], "cm")
        sub = u[-1]
        assert float(sub.values[0]) == 3.0

    def test_index_out_of_range(self):
        u = Unit([1], "cm")
        with pytest.raises(IndexError):
            u[5]

    def test_slice_index(self):
        u = Unit([1, 2, 3], "cm")
        sub = u[0:2]
        assert len(sub) == 2

    def test_list_index(self):
        u = Unit([1, 2, 3], "cm")
        sub = u[[0, 2]]
        assert len(sub) == 2

    def test_setitem_int(self):
        u = Unit([1, 2], "cm")
        u[0] = Unit(5, "cm")
        assert float(u.values[0]) == 5.0

    def test_setitem_negative(self):
        u = Unit([1, 2], "cm")
        u[-1] = Unit(10, "inches")
        assert u.units_list[-1] == "inches"

    def test_setitem_slice(self):
        u = Unit([1, 2, 3], "cm")
        u[0:2] = Unit(9, "mm")
        assert u.units_list[0] == "mm"
        assert u.units_list[1] == "mm"

    def test_setitem_non_unit_raises(self):
        u = Unit([1, 2], "cm")
        with pytest.raises(TypeError, match="must be a Unit"):
            u[0] = 5

    def test_setitem_invalid_index_raises(self):
        u = Unit([1, 2], "cm")
        with pytest.raises(TypeError, match="unsupported index"):
            u["bad"] = Unit(1, "cm")


# ---------------------------------------------------------------------------
# Unit arithmetic
# ---------------------------------------------------------------------------


class TestUnitArithmetic:
    def test_add_same_type(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "cm")
        result = u1 + u2
        assert float(result.values[0]) == 3.0

    def test_add_different_types(self):
        u1 = Unit(1, "cm")
        u2 = Unit(1, "npc")
        result = u1 + u2
        assert result.units_list[0] == "sum"

    def test_radd_zero(self):
        u = Unit(1, "cm")
        result = 0 + u
        assert float(result.values[0]) == 1.0

    def test_radd_unit(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "cm")
        # This triggers __radd__ on u2
        result = u1.__radd__(u2)
        assert float(result.values[0]) == 3.0

    def test_sub(self):
        u1 = Unit(3, "cm")
        u2 = Unit(1, "cm")
        result = u1 - u2
        assert float(result.values[0]) == 2.0

    def test_rsub(self):
        u1 = Unit(1, "cm")
        u2 = Unit(3, "cm")
        # u2.__rsub__(u1) computes u1 - u2 = 1 - 3 = -2
        result = u2.__rsub__(u1)
        assert float(result.values[0]) == -2.0

    def test_neg(self):
        u = Unit(5, "cm")
        result = -u
        assert float(result.values[0]) == -5.0

    def test_pos(self):
        u = Unit(5, "cm")
        result = +u
        assert float(result.values[0]) == 5.0

    def test_mul_scalar(self):
        u = Unit(2, "cm")
        result = u * 3
        assert float(result.values[0]) == 6.0

    def test_rmul_scalar(self):
        u = Unit(2, "cm")
        result = 3 * u
        assert float(result.values[0]) == 6.0

    def test_mul_unit_raises(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "cm")
        with pytest.raises(TypeError, match="Cannot multiply"):
            u1 * u2

    def test_mul_non_numeric_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__mul__("bad") is NotImplemented

    def test_rmul_non_numeric_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__rmul__("bad") is NotImplemented

    def test_div_scalar(self):
        u = Unit(6, "cm")
        result = u / 2
        assert float(result.values[0]) == 3.0

    def test_div_zero_raises(self):
        u = Unit(1, "cm")
        with pytest.raises(ZeroDivisionError):
            u / 0

    def test_div_unit_raises(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "cm")
        with pytest.raises(TypeError, match="Cannot divide"):
            u1 / u2

    def test_rdiv_raises(self):
        u = Unit(1, "cm")
        with pytest.raises(TypeError, match="Cannot divide"):
            5 / u

    def test_div_non_numeric_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__truediv__("bad") is NotImplemented

    def test_add_non_unit_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__add__("bad") is NotImplemented

    def test_radd_non_unit_non_zero_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__radd__("bad") is NotImplemented

    def test_sub_non_unit_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__sub__("bad") is NotImplemented

    def test_rsub_non_unit_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__rsub__("bad") is NotImplemented

    def test_mul_bool_returns_not_implemented(self):
        u = Unit(1, "cm")
        assert u.__mul__(True) is NotImplemented


# ---------------------------------------------------------------------------
# Unit equality
# ---------------------------------------------------------------------------


class TestUnitEquality:
    def test_equal(self):
        u1 = Unit(1, "cm")
        u2 = Unit(1, "cm")
        assert u1 == u2

    def test_not_equal_values(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "cm")
        assert u1 != u2

    def test_not_equal_units(self):
        u1 = Unit(1, "cm")
        u2 = Unit(1, "inches")
        assert u1 != u2

    def test_not_equal_length(self):
        u1 = Unit(1, "cm")
        u2 = Unit([1, 2], "cm")
        assert u1 != u2

    def test_eq_non_unit(self):
        u = Unit(1, "cm")
        assert u.__eq__("bad") is NotImplemented

    def test_ne_non_unit(self):
        u = Unit(1, "cm")
        assert u.__ne__("bad") is NotImplemented

    def test_hash_none(self):
        u = Unit(1, "cm")
        with pytest.raises(TypeError):
            hash(u)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestUnitHelpers:
    def test_copy(self):
        u = Unit(1, "cm")
        c = u.copy()
        assert c == u
        assert c is not u

    def test_is_absolute(self):
        u = Unit(1, "cm")
        assert u.is_absolute() is True

    def test_not_absolute(self):
        u = Unit(1, "npc")
        assert u.is_absolute() is False


# ---------------------------------------------------------------------------
# Module functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_is_unit(self):
        assert is_unit(Unit(1, "cm")) is True
        assert is_unit(42) is False

    def test_unit_type_single(self):
        assert unit_type(Unit(1, "cm")) == "cm"

    def test_unit_type_multiple(self):
        result = unit_type(Unit([1, 2], ["cm", "inches"]))
        assert result == ["cm", "inches"]

    def test_unit_type_non_unit_raises(self):
        with pytest.raises(TypeError, match="must be a Unit"):
            unit_type(42)

    def test_unit_c(self):
        result = unit_c(Unit(1, "cm"), Unit(2, "inches"))
        assert len(result) == 2
        assert result.units_list == ["cm", "inches"]

    def test_unit_c_empty_raises(self):
        with pytest.raises(ValueError):
            unit_c()

    def test_unit_c_non_unit_raises(self):
        with pytest.raises(TypeError, match="must be Unit"):
            unit_c(42)

    def test_unit_length(self):
        assert unit_length(Unit([1, 2, 3], "cm")) == 3

    def test_unit_length_non_unit_raises(self):
        with pytest.raises(TypeError, match="must be a Unit"):
            unit_length(42)

    def test_unit_pmax(self):
        result = unit_pmax(Unit([1, 4], "cm"), Unit([3, 2], "cm"))
        assert float(result.values[0]) == 3.0
        assert float(result.values[1]) == 4.0

    def test_unit_pmin(self):
        result = unit_pmin(Unit([1, 4], "cm"), Unit([3, 2], "cm"))
        assert float(result.values[0]) == 1.0
        assert float(result.values[1]) == 2.0

    def test_unit_psum(self):
        result = unit_psum(Unit([1, 2], "cm"), Unit([3, 4], "cm"))
        assert float(result.values[0]) == 4.0
        assert float(result.values[1]) == 6.0

    def test_unit_rep(self):
        u = Unit(1, "cm")
        result = unit_rep(u, 3)
        assert len(result) == 3

    def test_unit_rep_non_unit_raises(self):
        with pytest.raises(TypeError, match="must be a Unit"):
            unit_rep(42, 3)

    def test_unit_rep_zero_returns_empty(self):
        # R: rep(unit(1, "cm"), 0) returns empty unit
        result = unit_rep(Unit(1, "cm"), 0)
        assert len(result) == 0

    def test_parallel_op_single(self):
        u = Unit([1, 2], "cm")
        result = _parallel_op("sum", u)
        assert result == u

    def test_parallel_op_empty_raises(self):
        with pytest.raises(ValueError):
            _parallel_op("sum")


# ---------------------------------------------------------------------------
# String metric constructors
# ---------------------------------------------------------------------------


class TestStringConstructors:
    def test_string_width_single(self):
        u = string_width("hello")
        assert len(u) == 1
        assert u.units_list[0] == "strwidth"
        assert u.data[0] == "hello"

    def test_string_width_multiple(self):
        u = string_width(["a", "b", "c"])
        assert len(u) == 3

    def test_string_height_single(self):
        u = string_height("hello")
        assert u.units_list[0] == "strheight"

    def test_string_height_multiple(self):
        u = string_height(["a", "b"])
        assert len(u) == 2

    def test_string_ascent(self):
        u = string_ascent("test")
        assert u.units_list[0] == "strascent"

    def test_string_ascent_multiple(self):
        u = string_ascent(["a", "b"])
        assert len(u) == 2

    def test_string_descent(self):
        u = string_descent("test")
        assert u.units_list[0] == "strdescent"

    def test_string_descent_multiple(self):
        u = string_descent(["a", "b"])
        assert len(u) == 2


# ---------------------------------------------------------------------------
# absolute_size
# ---------------------------------------------------------------------------


class TestAbsoluteSize:
    def test_absolute(self):
        u = Unit(2, "cm")
        result = absolute_size(u)
        assert float(result.values[0]) == 2.0

    def test_non_absolute_replaced_with_null(self):
        # R: non-absolute units are replaced with unit(1, "null"),
        # not zeroed (grid/src/unit.c:1819-1820).
        u = Unit(2, "npc")
        result = absolute_size(u)
        assert float(result.values[0]) == 1.0
        assert result._units[0] == "null"

    def test_non_unit_raises(self):
        with pytest.raises(TypeError, match="must be a Unit"):
            absolute_size(42)


# ---------------------------------------------------------------------------
# convert_unit
# ---------------------------------------------------------------------------


class TestConvertUnit:
    def test_cm_to_inches(self):
        u = Unit(2.54, "cm")
        result = convert_unit(u, "inches")
        assert isinstance(result, Unit)
        assert float(result.values[0]) == pytest.approx(1.0, abs=0.01)

    def test_inches_to_cm(self):
        u = Unit(1, "inches")
        result = convert_unit(u, "cm")
        assert float(result.values[0]) == pytest.approx(2.54, abs=0.01)

    def test_same_unit(self):
        u = Unit(5, "cm")
        result = convert_unit(u, "cm")
        assert float(result.values[0]) == 5.0

    def test_value_only(self):
        u = Unit(2.54, "cm")
        result = convert_unit(u, "inches", valueOnly=True)
        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_context_dependent_auto_opens_default_device(self):
        # R parity: convertUnit(unit(1,"npc"), "cm") with no prior device
        # auto-opens a 7x7 in PDF-equivalent via GEcurrentDevice() and
        # resolves against it. 1 npc width = 7 in = 17.78 cm. (unit.R:59-75
        # → src/unit.c L_convert). Previous Python behaviour of returning
        # the unit unchanged with a warning was a divergence.
        from grid_py._state import get_state
        get_state()._renderer = None

        u = Unit(1, "npc")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = convert_unit(u, "cm")
        assert isinstance(result, Unit)
        assert float(result.values[0]) == pytest.approx(17.78, abs=1e-3)
        assert len(w) == 0, "R-faithful auto-open should not warn"

    def test_same_non_absolute_unit(self):
        u = Unit(0.5, "npc")
        result = convert_unit(u, "npc")
        assert float(result.values[0]) == 0.5

    def test_non_unit_raises(self):
        with pytest.raises(TypeError, match="must be a Unit"):
            convert_unit(42, "cm")

    def test_convert_x(self):
        u = Unit(1, "inches")
        result = convert_x(u, "cm")
        assert isinstance(result, Unit)

    def test_convert_y(self):
        u = Unit(1, "inches")
        result = convert_y(u, "cm")
        assert isinstance(result, Unit)

    def test_convert_width(self):
        u = Unit(1, "inches")
        result = convert_width(u, "cm")
        assert isinstance(result, Unit)

    def test_convert_height(self):
        u = Unit(1, "inches")
        result = convert_height(u, "cm")
        assert isinstance(result, Unit)

    def test_convert_x_value_only(self):
        u = Unit(2.54, "cm")
        result = convert_x(u, "inches", valueOnly=True)
        assert isinstance(result, np.ndarray)

    def test_convert_y_value_only(self):
        u = Unit(2.54, "cm")
        result = convert_y(u, "inches", valueOnly=True)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Compound unit operations
# ---------------------------------------------------------------------------


class TestCompoundUnit:
    def test_make_compound_sum_same_type(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "cm")
        result = _make_compound("sum", u1, u2)
        assert float(result.values[0]) == 3.0
        assert result.units_list[0] == "cm"

    def test_make_compound_min_same_type(self):
        u1 = Unit(3, "cm")
        u2 = Unit(1, "cm")
        result = _make_compound("min", u1, u2)
        assert float(result.values[0]) == 1.0

    def test_make_compound_max_same_type(self):
        u1 = Unit(1, "cm")
        u2 = Unit(3, "cm")
        result = _make_compound("max", u1, u2)
        assert float(result.values[0]) == 3.0

    def test_make_compound_different_types(self):
        u1 = Unit(1, "cm")
        u2 = Unit(1, "npc")
        result = _make_compound("sum", u1, u2)
        assert result.units_list[0] == "sum"

    def test_compound_recycling(self):
        u1 = Unit([1, 2], "cm")
        u2 = Unit(3, "cm")
        result = _make_compound("sum", u1, u2)
        assert len(result) == 2
        assert float(result.values[0]) == 4.0
        assert float(result.values[1]) == 5.0

    def test_desc_element_compound_with_scale(self):
        u1 = Unit(1, "cm")
        u2 = Unit(2, "npc")
        compound = u1 + u2
        # Compound desc should contain "sum"
        desc = compound._desc_element(0)
        assert "sum" in desc

    def test_desc_element_compound_fallback(self):
        # Create a compound with non-unit data
        u = Unit.__new__(Unit)
        u._values = np.array([1.0])
        u._units = ["sum"]
        u._data = ["not_a_unit"]
        u._is_absolute = False
        desc = u._desc_element(0)
        assert "sum" in desc


# ---------------------------------------------------------------------------
# _resolve_alias
# ---------------------------------------------------------------------------


class TestResolveAlias:
    def test_known_type(self):
        assert _resolve_alias("cm") == "cm"

    def test_alias(self):
        assert _resolve_alias("in") == "inches"
        assert _resolve_alias("millimeters") == "mm"

    def test_case_insensitive(self):
        assert _resolve_alias("CM") == "cm"
        assert _resolve_alias("Inches") == "inches"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown unit type"):
            _resolve_alias("bogus")
