"""Comprehensive tests for grid_py._units (the Unit system).

Covers construction, arithmetic, indexing, string representation,
helper functions, conversion, and edge cases.  Ported / inspired by
grid/tests/units.R where applicable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sys, os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_REPO, "grid_py")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from grid_py import (
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
    absolute_size,
    convert_unit,
)


# ===================================================================
# Construction
# ===================================================================


class TestUnitConstruction:
    """Tests for Unit.__init__ with various inputs."""

    def test_single_value_single_unit(self):
        u = Unit(1, "cm")
        assert len(u) == 1
        assert u.values[0] == pytest.approx(1.0)
        assert u.units_list == ["cm"]

    def test_single_value_alias_inch(self):
        u = Unit(2, "inch")
        assert u.units_list == ["inches"]
        assert u.values[0] == pytest.approx(2.0)

    def test_single_value_alias_pt(self):
        u = Unit(12, "pt")
        assert u.units_list == ["points"]

    def test_multiple_values_single_unit(self):
        u = Unit([1, 2, 3, 4], "npc")
        assert len(u) == 4
        np.testing.assert_array_almost_equal(u.values, [1, 2, 3, 4])
        assert u.units_list == ["npc"] * 4

    def test_multiple_values_multiple_units(self):
        u = Unit([1, 2, 3, 4], ["npc", "cm", "npc", "cm"])
        assert len(u) == 4
        assert u.units_list == ["npc", "cm", "npc", "cm"]

    def test_unit_recycles_shorter_units_list(self):
        u = Unit([1, 2, 3, 4], ["npc", "cm"])
        assert len(u) == 4
        assert u.units_list == ["npc", "cm", "npc", "cm"]

    def test_numpy_array_input(self):
        arr = np.array([0.5, 1.5])
        u = Unit(arr, "mm")
        assert len(u) == 2
        np.testing.assert_array_almost_equal(u.values, [0.5, 1.5])

    def test_from_existing_unit(self):
        """Unit(existing_unit, units) should copy it."""
        orig = Unit(3, "cm")
        copy = Unit(orig, "cm")
        assert copy == orig
        # Mutating copy should not affect original
        copy._values[0] = 99.0
        assert orig.values[0] == pytest.approx(3.0)

    def test_copy_method(self):
        """Unit.copy() returns an independent copy."""
        orig = Unit(3, "cm")
        cp = orig.copy()
        assert cp == orig
        cp._values[0] = 99.0
        assert orig.values[0] == pytest.approx(3.0)

    def test_invalid_unit_type_raises(self):
        with pytest.raises(ValueError, match="Unknown unit type"):
            Unit(1, "foobar")

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            Unit([], "cm")


# ===================================================================
# Arithmetic
# ===================================================================


class TestUnitArithmetic:
    """Tests for +, -, *, /, negation."""

    def test_add_same_type(self):
        a = Unit(1, "cm")
        b = Unit(2, "cm")
        c = a + b
        assert len(c) == 1
        assert c.values[0] == pytest.approx(3.0)
        assert c.units_list == ["cm"]

    def test_add_different_type_creates_compound(self):
        a = Unit(1, "cm")
        b = Unit(1, "inches")
        c = a + b
        assert len(c) == 1
        # Compound unit stores the operation in the "sum" type
        assert c.units_list == ["sum"]

    def test_subtract_same_type(self):
        a = Unit(5, "mm")
        b = Unit(2, "mm")
        c = a - b
        assert c.values[0] == pytest.approx(3.0)

    def test_subtract_different_type_creates_compound(self):
        a = Unit(1, "cm")
        b = Unit(1, "npc")
        c = a - b
        assert c.units_list == ["sum"]

    def test_scalar_mul_right(self):
        u = Unit(3, "cm")
        result = u * 2
        assert result.values[0] == pytest.approx(6.0)
        assert result.units_list == ["cm"]

    def test_scalar_mul_left(self):
        u = Unit(3, "cm")
        result = 2 * u
        assert result.values[0] == pytest.approx(6.0)

    def test_scalar_mul_float(self):
        u = Unit(4, "inches")
        result = 0.5 * u
        assert result.values[0] == pytest.approx(2.0)

    def test_div_by_scalar(self):
        u = Unit(6, "cm")
        result = u / 3
        assert result.values[0] == pytest.approx(2.0)

    def test_div_by_zero_raises(self):
        u = Unit(1, "cm")
        with pytest.raises(ZeroDivisionError):
            u / 0

    def test_mul_two_units_raises(self):
        with pytest.raises(TypeError):
            Unit(1, "cm") * Unit(2, "cm")

    def test_div_by_unit_raises(self):
        with pytest.raises(TypeError):
            Unit(1, "cm") / Unit(2, "cm")


# ===================================================================
# Negation
# ===================================================================


class TestUnitNegation:

    def test_negation(self):
        u = Unit(3, "cm")
        neg = -u
        assert neg.values[0] == pytest.approx(-3.0)
        assert neg.units_list == ["cm"]

    def test_negation_does_not_mutate_original(self):
        u = Unit(5, "mm")
        _ = -u
        assert u.values[0] == pytest.approx(5.0)

    def test_double_negation(self):
        u = Unit(7, "inches")
        assert (-(-u)).values[0] == pytest.approx(7.0)


# ===================================================================
# Indexing and slicing
# ===================================================================


class TestUnitIndexing:

    def test_single_index(self):
        u = Unit([10, 20, 30], "cm")
        elem = u[1]
        assert len(elem) == 1
        assert elem.values[0] == pytest.approx(20.0)

    def test_negative_index(self):
        u = Unit([10, 20, 30], "cm")
        elem = u[-1]
        assert elem.values[0] == pytest.approx(30.0)

    def test_slice(self):
        u = Unit([1, 2, 3, 4, 5], "mm")
        s = u[1:4]
        assert len(s) == 3
        np.testing.assert_array_almost_equal(s.values, [2, 3, 4])

    def test_index_out_of_range_raises(self):
        u = Unit(1, "cm")
        with pytest.raises(IndexError):
            u[5]

    def test_setitem_single(self):
        u = Unit([1, 2, 3], ["cm", "inches", "npc"])
        u[1] = Unit(99, "points")
        assert u.values[1] == pytest.approx(99.0)
        assert u.units_list[1] == "points"

    def test_setitem_slice(self):
        u = Unit([1, 2, 3, 4, 5], "mm")
        u[1:4] = Unit([20, 30, 40], "cm")
        assert u.units_list[1] == "cm"
        assert u.values[2] == pytest.approx(30.0)

    def test_list_index(self):
        u = Unit([10, 20, 30, 40], "cm")
        sub = u[[0, 3]]
        assert len(sub) == 2
        np.testing.assert_array_almost_equal(sub.values, [10, 40])


# ===================================================================
# Length
# ===================================================================


class TestUnitLength:

    def test_len_single(self):
        assert len(Unit(1, "cm")) == 1

    def test_len_multiple(self):
        assert len(Unit([1, 2, 3], "npc")) == 3

    def test_unit_length_function(self):
        u = Unit([1, 2, 3, 4], "cm")
        assert unit_length(u) == 4


# ===================================================================
# String representation
# ===================================================================


class TestUnitStringRepr:

    def test_as_character_simple(self):
        u = Unit(2.5, "cm")
        s = u.as_character()
        assert "2.5" in s
        assert "cm" in s

    def test_str_calls_as_character(self):
        u = Unit(1, "npc")
        assert str(u) == u.as_character()

    def test_repr(self):
        u = Unit(1, "cm")
        r = repr(u)
        assert "Unit" in r
        assert "cm" in r

    def test_as_character_multi(self):
        u = Unit([1, 2], ["cm", "inches"])
        s = u.as_character()
        assert "cm" in s
        assert "inches" in s


# ===================================================================
# unit_c -- concatenation
# ===================================================================


class TestUnitC:

    def test_concat_two(self):
        a = Unit(1, "cm")
        b = Unit(2, "inches")
        c = unit_c(a, b)
        assert len(c) == 2
        assert c.units_list == ["cm", "inches"]
        np.testing.assert_array_almost_equal(c.values, [1, 2])

    def test_concat_three(self):
        a = Unit(1, "cm")
        b = Unit(2, "mm")
        c = Unit(3, "npc")
        result = unit_c(a, b, c)
        assert len(result) == 3

    def test_concat_preserves_multi_element(self):
        a = Unit([1, 2], "cm")
        b = Unit(3, "inches")
        c = unit_c(a, b)
        assert len(c) == 3
        assert c.units_list == ["cm", "cm", "inches"]

    def test_concat_no_args_raises(self):
        with pytest.raises(ValueError):
            unit_c()

    def test_concat_non_unit_raises(self):
        with pytest.raises(TypeError):
            unit_c(Unit(1, "cm"), 42)


# ===================================================================
# unit_pmax, unit_pmin, unit_psum
# ===================================================================


class TestUnitParallel:

    def test_pmax_same_type(self):
        a = Unit([1, 4], "cm")
        b = Unit([3, 2], "cm")
        result = unit_pmax(a, b)
        np.testing.assert_array_almost_equal(result.values, [3, 4])

    def test_pmin_same_type(self):
        a = Unit([1, 4], "cm")
        b = Unit([3, 2], "cm")
        result = unit_pmin(a, b)
        np.testing.assert_array_almost_equal(result.values, [1, 2])

    def test_psum_same_type(self):
        a = Unit([1, 2], "cm")
        b = Unit([3, 4], "cm")
        result = unit_psum(a, b)
        np.testing.assert_array_almost_equal(result.values, [4, 6])

    def test_pmax_single_element(self):
        """Ported from R: length(unit.pmax(x)) == 1 for single-element input (PR#14443)."""
        x = Unit(1, "cm")
        result = unit_pmax(x)
        assert len(result) == 1

    def test_pmin_single_element(self):
        x = Unit(1, "cm")
        result = unit_pmin(x)
        assert len(result) == 1

    def test_pmax_different_types_compound(self):
        a = Unit(1, "cm")
        b = Unit(1, "inches")
        result = unit_pmax(a, b)
        assert len(result) == 1
        assert result.units_list == ["max"]


# ===================================================================
# unit_rep
# ===================================================================


class TestUnitRep:

    def test_rep_simple(self):
        u = Unit(1, "cm")
        r = unit_rep(u, 3)
        assert len(r) == 3
        np.testing.assert_array_almost_equal(r.values, [1, 1, 1])
        assert r.units_list == ["cm"] * 3

    def test_rep_multi_element(self):
        u = Unit([1, 2], "mm")
        r = unit_rep(u, 2)
        assert len(r) == 4
        np.testing.assert_array_almost_equal(r.values, [1, 2, 1, 2])

    def test_rep_mixed_units(self):
        u = Unit([1, 2], ["cm", "inches"])
        r = unit_rep(u, 3)
        assert len(r) == 6
        assert r.units_list == ["cm", "inches", "cm", "inches", "cm", "inches"]

    def test_rep_zero_raises(self):
        with pytest.raises(ValueError):
            unit_rep(Unit(1, "cm"), 0)


# ===================================================================
# is_unit
# ===================================================================


class TestIsUnit:

    def test_unit_object(self):
        assert is_unit(Unit(1, "cm")) is True

    def test_number(self):
        assert is_unit(42) is False

    def test_string(self):
        assert is_unit("cm") is False

    def test_none(self):
        assert is_unit(None) is False


# ===================================================================
# unit_type
# ===================================================================


class TestUnitType:

    def test_single_element(self):
        assert unit_type(Unit(1, "cm")) == "cm"

    def test_multiple_elements(self):
        u = Unit([1, 2], ["cm", "inches"])
        result = unit_type(u)
        assert result == ["cm", "inches"]

    def test_alias_resolved(self):
        u = Unit(1, "in")
        assert unit_type(u) == "inches"

    def test_non_unit_raises(self):
        with pytest.raises(TypeError):
            unit_type(42)


# ===================================================================
# convert_unit -- absolute conversions
# ===================================================================


class TestConvertUnit:

    def test_inches_to_cm(self):
        u = Unit(1, "inches")
        result = convert_unit(u, "cm")
        assert result.values[0] == pytest.approx(2.54)
        assert result.units_list == ["cm"]

    def test_cm_to_inches(self):
        u = Unit(2.54, "cm")
        result = convert_unit(u, "inches")
        assert result.values[0] == pytest.approx(1.0)

    def test_mm_to_cm(self):
        u = Unit(10, "mm")
        result = convert_unit(u, "cm")
        assert result.values[0] == pytest.approx(1.0)

    def test_cm_to_mm(self):
        u = Unit(1, "cm")
        result = convert_unit(u, "mm")
        assert result.values[0] == pytest.approx(10.0)

    def test_points_to_inches(self):
        u = Unit(72.27, "points")
        result = convert_unit(u, "inches")
        assert result.values[0] == pytest.approx(1.0)

    def test_roundtrip_cm_inches_cm(self):
        u = Unit(5, "cm")
        r1 = convert_unit(u, "inches")
        r2 = convert_unit(r1, "cm")
        assert r2.values[0] == pytest.approx(5.0)

    def test_value_only(self):
        u = Unit(1, "inches")
        arr = convert_unit(u, "cm", valueOnly=True)
        assert isinstance(arr, np.ndarray)
        assert arr[0] == pytest.approx(2.54)

    def test_same_unit_noop(self):
        u = Unit(3.14, "cm")
        result = convert_unit(u, "cm")
        assert result.values[0] == pytest.approx(3.14)

    def test_multi_element_conversion(self):
        u = Unit([1, 2, 3], "cm")
        result = convert_unit(u, "mm")
        np.testing.assert_array_almost_equal(result.values, [10, 20, 30])

    def test_alias_target_accepted(self):
        u = Unit(2.54, "cm")
        result = convert_unit(u, "in")
        assert result.values[0] == pytest.approx(1.0)


# ===================================================================
# string_width / string_height
# ===================================================================


class TestStringMetrics:

    def test_string_width_single(self):
        u = string_width("hello")
        assert is_unit(u)
        assert len(u) == 1
        assert u.units_list == ["strwidth"]
        assert u.data == ["hello"]

    def test_string_width_multiple(self):
        u = string_width(["a", "bb", "ccc"])
        assert len(u) == 3
        assert u.units_list == ["strwidth"] * 3
        assert u.data == ["a", "bb", "ccc"]

    def test_string_height_single(self):
        u = string_height("world")
        assert is_unit(u)
        assert u.units_list == ["strheight"]
        assert u.data == ["world"]


# ===================================================================
# absolute_size
# ===================================================================


class TestAbsoluteSize:

    def test_absolute_unit_unchanged(self):
        u = Unit(2, "cm")
        result = absolute_size(u)
        assert result.values[0] == pytest.approx(2.0)

    def test_context_unit_zeroed(self):
        u = Unit(1, "npc")
        result = absolute_size(u)
        assert result.values[0] == pytest.approx(0.0)

    def test_mixed_units(self):
        u = Unit([3, 1], ["cm", "npc"])
        result = absolute_size(u)
        assert result.values[0] == pytest.approx(3.0)
        assert result.values[1] == pytest.approx(0.0)

    def test_non_unit_raises(self):
        with pytest.raises(TypeError):
            absolute_size(42)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:

    def test_single_value_unit_length(self):
        assert len(Unit(0, "npc")) == 1

    def test_equality_same(self):
        a = Unit(1, "cm")
        b = Unit(1, "cm")
        assert a == b

    def test_equality_different_value(self):
        a = Unit(1, "cm")
        b = Unit(2, "cm")
        assert a != b

    def test_equality_different_type(self):
        a = Unit(1, "cm")
        b = Unit(1, "inches")
        assert a != b

    def test_equality_different_length(self):
        a = Unit(1, "cm")
        b = Unit([1, 2], "cm")
        assert a != b

    def test_copy_independent(self):
        u = Unit(1, "cm")
        c = u.copy()
        c._values[0] = 999.0
        assert u.values[0] == pytest.approx(1.0)

    def test_is_absolute_true(self):
        assert Unit(1, "cm").is_absolute() is True
        assert Unit(1, "inches").is_absolute() is True
        assert Unit(1, "mm").is_absolute() is True

    def test_is_absolute_false(self):
        assert Unit(1, "npc").is_absolute() is False
        assert Unit(1, "lines").is_absolute() is False

    def test_add_vectorized_recycling(self):
        """unit(1, 'npc') + unit(1:2, 'inch') should recycle the shorter operand."""
        a = Unit(1, "npc")
        b = Unit([1, 2], "inches")
        c = a + b
        assert len(c) == 2

    def test_sum_builtin_works(self):
        """Python's built-in sum() should work via __radd__(0, ...)."""
        units = [Unit(1, "cm"), Unit(2, "cm"), Unit(3, "cm")]
        total = sum(units)
        assert is_unit(total)
        assert total.values[0] == pytest.approx(6.0)
