"""Tests for grid_py Gpar (graphical parameters) module."""

from __future__ import annotations

import sys
import os

import pytest
import numpy as np

# Ensure grid_py is importable from the sibling directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grid_py"))

from grid_py._gpar import Gpar, get_gpar, _default_gpar, _GPAR_NAMES


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestGparConstruction:
    """Basic construction of Gpar objects."""

    def test_empty_gpar(self):
        gp = Gpar()
        assert gp.params == {}
        assert repr(gp) == "Gpar()"

    def test_single_col(self):
        gp = Gpar(col="red")
        assert gp.get("col") == "red"

    def test_multiple_params(self):
        gp = Gpar(col="blue", lwd=2.0, alpha=0.5)
        assert gp.get("col") == "blue"
        assert gp.get("lwd") == 2.0
        assert gp.get("alpha") == 0.5

    def test_vector_col(self):
        gp = Gpar(col=["red", "blue", "green"])
        assert gp.get("col") == ["red", "blue", "green"]

    def test_fontface_string_plain(self):
        gp = Gpar(fontface="bold")
        assert gp.get("font") == 2

    def test_fontface_int(self):
        gp = Gpar(fontface=3)
        assert gp.get("font") == 3

    def test_fontface_and_font_mutual_exclusion(self):
        with pytest.raises(ValueError, match="must specify only one"):
            Gpar(fontface="bold", font=2)

    def test_fill_param(self):
        gp = Gpar(fill="yellow")
        assert gp.get("fill") == "yellow"

    def test_fontfamily_coerced_to_str(self):
        gp = Gpar(fontfamily="serif")
        assert gp.get("fontfamily") == "serif"

    def test_lineheight_numeric(self):
        gp = Gpar(lineheight=1.5)
        assert gp.get("lineheight") == 1.5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGparValidation:
    """Validation of individual gpar parameters."""

    def test_alpha_valid_range(self):
        gp = Gpar(alpha=0.0)
        assert gp.get("alpha") == 0.0
        gp = Gpar(alpha=1.0)
        assert gp.get("alpha") == 1.0

    def test_alpha_below_zero(self):
        with pytest.raises(ValueError, match="invalid 'alpha'"):
            Gpar(alpha=-0.1)

    def test_alpha_above_one(self):
        with pytest.raises(ValueError, match="invalid 'alpha'"):
            Gpar(alpha=1.1)

    def test_linemitre_valid(self):
        gp = Gpar(linemitre=1.0)
        assert gp.get("linemitre") == 1.0

    def test_linemitre_below_one(self):
        with pytest.raises(ValueError, match="invalid 'linemitre'"):
            Gpar(linemitre=0.5)

    def test_lty_valid_names(self):
        for lty in ("solid", "dashed", "dotted", "dotdash", "longdash", "twodash"):
            gp = Gpar(lty=lty)
            assert gp.get("lty") == lty

    def test_lty_hex_string(self):
        gp = Gpar(lty="44")
        assert gp.get("lty") == "44"

    def test_lty_invalid_string(self):
        with pytest.raises(ValueError, match="invalid line type"):
            Gpar(lty="zigzag")

    def test_lty_numeric(self):
        gp = Gpar(lty=2)
        assert gp.get("lty") == 2

    def test_lineend_valid(self):
        for le in ("round", "butt", "square"):
            gp = Gpar(lineend=le)
            assert gp.get("lineend") == le

    def test_lineend_invalid(self):
        with pytest.raises(ValueError, match="invalid 'lineend'"):
            Gpar(lineend="pointy")

    def test_linejoin_valid(self):
        for lj in ("round", "mitre", "bevel"):
            gp = Gpar(linejoin=lj)
            assert gp.get("linejoin") == lj

    def test_linejoin_invalid(self):
        with pytest.raises(ValueError, match="invalid 'linejoin'"):
            Gpar(linejoin="wavy")

    def test_unknown_param_rejected(self):
        with pytest.raises(TypeError, match="unknown graphical parameter"):
            Gpar(color="red")  # should be 'col', not 'color'

    def test_length_zero_param_rejected(self):
        with pytest.raises(ValueError, match="must not be length 0"):
            Gpar(col=[])

    def test_alpha_non_numeric(self):
        with pytest.raises(TypeError, match="'alpha' must be numeric"):
            Gpar(alpha="half")

    def test_lwd_non_numeric(self):
        with pytest.raises(TypeError, match="'lwd' must be numeric"):
            Gpar(lwd="thick")

    def test_font_non_integer(self):
        with pytest.raises(TypeError, match="'font' must be integer"):
            Gpar(font="bold")

    def test_fontface_invalid_string(self):
        with pytest.raises(ValueError, match="invalid fontface"):
            Gpar(fontface="comic")


# ---------------------------------------------------------------------------
# Subscripting
# ---------------------------------------------------------------------------


class TestGparSubscripting:
    """Gpar __getitem__ (subscripting) behaviour."""

    def test_subscript_scalar(self):
        gp = Gpar(col="red", lwd=2)
        sub = gp[0]
        assert sub.get("col") == "red"
        assert sub.get("lwd") == 2

    def test_subscript_vector(self):
        gp = Gpar(col=["red", "blue", "green"], lwd=1)
        sub = gp[1]
        assert sub.get("col") == "blue"
        assert sub.get("lwd") == 1

    def test_subscript_recycling(self):
        gp = Gpar(col=["red", "blue"], lwd=[1, 2, 3])
        # length is 3 (max), col recycled to ["red","blue","red"]
        sub = gp[2]
        assert sub.get("col") == "red"
        assert sub.get("lwd") == 3

    def test_subscript_negative_index(self):
        gp = Gpar(col=["a", "b", "c"])
        sub = gp[-1]
        assert sub.get("col") == "c"

    def test_subscript_out_of_range(self):
        gp = Gpar(col="red")
        with pytest.raises(IndexError):
            gp[5]

    def test_subscript_empty_gpar(self):
        gp = Gpar()
        sub = gp[0]  # empty gpar returns empty gpar
        assert len(sub) == 0


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


class TestGparLength:
    """Gpar __len__ behaviour."""

    def test_length_empty(self):
        assert len(Gpar()) == 0

    def test_length_scalar(self):
        assert len(Gpar(col="red")) == 1

    def test_length_vector(self):
        assert len(Gpar(col=["red", "blue", "green"])) == 3

    def test_length_mixed(self):
        gp = Gpar(col=["r", "g"], lwd=[1, 2, 3])
        assert len(gp) == 3


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


class TestGparMerge:
    """Gpar _merge (child overrides parent) behaviour."""

    def test_child_overrides_parent(self):
        parent = Gpar(col="black", lwd=1)
        child = Gpar(col="red")
        merged = child._merge(parent)
        assert merged.get("col") == "red"
        assert merged.get("lwd") == 1

    def test_cumulative_alpha(self):
        parent = Gpar(alpha=0.5)
        child = Gpar(alpha=0.5)
        merged = child._merge(parent)
        assert merged.get("alpha") == pytest.approx(0.25)

    def test_cumulative_cex(self):
        parent = Gpar(cex=2.0)
        child = Gpar(cex=1.5)
        merged = child._merge(parent)
        assert merged.get("cex") == pytest.approx(3.0)

    def test_cumulative_lex(self):
        parent = Gpar(lex=2.0)
        child = Gpar(lex=0.5)
        merged = child._merge(parent)
        assert merged.get("lex") == pytest.approx(1.0)

    def test_parent_only_params_inherited(self):
        parent = Gpar(fill="yellow", lty="dashed")
        child = Gpar(col="red")
        merged = child._merge(parent)
        assert merged.get("fill") == "yellow"
        assert merged.get("lty") == "dashed"
        assert merged.get("col") == "red"


# ---------------------------------------------------------------------------
# Default gpar and get_gpar
# ---------------------------------------------------------------------------


class TestDefaultGpar:
    """Default graphical parameters and get_gpar()."""

    def test_default_gpar_fontsize(self):
        d = _default_gpar()
        assert d.get("fontsize") == 12

    def test_default_gpar_col(self):
        d = _default_gpar()
        assert d.get("col") == "black"

    def test_default_gpar_fill(self):
        d = _default_gpar()
        assert d.get("fill") == "transparent"

    def test_default_gpar_lty(self):
        d = _default_gpar()
        assert d.get("lty") == "solid"

    def test_default_gpar_alpha(self):
        d = _default_gpar()
        assert d.get("alpha") == 1.0

    def test_default_gpar_linemitre(self):
        d = _default_gpar()
        assert d.get("linemitre") == 10.0

    def test_get_gpar_all(self):
        gp = get_gpar()
        assert "col" in gp
        assert "fontsize" in gp

    def test_get_gpar_subset(self):
        gp = get_gpar(names=["col", "lwd"])
        assert gp.get("col") == "black"
        assert gp.get("lwd") == 1.0
        # Other params should not be present.
        assert gp.get("fontsize") is None

    def test_get_gpar_invalid_name(self):
        with pytest.raises(ValueError, match="invalid gpar name"):
            get_gpar(names=["col", "bogus"])


# ---------------------------------------------------------------------------
# Contains / names / equality / repr
# ---------------------------------------------------------------------------


class TestGparMisc:
    """Miscellaneous Gpar protocol tests."""

    def test_contains(self):
        gp = Gpar(col="red")
        assert "col" in gp
        assert "lwd" not in gp

    def test_names(self):
        gp = Gpar(col="red", lwd=2)
        assert set(gp.names()) == {"col", "lwd"}

    def test_equality(self):
        assert Gpar(col="red") == Gpar(col="red")

    def test_inequality(self):
        assert Gpar(col="red") != Gpar(col="blue")

    def test_repr_roundtrip_style(self):
        gp = Gpar(col="red", lwd=2.0)
        r = repr(gp)
        assert "Gpar(" in r
        assert "col='red'" in r

    def test_str_same_as_repr(self):
        gp = Gpar(col="red")
        assert str(gp) == repr(gp)

    def test_params_returns_copy(self):
        gp = Gpar(col="red")
        p = gp.params
        p["col"] = "blue"
        # Original should be unchanged.
        assert gp.get("col") == "red"

    def test_none_value_skipped(self):
        gp = Gpar(col=None)
        assert "col" not in gp
