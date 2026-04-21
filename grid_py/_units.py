"""
Unit system for grid_py -- Python port of R's grid unit infrastructure.

This module provides the fundamental ``Unit`` class and associated helper
functions that mirror R's ``grid::unit()`` family.  A ``Unit`` stores one or
more scalar values together with their unit types and optional reference data
(used by contextual units such as ``"strwidth"`` or ``"grobwidth"``).

Arithmetic on units follows R semantics:

* ``unit + unit`` produces a compound *sum* unit.
* ``scalar * unit`` (or ``unit * scalar``) scales the numeric values.
* ``-unit`` negates the numeric values.
* ``unit / scalar`` divides the numeric values.

Absolute-unit conversions (cm, inches, mm, points, picas, ...) are carried
out eagerly.  Context-dependent conversions (npc, native, lines, ...) are
deferred -- the unit is returned unchanged when no viewport context is
available.

Notes
-----
The module is intentionally self-contained so that it can be imported very
early during package initialisation without circular-dependency issues.
"""

from __future__ import annotations

import copy
import math
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "Unit",
    "is_unit",
    "unit_type",
    "unit_c",
    "unit_length",
    "unit_pmax",
    "unit_pmin",
    "unit_psum",
    "unit_rep",
    "string_width",
    "string_height",
    "string_ascent",
    "string_descent",
    "absolute_size",
    "convert_unit",
    "convert_x",
    "convert_y",
    "convert_width",
    "convert_height",
    "device_loc",
    "device_dim",
    "convert_theta",
    "unit_summary_min",
    "unit_summary_max",
    "unit_summary_sum",
]

# ---------------------------------------------------------------------------
# Valid unit type strings
# ---------------------------------------------------------------------------

VALID_UNIT_TYPES: Tuple[str, ...] = (
    "npc",
    "cm",
    "inches",
    "mm",
    "points",
    "picas",
    "bigpts",
    "dida",
    "cicero",
    "scaledpts",
    "lines",
    "char",
    "native",
    "null",
    "snpc",
    "strwidth",
    "strheight",
    "strdescent",
    "strascent",
    "vplayoutwidth",
    "vplayoutheight",
    "grobx",
    "groby",
    "grobwidth",
    "grobheight",
    "grobascent",
    "grobdescent",
    "mylines",
    "mychar",
    "mystrwidth",
    "mystrheight",
    "sum",
    "min",
    "max",
)

# Convenient lookup set for O(1) membership tests
_VALID_UNIT_SET: frozenset = frozenset(VALID_UNIT_TYPES)

# Aliases accepted on input (mapped to canonical names)
_UNIT_ALIASES: Dict[str, str] = {
    "in": "inches",
    "inch": "inches",
    "centimetre": "cm",
    "centimetres": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "millimetre": "mm",
    "millimetres": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "point": "points",
    "pt": "points",
    "line": "lines",
}

# ---------------------------------------------------------------------------
# Absolute-unit conversion factors  (everything relative to inches)
# ---------------------------------------------------------------------------
# Reference:
#   1 inch  = 2.54 cm = 25.4 mm = 72.27 pt (TeX point)
#   1 pica  = 12 pt
#   1 bigpt = 1/72 inch  (PostScript point)
#   1 dida  = 1238/1157 pt
#   1 cicero = 12 dida
#   1 scaledpt = 1/65536 pt

_INCHES_PER: Dict[str, float] = {
    "inches": 1.0,
    "cm": 1.0 / 2.54,
    "mm": 1.0 / 25.4,
    "points": 1.0 / 72.27,
    "picas": 12.0 / 72.27,
    "bigpts": 1.0 / 72.0,
    "dida": (1238.0 / 1157.0) / 72.27,
    "cicero": 12.0 * (1238.0 / 1157.0) / 72.27,
    "scaledpts": 1.0 / (72.27 * 65536.0),
}

# Set of unit types that can be converted without a viewport context
_ABSOLUTE_UNIT_TYPES: frozenset = frozenset(_INCHES_PER.keys())

# Unit types resolved by measuring a string or querying a grob
_STR_METRIC_TYPES: frozenset = frozenset(
    {"strwidth", "strheight", "strascent", "strdescent"}
)
_GROB_METRIC_TYPES: frozenset = frozenset(
    {"grobwidth", "grobheight", "grobascent", "grobdescent"}
)


def _eval_str_metric(unit_type: str, data: Any, scale: float = 1.0) -> float:
    """Evaluate a string-metric unit to an inch value.

    Mirrors R's ``GEStrWidth`` / ``GEStrHeight`` (src/main/engine.c), which
    back ``stringWidth`` / ``stringHeight`` for text units:

      - split on ``\\n`` into lines,
      - width  = max(per-line widths),
      - height = ink(first line) + (n - 1) × cex × lineheight × fontsize × 1.2 / 72

    Uses the current viewport's gpar (fontsize, cex, lineheight) to match
    R's behaviour where ``stringWidth`` inherits typography from the
    enclosing gpar context.

    Uses a lazy import of :func:`._size.calc_string_metric` to avoid
    circular dependencies (``_size`` imports ``Unit`` from this module).

    Parameters
    ----------
    unit_type : str
        One of ``"strwidth"``, ``"strheight"``, ``"strascent"``,
        ``"strdescent"``.
    data : object
        The string stored as auxiliary data in the unit.
    scale : float
        Multiplicative factor (the unit's numeric value).

    Returns
    -------
    float
        Measurement in inches, scaled by *scale*.
    """
    from ._size import calc_string_metric  # lazy – avoids circular import

    text = str(data) if data is not None else ""

    # Inherit fontsize / cex / lineheight from the current viewport gpar
    # stack, matching R's ``stringWidth`` which uses the enclosing gpar.
    try:
        from ._gpar import get_gpar
        gp = get_gpar()
    except Exception:
        gp = None

    fontsize = 12.0
    cex = 1.0
    lineheight = 1.2
    if gp is not None:
        fs = gp.get("fontsize", None)
        if fs is not None:
            fontsize = float(fs[0] if isinstance(fs, (list, tuple)) else fs)
        cx = gp.get("cex", None)
        if cx is not None:
            cex = float(cx[0] if isinstance(cx, (list, tuple)) else cx)
        lh = gp.get("lineheight", None)
        if lh is not None:
            lineheight = float(lh[0] if isinstance(lh, (list, tuple)) else lh)

    lines = text.split("\n") if text else [""]
    n = len(lines)
    m0 = calc_string_metric(lines[0], gp=gp)

    if unit_type == "strwidth":
        w = max(calc_string_metric(ln, gp=gp)["width"] for ln in lines)
        return w * scale
    elif unit_type == "strheight":
        ink_first = m0["ascent"] + m0["descent"]
        inter_line_gap = cex * lineheight * fontsize * 1.2 / 72.0
        return (ink_first + (n - 1) * inter_line_gap) * scale
    elif unit_type == "strascent":
        return m0["ascent"] * scale
    elif unit_type == "strdescent":
        return m0["descent"] * scale
    return 0.0  # pragma: no cover


def _eval_grob_metric(unit_type: str, grob: Any) -> Optional["Unit"]:
    """Evaluate a grob-metric unit by calling the appropriate detail dispatcher.

    Uses a lazy import of dispatchers from :mod:`._size` to avoid
    circular dependencies.

    Parameters
    ----------
    unit_type : str
        One of ``"grobwidth"``, ``"grobheight"``, ``"grobascent"``,
        ``"grobdescent"``.
    grob : object
        The grob stored as auxiliary data in the unit.

    Returns
    -------
    Unit or None
        The measured dimension as a :class:`Unit`, or ``None`` if the
        grob is ``None``.
    """
    from ._size import (  # lazy – avoids circular import
        width_details,
        height_details,
        ascent_details,
        descent_details,
    )

    if grob is None:
        return None
    _dispatch = {
        "grobwidth": width_details,
        "grobheight": height_details,
        "grobascent": ascent_details,
        "grobdescent": descent_details,
    }
    return _dispatch[unit_type](grob)


def _try_resolve_with_renderer(
    x: "Unit",
    i: int,
    src_unit: str,
    target: str,
    axis: str,
    type_: str,
) -> Optional[float]:
    """Resolve a context-dependent unit via the active renderer.

    Implements R's ``L_convert`` two-stage pipeline (grid.c:1384-1575):
      Stage 1: any unit → inches  (via renderer._resolve_to_inches_idx)
      Stage 2: inches → target    (via _vp_calc inverse transforms)

    Returns the converted value in *target* units, or ``None`` if no
    renderer is available.
    """
    from ._state import get_state
    from ._vp_calc import (
        _transform_xy_from_inches,
        _transform_wh_from_inches,
        _transform_xy_to_npc,
        _transform_wh_to_npc,
        _transform_xy_from_npc,
        _transform_wh_from_npc,
    )

    state = get_state()
    renderer = state.get_renderer()

    if renderer is None or not hasattr(renderer, "_resolve_to_inches_idx"):
        return None

    # Build a single-element Unit for the source
    elem = Unit(x._values[i], src_unit, data=x._data[i])
    is_dim = type_ in ("dimension",)

    # Get viewport context from renderer
    vtr = renderer._vp_transform_stack[-1]
    vpc = vtr.vpc

    # Determine axis parameters (R grid.c:1426-1427)
    #   axis encoding: 0=x-loc, 1=y-loc, 2=x-dim, 3=y-dim
    if axis == "x":
        scalemin, scalemax = vpc.xscalemin, vpc.xscalemax
        this_cm = vtr.width_cm
        other_cm = vtr.height_cm
    else:
        scalemin, scalemax = vpc.yscalemin, vpc.yscalemax
        this_cm = vtr.height_cm
        other_cm = vtr.width_cm

    # R grid.c:1438-1444 -- special case: relative-to-relative with zero dim
    from_is_relative = src_unit in ("native", "npc")
    to_is_relative = target in ("native", "npc")
    rel_convert = (from_is_relative and to_is_relative
                   and this_cm < 1e-6)

    # Stage 1: convert source → inches (or NPC for relConvert)
    if rel_convert:
        if is_dim:
            stage1 = _transform_wh_to_npc(
                float(x._values[i]), src_unit, scalemin, scalemax)
        else:
            stage1 = _transform_xy_to_npc(
                float(x._values[i]), src_unit, scalemin, scalemax)
    else:
        # Use renderer's full pipeline to get inches
        gp = state.get_gpar()
        stage1 = renderer._resolve_to_inches_idx(elem, 0, axis, is_dim, gp)

    # Stage 2: inches (or NPC) → target unit
    if rel_convert:
        if is_dim:
            return _transform_wh_from_npc(stage1, target, scalemin, scalemax)
        else:
            return _transform_xy_from_npc(stage1, target, scalemin, scalemax)
    else:
        fontsize, cex, lineheight = renderer._gpar_font_params(state.get_gpar())
        scale = renderer._get_scale()
        if is_dim:
            return _transform_wh_from_inches(
                stage1, target, scalemin, scalemax,
                fontsize, cex, lineheight,
                this_cm, other_cm, scale,
            )
        else:
            return _transform_xy_from_inches(
                stage1, target, scalemin, scalemax,
                fontsize, cex, lineheight,
                this_cm, other_cm, scale,
            )


def _resolve_alias(unit_str: str) -> str:
    """Return the canonical unit-type string, resolving common aliases.

    Parameters
    ----------
    unit_str : str
        Raw unit name (e.g. ``"in"``, ``"pt"``, ``"centimeters"``).

    Returns
    -------
    str
        Canonical unit name (e.g. ``"inches"``, ``"points"``, ``"cm"``).

    Raises
    ------
    ValueError
        If *unit_str* is not a recognised unit type or alias.
    """
    low = unit_str.strip().lower()
    if low in _VALID_UNIT_SET:
        return low
    if low in _UNIT_ALIASES:
        return _UNIT_ALIASES[low]
    raise ValueError(
        f"Unknown unit type {unit_str!r}. "
        f"Valid types: {', '.join(sorted(VALID_UNIT_TYPES))}"
    )


# ---------------------------------------------------------------------------
# The Unit class
# ---------------------------------------------------------------------------


class Unit:
    """Representation of one or more grid unit values.

    A ``Unit`` bundles numeric *values* with their *unit types* and optional
    *data* references (e.g. a string for ``"strwidth"`` or a grob for
    ``"grobwidth"``).  It mirrors R's ``grid::unit`` objects.

    Parameters
    ----------
    x : float, int, Sequence[float], or np.ndarray
        Numeric value(s).  Scalars are promoted to length-1 arrays.
    units : str or Sequence[str]
        Unit type(s).  A single string is recycled to match the length of *x*.
    data : Any or Sequence[Any], optional
        Auxiliary data attached to each element (used by contextual units
        such as ``"strwidth"``).  ``None`` entries are allowed.

    Raises
    ------
    ValueError
        If *x* or *units* are empty, or if *units* contains an unknown type.

    Examples
    --------
    >>> u = Unit(1, "cm")
    >>> u
    Unit([1.0], ['cm'])
    >>> Unit([0.5, 1.0], ["npc", "cm"])
    Unit([0.5, 1.0], ['npc', 'cm'])
    """

    # ---- internal slots ----------------------------------------------------
    __slots__ = ("_values", "_units", "_data", "_is_absolute")

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        x: Union[float, int, Sequence[float], np.ndarray],
        units: Union[str, Sequence[str]],
        data: Optional[Union[Any, Sequence[Any]]] = None,
    ) -> None:
        # If *x* is already a Unit, return a shallow copy (mirrors R behaviour
        # where ``unit(u)`` simply returns *u*).
        if isinstance(x, Unit):
            self._values = x._values.copy()
            self._units = list(x._units)
            self._data = list(x._data)
            self._is_absolute = x._is_absolute
            return

        # Coerce values --------------------------------------------------
        if isinstance(x, np.ndarray):
            vals = x.astype(np.float64).ravel()
        elif isinstance(x, (list, tuple)):
            vals = np.asarray(x, dtype=np.float64)
        else:
            vals = np.asarray([x], dtype=np.float64)

        if vals.size == 0:
            raise ValueError("'x' must have length > 0")

        # Coerce units ---------------------------------------------------
        if isinstance(units, str):
            resolved = _resolve_alias(units)
            unit_list = [resolved] * len(vals)
        else:
            unit_list = [_resolve_alias(u) for u in units]
            if len(unit_list) == 0:
                raise ValueError("'units' must have length > 0")
            # Recycle to match length of vals
            if len(unit_list) < len(vals):
                reps = math.ceil(len(vals) / len(unit_list))
                unit_list = (unit_list * reps)[: len(vals)]
            elif len(unit_list) > len(vals):
                # Recycle values to match units length
                reps = math.ceil(len(unit_list) / len(vals))
                vals = np.tile(vals, reps)[: len(unit_list)]

        # Coerce data ----------------------------------------------------
        if data is None:
            data_list: List[Any] = [None] * len(unit_list)
        elif isinstance(data, (list, tuple)):
            data_list = list(data)
            if len(data_list) < len(unit_list):
                reps = math.ceil(len(unit_list) / max(len(data_list), 1))
                data_list = (data_list * reps)[: len(unit_list)]
        else:
            data_list = [data] * len(unit_list)

        self._values: np.ndarray = vals
        self._units: List[str] = unit_list
        self._data: List[Any] = data_list
        self._is_absolute: bool = all(u in _ABSOLUTE_UNIT_TYPES for u in unit_list)

    # ---------------------------------------------------------------- properties
    @property
    def values(self) -> np.ndarray:
        """Numeric values as a 1-D ``numpy.float64`` array."""
        return self._values

    @property
    def units_list(self) -> List[str]:
        """List of unit-type strings (one per element)."""
        return self._units

    @property
    def data(self) -> List[Any]:
        """Auxiliary data list (one entry per element; may contain ``None``)."""
        return self._data

    # ---------------------------------------------------------------- length
    def __len__(self) -> int:
        return len(self._values)

    # ---------------------------------------------------------------- repr / str
    def __repr__(self) -> str:
        vals = [float(v) for v in self._values]
        return f"Unit({vals}, {self._units})"

    def __str__(self) -> str:
        return self.as_character()

    def as_character(self) -> str:
        """Return an R-compatible character representation.

        Returns
        -------
        str
            A string such as ``"1cm"`` or ``"0.5npc+1cm"`` when the unit
            contains a compound *sum* / *min* / *max* type.

        Examples
        --------
        >>> Unit(2.5, "cm").as_character()
        '2.5cm'
        """
        parts: List[str] = []
        for i in range(len(self._values)):
            parts.append(self._desc_element(i))
        return ", ".join(parts)

    def _desc_element(self, idx: int) -> str:
        """Format a single element as an R-style string."""
        val = self._values[idx]
        utype = self._units[idx]
        d = self._data[idx]

        if utype in ("sum", "min", "max"):
            # Compound unit -- data should be a Unit
            if isinstance(d, Unit):
                inner = ", ".join(d._desc_element(j) for j in range(len(d)))
                prefix = "" if val == 1.0 else f"{val}*"
                return f"{prefix}{utype}({inner})"
            # Fallback
            return f"{val}{utype}"

        # String-based units include data in the representation
        if utype in (
            "strwidth",
            "strheight",
            "strascent",
            "strdescent",
            "mystrwidth",
            "mystrheight",
        ) and d is not None:
            return f"{val}{utype}({d!r})"

        # Grob-based units
        if utype in (
            "grobx",
            "groby",
            "grobwidth",
            "grobheight",
            "grobascent",
            "grobdescent",
        ) and d is not None:
            return f"{val}{utype}({d!r})"

        return f"{val}{utype}"

    # ---------------------------------------------------------------- indexing
    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> "Unit":
        """Return a new ``Unit`` containing the selected element(s).

        Parameters
        ----------
        index : int, slice, or sequence of int
            Element selector.

        Returns
        -------
        Unit
            A new unit with the selected elements.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if isinstance(index, (int, np.integer)):
            if index < 0:
                index = len(self) + index
            if index < 0 or index >= len(self):
                raise IndexError(
                    f"index {index} is out of bounds for Unit of length {len(self)}"
                )
            new = Unit.__new__(Unit)
            new._values = self._values[index : index + 1].copy()
            new._units = [self._units[index]]
            new._data = [self._data[index]]
            new._is_absolute = self._units[index] in _ABSOLUTE_UNIT_TYPES
            return new

        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
        else:
            indices = list(index)

        vals = self._values[index] if isinstance(index, slice) else self._values[list(indices)]
        u_list = [self._units[i] for i in indices]
        d_list = [self._data[i] for i in indices]

        new = Unit.__new__(Unit)
        new._values = vals.copy() if isinstance(vals, np.ndarray) else np.asarray(vals, dtype=np.float64)
        new._units = u_list
        new._data = d_list
        new._is_absolute = all(u in _ABSOLUTE_UNIT_TYPES for u in u_list)
        return new

    def __setitem__(
        self, index: Union[int, slice], value: "Unit"
    ) -> None:
        """Set element(s) of this unit in-place.

        Parameters
        ----------
        index : int or slice
            Element selector.
        value : Unit
            Replacement unit value(s).
        """
        if not isinstance(value, Unit):
            raise TypeError("replacement value must be a Unit")

        if isinstance(index, (int, np.integer)):
            if index < 0:
                index = len(self) + index
            self._values[index] = value._values[0]
            self._units[index] = value._units[0]
            self._data[index] = value._data[0]
        elif isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            for j, i in enumerate(indices):
                src = j % len(value)
                self._values[i] = value._values[src]
                self._units[i] = value._units[src]
                self._data[i] = value._data[src]
        else:
            raise TypeError(f"unsupported index type {type(index)}")

        self._is_absolute = all(u in _ABSOLUTE_UNIT_TYPES for u in self._units)

    # ================================================================
    #  Arithmetic operators
    # ================================================================

    # ---- addition: unit + unit -> compound sum -------------------------
    def __add__(self, other: Any) -> "Unit":
        if isinstance(other, Unit):
            return _make_compound("sum", self, other)
        return NotImplemented

    def __radd__(self, other: Any) -> "Unit":
        if isinstance(other, Unit):
            return _make_compound("sum", other, self)
        if other == 0:
            # Supports sum() on iterables of Units
            return self.copy()
        return NotImplemented

    # ---- subtraction: unit - unit -> compound sum (with negated rhs) ---
    def __sub__(self, other: Any) -> "Unit":
        if isinstance(other, Unit):
            return _make_compound("sum", self, -other)
        return NotImplemented

    def __rsub__(self, other: Any) -> "Unit":
        if isinstance(other, Unit):
            return _make_compound("sum", other, -self)
        return NotImplemented

    # ---- negation ------------------------------------------------------
    def __neg__(self) -> "Unit":
        new = self.copy()
        new._values = -new._values
        return new

    def __pos__(self) -> "Unit":
        return self.copy()

    # ---- multiplication: scalar * unit or unit * scalar ----------------
    def __mul__(self, other: Any) -> "Unit":
        if isinstance(other, Number) and not isinstance(other, bool):
            new = self.copy()
            new._values = new._values * float(other)
            return new
        if isinstance(other, Unit):
            raise TypeError("Cannot multiply two Unit objects; one operand must be numeric")
        return NotImplemented

    def __rmul__(self, other: Any) -> "Unit":
        if isinstance(other, Number) and not isinstance(other, bool):
            return self.__mul__(other)
        return NotImplemented

    # ---- division: unit / scalar ---------------------------------------
    def __truediv__(self, other: Any) -> "Unit":
        if isinstance(other, Number) and not isinstance(other, bool):
            if float(other) == 0.0:
                raise ZeroDivisionError("division by zero")
            new = self.copy()
            new._values = new._values / float(other)
            return new
        if isinstance(other, Unit):
            raise TypeError("Cannot divide by a Unit object")
        return NotImplemented

    def __rtruediv__(self, other: Any) -> "Unit":
        raise TypeError("Cannot divide by a Unit object")

    # ---- equality (element-wise, mainly for testing) -------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        if len(self) != len(other):
            return False
        return (
            np.allclose(self._values, other._values)
            and self._units == other._units
        )

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented  # type: ignore[return-value]
        return not eq

    # ================================================================
    #  Helpers
    # ================================================================

    def copy(self) -> "Unit":
        """Return a shallow copy of this unit.

        Returns
        -------
        Unit
            Independent copy sharing no mutable state with the original.
        """
        new = Unit.__new__(Unit)
        new._values = self._values.copy()
        new._units = list(self._units)
        new._data = list(self._data)
        new._is_absolute = self._is_absolute
        return new

    def is_absolute(self) -> bool:
        """Return ``True`` if every element is an absolute (physical) unit.

        Returns
        -------
        bool
        """
        return self._is_absolute

    # Allow hashing to fail (mutable container)
    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Compound-unit helper
# ---------------------------------------------------------------------------


def _make_compound(op: str, lhs: Unit, rhs: Unit) -> Unit:
    """Create a compound unit (*sum*, *min*, or *max*) from two operands.

    If both operands share identical simple absolute unit types the operation
    is performed eagerly (e.g. ``1cm + 2cm -> 3cm``).

    Parameters
    ----------
    op : str
        One of ``"sum"``, ``"min"``, ``"max"``.
    lhs : Unit
        Left-hand operand.
    rhs : Unit
        Right-hand operand.

    Returns
    -------
    Unit
        Resulting (possibly compound) unit.
    """
    # Fast path: identical simple unit types -- compute eagerly
    if (
        len(set(lhs._units)) == 1
        and len(set(rhs._units)) == 1
        and lhs._units[0] == rhs._units[0]
        and lhs._units[0] not in ("sum", "min", "max")
    ):
        utype = lhs._units[0]
        # Recycle to common length
        n = max(len(lhs), len(rhs))
        lv = np.resize(lhs._values, n)
        rv = np.resize(rhs._values, n)
        if op == "sum":
            vals = lv + rv
        elif op == "min":
            vals = np.minimum(lv, rv)
        else:
            vals = np.maximum(lv, rv)
        new = Unit.__new__(Unit)
        new._values = vals
        new._units = [utype] * n
        new._data = [None] * n
        new._is_absolute = utype in _ABSOLUTE_UNIT_TYPES
        return new

    # General path: build a compound unit for each parallel pair
    n = max(len(lhs), len(rhs))
    compound_vals = np.ones(n, dtype=np.float64)
    compound_units: List[str] = [op] * n
    compound_data: List[Any] = []

    for i in range(n):
        li = i % len(lhs)
        ri = i % len(rhs)
        pair = unit_c(lhs[li], rhs[ri])
        compound_data.append(pair)

    new = Unit.__new__(Unit)
    new._values = compound_vals
    new._units = compound_units
    new._data = compound_data
    new._is_absolute = False
    return new


# ===================================================================
# Module-level helper functions
# ===================================================================


def is_unit(x: Any) -> bool:
    """Check whether *x* is a ``Unit`` instance.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a ``Unit``.

    Examples
    --------
    >>> is_unit(Unit(1, "cm"))
    True
    >>> is_unit(42)
    False
    """
    return isinstance(x, Unit)


def unit_type(x: Unit, recurse: bool = False) -> Union[str, List[str], List[Any]]:
    """Return the unit type(s) of *x*.

    Port of R ``unitType()`` (unit.R:197-226).

    Parameters
    ----------
    x : Unit
        A unit object.
    recurse : bool
        If ``True``, compound units (sum/min/max) are recursively
        expanded to reveal the underlying unit types.  Returns a list
        of lists for compound elements.  Default ``False``.

    Returns
    -------
    str or list
        When *recurse* is ``False``: a single string (length 1) or
        list of strings.
        When *recurse* is ``True``: a list where compound elements
        are themselves lists of their constituent unit types.

    Raises
    ------
    TypeError
        If *x* is not a ``Unit``.

    Examples
    --------
    >>> unit_type(Unit(1, "cm"))
    'cm'
    >>> unit_type(Unit([1, 2], ["cm", "inches"]))
    ['cm', 'inches']
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")

    if not recurse:
        if len(x) == 1:
            return x._units[0]
        return list(x._units)

    # recurse=True: expand compound units (R unit.R:211-224)
    result = []
    for i in range(len(x)):
        utype = x._units[i]
        if utype in ("sum", "min", "max"):
            # Compound unit: recurse into the child Unit stored in _data
            child = x._data[i]
            if isinstance(child, Unit):
                result.append(unit_type(child, recurse=True))
            else:
                result.append(utype)
        else:
            result.append(utype)
    return result


def unit_c(*args: Unit) -> Unit:
    """Concatenate one or more ``Unit`` objects into a single ``Unit``.

    Parameters
    ----------
    *args : Unit
        Units to concatenate.

    Returns
    -------
    Unit
        A new unit containing all elements in order.

    Raises
    ------
    TypeError
        If any argument is not a ``Unit``.
    ValueError
        If no arguments are provided.

    Examples
    --------
    >>> unit_c(Unit(1, "cm"), Unit(2, "inches"))
    Unit([1.0, 2.0], ['cm', 'inches'])
    """
    if len(args) == 0:
        raise ValueError("unit_c requires at least one argument")

    all_vals: List[np.ndarray] = []
    all_units: List[str] = []
    all_data: List[Any] = []

    for a in args:
        if not isinstance(a, Unit):
            raise TypeError(f"All arguments must be Unit objects, got {type(a)}")
        all_vals.append(a._values)
        all_units.extend(a._units)
        all_data.extend(a._data)

    new = Unit.__new__(Unit)
    new._values = np.concatenate(all_vals)
    new._units = all_units
    new._data = all_data
    new._is_absolute = all(u in _ABSOLUTE_UNIT_TYPES for u in all_units)
    return new


def unit_length(x: Unit) -> int:
    """Return the number of elements in a ``Unit``.

    Parameters
    ----------
    x : Unit
        A unit object.

    Returns
    -------
    int
        Number of unit values.

    Examples
    --------
    >>> unit_length(Unit([1, 2, 3], "cm"))
    3
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")
    return len(x)


def unit_pmax(*args: Unit) -> Unit:
    """Parallel (element-wise) maximum of units.

    Parameters
    ----------
    *args : Unit
        Two or more units of the same length.

    Returns
    -------
    Unit
        Element-wise maximum (compound *max* units when types differ).

    Examples
    --------
    >>> unit_pmax(Unit([1, 4], "cm"), Unit([3, 2], "cm"))
    Unit([3.0, 4.0], ['cm', 'cm'])
    """
    return _parallel_op("max", *args)


def unit_pmin(*args: Unit) -> Unit:
    """Parallel (element-wise) minimum of units.

    Parameters
    ----------
    *args : Unit
        Two or more units of the same length.

    Returns
    -------
    Unit
        Element-wise minimum (compound *min* units when types differ).

    Examples
    --------
    >>> unit_pmin(Unit([1, 4], "cm"), Unit([3, 2], "cm"))
    Unit([1.0, 2.0], ['cm', 'cm'])
    """
    return _parallel_op("min", *args)


def unit_psum(*args: Unit) -> Unit:
    """Parallel (element-wise) sum of units.

    Parameters
    ----------
    *args : Unit
        Two or more units of the same length.

    Returns
    -------
    Unit
        Element-wise sum (compound *sum* units when types differ).

    Examples
    --------
    >>> unit_psum(Unit([1, 2], "cm"), Unit([3, 4], "cm"))
    Unit([4.0, 6.0], ['cm', 'cm'])
    """
    return _parallel_op("sum", *args)


def _parallel_op(op: str, *args: Unit) -> Unit:
    """Internal implementation for parallel min / max / sum.

    Parameters
    ----------
    op : str
        ``"sum"``, ``"min"``, or ``"max"``.
    *args : Unit
        Units to combine element-wise.

    Returns
    -------
    Unit
    """
    if len(args) == 0:
        raise ValueError(f"unit_p{op} requires at least one argument")
    if len(args) == 1:
        return args[0].copy()

    result = args[0]
    for a in args[1:]:
        result = _make_compound(op, result, a)
    return result


def unit_rep(
    x: Unit,
    times: int = 1,
    length_out: Optional[int] = None,
    each: int = 1,
) -> Unit:
    """Repeat a ``Unit`` object.

    Port of R ``rep.unit()`` (unit.R:539-542).  Mirrors the semantics
    of R's ``rep(x, times, length.out, each)``.

    Parameters
    ----------
    x : Unit
        The unit to repeat.
    times : int
        Number of times to repeat the (possibly each-expanded) unit.
    length_out : int or None
        If given, truncate or recycle the result to this length.
    each : int
        Replicate each element *each* times before tiling.

    Returns
    -------
    Unit
        A unit whose elements are *x* repeated.

    Examples
    --------
    >>> unit_rep(Unit(1, "cm"), 3)
    Unit([1.0, 1.0, 1.0], ['cm', 'cm', 'cm'])
    >>> unit_rep(Unit([1, 2], "cm"), each=2)
    Unit([1.0, 1.0, 2.0, 2.0], ['cm', 'cm', 'cm', 'cm'])
    >>> unit_rep(Unit([1, 2, 3], "cm"), length_out=5)
    Unit([1.0, 2.0, 3.0, 1.0, 2.0], ['cm', 'cm', 'cm', 'cm', 'cm'])
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")

    # Build index vector mirroring R's rep(seq_along(x), times, length.out, each)
    n = len(x)
    base = list(range(n))

    # Apply each: replicate each element
    if each > 1:
        base = [i for i in base for _ in range(each)]

    # Apply times: tile the whole sequence (times=0 → empty)
    if times == 0:
        base = []
    elif times > 1:
        base = base * times

    # Apply length_out: truncate or recycle
    if length_out is not None:
        if length_out <= 0:
            base = []
        elif len(base) == 0:
            # Recycle from original if times made it empty
            base = list(range(n))
            full_cycles = length_out // len(base)
            remainder = length_out % len(base)
            base = base * full_cycles + base[:remainder]
        else:
            full_cycles = length_out // len(base)
            remainder = length_out % len(base)
            base = base * full_cycles + base[:remainder]

    if len(base) == 0:
        # Return empty Unit
        new = Unit.__new__(Unit)
        new._values = np.array([], dtype=np.float64)
        new._units = []
        new._data = []
        new._is_absolute = False
        return new

    return x[base]


# ===================================================================
# Summary.unit: min / max / sum  (port of R unit.R:300-347)
# ===================================================================


def unit_summary_min(*args: Unit) -> Unit:
    """Return the minimum across all elements of all input units.

    Port of R ``Summary.unit`` for ``min`` (unit.R:300-347).
    Unlike :func:`unit_pmin` (element-wise), this returns a single
    scalar unit representing the global minimum.

    Parameters
    ----------
    *args : Unit
        One or more unit objects.

    Returns
    -------
    Unit
        A single-element unit with the minimum value.

    Examples
    --------
    >>> unit_summary_min(Unit([3, 1, 4], "cm"))
    Unit([1.0], ['cm'])
    """
    return _summary_op("min", *args)


def unit_summary_max(*args: Unit) -> Unit:
    """Return the maximum across all elements of all input units.

    Port of R ``Summary.unit`` for ``max`` (unit.R:300-347).

    Parameters
    ----------
    *args : Unit
        One or more unit objects.

    Returns
    -------
    Unit
        A single-element unit with the maximum value.

    Examples
    --------
    >>> unit_summary_max(Unit([3, 1, 4], "cm"))
    Unit([4.0], ['cm'])
    """
    return _summary_op("max", *args)


def unit_summary_sum(*args: Unit) -> Unit:
    """Return the sum of all elements of all input units.

    Port of R ``Summary.unit`` for ``sum`` (unit.R:300-347).

    Parameters
    ----------
    *args : Unit
        One or more unit objects.

    Returns
    -------
    Unit
        A single-element unit with the total sum.

    Examples
    --------
    >>> unit_summary_sum(Unit([1, 2, 3], "cm"))
    Unit([6.0], ['cm'])
    """
    return _summary_op("sum", *args)


def _summary_op(op: str, *args: Unit) -> Unit:
    """Internal implementation for Summary.unit (min/max/sum).

    Port of R ``Summary.unit`` (unit.R:300-347).

    Optimisation: if all elements across all inputs share the same
    simple unit type, the operation is applied directly on the numeric
    values.  Otherwise, a compound unit is created.
    """
    if len(args) == 0:
        raise ValueError(f"unit {op} requires at least one argument")

    # Filter out None args (R: units[!vapply(units, is.null, ...)])
    units = [a for a in args if a is not None and isinstance(a, Unit)]
    if len(units) == 0:
        raise ValueError(f"unit {op} requires at least one Unit argument")

    # Concatenate all elements
    combined = unit_c(*units)

    # Optimisation: identical simple unit types (R unit.R:308-320)
    all_types = set(combined._units)
    if len(all_types) == 1 and list(all_types)[0] not in ("sum", "min", "max"):
        utype = combined._units[0]
        vals = combined._values
        if op == "sum":
            result_val = float(np.sum(vals))
        elif op == "min":
            result_val = float(np.min(vals))
        else:  # max
            result_val = float(np.max(vals))
        return Unit(result_val, utype)

    # General case: create compound unit (R unit.R:321-347)
    # The compound wraps all elements under a single min/max/sum operation
    op_code = {"sum": "sum", "min": "min", "max": "max"}[op]
    return Unit(1.0, op_code, data=combined)


# ===================================================================
# String-metric convenience constructors
# ===================================================================


def string_width(string: Union[str, Sequence[str]]) -> Unit:
    """Create a ``"strwidth"`` unit for the given string(s).

    Parameters
    ----------
    string : str or sequence of str
        The string(s) whose rendered width the unit represents.

    Returns
    -------
    Unit
        A unit of type ``"strwidth"`` with value 1 for each string.

    Examples
    --------
    >>> string_width("hello")
    Unit([1.0], ['strwidth'])
    """
    if isinstance(string, str):
        strings = [string]
    else:
        strings = list(string)
    n = len(strings)
    return Unit(np.ones(n), ["strwidth"] * n, data=strings)


def string_height(string: Union[str, Sequence[str]]) -> Unit:
    """Create a ``"strheight"`` unit for the given string(s).

    Parameters
    ----------
    string : str or sequence of str
        The string(s) whose rendered height the unit represents.

    Returns
    -------
    Unit
        A unit of type ``"strheight"`` with value 1 for each string.
    """
    if isinstance(string, str):
        strings = [string]
    else:
        strings = list(string)
    n = len(strings)
    return Unit(np.ones(n), ["strheight"] * n, data=strings)


def string_ascent(string: Union[str, Sequence[str]]) -> Unit:
    """Create a ``"strascent"`` unit for the given string(s).

    Parameters
    ----------
    string : str or sequence of str
        The string(s) whose rendered ascent the unit represents.

    Returns
    -------
    Unit
        A unit of type ``"strascent"`` with value 1 for each string.
    """
    if isinstance(string, str):
        strings = [string]
    else:
        strings = list(string)
    n = len(strings)
    return Unit(np.ones(n), ["strascent"] * n, data=strings)


def string_descent(string: Union[str, Sequence[str]]) -> Unit:
    """Create a ``"strdescent"`` unit for the given string(s).

    Parameters
    ----------
    string : str or sequence of str
        The string(s) whose rendered descent the unit represents.

    Returns
    -------
    Unit
        A unit of type ``"strdescent"`` with value 1 for each string.
    """
    if isinstance(string, str):
        strings = [string]
    else:
        strings = list(string)
    n = len(strings)
    return Unit(np.ones(n), ["strdescent"] * n, data=strings)


# ===================================================================
# Absolute-size helper
# ===================================================================


def _is_absolute_unit_type(utype: str) -> bool:
    """Check whether a unit type is "absolute" in R's sense.

    R's ``isAbsolute()`` (grid.h:218) treats these as absolute:
    cm, inches, mm, points, lines, null, char, strwidth, strheight,
    strascent, strdescent, and all ``my*`` variants (>1000).

    NON-absolute (context-dependent on parent size): npc, native, snpc.
    Also non-absolute: grobwidth, grobheight, grobx, groby, grobascent,
    grobdescent (depend on grob measurement in context).

    Note: ``"null"`` IS absolute (it's resolved by GridLayout, not parent size).
    Note: ``"lines"`` IS absolute (depends on fontsize only, not parent size).
    """
    _NON_ABSOLUTE = frozenset({
        "npc", "native", "snpc",
        "grobwidth", "grobheight", "grobx", "groby",
        "grobascent", "grobdescent",
        # Compound types need recursion — not absolute in isolation.
        "sum", "min", "max",
    })
    return utype not in _NON_ABSOLUTE


def absolute_size(x: Unit) -> Unit:
    """Convert a Unit to its absolute form.

    Absolute units (cm, inches, lines, null, points, etc.) pass through
    unchanged.  Non-absolute units (npc, native, snpc, grobwidth, etc.)
    are replaced with ``unit(1, "null")``.

    For compound (sum/min/max) units, the function recurses into the
    operands, preserving absolute leaves and replacing non-absolute
    leaves with null.

    This matches R's ``absolute.size()`` / ``absolute.units()``
    (grid/R/size.R:130, grid/src/unit.c:1777-1831).

    Parameters
    ----------
    x : Unit
        A unit object.

    Returns
    -------
    Unit
        A copy of *x* with non-absolute elements replaced by ``null``.

    Examples
    --------
    >>> absolute_size(Unit(2, "cm"))
    Unit([2.0], ['cm'])
    >>> absolute_size(Unit(0.5, "npc"))
    Unit([1.0], ['null'])
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")

    n = len(x)

    # Fast path: all absolute → return as-is (R: line 1803)
    if all(_is_absolute_unit_type(x._units[i]) for i in range(n)):
        return x

    new = x.copy()
    for i in range(n):
        utype = new._units[i]
        if utype in ("sum", "min", "max"):
            # Arithmetic compound: recurse into operands (R: lines 1814-1818)
            data = new._data[i]
            if data is not None and isinstance(data, Unit):
                new._data[i] = absolute_size(data)
        elif not _is_absolute_unit_type(utype):
            # Non-absolute scalar: replace with unit(1, "null") (R: lines 1819-1820)
            new._values[i] = 1.0
            new._units[i] = "null"
            new._data[i] = None
    return new


# ===================================================================
# Unit conversion
# ===================================================================


def _convert_absolute(value: float, unit_from: str, unit_to: str) -> float:
    """Convert a single value between two absolute unit types.

    Parameters
    ----------
    value : float
        Numeric value in *unit_from* units.
    unit_from : str
        Source unit type (must be in ``_ABSOLUTE_UNIT_TYPES``).
    unit_to : str
        Target unit type (must be in ``_ABSOLUTE_UNIT_TYPES``).

    Returns
    -------
    float
        The value expressed in *unit_to* units.
    """
    if unit_from == unit_to:
        return value
    inches = value * _INCHES_PER[unit_from]
    return inches / _INCHES_PER[unit_to]


def convert_unit(
    x: Unit,
    unitTo: str,
    axisFrom: str = "x",
    typeFrom: str = "location",
    axisTo: Optional[str] = None,
    typeTo: Optional[str] = None,
    valueOnly: bool = False,
) -> Union[Unit, np.ndarray]:
    """Convert a ``Unit`` to a different unit type.

    Absolute-to-absolute conversions (e.g. cm to inches) are performed
    immediately.  Context-dependent conversions (involving npc, native,
    lines, etc.) use the active renderer's viewport context when
    available; otherwise a warning is issued and the unit is returned
    unchanged.

    Parameters
    ----------
    x : Unit
        The unit to convert.
    unitTo : str
        Target unit type string.
    axisFrom : str, optional
        Source axis (``"x"`` or ``"y"``).  Default ``"x"``.
    typeFrom : str, optional
        Source type (``"location"`` or ``"dimension"``).  Default
        ``"location"``.
    axisTo : str, optional
        Target axis.  Defaults to *axisFrom*.
    typeTo : str, optional
        Target type.  Defaults to *typeFrom*.
    valueOnly : bool, optional
        If ``True`` return a bare ``numpy`` array instead of a ``Unit``.

    Returns
    -------
    Unit or numpy.ndarray
        Converted unit, or numeric array when *valueOnly* is ``True``.

    Examples
    --------
    >>> convert_unit(Unit(1, "inches"), "cm")
    Unit([2.54], ['cm'])
    >>> convert_unit(Unit(2.54, "cm"), "inches", valueOnly=True)
    array([1.])
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")

    if axisTo is None:
        axisTo = axisFrom
    if typeTo is None:
        typeTo = typeFrom

    target = _resolve_alias(unitTo)
    n = len(x)
    result_vals = np.empty(n, dtype=np.float64)
    converted = True

    for i in range(n):
        src_unit = x._units[i]
        if src_unit in _ABSOLUTE_UNIT_TYPES and target in _ABSOLUTE_UNIT_TYPES:
            # Fast path: absolute → absolute (no context needed)
            result_vals[i] = _convert_absolute(x._values[i], src_unit, target)
        elif src_unit == target:
            # Same unit type -- no conversion needed
            result_vals[i] = x._values[i]
        else:
            # All other conversions go through the full two-stage pipeline
            # (R grid.c:1384-1575 L_convert):
            #   Stage 1: source → inches (via renderer context)
            #   Stage 2: inches → target (via inverse transforms)
            # This handles: npc, native, lines, char, snpc, strwidth,
            # grobwidth, compound units, absolute→context, context→absolute
            resolved = _try_resolve_with_renderer(
                x, i, src_unit, target, axisFrom, typeFrom,
            )
            if resolved is not None:
                result_vals[i] = resolved
            elif src_unit in _STR_METRIC_TYPES:
                # Fallback without renderer: string metric → inches → target
                inches_val = _eval_str_metric(src_unit, x._data[i], x._values[i])
                if target in _ABSOLUTE_UNIT_TYPES:
                    result_vals[i] = inches_val / _INCHES_PER[target]
                else:
                    result_vals[i] = inches_val
                    converted = False
            elif src_unit in _GROB_METRIC_TYPES:
                # Fallback: grob metric → inches → target
                metric_unit = _eval_grob_metric(src_unit, x._data[i])
                if (
                    metric_unit is not None
                    and len(metric_unit) > 0
                    and metric_unit._units[0] in _ABSOLUTE_UNIT_TYPES
                ):
                    src_inches = (
                        metric_unit._values[0]
                        * _INCHES_PER[metric_unit._units[0]]
                    )
                    src_inches *= x._values[i]
                    if target in _ABSOLUTE_UNIT_TYPES:
                        result_vals[i] = src_inches / _INCHES_PER[target]
                    else:
                        result_vals[i] = src_inches
                        converted = False
                else:
                    result_vals[i] = x._values[i]
                    converted = False
            elif src_unit in _ABSOLUTE_UNIT_TYPES:
                # Absolute source → context-dependent target (no renderer)
                result_vals[i] = x._values[i]
                converted = False
            else:
                result_vals[i] = x._values[i]
                converted = False

    if valueOnly:
        return result_vals

    if converted:
        return Unit(result_vals, target)
    else:
        # Return original unchanged when full conversion is not possible
        import warnings

        warnings.warn(
            f"Cannot convert between {set(x._units)} and {target!r} "
            "without a viewport context; returning unit unchanged.",
            stacklevel=2,
        )
        return x.copy()


def convert_x(x: Unit, unitTo: str, valueOnly: bool = False) -> Union[Unit, np.ndarray]:
    """Convert an x-axis location unit.

    Parameters
    ----------
    x : Unit
        Source unit.
    unitTo : str
        Target unit type.
    valueOnly : bool, optional
        Return bare numeric array if ``True``.

    Returns
    -------
    Unit or numpy.ndarray
    """
    return convert_unit(x, unitTo, "x", "location", "x", "location", valueOnly=valueOnly)


def convert_y(x: Unit, unitTo: str, valueOnly: bool = False) -> Union[Unit, np.ndarray]:
    """Convert a y-axis location unit.

    Parameters
    ----------
    x : Unit
        Source unit.
    unitTo : str
        Target unit type.
    valueOnly : bool, optional
        Return bare numeric array if ``True``.

    Returns
    -------
    Unit or numpy.ndarray
    """
    return convert_unit(x, unitTo, "y", "location", "y", "location", valueOnly=valueOnly)


def convert_width(x: Unit, unitTo: str, valueOnly: bool = False) -> Union[Unit, np.ndarray]:
    """Convert a width (x-axis dimension) unit.

    Parameters
    ----------
    x : Unit
        Source unit.
    unitTo : str
        Target unit type.
    valueOnly : bool, optional
        Return bare numeric array if ``True``.

    Returns
    -------
    Unit or numpy.ndarray
    """
    return convert_unit(
        x, unitTo, "x", "dimension", "x", "dimension", valueOnly=valueOnly
    )


def convert_height(x: Unit, unitTo: str, valueOnly: bool = False) -> Union[Unit, np.ndarray]:
    """Convert a height (y-axis dimension) unit.

    Parameters
    ----------
    x : Unit
        Source unit.
    unitTo : str
        Target unit type.
    valueOnly : bool, optional
        Return bare numeric array if ``True``.

    Returns
    -------
    Unit or numpy.ndarray
    """
    return convert_unit(
        x, unitTo, "y", "dimension", "y", "dimension", valueOnly=valueOnly
    )


# ---------------------------------------------------------------------------
# convertTheta -- port of R unit.R:617-629
# ---------------------------------------------------------------------------


_THETA_ALIASES: Dict[str, float] = {
    "east": 0.0,
    "north": 90.0,
    "west": 180.0,
    "south": 270.0,
}


def convert_theta(theta: Any) -> float:
    """Convert a theta angle to numeric degrees in [0, 360).

    Port of R ``convertTheta()`` (unit.R:617-629).
    Accepts character shortcuts ``"east"`` (0), ``"north"`` (90),
    ``"west"`` (180), ``"south"`` (270) or numeric values.

    Parameters
    ----------
    theta : str or float
        Angle specification.

    Returns
    -------
    float
        Angle in degrees, normalised to [0, 360).

    Raises
    ------
    ValueError
        If *theta* is an unrecognised string.

    Examples
    --------
    >>> convert_theta("north")
    90.0
    >>> convert_theta(450)
    90.0
    """
    if isinstance(theta, str):
        val = _THETA_ALIASES.get(theta.lower())
        if val is None:
            raise ValueError(f"invalid theta: {theta!r}")
        return val
    return float(theta) % 360.0


# ---------------------------------------------------------------------------
# deviceLoc / deviceDim -- port of R unit.R:117-151 + grid.c:1580-1677
# ---------------------------------------------------------------------------


def device_loc(
    x: Unit,
    y: Unit,
    value_only: bool = False,
    device: bool = False,
) -> dict:
    """Convert grid locations to absolute device coordinates.

    Port of R ``deviceLoc()`` (unit.R:117-133) + ``L_devLoc`` (grid.c:1580-1628).
    For each (x[i], y[i]) pair:
      1. Convert x to inches via transformXtoINCHES
      2. Convert y to inches via transformYtoINCHES
      3. Apply the viewport 3×3 transform (transformLocn: location → trans)
      4. Optionally convert to device coordinates

    Parameters
    ----------
    x, y : Unit
        Location units.
    value_only : bool
        If True, return raw numeric arrays. Otherwise return Unit objects.
    device : bool
        If True, return in device-native coordinates (pixels).
        If False, return in absolute inches.

    Returns
    -------
    dict
        ``{'x': ..., 'y': ...}`` — each is either a Unit or ndarray.
    """
    from ._state import get_state
    from ._vp_calc import location, trans

    state = get_state()
    renderer = state.get_renderer()

    if renderer is None:
        raise RuntimeError("deviceLoc requires an active renderer")

    vtr = renderer._vp_transform_stack[-1]
    gp = state.get_gpar()

    nx = len(x)
    ny = len(y)
    maxn = max(nx, ny)

    out_x = np.empty(maxn, dtype=np.float64)
    out_y = np.empty(maxn, dtype=np.float64)

    for i in range(maxn):
        # Stage 1: resolve to inches within viewport
        # R grid.c:1612-1616  transformLocn()
        xx = renderer._resolve_to_inches_idx(x, i % nx, "x", False, gp)
        yy = renderer._resolve_to_inches_idx(y, i % ny, "y", False, gp)

        # Stage 2: apply viewport 3×3 transform to get absolute inches
        # R unit.c:1168-1171  location→trans
        loc = location(xx, yy)
        abs_loc = trans(loc, vtr.transform)
        xx = abs_loc[0]
        yy = abs_loc[1]

        if device:
            # Convert absolute inches to device pixels
            # R grid.c:1618-1619  toDeviceX/Y
            xx = renderer.inches_to_dev_x(xx)
            yy = renderer.inches_to_dev_y(yy)

        out_x[i] = xx
        out_y[i] = yy

    if value_only:
        return {"x": out_x, "y": out_y}
    else:
        if device:
            return {"x": Unit(out_x, "native"), "y": Unit(out_y, "native")}
        else:
            return {"x": Unit(out_x, "inches"), "y": Unit(out_y, "inches")}


def device_dim(
    w: Unit,
    h: Unit,
    value_only: bool = False,
    device: bool = False,
) -> dict:
    """Convert grid dimensions to absolute device dimensions.

    Port of R ``deviceDim()`` (unit.R:135-151) + ``L_devDim`` (grid.c:1630-1677).
    For each (w[i], h[i]) pair:
      1. Convert w to inches via transformWidthtoINCHES
      2. Convert h to inches via transformHeighttoINCHES
      3. Apply rotation transform (transformDimn)
      4. Optionally convert to device units

    Parameters
    ----------
    w, h : Unit
        Dimension units.
    value_only : bool
        If True, return raw numeric arrays. Otherwise return Unit objects.
    device : bool
        If True, return in device-native units (pixels).
        If False, return in absolute inches.

    Returns
    -------
    dict
        ``{'w': ..., 'h': ...}`` — each is either a Unit or ndarray.
    """
    import math
    from ._state import get_state
    from ._vp_calc import location, rotation, trans

    state = get_state()
    renderer = state.get_renderer()

    if renderer is None:
        raise RuntimeError("deviceDim requires an active renderer")

    vtr = renderer._vp_transform_stack[-1]
    gp = state.get_gpar()
    rotation_angle = vtr.rotation_angle

    nw = len(w)
    nh = len(h)
    maxn = max(nw, nh)

    out_w = np.empty(maxn, dtype=np.float64)
    out_h = np.empty(maxn, dtype=np.float64)

    for i in range(maxn):
        # Stage 1: resolve to inches within viewport
        ww = renderer._resolve_to_inches_idx(w, i % nw, "x", True, gp)
        hh = renderer._resolve_to_inches_idx(h, i % nh, "y", True, gp)

        # Stage 2: apply rotation (R unit.c:1208-1212 transformDimn)
        # R: location(ww,hh,din); rotation(angle,r); trans(din,r,dout);
        din = location(ww, hh)
        rot = rotation(rotation_angle)
        dout = trans(din, rot)
        ww = dout[0]
        hh = dout[1]

        if device:
            # Convert absolute inches to device pixels
            ww = renderer.inches_to_dev_w(ww)
            hh = renderer.inches_to_dev_h(hh)

        out_w[i] = ww
        out_h[i] = hh

    if value_only:
        return {"w": out_w, "h": out_h}
    else:
        if device:
            return {"w": Unit(out_w, "native"), "h": Unit(out_h, "native")}
        else:
            return {"w": Unit(out_w, "inches"), "h": Unit(out_h, "inches")}
