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
    "mystrascent",
    "mystrdescent",
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
    m = calc_string_metric(text)
    if unit_type == "strwidth":
        return m["width"] * scale
    elif unit_type == "strheight":
        return (m["ascent"] + m["descent"]) * scale
    elif unit_type == "strascent":
        return m["ascent"] * scale
    elif unit_type == "strdescent":
        return m["descent"] * scale
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
    """Try to resolve a context-dependent unit via the active renderer.

    Returns the converted value in *target* units, or ``None`` if no
    renderer is available.
    """
    from ._state import get_state
    state = get_state()
    renderer = state.get_renderer()

    if renderer is None or not hasattr(renderer, "resolve_to_npc"):
        return None

    # Build a single-element Unit for the source
    elem = Unit(x._values[i], src_unit, data=x._data[i])
    is_dim = type_ in ("dimension",)

    # Resolve source -> NPC
    npc_val = renderer.resolve_to_npc(elem, axis=axis, is_dim=is_dim)

    # Convert NPC -> target
    if target == "npc":
        return npc_val
    elif target in _ABSOLUTE_UNIT_TYPES:
        # NPC -> inches -> target
        _x0, _y0, vp_pw, vp_ph = renderer.get_viewport_bounds()
        vp_px = vp_pw if axis == "x" else vp_ph
        vp_inches = vp_px / renderer.dpi
        inches = npc_val * vp_inches
        return inches / _INCHES_PER[target]
    else:
        # Target is also context-dependent -- just return the NPC value
        return npc_val


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
            "mystrascent",
            "mystrdescent",
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


def unit_type(x: Unit) -> Union[str, List[str]]:
    """Return the unit type(s) of *x*.

    Parameters
    ----------
    x : Unit
        A unit object.

    Returns
    -------
    str or list of str
        A single string when *x* has length 1, otherwise a list.

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
    if len(x) == 1:
        return x._units[0]
    return list(x._units)


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


def unit_rep(x: Unit, times: int = 1) -> Unit:
    """Repeat a ``Unit`` *times* times.

    Parameters
    ----------
    x : Unit
        The unit to repeat.
    times : int
        Number of repetitions.

    Returns
    -------
    Unit
        A unit whose elements are *x* repeated.

    Examples
    --------
    >>> unit_rep(Unit(1, "cm"), 3)
    Unit([1.0, 1.0, 1.0], ['cm', 'cm', 'cm'])
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")
    if times <= 0:
        raise ValueError("times must be a positive integer")
    indices = list(range(len(x))) * times
    return x[indices]


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


def absolute_size(x: Unit) -> Unit:
    """Mark a ``Unit`` as absolute.

    For units that are already absolute this is a no-op.  For
    context-dependent units, only the absolute components are retained
    (non-absolute components are set to zero).

    Parameters
    ----------
    x : Unit
        A unit object.

    Returns
    -------
    Unit
        A copy of *x* with non-absolute elements zeroed out.

    Examples
    --------
    >>> absolute_size(Unit(2, "cm"))
    Unit([2.0], ['cm'])
    """
    if not isinstance(x, Unit):
        raise TypeError("x must be a Unit")
    new = x.copy()
    for i in range(len(new)):
        if new._units[i] not in _ABSOLUTE_UNIT_TYPES:
            new._values[i] = 0.0
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
            result_vals[i] = _convert_absolute(x._values[i], src_unit, target)
        elif src_unit == target:
            # Same unit type -- no conversion needed
            result_vals[i] = x._values[i]
        elif src_unit in _STR_METRIC_TYPES:
            # Evaluate string metric → inches, then convert to target
            inches_val = _eval_str_metric(src_unit, x._data[i], x._values[i])
            if target in _ABSOLUTE_UNIT_TYPES:
                result_vals[i] = inches_val / _INCHES_PER[target]
            else:
                result_vals[i] = inches_val
                converted = False
        elif src_unit in _GROB_METRIC_TYPES:
            # Evaluate grob metric → Unit, extract inches, convert
            metric_unit = _eval_grob_metric(src_unit, x._data[i])
            if (
                metric_unit is not None
                and len(metric_unit) > 0
                and metric_unit._units[0] in _ABSOLUTE_UNIT_TYPES
            ):
                # Convert the measured value to inches first
                src_inches = (
                    metric_unit._values[0]
                    * _INCHES_PER[metric_unit._units[0]]
                )
                # Apply the scale factor from the original unit
                src_inches *= x._values[i]
                if target in _ABSOLUTE_UNIT_TYPES:
                    result_vals[i] = src_inches / _INCHES_PER[target]
                else:
                    result_vals[i] = src_inches
                    converted = False
            else:
                result_vals[i] = x._values[i]
                converted = False
        else:
            # Context-dependent: try to resolve via active renderer
            resolved = _try_resolve_with_renderer(
                x, i, src_unit, target, axisFrom, typeFrom,
            )
            if resolved is not None:
                result_vals[i] = resolved
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
