"""
Gradient and pattern fill system for grid_py.

Python port of R's ``grid`` package pattern infrastructure
(``grid/R/patterns.R``).  Provides linear gradients, radial gradients,
and tiling patterns that can be used as fill values in graphical
parameter (gpar) objects.

Classes
-------
LinearGradient
    A two-point linear colour gradient.
RadialGradient
    A two-circle radial colour gradient.
Pattern
    A tiling pattern based on an arbitrary grob.

Functions
---------
linear_gradient
    Factory for :class:`LinearGradient`.
radial_gradient
    Factory for :class:`RadialGradient`.
pattern
    Factory for :class:`Pattern`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np

from ._units import Unit, is_unit

__all__: List[str] = [
    "LinearGradient",
    "RadialGradient",
    "Pattern",
    "linear_gradient",
    "radial_gradient",
    "pattern",
]

# Valid *extend* modes, matching R's ``match.arg`` choices.
_VALID_EXTEND: tuple[str, ...] = ("pad", "repeat", "reflect", "none")


def _validate_extend(extend: str) -> str:
    """Return *extend* if valid, otherwise raise ``ValueError``.

    Parameters
    ----------
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.

    Returns
    -------
    str
        The validated extend string.

    Raises
    ------
    ValueError
        If *extend* is not one of the four recognised modes.
    """
    if extend not in _VALID_EXTEND:
        raise ValueError(
            f"extend must be one of {_VALID_EXTEND!r}, got {extend!r}"
        )
    return extend


def _ensure_unit(value: Any, default_units: str) -> Unit:
    """Coerce *value* to a :class:`Unit` if it is not already one.

    Parameters
    ----------
    value : Any
        A :class:`Unit` instance, or a numeric scalar that will be
        wrapped in ``Unit(value, default_units)``.
    default_units : str
        Unit type used when *value* is not already a :class:`Unit`.

    Returns
    -------
    Unit
        The (possibly newly created) unit.
    """
    if is_unit(value):
        return value
    return Unit(value, default_units)


def _make_stops(
    colours: Sequence[str],
    stops: Optional[Sequence[float]],
) -> tuple[list[str], list[float]]:
    """Normalise *colours* and *stops* following R semantics.

    Both sequences are recycled to the length of the longer one.  If
    *stops* is ``None``, evenly-spaced values in [0, 1] are generated.

    Parameters
    ----------
    colours : Sequence[str]
        Colour specification strings.
    stops : Sequence[float] or None
        Gradient stop positions in [0, 1].

    Returns
    -------
    tuple[list[str], list[float]]
        ``(colours, stops)`` recycled to the same length.

    Raises
    ------
    ValueError
        If the resulting length is less than 1.
    """
    n_colours = len(colours)
    if stops is None:
        n_stops = n_colours
        stops_arr: np.ndarray = np.linspace(0.0, 1.0, n_stops)
    else:
        stops_arr = np.asarray(stops, dtype=float)
        n_stops = len(stops_arr)

    nstops = max(n_colours, n_stops)
    if nstops < 1:
        raise ValueError("colours and stops must be at least length 1")

    # Recycle both to *nstops* (mirroring R's ``rep(x, length.out=n)``).
    colours_out: list[str] = [
        colours[i % n_colours] for i in range(nstops)
    ]
    if len(stops_arr) < nstops:
        stops_arr = np.resize(stops_arr, nstops)
    stops_out: list[float] = stops_arr[:nstops].tolist()

    return colours_out, stops_out


# ======================================================================
# LinearGradient
# ======================================================================


class LinearGradient:
    """A linear colour gradient defined by two endpoints.

    This corresponds to R's ``grid::linearGradient()`` and the internal
    class ``GridLinearGradient``.

    Parameters
    ----------
    colours : list[str]
        Colour strings (e.g. ``["black", "white"]``).
    stops : list[float] or None
        Gradient stop positions in [0, 1].  ``None`` (default) produces
        evenly-spaced stops matching the length of *colours*.
    x1 : Unit or float or None
        Horizontal start of the gradient line.  Defaults to
        ``Unit(0, "npc")``.
    y1 : Unit or float or None
        Vertical start of the gradient line.  Defaults to
        ``Unit(0, "npc")``.
    x2 : Unit or float or None
        Horizontal end of the gradient line.  Defaults to
        ``Unit(1, "npc")``.
    y2 : Unit or float or None
        Vertical end of the gradient line.  Defaults to
        ``Unit(1, "npc")``.
    default_units : str
        Unit type applied when a coordinate is given as a plain number.
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.
    group : bool
        If ``True`` the gradient is resolved relative to the bounding
        box of *all* shapes; if ``False`` it is resolved per shape.

    Raises
    ------
    ValueError
        If *extend* is invalid, or *colours*/*stops* have length < 1,
        or any coordinate has length != 1.
    """

    def __init__(
        self,
        colours: list[str],
        stops: Optional[list[float]] = None,
        x1: Optional[Union[Unit, float]] = None,
        y1: Optional[Union[Unit, float]] = None,
        x2: Optional[Union[Unit, float]] = None,
        y2: Optional[Union[Unit, float]] = None,
        default_units: str = "npc",
        extend: str = "pad",
        group: bool = True,
    ) -> None:
        self.colours, self.stops = _make_stops(colours, stops)

        self.x1: Unit = _ensure_unit(
            x1 if x1 is not None else 0.0, default_units
        )
        self.y1: Unit = _ensure_unit(
            y1 if y1 is not None else 0.0, default_units
        )
        self.x2: Unit = _ensure_unit(
            x2 if x2 is not None else 1.0, default_units
        )
        self.y2: Unit = _ensure_unit(
            y2 if y2 is not None else 1.0, default_units
        )

        # Each coordinate must be scalar (length 1).
        for name, val in (
            ("x1", self.x1),
            ("y1", self.y1),
            ("x2", self.x2),
            ("y2", self.y2),
        ):
            if len(val) != 1:
                raise ValueError(
                    f"{name} must be length 1, got length {len(val)}"
                )

        self.extend: str = _validate_extend(extend)
        self.group: bool = bool(group)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LinearGradient(colours={self.colours!r}, "
            f"stops={self.stops!r}, "
            f"x1={self.x1!r}, y1={self.y1!r}, "
            f"x2={self.x2!r}, y2={self.y2!r}, "
            f"extend={self.extend!r}, group={self.group!r})"
        )


# ======================================================================
# RadialGradient
# ======================================================================


class RadialGradient:
    """A radial colour gradient defined by two circles.

    This corresponds to R's ``grid::radialGradient()`` and the internal
    class ``GridRadialGradient``.

    Parameters
    ----------
    colours : list[str]
        Colour strings.
    stops : list[float] or None
        Gradient stop positions in [0, 1].
    cx1 : Unit or float or None
        Horizontal centre of the inner circle.  Default ``0.5 npc``.
    cy1 : Unit or float or None
        Vertical centre of the inner circle.  Default ``0.5 npc``.
    r1 : Unit or float or None
        Radius of the inner circle.  Default ``0 npc``.
    cx2 : Unit or float or None
        Horizontal centre of the outer circle.  Default ``0.5 npc``.
    cy2 : Unit or float or None
        Vertical centre of the outer circle.  Default ``0.5 npc``.
    r2 : Unit or float or None
        Radius of the outer circle.  Default ``0.5 npc``.
    default_units : str
        Unit type applied when a parameter is given as a plain number.
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.
    group : bool
        If ``True`` the gradient is resolved relative to the bounding
        box of *all* shapes; if ``False`` it is resolved per shape.

    Raises
    ------
    ValueError
        If *extend* is invalid, colours/stops have length < 1, or any
        coordinate/radius has length != 1.
    """

    def __init__(
        self,
        colours: list[str],
        stops: Optional[list[float]] = None,
        cx1: Optional[Union[Unit, float]] = None,
        cy1: Optional[Union[Unit, float]] = None,
        r1: Optional[Union[Unit, float]] = None,
        cx2: Optional[Union[Unit, float]] = None,
        cy2: Optional[Union[Unit, float]] = None,
        r2: Optional[Union[Unit, float]] = None,
        default_units: str = "npc",
        extend: str = "pad",
        group: bool = True,
    ) -> None:
        self.colours, self.stops = _make_stops(colours, stops)

        self.cx1: Unit = _ensure_unit(
            cx1 if cx1 is not None else 0.5, default_units
        )
        self.cy1: Unit = _ensure_unit(
            cy1 if cy1 is not None else 0.5, default_units
        )
        self.r1: Unit = _ensure_unit(
            r1 if r1 is not None else 0.0, default_units
        )
        self.cx2: Unit = _ensure_unit(
            cx2 if cx2 is not None else 0.5, default_units
        )
        self.cy2: Unit = _ensure_unit(
            cy2 if cy2 is not None else 0.5, default_units
        )
        self.r2: Unit = _ensure_unit(
            r2 if r2 is not None else 0.5, default_units
        )

        for name, val in (
            ("cx1", self.cx1),
            ("cy1", self.cy1),
            ("r1", self.r1),
            ("cx2", self.cx2),
            ("cy2", self.cy2),
            ("r2", self.r2),
        ):
            if len(val) != 1:
                raise ValueError(
                    f"{name} must be length 1, got length {len(val)}"
                )

        self.extend: str = _validate_extend(extend)
        self.group: bool = bool(group)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RadialGradient(colours={self.colours!r}, "
            f"stops={self.stops!r}, "
            f"cx1={self.cx1!r}, cy1={self.cy1!r}, r1={self.r1!r}, "
            f"cx2={self.cx2!r}, cy2={self.cy2!r}, r2={self.r2!r}, "
            f"extend={self.extend!r}, group={self.group!r})"
        )


# ======================================================================
# Pattern (tiling pattern)
# ======================================================================

# Justification helpers -- resolve a single *just* string to (hjust, vjust).
_JUST_H: dict[str, float] = {
    "left": 0.0,
    "right": 1.0,
    "centre": 0.5,
    "center": 0.5,
    "top": 0.5,
    "bottom": 0.5,
}

_JUST_V: dict[str, float] = {
    "left": 0.5,
    "right": 0.5,
    "centre": 0.5,
    "center": 0.5,
    "top": 1.0,
    "bottom": 0.0,
}


def _resolve_just(
    just: Union[str, tuple[float, float]],
) -> tuple[float, float]:
    """Return ``(hjust, vjust)`` from a justification specification.

    Parameters
    ----------
    just : str or tuple[float, float]
        A justification string (``"centre"``, ``"left"``, etc.) or an
        explicit ``(hjust, vjust)`` pair.

    Returns
    -------
    tuple[float, float]
        Numeric ``(hjust, vjust)`` in [0, 1].

    Raises
    ------
    ValueError
        If *just* is an unrecognised string.
    """
    if isinstance(just, str):
        j = just.lower()
        if j not in _JUST_H:
            raise ValueError(
                f"Unrecognised justification string: {just!r}"
            )
        return _JUST_H[j], _JUST_V[j]
    # Assume numeric pair.
    return (float(just[0]), float(just[1]))


class Pattern:
    """A tiling pattern fill based on an arbitrary grob.

    This corresponds to R's ``grid::pattern()`` and the internal class
    ``GridTilingPattern``.

    Parameters
    ----------
    grob : Any
        A grob (graphical object) to use as the repeating tile.
    x : Unit or float or None
        Horizontal position of the tile.  Default ``0.5 npc``.
    y : Unit or float or None
        Vertical position of the tile.  Default ``0.5 npc``.
    width : Unit or float or None
        Width of the tile.  Default ``1 npc``.
    height : Unit or float or None
        Height of the tile.  Default ``1 npc``.
    default_units : str
        Unit type applied when a dimension is given as a plain number.
    just : str or tuple[float, float]
        Justification of the tile relative to ``(x, y)``.  Accepts
        standard strings such as ``"centre"``, ``"left"``, etc., or an
        explicit ``(hjust, vjust)`` pair.
    extend : str
        One of ``"pad"``, ``"repeat"``, ``"reflect"``, or ``"none"``.
    group : bool
        If ``True`` the pattern is resolved relative to the bounding
        box of *all* shapes; if ``False`` it is resolved per shape.

    Raises
    ------
    ValueError
        If *extend* is invalid or any coordinate/dimension has
        length != 1.
    """

    def __init__(
        self,
        grob: Any,
        x: Optional[Union[Unit, float]] = None,
        y: Optional[Union[Unit, float]] = None,
        width: Optional[Union[Unit, float]] = None,
        height: Optional[Union[Unit, float]] = None,
        default_units: str = "npc",
        just: Union[str, tuple[float, float]] = "centre",
        extend: str = "pad",
        group: bool = True,
    ) -> None:
        self.grob: Any = grob

        self.x: Unit = _ensure_unit(
            x if x is not None else 0.5, default_units
        )
        self.y: Unit = _ensure_unit(
            y if y is not None else 0.5, default_units
        )
        self.width: Unit = _ensure_unit(
            width if width is not None else 1.0, default_units
        )
        self.height: Unit = _ensure_unit(
            height if height is not None else 1.0, default_units
        )

        for name, val in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            if len(val) != 1:
                raise ValueError(
                    f"{name} must be length 1, got length {len(val)}"
                )

        self.hjust: float
        self.vjust: float
        self.hjust, self.vjust = _resolve_just(just)

        self.extend: str = _validate_extend(extend)
        self.group: bool = bool(group)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Pattern(grob={self.grob!r}, "
            f"x={self.x!r}, y={self.y!r}, "
            f"width={self.width!r}, height={self.height!r}, "
            f"hjust={self.hjust!r}, vjust={self.vjust!r}, "
            f"extend={self.extend!r}, group={self.group!r})"
        )


# ======================================================================
# Factory functions
# ======================================================================


def linear_gradient(
    colours: list[str] = None,
    stops: Optional[list[float]] = None,
    x1: Optional[Union[Unit, float]] = None,
    y1: Optional[Union[Unit, float]] = None,
    x2: Optional[Union[Unit, float]] = None,
    y2: Optional[Union[Unit, float]] = None,
    default_units: str = "npc",
    extend: str = "pad",
    group: bool = True,
) -> LinearGradient:
    """Create a :class:`LinearGradient`.

    This is a convenience wrapper matching R's
    ``grid::linearGradient()`` function signature.

    Parameters
    ----------
    colours : list[str], optional
        Colour strings.  Defaults to ``["black", "white"]``.
    stops : list[float] or None
        Gradient stop positions in [0, 1].
    x1 : Unit or float or None
        Horizontal start of the gradient line.
    y1 : Unit or float or None
        Vertical start of the gradient line.
    x2 : Unit or float or None
        Horizontal end of the gradient line.
    y2 : Unit or float or None
        Vertical end of the gradient line.
    default_units : str
        Unit type used for bare numeric coordinates.
    extend : str
        Gradient extension mode.
    group : bool
        Resolve gradient relative to all shapes (``True``) or per
        shape (``False``).

    Returns
    -------
    LinearGradient
        A new linear gradient object.

    Examples
    --------
    >>> lg = linear_gradient(["red", "blue"])
    >>> lg.colours
    ['red', 'blue']
    """
    if colours is None:
        colours = ["black", "white"]
    return LinearGradient(
        colours=colours,
        stops=stops,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        default_units=default_units,
        extend=extend,
        group=group,
    )


def radial_gradient(
    colours: list[str] = None,
    stops: Optional[list[float]] = None,
    cx1: Optional[Union[Unit, float]] = None,
    cy1: Optional[Union[Unit, float]] = None,
    r1: Optional[Union[Unit, float]] = None,
    cx2: Optional[Union[Unit, float]] = None,
    cy2: Optional[Union[Unit, float]] = None,
    r2: Optional[Union[Unit, float]] = None,
    default_units: str = "npc",
    extend: str = "pad",
    group: bool = True,
) -> RadialGradient:
    """Create a :class:`RadialGradient`.

    This is a convenience wrapper matching R's
    ``grid::radialGradient()`` function signature.

    Parameters
    ----------
    colours : list[str], optional
        Colour strings.  Defaults to ``["black", "white"]``.
    stops : list[float] or None
        Gradient stop positions in [0, 1].
    cx1 : Unit or float or None
        Horizontal centre of the inner circle.
    cy1 : Unit or float or None
        Vertical centre of the inner circle.
    r1 : Unit or float or None
        Radius of the inner circle.
    cx2 : Unit or float or None
        Horizontal centre of the outer circle.
    cy2 : Unit or float or None
        Vertical centre of the outer circle.
    r2 : Unit or float or None
        Radius of the outer circle.
    default_units : str
        Unit type used for bare numeric values.
    extend : str
        Gradient extension mode.
    group : bool
        Resolve gradient relative to all shapes (``True``) or per
        shape (``False``).

    Returns
    -------
    RadialGradient
        A new radial gradient object.

    Examples
    --------
    >>> rg = radial_gradient(["white", "black"])
    >>> rg.r2
    Unit([0.5], ['npc'])
    """
    if colours is None:
        colours = ["black", "white"]
    return RadialGradient(
        colours=colours,
        stops=stops,
        cx1=cx1,
        cy1=cy1,
        r1=r1,
        cx2=cx2,
        cy2=cy2,
        r2=r2,
        default_units=default_units,
        extend=extend,
        group=group,
    )


def pattern(
    grob: Any,
    x: Optional[Union[Unit, float]] = None,
    y: Optional[Union[Unit, float]] = None,
    width: Optional[Union[Unit, float]] = None,
    height: Optional[Union[Unit, float]] = None,
    default_units: str = "npc",
    just: Union[str, tuple[float, float]] = "centre",
    extend: str = "pad",
    group: bool = True,
) -> Pattern:
    """Create a :class:`Pattern` (tiling fill).

    This is a convenience wrapper matching R's ``grid::pattern()``
    function signature.

    Parameters
    ----------
    grob : Any
        A grob to use as the repeating tile.
    x : Unit or float or None
        Horizontal position of the tile.
    y : Unit or float or None
        Vertical position of the tile.
    width : Unit or float or None
        Width of the tile.
    height : Unit or float or None
        Height of the tile.
    default_units : str
        Unit type used for bare numeric values.
    just : str or tuple[float, float]
        Tile justification relative to ``(x, y)``.
    extend : str
        Pattern extension mode.
    group : bool
        Resolve pattern relative to all shapes (``True``) or per
        shape (``False``).

    Returns
    -------
    Pattern
        A new tiling pattern object.

    Examples
    --------
    >>> pat = pattern("placeholder_grob")
    >>> pat.hjust
    0.5
    """
    return Pattern(
        grob=grob,
        x=x,
        y=y,
        width=width,
        height=height,
        default_units=default_units,
        just=just,
        extend=extend,
        group=group,
    )
