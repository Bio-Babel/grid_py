"""Primitive grob constructors and ``grid_*`` drawing functions.

This module is the Python port of R's ``grid/R/primitives.R``,
``grid/R/roundRect.R``, and ``grid/R/function.R`` (~1699 lines combined).
It provides every primitive grob type available in the *grid* graphics
system together with convenience ``grid_*`` functions that optionally
draw them immediately.

Each primitive follows a consistent three-part pattern:

1. A :class:`~grid_py._grob.Grob` instance with a specific ``_grid_class``.
2. A *constructor* function (e.g. :func:`rect_grob`) that builds the grob.
3. A ``grid_*`` wrapper (e.g. :func:`grid_rect`) with a ``draw`` parameter.

Notes
-----
Actual rendering is deferred -- when ``draw=True`` the grob is appended to
the module-level ``_display_list`` and will be consumed by a future rendering
back-end.  When ``draw=False`` the grob is simply returned.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Union,
)

from ._arrow import Arrow
from ._gpar import Gpar
from ._grob import Grob, GTree
from ._just import valid_just
from ._units import Unit, is_unit

__all__ = [
    # move.to
    "move_to_grob",
    "grid_move_to",
    # line.to
    "line_to_grob",
    "grid_line_to",
    # lines
    "lines_grob",
    "grid_lines",
    # polyline
    "polyline_grob",
    "grid_polyline",
    # segments
    "segments_grob",
    "grid_segments",
    # arrows (defunct, kept for API completeness)
    "arrows_grob",
    "grid_arrows",
    # points
    "points_grob",
    "grid_points",
    # rect
    "rect_grob",
    "grid_rect",
    # roundrect
    "roundrect_grob",
    "grid_roundrect",
    # circle
    "circle_grob",
    "grid_circle",
    # polygon
    "polygon_grob",
    "grid_polygon",
    # path
    "path_grob",
    "grid_path",
    # text
    "text_grob",
    "grid_text",
    # raster
    "raster_grob",
    "grid_raster",
    # clip
    "clip_grob",
    "grid_clip",
    # null
    "null_grob",
    "grid_null",
    # function
    "function_grob",
    "grid_function",
]

# ---------------------------------------------------------------------------
# Module-level display list (placeholder for future rendering back-end)
# ---------------------------------------------------------------------------

_display_list: List[Grob] = []


def _grid_draw(grob: Grob) -> None:
    """Draw *grob* via the rendering back-end and record it.

    Parameters
    ----------
    grob : Grob
        The graphical object to draw and record.
    """
    from ._draw import grid_draw  # lazy import to avoid circular dependency

    grid_draw(grob, recording=True)


# ---------------------------------------------------------------------------
# Helper: ensure a value is a Unit
# ---------------------------------------------------------------------------


def _ensure_unit(
    x: Any,
    default_units: str,
) -> Unit:
    """Convert *x* to a :class:`Unit` if it is not already one.

    Parameters
    ----------
    x : Any
        A numeric scalar, sequence of numerics, or an existing ``Unit``.
    default_units : str
        The unit string to use when *x* is not already a ``Unit``
        (e.g. ``"npc"``, ``"inches"``).

    Returns
    -------
    Unit
        The value wrapped as a ``Unit``.
    """
    if is_unit(x):
        return x
    return Unit(x, default_units)


# ===================================================================== #
#  move.to primitive                                                     #
# ===================================================================== #


def move_to_grob(
    x: Any = 0,
    y: Any = 0,
    default_units: str = "npc",
    name: Optional[str] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *move.to* grob.

    A *move.to* grob sets the current drawing position without producing
    any visible output.  It is the counterpart to :func:`line_to_grob`.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinate.  Converted using *default_units* when not
        already a ``Unit``.
    y : Unit or numeric
        Vertical coordinate.
    default_units : str
        Unit type applied to bare numeric *x* / *y*.
    name : str or None
        Grob name.  Auto-generated when ``None``.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="move.to"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    return Grob(x=x, y=y, name=name, vp=vp, _grid_class="move.to")


def grid_move_to(
    x: Any = 0,
    y: Any = 0,
    default_units: str = "npc",
    name: Optional[str] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *move.to* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinate.
    y : Unit or numeric
        Vertical coordinate.
    default_units : str
        Default unit type for bare numerics.
    name : str or None
        Grob name.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob (returned invisibly when *draw* is ``True``).
    """
    grob = move_to_grob(x=x, y=y, default_units=default_units, name=name, vp=vp)
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  line.to primitive                                                     #
# ===================================================================== #


def line_to_grob(
    x: Any = 1,
    y: Any = 1,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *line.to* grob.

    A *line.to* grob draws a straight line from the current drawing
    position (set by :func:`move_to_grob`) to the specified location.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal endpoint.
    y : Unit or numeric
        Vertical endpoint.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="line.to"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    return Grob(
        x=x, y=y, arrow=arrow,
        name=name, gp=gp, vp=vp, _grid_class="line.to",
    )


def grid_line_to(
    x: Any = 1,
    y: Any = 1,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *line.to* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal endpoint.
    y : Unit or numeric
        Vertical endpoint.
    default_units : str
        Default unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = line_to_grob(
        x=x, y=y, default_units=default_units, arrow=arrow,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  lines primitive                                                       #
# ===================================================================== #


def lines_grob(
    x: Optional[Any] = None,
    y: Optional[Any] = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *lines* grob.

    A *lines* grob draws a single connected polyline through the given
    coordinates.

    Parameters
    ----------
    x : Unit, numeric, or None
        Horizontal coordinates.  Defaults to ``Unit([0, 1], "npc")``
        when ``None``.
    y : Unit, numeric, or None
        Vertical coordinates.  Defaults to ``Unit([0, 1], "npc")``
        when ``None``.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="lines"``.
    """
    if x is None:
        x = Unit([0, 1], "npc")
    else:
        x = _ensure_unit(x, default_units)
    if y is None:
        y = Unit([0, 1], "npc")
    else:
        y = _ensure_unit(y, default_units)
    return Grob(
        x=x, y=y, arrow=arrow,
        name=name, gp=gp, vp=vp, _grid_class="lines",
    )


def grid_lines(
    x: Optional[Any] = None,
    y: Optional[Any] = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *lines* grob.

    Parameters
    ----------
    x : Unit, numeric, or None
        Horizontal coordinates.  Defaults to ``Unit([0, 1], "npc")``.
    y : Unit, numeric, or None
        Vertical coordinates.  Defaults to ``Unit([0, 1], "npc")``.
    default_units : str
        Default unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = lines_grob(
        x=x, y=y, default_units=default_units, arrow=arrow,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  polyline primitive                                                    #
# ===================================================================== #


def polyline_grob(
    x: Any = None,
    y: Any = None,
    id: Optional[Sequence[int]] = None,
    id_lengths: Optional[Sequence[int]] = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *polyline* grob.

    A *polyline* grob draws one or more disconnected polylines through
    the given coordinates.  Sub-polylines are identified either by *id*
    (a per-point group label) or *id_lengths* (lengths of consecutive
    runs).  Only one of *id* and *id_lengths* may be specified.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinates.  Defaults to ``Unit([0, 1], "npc")``.
    y : Unit or numeric
        Vertical coordinates.  Defaults to ``Unit([0, 1], "npc")``.
    id : sequence of int or None
        Per-point group identifier.
    id_lengths : sequence of int or None
        Lengths of consecutive sub-polylines.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="polyline"``.

    Raises
    ------
    ValueError
        If both *id* and *id_lengths* are specified.
    """
    if x is None:
        x = Unit([0, 1], "npc")
    else:
        x = _ensure_unit(x, default_units)
    if y is None:
        y = Unit([0, 1], "npc")
    else:
        y = _ensure_unit(y, default_units)
    if id is not None and id_lengths is not None:
        raise ValueError(
            "it is invalid to specify both 'id' and 'id_lengths'"
        )
    return Grob(
        x=x, y=y, id=id, id_lengths=id_lengths, arrow=arrow,
        name=name, gp=gp, vp=vp, _grid_class="polyline",
    )


def grid_polyline(
    x: Any = None,
    y: Any = None,
    id: Optional[Sequence[int]] = None,
    id_lengths: Optional[Sequence[int]] = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *polyline* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinates.
    y : Unit or numeric
        Vertical coordinates.
    id : sequence of int or None
        Per-point group identifier.
    id_lengths : sequence of int or None
        Lengths of consecutive sub-polylines.
    default_units : str
        Default unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = polyline_grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        default_units=default_units, arrow=arrow,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  segments primitive                                                    #
# ===================================================================== #


def segments_grob(
    x0: Any = 0,
    y0: Any = 0,
    x1: Any = 1,
    y1: Any = 1,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *segments* grob.

    A *segments* grob draws one or more independent line segments, each
    defined by start ``(x0, y0)`` and end ``(x1, y1)`` coordinates.

    Parameters
    ----------
    x0 : Unit or numeric
        Horizontal start coordinate(s).
    y0 : Unit or numeric
        Vertical start coordinate(s).
    x1 : Unit or numeric
        Horizontal end coordinate(s).
    y1 : Unit or numeric
        Vertical end coordinate(s).
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="segments"``.
    """
    x0 = _ensure_unit(x0, default_units)
    y0 = _ensure_unit(y0, default_units)
    x1 = _ensure_unit(x1, default_units)
    y1 = _ensure_unit(y1, default_units)
    return Grob(
        x0=x0, y0=y0, x1=x1, y1=y1, arrow=arrow,
        name=name, gp=gp, vp=vp, _grid_class="segments",
    )


def grid_segments(
    x0: Any = 0,
    y0: Any = 0,
    x1: Any = 1,
    y1: Any = 1,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *segments* grob.

    Parameters
    ----------
    x0 : Unit or numeric
        Horizontal start coordinate(s).
    y0 : Unit or numeric
        Vertical start coordinate(s).
    x1 : Unit or numeric
        Horizontal end coordinate(s).
    y1 : Unit or numeric
        Vertical end coordinate(s).
    default_units : str
        Default unit type for bare numerics.
    arrow : Arrow or None
        Optional arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = segments_grob(
        x0=x0, y0=y0, x1=x1, y1=y1,
        default_units=default_units, arrow=arrow,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  arrows primitive (defunct in R -- kept for API completeness)           #
# ===================================================================== #


def arrows_grob(
    x: Any = None,
    y: Any = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create an *arrows* grob.

    .. deprecated::
        The standalone arrows grob is defunct in R (>= 2.3.0).  Use the
        ``arrow`` argument on line-drawing primitives instead.  This
        constructor is retained for API completeness and backward
        compatibility.

    Parameters
    ----------
    x : Unit, numeric, or None
        Horizontal coordinates.  Defaults to ``Unit([0.25, 0.75], "npc")``.
    y : Unit, numeric, or None
        Vertical coordinates.  Defaults to ``Unit(0.5, "npc")``.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Arrow-head specification.  When ``None`` a default arrow is used
        internally at draw time.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="arrows"``.
    """
    import warnings

    warnings.warn(
        "'arrows_grob' is deprecated; use the 'arrow' argument on "
        "line-drawing primitives instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if x is None:
        x = Unit([0.25, 0.75], "npc")
    else:
        x = _ensure_unit(x, default_units)
    if y is None:
        y = Unit(0.5, "npc")
    else:
        y = _ensure_unit(y, default_units)
    return Grob(
        x=x, y=y, arrow=arrow,
        name=name, gp=gp, vp=vp, _grid_class="arrows",
    )


def grid_arrows(
    x: Any = None,
    y: Any = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw an *arrows* grob.

    .. deprecated::
        Use the ``arrow`` argument on line-drawing primitives instead.

    Parameters
    ----------
    x : Unit, numeric, or None
        Horizontal coordinates.
    y : Unit, numeric, or None
        Vertical coordinates.
    default_units : str
        Default unit type for bare numerics.
    arrow : Arrow or None
        Arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    import warnings

    warnings.warn(
        "'grid_arrows' is deprecated; use the 'arrow' argument on "
        "line-drawing primitives instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    grob = arrows_grob(
        x=x, y=y, default_units=default_units, arrow=arrow,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  points primitive                                                      #
# ===================================================================== #


def points_grob(
    x: Any = None,
    y: Any = None,
    size: Optional[Any] = None,
    default_units: str = "npc",
    pch: Union[int, str] = 1,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *points* grob.

    A *points* grob draws symbols (plotting characters) at the given
    locations, analogous to R's ``points()``.

    Parameters
    ----------
    x : Unit, numeric, or None
        Horizontal coordinates.  When ``None`` a default is chosen.
    y : Unit, numeric, or None
        Vertical coordinates.  When ``None`` a default is chosen.
    size : Unit, numeric, or None
        Symbol size.  Defaults to ``Unit(1, "char")`` when ``None``.
    default_units : str
        Unit type for bare numerics (default ``"npc"``).
    pch : int or str
        Plotting character / symbol code.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="points"``.
    """
    if x is None:
        x = Unit(0.5, "npc")
    else:
        x = _ensure_unit(x, default_units)
    if y is None:
        y = Unit(0.5, "npc")
    else:
        y = _ensure_unit(y, default_units)
    if size is None:
        size = Unit(1, "char")
    else:
        size = _ensure_unit(size, default_units)
    return Grob(
        x=x, y=y, pch=pch, size=size,
        name=name, gp=gp, vp=vp, _grid_class="points",
    )


def grid_points(
    x: Any = None,
    y: Any = None,
    size: Optional[Any] = None,
    default_units: str = "npc",
    pch: Union[int, str] = 1,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *points* grob.

    Parameters
    ----------
    x : Unit, numeric, or None
        Horizontal coordinates.
    y : Unit, numeric, or None
        Vertical coordinates.
    size : Unit, numeric, or None
        Symbol size.
    default_units : str
        Default unit type for bare numerics.
    pch : int or str
        Plotting character / symbol code.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = points_grob(
        x=x, y=y, size=size, default_units=default_units, pch=pch,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  rect primitive                                                        #
# ===================================================================== #


def rect_grob(
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *rect* grob.

    A *rect* grob draws one or more axis-aligned rectangles.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit or numeric
        Rectangle width(s).
    height : Unit or numeric
        Rectangle height(s).
    default_units : str
        Unit type for bare numerics.
    just : str or sequence of str
        Justification specification (e.g. ``"centre"``, ``"left"``,
        ``["left", "bottom"]``).
    hjust : float or None
        Explicit horizontal justification override.
    vjust : float or None
        Explicit vertical justification override.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="rect"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    width = _ensure_unit(width, default_units)
    height = _ensure_unit(height, default_units)
    return Grob(
        x=x, y=y, width=width, height=height,
        just=just, hjust=hjust, vjust=vjust,
        name=name, gp=gp, vp=vp, _grid_class="rect",
    )


def grid_rect(
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *rect* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit or numeric
        Rectangle width(s).
    height : Unit or numeric
        Rectangle height(s).
    default_units : str
        Default unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = rect_grob(
        x=x, y=y, width=width, height=height,
        default_units=default_units, just=just,
        hjust=hjust, vjust=vjust,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  roundrect primitive                                                   #
# ===================================================================== #


def roundrect_grob(
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    default_units: str = "npc",
    r: Optional[Any] = None,
    just: Union[str, Sequence[str]] = "centre",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *roundrect* grob.

    A *roundrect* grob draws a single rectangle with rounded corners.
    Corner radius *r* is best specified as an absolute unit or ``"snpc"``
    to avoid distortion.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit or numeric
        Rectangle width.
    height : Unit or numeric
        Rectangle height.
    default_units : str
        Unit type for bare numerics.
    r : Unit, numeric, or None
        Corner radius.  Defaults to ``Unit(0.1, "snpc")`` when ``None``.
    just : str or sequence of str
        Justification specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="roundrect"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    width = _ensure_unit(width, default_units)
    height = _ensure_unit(height, default_units)
    if r is None:
        r = Unit(0.1, "snpc")
    elif not is_unit(r):
        r = Unit(r, default_units)
    return Grob(
        x=x, y=y, width=width, height=height, r=r, just=just,
        name=name, gp=gp, vp=vp, _grid_class="roundrect",
    )


def grid_roundrect(
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    default_units: str = "npc",
    r: Optional[Any] = None,
    just: Union[str, Sequence[str]] = "centre",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *roundrect* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit or numeric
        Rectangle width.
    height : Unit or numeric
        Rectangle height.
    default_units : str
        Default unit type for bare numerics.
    r : Unit, numeric, or None
        Corner radius.  Defaults to ``Unit(0.1, "snpc")``.
    just : str or sequence of str
        Justification specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = roundrect_grob(
        x=x, y=y, width=width, height=height,
        default_units=default_units, r=r, just=just,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  circle primitive                                                      #
# ===================================================================== #


def circle_grob(
    x: Any = 0.5,
    y: Any = 0.5,
    r: Any = 0.5,
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *circle* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal centre.
    y : Unit or numeric
        Vertical centre.
    r : Unit or numeric
        Radius.
    default_units : str
        Unit type for bare numerics.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="circle"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    r = _ensure_unit(r, default_units)
    return Grob(
        x=x, y=y, r=r,
        name=name, gp=gp, vp=vp, _grid_class="circle",
    )


def grid_circle(
    x: Any = 0.5,
    y: Any = 0.5,
    r: Any = 0.5,
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *circle* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal centre.
    y : Unit or numeric
        Vertical centre.
    r : Unit or numeric
        Radius.
    default_units : str
        Default unit type for bare numerics.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = circle_grob(
        x=x, y=y, r=r, default_units=default_units,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  polygon primitive                                                     #
# ===================================================================== #


def polygon_grob(
    x: Any = None,
    y: Any = None,
    id: Optional[Sequence[int]] = None,
    id_lengths: Optional[Sequence[int]] = None,
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *polygon* grob.

    A *polygon* grob draws one or more filled polygons.  Sub-polygons
    are identified by *id* or *id_lengths* (mutually exclusive).

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinates.  Defaults to a diamond shape.
    y : Unit or numeric
        Vertical coordinates.  Defaults to a diamond shape.
    id : sequence of int or None
        Per-point group identifier.
    id_lengths : sequence of int or None
        Lengths of consecutive sub-polygons.
    default_units : str
        Unit type for bare numerics.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="polygon"``.

    Raises
    ------
    ValueError
        If both *id* and *id_lengths* are specified.
    """
    if x is None:
        x = Unit([0, 0.5, 1, 0.5], "npc")
    else:
        x = _ensure_unit(x, default_units)
    if y is None:
        y = Unit([0.5, 1, 0.5, 0], "npc")
    else:
        y = _ensure_unit(y, default_units)
    if id is not None and id_lengths is not None:
        raise ValueError(
            "it is invalid to specify both 'id' and 'id_lengths'"
        )
    return Grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        name=name, gp=gp, vp=vp, _grid_class="polygon",
    )


def grid_polygon(
    x: Any = None,
    y: Any = None,
    id: Optional[Sequence[int]] = None,
    id_lengths: Optional[Sequence[int]] = None,
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *polygon* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinates.
    y : Unit or numeric
        Vertical coordinates.
    id : sequence of int or None
        Per-point group identifier.
    id_lengths : sequence of int or None
        Lengths of consecutive sub-polygons.
    default_units : str
        Default unit type for bare numerics.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = polygon_grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        default_units=default_units,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  path primitive                                                        #
# ===================================================================== #


def path_grob(
    x: Any,
    y: Any,
    id: Optional[Sequence[int]] = None,
    id_lengths: Optional[Sequence[int]] = None,
    path_id: Optional[Sequence[int]] = None,
    path_id_lengths: Optional[Sequence[int]] = None,
    rule: str = "winding",
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *path* grob.

    A *path* grob draws a complex filled shape defined by one or more
    sub-paths.  The fill rule (*rule*) determines how overlapping
    sub-paths interact.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinates.
    y : Unit or numeric
        Vertical coordinates.
    id : sequence of int or None
        Per-point sub-path identifier.
    id_lengths : sequence of int or None
        Lengths of consecutive sub-paths.
    path_id : sequence of int or None
        Identifier for grouping sub-paths into separate compound paths.
    path_id_lengths : sequence of int or None
        Lengths of consecutive compound path groups.
    rule : {"winding", "evenodd"}
        Fill rule.
    default_units : str
        Unit type for bare numerics.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="pathgrob"``.

    Raises
    ------
    ValueError
        If both *id* and *id_lengths* are specified, or if *rule* is
        invalid.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    if id is not None and id_lengths is not None:
        raise ValueError(
            "it is invalid to specify both 'id' and 'id_lengths'"
        )
    if rule not in ("winding", "evenodd"):
        raise ValueError(f"'rule' must be 'winding' or 'evenodd', got {rule!r}")
    return Grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        path_id=path_id, path_id_lengths=path_id_lengths,
        rule=rule,
        name=name, gp=gp, vp=vp, _grid_class="pathgrob",
    )


def grid_path(
    x: Any = None,
    y: Any = None,
    id: Optional[Sequence[int]] = None,
    id_lengths: Optional[Sequence[int]] = None,
    path_id: Optional[Sequence[int]] = None,
    path_id_lengths: Optional[Sequence[int]] = None,
    rule: str = "winding",
    default_units: str = "npc",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *path* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal coordinates.
    y : Unit or numeric
        Vertical coordinates.
    id : sequence of int or None
        Per-point sub-path identifier.
    id_lengths : sequence of int or None
        Lengths of consecutive sub-paths.
    path_id : sequence of int or None
        Identifier for compound path groups.
    path_id_lengths : sequence of int or None
        Lengths of compound path groups.
    rule : {"winding", "evenodd"}
        Fill rule.
    default_units : str
        Default unit type for bare numerics.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    if x is None:
        x = Unit([0, 0.5, 1, 0.5], "npc")
    if y is None:
        y = Unit([0.5, 1, 0.5, 0], "npc")
    grob = path_grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        path_id=path_id, path_id_lengths=path_id_lengths,
        rule=rule, default_units=default_units,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  text primitive                                                        #
# ===================================================================== #


def text_grob(
    label: Any,
    x: Any = 0.5,
    y: Any = 0.5,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    rot: float = 0,
    check_overlap: bool = False,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *text* grob.

    A *text* grob renders one or more text strings at the given
    positions.

    Parameters
    ----------
    label : str or sequence of str
        Text string(s) to display.
    x : Unit or numeric
        Horizontal position(s).
    y : Unit or numeric
        Vertical position(s).
    default_units : str
        Unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    rot : float
        Rotation angle in degrees.
    check_overlap : bool
        If ``True``, labels that overlap previously drawn labels are
        suppressed.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="text"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    return Grob(
        label=label, x=x, y=y,
        just=just, hjust=hjust, vjust=vjust,
        rot=float(rot), check_overlap=bool(check_overlap),
        name=name, gp=gp, vp=vp, _grid_class="text",
    )


def grid_text(
    label: Any,
    x: Any = 0.5,
    y: Any = 0.5,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    rot: float = 0,
    check_overlap: bool = False,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *text* grob.

    Parameters
    ----------
    label : str or sequence of str
        Text string(s) to display.
    x : Unit or numeric
        Horizontal position(s).
    y : Unit or numeric
        Vertical position(s).
    default_units : str
        Default unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    rot : float
        Rotation angle in degrees.
    check_overlap : bool
        Whether to suppress overlapping labels.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = text_grob(
        label=label, x=x, y=y, default_units=default_units,
        just=just, hjust=hjust, vjust=vjust,
        rot=rot, check_overlap=check_overlap,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  raster primitive                                                      #
# ===================================================================== #


def raster_grob(
    image: Any,
    x: Any = 0.5,
    y: Any = 0.5,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    interpolate: bool = True,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *raster* grob.

    A *raster* grob draws a raster image (2-D pixel array) at the
    specified location.  When *width* or *height* is ``None``, the
    missing dimension is inferred from the image aspect ratio at draw
    time.

    Parameters
    ----------
    image : array-like
        The raster image data (e.g. a NumPy array or PIL Image).
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit, numeric, or None
        Image width.  ``None`` means infer from aspect ratio.
    height : Unit, numeric, or None
        Image height.  ``None`` means infer from aspect ratio.
    default_units : str
        Unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    interpolate : bool
        If ``True`` (default), the image is linearly interpolated when
        scaled.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="rastergrob"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    if width is not None:
        width = _ensure_unit(width, default_units)
    if height is not None:
        height = _ensure_unit(height, default_units)
    return Grob(
        raster=image, x=x, y=y, width=width, height=height,
        just=just, hjust=hjust, vjust=vjust,
        interpolate=bool(interpolate),
        name=name, gp=gp, vp=vp, _grid_class="rastergrob",
    )


def grid_raster(
    image: Any,
    x: Any = 0.5,
    y: Any = 0.5,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    interpolate: bool = True,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *raster* grob.

    Parameters
    ----------
    image : array-like
        The raster image data.
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit, numeric, or None
        Image width.
    height : Unit, numeric, or None
        Image height.
    default_units : str
        Default unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    interpolate : bool
        Whether to interpolate when scaling.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = raster_grob(
        image=image, x=x, y=y, width=width, height=height,
        default_units=default_units, just=just,
        hjust=hjust, vjust=vjust, interpolate=interpolate,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  clip primitive                                                        #
# ===================================================================== #


def clip_grob(
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    name: Optional[str] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *clip* grob.

    A *clip* grob sets the clipping region to the given rectangle.
    Subsequent drawing is clipped to this area until the viewport is
    popped or the clipping region is reset.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit or numeric
        Clip rectangle width.
    height : Unit or numeric
        Clip rectangle height.
    default_units : str
        Unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    name : str or None
        Grob name.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="clip"``.

    Notes
    -----
    The *clip* grob does **not** accept a *gp* argument -- graphical
    parameters are irrelevant for clipping.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    width = _ensure_unit(width, default_units)
    height = _ensure_unit(height, default_units)
    return Grob(
        x=x, y=y, width=width, height=height,
        just=just, hjust=hjust, vjust=vjust,
        name=name, vp=vp, _grid_class="clip",
    )


def grid_clip(
    x: Any = 0.5,
    y: Any = 0.5,
    width: Any = 1,
    height: Any = 1,
    default_units: str = "npc",
    just: Union[str, Sequence[str]] = "centre",
    hjust: Optional[float] = None,
    vjust: Optional[float] = None,
    name: Optional[str] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *clip* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal anchor.
    y : Unit or numeric
        Vertical anchor.
    width : Unit or numeric
        Clip rectangle width.
    height : Unit or numeric
        Clip rectangle height.
    default_units : str
        Default unit type for bare numerics.
    just : str or sequence of str
        Justification specification.
    hjust : float or None
        Horizontal justification override.
    vjust : float or None
        Vertical justification override.
    name : str or None
        Grob name.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = clip_grob(
        x=x, y=y, width=width, height=height,
        default_units=default_units, just=just,
        hjust=hjust, vjust=vjust,
        name=name, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  null primitive                                                        #
# ===================================================================== #


def null_grob(
    x: Any = 0.5,
    y: Any = 0.5,
    default_units: str = "npc",
    name: Optional[str] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *null* grob.

    A *null* grob has guaranteed zero width and height and draws
    nothing.  It is useful as a positional anchor for other grobs.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal position.
    y : Unit or numeric
        Vertical position.
    default_units : str
        Unit type for bare numerics.
    name : str or None
        Grob name.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="null"``.
    """
    x = _ensure_unit(x, default_units)
    y = _ensure_unit(y, default_units)
    return Grob(x=x, y=y, name=name, vp=vp, _grid_class="null")


def grid_null(
    x: Any = 0.5,
    y: Any = 0.5,
    default_units: str = "npc",
    name: Optional[str] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *null* grob.

    Parameters
    ----------
    x : Unit or numeric
        Horizontal position.
    y : Unit or numeric
        Vertical position.
    default_units : str
        Default unit type for bare numerics.
    name : str or None
        Grob name.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = null_grob(x=x, y=y, default_units=default_units, name=name, vp=vp)
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  function grob                                                         #
# ===================================================================== #


def function_grob(
    fn: Callable[..., Any],
    n: int = 101,
    range: Union[str, Sequence[float]] = "x",
    units: str = "native",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *function* grob.

    A *function* grob evaluates a user-supplied function over a
    regularly-spaced grid and draws the resulting curve as a *lines*
    grob at draw time.

    Parameters
    ----------
    fn : callable
        A function that accepts a 1-D numeric input and returns a dict
        (or object) with ``"x"`` and ``"y"`` keys/attributes giving the
        output coordinates.
    n : int
        Number of evaluation points (default 101).
    range : {"x", "y"} or sequence of float
        Input range.  ``"x"`` or ``"y"`` means use the corresponding
        viewport scale; a 2-element numeric sequence gives explicit
        bounds.
    units : str
        Unit type for the coordinates produced by *fn*.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="functiongrob"``.

    Raises
    ------
    ValueError
        If *n* < 1.
    TypeError
        If *fn* is not callable.
    """
    if n < 1:
        raise ValueError("'n' must be >= 1")
    if not callable(fn):
        raise TypeError("'fn' must be callable")
    return Grob(
        f=fn, n=n, range=range, units=units,
        name=name, gp=gp, vp=vp, _grid_class="functiongrob",
    )


def grid_function(
    fn: Callable[..., Any],
    n: int = 101,
    range: Union[str, Sequence[float]] = "x",
    units: str = "native",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *function* grob.

    Parameters
    ----------
    fn : callable
        Evaluation function (see :func:`function_grob`).
    n : int
        Number of evaluation points.
    range : {"x", "y"} or sequence of float
        Input range.
    units : str
        Unit type for output coordinates.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The created grob.
    """
    grob = function_grob(
        fn=fn, n=n, range=range, units=units,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  Path fill / stroke primitives (R 4.2+, grid/R/path.R)                #
# ===================================================================== #


def as_path(
    x: Any,
    gp: Optional[Gpar] = None,
    rule: str = "winding",
) -> dict:
    """Mark a grob as a single path.

    Mirrors R's ``as.path()`` (``path.R:3-9``).

    Parameters
    ----------
    x : Grob
        The grob to convert.
    gp : Gpar or None
        Graphical parameters.
    rule : str
        Fill rule: ``"winding"`` (default) or ``"evenodd"``.

    Returns
    -------
    dict
        A ``GridPath`` descriptor with keys ``grob``, ``gp``, ``rule``.
    """
    return {
        "grob": x,
        "gp": gp if gp is not None else Gpar(),
        "rule": rule,
        "_class": "GridPath",
    }


def stroke_grob(
    x: Any,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a stroke-only path grob.

    Draws the outline of the nested grob without filling.
    Mirrors R's ``strokeGrob()`` (``path.R:20-35``).

    Parameters
    ----------
    x : Grob or GridPath dict
        The grob (or ``as_path`` result) defining the path.
    name, gp, vp
        Standard grob parameters.

    Returns
    -------
    Grob
        A grob with ``_grid_class="GridStroke"``.
    """
    if isinstance(x, dict) and x.get("_class") == "GridPath":
        return Grob(
            path=x["grob"], name=name, gp=x["gp"], vp=vp,
            _grid_class="GridStroke",
        )
    return Grob(
        path=x, name=name, gp=gp if gp is not None else Gpar(), vp=vp,
        _grid_class="GridStroke",
    )


def grid_stroke(x: Any, **kwargs: Any) -> Grob:
    """Create and draw a stroke-only path grob.

    Mirrors R's ``grid.stroke()``.
    """
    grob = stroke_grob(x, **kwargs)
    _grid_draw(grob)
    return grob


def fill_grob(
    x: Any,
    rule: str = "winding",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a fill-only path grob.

    Fills the interior of the nested grob without stroking the outline.
    Mirrors R's ``fillGrob()`` (``path.R:49-64``).

    Parameters
    ----------
    x : Grob or GridPath dict
        The grob (or ``as_path`` result) defining the path.
    rule : str
        Fill rule: ``"winding"`` or ``"evenodd"``.
    name, gp, vp
        Standard grob parameters.

    Returns
    -------
    Grob
        A grob with ``_grid_class="GridFill"``.
    """
    if isinstance(x, dict) and x.get("_class") == "GridPath":
        return Grob(
            path=x["grob"], rule=x["rule"], name=name, gp=x["gp"], vp=vp,
            _grid_class="GridFill",
        )
    return Grob(
        path=x, rule=rule, name=name,
        gp=gp if gp is not None else Gpar(), vp=vp,
        _grid_class="GridFill",
    )


def grid_fill(x: Any, **kwargs: Any) -> Grob:
    """Create and draw a fill-only path grob.

    Mirrors R's ``grid.fill()``.
    """
    grob = fill_grob(x, **kwargs)
    _grid_draw(grob)
    return grob


def fill_stroke_grob(
    x: Any,
    rule: str = "winding",
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a fill-and-stroke path grob.

    Fills and then strokes the nested grob as a single combined path.
    Mirrors R's ``fillStrokeGrob()`` (``path.R:80-95``).

    Parameters
    ----------
    x : Grob or GridPath dict
        The grob (or ``as_path`` result) defining the path.
    rule : str
        Fill rule: ``"winding"`` or ``"evenodd"``.
    name, gp, vp
        Standard grob parameters.

    Returns
    -------
    Grob
        A grob with ``_grid_class="GridFillStroke"``.
    """
    if isinstance(x, dict) and x.get("_class") == "GridPath":
        return Grob(
            path=x["grob"], rule=x["rule"], name=name, gp=x["gp"], vp=vp,
            _grid_class="GridFillStroke",
        )
    return Grob(
        path=x, rule=rule, name=name,
        gp=gp if gp is not None else Gpar(), vp=vp,
        _grid_class="GridFillStroke",
    )


def grid_fill_stroke(x: Any, **kwargs: Any) -> Grob:
    """Create and draw a fill-and-stroke path grob.

    Mirrors R's ``grid.fillStroke()``.
    """
    grob = fill_stroke_grob(x, **kwargs)
    _grid_draw(grob)
    return grob
