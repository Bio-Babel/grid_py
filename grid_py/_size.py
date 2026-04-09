"""Size and metric computation for grid_py (port of R's grid ``size.R``).

This module provides functions for computing grob dimensions (width, height,
ascent, descent) and text string metrics using Cairo's font engine.
These mirror the ``widthDetails``, ``heightDetails``, ``xDetails``,
``yDetails``, ``ascentDetails``, and ``descentDetails`` generics in R's
grid package.

The ``calc_string_metric`` function measures text using Cairo's FreeType-backed
font engine and returns ascent, descent, and width in inches.  The ``grob_*``
helpers create :class:`Unit` objects whose unit type references a grob,
paralleling R's ``"grobwidth"``, ``"grobheight"``, etc. unit family.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import cairo

from ._gpar import Gpar
from ._units import Unit

__all__ = [
    "calc_string_metric",
    "grob_width",
    "grob_height",
    "grob_x",
    "grob_y",
    "grob_ascent",
    "grob_descent",
    "width_details",
    "height_details",
    "ascent_details",
    "descent_details",
    "x_details",
    "y_details",
    "absolute_size",
]


# ---------------------------------------------------------------------------
# Cairo font helpers
# ---------------------------------------------------------------------------

# Shared measurement surface (tiny ImageSurface; never written to file).
_MEASURE_SURFACE: Optional[cairo.ImageSurface] = None
_MEASURE_CTX: Optional[cairo.Context] = None


def _get_measure_ctx() -> cairo.Context:
    """Return a Cairo context used solely for text measurement."""
    global _MEASURE_SURFACE, _MEASURE_CTX
    if _MEASURE_CTX is None:
        _MEASURE_SURFACE = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        _MEASURE_CTX = cairo.Context(_MEASURE_SURFACE)
    return _MEASURE_CTX


def _apply_font_from_gpar(
    ctx: cairo.Context,
    gp: Optional[Gpar] = None,
) -> float:
    """Configure *ctx*'s font from a :class:`Gpar` and return the font size
    in **points**.

    Parameters
    ----------
    ctx : cairo.Context
        The Cairo context to configure.
    gp : Gpar or None
        Graphical parameters.  If ``None``, defaults (sans-serif, 12 pt)
        are used.

    Returns
    -------
    float
        The resolved font size in points.
    """
    family = "sans-serif"
    slant = cairo.FONT_SLANT_NORMAL
    weight = cairo.FONT_WEIGHT_NORMAL
    fontsize = 12.0  # points

    if gp is not None:
        ff = gp.get("fontfamily", None)
        if ff is not None:
            family = str(ff[0] if isinstance(ff, (list, tuple)) else ff)

        fs = gp.get("fontsize", None)
        if fs is not None:
            fontsize = float(fs[0] if isinstance(fs, (list, tuple)) else fs)

        cex = gp.get("cex", None)
        if cex is not None:
            fontsize *= float(cex[0] if isinstance(cex, (list, tuple)) else cex)

        face = gp.get("fontface", None)
        if face is not None:
            val = face[0] if isinstance(face, (list, tuple)) else face
            if isinstance(val, str):
                val = val.lower()
            if val in (2, "bold"):
                weight = cairo.FONT_WEIGHT_BOLD
            elif val in (3, "italic", "oblique"):
                slant = cairo.FONT_SLANT_ITALIC
            elif val in (4, "bold.italic"):
                weight = cairo.FONT_WEIGHT_BOLD
                slant = cairo.FONT_SLANT_ITALIC

    ctx.select_font_face(family, slant, weight)
    ctx.set_font_size(fontsize)  # in points (measurement context has no scaling)
    return fontsize


# ---------------------------------------------------------------------------
# String metrics
# ---------------------------------------------------------------------------


def calc_string_metric(
    text: str,
    gp: Optional[Gpar] = None,
) -> Dict[str, float]:
    """Compute text metrics (ascent, descent, width) in inches.

    Uses Cairo's FreeType-backed font engine to measure the given *text*
    string with the font described by *gp*.

    Parameters
    ----------
    text : str
        The string to measure.
    gp : Gpar or None, optional
        Graphical parameters controlling the font family, size, and style.
        When ``None``, Cairo defaults (sans-serif, 12 pt) are used.

    Returns
    -------
    dict
        A dictionary with keys ``"ascent"``, ``"descent"``, and ``"width"``,
        each a ``float`` value in inches.

    Examples
    --------
    >>> m = calc_string_metric("Hello")
    >>> sorted(m.keys())
    ['ascent', 'descent', 'width']
    """
    ctx = _get_measure_ctx()
    _apply_font_from_gpar(ctx, gp)

    # font_extents: (ascent, descent, height, max_x_advance, max_y_advance)
    fe = ctx.font_extents()
    # text_extents: (x_bearing, y_bearing, width, height, x_advance, y_advance)
    te = ctx.text_extents(text)

    pts_per_inch = 72.0
    ascent = fe[0] / pts_per_inch
    descent = fe[1] / pts_per_inch
    width = te[4] / pts_per_inch  # x_advance

    return {"ascent": ascent, "descent": descent, "width": width}


# ---------------------------------------------------------------------------
# Generic detail dispatchers (mirroring R's S3 method dispatch)
# ---------------------------------------------------------------------------


def width_details(x: Any) -> Unit:
    """Return the width of grob *x*.

    Parameters
    ----------
    x : Grob
        A graphical object.  If it defines a ``width_details`` method, that
        is called; otherwise a ``Unit(1, "null")`` is returned.

    Returns
    -------
    Unit
        The width as a grid unit.
    """
    if hasattr(x, "width_details") and callable(x.width_details):
        return x.width_details()
    return Unit(1, "null")


def height_details(x: Any) -> Unit:
    """Return the height of grob *x*.

    Parameters
    ----------
    x : Grob
        A graphical object.

    Returns
    -------
    Unit
        The height as a grid unit.
    """
    if hasattr(x, "height_details") and callable(x.height_details):
        return x.height_details()
    return Unit(1, "null")


def ascent_details(x: Any) -> Unit:
    """Return the text ascent of grob *x*.

    Parameters
    ----------
    x : Grob
        A graphical object.

    Returns
    -------
    Unit
        The ascent as a grid unit.  Falls back to ``height_details`` for
        grobs that do not define ``ascent_details``.
    """
    if hasattr(x, "ascent_details") and callable(x.ascent_details):
        return x.ascent_details()
    return height_details(x)


def descent_details(x: Any) -> Unit:
    """Return the text descent of grob *x*.

    Parameters
    ----------
    x : Grob
        A graphical object.

    Returns
    -------
    Unit
        The descent as a grid unit.  Default is ``Unit(0, "inches")``.
    """
    if hasattr(x, "descent_details") and callable(x.descent_details):
        return x.descent_details()
    return Unit(0, "inches")


def x_details(x: Any, theta: float = 0) -> Unit:
    """Return the x position on the edge of grob *x* at angle *theta*.

    Parameters
    ----------
    x : Grob
        A graphical object.
    theta : float, optional
        Angle in degrees (default ``0``).

    Returns
    -------
    Unit
        The x position as a grid unit.  Default is ``Unit(0.5, "npc")``.
    """
    if hasattr(x, "x_details") and callable(x.x_details):
        return x.x_details(theta)
    return Unit(0.5, "npc")


def y_details(x: Any, theta: float = 0) -> Unit:
    """Return the y position on the edge of grob *x* at angle *theta*.

    Parameters
    ----------
    x : Grob
        A graphical object.
    theta : float, optional
        Angle in degrees (default ``0``).

    Returns
    -------
    Unit
        The y position as a grid unit.  Default is ``Unit(0.5, "npc")``.
    """
    if hasattr(x, "y_details") and callable(x.y_details):
        return x.y_details(theta)
    return Unit(0.5, "npc")


# ---------------------------------------------------------------------------
# grob_* convenience constructors
# ---------------------------------------------------------------------------


def grob_width(x: Any) -> Unit:
    """Create a ``"grobwidth"`` unit referencing grob *x*.

    Parameters
    ----------
    x : Grob
        The graphical object whose width is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobwidth"`` with *x* stored as auxiliary data.

    Examples
    --------
    >>> from grid_py._grob import Grob
    >>> g = Grob(name="test")
    >>> u = grob_width(g)
    >>> u._units[0]
    'grobwidth'
    """
    return Unit(1, "grobwidth", data=x)


def grob_height(x: Any) -> Unit:
    """Create a ``"grobheight"`` unit referencing grob *x*.

    Parameters
    ----------
    x : Grob
        The graphical object whose height is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobheight"`` with *x* stored as auxiliary data.
    """
    return Unit(1, "grobheight", data=x)


def grob_x(x: Any, theta: float = 0) -> Unit:
    """Create a ``"grobx"`` unit referencing grob *x* at angle *theta*.

    Parameters
    ----------
    x : Grob
        The graphical object.
    theta : float, optional
        Angle in degrees (default ``0``).

    Returns
    -------
    Unit
        A unit of type ``"grobx"`` with ``(x, theta)`` stored as data.
    """
    return Unit(theta, "grobx", data=x)


def grob_y(x: Any, theta: float = 0) -> Unit:
    """Create a ``"groby"`` unit referencing grob *x* at angle *theta*.

    Parameters
    ----------
    x : Grob
        The graphical object.
    theta : float, optional
        Angle in degrees (default ``0``).

    Returns
    -------
    Unit
        A unit of type ``"groby"`` with ``(x, theta)`` stored as data.
    """
    return Unit(theta, "groby", data=x)


def grob_ascent(x: Any) -> Unit:
    """Create a ``"grobascent"`` unit referencing grob *x*.

    Parameters
    ----------
    x : Grob
        The graphical object whose text ascent is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobascent"`` with *x* stored as auxiliary data.
    """
    return Unit(1, "grobascent", data=x)


def grob_descent(x: Any) -> Unit:
    """Create a ``"grobdescent"`` unit referencing grob *x*.

    Parameters
    ----------
    x : Grob
        The graphical object whose text descent is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobdescent"`` with *x* stored as auxiliary data.
    """
    return Unit(1, "grobdescent", data=x)


# ---------------------------------------------------------------------------
# absolute_size
# ---------------------------------------------------------------------------


def absolute_size(u: Unit) -> Unit:
    """Return absolute components of *u*; replace relative ones with null.

    For units that do not depend on the parent drawing context (e.g.
    ``"inches"``, ``"cm"``, ``"mm"``), the value is returned unchanged.
    Context-dependent units (e.g. ``"npc"``, ``"native"``) are replaced
    with ``Unit(1, "null")``.  This mirrors R's ``absolute.size()``.

    Parameters
    ----------
    u : Unit
        The unit to filter.

    Returns
    -------
    Unit
        A new unit with only absolute components retained.
    """
    from ._units import absolute_size as _absolute_size  # avoid shadowing

    return _absolute_size(u)
