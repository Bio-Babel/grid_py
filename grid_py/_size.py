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
# _grid_class-specific metric implementations
# (mirrors R's S3 methods in primitives.R)
# ---------------------------------------------------------------------------


def _normalise_labels(grob: Any) -> list:
    """Extract *label* from a grob as a plain Python list of strings."""
    labels = getattr(grob, "label", "")
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, (list, tuple)):
        return [str(l) for l in labels]
    # numpy array or other iterable
    try:
        return [str(l) for l in labels]
    except TypeError:
        return [str(labels)]


# -- text grob (R: primitives.R:1430-1470) ---------------------------------

def _text_width_details(grob: Any) -> Unit:
    """Width of a text grob: max width across all labels.

    Mirrors ``widthDetails.text`` (R ``primitives.R:1430``).
    """
    labels = _normalise_labels(grob)
    gp = getattr(grob, "gp", None)
    if not labels:
        return Unit(0, "inches")
    max_width = 0.0
    for lab in labels:
        m = calc_string_metric(lab, gp=gp)
        max_width = max(max_width, m["width"])
    return Unit(max_width, "inches")


def _text_height_details(grob: Any) -> Unit:
    """Height of a text grob: ascent + descent of the first label.

    Mirrors ``heightDetails.text`` (R ``primitives.R:1442``).
    """
    labels = _normalise_labels(grob)
    gp = getattr(grob, "gp", None)
    text = labels[0] if labels else ""
    m = calc_string_metric(text, gp=gp)
    return Unit(m["ascent"] + m["descent"], "inches")


def _text_ascent_details(grob: Any) -> Unit:
    """Ascent of a text grob.

    For a single label, returns the font ascent.  For multiple labels,
    falls back to ``_text_height_details``.

    Mirrors ``ascentDetails.text`` (R ``primitives.R:1454``).
    """
    labels = _normalise_labels(grob)
    gp = getattr(grob, "gp", None)
    if len(labels) == 1:
        m = calc_string_metric(labels[0], gp=gp)
        return Unit(m["ascent"], "inches")
    return _text_height_details(grob)


def _text_descent_details(grob: Any) -> Unit:
    """Descent of a text grob.

    For a single label, returns the font descent.  For multiple labels,
    returns ``Unit(0, "inches")``.

    Mirrors ``descentDetails.text`` (R ``primitives.R:1463``).
    """
    labels = _normalise_labels(grob)
    gp = getattr(grob, "gp", None)
    if len(labels) == 1:
        m = calc_string_metric(labels[0], gp=gp)
        return Unit(m["descent"], "inches")
    return Unit(0, "inches")


# -- null grob (R: primitives.R:1676-1682) ---------------------------------

def _null_width_details(grob: Any) -> Unit:
    """Width of a null grob: always zero.

    Mirrors ``widthDetails.null`` (R ``primitives.R:1676``).
    """
    return Unit(0, "inches")


def _null_height_details(grob: Any) -> Unit:
    """Height of a null grob: always zero.

    Mirrors ``heightDetails.null`` (R ``primitives.R:1680``).
    """
    return Unit(0, "inches")


# -- rect grob (R: primitives.R:1146-1166) ---------------------------------

def _rect_width_details(grob: Any) -> Unit:
    """Width of a rect grob: its own *width* attribute.

    Mirrors ``widthDetails.rect`` (R ``primitives.R:1146``).
    """
    w = getattr(grob, "width", None)
    if w is not None and isinstance(w, Unit):
        return w
    return Unit(1, "npc")


def _rect_height_details(grob: Any) -> Unit:
    """Height of a rect grob: its own *height* attribute.

    Mirrors ``heightDetails.rect`` (R ``primitives.R:1157``).
    """
    h = getattr(grob, "height", None)
    if h is not None and isinstance(h, Unit):
        return h
    return Unit(1, "npc")


# -- coordinate-based bounding box helpers ----------------------------------
#
# R uses C_locnBounds (for lines, points, polygon, polyline, segments)
# and C_circleBounds (for circles).  Both resolve all coordinates to
# inches in the current viewport context, then compute min/max.
#
# In grid_py we achieve the same by obtaining the active renderer
# and calling ``renderer.resolve_to_npc()`` on each coordinate.


def _get_renderer() -> Any:
    """Return the active renderer, or ``None`` if none is bound."""
    from ._state import get_state
    state = get_state()
    return state.get_renderer()


def _locn_bounds_width(x_unit: Any, renderer: Any, gp: Any = None) -> float:
    """Compute the width (in inches) of a set of x-coordinates.

    Mirrors R ``C_locnBounds`` returning ``bounds[3]`` (width).
    """
    from ._units import Unit
    if not isinstance(x_unit, Unit) or len(x_unit) == 0:
        return 0.0

    npc_vals = []
    for i in range(len(x_unit)):
        elem = Unit(x_unit._values[i], x_unit._units[i], data=x_unit._data[i])
        npc_vals.append(renderer.resolve_to_npc(elem, axis="x", is_dim=False, gp=gp))

    if not npc_vals:
        return 0.0
    npc_width = max(npc_vals) - min(npc_vals)
    # Convert NPC width to inches
    _x0, _y0, pw, _ph = renderer.get_viewport_bounds()
    return npc_width * pw / renderer.dpi


def _locn_bounds_height(y_unit: Any, renderer: Any, gp: Any = None) -> float:
    """Compute the height (in inches) of a set of y-coordinates.

    Mirrors R ``C_locnBounds`` returning ``bounds[4]`` (height).
    """
    from ._units import Unit
    if not isinstance(y_unit, Unit) or len(y_unit) == 0:
        return 0.0

    npc_vals = []
    for i in range(len(y_unit)):
        elem = Unit(y_unit._values[i], y_unit._units[i], data=y_unit._data[i])
        npc_vals.append(renderer.resolve_to_npc(elem, axis="y", is_dim=False, gp=gp))

    if not npc_vals:
        return 0.0
    npc_height = max(npc_vals) - min(npc_vals)
    _x0, _y0, _pw, ph = renderer.get_viewport_bounds()
    return npc_height * ph / renderer.dpi


# -- lines grob (R: primitives.R:186-200, uses C_locnBounds) ---------------

def _lines_width_details(grob: Any) -> Unit:
    """Width of a lines/polyline grob: bounding box of x-coordinates.

    Mirrors ``widthDetails.lines`` (R ``primitives.R:186``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x_unit = getattr(grob, "x", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_width(x_unit, renderer, gp), "inches")


def _lines_height_details(grob: Any) -> Unit:
    """Height of a lines/polyline grob.

    Mirrors ``heightDetails.lines`` (R ``primitives.R:194``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y_unit = getattr(grob, "y", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_height(y_unit, renderer, gp), "inches")


# -- points grob (R: primitives.R:1546-1560, uses C_locnBounds) ------------

def _points_width_details(grob: Any) -> Unit:
    """Width of a points grob: bounding box of x-coordinates.

    Mirrors ``widthDetails.points`` (R ``primitives.R:1546``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x_unit = getattr(grob, "x", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_width(x_unit, renderer, gp), "inches")


def _points_height_details(grob: Any) -> Unit:
    """Height of a points grob.

    Mirrors ``heightDetails.points`` (R ``primitives.R:1554``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y_unit = getattr(grob, "y", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_height(y_unit, renderer, gp), "inches")


# -- polygon grob (R: primitives.R:607-621, uses C_locnBounds) -------------

def _polygon_width_details(grob: Any) -> Unit:
    """Width of a polygon grob.

    Mirrors ``widthDetails.polygon`` (R ``primitives.R:607``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x_unit = getattr(grob, "x", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_width(x_unit, renderer, gp), "inches")


def _polygon_height_details(grob: Any) -> Unit:
    """Height of a polygon grob.

    Mirrors ``heightDetails.polygon`` (R ``primitives.R:615``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y_unit = getattr(grob, "y", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_height(y_unit, renderer, gp), "inches")


# -- segments grob (R: primitives.R:367-381, uses segmentBounds helper) -----

def _segments_width_details(grob: Any) -> Unit:
    """Width of a segments grob: bounding box of all endpoints.

    Mirrors ``widthDetails.segments`` (R ``primitives.R:367``).
    R's ``segmentBounds`` concatenates x0,x1 and y0,y1 into single
    vectors, then calls ``C_locnBounds``.
    """
    from ._units import Unit as _Unit, unit_c
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x0 = getattr(grob, "x0", None)
    x1 = getattr(grob, "x1", None)
    gp = getattr(grob, "gp", None)
    if x0 is not None and x1 is not None:
        if isinstance(x0, _Unit) and isinstance(x1, _Unit):
            combined_x = unit_c(x0, x1)
        else:
            combined_x = x0
    elif x0 is not None:
        combined_x = x0
    elif x1 is not None:
        combined_x = x1
    else:
        return Unit(0, "inches")
    return Unit(_locn_bounds_width(combined_x, renderer, gp), "inches")


def _segments_height_details(grob: Any) -> Unit:
    """Height of a segments grob.

    Mirrors ``heightDetails.segments`` (R ``primitives.R:375``).
    """
    from ._units import Unit as _Unit, unit_c
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y0 = getattr(grob, "y0", None)
    y1 = getattr(grob, "y1", None)
    gp = getattr(grob, "gp", None)
    if y0 is not None and y1 is not None:
        if isinstance(y0, _Unit) and isinstance(y1, _Unit):
            combined_y = unit_c(y0, y1)
        else:
            combined_y = y0
    elif y0 is not None:
        combined_y = y0
    elif y1 is not None:
        combined_y = y1
    else:
        return Unit(0, "inches")
    return Unit(_locn_bounds_height(combined_y, renderer, gp), "inches")


# -- circle grob (R: primitives.R:1062-1076, uses C_circleBounds) ----------

def _circle_width_details(grob: Any) -> Unit:
    """Width of a circle grob: bounding box considering radius.

    Mirrors ``widthDetails.circle`` (R ``primitives.R:1062``).
    R's ``C_circleBounds`` computes ``(max(x+r) - min(x-r))``.
    """
    from ._units import Unit as _Unit
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")

    x_unit = getattr(grob, "x", None)
    r_unit = getattr(grob, "r", None)
    gp = getattr(grob, "gp", None)

    if x_unit is None or not isinstance(x_unit, _Unit):
        return Unit(0, "inches")

    # Resolve each center and radius to NPC, then compute bounds
    n = len(x_unit)
    r_npc_arr = []
    if r_unit is not None and isinstance(r_unit, _Unit):
        for i in range(len(r_unit)):
            elem = _Unit(r_unit._values[i], r_unit._units[i], data=r_unit._data[i])
            r_npc_arr.append(renderer.resolve_to_npc(elem, axis="x", is_dim=True, gp=gp))
    if not r_npc_arr:
        r_npc_arr = [0.0]

    x_npc_arr = []
    for i in range(n):
        elem = _Unit(x_unit._values[i], x_unit._units[i], data=x_unit._data[i])
        x_npc_arr.append(renderer.resolve_to_npc(elem, axis="x", is_dim=False, gp=gp))

    # Compute bounding box with radius
    max_vals = []
    min_vals = []
    for i in range(n):
        r_npc = r_npc_arr[i % len(r_npc_arr)]
        max_vals.append(x_npc_arr[i] + r_npc)
        min_vals.append(x_npc_arr[i] - r_npc)

    npc_width = max(max_vals) - min(min_vals)
    _x0, _y0, vp_pw, _ph = renderer.get_viewport_bounds()
    return Unit(npc_width * vp_pw / renderer.dpi, "inches")


def _circle_height_details(grob: Any) -> Unit:
    """Height of a circle grob.

    Mirrors ``heightDetails.circle`` (R ``primitives.R:1070``).
    """
    from ._units import Unit as _Unit
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")

    y_unit = getattr(grob, "y", None)
    r_unit = getattr(grob, "r", None)
    gp = getattr(grob, "gp", None)

    if y_unit is None or not isinstance(y_unit, _Unit):
        return Unit(0, "inches")

    n = len(y_unit)
    r_npc_arr = []
    if r_unit is not None and isinstance(r_unit, _Unit):
        for i in range(len(r_unit)):
            elem = _Unit(r_unit._values[i], r_unit._units[i], data=r_unit._data[i])
            r_npc_arr.append(renderer.resolve_to_npc(elem, axis="y", is_dim=True, gp=gp))
    if not r_npc_arr:
        r_npc_arr = [0.0]

    y_npc_arr = []
    for i in range(n):
        elem = _Unit(y_unit._values[i], y_unit._units[i], data=y_unit._data[i])
        y_npc_arr.append(renderer.resolve_to_npc(elem, axis="y", is_dim=False, gp=gp))

    max_vals = []
    min_vals = []
    for i in range(n):
        r_npc = r_npc_arr[i % len(r_npc_arr)]
        max_vals.append(y_npc_arr[i] + r_npc)
        min_vals.append(y_npc_arr[i] - r_npc)

    npc_height = max(max_vals) - min(min_vals)
    _x0, _y0, _pw, vp_ph = renderer.get_viewport_bounds()
    return Unit(npc_height * vp_ph / renderer.dpi, "inches")


# -- roundrect grob (same as rect: returns own width/height) ----------------

def _roundrect_width_details(grob: Any) -> Unit:
    """Width of a roundrect grob: its own *width* attribute.

    Mirrors ``widthDetails.roundrect`` — same as rect.
    """
    return _rect_width_details(grob)


def _roundrect_height_details(grob: Any) -> Unit:
    """Height of a roundrect grob: its own *height* attribute."""
    return _rect_height_details(grob)


# -- pathgrob (coordinate bounding box, same pattern as polygon) ------------

def _path_width_details(grob: Any) -> Unit:
    """Width of a path grob: bounding box of x-coordinates.

    Mirrors ``widthDetails.path`` (uses ``C_locnBounds``).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x_unit = getattr(grob, "x", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_width(x_unit, renderer, gp), "inches")


def _path_height_details(grob: Any) -> Unit:
    """Height of a path grob."""
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y_unit = getattr(grob, "y", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_height(y_unit, renderer, gp), "inches")


# -- rastergrob (returns own width/height, same as rect) --------------------

def _raster_width_details(grob: Any) -> Unit:
    """Width of a raster grob: its own *width* attribute.

    Mirrors ``widthDetails.rastergrob``.
    """
    return _rect_width_details(grob)


def _raster_height_details(grob: Any) -> Unit:
    """Height of a raster grob: its own *height* attribute."""
    return _rect_height_details(grob)


# -- xspline grob (R: primitives.R:845-861, uses C_xsplineBounds) ----------

def _xspline_width_details(grob: Any) -> Unit:
    """Width of an xspline grob: bounding box of evaluated spline points.

    R calls ``C_xsplineBounds`` which evaluates the spline and computes
    bounds.  We use ``xspline_points()`` for the same effect.

    Mirrors ``widthDetails.xspline`` (R ``primitives.R:845``).
    """
    try:
        from ._curve import xspline_points
        pts = xspline_points(grob)
        if len(pts["x"]) == 0:
            return Unit(0, "inches")
        renderer = _get_renderer()
        if renderer is not None:
            _, _, vp_pw, _ = renderer.get_viewport_bounds()
            npc_width = float(pts["x"].max() - pts["x"].min())
            return Unit(npc_width * vp_pw / renderer.dpi, "inches")
        return Unit(float(pts["x"].max() - pts["x"].min()), "npc")
    except Exception:
        return Unit(0, "inches")


def _xspline_height_details(grob: Any) -> Unit:
    """Height of an xspline grob.

    Mirrors ``heightDetails.xspline`` (R ``primitives.R:853``).
    """
    try:
        from ._curve import xspline_points
        pts = xspline_points(grob)
        if len(pts["y"]) == 0:
            return Unit(0, "inches")
        renderer = _get_renderer()
        if renderer is not None:
            _, _, _, vp_ph = renderer.get_viewport_bounds()
            npc_height = float(pts["y"].max() - pts["y"].min())
            return Unit(npc_height * vp_ph / renderer.dpi, "inches")
        return Unit(float(pts["y"].max() - pts["y"].min()), "npc")
    except Exception:
        return Unit(0, "inches")


# -- bezier grob (R: primitives.R:997-1003, expands via splinegrob()) ------

def _bezier_width_details(grob: Any) -> Unit:
    """Width of a bezier grob: bounding box of evaluated Bezier points.

    R's ``widthDetails.beziergrob`` calls ``splinegrob()`` to expand to
    an xsplineGrob, then delegates.  We use ``bezier_points()`` directly.

    Mirrors ``widthDetails.beziergrob`` (R ``primitives.R:997``).
    """
    try:
        from ._curve import bezier_points
        pts = bezier_points(grob)
        if len(pts["x"]) == 0:
            return Unit(0, "inches")
        renderer = _get_renderer()
        if renderer is not None:
            _, _, vp_pw, _ = renderer.get_viewport_bounds()
            npc_width = float(pts["x"].max() - pts["x"].min())
            return Unit(npc_width * vp_pw / renderer.dpi, "inches")
        return Unit(float(pts["x"].max() - pts["x"].min()), "npc")
    except Exception:
        return Unit(0, "inches")


def _bezier_height_details(grob: Any) -> Unit:
    """Height of a bezier grob.

    Mirrors ``heightDetails.beziergrob`` (R ``primitives.R:1001``).
    """
    try:
        from ._curve import bezier_points
        pts = bezier_points(grob)
        if len(pts["y"]) == 0:
            return Unit(0, "inches")
        renderer = _get_renderer()
        if renderer is not None:
            _, _, _, vp_ph = renderer.get_viewport_bounds()
            npc_height = float(pts["y"].max() - pts["y"].min())
            return Unit(npc_height * vp_ph / renderer.dpi, "inches")
        return Unit(float(pts["y"].max() - pts["y"].min()), "npc")
    except Exception:
        return Unit(0, "inches")


# -- curve grob (R: curve.R:481-495, expands via calcCurveGrob()) ----------

def _curve_width_details(grob: Any) -> Unit:
    """Width of a curve grob: expand to control points, compute bbox.

    R's ``widthDetails.curve`` calls ``calcCurveGrob()`` to build the
    expanded gTree, then delegates to children's widthDetails.  We
    compute the curve control points directly.

    Mirrors ``widthDetails.curve`` (R ``curve.R:481``).
    """
    try:
        from ._curve import _calc_curve_points
        from ._units import Unit as _Unit
        x1 = getattr(grob, "x1", None)
        y1 = getattr(grob, "y1", None)
        x2 = getattr(grob, "x2", None)
        y2 = getattr(grob, "y2", None)
        if x1 is None or x2 is None:
            return Unit(0, "inches")
        x1v = float(x1._values[0]) if isinstance(x1, _Unit) else float(x1)
        y1v = float(y1._values[0]) if isinstance(y1, _Unit) else float(y1)
        x2v = float(x2._values[0]) if isinstance(x2, _Unit) else float(x2)
        y2v = float(y2._values[0]) if isinstance(y2, _Unit) else float(y2)

        px, py = _calc_curve_points(
            x1v, y1v, x2v, y2v,
            curvature=float(getattr(grob, "curvature", 1.0)),
            angle=float(getattr(grob, "angle", 90.0)),
            ncp=int(getattr(grob, "ncp", 1)),
            shape=float(getattr(grob, "shape", 0.5)),
            square=bool(getattr(grob, "square", True)),
            squareShape=float(getattr(grob, "squareShape", 1.0)),
            inflect=bool(getattr(grob, "inflect", False)),
            open_=bool(getattr(grob, "open_", getattr(grob, "open", True))),
        )
        if len(px) == 0:
            return Unit(0, "inches")
        renderer = _get_renderer()
        if renderer is not None:
            _, _, vp_pw, _ = renderer.get_viewport_bounds()
            npc_width = float(px.max() - px.min())
            return Unit(npc_width * vp_pw / renderer.dpi, "inches")
        return Unit(float(px.max() - px.min()), "npc")
    except Exception:
        return Unit(0, "inches")


def _curve_height_details(grob: Any) -> Unit:
    """Height of a curve grob.

    Mirrors ``heightDetails.curve`` (R ``curve.R:489``).
    """
    try:
        from ._curve import _calc_curve_points
        from ._units import Unit as _Unit
        x1 = getattr(grob, "x1", None)
        y1 = getattr(grob, "y1", None)
        x2 = getattr(grob, "x2", None)
        y2 = getattr(grob, "y2", None)
        if y1 is None or y2 is None:
            return Unit(0, "inches")
        x1v = float(x1._values[0]) if isinstance(x1, _Unit) else float(x1)
        y1v = float(y1._values[0]) if isinstance(y1, _Unit) else float(y1)
        x2v = float(x2._values[0]) if isinstance(x2, _Unit) else float(x2)
        y2v = float(y2._values[0]) if isinstance(y2, _Unit) else float(y2)

        px, py = _calc_curve_points(
            x1v, y1v, x2v, y2v,
            curvature=float(getattr(grob, "curvature", 1.0)),
            angle=float(getattr(grob, "angle", 90.0)),
            ncp=int(getattr(grob, "ncp", 1)),
            shape=float(getattr(grob, "shape", 0.5)),
            square=bool(getattr(grob, "square", True)),
            squareShape=float(getattr(grob, "squareShape", 1.0)),
            inflect=bool(getattr(grob, "inflect", False)),
            open_=bool(getattr(grob, "open_", getattr(grob, "open", True))),
        )
        if len(py) == 0:
            return Unit(0, "inches")
        renderer = _get_renderer()
        if renderer is not None:
            _, _, _, vp_ph = renderer.get_viewport_bounds()
            npc_height = float(py.max() - py.min())
            return Unit(npc_height * vp_ph / renderer.dpi, "inches")
        return Unit(float(py.max() - py.min()), "npc")
    except Exception:
        return Unit(0, "inches")


# ---------------------------------------------------------------------------
# _grid_class dispatch tables
# ---------------------------------------------------------------------------

_WIDTH_DISPATCH: Dict[str, Any] = {
    "text": _text_width_details,
    "null": _null_width_details,
    "rect": _rect_width_details,
    "roundrect": _roundrect_width_details,
    "lines": _lines_width_details,
    "polyline": _lines_width_details,
    "points": _points_width_details,
    "polygon": _polygon_width_details,
    "segments": _segments_width_details,
    "circle": _circle_width_details,
    "pathgrob": _path_width_details,
    "rastergrob": _raster_width_details,
    "xspline": _xspline_width_details,
    "beziergrob": _bezier_width_details,
    "curve": _curve_width_details,
}

_HEIGHT_DISPATCH: Dict[str, Any] = {
    "text": _text_height_details,
    "null": _null_height_details,
    "rect": _rect_height_details,
    "roundrect": _roundrect_height_details,
    "lines": _lines_height_details,
    "polyline": _lines_height_details,
    "points": _points_height_details,
    "polygon": _polygon_height_details,
    "segments": _segments_height_details,
    "circle": _circle_height_details,
    "pathgrob": _path_height_details,
    "rastergrob": _raster_height_details,
    "xspline": _xspline_height_details,
    "beziergrob": _bezier_height_details,
    "curve": _curve_height_details,
}

_ASCENT_DISPATCH: Dict[str, Any] = {
    "text": _text_ascent_details,
}

_DESCENT_DISPATCH: Dict[str, Any] = {
    "text": _text_descent_details,
}


# ---------------------------------------------------------------------------
# Generic detail dispatchers (mirroring R's S3 method dispatch)
# ---------------------------------------------------------------------------


def width_details(x: Any) -> Unit:
    """Return the width of grob *x*.

    Dispatches first by ``_grid_class`` attribute (text, null, rect),
    then by ``width_details`` method on the object, and finally falls
    back to ``Unit(1, "null")``.

    Parameters
    ----------
    x : Grob
        A graphical object.

    Returns
    -------
    Unit
        The width as a grid unit.
    """
    cls = getattr(x, "_grid_class", None)
    handler = _WIDTH_DISPATCH.get(cls)
    if handler is not None:
        return handler(x)
    if hasattr(x, "width_details") and callable(x.width_details):
        result = x.width_details()
        if result is not None:
            return result
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
    cls = getattr(x, "_grid_class", None)
    handler = _HEIGHT_DISPATCH.get(cls)
    if handler is not None:
        return handler(x)
    if hasattr(x, "height_details") and callable(x.height_details):
        result = x.height_details()
        if result is not None:
            return result
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
    cls = getattr(x, "_grid_class", None)
    handler = _ASCENT_DISPATCH.get(cls)
    if handler is not None:
        return handler(x)
    if hasattr(x, "ascent_details") and callable(x.ascent_details):
        result = x.ascent_details()
        if result is not None:
            return result
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
    cls = getattr(x, "_grid_class", None)
    handler = _DESCENT_DISPATCH.get(cls)
    if handler is not None:
        return handler(x)
    if hasattr(x, "descent_details") and callable(x.descent_details):
        result = x.descent_details()
        if result is not None:
            return result
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
