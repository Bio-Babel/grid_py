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

import math
from typing import Any, Dict, Optional, Union

import cairo
import numpy as np

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

    Returns **text-specific** ascent and descent (like R's ``GEStrMetric``),
    not font-level values.  For example, ``"H"`` has descent ≈ 0 because
    it has no descender strokes, while ``"g"`` has a positive descent.

    To avoid integer-quantisation artifacts from Cairo's toy font API at
    small point sizes, measurements are taken at a scaled-up font size
    and normalised back.

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
    fontsize = _apply_font_from_gpar(ctx, gp)

    # --- Scaled measurement to defeat integer quantisation -------
    # Cairo's toy font API quantises font_extents and text_extents
    # to integer user-units.  At small point sizes (e.g. 8.8 pt)
    # this introduces large relative errors.  By measuring at
    # SCALE× the requested size and dividing back, we recover
    # sub-pixel precision.
    _SCALE = 100
    ctx.set_font_size(fontsize * _SCALE)

    # text_extents: (x_bearing, y_bearing, width, height, x_advance, y_advance)
    te = ctx.text_extents(text)

    pts_per_inch = 72.0

    # Text-specific ascent and descent from text_extents
    # (matches R's GEStrMetric which returns per-string metrics):
    #   ascent  = -y_bearing  (baseline to top of ink)
    #   descent = height + y_bearing  (baseline to bottom of ink; ≥0)
    #   width   = x_advance  (total advance width)
    ascent = (-te[1]) / _SCALE / pts_per_inch
    descent = max(0.0, (te[3] + te[1])) / _SCALE / pts_per_inch
    width = te[4] / _SCALE / pts_per_inch

    # Restore the original font size on the shared context.
    ctx.set_font_size(fontsize)

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

def _resolve_grob_gp(grob: Any) -> "Optional[Gpar]":
    """Resolve a grob's gpar by merging with the current viewport stack.

    Port of R's ``resolveGPar`` — the grob's own gp overrides inherited
    values from the viewport gpar stack, matching R's ``C_textBounds``
    which always uses the fully-resolved gpar.
    """
    grob_gp = getattr(grob, "gp", None)
    try:
        from ._gpar import get_gpar
        vp_gp = get_gpar()
    except Exception:
        return grob_gp
    if grob_gp is None:
        return vp_gp
    if vp_gp is None:
        return grob_gp
    return grob_gp._merge(vp_gp)


def _text_bbox(grob: Any) -> tuple:
    """Compute (width, height) of the text bounding box in inches.

    Port of R's ``widthDetails.text`` / ``heightDetails.text``
    (primitives.R:1430-1452) → ``C_textBounds``: the union of every
    placement's justified+rotated label box INCLUDING the anchor
    positions (see ``_text_placement_corners``).

    R's ``grobHeight`` on a text grob
    returns the **ascent only** (glyph extent above baseline), never
    the descent. ``grobDescent`` is a separate method (see
    ``descentDetails.text``) and is exposed independently so that
    callers can add it when needed. ggplot2's ``titleGrob``
    (margins.R:115-132) relies on this convention — it uses
    ``unit(1, "grobheight", grob) + y_descent`` to assemble the final
    height — so any deviation here double-counts the descent.

    Empirical verification (R 4.4 ``cairo_png`` at 150 dpi, default
    Helvetica, fontsize 13.2 pt):

      grobHeight(textGrob("A long title", fs=13.2))         = 3.293 mm
      grobHeight(textGrob("gjpqy",        fs=13.2))         = 3.293 mm
      grobHeight(textGrob("y",            fs=13.2))         = 3.293 mm

    — i.e. the value is a font-level constant, independent of the
    label content. Our cairo-backed ``calc_string_metric`` returns a
    per-label ascent that varies slightly with glyph mix, which is the
    closest we can get without implementing a full AFM font-metric
    path; residual ≤ 0.3 mm discrepancy is a font-file difference
    (AFM Helvetica vs. cairo's "Sans" fallback) and outside the
    scope of this bbox function.

    Height formula (port of R's ``GEStrHeight``):

      width  = max(per-line ink widths)
      height = ascent(first line)
               + (n - 1) × cex × lineheight × fontsize × 1.2 / 72

    The per-extra-line gap ``1.2 × fontsize / 72`` is R's device-level
    ``cra[1] × ipr[1] / default_ps`` collapsed for the standard cairo /
    PostScript setups (default_ps = 12 pt, cin[1] = 1.2 × 12 / 72 in).

    Single-line text is therefore independent of ``lineheight`` (matches
    R exactly), while each extra newline adds the lineheight-scaled gap.

    The grob's gp is merged with the current viewport stack gpar, so
    ``grobHeight`` inherits fontsize / lineheight from the viewport.
    """
    quads = _text_placement_corners(grob)
    if quads is None:
        return (0.0, 0.0)
    xs = [c for q in quads for c in q[0]]
    ys = [c for q in quads for c in q[1]]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _text_width_details(grob: Any) -> Unit:
    """Width of a text grob: rotated bounding box width.

    Port of R ``widthDetails.text`` (primitives.R:1430) which calls
    ``C_textBounds`` with rotation to compute the axis-aligned bbox.
    """
    w, _ = _text_bbox(grob)
    return Unit(w, "inches")


def _text_height_details(grob: Any) -> Unit:
    """Height of a text grob: rotated bounding box height.

    Port of R ``heightDetails.text`` (primitives.R:1442) which calls
    ``C_textBounds`` with rotation to compute the axis-aligned bbox.
    """
    _, h = _text_bbox(grob)
    return Unit(h, "inches")


def _text_ascent_details(grob: Any) -> Unit:
    """Ascent of a text grob.

    For a single label, returns the font ascent.  For multiple labels,
    falls back to ``_text_height_details``.

    Mirrors ``ascentDetails.text`` (R ``primitives.R:1454``).
    """
    labels = _normalise_labels(grob)
    gp = _resolve_grob_gp(grob)
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
    gp = _resolve_grob_gp(grob)
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
    """Width of a rect grob: actual bounding box width in inches.

    Port of R ``widthDetails.rect`` (primitives.R:1146) which calls
    ``C_rectBounds``.  Resolves x, width, hjust to inches, computes
    the bounding box ``xmax - xmin`` across all rectangles.
    """
    from ._just import resolve_hjust, resolve_vjust

    renderer = _get_renderer()
    if renderer is None:
        # Fallback: return the raw width attribute
        w = getattr(grob, "width", None)
        if w is not None and isinstance(w, Unit):
            return w
        return Unit(1, "npc")

    gp = getattr(grob, "gp", None)
    x_unit = getattr(grob, "x", None)
    w_unit = getattr(grob, "width", None)
    if x_unit is None or w_unit is None:
        return Unit(0, "inches")

    just = getattr(grob, "just", None) or "centre"
    hjust_val = getattr(grob, "hjust", None)
    hjust = resolve_hjust(just, hjust_val)

    n = max(len(x_unit), len(w_unit))
    xmin = float("inf")
    xmax = float("-inf")
    for i in range(n):
        cx = renderer._resolve_to_inches_idx(x_unit, i % len(x_unit), "x", False, gp)
        w = renderer._resolve_to_inches_idx(w_unit, i % len(w_unit), "x", True, gp)
        left = cx - hjust * w
        right = left + w
        if left < xmin:
            xmin = left
        if right > xmax:
            xmax = right

    if xmin == float("inf"):
        return Unit(0, "inches")
    return Unit(xmax - xmin, "inches")


def _rect_height_details(grob: Any) -> Unit:
    """Height of a rect grob: actual bounding box height in inches.

    Port of R ``heightDetails.rect`` (primitives.R:1157) which calls
    ``C_rectBounds``.
    """
    from ._just import resolve_hjust, resolve_vjust

    renderer = _get_renderer()
    if renderer is None:
        h = getattr(grob, "height", None)
        if h is not None and isinstance(h, Unit):
            return h
        return Unit(1, "npc")

    gp = getattr(grob, "gp", None)
    y_unit = getattr(grob, "y", None)
    h_unit = getattr(grob, "height", None)
    if y_unit is None or h_unit is None:
        return Unit(0, "inches")

    just = getattr(grob, "just", None) or "centre"
    vjust_val = getattr(grob, "vjust", None)
    vjust = resolve_vjust(just, vjust_val)

    n = max(len(y_unit), len(h_unit))
    ymin = float("inf")
    ymax = float("-inf")
    for i in range(n):
        cy = renderer._resolve_to_inches_idx(y_unit, i % len(y_unit), "y", False, gp)
        h = renderer._resolve_to_inches_idx(h_unit, i % len(h_unit), "y", True, gp)
        bottom = cy - vjust * h
        top = bottom + h
        if bottom < ymin:
            ymin = bottom
        if top > ymax:
            ymax = top

    if ymin == float("inf"):
        return Unit(0, "inches")
    return Unit(ymax - ymin, "inches")


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


def _locn_bounds_inches(
    unit_obj: Any, renderer: Any, axis: str, gp: Any = None,
) -> tuple:
    """Resolve all elements of a Unit to inches and return (min, max).

    Port of R ``C_locnBounds`` (grid.c:5296-5376): resolves each coordinate
    to inches via ``transformXtoINCHES``/``transformYtoINCHES``, then computes
    the bounding box.

    Parameters
    ----------
    unit_obj : Unit
        Coordinate unit.
    renderer : object
        Active renderer with ``_resolve_to_inches_idx``.
    axis : str
        ``"x"`` or ``"y"``.
    gp : object, optional
        Graphical parameters.

    Returns
    -------
    tuple
        ``(min_inches, max_inches)`` or ``(0.0, 0.0)`` if empty.
    """
    from ._units import Unit
    if not isinstance(unit_obj, Unit) or len(unit_obj) == 0:
        return (0.0, 0.0)

    xmin = float("inf")
    xmax = float("-inf")
    n = len(unit_obj)
    for i in range(n):
        val = renderer._resolve_to_inches_idx(unit_obj, i, axis, False, gp)
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val

    if xmin == float("inf"):
        return (0.0, 0.0)
    return (xmin, xmax)


def _locn_bounds_width(x_unit: Any, renderer: Any, gp: Any = None) -> float:
    """Compute the width (in inches) of a set of x-coordinates.

    Port of R ``C_locnBounds`` returning ``bounds[3]`` (width = xmax - xmin).
    Uses the inches-based pipeline (not NPC).
    """
    lo, hi = _locn_bounds_inches(x_unit, renderer, "x", gp)
    return hi - lo


def _locn_bounds_height(y_unit: Any, renderer: Any, gp: Any = None) -> float:
    """Compute the height (in inches) of a set of y-coordinates.

    Port of R ``C_locnBounds`` returning ``bounds[4]`` (height = ymax - ymin).
    Uses the inches-based pipeline (not NPC).
    """
    lo, hi = _locn_bounds_inches(y_unit, renderer, "y", gp)
    return hi - lo


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

    Port of R ``widthDetails.circle`` (primitives.R:1062).
    R's ``C_circleBounds`` computes ``max(cx+r) - min(cx-r)`` in inches.
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

    n = len(x_unit)
    nr = len(r_unit) if r_unit is not None and isinstance(r_unit, _Unit) else 0

    xmin = float("inf")
    xmax = float("-inf")
    for i in range(n):
        cx = renderer._resolve_to_inches_idx(x_unit, i, "x", False, gp)
        if nr > 0:
            # R: r = pmin(convertWidth(r), convertHeight(r))
            rw = renderer._resolve_to_inches_idx(r_unit, i % nr, "x", True, gp)
            rh = renderer._resolve_to_inches_idx(r_unit, i % nr, "y", True, gp)
            r = min(rw, rh)
        else:
            r = 0.0
        left = cx - r
        right = cx + r
        if left < xmin:
            xmin = left
        if right > xmax:
            xmax = right

    if xmin == float("inf"):
        return Unit(0, "inches")
    return Unit(xmax - xmin, "inches")


def _circle_height_details(grob: Any) -> Unit:
    """Height of a circle grob.

    Port of R ``heightDetails.circle`` (primitives.R:1070).
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
    nr = len(r_unit) if r_unit is not None and isinstance(r_unit, _Unit) else 0

    ymin = float("inf")
    ymax = float("-inf")
    for i in range(n):
        cy = renderer._resolve_to_inches_idx(y_unit, i, "y", False, gp)
        if nr > 0:
            rw = renderer._resolve_to_inches_idx(r_unit, i % nr, "x", True, gp)
            rh = renderer._resolve_to_inches_idx(r_unit, i % nr, "y", True, gp)
            r = min(rw, rh)
        else:
            r = 0.0
        bottom = cy - r
        top = cy + r
        if bottom < ymin:
            ymin = bottom
        if top > ymax:
            ymax = top

    if ymin == float("inf"):
        return Unit(0, "inches")
    return Unit(ymax - ymin, "inches")


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
    """Width of a raster grob.

    Port of R ``widthDetails.rastergrob`` (primitives.R:1313) — uses
    ``C_rectBounds`` after resolving raster size.  Same logic as rect.
    """
    return _rect_width_details(grob)


def _raster_height_details(grob: Any) -> Unit:
    """Height of a raster grob.

    Port of R ``heightDetails.rastergrob`` (primitives.R:1325).
    """
    return _rect_height_details(grob)


# -- xspline grob (R: primitives.R:845-861, uses C_xsplineBounds) ----------

def _xspline_eval_bounds(grob: Any) -> Optional[tuple]:
    """Evaluate an xspline grob's curve and return its inch bounds.

    Port of R ``widthDetails.xspline`` / ``heightDetails.xspline``
    (primitives.R:845-861) → ``C_xsplineBounds`` (grid.c gridXspline
    with ``draw=FALSE``): the control points are converted to inches in
    the current context, the spline is evaluated — R passes
    ``list(seq_along(x$x))``, i.e. ALL control points as a single spline
    even when ``id`` is set — and the bounds of the evaluated points are
    returned as ``(xmin, xmax, ymin, ymax)`` (``None`` when there are no
    finite points).
    """
    from ._curve import _calc_xspline_points, _device_size_in
    from ._units import convert_x, convert_y

    xx = np.atleast_1d(np.asarray(
        convert_x(grob.x, "inches", valueOnly=True), dtype=np.float64))
    yy = np.atleast_1d(np.asarray(
        convert_y(grob.y, "inches", valueOnly=True), dtype=np.float64))
    shape = np.resize(
        np.atleast_1d(np.asarray(getattr(grob, "shape", 0.0),
                                 dtype=np.float64)),
        len(xx))
    px, py = _calc_xspline_points(
        xx, yy, shape,
        open_=bool(getattr(grob, "open_", True)),
        repEnds=bool(getattr(grob, "repEnds", True)),
        units_per_inch=1.0,
        device_size_in=_device_size_in(),
    )
    finite = np.isfinite(px) & np.isfinite(py)
    if not np.any(finite):
        return None
    px = px[finite]
    py = py[finite]
    return (float(px.min()), float(px.max()),
            float(py.min()), float(py.max()))


def _xspline_width_details(grob: Any) -> Unit:
    """Width of an xspline grob: bounds of the EVALUATED curve.

    Port of R ``widthDetails.xspline`` (primitives.R:845-852).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    bounds = _xspline_eval_bounds(grob)
    if bounds is None:
        return Unit(0, "inches")
    return Unit(bounds[1] - bounds[0], "inches")


def _xspline_height_details(grob: Any) -> Unit:
    """Height of an xspline grob: bounds of the EVALUATED curve.

    Port of R ``heightDetails.xspline`` (primitives.R:854-861).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    bounds = _xspline_eval_bounds(grob)
    if bounds is None:
        return Unit(0, "inches")
    return Unit(bounds[3] - bounds[2], "inches")


# -- bezier grob (R: primitives.R:997-1003, expands via splinegrob()) ------

def _bezier_width_details(grob: Any) -> Unit:
    """Width of a bezier grob.

    Port of R ``widthDetails.beziergrob`` (primitives.R:997-999):
    delegates to the X-spline approximation of the Bezier.
    """
    from ._curve import _splinegrob
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    return _xspline_width_details(_splinegrob(grob))


def _bezier_height_details(grob: Any) -> Unit:
    """Height of a bezier grob.

    Port of R ``heightDetails.beziergrob`` (primitives.R:1001-1003):
    delegates to the X-spline approximation of the Bezier.
    """
    from ._curve import _splinegrob
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    return _xspline_height_details(_splinegrob(grob))


# -- curve grob (R: curve.R:481-495, expands via calcCurveGrob()) ----------

def _curve_width_details(grob: Any) -> Unit:
    """Width of a curve grob.

    Port of R ``widthDetails.curve`` (curve.R:481).  R expands to a
    child grob then delegates.  We use the endpoint bounding box.
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    from ._units import Unit as _Unit, unit_c
    x1 = getattr(grob, "x1", None)
    x2 = getattr(grob, "x2", None)
    gp = getattr(grob, "gp", None)
    if x1 is not None and x2 is not None and isinstance(x1, _Unit) and isinstance(x2, _Unit):
        combined = unit_c(x1, x2)
        return Unit(_locn_bounds_width(combined, renderer, gp), "inches")
    return Unit(0, "inches")


def _curve_height_details(grob: Any) -> Unit:
    """Height of a curve grob.

    Port of R ``heightDetails.curve`` (curve.R:489).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    from ._units import Unit as _Unit, unit_c
    y1 = getattr(grob, "y1", None)
    y2 = getattr(grob, "y2", None)
    gp = getattr(grob, "gp", None)
    if y1 is not None and y2 is not None and isinstance(y1, _Unit) and isinstance(y2, _Unit):
        combined = unit_c(y1, y2)
        return Unit(_locn_bounds_height(combined, renderer, gp), "inches")
    return Unit(0, "inches")


# ---------------------------------------------------------------------------
# xDetails / yDetails edge geometry — "the point on the edge of a grob at
# angle theta" (degrees, 0 = East, 90 = North).  Ports of grid.c rectEdge /
# circleEdge / polygonEdge / hullEdge plus the !draw branches of the
# per-class C bounds functions (L_locnBounds, gridRect, gridCircle,
# gridText, gridXspline).  All coordinates are inches within the current
# viewport.  Results are NOT divided by GSS_SCALE, consistent with the
# width/height handlers above (zoom is handled at unit-resolution).
# ---------------------------------------------------------------------------


def _rect_edge(xmin: float, ymin: float, xmax: float, ymax: float,
               theta: float) -> tuple:
    """Point on the edge of a rectangle at angle *theta*.

    Port of ``rectEdge`` (grid.c:1751-1800), including the exact special
    cases for 0/90/180/270 degrees.
    """
    xm = (xmin + xmax) / 2.0
    ym = (ymin + ymax) / 2.0
    dx = (xmax - xmin) / 2.0
    dy = (ymax - ymin) / 2.0
    if theta == 0:
        return (xmax, ym)
    if theta == 270:
        return (xm, ymin)
    if theta == 180:
        return (xmin, ym)
    if theta == 90:
        return (xm, ymax)
    # C computes dy/dx without a zero guard (IEEE inf); replicate that
    cutoff = math.inf if dx == 0.0 else dy / dx
    angle = theta / 180.0 * math.pi
    tan_theta = math.tan(angle)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    if abs(tan_theta) < cutoff:
        if cos_theta > 0:
            return (xmax, ym + tan_theta * dx)
        return (xmin, ym - tan_theta * dx)
    if sin_theta > 0:
        return (xm + dy / tan_theta, ymax)
    return (xm - dy / tan_theta, ymin)


def _circle_edge(x: float, y: float, r: float, theta: float) -> tuple:
    """Point on the circle at angle *theta* (``circleEdge``, grid.c:1809)."""
    angle = theta / 180.0 * math.pi
    return (x + r * math.cos(angle), y + r * math.sin(angle))


def _polygon_edge(x: Any, y: Any, theta: float) -> tuple:
    """Point on the edge of a *convex* polygon at angle *theta*.

    Port of ``polygonEdge`` (grid.c:1826-1949).  Vertices must be in
    CLOCKWISE order (chull() convention): the scan takes the FIRST edge
    subtending *theta* about the bbox centre, so vertex order matters.
    """
    n = len(x)
    xmin = math.inf
    xmax = -math.inf
    ymin = math.inf
    ymax = -math.inf
    # explicit comparisons like C: NaN vertices are silently skipped
    for i in range(n):
        if x[i] < xmin:
            xmin = x[i]
        if x[i] > xmax:
            xmax = x[i]
        if y[i] < ymin:
            ymin = y[i]
        if y[i] > ymax:
            ymax = y[i]
    xm = (xmin + xmax) / 2.0
    ym = (ymin + ymax) / 2.0
    # degenerate branches: very tall-and-narrow / short-and-wide polygons
    wdiff = abs(xmin - xmax)
    hdiff = abs(ymin - ymax)
    if wdiff < 1e-6 or hdiff / wdiff > 1000:
        edgex = xmin
        if theta == 90:
            edgey = ymax
        elif theta == 270:
            edgey = ymin
        else:
            edgey = ym
        return (edgex, edgey)
    if hdiff < 1e-6 or wdiff / hdiff > 1000:
        edgey = ymin
        if theta == 0:
            edgex = xmax
        elif theta == 180:
            edgex = xmin
        else:
            edgex = xm
        return (edgex, edgey)
    angle = theta / 180.0 * math.pi
    found = False
    v1 = v2 = 0
    for i in range(n):
        v1 = i
        v2 = 0 if i + 1 == n else i + 1
        vangle1 = math.atan2(y[v1] - ym, x[v1] - xm)
        if vangle1 < 0:
            vangle1 += 2.0 * math.pi
        vangle2 = math.atan2(y[v2] - ym, x[v2] - xm)
        if vangle2 < 0:
            vangle2 += 2.0 * math.pi
        if ((vangle1 >= vangle2 and
             vangle1 >= angle and vangle2 <= angle) or
            (vangle1 < vangle2 and
             ((vangle1 >= angle and 0 <= angle) or
              (vangle2 <= angle and 2.0 * math.pi >= angle)))):
            found = True
            break
    if not found:
        raise ValueError("polygon edge not found")
    # intersect the centre->rectEdge segment with the found edge
    x1, y1 = xm, ym
    x2, y2 = _rect_edge(xmin, ymin, xmax, ymax, theta)
    x3, y3 = x[v1], y[v1]
    x4, y4 = x[v2], y[v2]
    numa = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    # C relies on IEEE division producing inf/NaN, then errors on it
    ua = math.inf if denom == 0.0 else numa / denom
    if not math.isfinite(ua):
        raise ValueError("polygon edge not found (zero-width or zero-height?)")
    return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))


def _chull_split(xs: Any, ys: Any, m: int, in_arr: list, in_base: int,
                 ii: int, jj: int, s: int,
                 iabv: list, iabv_base: int,
                 ibel: list, ibel_base: int) -> tuple:
    """Partition points by the line through vertices *ii* and *jj*.

    Verbatim port of ``split`` (grDevices chull.c); the ``*_base`` args
    play the role of the C base pointers (``&in[1]``, ``&ib[nib]``, ...).
    Returns ``(na, maxa, nb, maxb)``: subset sizes and 1-based positions
    of the farthest point in each subset (0 when empty).
    """
    xt = xs[ii]
    vert = xs[jj] == xt
    d1 = ys[jj] - ys[ii]
    a = b = 0.0
    neg_dir = False
    if vert:
        neg_dir = (s > 0 and d1 < 0.0) or (s < 0 and d1 > 0.0)
    else:
        a = d1 / (xs[jj] - xt)
        b = ys[ii] - a * xt
    up = 0.0
    na = 0
    maxa = 0
    down = 0.0
    nb = 0
    maxb = 0
    for i in range(m):
        is_ = in_arr[in_base + i]
        if vert:
            z = (xt - xs[is_]) if neg_dir else (xs[is_] - xt)
        else:
            z = ys[is_] - a * xs[is_] - b
        if z > 0.0:  # the point is ABOVE the line
            if s == -2:
                continue
            iabv[iabv_base + na] = is_
            na += 1
            if z >= up:
                up = z
                maxa = na
        elif s != 2 and z < 0.0:  # the point is BELOW the line
            ibel[ibel_base + nb] = is_
            nb += 1
            if z <= down:
                down = z
                maxb = nb
    return na, maxa, nb, maxb


def _chull(x: Any, y: Any) -> list:
    """Convex hull indices (0-based) exactly as R's ``grDevices::chull()``.

    Line-by-line port of ``in_chull`` (grDevices chull.c — ACM TOMS 523,
    Eddy 1977) plus the SEXP wrapper's reversal and the R-level angle
    sort.  The exact clockwise vertex ORDER matters: ``polygonEdge``
    takes the first edge its scan matches.
    """
    n = len(x)
    if n == 0:
        return []
    # 1-based arrays (index 0 unused) mirroring the f2c pointer adjustments
    xs = [0.0] * (n + 1)
    ys = [0.0] * (n + 1)
    for i in range(n):
        xs[i + 1] = float(x[i])
        ys[i + 1] = float(y[i])
    m = n
    in_arr = [0] * (n + 2)
    for i in range(1, n + 1):
        in_arr[i] = i
    ia = [0] * (n + 2)
    ib = [0] * (n + 2)
    ih = [0] * (n + 2)
    il = [0] * (n + 2)
    nh = 0

    def finis() -> list:
        # chull.c Finis (reorder ih along the linked list il), the SEXP
        # wrapper's reversal, then the R-level chull() angle sort:
        # res[order(atan2(dy, -dx))] about the hull-vertex mean
        nonlocal nh
        nh -= 1
        ia_tmp = [0] * (nh + 1)
        for i in range(1, nh + 1):
            ia_tmp[i] = ih[i]
        j = il[1]
        for i in range(2, nh + 1):
            ih[i] = ia_tmp[j]
            j = il[j]
        res = [ih[nh - i] - 1 for i in range(nh)]
        if len(res) < 2:
            return res
        cx = sum(x[i] for i in res) / len(res)
        cy = sum(y[i] for i in res) / len(res)
        # stable sort = R order()'s tie behaviour
        return sorted(res, key=lambda i: math.atan2(y[i] - cy,
                                                    -(x[i] - cx)))

    if m == 1:
        # L_1pt
        nh = 2
        ih[1] = in_arr[1]
        il[1] = 1
        return finis()

    il[1] = 2
    il[2] = 1
    kn = in_arr[1]
    kx = in_arr[2]
    if m == 2:
        # L_2pts
        ih[1] = kx
        ih[2] = kn
        if xs[kn] == xs[kx] and ys[kn] == ys[kx]:
            nh = 2
        else:
            nh = 3
        return finis()

    mp1 = m + 1
    min_ = 1
    mx = 1
    kx = in_arr[1]
    maxe = False
    mine = False
    # initial partition vertices: max-x (kx) and min-x (kn) points
    for i in range(2, m + 1):
        j = in_arr[i]
        d1 = xs[j] - xs[kx]
        if d1 < 0.0:
            pass
        elif d1 == 0.0:
            maxe = True
        else:
            maxe = False
            mx = i
            kx = j
        d1 = xs[j] - xs[kn]
        if d1 < 0.0:
            mine = False
            min_ = i
            kn = j
        elif d1 == 0.0:
            mine = True

    if kx == kn:
        # L_vertical: all the points lie on a vertical line
        kx = in_arr[1]
        kn = in_arr[1]
        for i in range(1, m + 1):
            j = in_arr[i]
            if ys[j] > ys[kx]:
                mx = i
                kx = j
            if ys[j] < ys[kn]:
                min_ = i
                kn = j
        if kx == kn:
            # L_1pt
            nh = 2
            ih[1] = in_arr[1]
            il[1] = 1
            return finis()
        # L_2pts
        ih[1] = kx
        ih[2] = kn
        if xs[kn] == xs[kx] and ys[kn] == ys[kx]:
            nh = 2
        else:
            nh = 3
        return finis()

    if maxe or mine:
        if maxe:  # tie-break equal max-x by largest y
            for i in range(1, m + 1):
                j = in_arr[i]
                if xs[j] != xs[kx]:
                    continue
                if ys[j] <= ys[kx]:
                    continue
                mx = i
                kx = j
        if mine:  # tie-break equal min-x by smallest y
            for i in range(1, m + 1):
                j = in_arr[i]
                if xs[j] != xs[kn]:
                    continue
                if ys[j] >= ys[kn]:
                    continue
                min_ = i
                kn = j

    # L7
    ih[1] = kx
    ih[2] = kn
    nh = 3
    inh = 1
    nib = 1
    ma = m
    in_arr[mx] = in_arr[m]
    in_arr[m] = kx
    mm = m - 2
    if min_ == m:
        min_ = mx
    in_arr[min_] = in_arr[m - 1]
    in_arr[m - 1] = kn
    # root partition
    mb, mxa, nb_out, mxbb = _chull_split(
        xs, ys, mm, in_arr, 1, ih[1], ih[2], 0, ia, 1, ib, 1)
    ia[ma] = nb_out
    mxb = 0
    mbb = 0

    # left half of the tree
    goto_l12 = False
    while not goto_l12:  # L8
        nib += ia[ma]
        ma -= 1
        restart_l8 = False
        while True:  # do { ... } while(true)
            if mxa != 0:
                il[nh] = il[inh]
                il[inh] = nh
                ih[nh] = ia[mxa]
                ia[mxa] = ia[mb]
                mb -= 1
                nh += 1
                if mb != 0:
                    ilinh = il[inh]
                    mbb, mxa, nb_out, mxb = _chull_split(
                        xs, ys, mb, ia, 1, ih[inh], ih[ilinh], 1,
                        ia, 1, ib, nib)
                    ia[ma] = nb_out
                    mb = mbb
                    restart_l8 = True  # goto L8
                    break
                inh = il[inh]
            while True:  # inner do { ... } while(ia[ma] == 0)
                inh = il[inh]
                ma += 1
                nib -= ia[ma]
                if ma >= m:
                    goto_l12 = True
                    break
                if ia[ma] != 0:
                    break
            if goto_l12:
                break
            ilinh = il[inh]
            # s=2: right son of a right son lies inside a triangle -> neglected
            mb, mxa, nb_out, mxb = _chull_split(
                xs, ys, ia[ma], ib, nib, ih[inh], ih[ilinh], 2,
                ia, 1, ib, nib)
            ia[ma] = nb_out
        if not restart_l8 and not goto_l12:
            break

    # right half of the tree (L12)
    mxb = mxbb
    ma = m
    mb = ia[ma]
    nia = 1
    ia[ma] = 0
    finis_now = False
    while not finis_now:  # L13
        nia += ia[ma]
        ma -= 1
        restart_l13 = False
        while True:  # do { ... } while(true)
            if mxb != 0:
                il[nh] = il[inh]
                il[inh] = nh
                ih[nh] = ib[mxb]
                ib[mxb] = ib[mb]
                mb -= 1
                nh += 1
                if mb != 0:
                    ilinh = il[inh]
                    na_out, mxa, mbb, mxb = _chull_split(
                        xs, ys, mb, ib, nib, ih[inh], ih[ilinh], -1,
                        ia, nia, ib, nib)
                    ia[ma] = na_out
                    mb = mbb
                    restart_l13 = True  # goto L13
                    break
                inh = il[inh]
            while True:  # inner do { ... } while(ia[ma] == 0)
                inh = il[inh]
                ma += 1
                # Next two lines swapped in R 4.0.0 (nia unused in Finis)
                if ma == mp1:
                    finis_now = True
                    break
                nia -= ia[ma]
                if ia[ma] != 0:
                    break
            if finis_now:
                break
            ilinh = il[inh]
            # s=-2: left son of a left son lies inside a triangle -> neglected
            mbb, mxa, mb, mxb = _chull_split(
                xs, ys, ia[ma], ia, nia, ih[inh], ih[ilinh], -2,
                ia, nia, ib, nib)
        if not restart_l13 and not finis_now:
            break

    return finis()


def _hull_edge(x: Any, y: Any, theta: float) -> tuple:
    """Edge of the convex hull of (x, y) at angle *theta*.

    Port of ``hullEdge`` (grid.c:1952-2007).  R computes the hull on the
    non-finite-filtered points but then indexes the ORIGINAL arrays with
    those hull indices (grid.c:1993-1996) — replicated verbatim.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(xs) & np.isfinite(ys)
    xk = xs[keep]
    yk = ys[keep]
    hull = _chull(xk, yk)
    hx = [float(xs[i]) for i in hull]
    hy = [float(ys[i]) for i in hull]
    return _polygon_edge(hx, hy, theta)


# -- per-class edge functions ------------------------------------------
# Each returns (edgex, edgey) in inches within the current viewport, or
# None when there is nothing to measure (R: NULL -> unit(0.5, "npc")).


def _resolve_locations_inches(x_unit: Any, y_unit: Any, renderer: Any,
                              gp: Any) -> Optional[tuple]:
    """Resolve paired location units to inches with R-style recycling.

    Transform loop of ``L_locnBounds`` (grid.c:5330-5360); non-finite
    pairs stay in the arrays and only reduce the finite count ``nloc``.
    Returns ``(xx, yy, nloc)`` or None for empty input.
    """
    from ._units import Unit as _U
    if not isinstance(x_unit, _U) or not isinstance(y_unit, _U):
        return None
    nx = max(len(x_unit), len(y_unit))
    if nx == 0 or len(x_unit) == 0 or len(y_unit) == 0:
        return None
    xx = np.empty(nx, dtype=np.float64)
    yy = np.empty(nx, dtype=np.float64)
    nloc = 0
    for i in range(nx):
        xx[i] = renderer._resolve_to_inches_idx(
            x_unit, i % len(x_unit), "x", False, gp)
        yy[i] = renderer._resolve_to_inches_idx(
            y_unit, i % len(y_unit), "y", False, gp)
        if np.isfinite(xx[i]) and np.isfinite(yy[i]):
            nloc += 1
    return (xx, yy, nloc)


def _edge_locn(grob: Any, theta: float) -> Optional[tuple]:
    """Hull edge of a grob's (x, y) locations.

    Port of ``xDetails.lines``/``.polyline``/``.polygon``/``.pathgrob``/
    ``.points``/``.null``, all via ``C_locnBounds`` (grid.c:5296-5380).
    """
    renderer = _get_renderer()
    if renderer is None:
        return None
    resolved = _resolve_locations_inches(
        getattr(grob, "x", None), getattr(grob, "y", None),
        renderer, getattr(grob, "gp", None))
    if resolved is None:
        return None
    xx, yy, nloc = resolved
    if nloc == 0:
        return None
    return _hull_edge(xx, yy, theta)


def _edge_segments(grob: Any, theta: float) -> Optional[tuple]:
    """Hull edge of segment endpoints.

    Port of R ``segmentBounds`` (primitives.R:341-349): endpoints are
    recycled to a common length, concatenated, then treated as locations.
    """
    from ._units import Unit as _U, unit_c
    renderer = _get_renderer()
    if renderer is None:
        return None
    x0 = getattr(grob, "x0", None)
    x1 = getattr(grob, "x1", None)
    y0 = getattr(grob, "y0", None)
    y1 = getattr(grob, "y1", None)
    if not all(isinstance(u, _U) for u in (x0, x1, y0, y1)):
        return None
    n = max(len(x0), len(x1), len(y0), len(y1))
    if n == 0:
        return None

    def _rep(u: Any) -> Any:
        idx = [i % len(u) for i in range(n)]
        return unit_c(*[u[i] for i in idx]) if len(u) != n else u

    resolved = _resolve_locations_inches(
        unit_c(_rep(x0), _rep(x1)), unit_c(_rep(y0), _rep(y1)),
        renderer, getattr(grob, "gp", None))
    if resolved is None:
        return None
    xx, yy, nloc = resolved
    if nloc == 0:
        return None
    return _hull_edge(xx, yy, theta)


def _edge_rect(grob: Any, theta: float) -> Optional[tuple]:
    """Edge of a rect grob.

    Port of ``gridRect`` with ``draw=FALSE`` (grid.c:3296-3356): each
    rect is justified, a single rect gets its own ``rectEdge``, several
    rects get the union-bbox ``rectEdge``.
    """
    from ._just import resolve_hjust, resolve_vjust
    from ._units import Unit as _U
    renderer = _get_renderer()
    if renderer is None:
        return None
    x_u = getattr(grob, "x", None)
    y_u = getattr(grob, "y", None)
    w_u = getattr(grob, "width", None)
    h_u = getattr(grob, "height", None)
    if not all(isinstance(u, _U) for u in (x_u, y_u, w_u, h_u)):
        return None
    gp = getattr(grob, "gp", None)
    just = getattr(grob, "just", None)
    if just is None:
        just = "centre"
    hj = float(resolve_hjust(just, getattr(grob, "hjust", None)))
    vj = float(resolve_vjust(just, getattr(grob, "vjust", None)))

    maxn = max(len(x_u), len(y_u), len(w_u), len(h_u))
    if maxn == 0:
        return None
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    edge = None
    nrect = 0
    for i in range(maxn):
        xx = renderer._resolve_to_inches_idx(x_u, i % len(x_u), "x", False, gp)
        yy = renderer._resolve_to_inches_idx(y_u, i % len(y_u), "y", False, gp)
        ww = renderer._resolve_to_inches_idx(w_u, i % len(w_u), "x", True, gp)
        hh = renderer._resolve_to_inches_idx(h_u, i % len(h_u), "y", True, gp)
        xx = xx - hj * ww
        yy = yy - vj * hh
        if (np.isfinite(xx) and np.isfinite(yy)
                and np.isfinite(ww) and np.isfinite(hh)):
            xmin = min(xmin, xx, xx + ww)
            xmax = max(xmax, xx, xx + ww)
            ymin = min(ymin, yy, yy + hh)
            ymax = max(ymax, yy, yy + hh)
            edge = _rect_edge(xx, yy, xx + ww, yy + hh, theta)
            nrect += 1
    if nrect == 0:
        return None
    if nrect > 1:
        edge = _rect_edge(xmin, ymin, xmax, ymax, theta)
    return edge


def _edge_circle(grob: Any, theta: float) -> Optional[tuple]:
    """Edge of a circle grob: on the circle when single, union-bbox
    ``rectEdge`` when several.

    Port of ``gridCircle`` with ``draw=FALSE`` (grid.c:3020-3096):
    r = min(|r as width|, |r as height|); like C, the single-circle edge
    uses the LAST loop iteration's values.
    """
    from ._units import Unit as _U
    renderer = _get_renderer()
    if renderer is None:
        return None
    x_u = getattr(grob, "x", None)
    y_u = getattr(grob, "y", None)
    r_u = getattr(grob, "r", None)
    if not all(isinstance(u, _U) for u in (x_u, y_u, r_u)):
        return None
    gp = getattr(grob, "gp", None)
    nx = max(len(x_u), len(y_u), len(r_u))
    if nx == 0:
        return None
    nr = len(r_u)
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    ncirc = 0
    xx = yy = rr = float("nan")
    for i in range(nx):
        xx = renderer._resolve_to_inches_idx(x_u, i % len(x_u), "x", False, gp)
        yy = renderer._resolve_to_inches_idx(y_u, i % len(y_u), "y", False, gp)
        rr1 = renderer._resolve_to_inches_idx(r_u, i % nr, "x", True, gp)
        rr2 = renderer._resolve_to_inches_idx(r_u, i % nr, "y", True, gp)
        rr = min(abs(rr1), abs(rr2))
        if np.isfinite(xx) and np.isfinite(yy) and np.isfinite(rr):
            xmin = min(xmin, xx - rr, xx + rr)
            xmax = max(xmax, xx - rr, xx + rr)
            ymin = min(ymin, yy - rr, yy + rr)
            ymax = max(ymax, yy - rr, yy + rr)
            ncirc += 1
    if ncirc == 0:
        return None
    if ncirc == 1:
        return _circle_edge(xx, yy, rr, theta)
    return _rect_edge(xmin, ymin, xmax, ymax, theta)


def _text_label_extent(label: str, gp: Any, cex: float, lineheight: float,
                       fontsize: float) -> tuple:
    """(width, height) of one label in inches (GEStrWidth / GEStrHeight).

    Height is the first line's ASCENT plus lineheight-scaled gaps — R's
    GEStrHeight never includes the descent (ggplot2's titleGrob adds
    grobDescent separately, so including it here would double-count).
    Single metric source for both the text bbox and the text edge.
    """
    lines = label.split("\n") if label else [""]
    w = max(calc_string_metric(ln, gp=gp)["width"] for ln in lines)
    m0 = calc_string_metric(lines[0], gp=gp)
    gap = cex * lineheight * fontsize * 1.2 / 72.0
    h = m0["ascent"] + (len(lines) - 1) * gap
    return (w, h)


def _gp_text_params(gp: Any) -> tuple:
    """(cex, lineheight, fontsize) from gpar, with R defaults."""
    cex = 1.0
    lineheight = 1.2
    fontsize = 12.0
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
    return (cex, lineheight, fontsize)


def _text_placement_corners(grob: Any) -> Optional[list]:
    """Corner quads of every text placement, in inches.

    Port of the placement loop shared by ``gridText`` with ``draw=FALSE``
    (grid.c:3746-3826) and ``textRect`` (util.c:178-260): one placement
    per ``max(len(x), len(y))`` — R recycles labels and rot into the
    anchor count, NOT the other way around — each with the recycled
    label's box justified, rotated, then translated to its anchor.
    Placements with non-finite anchors are dropped.  Returns a list of
    ``(cx4, cy4)`` quads (anti-clockwise bl, br, tr, tl), or ``None``
    when there is nothing to measure.

    Without an active renderer, anchors resolve to (0, 0) so the result
    degenerates to metric-only boxes (keeps device-less callers of the
    width/height details working; R always has a device).
    """
    from ._just import resolve_hjust, resolve_vjust
    from ._units import Unit as _U

    labels = _normalise_labels(grob)
    if not labels:
        return None
    gp = _resolve_grob_gp(grob)
    just = getattr(grob, "just", None)
    if just is None:
        just = "centre"
    hj = float(resolve_hjust(just, getattr(grob, "hjust", None)))
    vj = float(resolve_vjust(just, getattr(grob, "vjust", None)))
    rots = np.atleast_1d(np.asarray(getattr(grob, "rot", 0.0),
                                    dtype=np.float64))
    cex, lineheight, fontsize = _gp_text_params(gp)

    renderer = _get_renderer()
    x_u = getattr(grob, "x", None)
    y_u = getattr(grob, "y", None)
    has_anchors = (isinstance(x_u, _U) and isinstance(y_u, _U)
                   and len(x_u) > 0 and len(y_u) > 0)
    nx = max(len(x_u), len(y_u)) if has_anchors else len(labels)

    quads = []
    for i in range(nx):
        if has_anchors and renderer is not None:
            xx = renderer._resolve_to_inches_idx(
                x_u, i % len(x_u), "x", False, gp)
            yy = renderer._resolve_to_inches_idx(
                y_u, i % len(y_u), "y", False, gp)
        else:
            xx = yy = 0.0
        if not (np.isfinite(xx) and np.isfinite(yy)):
            continue
        w, h = _text_label_extent(labels[i % len(labels)], gp,
                                  cex, lineheight, fontsize)
        # textRect corners, anti-clockwise (bl, br, tr, tl) with sign
        # handling for negative extents
        if w >= 0:
            if h >= 0:
                corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
            else:
                corners = [(0.0, h), (w, h), (w, 0.0), (0.0, 0.0)]
        else:
            if h >= 0:
                corners = [(w, 0.0), (0.0, 0.0), (0.0, h), (w, h)]
            else:
                corners = [(w, h), (0.0, h), (0.0, 0.0), (w, 0.0)]
        rot = float(rots[i % len(rots)])
        rad = math.radians(rot)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        cx_pts = []
        cy_pts = []
        for px, py in corners:
            # justify, then rotate, then translate to the anchor
            jx = px - hj * w
            jy = py - vj * h
            cx_pts.append(jx * cos_r - jy * sin_r + xx)
            cy_pts.append(jx * sin_r + jy * cos_r + yy)
        quads.append((cx_pts, cy_pts))
    return quads or None


def _edge_text(grob: Any, theta: float) -> Optional[tuple]:
    """Edge of a text grob (rotated per-label boxes).

    Port of ``gridText`` with ``draw=FALSE`` (grid.c:3777-3826): a
    single placement gets ``polygonEdge`` on its box corners, several
    placements the union-bbox ``rectEdge``.
    """
    if _get_renderer() is None:
        return None
    quads = _text_placement_corners(grob)
    if quads is None:
        return None
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    edge = None
    for cx_pts, cy_pts in quads:
        xmin = min(xmin, *cx_pts)
        xmax = max(xmax, *cx_pts)
        ymin = min(ymin, *cy_pts)
        ymax = max(ymax, *cy_pts)
        # polygonEdge needs CLOCKWISE order: tl, tr, br, bl; like R it
        # runs for every placement (and may raise) even when the union
        # rectEdge supersedes it below
        edge = _polygon_edge(
            [cx_pts[3], cx_pts[2], cx_pts[1], cx_pts[0]],
            [cy_pts[3], cy_pts[2], cy_pts[1], cy_pts[0]],
            theta)
    if len(quads) > 1:
        edge = _rect_edge(xmin, ymin, xmax, ymax, theta)
    return edge


def _edge_xspline(grob: Any, theta: float) -> Optional[tuple]:
    """Edge of an xspline grob: hull edge of the EVALUATED curve for a
    single spline, union-bbox ``rectEdge`` for several ``id`` groups.

    Port of ``xDetails.xspline`` (primitives.R:827-843) via
    ``gridXspline`` with ``draw=FALSE`` (grid.c:2532-2596).
    """
    from ._curve import (
        _calc_xspline_points, _device_size_in, _xspline_index,
    )
    from ._units import convert_x, convert_y
    renderer = _get_renderer()
    if renderer is None:
        return None
    xx = np.atleast_1d(np.asarray(
        convert_x(grob.x, "inches", valueOnly=True), dtype=np.float64))
    yy = np.atleast_1d(np.asarray(
        convert_y(grob.y, "inches", valueOnly=True), dtype=np.float64))
    shape = np.resize(
        np.atleast_1d(np.asarray(getattr(grob, "shape", 0.0),
                                 dtype=np.float64)),
        len(xx))
    open_ = bool(getattr(grob, "open_", True))
    rep_ends = bool(getattr(grob, "repEnds", True))
    device_size = _device_size_in()
    groups = _xspline_index(grob)
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    edge = None
    nloc = 0
    for idx in groups:
        px, py = _calc_xspline_points(
            xx[idx], yy[idx], shape[idx], open_, rep_ends,
            units_per_inch=1.0, device_size_in=device_size,
        )
        if len(px) <= 1:
            continue  # GEXspline returns NULL for <=1 point
        finite = np.isfinite(px) & np.isfinite(py)
        if np.any(finite):
            xmin = min(xmin, float(px[finite].min()))
            xmax = max(xmax, float(px[finite].max()))
            ymin = min(ymin, float(py[finite].min()))
            ymax = max(ymax, float(py[finite].max()))
            nloc += int(finite.sum())
        edge = _hull_edge(px, py, theta)
    if nloc == 0:
        return None
    if len(groups) > 1:
        edge = _rect_edge(xmin, ymin, xmax, ymax, theta)
    return edge


def _round_corner(num: int, x: float, y: float, r: float) -> tuple:
    """One roundrect corner arc — port of R ``roundCorner``
    (roundrect.R:70-92); slices are R's 1-based inclusive subsets.
    """
    n = 40
    t = np.linspace(0.0, 2.0 * np.pi, n)
    if num == 1:
        xc, yc = x + r, y + r
        sl = slice(19, 30)  # R (n/2):(3*n/4)
    elif num == 2:
        xc, yc = x + r, y - r
        sl = slice(9, 20)   # R (n/4):(n/2)
    elif num == 3:
        xc, yc = x - r, y - r
        sl = slice(0, 10)   # R 1:(n/4)
    else:
        xc, yc = x - r, y + r
        sl = slice(29, 40)  # R (3*n/4):n
    return (xc + np.cos(t[sl]) * r, yc + np.sin(t[sl]) * r)


def _edge_roundrect(grob: Any, theta: float) -> Optional[tuple]:
    """Hull edge of a roundrect's rounded boundary (``rrpoints``).

    Port of ``xDetails.roundrect`` (roundrect.R:125-140).  R builds the
    boundary inside the implicit viewport pushed by
    ``makeContext.roundrect``; this port has no such viewport, so the
    equivalent justified rectangle is computed in the current context.
    """
    from ._just import resolve_hjust, resolve_vjust
    from ._units import Unit as _U
    renderer = _get_renderer()
    if renderer is None:
        return None
    x_u = getattr(grob, "x", None)
    y_u = getattr(grob, "y", None)
    w_u = getattr(grob, "width", None)
    h_u = getattr(grob, "height", None)
    r_u = getattr(grob, "r", None)
    if not all(isinstance(u, _U) for u in (x_u, y_u, w_u, h_u, r_u)):
        return None
    gp = getattr(grob, "gp", None)
    just = getattr(grob, "just", None)
    if just is None:
        just = "centre"
    hj = float(resolve_hjust(just, getattr(grob, "hjust", None)))
    vj = float(resolve_vjust(just, getattr(grob, "vjust", None)))
    xx = renderer._resolve_to_inches_idx(x_u, 0, "x", False, gp)
    yy = renderer._resolve_to_inches_idx(y_u, 0, "y", False, gp)
    ww = renderer._resolve_to_inches_idx(w_u, 0, "x", True, gp)
    hh = renderer._resolve_to_inches_idx(h_u, 0, "y", True, gp)
    if not (np.isfinite(xx) and np.isfinite(yy)
            and np.isfinite(ww) and np.isfinite(hh)):
        return None
    left = xx - hj * ww
    bottom = yy - vj * hh
    right = left + ww
    top = bottom + hh
    # rrpoints: r = min(radius as width, radius as height)
    r = min(renderer._resolve_to_inches_idx(r_u, 0, "x", True, gp),
            renderer._resolve_to_inches_idx(r_u, 0, "y", True, gp))
    c1x, c1y = _round_corner(1, left, bottom, r)
    c2x, c2y = _round_corner(2, left, top, r)
    c3x, c3y = _round_corner(3, right, top, r)
    c4x, c4y = _round_corner(4, right, bottom, r)
    bx = np.concatenate([
        [left + r, right - r], c4x, [right, right], c3x,
        [right - r, left + r], c2x, [left, left], c1x])
    by = np.concatenate([
        [bottom, bottom], c4y, [bottom + r, top - r], c3y,
        [top, top], c2y, [top - r, bottom + r], c1y])
    return _hull_edge(bx, by, theta)


_EDGE_DISPATCH: Dict[str, Any] = {
    "lines": _edge_locn,
    "polyline": _edge_locn,
    "polygon": _edge_locn,
    "pathgrob": _edge_locn,
    "points": _edge_locn,
    "null": _edge_locn,
    "segments": _edge_segments,
    "rect": _edge_rect,
    # rect logic: this port stores explicit raster width/height units
    # (same convention as _raster_width_details)
    "rastergrob": _edge_rect,
    "circle": _edge_circle,
    "text": _edge_text,
    "xspline": _edge_xspline,
    "roundrect": _edge_roundrect,
}


def _details_delegate(x: Any) -> Any:
    """The grob whose x/yDetails stand in for a delegating grob.

    curve (curve.R:463-478): the single child of the expanded curve, or
    the gTree itself (-> 0.5npc default) when there are several children.
    functiongrob (function.R:49-57): the generated lines grob.
    """
    if getattr(x, "_grid_class", None) == "functiongrob":
        return x.make_content()
    content = x.make_content()
    order = getattr(content, "_children_order", [])
    if len(order) == 1:
        return content._children[order[0]]
    return content


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


def _xy_details(x: Any, theta: float, axis_idx: int) -> Unit:
    """Shared implementation of :func:`x_details` / :func:`y_details`.

    Mirrors R's ``xDetails``/``yDetails`` S3 dispatch (size.R:37-55):
    class handlers, then the delegating classes (curve, functiongrob),
    then a grob method override (e.g. beziergrob), then the
    ``unit(0.5, "npc")`` default.
    """
    # normalise like grobX()'s convertTheta so rectEdge's exact
    # 0/90/180/270 special cases stay reachable
    theta = float(theta) % 360.0
    cls = getattr(x, "_grid_class", None)
    edge_fn = _EDGE_DISPATCH.get(cls)
    if edge_fn is not None:
        edge = edge_fn(x, theta)
        if edge is None:
            return Unit(0.5, "npc")
        return Unit(float(edge[axis_idx]), "inches")
    if cls in ("curve", "functiongrob"):
        return _xy_details(_details_delegate(x), theta, axis_idx)
    method = getattr(x, "x_details" if axis_idx == 0 else "y_details", None)
    if callable(method):
        result = method(theta)
        # the base Grob method returns None; only a real override counts
        if result is not None:
            return result
    return Unit(0.5, "npc")


def x_details(x: Any, theta: float = 0) -> Unit:
    """x position on the edge of grob *x* at angle *theta* (degrees,
    0 = East / 90 = North).

    Port of R ``xDetails`` (size.R:37-45) and its per-class methods.
    Returns inches within the current viewport context, or
    ``Unit(0.5, "npc")`` when there is no edge to measure.
    """
    return _xy_details(x, theta, 0)


def y_details(x: Any, theta: float = 0) -> Unit:
    """y position on the edge of grob *x* at angle *theta* (degrees,
    0 = East / 90 = North).

    Port of R ``yDetails`` (size.R:47-55) and its per-class methods.
    Returns inches within the current viewport context, or
    ``Unit(0.5, "npc")`` when there is no edge to measure.
    """
    return _xy_details(x, theta, 1)


# ---------------------------------------------------------------------------
# grob_* convenience constructors
# ---------------------------------------------------------------------------


def grob_width(x: Any) -> Unit:
    """Create a ``"grobwidth"`` unit referencing *x*.

    Port of R ``grobWidth()`` (unit.R:674-692).
    Accepts Grob, GList, GPath, or string (auto-wrapped in GPath).

    Parameters
    ----------
    x : Grob, GList, GPath, or str
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
    from ._grob import Grob, GList
    from ._path import GPath

    if isinstance(x, GList):
        # R unit.R:682-684: rep_len(1, length(x)) with data=x
        return Unit([1.0] * len(x), ["grobwidth"] * len(x),
                    data=[x[i] for i in range(len(x))])
    if isinstance(x, (Grob, GPath)):
        return Unit(1, "grobwidth", data=x)
    # Default: wrap string in GPath (R unit.R:690-692)
    return Unit(1, "grobwidth", data=GPath(str(x)))


def grob_height(x: Any) -> Unit:
    """Create a ``"grobheight"`` unit referencing *x*.

    Port of R ``grobHeight()`` (unit.R:695-713).

    Parameters
    ----------
    x : Grob, GList, GPath, or str
        The graphical object whose height is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobheight"`` with *x* stored as auxiliary data.
    """
    from ._grob import Grob, GList
    from ._path import GPath

    if isinstance(x, GList):
        return Unit([1.0] * len(x), ["grobheight"] * len(x),
                    data=[x[i] for i in range(len(x))])
    if isinstance(x, (Grob, GPath)):
        return Unit(1, "grobheight", data=x)
    return Unit(1, "grobheight", data=GPath(str(x)))


def grob_x(x: Any, theta: Any = 0) -> Unit:
    """Create a ``"grobx"`` unit referencing *x* at angle *theta*.

    Port of R ``grobX()`` (unit.R:632-650).

    Parameters
    ----------
    x : Grob, GList, GPath, or str
        The graphical object.
    theta : str or float
        Angle in degrees or one of ``"east"``, ``"north"``, ``"west"``,
        ``"south"``.  Normalised to [0, 360).

    Returns
    -------
    Unit
        A unit of type ``"grobx"``.
    """
    from ._grob import Grob, GList
    from ._path import GPath
    from ._units import convert_theta

    t = convert_theta(theta)
    if isinstance(x, GList):
        return Unit([t] * len(x), ["grobx"] * len(x),
                    data=[x[i] for i in range(len(x))])
    if isinstance(x, (Grob, GPath)):
        return Unit(t, "grobx", data=x)
    return Unit(t, "grobx", data=GPath(str(x)))


def grob_y(x: Any, theta: Any = 0) -> Unit:
    """Create a ``"groby"`` unit referencing *x* at angle *theta*.

    Port of R ``grobY()`` (unit.R:653-671).

    Parameters
    ----------
    x : Grob, GList, GPath, or str
        The graphical object.
    theta : str or float
        Angle in degrees or one of ``"east"``, ``"north"``, ``"west"``,
        ``"south"``.  Normalised to [0, 360).

    Returns
    -------
    Unit
        A unit of type ``"groby"``.
    """
    from ._grob import Grob, GList
    from ._path import GPath
    from ._units import convert_theta

    t = convert_theta(theta)
    if isinstance(x, GList):
        return Unit([t] * len(x), ["groby"] * len(x),
                    data=[x[i] for i in range(len(x))])
    if isinstance(x, (Grob, GPath)):
        return Unit(t, "groby", data=x)
    return Unit(t, "groby", data=GPath(str(x)))


def grob_ascent(x: Any) -> Unit:
    """Create a ``"grobascent"`` unit referencing *x*.

    Port of R ``grobAscent()`` (unit.R:716+).

    Parameters
    ----------
    x : Grob, GList, GPath, or str
        The graphical object whose text ascent is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobascent"`` with *x* stored as auxiliary data.
    """
    from ._grob import Grob, GList
    from ._path import GPath

    if isinstance(x, GList):
        return Unit([1.0] * len(x), ["grobascent"] * len(x),
                    data=[x[i] for i in range(len(x))])
    if isinstance(x, (Grob, GPath)):
        return Unit(1, "grobascent", data=x)
    return Unit(1, "grobascent", data=GPath(str(x)))


def grob_descent(x: Any) -> Unit:
    """Create a ``"grobdescent"`` unit referencing *x*.

    Port of R ``grobDescent()`` (unit.R:737+).

    Parameters
    ----------
    x : Grob, GList, GPath, or str
        The graphical object whose text descent is referenced.

    Returns
    -------
    Unit
        A unit of type ``"grobdescent"`` with *x* stored as auxiliary data.
    """
    from ._grob import Grob, GList
    from ._path import GPath

    if isinstance(x, GList):
        return Unit([1.0] * len(x), ["grobdescent"] * len(x),
                    data=[x[i] for i in range(len(x))])
    if isinstance(x, (Grob, GPath)):
        return Unit(1, "grobdescent", data=x)
    return Unit(1, "grobdescent", data=GPath(str(x)))


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
