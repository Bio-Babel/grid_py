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

    Mirrors what R's ``heightDetails.text`` / ``widthDetails.text`` return
    (empirically verified against R 4.4 ``cairo_png`` at 150 dpi):

      width  = max(per-line ink widths)           — R's ``GEStrWidth``
      height = ink_height(first line)
               + (n - 1) × cex × lineheight × fontsize × 1.2 / 72
                                                    — R's ``GEStrHeight``

    The per-extra-line gap ``1.2 × fontsize / 72`` is R's device-level
    ``cra[1] × ipr[1] / default_ps`` collapsed for the standard cairo /
    PostScript setups (default_ps = 12 pt, cin[1] = 1.2 × 12 / 72 in).

    Single-line text is therefore independent of ``lineheight`` (matches
    R exactly), while each extra newline adds the lineheight-scaled gap.

    The grob's gp is merged with the current viewport stack gpar, so
    ``grobHeight`` inherits fontsize / lineheight from the viewport.
    """
    import math
    labels = _normalise_labels(grob)
    gp = _resolve_grob_gp(grob)
    rot = float(getattr(grob, "rot", 0.0))

    if not labels:
        return (0.0, 0.0)

    # Resolve cex and lineheight; fontsize is read per-line by calc_string_metric.
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

    # Per-extra-line gap in inches — matches R's ``cra[1] × ipr[1] / default_ps``
    # collapsed for the standard device (= fontsize × 1.2 / 72).
    inter_line_gap = cex * lineheight * fontsize * 1.2 / 72.0

    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")

    for lab in labels:
        lines = lab.split("\n") if lab else [""]
        n_lines = len(lines)
        # Width: max per-line width (R's ``GEStrWidth`` walks each line).
        w = max(calc_string_metric(ln, gp=gp)["width"] for ln in lines)
        # Height: ink of first line + gap × (n - 1).  R uses the first
        # line's ink bounds (``ascent + descent``) and extends by per-line
        # gaps for additional lines.
        m0 = calc_string_metric(lines[0], gp=gp)
        h = m0["ascent"] + m0["descent"] + (n_lines - 1) * inter_line_gap

        if rot == 0.0:
            # No rotation: bbox is just the text extent
            corners_x = [0, w, w, 0]
            corners_y = [0, 0, h, h]
        else:
            rad = math.radians(rot)
            cos_r = math.cos(rad)
            sin_r = math.sin(rad)
            # Four corners of the unrotated text rectangle
            corners = [(0, 0), (w, 0), (w, h), (0, h)]
            corners_x = [cx_ * cos_r - cy_ * sin_r for cx_, cy_ in corners]
            corners_y = [cx_ * sin_r + cy_ * cos_r for cx_, cy_ in corners]

        for cx_ in corners_x:
            if cx_ < xmin:
                xmin = cx_
            if cx_ > xmax:
                xmax = cx_
        for cy_ in corners_y:
            if cy_ < ymin:
                ymin = cy_
            if cy_ > ymax:
                ymax = cy_

    if xmin == float("inf"):
        return (0.0, 0.0)
    return (xmax - xmin, ymax - ymin)


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

def _xspline_width_details(grob: Any) -> Unit:
    """Width of an xspline grob: bounding box of control points.

    Port of R ``widthDetails.xspline`` (primitives.R:845) which uses
    ``C_xsplineBounds``.  We approximate by using the control point
    bounding box (same as C_locnBounds on the control points).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x_unit = getattr(grob, "x", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_width(x_unit, renderer, gp), "inches")


def _xspline_height_details(grob: Any) -> Unit:
    """Height of an xspline grob.

    Port of R ``heightDetails.xspline`` (primitives.R:854).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y_unit = getattr(grob, "y", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_height(y_unit, renderer, gp), "inches")


# -- bezier grob (R: primitives.R:997-1003, expands via splinegrob()) ------

def _bezier_width_details(grob: Any) -> Unit:
    """Width of a bezier grob: bounding box of control points.

    Port of R ``widthDetails.beziergrob`` (primitives.R:997).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    x_unit = getattr(grob, "x", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_width(x_unit, renderer, gp), "inches")


def _bezier_height_details(grob: Any) -> Unit:
    """Height of a bezier grob.

    Port of R ``heightDetails.beziergrob`` (primitives.R:1001).
    """
    renderer = _get_renderer()
    if renderer is None:
        return Unit(0, "inches")
    y_unit = getattr(grob, "y", None)
    gp = getattr(grob, "gp", None)
    return Unit(_locn_bounds_height(y_unit, renderer, gp), "inches")


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
