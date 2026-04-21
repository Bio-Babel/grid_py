"""Viewport coordinate transform calculations -- port of R's matrix.c + viewport.c.

This module provides the 3×3 affine transform matrix operations and viewport
transform computation that form the core of R grid's coordinate pipeline.

All transforms use the **row-vector** convention from R's grid:
    point_out = point_in @ matrix
where point = [x, y, 1].  Translation terms live in row 2 (m[2,0], m[2,1]).

References
----------
R source: ``grid/src/matrix.c`` (identity, translation, rotation, scaling,
multiply, location, trans), ``grid/src/viewport.c`` (calcViewportTransform).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    # Matrix operations (port of matrix.c)
    "identity",
    "translation",
    "rotation",
    "scaling",
    "multiply",
    "location",
    "trans",
    "inv_transform",
    "copy_transform",
    # Viewport transform (port of viewport.c:calcViewportTransform)
    "calc_viewport_transform",
    # Inverse transforms (port of unit.c:1226-1475)
    "_transform_from_inches",
    "_transform_xy_from_inches",
    "_transform_wh_from_inches",
    "_transform_xy_to_npc",
    "_transform_wh_to_npc",
    "_transform_xy_from_npc",
    "_transform_wh_from_npc",
    # Viewport context
    "ViewportContext",
    "ViewportTransformResult",
]


# ============================================================================
# Type aliases
# ============================================================================

# LTransform = double[3][3]  -->  np.ndarray shape (3,3) float64
# LLocation  = double[3]     -->  np.ndarray shape (3,) float64


# ============================================================================
# Matrix operations -- direct port of R grid/src/matrix.c
# ============================================================================
# Each function mirrors the corresponding C function exactly.
# Comments reference R source line numbers for traceability.


def identity() -> np.ndarray:
    """Return 3×3 identity matrix.

    Port of ``matrix.c:62 identity()``.
    """
    return np.eye(3, dtype=np.float64)


def translation(tx: float, ty: float) -> np.ndarray:
    """Return 3×3 translation matrix.

    Port of ``matrix.c:73 translation()``:
    ``identity(m); m[2][0] = tx; m[2][1] = ty;``
    """
    m = np.eye(3, dtype=np.float64)
    m[2, 0] = tx
    m[2, 1] = ty
    return m


def scaling(sx: float, sy: float) -> np.ndarray:
    """Return 3×3 scaling matrix.

    Port of ``matrix.c:80 scaling()``:
    ``identity(m); m[0][0] = sx; m[1][1] = sy;``
    """
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    return m


def rotation(theta_degrees: float) -> np.ndarray:
    """Return 3×3 rotation matrix for *theta_degrees* degrees.

    Port of ``matrix.c:87 rotation()``:
    ``thetarad = theta/180*PI; m[0][0] = cos; m[0][1] = sin;
    m[1][0] = -sin; m[1][1] = cos;``
    """
    rad = theta_degrees / 180.0 * math.pi
    c = math.cos(rad)
    s = math.sin(rad)
    m = np.eye(3, dtype=np.float64)
    m[0, 0] = c
    m[0, 1] = s
    m[1, 0] = -s
    m[1, 1] = c
    return m


def multiply(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Multiply two 3×3 matrices: ``result = m1 @ m2``.

    Port of ``matrix.c:99 multiply()``.
    Uses numpy matmul for clarity; result is identical to the
    hand-unrolled R code.
    """
    return m1 @ m2


def location(x: float, y: float) -> np.ndarray:
    """Create a homogeneous location vector [x, y, 1].

    Port of ``matrix.c:112 location()``.
    """
    return np.array([x, y, 1.0], dtype=np.float64)


def trans(vin: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Transform a location vector by a matrix: ``vout = vin @ m``.

    Port of ``matrix.c:119 trans()``:
    ``vout[0] = vin[0]*m[0][0] + vin[1]*m[1][0] + vin[2]*m[2][0]; ...``
    """
    return vin @ m


def inv_transform(t: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 3×3 transform matrix.

    Port of ``matrix.c:44 invTransform()`` using the explicit
    cofactor/determinant formula.
    """
    det = (t[0, 0] * (t[2, 2] * t[1, 1] - t[2, 1] * t[1, 2])
           - t[1, 0] * (t[2, 2] * t[0, 1] - t[2, 1] * t[0, 2])
           + t[2, 0] * (t[1, 2] * t[0, 1] - t[1, 1] * t[0, 2]))
    if det == 0:
        raise ValueError("singular transformation matrix")
    inv = np.empty((3, 3), dtype=np.float64)
    inv[0, 0] = (t[2, 2] * t[1, 1] - t[2, 1] * t[1, 2]) / det
    inv[0, 1] = -(t[2, 2] * t[0, 1] - t[2, 1] * t[0, 2]) / det
    inv[0, 2] = (t[1, 2] * t[0, 1] - t[1, 1] * t[0, 2]) / det
    inv[1, 0] = -(t[2, 2] * t[1, 0] - t[2, 0] * t[1, 2]) / det
    inv[1, 1] = (t[2, 2] * t[0, 0] - t[2, 0] * t[0, 2]) / det
    inv[1, 2] = -(t[1, 2] * t[0, 0] - t[1, 0] * t[0, 2]) / det
    inv[2, 0] = (t[2, 1] * t[1, 0] - t[2, 0] * t[1, 1]) / det
    inv[2, 1] = -(t[2, 1] * t[0, 0] - t[2, 0] * t[0, 1]) / det
    inv[2, 2] = (t[1, 1] * t[0, 0] - t[1, 0] * t[0, 1]) / det
    return inv


def copy_transform(t: np.ndarray) -> np.ndarray:
    """Copy a 3×3 transform matrix.

    Port of ``matrix.c:36 copyTransform()``.
    """
    return t.copy()


# ============================================================================
# Justification helper -- port of R grid/src/viewport.c justification()
# ============================================================================

def justification(width: float, height: float,
                  hjust: float, vjust: float) -> Tuple[float, float]:
    """Compute justification offsets in the same units as width/height.

    Port of R's ``justification()`` (grid.h / viewport.c).
    Returns (xadj, yadj) where:
        xadj = -hjust * width
        yadj = -vjust * height
    """
    return (-hjust * width, -vjust * height)


# ============================================================================
# Viewport context (port of LViewportContext)
# ============================================================================

class ViewportContext:
    """Stores xscale/yscale ranges for native unit resolution.

    Port of ``grid.h:260 LViewportContext``.
    """
    __slots__ = ("xscalemin", "xscalemax", "yscalemin", "yscalemax")

    def __init__(self, xscale: Tuple[float, float] = (0.0, 1.0),
                 yscale: Tuple[float, float] = (0.0, 1.0)):
        self.xscalemin = float(xscale[0])
        self.xscalemax = float(xscale[1])
        self.yscalemin = float(yscale[0])
        self.yscalemax = float(yscale[1])


class ViewportTransformResult:
    """Result of calcViewportTransform: the computed viewport state.

    Stores the same values that R writes back to the viewport SEXP
    (viewport.c:370-380).
    """
    __slots__ = ("width_cm", "height_cm", "rotation_angle",
                 "transform", "vpc")

    def __init__(self, width_cm: float, height_cm: float,
                 rotation_angle: float, transform: np.ndarray,
                 vpc: ViewportContext):
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.rotation_angle = rotation_angle
        self.transform = transform
        self.vpc = vpc


# ============================================================================
# Unit-to-inches conversion -- port of unit.c:transform() switch
# ============================================================================
# This is the key function that converts any grid unit value to inches
# within a viewport context, WITHOUT requiring an R graphics device.

_INCHES_PER = {
    "inches": 1.0,
    "cm": 1.0 / 2.54,
    "mm": 1.0 / 25.4,
    "points": 1.0 / 72.27,        # TeX points
    "picas": 12.0 / 72.27,
    "bigpts": 1.0 / 72.0,         # PostScript/CSS points
    "dida": (1238.0 / 1157.0) / 72.27,
    "cicero": 12.0 * (1238.0 / 1157.0) / 72.27,
    "scaledpts": 1.0 / (72.27 * 65536.0),
}


def transform_x_to_inches(
    unit_obj: Any,
    index: int,
    vpc: ViewportContext,
    gc_fontsize: float,
    gc_cex: float,
    gc_lineheight: float,
    parent_width_cm: float,
    parent_height_cm: float,
    str_metric_fn: Any = None,
    grob_metric_fn: Any = None,
) -> float:
    """Convert a Unit element to inches along the X axis.

    Port of ``unit.c:transformXtoINCHES`` which calls
    ``transform()`` with ``thisCM = parentWidthCM``.

    Parameters
    ----------
    unit_obj : Unit
        The unit object (must have _values, _units, _data).
    index : int
        Element index within the unit vector.
    vpc : ViewportContext
        The parent viewport's native scale ranges.
    gc_fontsize, gc_cex, gc_lineheight : float
        Font metrics from the gpar context (gc->ps, gc->cex, gc->lineheight).
    parent_width_cm, parent_height_cm : float
        Parent viewport dimensions in CM.
    str_metric_fn : callable or None
        Function(text, gp) -> dict with 'width','ascent','descent' in inches.
    grob_metric_fn : callable or None
        Function(grob, what) -> Unit for grob-based metrics.

    Returns
    -------
    float
        The value in inches.
    """
    return _transform_to_inches(
        unit_obj, index, vpc,
        gc_fontsize, gc_cex, gc_lineheight,
        this_cm=parent_width_cm,
        other_cm=parent_height_cm,
        axis="x", is_dim=False,
        str_metric_fn=str_metric_fn,
        grob_metric_fn=grob_metric_fn,
    )


def transform_y_to_inches(
    unit_obj: Any, index: int, vpc: ViewportContext,
    gc_fontsize: float, gc_cex: float, gc_lineheight: float,
    parent_width_cm: float, parent_height_cm: float,
    str_metric_fn: Any = None, grob_metric_fn: Any = None,
) -> float:
    """Convert a Unit element to inches along the Y axis.

    Port of ``unit.c:transformYtoINCHES``.
    """
    return _transform_to_inches(
        unit_obj, index, vpc,
        gc_fontsize, gc_cex, gc_lineheight,
        this_cm=parent_height_cm,
        other_cm=parent_width_cm,
        axis="y", is_dim=False,
        str_metric_fn=str_metric_fn,
        grob_metric_fn=grob_metric_fn,
    )


def transform_width_to_inches(
    unit_obj: Any, index: int, vpc: ViewportContext,
    gc_fontsize: float, gc_cex: float, gc_lineheight: float,
    parent_width_cm: float, parent_height_cm: float,
    str_metric_fn: Any = None, grob_metric_fn: Any = None,
) -> float:
    """Convert a Unit element to inches as an X-axis dimension.

    Port of ``unit.c:transformWidthtoINCHES``.
    """
    return _transform_to_inches(
        unit_obj, index, vpc,
        gc_fontsize, gc_cex, gc_lineheight,
        this_cm=parent_width_cm,
        other_cm=parent_height_cm,
        axis="x", is_dim=True,
        str_metric_fn=str_metric_fn,
        grob_metric_fn=grob_metric_fn,
    )


def transform_height_to_inches(
    unit_obj: Any, index: int, vpc: ViewportContext,
    gc_fontsize: float, gc_cex: float, gc_lineheight: float,
    parent_width_cm: float, parent_height_cm: float,
    str_metric_fn: Any = None, grob_metric_fn: Any = None,
) -> float:
    """Convert a Unit element to inches as a Y-axis dimension.

    Port of ``unit.c:transformHeighttoINCHES``.
    """
    return _transform_to_inches(
        unit_obj, index, vpc,
        gc_fontsize, gc_cex, gc_lineheight,
        this_cm=parent_height_cm,
        other_cm=parent_width_cm,
        axis="y", is_dim=True,
        str_metric_fn=str_metric_fn,
        grob_metric_fn=grob_metric_fn,
    )


def _transform_to_inches(
    unit_obj: Any,
    index: int,
    vpc: ViewportContext,
    gc_fontsize: float,
    gc_cex: float,
    gc_lineheight: float,
    this_cm: float,
    other_cm: float,
    axis: str,
    is_dim: bool,
    str_metric_fn: Any = None,
    grob_metric_fn: Any = None,
    scale: float = 1.0,
) -> float:
    """Core unit-to-inches conversion -- port of ``unit.c:transform()``.

    This implements the big switch statement (unit.c:658-800) that
    converts each unit type to inches, plus the GSS_SCALE post-scaling
    for physical units (unit.c:804-814).

    Parameters
    ----------
    unit_obj : Unit
        The unit object.
    index : int
        Element index.
    vpc : ViewportContext
        Parent scale context.
    gc_fontsize, gc_cex, gc_lineheight : float
        Font context (gc->ps, gc->cex, gc->lineheight).
    this_cm, other_cm : float
        Viewport dimension in CM for the primary and orthogonal axis.
    axis : str
        ``"x"`` or ``"y"``.
    is_dim : bool
        True if converting a width/height (dimension), False for position.
    str_metric_fn : callable or None
        String metric query function.
    grob_metric_fn : callable or None
        Grob metric query function.
    scale : float
        GSS_SCALE zoom factor (R unit.c:804-814).  Default 1.0.

    Returns
    -------
    float
        Value in inches.
    """
    from ._units import Unit

    if not isinstance(unit_obj, Unit):
        return float(unit_obj)

    idx = index % len(unit_obj)
    value = float(unit_obj._values[idx])
    utype = unit_obj._units[idx]
    data = unit_obj._data[idx] if unit_obj._data is not None else None

    this_inches = this_cm / 2.54

    # ---- Absolute physical units (unit.c:670-682) ----
    # R unit.c:804-814: physical units are additionally scaled by GSS_SCALE
    if utype in _INCHES_PER:
        return value * _INCHES_PER[utype] * scale

    # ---- NPC (unit.c:667) ----
    # L_NPC: result * thisCM / 2.54
    if utype == "npc":
        if is_dim:
            return value * this_inches
        else:
            return value * this_inches

    # ---- SNPC (unit.c:689) ----
    # L_SNPC: result * min(thisCM, otherCM) / 2.54
    if utype == "snpc":
        return value * min(this_cm, other_cm) / 2.54

    # ---- Native (unit.c:837) ----
    # Maps value from [scalemin, scalemax] range to [0, thisCM/2.54]
    if utype == "native":
        scalemin = vpc.xscalemin if axis == "x" else vpc.yscalemin
        scalemax = vpc.xscalemax if axis == "x" else vpc.yscalemax
        srange = scalemax - scalemin
        if srange == 0:
            return 0.0
        if is_dim:
            return (value / srange) * this_inches
        else:
            return ((value - scalemin) / srange) * this_inches

    # ---- Char (unit.c:683) ----
    # L_CHAR: result * gc->ps * gc->cex / 72
    # Note: gc->ps = fontsize * GSS_SCALE (gpar.c:395), so char/lines
    # units are implicitly scaled by GSS_SCALE through gc->ps.
    if utype == "char":
        return value * gc_fontsize * scale * gc_cex / 72.0

    # ---- Lines (unit.c:687) ----
    # L_LINES: result * gc->ps * gc->cex * gc->lineheight / 72
    if utype == "lines":
        return value * gc_fontsize * scale * gc_cex * gc_lineheight / 72.0

    # ---- Null (unit.c:693) ----
    # L_NULL: contributes 0 inches in this context
    if utype == "null":
        return 0.0

    # ---- String metrics (unit.c:720-760) ----
    #
    # Mirrors R's ``GEStrWidth`` / ``GEStrHeight`` / ``GEStrMetric``:
    #
    #   - split the label on ``\n`` into lines;
    #   - ``strwidth``  = max(per-line widths)                       (GEStrWidth)
    #   - ``strheight`` = ink(first line) + (n−1) × cex × lineheight × ps × 1.2 / 72
    #                                                                (GEStrHeight)
    #   - ``strascent`` / ``strdescent`` = ink metric of the first line
    #                                                                (GEStrMetric)
    if utype in ("strwidth", "strheight", "strascent", "strdescent"):
        text = str(data) if data is not None else ""
        lines = text.split("\n") if text else [""]
        n = len(lines)
        if str_metric_fn is not None:
            m0 = str_metric_fn(lines[0], None)
            if utype == "strwidth":
                w = max(str_metric_fn(ln, None).get("width", 0.0) for ln in lines)
                return value * w
            elif utype == "strheight":
                ink_first = m0.get("ascent", 0.0) + m0.get("descent", 0.0)
                inter_line_gap = (
                    gc_cex * gc_lineheight * gc_fontsize * 1.2 / 72.0
                )
                return value * (ink_first + (n - 1) * inter_line_gap)
            elif utype == "strascent":
                return value * m0.get("ascent", 0.0)
            else:
                return value * m0.get("descent", 0.0)
        # Fallback: estimate from font size when no measurement callback is
        # available.  ``gc->ps = fontsize * GSS_SCALE`` in R.
        effective = gc_fontsize * scale * gc_cex
        char_width = effective * 0.6 / 72.0
        if utype == "strwidth":
            max_line_len = max((len(ln) for ln in lines), default=0)
            return value * max_line_len * char_width
        elif utype == "strheight":
            # First line approximated as one ``effective`` unit of height;
            # add the inter-line gap for extra lines.
            inter_line_gap = (
                gc_cex * gc_lineheight * gc_fontsize * 1.2 / 72.0
            )
            return value * (effective / 72.0 + (n - 1) * inter_line_gap)
        elif utype == "strascent":
            return value * effective * 0.75 / 72.0
        else:
            return value * effective * 0.25 / 72.0

    # ---- Grob metrics (unit.c:770-800) ----
    # R's evaluateGrobUnit (unit.c:325-590) does a full preDraw/postDraw
    # cycle on the grob, then calls widthDetails/heightDetails to get a
    # result Unit, converts it to inches *within the grob's viewport
    # context*, and returns the inches value.
    #
    # grob_metric_fn(grob, utype, value) must return **inches** directly
    # (it does the full preDraw/eval/postDraw/restore cycle internally).
    if utype in ("grobwidth", "grobheight", "grobascent", "grobdescent",
                  "grobx", "groby"):
        if grob_metric_fn is not None and data is not None:
            inches = grob_metric_fn(data, utype, value)
            if inches is not None:
                # For width / height / ascent / descent, ``value`` is a
                # scaling factor (usually 1) — the result scales linearly.
                # For grobx / groby, ``value`` carries the evaluation angle
                # (degrees) and the callback already returned the absolute
                # x/y coordinate in inches, so no extra scaling applies.
                if utype in ("grobx", "groby"):
                    return inches
                return value * inches
        return 0.0

    # ---- Compound units: sum, min, max ----
    if utype in ("sum", "min", "max"):
        child = data
        if isinstance(child, Unit):
            results = []
            for j in range(len(child)):
                r = _transform_to_inches(
                    child, j, vpc,
                    gc_fontsize, gc_cex, gc_lineheight,
                    this_cm, other_cm, axis, is_dim,
                    str_metric_fn, grob_metric_fn,
                )
                results.append(r)
            if utype == "sum":
                return value * sum(results)
            elif utype == "min":
                return value * min(results) if results else 0.0
            elif utype == "max":
                return value * max(results) if results else 0.0
        return 0.0

    # ---- mychar, mylines, mystrwidth, mystrheight (unit.c:804+) ----
    if utype == "mychar":
        return value * gc_fontsize * gc_cex / 72.0
    if utype == "mylines":
        return value * gc_fontsize * gc_cex * gc_lineheight / 72.0
    if utype in ("mystrwidth", "mystrheight"):
        # These use the grob's own gpar rather than the parent's.
        # Without that context, fall back to the str metric path.
        # R marks these as "FIXME: Remove this when I can" (unit.c:721,734).
        if str_metric_fn is not None and data is not None:
            text = str(data)
            m = str_metric_fn(text, None)
            if utype == "mystrwidth":
                return value * m.get("width", 0.0)
            else:
                return value * (m.get("ascent", 0.0) + m.get("descent", 0.0))
        return 0.0

    # ---- Fallback: treat as NPC ----
    return value * this_inches


# ============================================================================
# Inverse transform: inches → target unit  (port of unit.c:1226-1475)
# ============================================================================


def _transform_from_inches(
    value: float,
    unit: str,
    gc_fontsize: float,
    gc_cex: float,
    gc_lineheight: float,
    this_cm: float,
    other_cm: float,
    scale: float = 1.0,
) -> float:
    """Convert a value in inches to the given unit type.

    Port of R ``unit.c:1226-1333 transformFromINCHES()``.
    Handles absolute and font-relative units.  For physical units,
    applies the inverse of GSS_SCALE (unit.c:1313-1331).

    Parameters
    ----------
    value : float
        Value in inches.
    unit : str
        Target unit type string.
    gc_fontsize, gc_cex, gc_lineheight : float
        Font context parameters.
    this_cm, other_cm : float
        Viewport dimensions in CM for primary and orthogonal axis.
    scale : float
        GSS_SCALE zoom factor.

    Returns
    -------
    float
        The value in the target unit.
    """
    result = value

    if unit == "npc":
        # unit.c:1237  result/(thisCM/2.54)
        this_inches = this_cm / 2.54
        if this_inches < 1e-10:
            if result != 0:
                raise ValueError("Viewport has zero dimension(s)")
            return 0.0
        result = result / this_inches
    elif unit == "inches":
        pass  # unit.c:1243
    elif unit == "cm":
        result = result * 2.54  # unit.c:1240
    elif unit == "mm":
        result = result * 25.4  # unit.c:1270
    elif unit == "points":
        result = result * 72.27  # unit.c:1275
    elif unit == "picas":
        result = result / 12.0 * 72.27  # unit.c:1278
    elif unit == "bigpts":
        result = result * 72.0  # unit.c:1281
    elif unit == "dida":
        result = result / 1238.0 * 1157.0 * 72.27  # unit.c:1284
    elif unit == "cicero":
        result = result / 1238.0 * 1157.0 * 72.27 / 12.0  # unit.c:1287
    elif unit == "scaledpts":
        result = result * 65536.0 * 72.27  # unit.c:1290
    elif unit == "char":
        # unit.c:1253  (result*72)/(gc->ps*gc->cex)
        ps_cex = gc_fontsize * gc_cex
        if ps_cex < 1e-10:
            return 0.0
        result = (result * 72.0) / ps_cex
    elif unit == "lines":
        # unit.c:1256  (result*72)/(gc->ps*gc->cex*gc->lineheight)
        ps_cex_lh = gc_fontsize * gc_cex * gc_lineheight
        if ps_cex_lh < 1e-10:
            return 0.0
        result = (result * 72.0) / ps_cex_lh
    elif unit == "snpc":
        # unit.c:1258-1268
        if this_cm < 1e-6 or other_cm < 1e-6:
            if result != 0:
                raise ValueError("Viewport has zero dimension(s)")
            return 0.0
        min_inches = min(this_cm, other_cm) / 2.54
        result = result / min_inches
    else:
        raise ValueError(f"Cannot convert from inches to unit {unit!r}")

    # For physical units, reverse the GSS_SCALE (unit.c:1313-1331)
    _PHYSICAL_UNITS = {
        "inches", "cm", "mm", "points", "picas",
        "bigpts", "dida", "cicero", "scaledpts",
    }
    if unit in _PHYSICAL_UNITS and scale != 0:
        result = result / scale

    return result


def _transform_xy_from_inches(
    location_inches: float,
    unit: str,
    scalemin: float,
    scalemax: float,
    gc_fontsize: float,
    gc_cex: float,
    gc_lineheight: float,
    this_cm: float,
    other_cm: float,
    scale: float = 1.0,
) -> float:
    """Convert a location in inches to the target unit.

    Port of R ``unit.c:1348-1377 transformXYFromINCHES()``.
    Handles the special NATIVE case (scale mapping).
    """
    if unit == "native":
        this_inches = this_cm / 2.54
        if this_inches < 1e-10:
            if location_inches != 0:
                raise ValueError("Viewport has zero dimension(s)")
            return 0.0
        # unit.c:1369  scalemin + (result/(thisCM/2.54))*(scalemax - scalemin)
        return scalemin + (location_inches / this_inches) * (scalemax - scalemin)
    return _transform_from_inches(
        location_inches, unit,
        gc_fontsize, gc_cex, gc_lineheight,
        this_cm, other_cm, scale,
    )


def _transform_wh_from_inches(
    dimension_inches: float,
    unit: str,
    scalemin: float,
    scalemax: float,
    gc_fontsize: float,
    gc_cex: float,
    gc_lineheight: float,
    this_cm: float,
    other_cm: float,
    scale: float = 1.0,
) -> float:
    """Convert a dimension in inches to the target unit.

    Port of R ``unit.c:1379-1408 transformWidthHeightFromINCHES()``.
    Handles the special NATIVE case (dimension = range fraction).
    """
    if unit == "native":
        this_inches = this_cm / 2.54
        if this_inches < 1e-10:
            if dimension_inches != 0:
                raise ValueError("Viewport has zero dimension(s)")
            return 0.0
        # unit.c:1400  (result/(thisCM/2.54))*(scalemax - scalemin)
        return (dimension_inches / this_inches) * (scalemax - scalemin)
    return _transform_from_inches(
        dimension_inches, unit,
        gc_fontsize, gc_cex, gc_lineheight,
        this_cm, other_cm, scale,
    )


def _transform_xy_to_npc(
    value: float, from_unit: str,
    scalemin: float, scalemax: float,
) -> float:
    """Relative unit to NPC -- port of ``unit.c:1418-1431 transformXYtoNPC()``.

    Used when viewport has zero width/height to avoid divide-by-zero.
    """
    if from_unit == "npc":
        return value
    if from_unit == "native":
        srange = scalemax - scalemin
        if srange == 0:
            return 0.0
        return (value - scalemin) / srange
    raise ValueError(f"Cannot convert {from_unit!r} to NPC (zero-dimension special case)")


def _transform_wh_to_npc(
    value: float, from_unit: str,
    scalemin: float, scalemax: float,
) -> float:
    """Relative dimension to NPC -- port of ``unit.c:1433-1446 transformWHtoNPC()``."""
    if from_unit == "npc":
        return value
    if from_unit == "native":
        srange = scalemax - scalemin
        if srange == 0:
            return 0.0
        return value / srange
    raise ValueError(f"Cannot convert {from_unit!r} to NPC (zero-dimension special case)")


def _transform_xy_from_npc(
    value: float, to_unit: str,
    scalemin: float, scalemax: float,
) -> float:
    """NPC to relative unit -- port of ``unit.c:1448-1461 transformXYfromNPC()``."""
    if to_unit == "npc":
        return value
    if to_unit == "native":
        return scalemin + value * (scalemax - scalemin)
    raise ValueError(f"Cannot convert NPC to {to_unit!r} (zero-dimension special case)")


def _transform_wh_from_npc(
    value: float, to_unit: str,
    scalemin: float, scalemax: float,
) -> float:
    """NPC to relative dimension -- port of ``unit.c:1463-1475 transformWHfromNPC()``."""
    if to_unit == "npc":
        return value
    if to_unit == "native":
        return value * (scalemax - scalemin)
    raise ValueError(f"Cannot convert NPC to {to_unit!r} (zero-dimension special case)")


# ============================================================================
# calcViewportTransform -- port of viewport.c:214-382
# ============================================================================

def calc_viewport_transform(
    vp: Any,
    parent_transform: np.ndarray,
    parent_width_cm: float,
    parent_height_cm: float,
    parent_angle: float,
    parent_context: ViewportContext,
    gc_fontsize: float = 10.0,
    gc_cex: float = 1.0,
    gc_lineheight: float = 1.2,
    str_metric_fn: Any = None,
    grob_metric_fn: Any = None,
) -> ViewportTransformResult:
    """Compute the viewport's 3×3 transform matrix.

    This is the core function that mirrors R's ``calcViewportTransform``
    (viewport.c:214-382).  It converts the viewport's position and
    dimensions to inches, applies justification + rotation, and
    combines with the parent's transform.

    Parameters
    ----------
    vp : Viewport
        The viewport being pushed.
    parent_transform : ndarray (3,3)
        The parent viewport's accumulated transform.
    parent_width_cm, parent_height_cm : float
        Parent viewport dimensions in CM.
    parent_angle : float
        Parent's accumulated rotation angle in degrees.
    parent_context : ViewportContext
        Parent's xscale/yscale.
    gc_fontsize, gc_cex, gc_lineheight : float
        Parent gpar font metrics.
    str_metric_fn, grob_metric_fn : callable or None
        Metric query callbacks.

    Returns
    -------
    ViewportTransformResult
        Contains width_cm, height_cm, rotation_angle, transform, vpc.
    """
    # -- viewport.c:308-313: convert vp location to INCHES --
    vp_x_unit = getattr(vp, "_x", None)
    vp_y_unit = getattr(vp, "_y", None)
    vp_w_unit = getattr(vp, "_width", None)
    vp_h_unit = getattr(vp, "_height", None)

    # Default units from viewport
    from ._units import Unit
    if vp_x_unit is None:
        vp_x_unit = Unit(0.5, "npc")
    if vp_y_unit is None:
        vp_y_unit = Unit(0.5, "npc")
    if vp_w_unit is None:
        vp_w_unit = Unit(1.0, "npc")
    if vp_h_unit is None:
        vp_h_unit = Unit(1.0, "npc")

    x_inches = transform_x_to_inches(
        vp_x_unit, 0, parent_context,
        gc_fontsize, gc_cex, gc_lineheight,
        parent_width_cm, parent_height_cm,
        str_metric_fn, grob_metric_fn,
    )
    y_inches = transform_y_to_inches(
        vp_y_unit, 0, parent_context,
        gc_fontsize, gc_cex, gc_lineheight,
        parent_width_cm, parent_height_cm,
        str_metric_fn, grob_metric_fn,
    )

    # -- viewport.c:317-324: convert width/height to CM --
    # Note: R stores these in CM, converting from inches * 2.54
    vp_width_cm = transform_width_to_inches(
        vp_w_unit, 0, parent_context,
        gc_fontsize, gc_cex, gc_lineheight,
        parent_width_cm, parent_height_cm,
        str_metric_fn, grob_metric_fn,
    ) * 2.54

    vp_height_cm = transform_height_to_inches(
        vp_h_unit, 0, parent_context,
        gc_fontsize, gc_cex, gc_lineheight,
        parent_width_cm, parent_height_cm,
        str_metric_fn, grob_metric_fn,
    ) * 2.54

    # Non-finite check (viewport.c:327-331)
    if (not math.isfinite(x_inches) or not math.isfinite(y_inches)
            or not math.isfinite(vp_width_cm) or not math.isfinite(vp_height_cm)):
        raise ValueError("non-finite location and/or size for viewport")

    # -- viewport.c:334-335: justification offsets --
    just = getattr(vp, "_just", (0.5, 0.5))
    if isinstance(just, (list, tuple)) and len(just) >= 2:
        hjust, vjust = float(just[0]), float(just[1])
    else:
        hjust, vjust = 0.5, 0.5

    # justification() returns offsets in CM, then we convert to inches
    xadj, yadj = justification(vp_width_cm, vp_height_cm, hjust, vjust)

    # -- viewport.c:341-355: build transform chain --
    # thisLocation: translate to viewport position (in inches)
    this_location = translation(x_inches, y_inches)

    # thisRotation: viewport rotation
    vp_angle = float(getattr(vp, "_angle", 0))
    if vp_angle != 0:
        this_rotation = rotation(vp_angle)
    else:
        this_rotation = identity()

    # thisJustification: translate by justification offsets (CM -> inches)
    this_justification = translation(xadj / 2.54, yadj / 2.54)

    # viewport.c:349: Position relative to origin of rotation THEN rotate
    temp_transform = multiply(this_justification, this_rotation)

    # viewport.c:352: Translate to bottom-left corner
    this_transform = multiply(temp_transform, this_location)

    # viewport.c:355: Combine with parent's transform
    transform = multiply(this_transform, parent_transform)

    # viewport.c:358: Sum up the rotation angles
    rotation_angle = parent_angle + vp_angle

    # Build viewport context for children
    xscale = getattr(vp, "_xscale", [0.0, 1.0])
    yscale = getattr(vp, "_yscale", [0.0, 1.0])
    vpc = ViewportContext(
        xscale=(float(xscale[0]), float(xscale[1])),
        yscale=(float(yscale[0]), float(yscale[1])),
    )

    return ViewportTransformResult(
        width_cm=vp_width_cm,
        height_cm=vp_height_cm,
        rotation_angle=rotation_angle,
        transform=transform,
        vpc=vpc,
    )


def calc_root_transform(
    device_width_cm: float,
    device_height_cm: float,
) -> ViewportTransformResult:
    """Compute the root (device-level) viewport transform.

    Port of ``viewport.c:233-260``: when the parent is NULL (top-level
    viewport), the parent is the device itself.
    """
    return ViewportTransformResult(
        width_cm=device_width_cm,
        height_cm=device_height_cm,
        rotation_angle=0.0,
        transform=identity(),
        vpc=ViewportContext(xscale=(0.0, 1.0), yscale=(0.0, 1.0)),
    )
