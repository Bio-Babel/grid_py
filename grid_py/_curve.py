"""Curve, xspline, and bezier grobs for grid_py.

Python port of R's ``grid/R/curve.R`` (~535 lines).  Provides grob
constructors, ``grid_*`` drawing wrappers, point-extraction helpers, and the
internal control-point calculation routines that underpin curved connectors in
the *grid* graphics system.

The three main families are:

* **curve** -- a smooth curve between two endpoints, parameterised by
  curvature, angle, and number of control points.
* **xspline** -- an X-spline through arbitrary control points.
* **bezier** -- a cubic Bezier curve through four (or more) control points.
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from ._arrow import Arrow
from ._gpar import Gpar
from ._grob import GList, GTree, Grob
from ._primitives import lines_grob, segments_grob
from ._units import Unit, convert_x, convert_y, is_unit

__all__ = [
    # curve
    "curve_grob",
    "grid_curve",
    # xspline
    "xspline_grob",
    "grid_xspline",
    "xspline_points",
    # bezier
    "bezier_grob",
    "grid_bezier",
    "bezier_points",
    # utility
    "arc_curvature",
]

# ---------------------------------------------------------------------------
# Drawing helper (same as _primitives._grid_draw)
# ---------------------------------------------------------------------------


def _grid_draw(grob: Grob) -> None:
    """Draw *grob* immediately via the central dispatcher.

    Mirrors R's ``grid.draw()`` call inside ``grid.curve()`` /
    ``grid.xspline()`` / ``grid.bezier()``.
    """
    from ._draw import grid_draw  # lazy import to avoid circular dependency

    grid_draw(grob, recording=True)


# ---------------------------------------------------------------------------
# Helper: ensure a value is a Unit
# ---------------------------------------------------------------------------


def _ensure_unit(x: Any, default_units: str) -> Unit:
    """Convert *x* to a :class:`Unit` if it is not already one.

    Parameters
    ----------
    x : Any
        A numeric scalar, sequence of numerics, or an existing ``Unit``.
    default_units : str
        The unit string to use when *x* is not already a ``Unit``.

    Returns
    -------
    Unit
    """
    if is_unit(x):
        return x
    return Unit(x, default_units)


# ===================================================================== #
#  Internal: arc curvature utility                                       #
# ===================================================================== #


def arc_curvature(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
) -> float:
    """Compute the signed curvature of the arc through three points.

    Parameters
    ----------
    x1, y1 : float
        First point.
    x2, y2 : float
        Second point (apex).
    x3, y3 : float
        Third point.

    Returns
    -------
    float
        The signed curvature (positive = curves right, negative = curves
        left).  Returns ``0.0`` when the points are collinear or
        coincident.

    Notes
    -----
    Curvature is ``2 * signed_area / (d12 * d23 * d13)`` where
    ``signed_area`` is the cross-product triangle area.
    """
    # Twice the signed area of the triangle
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d12 = math.hypot(x2 - x1, y2 - y1)
    d23 = math.hypot(x3 - x2, y3 - y2)
    d13 = math.hypot(x3 - x1, y3 - y1)
    denom = d12 * d23 * d13
    if denom == 0.0:
        return 0.0
    return 2.0 * area2 / denom


# ===================================================================== #
#  Internal: control-point calculation (mirrors R's calcControlPoints)   #
# ===================================================================== #


def _calc_origin(
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    x2: NDArray[np.float64],
    y2: NDArray[np.float64],
    origin: float,
    hand: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the origin of rotation for control-point generation.

    Parameters
    ----------
    x1, y1, x2, y2 : ndarray
        Endpoint coordinates.
    origin : float
        Origin offset (derived from curvature).
    hand : str
        ``"left"`` or ``"right"``.

    Returns
    -------
    tuple of ndarray
        ``(ox, oy)`` origin coordinates.
    """
    xm = (x1 + x2) / 2.0
    ym = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1

    tmpox = xm + origin * dx / 2.0
    tmpoy = ym + origin * dy / 2.0

    # Handle special slope cases (vectorised)
    slope = np.where(dx != 0.0, dy / np.where(dx != 0.0, dx, 1.0), np.inf)
    finite_slope = np.isfinite(slope)
    oslope = np.where(slope != 0.0, -1.0 / np.where(slope != 0.0, slope, 1.0), np.inf)
    finite_oslope = np.isfinite(oslope)

    tmpox = np.where(~finite_slope, xm, tmpox)
    tmpoy = np.where(~finite_slope, ym + origin * dy / 2.0, tmpoy)
    tmpoy = np.where(finite_slope & ~finite_oslope, ym, tmpoy)

    # Rotate by -90 degrees about midpoint
    sintheta = -1.0
    ox = xm - (tmpoy - ym) * sintheta
    oy = ym + (tmpox - xm) * sintheta

    return ox, oy


def _calc_control_points(
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    x2: NDArray[np.float64],
    y2: NDArray[np.float64],
    curvature: float,
    angle: Optional[float],
    ncp: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute control points by rotating endpoints about an origin.

    Parameters
    ----------
    x1, y1, x2, y2 : ndarray
        Endpoint coordinates (in inches).
    curvature : float
        Signed curvature parameter.
    angle : float or None
        Angle in degrees (0-180).  ``None`` means auto-compute.
    ncp : int
        Number of control points per curve segment.

    Returns
    -------
    tuple of ndarray
        ``(cpx, cpy)`` arrays of control-point coordinates, flattened in
        row-major order.
    """
    xm = (x1 + x2) / 2.0
    ym = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    slope = np.where(dx != 0.0, dy / np.where(dx != 0.0, dx, 1.0), np.inf)

    # Angle computation
    if angle is None:
        angle_rad = np.where(
            slope < 0,
            2.0 * np.arctan(np.abs(slope)),
            2.0 * np.arctan(1.0 / np.where(slope != 0, np.abs(slope), 1e-30)),
        )
    else:
        angle_rad = np.full_like(x1, angle / 180.0 * math.pi)

    sina = np.sin(angle_rad)
    cosa = np.cos(angle_rad)
    cornerx = xm + (x1 - xm) * cosa - (y1 - ym) * sina
    cornery = ym + (y1 - ym) * cosa + (x1 - xm) * sina

    # Rotation angle to align region with axes
    denom_beta = cornerx - x1
    denom_beta = np.where(denom_beta == 0.0, 1e-30, denom_beta)
    beta = -np.arctan((cornery - y1) / denom_beta)
    sinb = np.sin(beta)
    cosb = np.cos(beta)

    # Rotate end point about start
    newx2 = x1 + dx * cosb - dy * sinb
    newy2 = y1 + dy * cosb + dx * sinb

    # Scale to make region square
    denom_scale = newx2 - x1
    denom_scale = np.where(denom_scale == 0.0, 1e-30, denom_scale)
    scalex = (newy2 - y1) / denom_scale
    scalex = np.where(scalex == 0.0, 1e-30, scalex)
    newx1 = x1 * scalex
    newx2 = newx2 * scalex

    # Origin in the "square" region
    ratio = 2.0 * (math.sin(math.atan(curvature)) ** 2)
    if ratio == 0.0:
        ratio = 1e-30
    origin = curvature - curvature / ratio
    hand = "right" if curvature > 0 else "left"

    ox, oy = _calc_origin(newx1, y1, newx2, newy2, origin, hand)

    # Direction and angular sweep for control points
    direction = 1.0 if hand == "right" else -1.0
    maxtheta = math.pi + math.copysign(1.0, origin * direction) * 2.0 * math.atan(abs(origin))
    # Port of R's ``seq(from, to, by)``:  ``seq(0, 0, by=0)`` returns
    # ``c(0)`` of length 1, not a length-``ncp+2`` ramp.
    step = direction * maxtheta / (ncp + 1)
    if step == 0.0:
        theta_all = np.array([0.0])
    else:
        theta_all = np.linspace(0.0, direction * maxtheta, ncp + 2)
    # R's ``[c(-1, -(ncp+2))]`` — drop first and last.  On a length-1
    # vector R silently allows out-of-range negative indices, yielding
    # an empty result.  ``theta_all[1:-1]`` matches both cases.
    theta = theta_all[1:-1]
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    # Matrix multiplication: ncurve x ncp
    # (newx1 - ox) is shape (ncurve,), costheta is shape (ncp,)
    cpx = ox[:, None] + np.outer(newx1 - ox, costheta) - np.outer(y1 - oy, sintheta)
    cpy = oy[:, None] + np.outer(y1 - oy, costheta) + np.outer(newx1 - ox, sintheta)

    # Reverse scaling
    cpx = cpx / scalex[:, None]

    # Reverse rotation
    sinnb = np.sin(-beta)
    cosnb = np.cos(-beta)
    finalcpx = x1[:, None] + (cpx - x1[:, None]) * cosnb[:, None] - (cpy - y1[:, None]) * sinnb[:, None]
    finalcpy = y1[:, None] + (cpy - y1[:, None]) * cosnb[:, None] + (cpx - x1[:, None]) * sinnb[:, None]

    return finalcpx.ravel(order="C"), finalcpy.ravel(order="C")


def _interleave(
    ncp: int,
    ncurve: int,
    val: NDArray[np.float64],
    sval: NDArray[np.float64],
    eval_: NDArray[np.float64],
    end: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Interleave control-point values with start/end extras.

    Parameters
    ----------
    ncp : int
        Number of control points per curve.
    ncurve : int
        Number of curves.
    val : ndarray
        Control-point values (ncp * ncurve).
    sval : ndarray
        Start values (ncurve).
    eval_ : ndarray
        End values (ncurve).
    end : ndarray of bool
        If ``True`` for curve *i*, append ``eval_[i]``; otherwise prepend
        ``sval[i]``.

    Returns
    -------
    ndarray
        Interleaved values, length ``(ncp + 1) * ncurve``.
    """
    sval = np.resize(sval, ncurve)
    eval_ = np.resize(eval_, ncurve)
    # Port of R's ``matrix(val, ncol=ncurve)``:  empty ``val`` yields a
    # ``0 × ncurve`` matrix (numpy's reshape would raise otherwise).
    if val.size == 0:
        m = np.empty((0, ncurve), dtype=np.float64)
    else:
        m = val.reshape((ncp, ncurve), order="F")
    result = np.empty((ncp + 1, ncurve), dtype=np.float64)
    for i in range(ncurve):
        if end[i]:
            col = np.concatenate([m[:, i], [eval_[i]]])
        else:
            col = np.concatenate([[sval[i]], m[:, i]])
        # R's ``result[,i] <- <shorter vector>`` recycles the rhs to
        # fill the column; for ``val`` empty the rhs is a length-1
        # scalar, which broadcasts naturally.
        if col.size == 1:
            result[:, i] = col[0]
        else:
            result[:, i] = col
    return result.ravel(order="F")


def _calc_square_control_points(
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    x2: NDArray[np.float64],
    y2: NDArray[np.float64],
    curvature: float,
    angle: Optional[float],
    ncp: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Compute "square" control points with an extra interleaved point.

    Parameters
    ----------
    x1, y1, x2, y2 : ndarray
        Endpoint coordinates.
    curvature : float
        Signed curvature.
    angle : float or None
        Angle in degrees.
    ncp : int
        Number of control points per segment.

    Returns
    -------
    tuple
        ``(cpx, cpy, end)`` where *end* is a boolean mask indicating
        whether the extra point was appended (True) or prepended (False).
    """
    dx = x2 - x1
    dy = y2 - y1
    slope = np.where(dx != 0.0, dy / np.where(dx != 0.0, dx, 1.0), np.inf)

    end = (slope > 1) | ((slope < 0) & (slope > -1))
    if curvature < 0:
        end = ~end

    abs_slope = np.abs(slope)
    sign_slope = np.sign(slope)

    startx = np.where(end, x1,
                       np.where(abs_slope > 1, x2 - dx, x2 - sign_slope * dy))
    starty = np.where(end, y1,
                       np.where(abs_slope > 1, y2 - sign_slope * dx, y2 - dy))
    endx = np.where(end,
                    np.where(abs_slope > 1, x1 + dx, x1 + sign_slope * dy),
                    x2)
    endy = np.where(end,
                    np.where(abs_slope > 1, y1 + sign_slope * dx, y1 + dy),
                    y2)

    cpx, cpy = _calc_control_points(startx, starty, endx, endy,
                                     curvature, angle, ncp)

    ncurve = len(x1)
    cpx = _interleave(ncp, ncurve, cpx, startx, endx, end)
    cpy = _interleave(ncp, ncurve, cpy, starty, endy, end)

    return cpx, cpy, end


# ===================================================================== #
#  Internal: curve point calculation                                     #
# ===================================================================== #


def _calc_curve_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    curvature: float = 1.0,
    angle: float = 90.0,
    ncp: int = 1,
    shape: float = 0.5,
    square: bool = True,
    squareShape: float = 1.0,
    inflect: bool = False,
    open_: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the full set of curve points (control + interpolation).

    This mirrors R's ``calcCurveGrob`` but returns the x-spline control
    points instead of building a grob tree.

    Parameters
    ----------
    x1, y1 : float
        Start point (in working coordinates, e.g. inches).
    x2, y2 : float
        End point.
    curvature : float
        Curvature parameter (0 = straight line).
    angle : float
        Angle in degrees (0--180).
    ncp : int
        Number of control points.
    shape : float
        X-spline shape parameter (-1 to 1).
    square : bool
        Whether to use "square" control-point placement.
    squareShape : float
        Shape for the extra square control point.
    inflect : bool
        Whether the curve should inflect at the midpoint.
    open_ : bool
        Whether the resulting spline is open.

    Returns
    -------
    tuple of ndarray
        ``(x_pts, y_pts)`` control-point arrays suitable for an x-spline.
    """
    ax1 = np.atleast_1d(np.asarray(x1, dtype=np.float64))
    ay1 = np.atleast_1d(np.asarray(y1, dtype=np.float64))
    ax2 = np.atleast_1d(np.asarray(x2, dtype=np.float64))
    ay2 = np.atleast_1d(np.asarray(y2, dtype=np.float64))

    # Outlaw identical endpoints
    if np.any((ax1 == ax2) & (ay1 == ay2)):
        raise ValueError("end points must not be identical")

    maxn = max(len(ax1), len(ay1), len(ax2), len(ay2))
    ax1 = np.resize(ax1, maxn)
    ay1 = np.resize(ay1, maxn)
    ax2 = np.resize(ax2, maxn)
    ay2 = np.resize(ay2, maxn)

    # Straight line
    if curvature == 0 or angle < 1 or angle > 179:
        return np.array([x1, x2], dtype=np.float64), np.array([y1, y2], dtype=np.float64)

    ncurve = maxn

    if inflect:
        xm = (ax1 + ax2) / 2.0
        ym = (ay1 + ay2) / 2.0
        shape_vec1 = np.tile(np.resize(np.atleast_1d(shape), ncp), ncurve)
        shape_vec2 = shape_vec1[::-1].copy()

        if square:
            cpx1, cpy1, end1 = _calc_square_control_points(
                ax1, ay1, xm, ym, curvature, angle, ncp)
            cpx2, cpy2, end2 = _calc_square_control_points(
                xm, ym, ax2, ay2, -curvature, angle, ncp)
            shape_vec1 = _interleave(ncp, ncurve, shape_vec1,
                                      np.full(ncurve, squareShape),
                                      np.full(ncurve, squareShape), end1)
            shape_vec2 = _interleave(ncp, ncurve, shape_vec2,
                                      np.full(ncurve, squareShape),
                                      np.full(ncurve, squareShape), end2)
            ncp_eff = ncp + 1
        else:
            cpx1, cpy1 = _calc_control_points(ax1, ay1, xm, ym,
                                                curvature, angle, ncp)
            cpx2, cpy2 = _calc_control_points(xm, ym, ax2, ay2,
                                                -curvature, angle, ncp)
            ncp_eff = ncp

        # Build arrays: x1, cps1, xm, cps2, x2
        all_x = np.concatenate([ax1, cpx1, xm, cpx2, ax2])
        all_y = np.concatenate([ay1, cpy1, ym, cpy2, ay2])
        all_shape = np.concatenate([
            np.zeros(ncurve), shape_vec1,
            np.zeros(ncurve), shape_vec2,
            np.zeros(ncurve),
        ])
        return all_x, all_y
    else:
        shape_vec = np.tile(np.resize(np.atleast_1d(shape), ncp), ncurve)

        if square:
            cpx, cpy, end = _calc_square_control_points(
                ax1, ay1, ax2, ay2, curvature, angle, ncp)
            shape_vec = _interleave(ncp, ncurve, shape_vec,
                                     np.full(ncurve, squareShape),
                                     np.full(ncurve, squareShape), end)
            ncp_eff = ncp + 1
        else:
            cpx, cpy = _calc_control_points(ax1, ay1, ax2, ay2,
                                             curvature, angle, ncp)
            ncp_eff = ncp

        all_x = np.concatenate([ax1, cpx, ax2])
        all_y = np.concatenate([ay1, cpy, ay2])
        return all_x, all_y


# ===================================================================== #
#  Internal: X-spline point calculation                                  #
# ===================================================================== #


def _calc_xspline_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    shape: Union[float, NDArray[np.float64]] = 0.0,
    open_: bool = True,
    repEnds: bool = True,
    units_per_inch: float = 1.0,
    device_size_in: Optional[Tuple[float, float]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate an X-spline through the given control points.

    Faithful port of R's ``GEXspline`` (engine.c:2048-2094) plus
    ``src/main/xspline.c`` (itself derived from XFig 3.2.4, which
    implements the Blanc & Schlick 1995 X-spline model verbatim).  The
    per-point ``shape`` parameter is in ``[-1, 1]`` with the standard
    interpretation:

    - ``shape < 0``: "interpolating" (curve passes through the point)
    - ``shape = 0``: control point is a sharp corner
    - ``shape > 0``: "approximating" (B-spline-like)

    Blending is done with the three polynomial kernels defined in the
    Blanc-Schlick paper — ``f_blend`` (quintic), ``g_blend`` (quintic),
    and ``h_blend`` (quartic).  These are **exact**, not a Catmull-Rom /
    B-spline / linear approximation.

    Like R (``COPY_CONTROL_POINT``, xspline.c:428-433), all internal
    math runs in xfig's 1200-points-per-inch space so the adaptive step
    computation samples at the same density as R, then results are
    converted back to the input units.

    Shapes are used exactly as given: R zeroes the first/last shape of
    each *open* spline at grob-validation time (``validDetails.xspline``,
    primitives.R:794-803), NOT inside the engine — build grobs through
    :func:`xspline_grob` to get that behaviour.

    Parameters
    ----------
    x, y : ndarray
        Control-point coordinates in any linear unit.
    shape : float or ndarray
        Per-control-point shape parameter(s) in ``[-1, 1]``.  Scalar is
        broadcast; a short vector is recycled (R engine recycles with
        ``s[j % length(s)]``, grid.c:2440).
    open_ : bool
        Open (True) or closed (False) spline.
    repEnds : bool
        For open splines, replicate the first and last control points so
        the curve passes through the endpoints.  Matches R's ``repEnds``.
    units_per_inch : float
        How many input units make one inch (1.0 for inches input;
        ``renderer._dev_units_per_inch`` for device coordinates).
    device_size_in : tuple of float or None
        ``(width, height)`` of the device in inches, used for R's
        step-count clamp at the device diagonal (xspline.c:311-327).
        ``None`` disables the clamp (points far off-device then sample
        more densely than R would).

    Returns
    -------
    tuple of ndarray
        ``(x_pts, y_pts)`` evaluated spline coordinates in input units.

    References
    ----------
    Blanc, C. and Schlick, C. (1995).  X-splines: A spline model designed
    for the end-user.  *Proceedings of SIGGRAPH 95*, pp. 377-386.

    R implementation: ``src/main/xspline.c`` + ``engine.c GEXspline``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        # R grid.c:2465-2467 raises for both the draw and bounds paths
        raise ValueError("non-finite control point in Xspline")

    if np.isscalar(shape):
        s = np.full(n, float(shape), dtype=np.float64)
    else:
        s = np.asarray(shape, dtype=np.float64)
        if len(s) < n:
            s = np.resize(s, n)

    # R converts device coordinates to xfig's 1200ppi space before any
    # spline math (COPY_CONTROL_POINT) and back on output (add_point).
    to_1200 = 1200.0 / float(units_per_inch)
    x1200 = x * to_1200
    y1200 = y * to_1200

    # Device-diagonal clamp for the step computation (xspline.c:311-327).
    if device_size_in is not None:
        dev_w = device_size_in[0] * 1200.0
        dev_h = device_size_in[1] * 1200.0
        dev_diag_1200 = math.sqrt(dev_w * dev_w + dev_h * dev_h)
    else:
        dev_diag_1200 = math.inf

    # R's precision parameter (LOW_PRECISION=1.0 is what ``GEXspline``
    # passes).  Step size is derived adaptively from segment geometry
    # (see ``_xsp_step``).
    precision = 1.0

    if open_:
        out_x, out_y = _xsp_compute_open(x1200, y1200, s, repEnds,
                                         precision, dev_diag_1200)
    else:
        out_x, out_y = _xsp_compute_closed(x1200, y1200, s,
                                           precision, dev_diag_1200)

    return out_x / to_1200, out_y / to_1200


# -- Blanc-Schlick polynomial blending kernels ------------------------------
#
# Direct port of ``f_blend`` / ``g_blend`` / ``h_blend`` in
# R's ``src/main/xspline.c`` (lines 138-159).  ``Q(s) = -s``.

def _xsp_f_blend(numerator: float, denominator: float) -> float:
    # f(u) = u^3 * (10 - p + (2p - 15) u + (6 - p) u^2),   p = 2*denom^2
    p = 2.0 * denominator * denominator
    u = numerator / denominator
    u2 = u * u
    return u * u2 * (10.0 - p + (2.0 * p - 15.0) * u + (6.0 - p) * u2)


def _xsp_g_blend(u: float, q: float) -> float:
    # g(u) = u * (q + u * (2q + u * (8 - 12q + u * (14q - 11 + u * (4 - 5q)))))
    return u * (q + u * (2.0 * q + u * (8.0 - 12.0 * q + u *
                 (14.0 * q - 11.0 + u * (4.0 - 5.0 * q)))))


def _xsp_h_blend(u: float, q: float) -> float:
    # h(u) = u * (q + u * (2q + u^2 * (-2q - u*q)))
    u2 = u * u
    return u * (q + u * (2.0 * q + u2 * (-2.0 * q - u * q)))


# -- Influence functions ----------------------------------------------------
#
# Direct port of ``negative_s1_influence`` / ``negative_s2_influence`` /
# ``positive_s1_influence`` / ``positive_s2_influence`` (xspline.c:161-197).
# ``Q(s) = -s`` is applied for the negative-s branches.

def _xsp_neg_s1(t: float, s1: float) -> Tuple[float, float]:
    q = -s1
    return _xsp_h_blend(-t, q), _xsp_g_blend(t, q)


def _xsp_neg_s2(t: float, s2: float) -> Tuple[float, float]:
    q = -s2
    return _xsp_g_blend(1.0 - t, q), _xsp_h_blend(t - 1.0, q)


def _xsp_pos_s1(k: float, t: float, s1: float) -> Tuple[float, float]:
    Tk = k + 1.0 + s1
    A0 = _xsp_f_blend(t + k + 1.0 - Tk, k - Tk) if (t + k + 1.0) < Tk else 0.0
    Tk = k + 1.0 - s1
    A2 = _xsp_f_blend(t + k + 1.0 - Tk, k + 2.0 - Tk)
    return A0, A2


def _xsp_pos_s2(k: float, t: float, s2: float) -> Tuple[float, float]:
    Tk = k + 2.0 + s2
    A1 = _xsp_f_blend(t + k + 1.0 - Tk, k + 1.0 - Tk)
    Tk = k + 2.0 - s2
    A3 = _xsp_f_blend(t + k + 1.0 - Tk, k + 3.0 - Tk) if (t + k + 1.0) > Tk else 0.0
    return A1, A3


def _xsp_weights(k: float, t: float, s1: float, s2: float
                 ) -> Tuple[float, float, float, float]:
    """Compute (A0, A1, A2, A3) blending weights for one ``(k, t, s1, s2)``."""
    if s1 < 0.0:
        A0, A2 = _xsp_neg_s1(t, s1)
    else:
        A0, A2 = _xsp_pos_s1(k, t, s1)
    if s2 < 0.0:
        A1, A3 = _xsp_neg_s2(t, s2)
    else:
        A1, A3 = _xsp_pos_s2(k, t, s2)
    return A0, A1, A2, A3


def _xsp_point(A: Tuple[float, float, float, float],
               px: Tuple[float, float, float, float],
               py: Tuple[float, float, float, float]
               ) -> Tuple[float, float]:
    """``point_computing`` / ``point_adding``: weighted blend normalised."""
    ws = A[0] + A[1] + A[2] + A[3]
    num_x = A[0] * px[0] + A[1] * px[1] + A[2] * px[2] + A[3] * px[3]
    num_y = A[0] * py[0] + A[1] * py[1] + A[2] * py[2] + A[3] * py[3]
    return num_x / ws, num_y / ws


# -- Adaptive step computation (xspline.c:224-342) --------------------------

_MAX_SPLINE_STEP = 0.2


def _xsp_step(k: int, px: Tuple[float, ...], py: Tuple[float, ...],
              s1: float, s2: float, precision: float,
              dev_diag_1200: float = math.inf) -> float:
    """Port of R's ``step_computing`` (xspline.c:224-341).

    The step is chosen so the polyline sampling resolution matches the
    physical distance from segment origin to extremity (in 1200ppi
    units), augmented by a curvature term (cosine of the
    origin-mid-extremity angle).  ``dev_diag_1200`` is the device
    diagonal in 1200ppi units: R clamps the distance there so control
    points far off-device do not produce "ridiculously many steps"
    (xspline.c:311-327).
    """
    if s1 == 0.0 and s2 == 0.0:
        return 1.0  # linear segment

    # origin (t=0)
    if s1 > 0.0:
        if s2 < 0.0:
            A0, A2 = _xsp_pos_s1(k, 0.0, s1)
            A1, A3 = _xsp_neg_s2(0.0, s2)
        else:
            A0, A2 = _xsp_pos_s1(k, 0.0, s1)
            A1, A3 = _xsp_pos_s2(k, 0.0, s2)
        xstart, ystart = _xsp_point((A0, A1, A2, A3), px, py)
    else:
        xstart, ystart = px[1], py[1]

    # extremity (t=1)
    if s2 > 0.0:
        if s1 < 0.0:
            A0, A2 = _xsp_neg_s1(1.0, s1)
            A1, A3 = _xsp_pos_s2(k, 1.0, s2)
        else:
            A0, A2 = _xsp_pos_s1(k, 1.0, s1)
            A1, A3 = _xsp_pos_s2(k, 1.0, s2)
        xend, yend = _xsp_point((A0, A1, A2, A3), px, py)
    else:
        xend, yend = px[2], py[2]

    # midpoint (t=0.5)
    if s2 > 0.0:
        if s1 < 0.0:
            A0, A2 = _xsp_neg_s1(0.5, s1)
            A1, A3 = _xsp_pos_s2(k, 0.5, s2)
        else:
            A0, A2 = _xsp_pos_s1(k, 0.5, s1)
            A1, A3 = _xsp_pos_s2(k, 0.5, s2)
    elif s1 < 0.0:
        A0, A2 = _xsp_neg_s1(0.5, s1)
        A1, A3 = _xsp_neg_s2(0.5, s2)
    else:
        A0, A2 = _xsp_pos_s1(k, 0.5, s1)
        A1, A3 = _xsp_neg_s2(0.5, s2)
    xmid, ymid = _xsp_point((A0, A1, A2, A3), px, py)

    xv1, yv1 = xstart - xmid, ystart - ymid
    xv2, yv2 = xend - xmid, yend - ymid
    scal = xv1 * xv2 + yv1 * yv2
    sides = math.sqrt((xv1 * xv1 + yv1 * yv1) * (xv2 * xv2 + yv2 * yv2))
    angle_cos = 0.0 if sides == 0.0 else scal / sides

    xlen = xend - xstart
    ylen = yend - ystart
    start_to_end_dist = math.sqrt(xlen * xlen + ylen * ylen)

    # Coordinates are already in 1200ppi units (converted by
    # ``_calc_xspline_points``, mirroring R's COPY_CONTROL_POINT).
    # Clamp remote origin/extremity pairs at the device diagonal
    # (xspline.c:311-327, "Paul 2009-01-25").
    if start_to_end_dist > dev_diag_1200:
        start_to_end_dist = dev_diag_1200

    # more steps if segment's origin and extremity are remote
    n_steps = math.sqrt(start_to_end_dist) / 2.0
    # more steps if the curve is high
    n_steps += int((1.0 + angle_cos) * 10.0)
    step = 1.0 if n_steps == 0 else precision / n_steps
    if step > _MAX_SPLINE_STEP or step == 0.0:
        step = _MAX_SPLINE_STEP
    return step


# -- Segment sampling (xspline.c:344-423) -----------------------------------

def _xsp_segment(step: float, k: int,
                 px: Tuple[float, ...], py: Tuple[float, ...],
                 s1: float, s2: float,
                 out_x: List[float], out_y: List[float]) -> None:
    """Port of ``spline_segment_computing`` — sample segment over ``t ∈ [0, 1)``.

    Every sample is appended unconditionally.  R's ``add_point`` has a
    "skip identical point" check, but it compares the stored *device*
    coordinate against the incoming *1200ppi* coordinate (xspline.c:81-87)
    so it effectively never fires; duplicate points are instead trimmed
    from the two ENDS by the grid.c layer (see ``_trim_identical_ends``).
    """
    t = 0.0
    while t < 1.0:
        A = _xsp_weights(k, t, s1, s2)
        bx, by = _xsp_point(A, px, py)
        out_x.append(bx)
        out_y.append(by)
        t += step


def _xsp_last_segment(step: float, k: int,
                      px: Tuple[float, ...], py: Tuple[float, ...],
                      s1: float, s2: float,
                      out_x: List[float], out_y: List[float]) -> None:
    """Port of ``spline_last_segment_computing`` — one point at t=1."""
    A = _xsp_weights(k, 1.0, s1, s2)
    bx, by = _xsp_point(A, px, py)
    out_x.append(bx)
    out_y.append(by)


# -- Open / closed drivers (xspline.c:455-547) ------------------------------

def _xsp_compute_open(
    x: NDArray[np.float64], y: NDArray[np.float64], s: NDArray[np.float64],
    repEnds: bool, precision: float, dev_diag_1200: float = math.inf,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port of ``compute_open_spline`` (xspline.c:459-521).

    Returns the raw evaluated points; duplicate-end trimming is a
    separate, caller-level concern (grid.c:2475-2504, see
    ``_trim_identical_ends``).
    """
    n = len(x)
    if repEnds and n < 2:
        raise ValueError("there must be at least two control points")
    if not repEnds and n < 4:
        raise ValueError("there must be at least four control points")

    out_x: List[float] = []
    out_y: List[float] = []

    if repEnds:
        # First control point is needed twice for the first segment.
        # px/py/ps arrays are the 4-point sliding window.
        px = [x[0], x[0], x[1], x[2 if n > 2 else 1]]
        py = [y[0], y[0], y[1], y[2 if n > 2 else 1]]
        ps = [s[0], s[0], s[1], s[2 if n > 2 else 1]]

        k = 0
        while True:
            step = _xsp_step(k, px, py, ps[1], ps[2], precision,
                             dev_diag_1200)
            _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                         out_x, out_y)
            if k + 3 >= n:
                break
            # R's ``NEXT_CONTROL_POINTS(K, N)`` macro (xspline.c:435-439):
            # ``px[0] = x[K % N]``, ``px[1] = x[(K+1) % N]``, etc.  K is the
            # CURRENT segment index — not incremented before indexing.  Note
            # this is why the sliding window overlaps between iterations.
            px = [x[k % n], x[(k + 1) % n], x[(k + 2) % n], x[(k + 3) % n]]
            py = [y[k % n], y[(k + 1) % n], y[(k + 2) % n], y[(k + 3) % n]]
            ps = [s[k % n], s[(k + 1) % n], s[(k + 2) % n], s[(k + 3) % n]]
            k += 1

        # Last control point needed twice for the last segment.
        if n == 2:
            px = [x[n - 2], x[n - 2], x[n - 1], x[n - 1]]
            py = [y[n - 2], y[n - 2], y[n - 1], y[n - 1]]
            ps = [s[n - 2], s[n - 2], s[n - 1], s[n - 1]]
        else:
            px = [x[n - 3], x[n - 2], x[n - 1], x[n - 1]]
            py = [y[n - 3], y[n - 2], y[n - 1], y[n - 1]]
            ps = [s[n - 3], s[n - 2], s[n - 1], s[n - 1]]
        step = _xsp_step(k, px, py, ps[1], ps[2], precision, dev_diag_1200)
        _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                     out_x, out_y)

        # Final point: add_point(px[3], py[3]) (xspline.c:507)
        out_x.append(float(px[3]))
        out_y.append(float(py[3]))
    else:
        # repEnds=False: no endpoint replication.  Exactly n-3 segments,
        # then one final-segment t=1 point.
        step = 0.0
        for k in range(n - 3):
            px = [x[k], x[k + 1], x[k + 2], x[k + 3]]
            py = [y[k], y[k + 1], y[k + 2], y[k + 3]]
            ps = [s[k], s[k + 1], s[k + 2], s[k + 3]]
            step = _xsp_step(k, px, py, ps[1], ps[2], precision,
                             dev_diag_1200)
            _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                         out_x, out_y)
        # Last segment's t=1 evaluation (xspline.c:513)
        k = n - 4
        px = [x[k], x[k + 1], x[k + 2], x[k + 3]]
        py = [y[k], y[k + 1], y[k + 2], y[k + 3]]
        ps = [s[k], s[k + 1], s[k + 2], s[k + 3]]
        _xsp_last_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                          out_x, out_y)

    return (np.asarray(out_x, dtype=np.float64),
            np.asarray(out_y, dtype=np.float64))


def _xsp_compute_closed(
    x: NDArray[np.float64], y: NDArray[np.float64], s: NDArray[np.float64],
    precision: float, dev_diag_1200: float = math.inf,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port of ``compute_closed_spline`` (xspline.c:523-549)."""
    n = len(x)
    if n < 3:
        raise ValueError("There must be at least three control points")

    out_x: List[float] = []
    out_y: List[float] = []

    # INIT_CONTROL_POINTS: (n-1, 0, 1, 2) mod n
    idx = [(n - 1) % n, 0 % n, 1 % n, 2 % n]
    px = [x[i] for i in idx]
    py = [y[i] for i in idx]
    ps = [s[i] for i in idx]

    for k in range(n):
        step = _xsp_step(k, px, py, ps[1], ps[2], precision, dev_diag_1200)
        _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                     out_x, out_y)
        # NEXT_CONTROL_POINTS(K, N) with the CURRENT k: (K..K+3) mod n —
        # segment k+1 then blends window (k, k+1, k+2, k+3), i.e. the n
        # windows are (n-1,0,1,2), (0,1,2,3), ..., (n-2,n-1,0,1).
        idx = [k % n, (k + 1) % n, (k + 2) % n, (k + 3) % n]
        px = [x[i] for i in idx]
        py = [y[i] for i in idx]
        ps = [s[i] for i in idx]

    return (np.asarray(out_x, dtype=np.float64),
            np.asarray(out_y, dtype=np.float64))


# ===================================================================== #
#  Internal: shared xspline helpers (trim / index / device size)         #
# ===================================================================== #


def _trim_identical_ends(
    xs: NDArray[np.float64], ys: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trim runs of identical points from both ENDS of a point list.

    Port of the "trim identical points from the ends (so arrow heads are
    drawn at correct angle)" block in ``gridXspline`` (grid.c:2475-2504).
    Interior duplicates are kept, exactly like R.
    """
    np_count = len(xs)
    start = 0
    end = np_count - 1
    while (np_count > 1 and xs[start] == xs[start + 1]
           and ys[start] == ys[start + 1]):
        start += 1
        np_count -= 1
    while (np_count > 1 and xs[end] == xs[end - 1]
           and ys[end] == ys[end - 1]):
        end -= 1
        np_count -= 1
    return xs[start:end + 1], ys[start:end + 1]


def _xspline_index(x: Grob) -> List[NDArray[np.intp]]:
    """Per-spline control-point index groups (0-based).

    Port of R ``xsplineIndex`` (primitives.R:807-820): one index vector
    per spline, derived from ``id`` (split by value, ascending) or
    ``id_lengths`` (consecutive runs); a single full-range group when
    neither is given.
    """
    n = len(x.x)
    id_ = getattr(x, "id", None)
    id_lengths = getattr(x, "id_lengths", None)
    if id_ is None and id_lengths is None:
        return [np.arange(n, dtype=np.intp)]
    if id_ is None:
        lengths = np.atleast_1d(np.asarray(id_lengths, dtype=np.intp))
        id_vec = np.repeat(np.arange(1, len(lengths) + 1), lengths)
    else:
        id_vec = np.atleast_1d(np.asarray(id_, dtype=np.intp))
    return [np.flatnonzero(id_vec == uid) for uid in np.unique(id_vec)]


def _device_size_in() -> Tuple[float, float]:
    """Current device ``(width, height)`` in inches.

    R equivalent: ``fromDeviceWidth(toDeviceWidth(1, GE_NDC, dd),
    GE_INCHES, dd)`` in ``step_computing`` (xspline.c:317-320).  Falls
    back to the grid state's device dimensions when no renderer is
    active (R always has a device open, so only the renderer branch has
    an R analogue).
    """
    from ._state import get_state

    state = get_state()
    renderer = state.get_renderer()
    if renderer is not None:
        return float(renderer.width_in), float(renderer.height_in)
    return (state._device_width_cm / 2.54, state._device_height_cm / 2.54)


# ===================================================================== #
#  curveGrob / grid.curve                                                #
# ===================================================================== #


def curve_grob(
    x1: Any = 0,
    y1: Any = 0,
    x2: Any = 1,
    y2: Any = 1,
    default_units: str = "npc",
    curvature: float = 1.0,
    angle: float = 90.0,
    ncp: int = 1,
    shape: float = 0.5,
    square: bool = True,
    squareShape: float = 1.0,
    inflect: bool = False,
    arrow: Optional[Arrow] = None,
    open_: bool = True,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GTree:
    """Create a *curve* grob (GTree).

    A curve grob draws a smooth curve between two endpoints.  The shape
    of the curve is controlled by ``curvature``, ``angle``, ``ncp``, and
    ``shape``.

    Parameters
    ----------
    x1, y1 : Unit or numeric
        Start-point coordinates.
    x2, y2 : Unit or numeric
        End-point coordinates.
    default_units : str
        Unit type for bare numerics (default ``"npc"``).
    curvature : float
        Amount of curvature.  0 = straight line, positive curves right,
        negative curves left.
    angle : float
        Angle in degrees (0--180) controlling the skewness of the curve.
    ncp : int
        Number of control points on the curve.
    shape : float
        X-spline shape parameter (-1 to 1).
    square : bool
        Whether to use "square" control-point placement for better
        aesthetics with right-angled curves.
    squareShape : float
        Shape for extra square control point (-1 to 1).
    inflect : bool
        Whether the curve should inflect at the midpoint.
    arrow : Arrow or None
        Arrow-head specification.
    open_ : bool
        Whether the spline is open.
    name : str or None
        Grob name (auto-generated when ``None``).
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    GTree
        A grob tree with ``_grid_class="curve"``.

    Raises
    ------
    ValueError
        If ``shape`` or ``squareShape`` is outside [-1, 1].
    """
    if not (-1 <= shape <= 1):
        raise ValueError("'shape' must be between -1 and 1")
    if not (-1 <= squareShape <= 1):
        raise ValueError("'squareShape' must be between -1 and 1")

    ux1 = _ensure_unit(x1, default_units)
    uy1 = _ensure_unit(y1, default_units)
    ux2 = _ensure_unit(x2, default_units)
    uy2 = _ensure_unit(y2, default_units)

    angle = angle % 180

    return _CurveGrob(
        name=name,
        gp=gp,
        vp=vp,
        _grid_class="curve",
        x1=ux1,
        y1=uy1,
        x2=ux2,
        y2=uy2,
        curvature=float(curvature),
        angle=float(angle),
        ncp=int(ncp),
        shape=float(shape),
        square=bool(square),
        squareShape=float(squareShape),
        inflect=bool(inflect),
        arrow=arrow,
        open_=bool(open_),
    )


class _CurveGrob(GTree):
    """GTree for ``_grid_class="curve"``.

    ``make_content`` lazily expands the curve into ``segments`` and / or
    ``xspline`` children at draw time, so endpoint unit conversion happens
    in the current viewport context.
    """

    def make_content(self) -> Grob:
        return _calc_curve_content(self)


def _calc_curve_content(x: "_CurveGrob") -> GTree:
    """Expand a curve grob into a gTree of segments / xspline children.

    curvature = 0 or near-flat angles produce a plain ``segments_grob``.
    Under ``square=True`` horizontal / vertical segments are peeled off
    (``_calc_control_points`` divides by dx / dy).  Other cases build an
    xspline from control points, optionally reflecting about the midpoint
    when ``inflect=True``.
    """
    x1_u = x.x1
    y1_u = x.y1
    x2_u = x.x2
    y2_u = x.y2
    curvature = float(x.curvature)
    angle = float(x.angle)
    ncp = int(x.ncp)
    shape = float(x.shape)
    square = bool(x.square)
    squareShape = float(x.squareShape)
    inflect = bool(x.inflect)
    arrow = x.arrow
    open_ = bool(x.open_)

    x1 = np.atleast_1d(np.asarray(convert_x(x1_u, "inches", valueOnly=True), dtype=float))
    y1 = np.atleast_1d(np.asarray(convert_y(y1_u, "inches", valueOnly=True), dtype=float))
    x2 = np.atleast_1d(np.asarray(convert_x(x2_u, "inches", valueOnly=True), dtype=float))
    y2 = np.atleast_1d(np.asarray(convert_y(y2_u, "inches", valueOnly=True), dtype=float))

    if np.any((x1 == x2) & (y1 == y2)):
        raise ValueError("end points must not be identical")

    maxn = int(max(len(x1), len(y1), len(x2), len(y2)))
    x1 = np.resize(x1, maxn)
    y1 = np.resize(y1, maxn)
    x2 = np.resize(x2, maxn)
    y2 = np.resize(y2, maxn)

    def _straight(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> Grob:
        return segments_grob(
            x0=a1, y0=b1, x1=a2, y1=b2,
            default_units="inches", arrow=arrow, name="segment",
        )

    children_list: List[Grob] = []

    if curvature == 0:
        children_list.append(_straight(x1, y1, x2, y2))
    else:
        if angle < 1 or angle > 179:
            children_list.append(_straight(x1, y1, x2, y2))
        else:
            straight_grob: Optional[Grob] = None
            if square:
                subset = (x1 == x2) | (y1 == y2)
                if np.any(subset):
                    straight_grob = _straight(x1[subset], y1[subset], x2[subset], y2[subset])
                    keep = ~subset
                    x1 = x1[keep]
                    y1 = y1[keep]
                    x2 = x2[keep]
                    y2 = y2[keep]

            ncurve = int(len(x1))
            if ncurve == 0:
                if straight_grob is not None:
                    children_list.append(straight_grob)
            else:
                base_shape = np.full(ncp * ncurve, shape, dtype=float)

                if inflect:
                    xm = (x1 + x2) / 2.0
                    ym = (y1 + y2) / 2.0
                    shape1 = base_shape.copy()
                    shape2 = base_shape[::-1].copy()

                    if square:
                        cpx1, cpy1, end1 = _calc_square_control_points(
                            x1, y1, xm, ym, curvature, angle, ncp,
                        )
                        cpx2, cpy2, end2 = _calc_square_control_points(
                            xm, ym, x2, y2, -curvature, angle, ncp,
                        )
                        shape1 = _interleave(
                            ncp, ncurve, shape1,
                            np.full(ncurve, squareShape),
                            np.full(ncurve, squareShape),
                            end1,
                        )
                        shape2 = _interleave(
                            ncp, ncurve, shape2,
                            np.full(ncurve, squareShape),
                            np.full(ncurve, squareShape),
                            end2,
                        )
                        ncp_eff = ncp + 1
                    else:
                        cpx1, cpy1 = _calc_control_points(
                            x1, y1, xm, ym, curvature, angle, ncp,
                        )
                        cpx2, cpy2 = _calc_control_points(
                            xm, ym, x2, y2, -curvature, angle, ncp,
                        )
                        ncp_eff = ncp

                    idset = np.arange(1, ncurve + 1, dtype=int)
                    spline_x = np.concatenate([x1, cpx1, xm, cpx2, x2])
                    spline_y = np.concatenate([y1, cpy1, ym, cpy2, y2])
                    rep_id = np.repeat(idset, ncp_eff)
                    spline_id = np.concatenate([idset, rep_id, idset, rep_id, idset])
                    spline_shape = np.concatenate([
                        np.zeros(ncurve),
                        shape1,
                        np.zeros(ncurve),
                        shape2,
                        np.zeros(ncurve),
                    ])
                    spline = xspline_grob(
                        x=spline_x, y=spline_y,
                        default_units="inches",
                        shape=spline_shape,
                        open_=open_, arrow=arrow,
                        name="xspline",
                    )
                    spline.id = spline_id
                    if straight_grob is not None:
                        children_list.extend([straight_grob, spline])
                    else:
                        children_list.append(spline)
                else:
                    shape_arr = base_shape
                    if square:
                        cpx, cpy, cend = _calc_square_control_points(
                            x1, y1, x2, y2, curvature, angle, ncp,
                        )
                        shape_arr = _interleave(
                            ncp, ncurve, shape_arr,
                            np.full(ncurve, squareShape),
                            np.full(ncurve, squareShape),
                            cend,
                        )
                        ncp_eff = ncp + 1
                    else:
                        cpx, cpy = _calc_control_points(
                            x1, y1, x2, y2, curvature, angle, ncp,
                        )
                        ncp_eff = ncp

                    idset = np.arange(1, ncurve + 1, dtype=int)
                    spline_x = np.concatenate([x1, cpx, x2])
                    spline_y = np.concatenate([y1, cpy, y2])
                    spline_id = np.concatenate([
                        idset,
                        np.repeat(idset, ncp_eff),
                        idset,
                    ])
                    spline_shape = np.concatenate([
                        np.zeros(ncurve),
                        shape_arr,
                        np.zeros(ncurve),
                    ])
                    spline = xspline_grob(
                        x=spline_x, y=spline_y,
                        default_units="inches",
                        shape=spline_shape,
                        open_=open_, arrow=arrow,
                        name="xspline",
                    )
                    spline.id = spline_id
                    if straight_grob is not None:
                        children_list.extend([straight_grob, spline])
                    else:
                        children_list.append(spline)

    return GTree(
        children=GList(*children_list),
        name=x.name, gp=x.gp, vp=x.vp,
    )


def grid_curve(
    x1: Any = 0,
    y1: Any = 0,
    x2: Any = 1,
    y2: Any = 1,
    default_units: str = "npc",
    curvature: float = 1.0,
    angle: float = 90.0,
    ncp: int = 1,
    shape: float = 0.5,
    square: bool = True,
    squareShape: float = 1.0,
    inflect: bool = False,
    arrow: Optional[Arrow] = None,
    open_: bool = True,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> GTree:
    """Create and optionally draw a *curve* grob.

    Parameters
    ----------
    x1, y1 : Unit or numeric
        Start-point coordinates.
    x2, y2 : Unit or numeric
        End-point coordinates.
    default_units : str
        Unit type for bare numerics.
    curvature : float
        Curvature parameter.
    angle : float
        Angle in degrees (0--180).
    ncp : int
        Number of control points.
    shape : float
        X-spline shape (-1 to 1).
    square : bool
        Use square control-point placement.
    squareShape : float
        Shape for extra square point.
    inflect : bool
        Inflect at midpoint.
    arrow : Arrow or None
        Arrow specification.
    open_ : bool
        Open spline.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), immediately record the grob for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    GTree
        The curve grob.
    """
    grob = curve_grob(
        x1=x1, y1=y1, x2=x2, y2=y2,
        default_units=default_units,
        curvature=curvature, angle=angle, ncp=ncp,
        shape=shape, square=square, squareShape=squareShape,
        inflect=inflect, arrow=arrow, open_=open_,
        name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


# ===================================================================== #
#  xsplineGrob / grid.xspline                                           #
# ===================================================================== #


class _XsplineGrob(Grob):
    """Grob for ``_grid_class="xspline"``.

    ``valid_details`` ports R ``validDetails.xspline``
    (primitives.R:772-805).  In particular, the first and last shape of
    every *open* spline (per ``id`` group) is forced to 0 at validation
    time, which is what makes open X-splines start and end at their end
    control points — the engine itself (``_calc_xspline_points``) uses
    shapes exactly as given, like R's C code.
    """

    def valid_details(self) -> None:
        if not is_unit(self.x) or not is_unit(self.y):
            raise TypeError("x and y must be units")
        if self.id is not None and self.id_lengths is not None:
            raise ValueError(
                "it is invalid to specify both 'id' and 'id.lengths'")
        nx = len(self.x)
        ny = len(self.y)
        if nx != ny:
            raise ValueError("'x' and 'y' must be same length")
        if self.id is not None:
            self.id = np.atleast_1d(np.asarray(self.id, dtype=np.int64))
            if len(self.id) != nx:
                raise ValueError(
                    "'x' and 'y' and 'id' must all be same length")
        if self.id_lengths is not None:
            self.id_lengths = np.atleast_1d(
                np.asarray(self.id_lengths, dtype=np.int64))
            if int(self.id_lengths.sum()) != nx:
                raise ValueError(
                    "'x' and 'y' and 'id.lengths' must specify same "
                    "overall length")
        if self.arrow is not None and not isinstance(self.arrow, Arrow):
            raise TypeError("invalid 'arrow' argument")
        shape = np.atleast_1d(np.asarray(self.shape, dtype=np.float64))
        if np.any((shape < -1) | (shape > 1)):
            raise ValueError("'shape' must be between -1 and 1")
        self.open_ = bool(self.open_)
        # Force all first and last shapes to be 0 for open xsplines
        if self.open_:
            shape = np.resize(shape, nx)
            for idx in _xspline_index(self):
                shape[int(idx.min())] = 0.0
                shape[int(idx.max())] = 0.0
        self.shape = shape


def xspline_grob(
    x: Any = (0, 0.5, 1, 0.5),
    y: Any = (0.5, 1, 0.5, 0),
    id: Optional[Any] = None,
    id_lengths: Optional[Any] = None,
    default_units: str = "npc",
    shape: Union[float, Sequence[float]] = 0.0,
    open_: bool = True,
    arrow: Optional[Arrow] = None,
    repEnds: bool = True,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create an *xspline* grob.

    An X-spline grob draws a smooth curve through control points whose
    shape is governed by per-point ``shape`` parameters.

    Parameters
    ----------
    x, y : Unit, numeric, or sequence
        Control-point coordinates.  Defaults mirror R's
        ``xsplineGrob()`` (primitives.R:863).
    id : array-like of int or None
        Group label for each control point.  Points sharing an ``id`` are
        rendered as one X-spline; the grob therefore renders one spline
        per unique ``id`` value.  Mirrors R ``xsplineGrob(id=...)``.
        Mutually exclusive with ``id_lengths``.
    id_lengths : array-like of int or None
        Run-length encoding of ``id``: the n-th entry is the number of
        consecutive control points belonging to spline n.  Mirrors R's
        ``xsplineGrob(id.lengths=...)``.
    default_units : str
        Unit type for bare numerics.
    shape : float or sequence of float
        Shape parameter(s) in [-1, 1].  A scalar is broadcast to all
        control points.  Following R ``validDetails.xspline``, the first
        and last shape of each open spline is forced to 0 on the created
        grob.
    open_ : bool
        Whether the spline is open (True) or closed (False).
    arrow : Arrow or None
        Arrow-head specification.
    repEnds : bool
        Whether to replicate endpoints so the spline passes through them.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="xspline"``.
    """
    ux = _ensure_unit(x, default_units)
    uy = _ensure_unit(y, default_units)

    id_arr = None if id is None else np.asarray(id, dtype=np.int64)
    id_lengths_arr = (
        None if id_lengths is None else np.asarray(id_lengths, dtype=np.int64)
    )

    return _XsplineGrob(
        x=ux,
        y=uy,
        id=id_arr,
        id_lengths=id_lengths_arr,
        shape=shape,
        open_=open_,
        arrow=arrow,
        repEnds=bool(repEnds),
        name=name,
        gp=gp,
        vp=vp,
        _grid_class="xspline",
    )


def grid_xspline(
    x: Any = (0, 0.5, 1, 0.5),
    y: Any = (0.5, 1, 0.5, 0),
    id: Optional[Any] = None,
    id_lengths: Optional[Any] = None,
    default_units: str = "npc",
    shape: Union[float, Sequence[float]] = 0.0,
    open_: bool = True,
    arrow: Optional[Arrow] = None,
    repEnds: bool = True,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw an *xspline* grob.

    Parameters
    ----------
    x, y : Unit, numeric, sequence, or None
        Control-point coordinates.
    default_units : str
        Unit type for bare numerics.
    shape : float or sequence of float
        Shape parameter(s).
    open_ : bool
        Open spline.
    arrow : Arrow or None
        Arrow specification.
    repEnds : bool
        Replicate endpoints.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), record for drawing.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The xspline grob.
    """
    grob = xspline_grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        default_units=default_units,
        shape=shape, open_=open_, arrow=arrow,
        repEnds=repEnds, name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


def xspline_points(
    x: Grob,
) -> Union[Dict[str, NDArray[np.float64]], List[Dict[str, NDArray[np.float64]]]]:
    """Extract the evaluated X-spline curve from an xspline grob.

    Port of R ``xsplinePoints`` (primitives.R:881-905): enforces the
    grob's ``vp`` and ``gp`` like ``drawGrob()`` (preDraw/postDraw),
    converts the control points to inches in that context, evaluates one
    spline per ``id`` group, and trims runs of identical points from the
    curve ends (grid.c:2475-2504).

    Parameters
    ----------
    x : Grob
        An xspline grob (``_grid_class="xspline"``).

    Returns
    -------
    dict or list of dict
        For a single spline, a dict with keys ``"x"`` and ``"y"`` holding
        the evaluated curve coordinates in INCHES (R returns
        ``unit(..., "inches")``).  For multiple splines (``id`` /
        ``id_lengths``), a list of such dicts — mirroring R, which
        returns a list of point sets and unwraps it when there is
        exactly one.

    Raises
    ------
    TypeError
        If *x* is not an xspline grob.
    """
    if not isinstance(x, Grob) or getattr(x, "_grid_class", None) != "xspline":
        raise TypeError("'x' must be an xspline grob")

    import copy as _copy

    from ._draw import _pop_grob_vp, _push_vp_gp
    from ._state import get_state

    state = get_state()
    # Mimic drawGrob() to ensure x$vp and x$gp enforced (primitives.R:882-887)
    saved_dl_on = state._dl_on
    state.set_display_list_on(False)
    saved_gpar = _copy.copy(state.get_gpar())
    pushed = False
    try:
        _push_vp_gp(x)  # preDraw(x)
        pushed = True

        xx = np.atleast_1d(np.asarray(
            convert_x(x.x, "inches", valueOnly=True), dtype=np.float64))
        yy = np.atleast_1d(np.asarray(
            convert_y(x.y, "inches", valueOnly=True), dtype=np.float64))
        # C recycles shape per control-point index (grid.c:2440)
        shape = np.resize(
            np.atleast_1d(np.asarray(getattr(x, "shape", 0.0),
                                     dtype=np.float64)),
            len(xx))
        open_ = bool(getattr(x, "open_", True))
        repEnds = bool(getattr(x, "repEnds", True))
        device_size = _device_size_in()

        results: List[Dict[str, NDArray[np.float64]]] = []
        for idx in _xspline_index(x):
            px, py = _calc_xspline_points(
                xx[idx], yy[idx], shape[idx], open_, repEnds,
                units_per_inch=1.0, device_size_in=device_size,
            )
            px, py = _trim_identical_ends(px, py)
            results.append({"x": px, "y": py})
    finally:
        # postDraw(x) + state restore
        if pushed and x.vp is not None:
            _pop_grob_vp(x.vp)
        state.replace_gpar(saved_gpar)
        state.set_display_list_on(saved_dl_on)

    if len(results) == 1:
        return results[0]
    return results


# ===================================================================== #
#  bezierGrob / grid.bezier                                              #
# ===================================================================== #
#
# A bezier grob that works off a (not-100% accurate) approximation using
# X-splines (R primitives.R:906-1028): the four Bezier control points of
# each curve are mapped through solve(Ms) %*% Mb — cubic-B-spline basis
# inverse times cubic-Bezier basis — giving X-spline control points, and
# the result is drawn as an X-spline with shape=1, repEnds=FALSE.

# X-Spline approx to Bezier (R primitives.R:913-917)
_BEZIER_MS = (1.0 / 6.0) * np.array(
    [[1, 4, 1, 0],
     [-3, 0, 3, 0],
     [3, -6, 3, 0],
     [-1, 3, -3, 1]], dtype=np.float64)
# Bezier control matrix (R primitives.R:919-923)
_BEZIER_MB = np.array(
    [[1, 0, 0, 0],
     [-3, 3, 0, 0],
     [3, -6, 3, 0],
     [-1, 3, -3, 1]], dtype=np.float64)
# R: Msinv %*% Mb  (Msinv <- solve(Ms), primitives.R:918)
_SPLINE_FROM_BEZIER = np.linalg.solve(_BEZIER_MS, _BEZIER_MB)


def _spline_points(
    xb: NDArray[np.float64],
    yb: NDArray[np.float64],
    id_index: List[NDArray[np.intp]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """X-spline control points from Bezier control points.

    Port of R ``splinePoints`` (primitives.R:925-936): applies
    ``Msinv %*% Mb`` to each 4-point group.
    """
    xs = np.concatenate([_SPLINE_FROM_BEZIER @ xb[i] for i in id_index])
    ys = np.concatenate([_SPLINE_FROM_BEZIER @ yb[i] for i in id_index])
    return xs, ys


def _splinegrob(x: Grob) -> Grob:
    """The X-spline grob that draws for a bezier grob.

    Port of R ``splinegrob`` (primitives.R:938-948): control points are
    converted to inches in the CURRENT viewport context (at draw time
    that is the bezier grob's own context, because ``make_content`` runs
    after preDraw has pushed ``vp``/``gp``), transformed per curve, and
    wrapped in an xspline grob with ``shape=1, repEnds=FALSE``.
    """
    xx = np.atleast_1d(np.asarray(
        convert_x(x.x, "inches", valueOnly=True), dtype=np.float64))
    yy = np.atleast_1d(np.asarray(
        convert_y(x.y, "inches", valueOnly=True), dtype=np.float64))
    sx, sy = _spline_points(xx, yy, _xspline_index(x))
    return xspline_grob(
        sx, sy, default_units="inches",
        id=x.id, id_lengths=x.id_lengths,
        shape=1, repEnds=False,
        arrow=x.arrow, name=x.name,
        gp=x.gp, vp=x.vp,
    )


class _BezierGrob(Grob):
    """Grob for ``_grid_class="beziergrob"``.

    Mirrors R's beziergrob: a plain grob whose ``make_content`` expands
    to the X-spline approximation (``makeContent.beziergrob``,
    primitives.R:985-987) so unit conversion happens in the current
    viewport context at draw time.
    """

    def valid_details(self) -> None:
        # Port of R validDetails.beziergrob (primitives.R:948-983)
        if not is_unit(self.x) or not is_unit(self.y):
            raise TypeError("x and y must be units")
        if self.id is not None and self.id_lengths is not None:
            raise ValueError(
                "it is invalid to specify both 'id' and 'id.lengths'")
        nx = len(self.x)
        ny = len(self.y)
        if nx != ny:
            raise ValueError("'x' and 'y' must be same length")
        if self.id is not None:
            self.id = np.atleast_1d(np.asarray(self.id, dtype=np.int64))
            if len(self.id) != nx:
                raise ValueError(
                    "'x' and 'y' and 'id' must all be same length")
        if self.id_lengths is not None:
            self.id_lengths = np.atleast_1d(
                np.asarray(self.id_lengths, dtype=np.int64))
            if int(self.id_lengths.sum()) != nx:
                raise ValueError(
                    "'x' and 'y' and 'id.lengths' must specify same "
                    "overall length")
        if self.id is None and self.id_lengths is None:
            if nx != 4:
                raise ValueError("must have exactly 4 control points")
        elif any(len(idx) != 4 for idx in _xspline_index(self)):
            raise ValueError(
                "must have exactly 4 control points per Bezier curve")
        if self.arrow is not None and not isinstance(self.arrow, Arrow):
            raise TypeError("invalid 'arrow' argument")

    def make_content(self) -> Grob:
        # Port of R makeContent.beziergrob (primitives.R:985-987)
        return _splinegrob(self)

    def x_details(self, theta: float = 0.0) -> Any:
        # Port of R xDetails.beziergrob (primitives.R:989-991)
        from ._size import x_details
        return x_details(_splinegrob(self), theta)

    def y_details(self, theta: float = 0.0) -> Any:
        # Port of R yDetails.beziergrob (primitives.R:993-995)
        from ._size import y_details
        return y_details(_splinegrob(self), theta)


def bezier_grob(
    x: Any = (0, 0.5, 1, 0.5),
    y: Any = (0.5, 1, 0.5, 0),
    id: Optional[Any] = None,
    id_lengths: Optional[Any] = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Create a *bezier* grob.

    A Bezier grob draws a cubic Bezier curve through the given control
    points: the curve interpolates the first and last point of each
    group of 4 and is attracted toward the middle two.  Like R, the
    curve is drawn as an X-spline approximation of the Bezier
    (see ``bezierGrob``, primitives.R:1005-1016), which is close to but
    not exactly the true Bezier.

    Parameters
    ----------
    x, y : Unit or numeric
        Control-point coordinates; exactly 4 per curve.  Defaults mirror
        R's ``bezierGrob()``.
    id : array-like of int or None
        Curve label per control point (4 points per label).  Mutually
        exclusive with ``id_lengths``.
    id_lengths : array-like of int or None
        Run-length encoding of ``id``; every entry must be 4.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Arrow-head specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        A grob with ``_grid_class="beziergrob"`` (like R, a plain grob:
        the X-spline content is generated at draw time).
    """
    ux = _ensure_unit(x, default_units)
    uy = _ensure_unit(y, default_units)

    id_arr = None if id is None else np.asarray(id, dtype=np.int64)
    id_lengths_arr = (
        None if id_lengths is None else np.asarray(id_lengths, dtype=np.int64)
    )

    return _BezierGrob(
        name=name,
        gp=gp,
        vp=vp,
        _grid_class="beziergrob",
        x=ux,
        y=uy,
        id=id_arr,
        id_lengths=id_lengths_arr,
        arrow=arrow,
    )


def grid_bezier(
    x: Any = (0, 0.5, 1, 0.5),
    y: Any = (0.5, 1, 0.5, 0),
    id: Optional[Any] = None,
    id_lengths: Optional[Any] = None,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> Grob:
    """Create and optionally draw a *bezier* grob.

    Parameters
    ----------
    x, y : Unit or numeric
        Control-point coordinates; exactly 4 per curve.
    id : array-like of int or None
        Curve label per control point.  Mutually exclusive with
        ``id_lengths``.
    id_lengths : array-like of int or None
        Run-length encoding of ``id``.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Arrow specification.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True`` (default), draw immediately (and record).
    vp : viewport or None
        Optional viewport.

    Returns
    -------
    Grob
        The bezier grob.
    """
    grob = bezier_grob(
        x=x, y=y, id=id, id_lengths=id_lengths,
        default_units=default_units,
        arrow=arrow, name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


def bezier_points(
    x: Grob,
) -> Union[Dict[str, NDArray[np.float64]], List[Dict[str, NDArray[np.float64]]]]:
    """Extract the evaluated curve points from a bezier grob.

    Port of R ``bezierPoints`` (primitives.R:1022-1027): the bezier grob
    is converted to its X-spline approximation (``splinegrob``) and the
    curve is traced with :func:`xspline_points`.  Note that, exactly as
    in R, the returned points come from the X-spline approximation of
    the Bezier — not from evaluating the true Bezier polynomial — and
    their number depends on the device size.

    Parameters
    ----------
    x : Grob
        A bezier grob (``_grid_class="beziergrob"``).

    Returns
    -------
    dict or list of dict
        For a single curve, a dict with ``"x"`` / ``"y"`` ndarrays in
        INCHES; for multiple curves, a list of such dicts (mirroring R).

    Raises
    ------
    TypeError
        If *x* is not a bezier grob.
    """
    if not isinstance(x, Grob) or getattr(x, "_grid_class", None) != "beziergrob":
        raise TypeError("'x' must be a beziergrob grob")

    sg = _splinegrob(x)
    # splinegrob() conversion happens in the caller's context; enforce
    # the bezier's vp for the trace exactly like R (primitives.R:1024-1026)
    sg.vp = x.vp
    return xspline_points(sg)
