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
# Module-level display list (shared with _primitives)
# ---------------------------------------------------------------------------

_display_list: List[Grob] = []


def _grid_draw(grob: Grob) -> None:
    """Append *grob* to the module-level display list."""
    _display_list.append(grob)


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
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate an X-spline through the given control points.

    Faithful port of R's ``src/main/xspline.c`` (itself derived from
    XFig 3.2.4, which in turn implements the Blanc & Schlick 1995
    X-spline model verbatim).  The per-point ``shape`` parameter is in
    ``[-1, 1]`` with the standard interpretation:

    - ``shape < 0``: "approximating" (B-spline-like)
    - ``shape = 0``: control point is a sharp corner
    - ``shape > 0``: "interpolating" (curve passes through)

    Blending is done with the three polynomial kernels defined in the
    Blanc-Schlick paper — ``f_blend`` (quintic), ``g_blend`` (quintic),
    and ``h_blend`` (quartic).  These are **exact**, not a Catmull-Rom /
    B-spline / linear approximation.

    Parameters
    ----------
    x, y : ndarray
        Control-point coordinates (inches, device, or any linear unit).
    shape : float or ndarray
        Per-control-point shape parameter(s) in ``[-1, 1]``.  Scalar is
        broadcast to all points.
    open_ : bool
        Open (True) or closed (False) spline.
    repEnds : bool
        For open splines, replicate the first and last control points so
        the curve passes through the endpoints.  Matches R's ``repEnds``.

    Returns
    -------
    tuple of ndarray
        ``(x_pts, y_pts)`` evaluated spline coordinates.

    References
    ----------
    Blanc, C. and Schlick, C. (1995).  X-splines: A spline model designed
    for the end-user.  *Proceedings of SIGGRAPH 95*, pp. 377-386.

    R implementation: ``src/main/xspline.c``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if n < 2:
        return x.copy(), y.copy()

    if np.isscalar(shape):
        s = np.full(n, float(shape), dtype=np.float64)
    else:
        s = np.asarray(shape, dtype=np.float64)
        if len(s) < n:
            s = np.resize(s, n)
    s = np.clip(s, -1.0, 1.0)

    # R forces the first and last control points' shape to 0 for OPEN
    # xsplines (primitives.R:795-803 ``validDetails.xspline``).  This
    # makes the curve pass exactly through the endpoints: at shape=0,
    # ``positive_s1/s2_influence`` at ``t=0`` reduces to ``A1=1`` and
    # all other weights 0, so the blend resolves to the (duplicated)
    # first control point.  Without this, open splines with nonzero
    # end shapes do not land on the endpoints.
    if open_ and n >= 1:
        s = s.copy()
        s[0] = 0.0
        s[-1] = 0.0

    # R's precision parameter (LOW_PRECISION=1.0 is the default for
    # ``GEXspline``).  Step size is derived adaptively from segment
    # geometry (see ``_xsp_step``).
    precision = 1.0

    if open_:
        out_x, out_y = _xsp_compute_open(x, y, s, repEnds, precision)
    else:
        out_x, out_y = _xsp_compute_closed(x, y, s, precision)

    return out_x, out_y


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
              s1: float, s2: float, precision: float) -> float:
    """Port of R's ``step_computing`` — adaptive step based on curve extent.

    The step is chosen so the polyline sampling resolution matches the
    physical distance from segment origin to extremity, augmented by a
    curvature term (cosine of the origin-mid-extremity angle).
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
    dist = math.sqrt(xlen * xlen + ylen * ylen)

    # R (via XFig) does all step math in 1200 ppi units.  Our coordinates
    # are in whatever linear unit the caller passed (usually inches), so
    # scale by 1200 here to reproduce R's sampling density.  Downstream
    # output coordinates are unaffected — only the step count changes.
    dist = dist * 1200.0

    # R's diagonal clamp (xspline.c:312-325) avoids runaway sampling when
    # control points are far outside the device; we approximate it with a
    # fixed cap equivalent to ~1.7 inches × 1200 diagonal.
    if dist > 2000.0:
        dist = 2000.0

    n_steps = math.sqrt(dist) / 2.0
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

    Emits points into ``out_x`` / ``out_y`` with de-duplication against the
    last emitted point (matches R's ``add_point`` which skips repeats).
    """
    t = 0.0
    while t < 1.0:
        A = _xsp_weights(k, t, s1, s2)
        bx, by = _xsp_point(A, px, py)
        if not out_x or out_x[-1] != bx or out_y[-1] != by:
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
    if not out_x or out_x[-1] != bx or out_y[-1] != by:
        out_x.append(bx)
        out_y.append(by)


# -- Open / closed drivers (xspline.c:455-547) ------------------------------

def _xsp_compute_open(
    x: NDArray[np.float64], y: NDArray[np.float64], s: NDArray[np.float64],
    repEnds: bool, precision: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
            step = _xsp_step(k, px, py, ps[1], ps[2], precision)
            _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                         out_x, out_y)
            if k + 3 >= n:
                break
            # R's ``NEXT_CONTROL_POINTS(K, N)`` macro (xspline.c:438-442):
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
        step = _xsp_step(k, px, py, ps[1], ps[2], precision)
        _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                     out_x, out_y)

        # Final point: px[3], py[3] (xspline.c:510)
        if not out_x or out_x[-1] != px[3] or out_y[-1] != py[3]:
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
            step = _xsp_step(k, px, py, ps[1], ps[2], precision)
            _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                         out_x, out_y)
        # Last segment's t=1 evaluation (xspline.c:516)
        k = n - 4
        px = [x[k], x[k + 1], x[k + 2], x[k + 3]]
        py = [y[k], y[k + 1], y[k + 2], y[k + 3]]
        ps = [s[k], s[k + 1], s[k + 2], s[k + 3]]
        _xsp_last_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                          out_x, out_y)

    # R trims leading / trailing duplicate points (grid.c:2494-2504).
    # We emulate that: remove consecutive duplicates at start only
    # (trailing dedup already happens in _xsp_segment's emit).
    while len(out_x) > 1 and out_x[0] == out_x[1] and out_y[0] == out_y[1]:
        out_x.pop(0)
        out_y.pop(0)

    return (np.asarray(out_x, dtype=np.float64),
            np.asarray(out_y, dtype=np.float64))


def _xsp_compute_closed(
    x: NDArray[np.float64], y: NDArray[np.float64], s: NDArray[np.float64],
    precision: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(x)
    if n < 3:
        raise ValueError("there must be at least three control points")

    out_x: List[float] = []
    out_y: List[float] = []

    # INIT_CONTROL_POINTS: (n-1, 0, 1, 2) mod n
    idx = [(n - 1) % n, 0 % n, 1 % n, 2 % n]
    px = [x[i] for i in idx]
    py = [y[i] for i in idx]
    ps = [s[i] for i in idx]

    for k in range(n):
        step = _xsp_step(k, px, py, ps[1], ps[2], precision)
        _xsp_segment(step, k, tuple(px), tuple(py), ps[1], ps[2],
                     out_x, out_y)
        # NEXT_CONTROL_POINTS(K, N): (K..K+3) mod n
        idx = [(k + 1) % n, (k + 2) % n, (k + 3) % n, (k + 4) % n]
        px = [x[i] for i in idx]
        py = [y[i] for i in idx]
        ps = [s[i] for i in idx]

    return (np.asarray(out_x, dtype=np.float64),
            np.asarray(out_y, dtype=np.float64))


# ===================================================================== #
#  Internal: Bezier point calculation (de Casteljau)                     #
# ===================================================================== #


def _calc_bezier_points(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n: int = 50,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate a Bezier curve using the de Casteljau algorithm.

    Parameters
    ----------
    x, y : ndarray
        Control-point coordinates.  Typically 4 points for a cubic
        Bezier, but any number >= 2 is accepted.
    n : int
        Number of evaluation points along the curve.

    Returns
    -------
    tuple of ndarray
        ``(x_pts, y_pts)`` evaluated Bezier curve coordinates.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    npts = len(x)

    if npts < 2:
        return x.copy(), y.copy()

    t_vals = np.linspace(0.0, 1.0, n)
    out_x = np.empty(n, dtype=np.float64)
    out_y = np.empty(n, dtype=np.float64)

    for k, t in enumerate(t_vals):
        # de Casteljau
        bx = x.copy()
        by = y.copy()
        for r in range(1, npts):
            bx[:npts - r] = (1 - t) * bx[:npts - r] + t * bx[1:npts - r + 1]
            by[:npts - r] = (1 - t) * by[:npts - r] + t * by[1:npts - r + 1]
        out_x[k] = bx[0]
        out_y[k] = by[0]

    return out_x, out_y


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


def xspline_grob(
    x: Optional[Any] = None,
    y: Optional[Any] = None,
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
    x, y : Unit, numeric, sequence, or None
        Control-point coordinates.  Defaults to ``Unit([0, 1], "npc")``
        when ``None``.
    id : array-like of int or None
        Group label for each control point.  Points sharing an ``id`` are
        rendered as one X-spline; the grob therefore renders one spline
        per unique ``id`` value.  Mirrors R ``xsplineGrob(id=...)``.
        Mutually meaningful with ``id_lengths``: pass at most one.
    id_lengths : array-like of int or None
        Run-length encoding of ``id``: the n-th entry is the number of
        consecutive control points belonging to spline n.  Mirrors R's
        ``xsplineGrob(id.lengths=...)``.
    default_units : str
        Unit type for bare numerics.
    shape : float or sequence of float
        Shape parameter(s) in [-1, 1].  A scalar is broadcast to all
        control points.
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
    if x is None:
        x = Unit([0, 1], "npc")
    else:
        x = _ensure_unit(x, default_units)
    if y is None:
        y = Unit([0, 1], "npc")
    else:
        y = _ensure_unit(y, default_units)

    # Normalise shape to a numpy array
    shape_arr = np.atleast_1d(np.asarray(shape, dtype=np.float64))
    if np.any((shape_arr < -1) | (shape_arr > 1)):
        raise ValueError("all 'shape' values must be between -1 and 1")

    id_arr = None if id is None else np.asarray(id, dtype=np.int64)
    id_lengths_arr = (
        None if id_lengths is None else np.asarray(id_lengths, dtype=np.int64)
    )

    return Grob(
        x=x,
        y=y,
        id=id_arr,
        id_lengths=id_lengths_arr,
        shape=shape_arr,
        open_=bool(open_),
        arrow=arrow,
        repEnds=bool(repEnds),
        name=name,
        gp=gp,
        vp=vp,
        _grid_class="xspline",
    )


def grid_xspline(
    x: Optional[Any] = None,
    y: Optional[Any] = None,
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


def xspline_points(x: Grob) -> Dict[str, NDArray[np.float64]]:
    """Extract evaluated X-spline points from an xspline grob.

    Parameters
    ----------
    x : Grob
        An xspline grob (``_grid_class="xspline"``).

    Returns
    -------
    dict
        Dictionary with keys ``"x"`` and ``"y"``, each an ndarray of
        evaluated spline coordinates.

    Raises
    ------
    TypeError
        If *x* is not an xspline grob.
    """
    if not isinstance(x, Grob) or getattr(x, "_grid_class", None) != "xspline":
        raise TypeError("'x' must be an xspline grob")

    # Extract numeric values from Unit objects
    ctrl_x = np.asarray(x.x.values if hasattr(x.x, "values") else x.x, dtype=np.float64)
    ctrl_y = np.asarray(x.y.values if hasattr(x.y, "values") else x.y, dtype=np.float64)
    shape = x.shape if hasattr(x, "shape") else 0.0
    open_ = getattr(x, "open_", True)
    repEnds = getattr(x, "repEnds", True)

    px, py = _calc_xspline_points(ctrl_x, ctrl_y, shape, open_, repEnds)
    return {"x": px, "y": py}


# ===================================================================== #
#  bezierGrob / grid.bezier                                              #
# ===================================================================== #


def bezier_grob(
    x: Any,
    y: Any,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GTree:
    """Create a *bezier* grob (GTree).

    A Bezier grob draws a cubic (or higher-order) Bezier curve through
    the given control points.

    Parameters
    ----------
    x, y : Unit or numeric
        Control-point coordinates.  For a cubic Bezier, supply exactly 4
        points; the curve interpolates the first and last and is
        attracted toward the middle two.
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
    GTree
        A grob tree with ``_grid_class="beziergrob"``.
    """
    ux = _ensure_unit(x, default_units)
    uy = _ensure_unit(y, default_units)

    return GTree(
        name=name,
        gp=gp,
        vp=vp,
        _grid_class="beziergrob",
        x=ux,
        y=uy,
        arrow=arrow,
    )


def grid_bezier(
    x: Any,
    y: Any,
    default_units: str = "npc",
    arrow: Optional[Arrow] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> GTree:
    """Create and optionally draw a *bezier* grob.

    Parameters
    ----------
    x, y : Unit or numeric
        Control-point coordinates.
    default_units : str
        Unit type for bare numerics.
    arrow : Arrow or None
        Arrow specification.
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
    GTree
        The bezier grob.
    """
    grob = bezier_grob(
        x=x, y=y, default_units=default_units,
        arrow=arrow, name=name, gp=gp, vp=vp,
    )
    if draw:
        _grid_draw(grob)
    return grob


def bezier_points(x: Grob, n: int = 50) -> Dict[str, NDArray[np.float64]]:
    """Extract evaluated Bezier curve points from a bezier grob.

    Parameters
    ----------
    x : Grob
        A bezier grob (``_grid_class="beziergrob"``).
    n : int
        Number of evaluation points (default 50).

    Returns
    -------
    dict
        Dictionary with keys ``"x"`` and ``"y"``, each an ndarray of
        evaluated Bezier coordinates.

    Raises
    ------
    TypeError
        If *x* is not a bezier grob.
    """
    if not isinstance(x, (Grob, GTree)) or getattr(x, "_grid_class", None) != "beziergrob":
        raise TypeError("'x' must be a beziergrob grob")

    ctrl_x = np.asarray(x.x.values if hasattr(x.x, "values") else x.x, dtype=np.float64)
    ctrl_y = np.asarray(x.y.values if hasattr(x.y, "values") else x.y, dtype=np.float64)

    px, py = _calc_bezier_points(ctrl_x, ctrl_y, n=n)
    return {"x": px, "y": py}
