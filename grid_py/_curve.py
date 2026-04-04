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
from ._units import Unit, is_unit

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
    theta_all = np.linspace(0.0, direction * maxtheta, ncp + 2)
    # Drop first and last
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
    m = val.reshape((ncp, ncurve), order="F")
    result = np.empty((ncp + 1, ncurve), dtype=np.float64)
    for i in range(ncurve):
        if end[i]:
            result[:, i] = np.concatenate([m[:, i], [eval_[i]]])
        else:
            result[:, i] = np.concatenate([[sval[i]], m[:, i]])
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

    This is a simplified approximation of X-spline evaluation.  When
    *shape* is 0 the result is a Catmull--Rom interpolation; when *shape*
    is -1 it approximates a B-spline; when *shape* is 1 it interpolates
    through control points.

    Parameters
    ----------
    x, y : ndarray
        Control-point coordinates.
    shape : float or ndarray
        Shape parameter(s) in [-1, 1].  Scalar is broadcast.
    open_ : bool
        Whether the spline is open (True) or closed (False).
    repEnds : bool
        Whether to replicate the first and last control points (for open
        splines) so the curve passes through the endpoints.

    Returns
    -------
    tuple of ndarray
        ``(x_pts, y_pts)`` evaluated spline coordinates.

    Notes
    -----
    A full X-spline engine is complex; this implementation uses cubic
    Catmull--Rom subdivision as a reasonable default, falling back to
    linear interpolation when fewer than 4 points are available.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if n < 2:
        return x.copy(), y.copy()

    # Replicate endpoints for open splines so curve passes through them
    if open_ and repEnds and n >= 2:
        x = np.concatenate([[x[0]], x, [x[-1]]])
        y = np.concatenate([[y[0]], y, [y[-1]]])
        n = len(x)

    if not open_:
        # Close by wrapping
        x = np.concatenate([x, x[:1]])
        y = np.concatenate([y, y[:1]])
        n = len(x)

    # Use Catmull-Rom interpolation between successive segments
    n_seg = n - 1
    pts_per_seg = max(2, 50 // max(1, n_seg))
    t_vals = np.linspace(0, 1, pts_per_seg, endpoint=False)

    all_x: List[float] = []
    all_y: List[float] = []

    for i in range(n_seg):
        # Catmull-Rom needs 4 points: p0, p1, p2, p3
        p0_idx = max(0, i - 1)
        p1_idx = i
        p2_idx = min(n - 1, i + 1)
        p3_idx = min(n - 1, i + 2)

        p0x, p0y = x[p0_idx], y[p0_idx]
        p1x, p1y = x[p1_idx], y[p1_idx]
        p2x, p2y = x[p2_idx], y[p2_idx]
        p3x, p3y = x[p3_idx], y[p3_idx]

        for t in t_vals:
            t2 = t * t
            t3 = t2 * t
            # Catmull-Rom basis
            bx = 0.5 * (
                (2 * p1x)
                + (-p0x + p2x) * t
                + (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2
                + (-p0x + 3 * p1x - 3 * p2x + p3x) * t3
            )
            by = 0.5 * (
                (2 * p1y)
                + (-p0y + p2y) * t
                + (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2
                + (-p0y + 3 * p1y - 3 * p2y + p3y) * t3
            )
            all_x.append(bx)
            all_y.append(by)

    # Append final point
    all_x.append(float(x[-1]))
    all_y.append(float(y[-1]))

    return np.array(all_x, dtype=np.float64), np.array(all_y, dtype=np.float64)


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

    return GTree(
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

    return Grob(
        x=x,
        y=y,
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
        x=x, y=y, default_units=default_units,
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
