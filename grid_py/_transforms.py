"""Affine transformation matrices for grid_py (port of R's grid group transforms).

This module provides functions that construct 3x3 affine transformation
matrices used by the group/define/use grob system.  Each function returns a
:class:`numpy.ndarray` of shape ``(3, 3)`` and dtype ``float64``.

The matrices follow the **row-vector** convention used by R's *grid* package:
a point ``[x, y, 1]`` is transformed via ``point @ matrix``.  Translation
terms therefore live in the bottom row (indices ``[2, 0]`` and ``[2, 1]``).

Group transforms
----------------
Low-level building blocks that mirror R's ``groupTranslate``,
``groupRotate``, ``groupScale``, ``groupShear``, and ``groupFlip``.

Definition transforms
---------------------
Transforms applied when defining a group (``defineGrob``).

Use transforms
--------------
Transforms applied when reusing a group (``useGrob``).

Viewport transforms
-------------------
Combined transforms that map from one viewport to another.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    # Group (primitive) transforms
    "group_translate",
    "group_rotate",
    "group_scale",
    "group_shear",
    "group_flip",
    # Definition transforms
    "defn_translate",
    "defn_rotate",
    "defn_scale",
    # Use transforms
    "use_translate",
    "use_rotate",
    "use_scale",
    # Viewport transforms
    "viewport_translate",
    "viewport_rotate",
    "viewport_scale",
    "viewport_transform",
]

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Matrix3x3 = NDArray[np.float64]

# ============================================================================
# Group (primitive) transforms
# ============================================================================


def group_translate(dx: float = 0, dy: float = 0) -> Matrix3x3:
    """Return a 3x3 translation matrix.

    Parameters
    ----------
    dx : float, optional
        Horizontal translation (default ``0``).
    dy : float, optional
        Vertical translation (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine translation matrix.

    Examples
    --------
    >>> group_translate(10, 20)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [10., 20.,  1.]])
    """
    mat = np.eye(3, dtype=np.float64)
    mat[2, 0] = dx
    mat[2, 1] = dy
    return mat


def group_rotate(angle: float = 0, device: bool = True) -> Matrix3x3:
    """Return a 3x3 rotation matrix.

    Parameters
    ----------
    angle : float, optional
        Rotation angle in **degrees** (default ``0``).  Positive values
        rotate counter-clockwise in the standard mathematical sense.
    device : bool, optional
        If ``True`` (default) the rotation follows device conventions
        (identical to R's ``groupRotate`` with ``device=TRUE`` when the
        device origin is at the bottom-left).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine rotation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(group_rotate(90) @ np.array([1, 0, 1]),
    ...             np.array([0, 1, 1]), atol=1e-15)
    True
    """
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    mat = np.eye(3, dtype=np.float64)
    mat[0, 0] = cos_t
    mat[0, 1] = sin_t
    mat[1, 0] = -sin_t
    mat[1, 1] = cos_t
    return mat


def group_scale(sx: float = 1, sy: float = 1) -> Matrix3x3:
    """Return a 3x3 scaling matrix.

    Parameters
    ----------
    sx : float, optional
        Horizontal scale factor (default ``1``).
    sy : float, optional
        Vertical scale factor (default ``1``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine scaling matrix.

    Examples
    --------
    >>> group_scale(2, 3)
    array([[2., 0., 0.],
           [0., 3., 0.],
           [0., 0., 1.]])
    """
    mat = np.eye(3, dtype=np.float64)
    mat[0, 0] = sx
    mat[1, 1] = sy
    return mat


def group_shear(sx: float = 0, sy: float = 0) -> Matrix3x3:
    """Return a 3x3 shear matrix.

    Parameters
    ----------
    sx : float, optional
        Shear factor along the *x*-axis (default ``0``).  This is placed
        at matrix position ``[1, 0]``, matching R's ``groupShear``.
    sy : float, optional
        Shear factor along the *y*-axis (default ``0``).  This is placed
        at matrix position ``[0, 1]``, matching R's ``groupShear``.

    Returns
    -------
    numpy.ndarray
        A 3x3 affine shear matrix.

    Examples
    --------
    >>> group_shear(0.5, 0)
    array([[1. , 0. , 0. ],
           [0.5, 1. , 0. ],
           [0. , 0. , 1. ]])
    """
    mat = np.eye(3, dtype=np.float64)
    mat[0, 1] = sy
    mat[1, 0] = sx
    return mat


def group_flip(flipX: bool = False, flipY: bool = False) -> Matrix3x3:
    """Return a 3x3 flip (reflection) matrix.

    Parameters
    ----------
    flipX : bool, optional
        If ``True``, negate the *x*-component (reflect across the *y*-axis).
    flipY : bool, optional
        If ``True``, negate the *y*-component (reflect across the *x*-axis).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine flip matrix.

    Examples
    --------
    >>> group_flip(True, False)
    array([[-1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """
    mat = np.eye(3, dtype=np.float64)
    if flipX:
        mat[0, 0] = -1.0
    if flipY:
        mat[1, 1] = -1.0
    return mat


# ============================================================================
# Definition transforms  (for defineGrob)
# ============================================================================


def defn_translate(dx: float = 0, dy: float = 0) -> Matrix3x3:
    """Return a 3x3 translation matrix for a group definition.

    In R this retrieves the definition viewport location from the group
    object.  Here we accept the displacements directly.

    Parameters
    ----------
    dx : float, optional
        Horizontal displacement (default ``0``).
    dy : float, optional
        Vertical displacement (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine translation matrix.
    """
    return group_translate(dx, dy)


def defn_rotate(angle: float = 0) -> Matrix3x3:
    """Return a 3x3 rotation matrix for a group definition.

    Parameters
    ----------
    angle : float, optional
        Rotation angle in degrees (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine rotation matrix.
    """
    return group_rotate(angle, device=True)


def defn_scale(sx: float = 1, sy: float = 1) -> Matrix3x3:
    """Return a 3x3 scaling matrix for a group definition.

    Parameters
    ----------
    sx : float, optional
        Horizontal scale factor (default ``1``).
    sy : float, optional
        Vertical scale factor (default ``1``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine scaling matrix.
    """
    return group_scale(sx, sy)


# ============================================================================
# Use transforms  (for useGrob)
# ============================================================================


def use_translate(dx: float = 0, dy: float = 0) -> Matrix3x3:
    """Return a 3x3 translation matrix for a group use.

    In R this retrieves the current viewport location.  Here we accept
    the displacements directly so the matrix can be composed offline.

    Parameters
    ----------
    dx : float, optional
        Horizontal displacement (default ``0``).
    dy : float, optional
        Vertical displacement (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine translation matrix.
    """
    return group_translate(dx, dy)


def use_rotate(angle: float = 0) -> Matrix3x3:
    """Return a 3x3 rotation matrix for a group use.

    Parameters
    ----------
    angle : float, optional
        Rotation angle in degrees (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine rotation matrix.
    """
    return group_rotate(angle, device=True)


def use_scale(sx: float = 1, sy: float = 1) -> Matrix3x3:
    """Return a 3x3 scaling matrix for a group use.

    Parameters
    ----------
    sx : float, optional
        Horizontal scale factor (default ``1``).
    sy : float, optional
        Vertical scale factor (default ``1``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine scaling matrix.
    """
    return group_scale(sx, sy)


# ============================================================================
# Viewport transforms
# ============================================================================


def viewport_translate(dx: float = 0, dy: float = 0) -> Matrix3x3:
    """Return a 3x3 translation suitable for a viewport transform.

    This is the combined inverse-definition-translate followed by the
    use-translate, collapsed into a single translation of the difference.

    Parameters
    ----------
    dx : float, optional
        Net horizontal displacement (default ``0``).
    dy : float, optional
        Net vertical displacement (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine translation matrix.
    """
    return group_translate(dx, dy)


def viewport_rotate(angle: float = 0) -> Matrix3x3:
    """Return a 3x3 rotation suitable for a viewport transform.

    Parameters
    ----------
    angle : float, optional
        Net rotation angle in degrees (default ``0``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine rotation matrix.
    """
    return group_rotate(angle, device=True)


def viewport_scale(sx: float = 1, sy: float = 1) -> Matrix3x3:
    """Return a 3x3 scaling suitable for a viewport transform.

    Parameters
    ----------
    sx : float, optional
        Horizontal scale factor (default ``1``).
    sy : float, optional
        Vertical scale factor (default ``1``).

    Returns
    -------
    numpy.ndarray
        A 3x3 affine scaling matrix.
    """
    return group_scale(sx, sy)


def viewport_transform(
    dx: float = 0,
    dy: float = 0,
    rotation: float = 0,
    sx: float = 1,
    sy: float = 1,
) -> Matrix3x3:
    """Return a combined 3x3 viewport transform matrix.

    The resulting matrix applies translation, rotation, and scaling in a
    single affine transform, following the composition order used by R's
    ``viewportTransform``:

        translate(dx, dy) @ rotate(rotation) @ scale(sx, sy)

    Because we use the row-vector convention (``point @ matrix``), the
    operations are applied left-to-right: first translate, then rotate,
    then scale.

    Parameters
    ----------
    dx : float, optional
        Horizontal translation (default ``0``).
    dy : float, optional
        Vertical translation (default ``0``).
    rotation : float, optional
        Rotation angle in degrees (default ``0``).
    sx : float, optional
        Horizontal scale factor (default ``1``).
    sy : float, optional
        Vertical scale factor (default ``1``).

    Returns
    -------
    numpy.ndarray
        A 3x3 combined affine transform matrix.

    Examples
    --------
    >>> import numpy as np
    >>> T = viewport_transform(dx=10, dy=20, rotation=0, sx=2, sy=3)
    >>> np.allclose(T, group_translate(10, 20) @ group_scale(2, 3))
    True
    """
    T = group_translate(dx, dy)
    R = group_rotate(rotation, device=True)
    S = group_scale(sx, sy)
    return T @ R @ S
