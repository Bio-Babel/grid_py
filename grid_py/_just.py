"""Justification utilities for grid_py.

This module ports the justification logic from R's *grid* package
(``src/library/grid/R/just.R``) to Python.  Justification values
control where a graphical object is placed relative to its anchor
point.

Numeric convention (matching R):

* 0 -- left / bottom
* 0.5 -- centre
* 1 -- right / top
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "valid_just",
    "resolve_hjust",
    "resolve_vjust",
    "resolve_raster_size",
]

# ------------------------------------------------------------------ #
# Internal look-up tables                                            #
# ------------------------------------------------------------------ #

# The order matches the R enum in lattice.h:
#   "left"=0, "right"=1, "bottom"=2, "top"=3, "centre"=4, "center"=5
_JUST_STRINGS: Tuple[str, ...] = (
    "left",
    "right",
    "bottom",
    "top",
    "centre",
    "center",
)

# Mapping from enum index to numeric justification
_HJUST_MAP = {0: 0.0, 1: 1.0, 4: 0.5, 5: 0.5}  # left  # right  # centre  # center
_VJUST_MAP = {2: 0.0, 3: 1.0, 4: 0.5, 5: 0.5}  # bottom  # top  # centre  # center

# When a single string is provided, R expands it to (hjust, vjust) as:
#   left   -> (left,   centre)
#   right  -> (right,  centre)
#   bottom -> (centre, bottom)
#   top    -> (centre, top)
#   centre -> (centre, centre)
#   center -> (centre, centre)
_SINGLE_EXPAND = {
    0: (0, 4),  # left   -> (left, centre)
    1: (1, 4),  # right  -> (right, centre)
    2: (4, 2),  # bottom -> (centre, bottom)
    3: (4, 3),  # top    -> (centre, top)
    4: (4, 4),  # centre -> (centre, centre)
    5: (4, 4),  # center -> (centre, centre)
}

# Valid enum indices for each axis
_VALID_HJUST_INDICES = {0, 1, 4, 5}  # left, right, centre, center
_VALID_VJUST_INDICES = {2, 3, 4, 5}  # bottom, top, centre, center

# Convenient string-to-float shortcuts
_STR_TO_HJUST = {"left": 0.0, "right": 1.0, "centre": 0.5, "center": 0.5}
_STR_TO_VJUST = {"bottom": 0.0, "top": 1.0, "centre": 0.5, "center": 0.5}

# Type aliases
JustSpec = Union[str, float, int, Sequence[Union[str, float, int]]]


# ------------------------------------------------------------------ #
# Internal helpers                                                   #
# ------------------------------------------------------------------ #


def _match_just_string(s: str) -> int:
    """Return the enum index for a justification string, or raise."""
    try:
        return _JUST_STRINGS.index(s)
    except ValueError:
        raise ValueError(
            f"invalid justification string {s!r}; "
            f"must be one of {_JUST_STRINGS}"
        ) from None


def _valid_charjust(just: Sequence[str]) -> Tuple[float, float]:
    """Validate and convert character justification (mirrors ``valid.charjust`` in R).

    Parameters
    ----------
    just : sequence of str
        One or two justification strings.

    Returns
    -------
    tuple of float
        ``(hjust, vjust)`` each in [0, 1].
    """
    n = len(just)
    if n == 0:
        return (0.5, 0.5)

    if n == 1:
        idx = _match_just_string(just[0])
        h_idx, v_idx = _SINGLE_EXPAND[idx]
    else:
        h_idx = _match_just_string(just[0])
        if h_idx not in _VALID_HJUST_INDICES:
            raise ValueError(
                f"invalid horizontal justification {just[0]!r}; "
                "must be 'left', 'right', 'centre', or 'center'"
            )
        v_idx = _match_just_string(just[1])
        if v_idx not in _VALID_VJUST_INDICES:
            raise ValueError(
                f"invalid vertical justification {just[1]!r}; "
                "must be 'bottom', 'top', 'centre', or 'center'"
            )

    hjust = _HJUST_MAP.get(h_idx)
    vjust = _VJUST_MAP.get(v_idx)
    if hjust is None or vjust is None:
        raise ValueError("invalid justification")
    return (hjust, vjust)


def _valid_numjust(
    just: Sequence[Union[int, float]],
) -> Tuple[float, float]:
    """Validate and convert numeric justification (mirrors ``valid.numjust`` in R).

    Parameters
    ----------
    just : sequence of int or float
        Zero, one, or two numeric justification values.

    Returns
    -------
    tuple of float
        ``(hjust, vjust)``.
    """
    n = len(just)
    if n == 0:
        return (0.5, 0.5)
    if n == 1:
        return (float(just[0]), 0.5)
    return (float(just[0]), float(just[1]))


# ------------------------------------------------------------------ #
# Public API                                                         #
# ------------------------------------------------------------------ #


def valid_just(just: JustSpec) -> Tuple[float, float]:
    """Validate and normalise a justification specification.

    This mirrors R's ``valid.just`` function.  Justification may be
    given as:

    * a single string (``"left"``, ``"right"``, ``"bottom"``, ``"top"``,
      ``"centre"``, ``"center"``),
    * a two-element sequence ``[hjust, vjust]`` where each element is a
      string *or* a number, or
    * a single number (treated as *hjust* with *vjust* defaulting to 0.5).

    Parameters
    ----------
    just : str, float, int, or sequence thereof
        Justification specification.

    Returns
    -------
    tuple of float
        ``(hjust, vjust)`` where 0 is left/bottom, 0.5 is centre, and
        1 is right/top.

    Raises
    ------
    ValueError
        If *just* contains an unrecognised string or an invalid
        combination of horizontal/vertical strings.

    Examples
    --------
    >>> valid_just("centre")
    (0.5, 0.5)
    >>> valid_just("left")
    (0.0, 0.5)
    >>> valid_just(["right", "top"])
    (1.0, 1.0)
    >>> valid_just(0.25)
    (0.25, 0.5)
    >>> valid_just([0.1, 0.9])
    (0.1, 0.9)
    """
    # Scalar string
    if isinstance(just, str):
        return _valid_charjust([just])

    # Scalar numeric
    if isinstance(just, (int, float, np.integer, np.floating)):
        return (float(just), 0.5)

    # Sequence -- coerce to list for uniform handling
    just_list: List[Union[str, float, int]] = list(just)
    if len(just_list) == 0:
        return (0.5, 0.5)

    # All-string path
    if all(isinstance(j, str) for j in just_list):
        return _valid_charjust(just_list)  # type: ignore[arg-type]

    # All-numeric path
    if all(isinstance(j, (int, float, np.integer, np.floating)) for j in just_list):
        return _valid_numjust(just_list)  # type: ignore[arg-type]

    # Mixed: try to convert everything to float (matches R's as.numeric)
    try:
        nums = [float(j) for j in just_list]
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid justification specification: {just!r}"
        ) from None
    return _valid_numjust(nums)


def resolve_hjust(
    just: JustSpec,
    hjust: Optional[float] = None,
) -> float:
    """Resolve horizontal justification to a numeric value.

    If *hjust* is provided it is returned directly; otherwise the
    horizontal component is extracted from *just* via `valid_just`.
    This mirrors R's ``resolveHJust``.

    Parameters
    ----------
    just : str, float, int, or sequence thereof
        General justification specification (see `valid_just`).
    hjust : float, optional
        Explicit horizontal justification override.  When not *None*
        this value is returned as-is.

    Returns
    -------
    float
        Horizontal justification in [0, 1].

    Examples
    --------
    >>> resolve_hjust("left")
    0.0
    >>> resolve_hjust("centre", hjust=0.3)
    0.3
    """
    if hjust is not None:
        return float(hjust)
    return valid_just(just)[0]


def resolve_vjust(
    just: JustSpec,
    vjust: Optional[float] = None,
) -> float:
    """Resolve vertical justification to a numeric value.

    If *vjust* is provided it is returned directly; otherwise the
    vertical component is extracted from *just* via `valid_just`.
    This mirrors R's ``resolveVJust``.

    Parameters
    ----------
    just : str, float, int, or sequence thereof
        General justification specification (see `valid_just`).
    vjust : float, optional
        Explicit vertical justification override.  When not *None*
        this value is returned as-is.

    Returns
    -------
    float
        Vertical justification in [0, 1].

    Examples
    --------
    >>> resolve_vjust("top")
    1.0
    >>> resolve_vjust("centre", vjust=0.8)
    0.8
    """
    if vjust is not None:
        return float(vjust)
    return valid_just(just)[1]


def resolve_raster_size(
    x: np.ndarray,
    default_size: Tuple[float, float],
    target_size: Tuple[Optional[float], Optional[float]],
) -> Tuple[float, float]:
    """Compute width and height for a raster grob.

    When either the target width or height is *None* the missing
    dimension is derived from the raster's aspect ratio and the
    available viewport size, matching R's ``resolveRasterSize``.

    Parameters
    ----------
    x : numpy.ndarray
        The raster data array.  Its shape ``(nrow, ncol, ...)`` is used
        to compute the aspect ratio (``nrow / ncol``).
    default_size : tuple of float
        ``(viewport_width, viewport_height)`` in inches, used when both
        *width* and *height* are *None*.
    target_size : tuple of float or None
        ``(width, height)`` requested by the user.  Either or both may
        be *None* to indicate "auto".

    Returns
    -------
    tuple of float
        ``(width, height)`` in the same units as *default_size* /
        *target_size*.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((200, 400, 3))
    >>> resolve_raster_size(img, (6.0, 4.0), (None, None))
    (6.0, 3.0)
    """
    nrow, ncol = x.shape[0], x.shape[1]
    raster_ratio = nrow / ncol  # height / width in pixels

    width, height = target_size

    if width is None and height is None:
        vp_width, vp_height = default_size
        vp_ratio = vp_height / vp_width
        if raster_ratio > vp_ratio:
            # raster is taller (relative) than viewport
            height = vp_height
            width = vp_height * ncol / nrow
        else:
            width = vp_width
            height = vp_width * nrow / ncol
    elif width is None:
        # height is known; derive width
        width = height * ncol / nrow
    elif height is None:
        # width is known; derive height
        height = width * nrow / ncol

    return (float(width), float(height))
