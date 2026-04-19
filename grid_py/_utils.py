"""
Utility helpers for the grid_py package.

This module provides internal helpers and public utility functions ported
from the R *grid* package's ``util.R`` and related sources.
"""

from __future__ import annotations

import math
from typing import Any, Generator, List, Optional, Sequence, Union

import numpy as np

__all__: list[str] = [
    "depth",
    "explode",
    "grid_pretty",
    "n2mfrow",
]

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_grid_path_sep: str = "::"
"""Path separator used by viewport paths and grob paths (mirrors ``.grid.pathSep`` in R)."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_name_counter(prefix: str = "GRID") -> Generator[str, None, None]:
    """Yield sequential auto-generated names.

    Parameters
    ----------
    prefix : str, optional
        Prefix for each name.  Default is ``"GRID"``.

    Yields
    ------
    str
        Names of the form ``"GRID.1"``, ``"GRID.2"``, ...
    """
    n = 0
    while True:
        n += 1
        yield f"{prefix}.{n}"


def _recycle(x: Union[np.ndarray, Sequence[Any]], length: int) -> np.ndarray:
    """Recycle *x* to the requested *length* (R-style recycling).

    Parameters
    ----------
    x : array_like
        Input values.
    length : int
        Desired output length.  Must be >= 0.

    Returns
    -------
    numpy.ndarray
        Array of *length* elements obtained by repeating *x* cyclically.

    Examples
    --------
    >>> _recycle([1, 2, 3], 7)
    array([1, 2, 3, 1, 2, 3, 1])
    """
    arr = np.asarray(x).ravel()
    if len(arr) == 0:
        return np.empty(0, dtype=arr.dtype)
    if length == 0:
        return np.empty(0, dtype=arr.dtype)
    # np.resize already recycles, but it silently returns empty for 0-length
    return np.resize(arr, length)


def _is_finite(x: Any) -> Union[bool, np.ndarray]:
    """Check for finite values, treating ``None`` as non-finite.

    Parameters
    ----------
    x : scalar or array_like
        Value(s) to test.  ``None`` is treated as non-finite.

    Returns
    -------
    bool or numpy.ndarray of bool
        ``True`` where values are finite, ``False`` otherwise.
    """
    if x is None:
        return False
    arr = np.asarray(x)
    # For non-numeric dtypes every element is "not finite"
    if not np.issubdtype(arr.dtype, np.number):
        return np.zeros(arr.shape, dtype=bool) if arr.ndim else False
    return np.isfinite(arr)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def depth(x: Any) -> int:
    """Return the depth of a path-like object.

    For plain strings the depth equals the number of components separated
    by ``"::"``.  For path objects (dicts with an ``"n"`` key) the stored
    depth is returned directly.

    Parameters
    ----------
    x : str or path-like
        A viewport-path or grob-path object.  Strings are split on the
        ``"::"`` separator to determine their depth.

    Returns
    -------
    int
        Number of components in the path.

    Examples
    --------
    >>> depth("A::B::C")
    3
    >>> depth("ROOT")
    1
    """
    # Path-like dict (vpPath / gPath style)
    if isinstance(x, dict) and "n" in x:
        return int(x["n"])

    # Plain string
    if isinstance(x, str):
        parts = x.split(_grid_path_sep)
        return len(parts)

    # Objects that expose a `.depth()` method or `.n` attribute
    if hasattr(x, "depth"):
        return x.depth()
    if hasattr(x, "n"):
        return int(x.n)

    raise TypeError(f"Cannot compute depth of {type(x).__name__!r}")


def explode(x: Any) -> List[str]:
    """Split a path into its individual components.

    This is the Python equivalent of the S3 generic ``explode()`` in R's
    grid package.

    Parameters
    ----------
    x : str or path-like
        A viewport-path or grob-path object.  Strings are split on the
        ``"::"`` separator.

    Returns
    -------
    list of str
        The individual path components.

    Examples
    --------
    >>> explode("A::B::C")
    ['A', 'B', 'C']
    >>> explode("ROOT")
    ['ROOT']
    """
    # Path-like dict (vpPath / gPath style)
    if isinstance(x, dict):
        if x.get("n", 0) == 1:
            return [x["name"]]
        parts = explode(x["path"])
        parts.append(x["name"])
        return parts

    if isinstance(x, str):
        return x.split(_grid_path_sep)

    # Objects that expose an `explode()` method
    if hasattr(x, "explode"):
        return list(x.explode())

    raise TypeError(f"Cannot explode {type(x).__name__!r}")


def grid_pretty(range_val: Sequence[float], n: int = 5) -> np.ndarray:
    """Return *pretty* tick mark positions for a numeric range.

    This mirrors R's ``grid.pretty(range, n)`` which internally calls
    ``pretty()``.  The algorithm chooses "nice" numbers that cover the
    given range.

    Parameters
    ----------
    range_val : sequence of float
        A two-element sequence ``[lo, hi]`` giving the data range.
    n : int, optional
        Target number of intervals (the result may have slightly more or
        fewer tick marks).  Default is 5.

    Returns
    -------
    numpy.ndarray
        Array of pretty tick positions.

    Raises
    ------
    ValueError
        If *range_val* is not numeric or does not have two elements.

    Examples
    --------
    >>> grid_pretty([0.0, 1.0])
    array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    """
    rng = np.asarray(range_val, dtype=float)
    if rng.size != 2:
        raise ValueError("'range_val' must have exactly two elements")
    if not np.all(np.isfinite(rng)):
        raise ValueError("'range_val' must be finite numeric")

    lo, hi = float(rng[0]), float(rng[1])

    if lo == hi:
        return np.array([lo])

    # ---- Port of R's pretty (src/appl/pretty.c R_pretty) ------------------
    # R's grid.pretty(range, n) is `pretty(range, n)` filtered to within
    # range; we follow the same two-step recipe.
    diff = hi - lo
    cell = max(abs(diff) / max(n, 1), 1e-10)

    base = 10.0 ** math.floor(math.log10(cell))
    # GEPretty (R src/main/engine.c) calls R_pretty0 with the
    # `high_u_fact = {0.8, 1.7}` bias, NOT R's user-facing pretty()
    # default of {1.5, 2.75}.  The smaller bias prefers denser ticks,
    # which is why grid.pretty(c(-7.49, 7.49)) returns step=2 (giving
    # 7 ticks) while pretty(c(-7.49, 7.49)) returns step=5 (5 ticks).
    h = 0.8
    h5 = 1.7
    unit = base
    if 2.0 * base - cell < h * (cell - unit):
        unit = 2.0 * base
    if 5.0 * base - cell < h5 * (cell - unit):
        unit = 5.0 * base
    if 10.0 * base - cell < h * (cell - unit):
        unit = 10.0 * base

    ns = math.floor(lo / unit + 1e-7)
    nu = math.ceil(hi / unit - 1e-7)

    lo_tick = ns * unit
    hi_tick = nu * unit

    ticks = np.arange(lo_tick, hi_tick + unit * 0.5, unit)
    # Clip floating-point noise at the boundaries
    ticks = np.round(ticks / unit) * unit

    # grid.pretty restricts to within the requested range
    # (grid R-3.6 src/library/grid/R/util.R: `res[res >= range[1] & res <= range[2]]`).
    ticks = ticks[(ticks >= lo) & (ticks <= hi)]

    return ticks


def n2mfrow(n: int) -> tuple[int, int]:
    """Compute a ``(nrow, ncol)`` layout to display *n* plots.

    This is a Python port of R's ``grDevices::n2mfrow``.

    Parameters
    ----------
    n : int
        Total number of plots.

    Returns
    -------
    tuple of (int, int)
        ``(nrow, ncol)`` suitable for passing to a layout function.

    Examples
    --------
    >>> n2mfrow(5)
    (3, 2)
    >>> n2mfrow(1)
    (1, 1)
    """
    if n <= 0:
        return (0, 0)
    if n <= 3:
        return (n, 1)
    if n <= 6:
        return (3, 2) if n > 4 else (2, 2)
    if n <= 12:
        ncol = 3
        nrow = math.ceil(n / ncol)
        return (nrow, ncol)

    # General case: roughly square
    ncol = math.ceil(math.sqrt(n))
    nrow = math.ceil(n / ncol)
    return (nrow, ncol)
