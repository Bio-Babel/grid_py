"""Layout system for grid_py -- Python port of R's ``grid::grid.layout``.

This module provides the :class:`GridLayout` class and associated accessor
functions that mirror R's ``grid.layout()`` constructor and its companion
helper functions (``layout.nrow``, ``layout.ncol``, etc.).

A layout partitions a rectangular region into a grid of rows and columns
whose sizes may be expressed in any unit supported by the grid unit system.
The *respect* mechanism allows certain cells to maintain their aspect ratio
when the viewport is resized.

References
----------
R source: ``src/library/grid/R/layout.R``
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ._just import valid_just
from ._units import Unit, is_unit

__all__ = [
    "GridLayout",
    "layout_nrow",
    "layout_ncol",
    "layout_widths",
    "layout_heights",
    "layout_respect",
    "layout_region",
]


class GridLayout:
    """A grid layout specification.

    Divides a rectangular region into *nrow* rows and *ncol* columns whose
    dimensions are given by *widths* and *heights* (as :class:`Unit` objects).

    Parameters
    ----------
    nrow : int
        Number of rows.
    ncol : int
        Number of columns.
    widths : Unit or None
        Column widths.  If *None* (the default), each column receives equal
        ``"null"`` space (``Unit([1]*ncol, "null")``).
    heights : Unit or None
        Row heights.  If *None* (the default), each row receives equal
        ``"null"`` space (``Unit([1]*nrow, "null")``).
    default_units : str
        Unit type applied to *widths* / *heights* when they are supplied as
        plain numeric values rather than :class:`Unit` objects.
    respect : bool or numpy.ndarray
        If ``False`` (the default), no aspect-ratio constraints are applied.
        If ``True``, all null-unit cells are respected.  An *nrow* x *ncol*
        integer matrix (``numpy.ndarray``) selects individual cells to respect.
    just : str or sequence of str
        Justification of the layout within its parent viewport.  Accepted
        values follow :func:`._just.valid_just` conventions (e.g.
        ``"centre"``, ``"left"``, ``["right", "top"]``).
    """

    __slots__ = (
        "_nrow",
        "_ncol",
        "_widths",
        "_heights",
        "_respect",
        "_valid_respect",
        "_respect_mat",
        "_just",
        "_valid_just",
    )

    def __init__(
        self,
        nrow: int = 1,
        ncol: int = 1,
        widths: Optional[Unit] = None,
        heights: Optional[Unit] = None,
        default_units: str = "null",
        respect: Union[bool, np.ndarray] = False,
        just: Union[str, Sequence[str]] = "centre",
    ) -> None:
        self._nrow: int = int(nrow)
        self._ncol: int = int(ncol)

        # -- widths ----------------------------------------------------------
        if widths is None:
            self._widths: Unit = Unit([1.0] * self._ncol, "null")
        elif is_unit(widths):
            self._widths = widths
        else:
            self._widths = Unit(widths, default_units)

        # -- heights ---------------------------------------------------------
        if heights is None:
            self._heights: Unit = Unit([1.0] * self._nrow, "null")
        elif is_unit(heights):
            self._heights = heights
        else:
            self._heights = Unit(heights, default_units)

        # -- respect ---------------------------------------------------------
        if isinstance(respect, np.ndarray):
            respect_arr = np.asarray(respect, dtype=np.int32)
            if respect_arr.shape != (self._nrow, self._ncol):
                raise ValueError(
                    "'respect' must be logical or an 'nrow' by 'ncol' matrix; "
                    f"got shape {respect_arr.shape}, expected "
                    f"({self._nrow}, {self._ncol})"
                )
            self._respect_mat: np.ndarray = respect_arr
            # R stores integer 2 to signal "matrix mode"
            self._respect: Union[bool, np.ndarray] = respect
            self._valid_respect: int = 2
        elif respect:
            self._respect_mat = np.zeros(
                (self._nrow, self._ncol), dtype=np.int32
            )
            self._respect = True
            self._valid_respect = 1
        else:
            self._respect_mat = np.zeros(
                (self._nrow, self._ncol), dtype=np.int32
            )
            self._respect = False
            self._valid_respect = 0

        # -- justification ---------------------------------------------------
        self._just = just
        self._valid_just: Tuple[float, float] = valid_just(just)

    # --------------------------------------------------------------------- #
    # Properties                                                            #
    # --------------------------------------------------------------------- #

    @property
    def nrow(self) -> int:
        """Number of rows in the layout."""
        return self._nrow

    @property
    def ncol(self) -> int:
        """Number of columns in the layout."""
        return self._ncol

    @property
    def widths(self) -> Unit:
        """Column widths as a :class:`Unit`."""
        return self._widths

    @property
    def heights(self) -> Unit:
        """Row heights as a :class:`Unit`."""
        return self._heights

    @property
    def respect(self) -> Union[bool, np.ndarray]:
        """Respect specification.

        Returns ``False`` (no respect), ``True`` (full respect), or an
        *nrow* x *ncol* integer matrix indicating per-cell respect.
        """
        if self._valid_respect == 0:
            return False
        if self._valid_respect == 1:
            return True
        return self._respect_mat

    @property
    def respect_mat(self) -> np.ndarray:
        """The *nrow* x *ncol* integer matrix of per-cell respect flags."""
        return self._respect_mat

    @property
    def dim(self) -> Tuple[int, int]:
        """Layout dimensions as ``(nrow, ncol)``."""
        return (self._nrow, self._ncol)

    # --------------------------------------------------------------------- #
    # Dunder methods                                                        #
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"GridLayout(nrow={self._nrow}, ncol={self._ncol}, "
            f"widths={self._widths!r}, heights={self._heights!r}, "
            f"respect={self._respect!r}, just={self._just!r})"
        )


# ======================================================================= #
# Module-level accessor functions                                         #
# ======================================================================= #


def layout_nrow(layout: GridLayout) -> int:
    """Return the number of rows in *layout*.

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.

    Returns
    -------
    int
        Number of rows.
    """
    return layout.nrow


def layout_ncol(layout: GridLayout) -> int:
    """Return the number of columns in *layout*.

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.

    Returns
    -------
    int
        Number of columns.
    """
    return layout.ncol


def layout_widths(layout: GridLayout) -> Unit:
    """Return the column widths of *layout*.

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.

    Returns
    -------
    Unit
        Column widths.
    """
    return layout.widths


def layout_heights(layout: GridLayout) -> Unit:
    """Return the row heights of *layout*.

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.

    Returns
    -------
    Unit
        Row heights.
    """
    return layout.heights


def layout_respect(layout: GridLayout) -> Union[bool, np.ndarray]:
    """Return the respect specification of *layout*.

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.

    Returns
    -------
    bool or numpy.ndarray
        ``False`` for no respect, ``True`` for full respect, or an
        *nrow* x *ncol* integer matrix for per-cell respect.
    """
    return layout.respect


# ======================================================================= #
# Three-phase layout algorithm (mirrors R layout.c:calcViewportLayout)   #
# ======================================================================= #


def _col_respected(col: int, layout: GridLayout) -> bool:
    """Check if column *col* (0-based) has any respected cell.

    Mirrors R ``layout.c:colRespected`` (lines 121-133).
    """
    if layout._valid_respect == 1:
        return True
    if layout._valid_respect == 2:
        return bool(np.any(layout._respect_mat[:, col] != 0))
    return False


def _row_respected(row: int, layout: GridLayout) -> bool:
    """Check if row *row* (0-based) has any respected cell.

    Mirrors R ``layout.c:rowRespected`` (lines 135-147).
    """
    if layout._valid_respect == 1:
        return True
    if layout._valid_respect == 2:
        return bool(np.any(layout._respect_mat[row, :] != 0))
    return False


def _calc_layout_sizes(
    layout: GridLayout,
    parent_w_px: float,
    parent_h_px: float,
    dpi: float = 150.0,
) -> Tuple[list, list]:
    """Three-phase layout negotiation algorithm.

    Mirrors R ``layout.c:calcViewportLayout`` (lines 492-591).

    Phase 1: Allocate absolute (non-null) units to device pixels.
    Phase 2: Allocate respected null units with unified aspect-ratio scale.
    Phase 3: Distribute remaining space among unrespected null units.

    Parameters
    ----------
    layout : GridLayout
        The layout specification.
    parent_w_px, parent_h_px : float
        Available space in device pixels.
    dpi : float
        Device resolution in dots per inch.

    Returns
    -------
    (col_widths, row_heights) : (list[float], list[float])
        Allocated sizes in device pixels.
    """
    from ._units import _INCHES_PER

    ncol = layout.nrow if False else layout.ncol  # kept explicit for clarity
    nrow = layout.nrow
    ncol = layout.ncol
    widths = layout.widths
    heights = layout.heights

    col_sizes = [0.0] * ncol
    row_sizes = [0.0] * nrow
    relative_w = [False] * ncol
    relative_h = [False] * nrow

    reduced_w = parent_w_px
    reduced_h = parent_h_px

    # ---- Phase 1: allocate absolute units (layout.c:allocateKnownWidths) --
    for i in range(ncol):
        utype = widths._units[i] if i < len(widths._units) else "null"
        if utype == "null":
            relative_w[i] = True
        else:
            val = float(widths._values[i])
            if utype == "npc":
                px = val * parent_w_px
            elif utype in _INCHES_PER:
                px = val * _INCHES_PER[utype] * dpi
            else:
                # Unknown type: treat as null
                relative_w[i] = True
                continue
            col_sizes[i] = px
            reduced_w -= px

    for j in range(nrow):
        utype = heights._units[j] if j < len(heights._units) else "null"
        if utype == "null":
            relative_h[j] = True
        else:
            val = float(heights._values[j])
            if utype == "npc":
                px = val * parent_h_px
            elif utype in _INCHES_PER:
                px = val * _INCHES_PER[utype] * dpi
            else:
                relative_h[j] = True
                continue
            row_sizes[j] = px
            reduced_h -= px

    reduced_w = max(reduced_w, 0.0)
    reduced_h = max(reduced_h, 0.0)

    # ---- Phase 2: allocate respected null units (layout.c:allocateRespected)
    if layout._valid_respect > 0 and (reduced_w > 0 or reduced_h > 0):
        sum_w = sum(
            float(widths._values[i])
            for i in range(ncol)
            if relative_w[i] and _col_respected(i, layout)
        )
        sum_h = sum(
            float(heights._values[j])
            for j in range(nrow)
            if relative_h[j] and _row_respected(j, layout)
        )

        temp_w = reduced_w
        temp_h = reduced_h

        if sum_w > 0 or sum_h > 0:
            # Determine limiting dimension (layout.c:221)
            if temp_h * sum_w > sum_h * temp_w:
                denom, mult = sum_w, temp_w
            else:
                denom, mult = sum_h, temp_h

            for i in range(ncol):
                if relative_w[i] and _col_respected(i, layout):
                    # Special case: sumHeight==0 (layout.c:240-243)
                    d, m = denom, mult
                    if sum_h == 0:
                        d, m = sum_w, temp_w
                    if d > 0:
                        col_sizes[i] = float(widths._values[i]) / d * m
                    reduced_w -= col_sizes[i]

            for j in range(nrow):
                if relative_h[j] and _row_respected(j, layout):
                    d, m = denom, mult
                    if sum_w == 0:
                        d, m = sum_h, temp_h
                    if d > 0:
                        row_sizes[j] = float(heights._values[j]) / d * m
                    reduced_h -= row_sizes[j]
    else:
        # No respect or no remaining space: respected nulls get 0
        for i in range(ncol):
            if relative_w[i] and _col_respected(i, layout):
                col_sizes[i] = 0.0
        for j in range(nrow):
            if relative_h[j] and _row_respected(j, layout):
                row_sizes[j] = 0.0

    reduced_w = max(reduced_w, 0.0)
    reduced_h = max(reduced_h, 0.0)

    # ---- Phase 3: allocate unrespected null units (layout.c:allocateRemaining)
    sum_unresp_w = sum(
        float(widths._values[i])
        for i in range(ncol)
        if relative_w[i] and not _col_respected(i, layout)
    )
    if sum_unresp_w > 0:
        for i in range(ncol):
            if relative_w[i] and not _col_respected(i, layout):
                col_sizes[i] = reduced_w * float(widths._values[i]) / sum_unresp_w
    else:
        for i in range(ncol):
            if relative_w[i] and not _col_respected(i, layout):
                col_sizes[i] = 0.0

    sum_unresp_h = sum(
        float(heights._values[j])
        for j in range(nrow)
        if relative_h[j] and not _row_respected(j, layout)
    )
    if sum_unresp_h > 0:
        for j in range(nrow):
            if relative_h[j] and not _row_respected(j, layout):
                row_sizes[j] = reduced_h * float(heights._values[j]) / sum_unresp_h
    else:
        for j in range(nrow):
            if relative_h[j] and not _row_respected(j, layout):
                row_sizes[j] = 0.0

    return col_sizes, row_sizes


def layout_region(
    layout: GridLayout,
    row: Union[int, Sequence[int]],
    col: Union[int, Sequence[int]],
    parent_w_px: Optional[float] = None,
    parent_h_px: Optional[float] = None,
    dpi: float = 150.0,
) -> Dict[str, Unit]:
    """Compute the region occupied by a range of layout cells.

    Mirrors R's ``layoutRegion`` function.  Row and column indices are
    **1-based** (following R convention).  A single integer selects one
    row/column; a two-element sequence ``[start, end]`` selects an inclusive
    range.

    When *parent_w_px* and *parent_h_px* are provided, the full three-phase
    layout negotiation algorithm is used (matching R's ``calcViewportLayout``
    in ``layout.c``), correctly handling mixed units (``cm`` + ``null`` +
    ``npc``), the ``respect`` parameter, and per-cell respect matrices.

    When these parameters are omitted, a simplified proportional allocation
    is used (backward compatible; only correct for uniform ``null`` units).

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.
    row : int or sequence of int
        1-based row index or ``[start, end]`` range (inclusive).
    col : int or sequence of int
        1-based column index or ``[start, end]`` range (inclusive).
    parent_w_px : float or None
        Parent viewport width in device pixels.  When provided, enables the
        full layout algorithm.
    parent_h_px : float or None
        Parent viewport height in device pixels.
    dpi : float
        Device resolution (dots per inch).

    Returns
    -------
    dict
        Dictionary with keys ``"left"``, ``"bottom"``, ``"width"``,
        ``"height"``, each containing a :class:`Unit` in ``"npc"`` units.
    """
    # Normalise to (start, end) -- 1-based inclusive
    if isinstance(row, (int, np.integer)):
        row_start, row_end = int(row), int(row)
    else:
        row_seq = list(row)
        row_start = int(row_seq[0])
        row_end = int(row_seq[-1]) if len(row_seq) > 1 else row_start

    if isinstance(col, (int, np.integer)):
        col_start, col_end = int(col), int(col)
    else:
        col_seq = list(col)
        col_start = int(col_seq[0])
        col_end = int(col_seq[-1]) if len(col_seq) > 1 else col_start

    # Convert to 0-based indices
    r0 = row_start - 1
    r1 = row_end  # exclusive upper bound
    c0 = col_start - 1
    c1 = col_end

    # -- Full three-phase algorithm when parent dimensions available --------
    if parent_w_px is not None and parent_h_px is not None:
        col_widths, row_heights = _calc_layout_sizes(
            layout, parent_w_px, parent_h_px, dpi,
        )
        total_w = sum(col_widths) or 1.0
        total_h = sum(row_heights) or 1.0

        left_frac = sum(col_widths[:c0]) / total_w
        width_frac = sum(col_widths[c0:c1]) / total_w
        top_frac = sum(row_heights[:r0]) / total_h
        height_frac = sum(row_heights[r0:r1]) / total_h
        bottom_frac = 1.0 - top_frac - height_frac

        return {
            "left": Unit(left_frac, "npc"),
            "bottom": Unit(bottom_frac, "npc"),
            "width": Unit(width_frac, "npc"),
            "height": Unit(height_frac, "npc"),
        }

    # -- Simplified proportional fallback (null units only) ----------------
    w_vals = np.asarray(layout.widths._values, dtype=np.float64)
    h_vals = np.asarray(layout.heights._values, dtype=np.float64)

    total_w = w_vals.sum() or 1.0
    total_h = h_vals.sum() or 1.0

    left_frac = w_vals[:c0].sum() / total_w
    width_frac = w_vals[c0:c1].sum() / total_w

    top_frac = h_vals[:r0].sum() / total_h
    height_frac = h_vals[r0:r1].sum() / total_h
    bottom_frac = 1.0 - top_frac - height_frac

    return {
        "left": Unit(left_frac, "npc"),
        "bottom": Unit(bottom_frac, "npc"),
        "width": Unit(width_frac, "npc"),
        "height": Unit(height_frac, "npc"),
    }
