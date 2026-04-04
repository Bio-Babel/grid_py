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


def layout_region(
    layout: GridLayout,
    row: Union[int, Sequence[int]],
    col: Union[int, Sequence[int]],
) -> Dict[str, Unit]:
    """Compute the region occupied by a range of layout cells.

    This mirrors R's ``layoutRegion`` function.  Row and column indices are
    **1-based** (following R convention).  A single integer selects one
    row/column; a two-element sequence ``[start, end]`` selects an inclusive
    range.

    The returned region is expressed in ``"npc"`` coordinates relative to
    the layout's total area.  This is a simplified, pure-Python
    implementation that only handles ``"null"`` unit columns/rows (the most
    common case).  Full support for mixed absolute/relative units would
    require the C-level layout engine.

    Parameters
    ----------
    layout : GridLayout
        A grid layout object.
    row : int or sequence of int
        1-based row index or ``[start, end]`` range (inclusive).
    col : int or sequence of int
        1-based column index or ``[start, end]`` range (inclusive).

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
    r1 = row_end  # exclusive upper bound (row_end is inclusive, so +1 -1+1)
    c0 = col_start - 1
    c1 = col_end

    # Extract raw numeric values from the Unit objects
    w_vals = np.asarray(layout.widths._values, dtype=np.float64)
    h_vals = np.asarray(layout.heights._values, dtype=np.float64)

    total_w = w_vals.sum()
    total_h = h_vals.sum()

    # Guard against zero totals
    if total_w == 0:
        total_w = 1.0
    if total_h == 0:
        total_h = 1.0

    # NPC fractions
    left_frac = w_vals[:c0].sum() / total_w
    width_frac = w_vals[c0:c1].sum() / total_w

    # Rows are counted top-to-bottom in R's layout but NPC y=0 is bottom.
    top_frac = h_vals[:r0].sum() / total_h
    height_frac = h_vals[r0:r1].sum() / total_h
    bottom_frac = 1.0 - top_frac - height_frac

    return {
        "left": Unit(left_frac, "npc"),
        "bottom": Unit(bottom_frac, "npc"),
        "width": Unit(width_frac, "npc"),
        "height": Unit(height_frac, "npc"),
    }
