"""
Arrow head specification for grid_py -- Python port of R's ``grid::arrow()``.

This module provides the :class:`Arrow` class and the :func:`arrow` factory
function, which describe the arrow heads that can be attached to line-based
grobs (segments, lines, curves, etc.).

The implementation mirrors the behaviour of R's ``arrow()`` constructor,
``length.arrow``, ``rep.arrow``, and ``[.arrow`` method defined in
``src/library/grid/R/primitives.R``.

Examples
--------
>>> from grid_py._arrow import arrow
>>> a = arrow()
>>> a
Arrow(angle=[30.0], length=Unit([0.25], 'inches'), ends=[2], type=[1])
>>> len(a)
1
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from ._units import Unit, unit_rep

__all__ = ["Arrow", "arrow"]


def _recycle_unit(u: Unit, n: int) -> Unit:
    """Recycle a Unit to length *n* using indexing with modular indices."""
    lu = len(u)
    if lu == n:
        return u
    indices = [i % lu for i in range(n)]
    return u[indices]

# Valid string values for *ends* and *type*, following R's match() semantics.
_VALID_ENDS = ("first", "last", "both")
_VALID_TYPES = ("open", "closed")


class Arrow:
    """Description of an arrow head to attach to a line-based grob.

    Parameters
    ----------
    angle : float or Sequence[float]
        Angle of the arrow head in degrees (the angle between the shaft and
        each edge of the arrow head).  Scalar or vector.
    length : Unit, optional
        Length of the arrow head measured along the edges.  Must be a
        :class:`Unit` object.  Defaults to ``Unit(0.25, "inches")``.
    ends : {"first", "last", "both"} or Sequence[str]
        Which end(s) of the line should receive an arrow head.  Encoded
        internally as integers: ``1`` = first, ``2`` = last, ``3`` = both,
        matching R's ``match()`` convention.
    type : {"open", "closed"} or Sequence[str]
        Whether the arrow head is open or closed.  Encoded internally as
        integers: ``1`` = open, ``2`` = closed.

    Raises
    ------
    TypeError
        If *length* is not a :class:`Unit` object.
    ValueError
        If *ends* or *type* contains invalid values.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        angle: Union[float, int, Sequence[float]] = 30,
        length: Optional[Unit] = None,
        ends: Union[str, Sequence[str]] = "last",
        type: Union[str, Sequence[str]] = "open",  # noqa: A002
    ) -> None:
        # --- angle --------------------------------------------------------
        if isinstance(angle, (int, float, np.integer, np.floating)):
            self._angle: np.ndarray = np.asarray([float(angle)], dtype=np.float64)
        else:
            self._angle = np.asarray(angle, dtype=np.float64).ravel()

        # --- length -------------------------------------------------------
        if length is None:
            length = Unit(0.25, "inches")
        if not isinstance(length, Unit):
            raise TypeError("'length' must be a Unit object")
        self._length: Unit = length

        # --- ends ---------------------------------------------------------
        self._ends: np.ndarray = self._encode_match(ends, _VALID_ENDS, "ends")

        # --- type ---------------------------------------------------------
        self._type: np.ndarray = self._encode_match(type, _VALID_TYPES, "type")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_match(
        value: Union[str, Sequence[str]],
        valid: tuple,
        label: str,
    ) -> np.ndarray:
        """Encode string(s) to 1-based integer codes, like R's ``match()``.

        Parameters
        ----------
        value : str or Sequence[str]
            Input value(s).
        valid : tuple of str
            Allowed string values.
        label : str
            Name used in error messages.

        Returns
        -------
        np.ndarray
            1-based integer codes corresponding to *value*.

        Raises
        ------
        ValueError
            If any element of *value* is not in *valid*.
        """
        if isinstance(value, str):
            value = [value]
        codes: List[int] = []
        for v in value:
            if v not in valid:
                raise ValueError(
                    f"invalid '{label}' argument: {v!r}; "
                    f"must be one of {valid}"
                )
            codes.append(valid.index(v) + 1)
        arr = np.asarray(codes, dtype=np.int64)
        if arr.size == 0:
            raise ValueError(f"'{label}' must have length > 0")
        return arr

    # ------------------------------------------------------------------
    # Properties (read-only access to internal data)
    # ------------------------------------------------------------------

    @property
    def angle(self) -> np.ndarray:
        """Arrow-head angle(s) in degrees."""
        return self._angle

    @property
    def length(self) -> Unit:
        """Arrow-head length as a :class:`Unit`."""
        return self._length

    @property
    def ends(self) -> np.ndarray:
        """Integer code(s) for which end gets an arrow (1=first, 2=last, 3=both)."""
        return self._ends

    @property
    def type(self) -> np.ndarray:
        """Integer code(s) for arrow type (1=open, 2=closed)."""
        return self._type

    # ------------------------------------------------------------------
    # length (len) -- mirrors R's length.arrow
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the effective vector length of this Arrow.

        Follows R's ``length.arrow`` which returns the maximum of the
        lengths of all component vectors (angle, length, ends, type).

        Returns
        -------
        int
            Effective length.
        """
        return int(
            max(
                len(self._angle),
                len(self._length),
                len(self._ends),
                len(self._type),
            )
        )

    # ------------------------------------------------------------------
    # Subscript -- mirrors R's `[.arrow`
    # ------------------------------------------------------------------

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> "Arrow":
        """Subset an Arrow, recycling components to a common length first.

        Parameters
        ----------
        index : int, slice, or Sequence[int]
            Index(es) to select.

        Returns
        -------
        Arrow
            A new :class:`Arrow` with the selected elements.
        """
        maxn = len(self)

        # Recycle each component to *maxn*.
        angle = np.resize(self._angle, maxn)
        length = _recycle_unit(self._length, maxn)
        ends = np.resize(self._ends, maxn)
        type_ = np.resize(self._type, maxn)

        # Apply the index.
        new = object.__new__(Arrow)
        new._angle = np.atleast_1d(angle[index])
        new._length = length[index]
        new._ends = np.atleast_1d(ends[index])
        new._type = np.atleast_1d(type_[index])
        return new

    # ------------------------------------------------------------------
    # rep -- mirrors R's rep.arrow
    # ------------------------------------------------------------------

    def rep(self, times: int = 1, length_out: Optional[int] = None) -> "Arrow":
        """Repeat the Arrow, recycling components to a common length first.

        Parameters
        ----------
        times : int, optional
            Number of times to repeat (default ``1``).
        length_out : int, optional
            Desired length of the result.  If given, *times* is ignored and
            the output is truncated or recycled to this length.

        Returns
        -------
        Arrow
            A new :class:`Arrow` with repeated elements.
        """
        maxn = len(self)

        # First recycle components to the common length.
        angle = np.resize(self._angle, maxn)
        ends = np.resize(self._ends, maxn)
        type_ = np.resize(self._type, maxn)
        length = _recycle_unit(self._length, maxn)

        # Then tile by *times*.
        angle = np.tile(angle, times)
        ends = np.tile(ends, times)
        type_ = np.tile(type_, times)
        length = unit_rep(length, times)

        # Trim / recycle to *length_out* if requested.
        if length_out is not None:
            angle = np.resize(angle, length_out)
            ends = np.resize(ends, length_out)
            type_ = np.resize(type_, length_out)
            length = _recycle_unit(length, length_out)

        new = object.__new__(Arrow)
        new._angle = angle
        new._length = length
        new._ends = ends
        new._type = type_
        return new

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Arrow(angle={self._angle.tolist()}, "
            f"length={self._length!r}, "
            f"ends={self._ends.tolist()}, "
            f"type={self._type.tolist()})"
        )


# ----------------------------------------------------------------------
# Factory function
# ----------------------------------------------------------------------


def arrow(
    angle: Union[float, int, Sequence[float]] = 30,
    length: Optional[Unit] = None,
    ends: Union[str, Sequence[str]] = "last",
    type: Union[str, Sequence[str]] = "open",  # noqa: A002
) -> Arrow:
    """Create an Arrow specification.

    This is the main user-facing factory function, equivalent to R's
    ``grid::arrow()``.

    Parameters
    ----------
    angle : float or Sequence[float], optional
        Angle of the arrow head in degrees (default ``30``).
    length : Unit, optional
        Length of the arrow head edges.  Defaults to
        ``Unit(0.25, "inches")``.
    ends : {"first", "last", "both"} or Sequence[str], optional
        Which end(s) of the line receive an arrow head (default ``"last"``).
    type : {"open", "closed"} or Sequence[str], optional
        Arrow head style (default ``"open"``).

    Returns
    -------
    Arrow
        A new :class:`Arrow` instance.

    Examples
    --------
    >>> from grid_py._arrow import arrow
    >>> a = arrow(angle=45, type="closed")
    >>> a
    Arrow(angle=[45.0], length=Unit([0.25], 'inches'), ends=[2], type=[2])
    """
    return Arrow(angle=angle, length=length, ends=ends, type=type)
