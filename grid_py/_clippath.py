"""Clipping-path system for grid_py.

Python port of R's ``grid`` package clipping-path infrastructure
(``grid/R/clippath.R``).  Provides the :class:`GridClipPath` class and
the :func:`as_clip_path` factory function for converting a grob into a
clipping path.

Classes
-------
GridClipPath
    A clipping path defined by a grob.

Functions
---------
as_clip_path
    Factory for :class:`GridClipPath`.
is_clip_path
    Type-check predicate.
"""

from __future__ import annotations

from typing import Any, List

__all__: List[str] = [
    "GridClipPath",
    "as_clip_path",
    "is_clip_path",
]


# ======================================================================
# GridClipPath
# ======================================================================


class GridClipPath:
    """A clipping path wrapping a grob.

    This is the Python equivalent of the R S3 class ``"GridClipPath"``
    produced internally by ``createClipPath()``.

    The clipping path constrains drawing to the region defined by the
    referenced grob's outline.

    Parameters
    ----------
    grob : Any
        The grob object whose outline defines the clip region.

    Raises
    ------
    TypeError
        If *grob* is ``None``.

    Examples
    --------
    >>> cp = GridClipPath("some_grob")
    >>> cp.grob
    'some_grob'
    """

    __slots__ = ("_grob",)

    def __init__(self, grob: Any) -> None:
        if grob is None:
            raise TypeError("a GridClipPath requires a non-None grob")
        self._grob: Any = grob

    # -- properties ---------------------------------------------------------

    @property
    def grob(self) -> Any:
        """The grob whose outline defines the clipping region.

        Returns
        -------
        Any
        """
        return self._grob

    # -- dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        return f"GridClipPath(grob={self._grob!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridClipPath):
            return self._grob == other._grob
        return NotImplemented

    def __hash__(self) -> int:
        return hash(id(self._grob))


# ======================================================================
# Factory function
# ======================================================================


def as_clip_path(x: Any) -> GridClipPath:
    """Convert a grob to a :class:`GridClipPath`.

    This mirrors the intent of R's ``createClipPath()`` from the *grid*
    package, exposed here as a user-facing factory function.

    If *x* is already a :class:`GridClipPath` it is returned as-is,
    avoiding unnecessary wrapping.

    Parameters
    ----------
    x : Any
        A grob instance, or an existing :class:`GridClipPath`.
        ``None`` is **not** accepted.

    Returns
    -------
    GridClipPath
        A clipping-path object wrapping *x*.

    Raises
    ------
    TypeError
        If *x* is ``None``.

    Examples
    --------
    >>> cp = as_clip_path("placeholder_grob")
    >>> cp.grob
    'placeholder_grob'

    Passing an existing ``GridClipPath`` returns it unchanged:

    >>> cp2 = as_clip_path(cp)
    >>> cp2 is cp
    True
    """
    if isinstance(x, GridClipPath):
        return x
    if x is None:
        raise TypeError("only a grob can be converted to a clipping path")
    return GridClipPath(grob=x)


# ======================================================================
# Predicate
# ======================================================================


def is_clip_path(x: Any) -> bool:
    """Return whether *x* is a :class:`GridClipPath` instance.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a :class:`GridClipPath`, ``False`` otherwise.

    Examples
    --------
    >>> is_clip_path(as_clip_path("grob"))
    True
    >>> is_clip_path("not a clip path")
    False
    """
    return isinstance(x, GridClipPath)
