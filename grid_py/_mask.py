"""Mask system for grid_py.

Python port of R's ``grid`` package mask infrastructure
(``grid/R/mask.R``).  Provides the :class:`GridMask` class and the
:func:`as_mask` factory function for converting a grob (or a boolean
sentinel) into an alpha or luminance mask.

Classes
-------
GridMask
    A mask defined by a reference grob and a compositing type.

Functions
---------
as_mask
    Factory for :class:`GridMask`.
is_mask
    Type-check predicate.
"""

from __future__ import annotations

from typing import Any, List, Union

__all__: List[str] = [
    "GridMask",
    "as_mask",
    "is_mask",
]

# Valid mask types, matching R's ``match.arg(type)`` in ``as.mask()``.
_VALID_TYPES: tuple[str, ...] = ("alpha", "luminance")


# ======================================================================
# GridMask
# ======================================================================


class GridMask:
    """A mask wrapping a grob for alpha or luminance compositing.

    This is the Python equivalent of the R S3 class ``"GridMask"``
    produced by ``as.mask()``.

    Parameters
    ----------
    ref : Any
        The reference object that defines the mask.  Typically a grob,
        but may also be ``True`` (use the default mask) or ``False``
        (disable masking).
    type : str, optional
        Compositing type, either ``"alpha"`` (default) or
        ``"luminance"``.

    Raises
    ------
    ValueError
        If *type* is not one of the accepted values.

    Examples
    --------
    >>> m = GridMask("some_grob", type="alpha")
    >>> m.type
    'alpha'
    >>> m.ref
    'some_grob'
    """

    __slots__ = ("_ref", "_type")

    def __init__(self, ref: Any, type: str = "alpha") -> None:  # noqa: A002
        if type not in _VALID_TYPES:
            raise ValueError(
                f"'type' must be one of {list(_VALID_TYPES)}, got {type!r}"
            )
        self._ref: Any = ref
        self._type: str = type

    # -- properties ---------------------------------------------------------

    @property
    def ref(self) -> Any:
        """The grob (or boolean sentinel) that defines the mask.

        Returns
        -------
        Any
        """
        return self._ref

    @property
    def type(self) -> str:
        """Compositing type (``"alpha"`` or ``"luminance"``).

        Returns
        -------
        str
        """
        return self._type

    # -- dunder methods -----------------------------------------------------

    def __repr__(self) -> str:
        return f"GridMask(ref={self._ref!r}, type={self._type!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GridMask):
            return self._ref == other._ref and self._type == other._type
        return NotImplemented

    def __hash__(self) -> int:
        return hash((id(self._ref), self._type))


# ======================================================================
# Factory function
# ======================================================================


def as_mask(
    x: Union[Any, bool],
    type: str = "alpha",  # noqa: A002
) -> GridMask:
    """Convert a grob (or boolean) to a :class:`GridMask`.

    This mirrors R's ``as.mask()`` function from the *grid* package.

    When *x* is a grob the resulting :class:`GridMask` will use that grob
    as the mask definition.  Passing ``True`` or ``False`` creates a
    sentinel mask that signals default or disabled masking behaviour to
    the graphics device.

    Parameters
    ----------
    x : Any
        A grob instance, or ``True``/``False`` for default/disabled
        masking.  ``None`` is **not** accepted.
    type : str, optional
        Compositing type -- ``"alpha"`` (default) or ``"luminance"``.

    Returns
    -------
    GridMask
        A new mask object wrapping *x*.

    Raises
    ------
    TypeError
        If *x* is ``None``.
    ValueError
        If *type* is not ``"alpha"`` or ``"luminance"``.

    Examples
    --------
    >>> mask = as_mask("placeholder_grob")
    >>> mask.type
    'alpha'
    >>> mask.ref
    'placeholder_grob'

    >>> mask_lum = as_mask("grob", type="luminance")
    >>> mask_lum.type
    'luminance'
    """
    if x is None:
        raise TypeError("only a grob (or True/False) can be converted to a mask")
    return GridMask(ref=x, type=type)


# ======================================================================
# Predicate
# ======================================================================


def is_mask(x: Any) -> bool:
    """Return whether *x* is a :class:`GridMask` instance.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        ``True`` if *x* is a :class:`GridMask`, ``False`` otherwise.

    Examples
    --------
    >>> is_mask(as_mask("grob"))
    True
    >>> is_mask("not a mask")
    False
    """
    return isinstance(x, GridMask)
