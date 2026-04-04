"""Path-related classes for grid_py (port of R's grid path system).

This module provides :class:`GPath` and :class:`VpPath` for addressing grobs
and viewports by hierarchical name paths, as well as :class:`GridPath` and
the helper :func:`as_path` for treating a grob as a single filled/stroked
path.  These are direct ports of the R ``gPath()``, ``vpPath()``, and
``as.path()`` facilities in the *grid* package.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

__all__ = [
    "GPath",
    "VpPath",
    "GridPath",
    "as_path",
    "is_closed",
    "PATH_SEP",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATH_SEP: str = "::"
"""Separator used between components in both grob and viewport paths."""

# ---------------------------------------------------------------------------
# Grob types considered "open" (i.e. not closed shapes).
# ---------------------------------------------------------------------------

_OPEN_TYPES: frozenset[str] = frozenset(
    {
        "move.to",
        "line.to",
        "lines",
        "polyline",
        "segments",
        "beziergrob",
    }
)


# ---------------------------------------------------------------------------
# GPath
# ---------------------------------------------------------------------------


class GPath:
    """A path addressing a grob inside a grob hierarchy.

    Components are joined with the ``"::"`` separator, mirroring R's
    ``gPath()`` constructor.

    Parameters
    ----------
    *args : str
        One or more path component strings.  Each string may itself
        contain ``"::"`` separators which will be split automatically.

    Examples
    --------
    >>> p = GPath("a", "b", "c")
    >>> str(p)
    'a::b::c'
    >>> p.name
    'c'
    >>> p.path
    'a::b'
    >>> p.n
    3
    """

    __slots__ = ("_components",)

    def __init__(self, *args: str) -> None:
        components: list[str] = []
        for a in args:
            if not isinstance(a, str):
                raise TypeError(f"invalid grob name: expected str, got {type(a).__name__}")
            components.extend(a.split(PATH_SEP))
        if len(components) < 1:
            raise ValueError("a grob path must contain at least one grob name")
        for c in components:
            if not c:
                raise ValueError("invalid grob name (empty string)")
        self._components: tuple[str, ...] = tuple(components)

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Last (leaf) component of the path.

        Returns
        -------
        str
        """
        return self._components[-1]

    @property
    def path(self) -> Optional[str]:
        """Parent path as a ``"::"``-separated string, or ``None``.

        Returns
        -------
        str or None
            ``None`` when the path has only one component.
        """
        if self._components.__len__() == 1:
            return None
        return PATH_SEP.join(self._components[:-1])

    @property
    def n(self) -> int:
        """Depth (number of components) of the path.

        Returns
        -------
        int
        """
        return len(self._components)

    @property
    def components(self) -> tuple[str, ...]:
        """All path components as a tuple of strings.

        Returns
        -------
        tuple[str, ...]
        """
        return self._components

    # -- dunder methods -----------------------------------------------------

    def __str__(self) -> str:
        return PATH_SEP.join(self._components)

    def __repr__(self) -> str:
        return f"GPath({PATH_SEP.join(self._components)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GPath):
            return self._components == other._components
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._components)

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# VpPath
# ---------------------------------------------------------------------------


class VpPath:
    """A path addressing a viewport inside the viewport tree.

    Components are joined with ``"::"`` separators, mirroring R's
    ``vpPath()`` constructor.

    Parameters
    ----------
    *args : str
        One or more path component strings.  Each string may itself
        contain ``"::"`` separators which will be split automatically.

    Examples
    --------
    >>> vp = VpPath("root", "panel", "strip")
    >>> str(vp)
    'root::panel::strip'
    >>> vp.name
    'strip'
    >>> vp[0:2]
    VpPath(root::panel)
    """

    __slots__ = ("_components",)

    def __init__(self, *args: str) -> None:
        components: list[str] = []
        for a in args:
            if not isinstance(a, str):
                raise TypeError(
                    f"invalid viewport name: expected str, got {type(a).__name__}"
                )
            components.extend(a.split(PATH_SEP))
        if len(components) < 1:
            raise ValueError(
                "a viewport path must contain at least one viewport name"
            )
        for c in components:
            if not c:
                raise ValueError("invalid viewport name (empty string)")
        self._components: tuple[str, ...] = tuple(components)

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Last (leaf) component of the path.

        Returns
        -------
        str
        """
        return self._components[-1]

    @property
    def path(self) -> Optional[str]:
        """Parent path as a ``"::"``-separated string, or ``None``.

        Returns
        -------
        str or None
            ``None`` when the path has only one component.
        """
        if len(self._components) == 1:
            return None
        return PATH_SEP.join(self._components[:-1])

    @property
    def n(self) -> int:
        """Depth (number of components) of the path.

        Returns
        -------
        int
        """
        return len(self._components)

    @property
    def components(self) -> tuple[str, ...]:
        """All path components as a tuple of strings.

        Returns
        -------
        tuple[str, ...]
        """
        return self._components

    # -- indexing ------------------------------------------------------------

    def __getitem__(self, index: Union[int, slice]) -> "VpPath":
        """Index or slice the path to obtain a sub-path.

        Parameters
        ----------
        index : int or slice
            Integer index or slice applied to the ordered components.

        Returns
        -------
        VpPath
            A new ``VpPath`` built from the selected components.

        Raises
        ------
        IndexError
            If the resulting selection is empty.
        """
        selected = self._components[index]
        if isinstance(selected, str):
            selected = (selected,)
        if len(selected) == 0:
            raise IndexError("resulting viewport path is empty")
        return VpPath(*selected)

    # -- dunder methods -----------------------------------------------------

    def __str__(self) -> str:
        return PATH_SEP.join(self._components)

    def __repr__(self) -> str:
        return f"VpPath({PATH_SEP.join(self._components)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VpPath):
            return self._components == other._components
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._components)

    def __len__(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# GridPath  (wraps a grob as a stroke/fill path)
# ---------------------------------------------------------------------------

_VALID_RULES: frozenset[str] = frozenset({"winding", "evenodd"})


class GridPath:
    """Wrapper that marks a grob for path-based stroke/fill operations.

    This is the Python equivalent of R's ``GridPath`` S3 class produced by
    ``as.path()``.

    Parameters
    ----------
    grob : Any
        The grob object to treat as a path.
    gp : Any, optional
        Graphical parameters (``Gpar`` instance or ``None``).
    rule : str, optional
        Fill rule, either ``"winding"`` (default) or ``"evenodd"``.

    Raises
    ------
    ValueError
        If *rule* is not one of the accepted values.
    """

    __slots__ = ("grob", "gp", "rule")

    def __init__(
        self,
        grob: Any,
        gp: Any = None,
        rule: str = "winding",
    ) -> None:
        if rule not in _VALID_RULES:
            raise ValueError(
                f"'rule' must be one of {sorted(_VALID_RULES)}, got {rule!r}"
            )
        self.grob: Any = grob
        self.gp: Any = gp
        self.rule: str = rule

    def __repr__(self) -> str:
        return f"GridPath(grob={self.grob!r}, rule={self.rule!r})"


# ---------------------------------------------------------------------------
# as_path  (factory function)
# ---------------------------------------------------------------------------


def as_path(
    x: Any,
    gp: Any = None,
    rule: str = "winding",
) -> GridPath:
    """Convert a grob to a :class:`GridPath` for stroke/fill operations.

    This mirrors R's ``as.path()`` function from the *grid* package.

    Parameters
    ----------
    x : Any
        A grob instance.  In the current implementation no strict type check
        is enforced beyond ensuring *x* is not ``None``.
    gp : Any, optional
        Graphical parameters to associate with the path.
    rule : str, optional
        Fill rule — ``"winding"`` (default) or ``"evenodd"``.

    Returns
    -------
    GridPath

    Raises
    ------
    TypeError
        If *x* is ``None``.
    ValueError
        If *rule* is invalid.
    """
    if x is None:
        raise TypeError("only a grob can be converted to a path")
    return GridPath(grob=x, gp=gp, rule=rule)


# ---------------------------------------------------------------------------
# is_closed  (S3-style dispatch)
# ---------------------------------------------------------------------------


def is_closed(x: Any) -> bool:
    """Return whether a grob represents a closed shape.

    Mimics R's ``isClosed()`` generic with method dispatch based on a
    ``_grid_class`` attribute (or, failing that, the Python class name).

    Open shapes (returning ``False``):
        ``move.to``, ``line.to``, ``lines``, ``polyline``, ``segments``,
        ``beziergrob``.

    Everything else (including ``rect``, ``circle``, ``polygon``, and any
    unknown type) defaults to ``True``, matching R's ``isClosed.default``.

    Parameters
    ----------
    x : Any
        A grob or grob-like object.  The function inspects
        ``x._grid_class`` first; if that attribute is absent it falls back
        to ``type(x).__name__``.

    Returns
    -------
    bool
        ``True`` for closed shapes, ``False`` for open ones.
    """
    cls_name: str = getattr(x, "_grid_class", type(x).__name__)
    return cls_name not in _OPEN_TYPES
