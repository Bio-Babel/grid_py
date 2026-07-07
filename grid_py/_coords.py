"""Coordinate query system for grid_py (port of R's ``grid/R/coords.R``).

This module provides data structures and functions for computing sets of points
around the perimeter (or along the length) of grobs.  The three container
classes -- :class:`GridCoords`, :class:`GridGrobCoords`, and
:class:`GridGTreeCoords` -- mirror R's ``GridCoords``, ``GridGrobCoords``,
and ``GridGTreeCoords`` S3 classes respectively.

``grob_coords`` is the user-level entry point that emulates drawing set-up
behaviour (pushing viewports, setting graphical parameters).
``grob_points`` skips that set-up and is intended for internal use when the
drawing context has already been established.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike

from ._grob import Grob, GTree, GList

__all__ = [
    "GridCoords",
    "GridGrobCoords",
    "GridGTreeCoords",
    "grob_coords",
    "grob_points",
    "grid_coords",
    "grid_grob_coords",
    "grid_gtree_coords",
    "empty_coords",
    "empty_grob_coords",
    "empty_gtree_coords",
    "is_empty_coords",
    "coords_bbox",
    "is_closed",
]

# ---------------------------------------------------------------------------
# Print indentation (mirrors R's ``coordPrintIndent``)
# ---------------------------------------------------------------------------

_COORD_PRINT_INDENT: str = "  "


# ---------------------------------------------------------------------------
# GridCoords -- low-level coordinate pair container
# ---------------------------------------------------------------------------


class GridCoords:
    """Container for a set of (x, y) coordinate pairs.

    This is the leaf node in the coordinate hierarchy.  It wraps two
    equal-length NumPy arrays of x and y values (typically in inches).

    Parameters
    ----------
    x : array_like
        X coordinates.
    y : array_like
        Y coordinates.
    name : str
        Human-readable label for this coordinate set.

    Raises
    ------
    ValueError
        If *x* and *y* do not have the same length.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        name: str = "coords",
    ) -> None:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        # Ensure 1-D
        x_arr = np.atleast_1d(x_arr)
        y_arr = np.atleast_1d(y_arr)
        if x_arr.shape != y_arr.shape:
            raise ValueError(
                f"x and y must have the same shape, got {x_arr.shape} and {y_arr.shape}"
            )
        self._x = x_arr
        self._y = y_arr
        self._name = name

    # -- properties ---------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """X coordinates as a NumPy array."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """Y coordinates as a NumPy array."""
        return self._y

    @property
    def name(self) -> str:
        """Human-readable label."""
        return self._name

    # -- query methods ------------------------------------------------------

    def get_x(self) -> np.ndarray:
        """Return the x coordinate array.

        Returns
        -------
        numpy.ndarray
            The x values.
        """
        return self._x

    def get_y(self) -> np.ndarray:
        """Return the y coordinate array.

        Returns
        -------
        numpy.ndarray
            The y values.
        """
        return self._y

    # -- transformation methods ---------------------------------------------

    def to_device(self, state: Any = None) -> "GridCoords":
        """Convert coordinates to device space.

        Parameters
        ----------
        state : object, optional
            Device / viewport state used for unit conversion.  When ``None``
            the coordinates are returned unchanged (no-op).

        Returns
        -------
        GridCoords
            New instance with device-space coordinates.
        """
        if self.is_empty():
            return self
        if state is None:
            return self
        # If a state object provides a ``device_loc`` method, use it.
        if hasattr(state, "device_loc"):
            dx, dy = state.device_loc(self._x, self._y)
            return GridCoords(dx, dy, name=self._name)
        return self

    def from_device(self, state: Any = None) -> "GridCoords":
        """Transform coordinates from device space using an inverse transform.

        In R this multiplies the coordinate matrix by ``solve(trans)``.

        Parameters
        ----------
        state : object, optional
            A 3x3 transformation matrix (ndarray) **or** an object with a
            ``transform`` attribute that is a 3x3 matrix.  When ``None`` the
            coordinates are returned unchanged.

        Returns
        -------
        GridCoords
            New instance with transformed coordinates.
        """
        if state is None:
            return self
        trans = state
        if hasattr(state, "transform"):
            trans = state.transform
        trans = np.asarray(trans, dtype=float)
        if trans.shape != (3, 3):
            raise ValueError("from_device requires a 3x3 transformation matrix")
        # R coords.R:183: cbind(x,y,1) %*% solve(trans)  →  pts @ inv(trans)
        inv = np.linalg.solve(trans, np.eye(3))
        pts = np.column_stack([self._x, self._y, np.ones(len(self._x))])
        result = pts @ inv
        return GridCoords(result[:, 0], result[:, 1], name=self._name)

    def is_empty(self) -> bool:
        """Return ``True`` when this represents the canonical empty coords.

        Returns
        -------
        bool
        """
        return (
            len(self._x) == 1
            and self._x[0] == 0.0
            and self._y[0] == 0.0
        )

    def transform_coords(self, tm: np.ndarray) -> "GridCoords":
        """Apply a 3x3 affine transformation matrix.

        Parameters
        ----------
        tm : numpy.ndarray
            A 3x3 affine transformation matrix.

        Returns
        -------
        GridCoords
            Transformed coordinates.
        """
        tm = np.asarray(tm, dtype=float)
        pts = np.column_stack([self._x, self._y, np.ones(len(self._x))])
        result = pts @ tm
        return GridCoords(result[:, 0], result[:, 1], name=self._name)

    def flatten(self) -> "GridCoords":
        """Return a flattened copy (no-op for leaf nodes).

        Returns
        -------
        GridCoords
        """
        return GridCoords(self._x.copy(), self._y.copy(), name=self._name)

    # -- dunder methods -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._x)

    def __repr__(self) -> str:
        def _fmt(arr: np.ndarray) -> str:
            if len(arr) > 3:
                head = " ".join(f"{v:.4g}" for v in arr[:3])
                return f"{head} ... [{len(arr)} values]"
            return " ".join(f"{v:.4g}" for v in arr) + f" [{len(arr)} values]"

        x_str = _fmt(self._x)
        y_str = _fmt(self._y)
        return f"x: {x_str}\ny: {y_str}"


# ---------------------------------------------------------------------------
# GridGrobCoords -- coordinates for a single grob (list of GridCoords)
# ---------------------------------------------------------------------------


class GridGrobCoords:
    """Container for coordinates of a single grob.

    Wraps an ordered list of :class:`GridCoords` (one per sub-shape).

    Parameters
    ----------
    coords_list : list of GridCoords or None
        The coordinate sets for each sub-shape.  ``None`` creates an empty
        container.
    name : str
        Name of the parent grob.
    rule : str or None
        Fill rule (``"winding"`` or ``"evenodd"``), mirroring R's
        ``attr(x, "rule")``.
    """

    def __init__(
        self,
        coords_list: Optional[List[GridCoords]] = None,
        name: str = "grobcoords",
        rule: Optional[str] = None,
    ) -> None:
        self._coords: List[GridCoords] = list(coords_list) if coords_list else []
        self._name = name
        self._rule = rule

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of the parent grob."""
        return self._name

    @property
    def rule(self) -> Optional[str]:
        """Fill rule (``'winding'`` or ``'evenodd'``), or ``None``."""
        return self._rule

    # -- query methods ------------------------------------------------------

    def get_x(self, subset: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return concatenated x values across all (or selected) sub-shapes.

        Parameters
        ----------
        subset : sequence of int, optional
            Indices of sub-shapes to include.  ``None`` means all.

        Returns
        -------
        numpy.ndarray
        """
        items = self._coords if subset is None else [self._coords[i] for i in subset]
        if not items:
            return np.array([], dtype=float)
        return np.concatenate([c.get_x() for c in items])

    def get_y(self, subset: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return concatenated y values across all (or selected) sub-shapes.

        Parameters
        ----------
        subset : sequence of int, optional
            Indices of sub-shapes to include.  ``None`` means all.

        Returns
        -------
        numpy.ndarray
        """
        items = self._coords if subset is None else [self._coords[i] for i in subset]
        if not items:
            return np.array([], dtype=float)
        return np.concatenate([c.get_y() for c in items])

    # -- transformation methods ---------------------------------------------

    def to_device(self, state: Any = None) -> "GridGrobCoords":
        """Convert all sub-shape coordinates to device space.

        Parameters
        ----------
        state : object, optional
            Passed through to :meth:`GridCoords.to_device`.

        Returns
        -------
        GridGrobCoords
        """
        return GridGrobCoords(
            [c.to_device(state) for c in self._coords],
            name=self._name,
            rule=self._rule,
        )

    def from_device(self, state: Any = None) -> "GridGrobCoords":
        """Transform all sub-shape coordinates from device space.

        Parameters
        ----------
        state : object, optional
            Passed through to :meth:`GridCoords.from_device`.

        Returns
        -------
        GridGrobCoords
        """
        return GridGrobCoords(
            [c.from_device(state) for c in self._coords],
            name=self._name,
            rule=self._rule,
        )

    def is_empty(self) -> bool:
        """Return ``True`` when every sub-shape is empty.

        Returns
        -------
        bool
        """
        if not self._coords:
            return True
        return all(c.is_empty() for c in self._coords)

    def transform_coords(self, tm: np.ndarray) -> "GridGrobCoords":
        """Apply an affine transformation to all sub-shapes.

        Parameters
        ----------
        tm : numpy.ndarray
            A 3x3 affine transformation matrix.

        Returns
        -------
        GridGrobCoords
        """
        return GridGrobCoords(
            [c.transform_coords(tm) for c in self._coords],
            name=self._name,
            rule=self._rule,
        )

    def flatten(self) -> "GridGrobCoords":
        """Return a flattened copy.

        Returns
        -------
        GridGrobCoords
        """
        return GridGrobCoords(
            [c.flatten() for c in self._coords],
            name=self._name,
            rule=self._rule,
        )

    # -- dunder methods -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._coords)

    def __iter__(self) -> Iterator[GridCoords]:
        return iter(self._coords)

    def __getitem__(self, index: int) -> GridCoords:
        return self._coords[index]

    def __repr__(self) -> str:
        lines: List[str] = []
        rule_str = f" (fill: {self._rule})" if self._rule else ""
        lines.append(f"grob {self._name}{rule_str}")
        for i, coord in enumerate(self._coords):
            label = str(i + 1)
            lines.append(f"{_COORD_PRINT_INDENT}shape {label}")
            for sub_line in repr(coord).split("\n"):
                lines.append(f"{_COORD_PRINT_INDENT}{_COORD_PRINT_INDENT}{sub_line}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GridGTreeCoords -- coordinates for a gTree (dict of child coords)
# ---------------------------------------------------------------------------


class GridGTreeCoords:
    """Container for coordinates of a gTree.

    Wraps a mapping from child names to their coordinate containers (which
    may themselves be :class:`GridGrobCoords` or :class:`GridGTreeCoords`).

    Parameters
    ----------
    coords_dict : dict or list or None
        Mapping of child names to coordinate containers, **or** a list of
        coordinate containers (keys are auto-generated).  ``None`` creates
        an empty container.
    name : str
        Name of the parent gTree.
    """

    def __init__(
        self,
        coords_dict: Union[
            Dict[str, Union[GridGrobCoords, "GridGTreeCoords"]],
            List[Union[GridGrobCoords, "GridGTreeCoords"]],
            None,
        ] = None,
        name: str = "gtreecoords",
    ) -> None:
        if coords_dict is None:
            self._children: Dict[str, Union[GridGrobCoords, GridGTreeCoords]] = {}
        elif isinstance(coords_dict, dict):
            self._children = dict(coords_dict)
        else:
            # Accept a list -- derive keys from child names or indices
            self._children = {}
            for i, child in enumerate(coords_dict):
                key = getattr(child, "name", str(i))
                self._children[key] = child
        self._name = name

    # -- properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of the parent gTree."""
        return self._name

    # -- query methods ------------------------------------------------------

    def get_x(self) -> np.ndarray:
        """Return concatenated x values across all children.

        Returns
        -------
        numpy.ndarray
        """
        if not self._children:
            return np.array([], dtype=float)
        return np.concatenate([c.get_x() for c in self._children.values()])

    def get_y(self) -> np.ndarray:
        """Return concatenated y values across all children.

        Returns
        -------
        numpy.ndarray
        """
        if not self._children:
            return np.array([], dtype=float)
        return np.concatenate([c.get_y() for c in self._children.values()])

    # -- transformation methods ---------------------------------------------

    def to_device(self, state: Any = None) -> "GridGTreeCoords":
        """Convert all child coordinates to device space.

        Parameters
        ----------
        state : object, optional
            Passed through to child ``to_device`` methods.

        Returns
        -------
        GridGTreeCoords
        """
        return GridGTreeCoords(
            {k: v.to_device(state) for k, v in self._children.items()},
            name=self._name,
        )

    def from_device(self, state: Any = None) -> "GridGTreeCoords":
        """Transform all child coordinates from device space.

        Parameters
        ----------
        state : object, optional
            Passed through to child ``from_device`` methods.

        Returns
        -------
        GridGTreeCoords
        """
        return GridGTreeCoords(
            {k: v.from_device(state) for k, v in self._children.items()},
            name=self._name,
        )

    def is_empty(self) -> bool:
        """Return ``True`` when every child reports empty.

        Returns
        -------
        bool
        """
        if not self._children:
            return True
        return all(c.is_empty() for c in self._children.values())

    def transform_coords(self, tm: np.ndarray) -> "GridGTreeCoords":
        """Apply an affine transformation to all children.

        Parameters
        ----------
        tm : numpy.ndarray
            A 3x3 affine transformation matrix.

        Returns
        -------
        GridGTreeCoords
        """
        return GridGTreeCoords(
            {k: v.transform_coords(tm) for k, v in self._children.items()},
            name=self._name,
        )

    def flatten(self) -> "GridGTreeCoords":
        """Return a flattened copy.

        Returns
        -------
        GridGTreeCoords
        """
        return GridGTreeCoords(
            {k: v.flatten() for k, v in self._children.items()},
            name=self._name,
        )

    # -- dunder methods -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._children)

    def __iter__(self) -> Iterator[str]:
        return iter(self._children)

    def __getitem__(self, key: str) -> Union[GridGrobCoords, "GridGTreeCoords"]:
        return self._children[key]

    def __repr__(self) -> str:
        lines: List[str] = []
        lines.append(f"gTree {self._name}")
        for child in self._children.values():
            for sub_line in repr(child).split("\n"):
                lines.append(f"{_COORD_PRINT_INDENT}{sub_line}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory / convenience functions
# ---------------------------------------------------------------------------


def grid_coords(x: ArrayLike, y: ArrayLike) -> GridCoords:
    """Create a :class:`GridCoords` instance.

    This is the public factory function mirroring R's ``gridCoords()``.

    Parameters
    ----------
    x : array_like
        X coordinates.
    y : array_like
        Y coordinates.

    Returns
    -------
    GridCoords
    """
    return GridCoords(x, y)


def grid_grob_coords(
    coords_list: List[GridCoords],
    name: str,
    rule: Optional[str] = None,
) -> GridGrobCoords:
    """Create a :class:`GridGrobCoords` instance.

    Parameters
    ----------
    coords_list : list of GridCoords
        Coordinate sets for each sub-shape.
    name : str
        Name of the owning grob.
    rule : str or None
        Fill rule.

    Returns
    -------
    GridGrobCoords
    """
    return GridGrobCoords(coords_list, name=name, rule=rule)


def grid_gtree_coords(
    coords_dict: Union[
        Dict[str, Union[GridGrobCoords, GridGTreeCoords]],
        List[Union[GridGrobCoords, GridGTreeCoords]],
    ],
    name: str,
) -> GridGTreeCoords:
    """Create a :class:`GridGTreeCoords` instance.

    Parameters
    ----------
    coords_dict : dict or list
        Mapping (or list) of child coordinate containers.
    name : str
        Name of the owning gTree.

    Returns
    -------
    GridGTreeCoords
    """
    return GridGTreeCoords(coords_dict, name=name)


# ---------------------------------------------------------------------------
# Canonical empty objects
# ---------------------------------------------------------------------------

#: The canonical "empty" coordinate pair (single (0, 0) point), mirroring
#: R's ``emptyCoords``.
_EMPTY_COORDS: GridCoords = GridCoords(np.array([0.0]), np.array([0.0]), name="empty")


def empty_coords() -> GridCoords:
    """Return the canonical empty :class:`GridCoords`.

    Returns
    -------
    GridCoords
        A coordinate set with a single ``(0, 0)`` point.
    """
    return _EMPTY_COORDS


def empty_grob_coords(name: str = "empty") -> GridGrobCoords:
    """Return an empty :class:`GridGrobCoords`.

    Parameters
    ----------
    name : str
        Label for the owning grob.

    Returns
    -------
    GridGrobCoords
    """
    return GridGrobCoords([_EMPTY_COORDS], name=name)


def empty_gtree_coords(name: str = "empty") -> GridGTreeCoords:
    """Return an empty :class:`GridGTreeCoords`.

    Parameters
    ----------
    name : str
        Label for the owning gTree.

    Returns
    -------
    GridGTreeCoords
    """
    return GridGTreeCoords([empty_grob_coords("0")], name=name)


def is_empty_coords(x: Union[GridCoords, GridGrobCoords, GridGTreeCoords]) -> bool:
    """Test whether a coordinate container is the canonical empty set.

    Dispatches to the ``is_empty()`` method of the container.

    Parameters
    ----------
    x : GridCoords or GridGrobCoords or GridGTreeCoords
        Coordinate container to test.

    Returns
    -------
    bool
    """
    return x.is_empty()


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def coords_bbox(
    x: Union[GridCoords, GridGrobCoords, GridGTreeCoords],
    subset: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Compute the axis-aligned bounding box of a coordinate set.

    Parameters
    ----------
    x : GridCoords or GridGrobCoords or GridGTreeCoords
        Coordinate container.
    subset : sequence of int, optional
        Passed to :meth:`GridGrobCoords.get_x` / ``get_y`` when applicable.

    Returns
    -------
    dict
        Keys ``'left'``, ``'bottom'``, ``'width'``, ``'height'``.
    """
    if isinstance(x, GridGrobCoords) and subset is not None:
        xx = x.get_x(subset)
        yy = x.get_y(subset)
    else:
        xx = x.get_x()
        yy = x.get_y()
    return {
        "left": float(np.min(xx)),
        "bottom": float(np.min(yy)),
        "width": float(np.ptp(xx)),
        "height": float(np.ptp(yy)),
    }


# ---------------------------------------------------------------------------
# isClosed dispatch
# ---------------------------------------------------------------------------

#: Grob class tags that are considered open (not closed).
_OPEN_GROB_CLASSES: frozenset[str] = frozenset(
    {
        "move.to",
        "line.to",
        "lines",
        "polyline",
        "segments",
        "beziergrob",
    }
)


def is_closed(x: Any) -> bool:
    """Determine the default ``closed`` value for a grob.

    Mirrors R's ``isClosed()`` generic.  Most grobs default to ``True``;
    line-like grobs default to ``False``.

    Parameters
    ----------
    x : Grob
        A graphical object.

    Returns
    -------
    bool
    """
    grid_class = getattr(x, "_grid_class", None) or ""
    if grid_class in _OPEN_GROB_CLASSES:
        return False
    # Special handling for xspline
    if grid_class == "xspline":
        return not getattr(x, "open_", True)
    # Special handling for points
    if grid_class == "points":
        pch = getattr(x, "pch", None)
        if pch in (3, 4, 8):
            return False
        return True
    return True


# ---------------------------------------------------------------------------
# grobCoords / grobPoints dispatchers
# ---------------------------------------------------------------------------


def grob_coords(x: Any, closed: Optional[bool] = None) -> Union[GridGrobCoords, GridGTreeCoords]:
    """Get the coordinates of a grob, performing drawing set-up.

    This is the user-level function that mirrors R's ``grobCoords()``.
    It dispatches based on the grob type: :class:`GList`, :class:`GTree`,
    and plain :class:`Grob` are all handled.

    Parameters
    ----------
    x : Grob or GTree or GList
        A graphical object.
    closed : bool, optional
        Whether to compute closed-shape coordinates.  Defaults to the
        result of :func:`is_closed`.

    Returns
    -------
    GridGrobCoords or GridGTreeCoords
        The computed coordinates.
    """
    if closed is None:
        closed = is_closed(x)

    if isinstance(x, GList):
        return _grob_coords_glist(x, closed)
    if isinstance(x, GTree):
        return _grob_coords_gtree(x, closed)
    if isinstance(x, Grob):
        # Check for custom grob_coords method first
        if hasattr(x, "grob_coords"):
            result = x.grob_coords(closed=closed)
            if result is not None:
                return result
        return _grob_coords_grob(x, closed)

    raise TypeError(f"grob_coords does not support {type(x).__name__}")


def _grob_coords_grob(x: Grob, closed: bool) -> GridGrobCoords:
    """Compute coordinates for a plain grob.

    Port of R ``grobCoords.grob`` (coords.R:272-304).
    Does full drawing setup:
      1. Save state (DL, gpar)
      2. preDraw (makeContext + push vp/gp)
      3. makeContent
      4. Call grobPoints to get coordinates
      5. If vp changed: toDevice → postDraw → fromDevice
      6. Otherwise: postDraw
    """
    import copy
    from ._state import get_state

    state = get_state()
    renderer = state.get_renderer()

    # Fallback when no renderer is available (no drawing context)
    if renderer is None:
        return grob_points(x, closed)

    # Save current transform for fromDevice (R coords.R:274)
    cur_transform = None
    if renderer is not None and hasattr(renderer, "_vp_transform_stack"):
        cur_transform = renderer._vp_transform_stack[-1].transform.copy()

    original_vp = x.vp

    # Same setup as drawGrob (R coords.R:276-284)
    saved_dl_on = state._dl_on
    state.set_display_list_on(False)
    saved_gpar = copy.copy(state.get_gpar())

    try:
        # preDraw: makeContext + push vp/gp + preDrawDetails
        x = x.make_context()
        from ._draw import _push_vp_gp
        _push_vp_gp(x)
        x.pre_draw_details()

        # makeContent
        x = x.make_content()

        # Check if vp changed (R coords.R:287)
        vpgrob = (x.vp is not None) or (original_vp is not x.vp)

        # Get points (R coords.R:290)
        pts = grob_points(x, closed)

        if vpgrob and not pts.is_empty():
            # Transform to device coordinates (R coords.R:292-294)
            pts = _to_device_grob_coords(pts, state)

        # postDraw: postDrawDetails + pop vp
        x.post_draw_details()
        if x.vp is not None:
            from ._draw import _pop_grob_vp
            _pop_grob_vp(x.vp)

        if vpgrob and not pts.is_empty() and cur_transform is not None:
            # Transform back from device (R coords.R:299-301)
            pts = _from_device_grob_coords(pts, cur_transform)

    finally:
        state.replace_gpar(saved_gpar)
        state.set_display_list_on(saved_dl_on)

    return pts


def _grob_coords_glist(x: GList, closed: bool) -> GridGTreeCoords:
    """Compute coordinates for a GList (mirrors R's grobCoords.gList)."""
    from ._grob import grob_name as _grob_name

    children = [grob_coords(child, closed) for child in x]
    return GridGTreeCoords(children, name=_grob_name())


def _grob_coords_gtree(x: GTree, closed: bool) -> GridGTreeCoords:
    """Compute coordinates for a gTree.

    Port of R ``grobCoords.gTree`` (coords.R:312-346).
    Does full drawing setup like drawGTree, then recursively
    calls grobCoords on children.
    """
    import copy
    from ._state import get_state

    state = get_state()
    renderer = state.get_renderer()

    # Fallback when no renderer is available (no drawing context)
    if renderer is None:
        children_order = getattr(x, "_children_order", [])
        children = getattr(x, "_children", {})
        if children and len(children) > 0:
            ordered = [children[k] for k in children_order if k in children]
            pts = [grob_coords(child, closed) for child in ordered]
            return GridGTreeCoords(pts, name=x.name)
        return empty_gtree_coords(x.name)

    cur_transform = None
    if renderer is not None and hasattr(renderer, "_vp_transform_stack"):
        cur_transform = renderer._vp_transform_stack[-1].transform.copy()

    original_vp = x.vp

    # Same setup as drawGTree (R coords.R:316-322)
    saved_dl_on = state._dl_on
    state.set_display_list_on(False)
    saved_gpar = copy.copy(state.get_gpar())
    saved_current_grob = getattr(state, "_current_grob", None)

    try:
        # preDraw.gTree: makeContext + setCurrentGrob + push vp/gp + childrenvp
        x = x.make_context()
        state._current_grob = x

        from ._draw import _push_vp_gp
        _push_vp_gp(x)

        # Push children viewport if present (R grob.R:1792-1801)
        children_vp = getattr(x, "childrenvp", None) or getattr(x, "children_vp", None)
        if children_vp is not None:
            from ._viewport import push_viewport, up_viewport
            from ._draw import _vp_depth
            temp_gp = copy.copy(state.get_gpar())
            push_viewport(children_vp, recording=False)
            up_viewport(_vp_depth(children_vp), recording=False)
            state.set_gpar(temp_gp)

        x.pre_draw_details()

        # makeContent (R coords.R:327)
        x = x.make_content()

        # Check if vp changed (R coords.R:330)
        vpgrob = (x.vp is not None) or (original_vp is not x.vp)

        # Get children coords (R coords.R:332-334)
        children_order = x._children_order if hasattr(x, "_children_order") else []
        children = x._children if hasattr(x, "_children") else {}

        if children and len(children) > 0:
            ordered = [children[k] for k in children_order if k in children]
            child_pts = [grob_coords(child, closed) for child in ordered]
            pts = GridGTreeCoords(child_pts, name=x.name)
        else:
            pts = empty_gtree_coords(x.name)

        if vpgrob and not pts.is_empty():
            pts = _to_device_gtree_coords(pts, state)

        # postDraw
        x.post_draw_details()
        if x.vp is not None:
            from ._draw import _pop_grob_vp
            _pop_grob_vp(x.vp)

        if vpgrob and not pts.is_empty() and cur_transform is not None:
            pts = _from_device_gtree_coords(pts, cur_transform)

    finally:
        state.replace_gpar(saved_gpar)
        state._current_grob = saved_current_grob
        state.set_display_list_on(saved_dl_on)

    return pts


# ---------------------------------------------------------------------------
# toDevice / fromDevice helpers for grobCoords
# Port of R coords.R:157-195
# ---------------------------------------------------------------------------


def _to_device_grob_coords(
    pts: GridGrobCoords, state: Any
) -> GridGrobCoords:
    """Convert grob coordinates to device inches via deviceLoc."""
    from ._units import device_loc, Unit

    new_coords = []
    for coord in pts:
        if coord.is_empty():
            new_coords.append(coord)
            continue
        # R coords.R:163-165: deviceLoc(unit(x,"in"), unit(y,"in"), valueOnly=TRUE)
        result = device_loc(
            Unit(coord.x, "inches"),
            Unit(coord.y, "inches"),
            value_only=True,
        )
        new_coords.append(GridCoords(result["x"], result["y"], name=coord.name))
    return GridGrobCoords(new_coords, name=pts.name, rule=pts.rule)


def _from_device_grob_coords(
    pts: GridGrobCoords, transform: np.ndarray
) -> GridGrobCoords:
    """Transform grob coordinates from device space.

    R coords.R:182-184: cbind(x,y,1) %*% solve(trans)
    """
    inv = np.linalg.solve(transform, np.eye(3))
    new_coords = []
    for coord in pts:
        if coord.is_empty():
            new_coords.append(coord)
            continue
        pts_matrix = np.column_stack([coord.x, coord.y, np.ones(len(coord.x))])
        result = pts_matrix @ inv
        new_coords.append(GridCoords(result[:, 0], result[:, 1], name=coord.name))
    return GridGrobCoords(new_coords, name=pts.name, rule=pts.rule)


def _to_device_gtree_coords(
    pts: GridGTreeCoords, state: Any
) -> GridGTreeCoords:
    """Convert gTree coordinates to device inches."""
    new_children = {}
    for key, child in pts._children.items():
        if isinstance(child, GridGrobCoords):
            new_children[key] = _to_device_grob_coords(child, state)
        elif isinstance(child, GridGTreeCoords):
            new_children[key] = _to_device_gtree_coords(child, state)
        else:
            new_children[key] = child
    return GridGTreeCoords(new_children, name=pts.name)


def _from_device_gtree_coords(
    pts: GridGTreeCoords, transform: np.ndarray
) -> GridGTreeCoords:
    """Transform gTree coordinates from device space."""
    new_children = {}
    for key, child in pts._children.items():
        if isinstance(child, GridGrobCoords):
            new_children[key] = _from_device_grob_coords(child, transform)
        elif isinstance(child, GridGTreeCoords):
            new_children[key] = _from_device_gtree_coords(child, transform)
        else:
            new_children[key] = child
    return GridGTreeCoords(new_children, name=pts.name)


def grob_points(x: Any, closed: Optional[bool] = None) -> GridGrobCoords:
    """Get boundary points of a grob without drawing set-up.

    This is for internal use when the drawing context is already
    established.  Mirrors R's ``grobPoints()``.

    Parameters
    ----------
    x : Grob or GTree or GList
        A graphical object.
    closed : bool, optional
        Whether to compute closed-shape coordinates.  Defaults to the
        result of :func:`is_closed`.

    Returns
    -------
    GridGrobCoords or GridGTreeCoords
        The computed boundary points.
    """
    if closed is None:
        closed = is_closed(x)

    if isinstance(x, GList):
        return _grob_points_glist(x, closed)
    if isinstance(x, GTree):
        return _grob_points_gtree(x, closed)

    # Dispatch by _grid_class to per-primitive implementations
    # (port of R coords.R:356-673)
    grid_class = getattr(x, "_grid_class", None) or ""
    dispatch_fn = _GROB_POINTS_DISPATCH.get(grid_class)
    if dispatch_fn is not None:
        return dispatch_fn(x, closed)

    # Allow grobs to provide their own custom implementation
    if hasattr(x, "grob_points"):
        result = x.grob_points(closed=closed)
        if result is not None:
            return result

    # Default: return empty
    name = getattr(x, "name", "unknown")
    return empty_grob_coords(name)


def _grob_points_glist(x: GList, closed: bool) -> GridGTreeCoords:
    """Compute points for a GList (mirrors R's grobPoints.gList)."""
    from ._grob import grob_name as _grob_name

    if len(x) > 0:
        return GridGTreeCoords(
            [grob_coords(child, closed) for child in x],
            name=_grob_name(),
        )
    return empty_gtree_coords(_grob_name())


def _grob_points_gtree(x: GTree, closed: bool) -> GridGTreeCoords:
    """Compute points for a gTree (mirrors R's grobPoints.gTree)."""
    children_order = getattr(x, "children_order", None)
    children = getattr(x, "children", None)

    if children is not None and len(children) > 0:
        if children_order is not None:
            ordered = [children[k] for k in children_order]
        else:
            ordered = list(children)
        pts = [grob_coords(child, closed) for child in ordered]
        return GridGTreeCoords(pts, name=x.name)
    return empty_gtree_coords(x.name)


# ---------------------------------------------------------------------------
# Per-primitive grobPoints implementations
# Port of R coords.R:356-673
# ---------------------------------------------------------------------------


def _grob_points_circle(x: Grob, closed: bool = True, n: int = 100) -> GridGrobCoords:
    """grobPoints.circle -- R coords.R:368-388.

    Returns n points around each circle perimeter (closed=True)
    or empty (closed=False).
    """
    if not closed:
        return empty_grob_coords(x.name)

    from ._units import convert_x, convert_y, convert_width, convert_height

    cx = convert_x(x.x, "inches", valueOnly=True)
    cy = convert_y(x.y, "inches", valueOnly=True)
    # R: r = pmin(convertWidth(r), convertHeight(r))
    rw = convert_width(x.r, "inches", valueOnly=True)
    rh = convert_height(x.r, "inches", valueOnly=True)
    r = np.minimum(rw, rh)

    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]

    # Recycle via broadcasting (R: cbind(cx, cy, r))
    data = np.column_stack([
        np.broadcast_to(cx, max(len(cx), len(cy), len(r))),
        np.broadcast_to(cy, max(len(cx), len(cy), len(r))),
        np.broadcast_to(r, max(len(cx), len(cy), len(r))),
    ])
    ncirc = data.shape[0]
    pts = []
    for i in range(ncirc):
        px = data[i, 0] + data[i, 2] * np.cos(t)
        py = data[i, 1] + data[i, 2] * np.sin(t)
        pts.append(GridCoords(px, py, name=str(i + 1)))
    return GridGrobCoords(pts, name=x.name)


def _grob_points_rect(x: Grob, closed: bool = True) -> GridGrobCoords:
    """grobPoints.rect -- R coords.R:529-547.

    Returns 4 corners of each rect (closed=True) or empty.
    """
    if not closed:
        return empty_grob_coords(x.name)

    from ._just import resolve_hjust, resolve_vjust
    from ._units import convert_x, convert_y, convert_width, convert_height

    just = getattr(x, "just", None) or "centre"
    hjust = resolve_hjust(just, getattr(x, "hjust", None))
    vjust = resolve_vjust(just, getattr(x, "vjust", None))

    w = convert_width(x.width, "inches", valueOnly=True)
    h = convert_height(x.height, "inches", valueOnly=True)
    xc = convert_x(x.x, "inches", valueOnly=True)
    yc = convert_y(x.y, "inches", valueOnly=True)

    left = xc - hjust * w
    bottom = yc - vjust * h
    right = left + w
    top = bottom + h

    # Recycle via column_stack (R: cbind(left, right, bottom, top))
    rects = np.column_stack([
        np.broadcast_to(left, max(len(left), len(right), len(bottom), len(top))),
        np.broadcast_to(right, max(len(left), len(right), len(bottom), len(top))),
        np.broadcast_to(bottom, max(len(left), len(right), len(bottom), len(top))),
        np.broadcast_to(top, max(len(left), len(right), len(bottom), len(top))),
    ])
    pts = []
    for i in range(rects.shape[0]):
        # R: xyListFromMatrix(rects, c(1,1,2,2), c(3,4,4,3))
        # Corners: (left,bottom), (left,top), (right,top), (right,bottom)
        px = np.array([rects[i, 0], rects[i, 0], rects[i, 1], rects[i, 1]])
        py = np.array([rects[i, 2], rects[i, 3], rects[i, 3], rects[i, 2]])
        pts.append(GridCoords(px, py, name=str(i + 1)))
    return GridGrobCoords(pts, name=x.name)


def _grob_points_lines(x: Grob, closed: bool = False) -> GridGrobCoords:
    """grobPoints.lines -- R coords.R:390-400."""
    if closed:
        return empty_grob_coords(x.name)

    from ._units import convert_x, convert_y

    xx = convert_x(x.x, "inches", valueOnly=True)
    yy = convert_y(x.y, "inches", valueOnly=True)
    # Recycle via column_stack
    lines = np.column_stack([
        np.broadcast_to(xx, max(len(xx), len(yy))),
        np.broadcast_to(yy, max(len(xx), len(yy))),
    ])
    return GridGrobCoords(
        [GridCoords(lines[:, 0], lines[:, 1], name="1")],
        name=x.name,
    )


def _grob_points_polyline(x: Grob, closed: bool = False) -> GridGrobCoords:
    """grobPoints.polyline -- R coords.R:402-429."""
    if closed:
        return empty_grob_coords(x.name)

    from ._units import convert_x, convert_y

    xx = convert_x(x.x, "inches", valueOnly=True)
    yy = convert_y(x.y, "inches", valueOnly=True)

    grob_id = getattr(x, "id", None)
    grob_id_lengths = getattr(x, "id_lengths", None)

    if grob_id is None and grob_id_lengths is None:
        return GridGrobCoords(
            [GridCoords(xx, yy, name="1")], name=x.name,
        )

    if grob_id is None:
        n = len(grob_id_lengths)
        grob_id = np.repeat(np.arange(1, n + 1), grob_id_lengths)
    else:
        grob_id = np.asarray(grob_id)

    unique_ids = np.unique(grob_id)
    if len(unique_ids) > 1:
        pts = []
        for uid in unique_ids:
            mask = grob_id == uid
            pts.append(GridCoords(xx[mask], yy[mask], name=str(uid)))
        return GridGrobCoords(pts, name=x.name)
    return GridGrobCoords(
        [GridCoords(xx, yy, name="1")], name=x.name,
    )


def _grob_points_polygon(x: Grob, closed: bool = True) -> GridGrobCoords:
    """grobPoints.polygon -- R coords.R:437-464."""
    if not closed:
        return empty_grob_coords(x.name)

    from ._units import convert_x, convert_y

    xx = convert_x(x.x, "inches", valueOnly=True)
    yy = convert_y(x.y, "inches", valueOnly=True)

    grob_id = getattr(x, "id", None)
    grob_id_lengths = getattr(x, "id_lengths", None)

    if grob_id is None and grob_id_lengths is None:
        return GridGrobCoords(
            [GridCoords(xx, yy, name="1")], name=x.name,
        )

    if grob_id is None:
        n = len(grob_id_lengths)
        grob_id = np.repeat(np.arange(1, n + 1), grob_id_lengths)
    else:
        grob_id = np.asarray(grob_id)

    unique_ids = np.unique(grob_id)
    if len(unique_ids) > 1:
        pts = []
        for uid in unique_ids:
            mask = grob_id == uid
            pts.append(GridCoords(xx[mask], yy[mask], name=str(uid)))
        return GridGrobCoords(pts, name=x.name)
    return GridGrobCoords(
        [GridCoords(xx, yy, name="1")], name=x.name,
    )


def _grob_points_segments(x: Grob, closed: bool = False) -> GridGrobCoords:
    """grobPoints.segments -- R coords.R:549-563."""
    if closed:
        return empty_grob_coords(x.name)

    from ._units import convert_x, convert_y

    x0 = convert_x(x.x0, "inches", valueOnly=True)
    x1 = convert_x(x.x1, "inches", valueOnly=True)
    y0 = convert_y(x.y0, "inches", valueOnly=True)
    y1 = convert_y(x.y1, "inches", valueOnly=True)

    # Recycle via column_stack (R: cbind(x0, x1, y0, y1))
    maxlen = max(len(x0), len(x1), len(y0), len(y1))
    xy = np.column_stack([
        np.broadcast_to(x0, maxlen),
        np.broadcast_to(x1, maxlen),
        np.broadcast_to(y0, maxlen),
        np.broadcast_to(y1, maxlen),
    ])
    pts = []
    for i in range(xy.shape[0]):
        # R: xyListFromMatrix(xy, 1:2, 3:4)
        px = np.array([xy[i, 0], xy[i, 1]])
        py = np.array([xy[i, 2], xy[i, 3]])
        pts.append(GridCoords(px, py, name=str(i + 1)))
    return GridGrobCoords(pts, name=x.name)


def _grob_points_pathgrob(x: Grob, closed: bool = True) -> GridGrobCoords:
    """grobPoints.pathgrob -- R coords.R:474-527.

    Handles pathId and id for multiple paths/shapes.
    """
    if not closed:
        return empty_grob_coords(x.name)

    from ._units import convert_x, convert_y

    xx = convert_x(x.x, "inches", valueOnly=True)
    yy = convert_y(x.y, "inches", valueOnly=True)
    rule = getattr(x, "rule", None)

    grob_id = getattr(x, "id", None)
    grob_id_lengths = getattr(x, "id_lengths", None)

    if grob_id is None and grob_id_lengths is None:
        return GridGrobCoords(
            [GridCoords(xx, yy, name="1")],
            name=x.name, rule=rule,
        )

    if grob_id is None:
        n = len(grob_id_lengths)
        grob_id = np.repeat(np.arange(1, n + 1), grob_id_lengths)
    else:
        grob_id = np.asarray(grob_id)

    unique_ids = np.unique(grob_id)
    pts = []
    for uid in unique_ids:
        mask = grob_id == uid
        pts.append(GridCoords(xx[mask], yy[mask], name=str(uid)))
    return GridGrobCoords(pts, name=x.name, rule=rule)


def _grob_points_text(x: Grob, closed: bool = True) -> GridGrobCoords:
    """grobPoints.text -- R coords.R:591-612.

    Returns bounding box (4 corners) if closed, empty otherwise.
    Uses string metrics to estimate the text bounds.
    """
    if not closed:
        return empty_grob_coords(x.name)

    from ._just import resolve_hjust, resolve_vjust
    from ._units import convert_x, convert_y, Unit
    from ._font_metrics import get_font_backend

    label = getattr(x, "label", "")
    if isinstance(label, (list, tuple)):
        label = label[0] if label else ""

    just = getattr(x, "just", None) or "centre"
    hjust = resolve_hjust(just, getattr(x, "hjust", None))
    vjust = resolve_vjust(just, getattr(x, "vjust", None))

    # Get text position in inches
    xpos = convert_x(x.x, "inches", valueOnly=True)
    ypos = convert_y(x.y, "inches", valueOnly=True)

    # Measure text
    backend = get_font_backend()
    gp = getattr(x, "gp", None)
    metric = backend.measure(str(label), gp)

    w = metric.get("width", 0.0)
    asc = metric.get("ascent", 0.0)
    desc = metric.get("descent", 0.0)
    h = asc + desc

    if w == 0 and h == 0:
        return empty_grob_coords(x.name)

    # Compute bounds based on justification
    left = float(xpos[0]) - hjust * w
    bottom = float(ypos[0]) - vjust * h
    right = left + w
    top = bottom + h

    return GridGrobCoords(
        [GridCoords(
            np.array([left, left, right, right]),
            np.array([bottom, top, top, bottom]),
            name="1",
        )],
        name=x.name,
    )


def _grob_points_xspline(x: Grob, closed: Optional[bool] = None) -> GridGrobCoords:
    """grobPoints.xspline -- R coords.R:565-586.

    Returns the traced spline curve via ``xsplinePoints`` (one point set
    per ``id`` group), exactly like R.
    """
    is_open = bool(getattr(x, "open_", True))
    if closed is None:
        closed = not is_open

    if (closed and not is_open) or (not closed and is_open):
        from ._curve import xspline_points
        trace = xspline_points(x)
        if isinstance(trace, dict):
            # Single X-spline
            return GridGrobCoords(
                [GridCoords(trace["x"], trace["y"], name="1")],
                name=x.name,
            )
        return GridGrobCoords(
            [GridCoords(t["x"], t["y"], name=str(i + 1))
             for i, t in enumerate(trace)],
            name=x.name,
        )
    return empty_grob_coords(x.name)


def _grob_points_rastergrob(x: Grob, closed: bool = True) -> GridGrobCoords:
    """grobPoints.rastergrob -- R coords.R:639-641. Always empty."""
    return empty_grob_coords(x.name)


def _grob_points_null(x: Grob, closed: bool = True) -> GridGrobCoords:
    """grobPoints.null -- R coords.R:647-649. Always empty."""
    return empty_grob_coords(x.name)


# ---------------------------------------------------------------------------
# Dispatcher: map _grid_class → grobPoints implementation
# ---------------------------------------------------------------------------

_GROB_POINTS_DISPATCH: Dict[str, Any] = {
    "circle": _grob_points_circle,
    "rect": _grob_points_rect,
    "lines": _grob_points_lines,
    "polyline": _grob_points_polyline,
    "polygon": _grob_points_polygon,
    "segments": _grob_points_segments,
    "pathgrob": _grob_points_pathgrob,
    "text": _grob_points_text,
    "xspline": _grob_points_xspline,
    "rastergrob": _grob_points_rastergrob,
    "null": _grob_points_null,
    "move.to": lambda x, closed=True: empty_grob_coords(x.name),
    "line.to": lambda x, closed=True: empty_grob_coords(x.name),
    "clip": lambda x, closed=True: empty_grob_coords(x.name),
}
