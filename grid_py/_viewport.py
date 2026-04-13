"""Viewport system for grid_py -- Python port of R's ``grid::viewport``.

This module provides the :class:`Viewport` class and associated container
classes (:class:`VpList`, :class:`VpStack`, :class:`VpTree`) that mirror
the viewport infrastructure in R's *grid* package.  It also exposes the
navigation functions (``push_viewport``, ``pop_viewport``, ``up_viewport``,
``down_viewport``, ``seek_viewport``) and query helpers
(``current_viewport``, ``current_vp_path``, etc.).

Viewports define nested rectangular sub-regions of the graphics device,
each with its own coordinate system, clipping behaviour, and graphical
parameter settings.

References
----------
R source: ``src/library/grid/R/viewport.R``, ``src/library/grid/R/grid.R``
"""

from __future__ import annotations

import copy
import math
import threading
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np

from ._gpar import Gpar
from ._just import valid_just
from ._layout import GridLayout
from ._path import VpPath
from ._units import Unit, is_unit

__all__ = [
    "Viewport",
    "VpList",
    "VpStack",
    "VpTree",
    "push_viewport",
    "pop_viewport",
    "down_viewport",
    "up_viewport",
    "seek_viewport",
    "current_viewport",
    "current_vp_path",
    "current_vp_tree",
    "current_transform",
    "current_rotation",
    "current_parent",
    "data_viewport",
    "plot_viewport",
    "edit_viewport",
    "show_viewport",
    "depth",
    "is_viewport",
]

# ---------------------------------------------------------------------------
# Module-level auto-name counter (thread-safe)
# ---------------------------------------------------------------------------

_vp_name_lock = threading.Lock()
_vp_name_index: int = 0


def _vp_auto_name() -> str:
    """Generate a unique viewport name of the form ``GRID.VP.<n>``.

    Returns
    -------
    str
        The generated name.

    Notes
    -----
    This mirrors R's ``vpAutoName()`` closure.  A module-level counter is
    used instead of a closure so that it can be reset for testing.  Access
    is serialised with a lock for thread safety.
    """
    global _vp_name_index
    with _vp_name_lock:
        _vp_name_index += 1
        return f"GRID.VP.{_vp_name_index}"


def _reset_vp_auto_name() -> None:
    """Reset the auto-name counter to zero (for testing)."""
    global _vp_name_index
    with _vp_name_lock:
        _vp_name_index = 0


# ---------------------------------------------------------------------------
# Clip value normalisation
# ---------------------------------------------------------------------------

_CLIP_MAP = {
    "on": True,
    "off": None,       # R uses NA; we use None
    "inherit": False,
}


def _valid_clip(clip: Any) -> Any:
    """Normalise *clip* to its internal representation.

    Parameters
    ----------
    clip : bool, str, or other
        ``"on"`` -> ``True``, ``"off"`` -> ``None``, ``"inherit"`` -> ``False``.
        Booleans and ``None`` pass through unchanged.

    Returns
    -------
    bool or None

    Raises
    ------
    ValueError
        If *clip* is a string that is not one of the accepted values.
    """
    if isinstance(clip, bool) or clip is None:
        return clip
    if isinstance(clip, str):
        val = _CLIP_MAP.get(clip.lower())
        if val is None and clip.lower() != "off":
            raise ValueError(
                f"invalid 'clip' value {clip!r}; "
                "must be 'on', 'off', 'inherit', or a boolean"
            )
        # "off" -> None is correct from the map
        return _CLIP_MAP.get(clip.lower(), None)
    raise ValueError(
        f"invalid 'clip' value {clip!r}; "
        "must be 'on', 'off', 'inherit', or a boolean"
    )


# ---------------------------------------------------------------------------
# Mask value normalisation
# ---------------------------------------------------------------------------

_MASK_MAP = {
    "inherit": True,
    "none": False,
}


def _valid_mask(mask: Any) -> Any:
    """Normalise *mask* to its internal representation.

    Parameters
    ----------
    mask : bool, str, or other
        ``"inherit"`` -> ``True``, ``"none"`` -> ``False``.
        Booleans pass through unchanged.  Other objects (e.g. mask grobs)
        are returned as-is.

    Returns
    -------
    bool or object

    Raises
    ------
    ValueError
        If *mask* is a string that is not one of the accepted values.
    """
    if isinstance(mask, bool):
        return mask
    if isinstance(mask, str):
        val = _MASK_MAP.get(mask.lower())
        if val is None:
            raise ValueError(
                f"invalid 'mask' value {mask!r}; "
                "must be 'inherit', 'none', or a boolean"
            )
        return val
    # Arbitrary mask objects (grobs, etc.) pass through
    return mask


# ---------------------------------------------------------------------------
# Viewport class
# ---------------------------------------------------------------------------


class Viewport:
    """A viewport specification -- a rectangular sub-region of a device.

    Parameters
    ----------
    x : Unit or float or None
        Horizontal position.  Defaults to ``Unit(0.5, "npc")``.
    y : Unit or float or None
        Vertical position.  Defaults to ``Unit(0.5, "npc")``.
    width : Unit or float or None
        Width.  Defaults to ``Unit(1, "npc")``.
    height : Unit or float or None
        Height.  Defaults to ``Unit(1, "npc")``.
    default_units : str
        Unit type applied when *x*, *y*, *width*, or *height* are given
        as plain numbers rather than :class:`Unit` objects.
    just : str or sequence
        Justification specification (see :func:`valid_just`).
    gp : Gpar or None
        Graphical parameter settings.
    clip : str or bool
        Clipping mode: ``"inherit"`` (default), ``"on"``, or ``"off"``.
    mask : bool or str
        Masking mode: ``"inherit"`` (default, mapped to ``True``),
        ``"none"`` (mapped to ``False``), or a mask grob.
    xscale : sequence of float or None
        Two-element ``[min, max]`` giving the native x-coordinate range.
        Defaults to ``[0, 1]``.
    yscale : sequence of float or None
        Two-element ``[min, max]`` giving the native y-coordinate range.
        Defaults to ``[0, 1]``.
    angle : float
        Rotation angle in degrees.
    layout : GridLayout or None
        Layout for arranging children of this viewport.
    layout_pos_row : int, sequence of int, or None
        Row position(s) of this viewport in the parent's layout.
    layout_pos_col : int, sequence of int, or None
        Column position(s) of this viewport in the parent's layout.
    name : str or None
        Viewport name.  Auto-generated (``"GRID.VP.<n>"``) if ``None``.

    Raises
    ------
    ValueError
        If any argument fails validation.
    TypeError
        If *gp* is not a :class:`Gpar` (or ``None``).

    Examples
    --------
    >>> vp = Viewport(width=Unit(0.8, "npc"), height=Unit(0.8, "npc"))
    >>> str(vp)
    'viewport[GRID.VP.1]'
    """

    # We store everything on the instance rather than using ``__slots__``
    # because pushed-viewport copies add additional runtime attributes
    # (``parentgpar``, ``trans``, ``children``, etc.) and slots would
    # prevent that.

    def __init__(
        self,
        x: Union[Unit, float, int, None] = None,
        y: Union[Unit, float, int, None] = None,
        width: Union[Unit, float, int, None] = None,
        height: Union[Unit, float, int, None] = None,
        default_units: str = "npc",
        just: Any = "centre",
        gp: Optional[Gpar] = None,
        clip: Any = "inherit",
        mask: Any = False,
        xscale: Optional[Sequence[float]] = None,
        yscale: Optional[Sequence[float]] = None,
        angle: float = 0,
        layout: Optional[GridLayout] = None,
        layout_pos_row: Optional[Union[int, Sequence[int]]] = None,
        layout_pos_col: Optional[Union[int, Sequence[int]]] = None,
        name: Optional[str] = None,
    ) -> None:
        # -- position / size defaults ----------------------------------------
        if x is None:
            x = Unit(0.5, "npc")
        if y is None:
            y = Unit(0.5, "npc")
        if width is None:
            width = Unit(1, "npc")
        if height is None:
            height = Unit(1, "npc")

        # Coerce plain numerics to Unit with *default_units*
        if not is_unit(x):
            x = Unit(x, default_units)
        if not is_unit(y):
            y = Unit(y, default_units)
        if not is_unit(width):
            width = Unit(width, default_units)
        if not is_unit(height):
            height = Unit(height, default_units)

        # -- validate scalar unit length -------------------------------------
        for arg_name, arg_val in [
            ("x", x), ("y", y), ("width", width), ("height", height),
        ]:
            if len(arg_val) != 1:
                raise ValueError(
                    f"'{arg_name}' must be a unit of length 1, "
                    f"got length {len(arg_val)}"
                )

        # -- gp ---------------------------------------------------------------
        if gp is None:
            gp = Gpar()
        if not isinstance(gp, Gpar):
            raise TypeError(
                f"invalid 'gp' value: expected Gpar, got {type(gp).__name__}"
            )

        # -- clip / mask ------------------------------------------------------
        clip = _valid_clip(clip)
        mask = _valid_mask(mask)

        # -- scales -----------------------------------------------------------
        if xscale is None:
            xscale = [0.0, 1.0]
        if yscale is None:
            yscale = [0.0, 1.0]

        xscale = [float(v) for v in xscale]
        yscale = [float(v) for v in yscale]

        if len(xscale) != 2 or not all(math.isfinite(v) for v in xscale):
            raise ValueError("invalid 'xscale' in viewport")
        if xscale[1] == xscale[0]:
            raise ValueError(
                "invalid 'xscale' in viewport: range must be non-zero"
            )

        if len(yscale) != 2 or not all(math.isfinite(v) for v in yscale):
            raise ValueError("invalid 'yscale' in viewport")
        if yscale[1] == yscale[0]:
            raise ValueError(
                "invalid 'yscale' in viewport: range must be non-zero"
            )

        # -- angle ------------------------------------------------------------
        angle = float(angle)
        if not math.isfinite(angle):
            raise ValueError("invalid 'angle' in viewport")

        # -- layout -----------------------------------------------------------
        if layout is not None and not isinstance(layout, GridLayout):
            raise ValueError("invalid 'layout' in viewport")

        # -- layout position ---------------------------------------------------
        if layout_pos_row is not None:
            if isinstance(layout_pos_row, (int, np.integer)):
                layout_pos_row = [int(layout_pos_row), int(layout_pos_row)]
            else:
                vals = [int(v) for v in layout_pos_row]
                layout_pos_row = [min(vals), max(vals)]
            if not all(math.isfinite(v) for v in layout_pos_row):
                raise ValueError("invalid 'layout_pos_row' in viewport")

        if layout_pos_col is not None:
            if isinstance(layout_pos_col, (int, np.integer)):
                layout_pos_col = [int(layout_pos_col), int(layout_pos_col)]
            else:
                vals = [int(v) for v in layout_pos_col]
                layout_pos_col = [min(vals), max(vals)]
            if not all(math.isfinite(v) for v in layout_pos_col):
                raise ValueError("invalid 'layout_pos_col' in viewport")

        # -- justification ----------------------------------------------------
        just_pair = valid_just(just)

        # -- name --------------------------------------------------------------
        if name is None:
            name = _vp_auto_name()
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"invalid viewport name: {name!r}"
            )

        # -- store validated fields -------------------------------------------
        self._x: Unit = x
        self._y: Unit = y
        self._width: Unit = width
        self._height: Unit = height
        self._default_units: str = default_units
        self._just: Tuple[float, float] = just_pair
        self._gp: Gpar = gp
        self._clip: Any = clip
        self._mask: Any = mask
        self._xscale: List[float] = xscale
        self._yscale: List[float] = yscale
        self._angle: float = angle
        self._layout: Optional[GridLayout] = layout
        self._layout_pos_row: Optional[List[int]] = layout_pos_row
        self._layout_pos_col: Optional[List[int]] = layout_pos_col
        self._name: str = name

        # -- pushed-viewport slots (filled in when the vp is pushed) ----------
        self.parentgpar: Optional[Gpar] = None
        self.gpar: Optional[Gpar] = None
        self.trans: Optional[np.ndarray] = None
        self.widths: Optional[Any] = None
        self.heights: Optional[Any] = None
        self.width_cm: Optional[float] = None
        self.height_cm: Optional[float] = None
        self.rotation: Optional[float] = None
        self.cliprect: Optional[Any] = None
        self.parent: Optional["Viewport"] = None
        self.children: Optional[dict] = None
        self.devwidth: Optional[float] = None
        self.devheight: Optional[float] = None
        self.clippath: Optional[Any] = None
        self.resolvedmask: Optional[Any] = None

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def x(self) -> Unit:
        """Horizontal position of the viewport.

        Returns
        -------
        Unit
        """
        return self._x

    @property
    def y(self) -> Unit:
        """Vertical position of the viewport.

        Returns
        -------
        Unit
        """
        return self._y

    @property
    def width(self) -> Unit:
        """Width of the viewport.

        Returns
        -------
        Unit
        """
        return self._width

    @property
    def height(self) -> Unit:
        """Height of the viewport.

        Returns
        -------
        Unit
        """
        return self._height

    @property
    def default_units(self) -> str:
        """Default unit type for numeric position/size arguments.

        Returns
        -------
        str
        """
        return self._default_units

    @property
    def just(self) -> Tuple[float, float]:
        """Justification as a ``(hjust, vjust)`` pair.

        Returns
        -------
        tuple of float
        """
        return self._just

    @property
    def gp(self) -> Gpar:
        """Graphical parameters associated with this viewport.

        Returns
        -------
        Gpar
        """
        return self._gp

    @property
    def clip(self) -> Any:
        """Clipping mode.

        Returns
        -------
        bool or None
            ``True`` for ``"on"``, ``None`` for ``"off"``, ``False`` for
            ``"inherit"``.
        """
        return self._clip

    @property
    def mask(self) -> Any:
        """Masking mode or mask object.

        Returns
        -------
        bool or object
        """
        return self._mask

    @property
    def xscale(self) -> List[float]:
        """Native x-coordinate range ``[min, max]``.

        Returns
        -------
        list of float
        """
        return list(self._xscale)

    @property
    def yscale(self) -> List[float]:
        """Native y-coordinate range ``[min, max]``.

        Returns
        -------
        list of float
        """
        return list(self._yscale)

    @property
    def angle(self) -> float:
        """Rotation angle in degrees.

        Returns
        -------
        float
        """
        return self._angle

    @property
    def layout(self) -> Optional[GridLayout]:
        """Layout for child arrangement, or ``None``.

        Returns
        -------
        GridLayout or None
        """
        return self._layout

    @property
    def layout_pos_row(self) -> Optional[List[int]]:
        """Row position(s) in parent layout, or ``None``.

        Returns
        -------
        list of int or None
        """
        return self._layout_pos_row

    @property
    def layout_pos_col(self) -> Optional[List[int]]:
        """Column position(s) in parent layout, or ``None``.

        Returns
        -------
        list of int or None
        """
        return self._layout_pos_col

    @property
    def name(self) -> str:
        """Name of the viewport.

        Returns
        -------
        str
        """
        return self._name

    # -----------------------------------------------------------------------
    # String representations
    # -----------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a short description (mirrors R's ``as.character.viewport``).

        Returns
        -------
        str
        """
        return f"viewport[{self._name}]"

    def __repr__(self) -> str:
        """Return a detailed description for debugging.

        Returns
        -------
        str
        """
        parts = [
            f"name={self._name!r}",
            f"x={self._x!r}",
            f"y={self._y!r}",
            f"width={self._width!r}",
            f"height={self._height!r}",
            f"just={self._just!r}",
            f"xscale={self._xscale!r}",
            f"yscale={self._yscale!r}",
            f"angle={self._angle!r}",
        ]
        return f"Viewport({', '.join(parts)})"

    # -----------------------------------------------------------------------
    # Copying
    # -----------------------------------------------------------------------

    def _copy(self) -> "Viewport":
        """Return a shallow copy of this viewport.

        Returns
        -------
        Viewport
        """
        return copy.copy(self)


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------


def is_viewport(obj: Any) -> bool:
    """Return ``True`` if *obj* is a :class:`Viewport` (or subclass).

    Parameters
    ----------
    obj : object
        Object to test.

    Returns
    -------
    bool
    """
    return isinstance(obj, Viewport)


def _viewport_or_path(obj: Any) -> bool:
    """Return ``True`` if *obj* is a Viewport or a VpPath."""
    return isinstance(obj, (Viewport, VpPath))


# ---------------------------------------------------------------------------
# VpList -- parallel push
# ---------------------------------------------------------------------------


class VpList:
    """A list of viewports to be pushed in parallel.

    When pushed, all viewports in the list are pushed as siblings of the
    current viewport.  For all but the last element the navigation returns
    to the common parent before pushing the next viewport; the final
    element's viewport becomes the current viewport after the push.

    Parameters
    ----------
    *vps : Viewport or VpPath
        One or more viewports (or viewport paths).

    Raises
    ------
    TypeError
        If any element is not a :class:`Viewport` or :class:`VpPath`.

    Examples
    --------
    >>> vl = VpList(Viewport(name="a"), Viewport(name="b"))
    >>> len(vl)
    2
    """

    def __init__(self, *vps: Union[Viewport, VpPath]) -> None:
        for v in vps:
            if not _viewport_or_path(v):
                raise TypeError(
                    f"only viewports allowed in VpList, got {type(v).__name__}"
                )
        self._vps: Tuple[Union[Viewport, VpPath], ...] = tuple(vps)

    # -- container protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._vps)

    def __getitem__(self, index: int) -> Union[Viewport, VpPath]:
        return self._vps[index]

    def __iter__(self) -> Iterator[Union[Viewport, VpPath]]:
        return iter(self._vps)

    # -- string representation -----------------------------------------------

    def __str__(self) -> str:
        """Mirrors R's ``as.character.vpList``.

        Returns
        -------
        str
        """
        inner = ", ".join(str(v) for v in self._vps)
        return f"({inner})"

    def __repr__(self) -> str:
        inner = ", ".join(repr(v) for v in self._vps)
        return f"VpList({inner})"


# ---------------------------------------------------------------------------
# VpStack -- sequential (nested) push
# ---------------------------------------------------------------------------


class VpStack:
    """A stack of viewports to be pushed sequentially (nested).

    Each viewport in the stack is pushed inside the preceding one, producing
    a chain of nested viewports.

    Parameters
    ----------
    *vps : Viewport or VpPath
        One or more viewports (or viewport paths).

    Raises
    ------
    TypeError
        If any element is not a :class:`Viewport` or :class:`VpPath`.

    Examples
    --------
    >>> vs = VpStack(Viewport(name="outer"), Viewport(name="inner"))
    >>> str(vs)
    'viewport[outer]->viewport[inner]'
    """

    def __init__(self, *vps: Union[Viewport, VpPath]) -> None:
        for v in vps:
            if not _viewport_or_path(v):
                raise TypeError(
                    f"only viewports allowed in VpStack, got {type(v).__name__}"
                )
        self._vps: Tuple[Union[Viewport, VpPath], ...] = tuple(vps)

    # -- container protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._vps)

    def __getitem__(self, index: int) -> Union[Viewport, VpPath]:
        return self._vps[index]

    def __iter__(self) -> Iterator[Union[Viewport, VpPath]]:
        return iter(self._vps)

    # -- string representation -----------------------------------------------

    def __str__(self) -> str:
        """Mirrors R's ``as.character.vpStack``.

        Returns
        -------
        str
        """
        return "->".join(str(v) for v in self._vps)

    def __repr__(self) -> str:
        inner = ", ".join(repr(v) for v in self._vps)
        return f"VpStack({inner})"


# ---------------------------------------------------------------------------
# VpTree -- parent + children (VpList)
# ---------------------------------------------------------------------------


class VpTree:
    """A viewport tree consisting of a parent viewport and a list of children.

    When pushed, the *parent* viewport is pushed first, then the *children*
    (a :class:`VpList`) are pushed inside it.

    Parameters
    ----------
    parent : Viewport
        The parent viewport.
    children : VpList
        The children to push inside *parent*.

    Raises
    ------
    TypeError
        If *parent* is not a Viewport/VpPath or *children* is not a
        :class:`VpList`.

    Examples
    --------
    >>> tree = VpTree(Viewport(name="p"), VpList(Viewport(name="c1")))
    >>> str(tree)
    'viewport[p]->(viewport[c1])'
    """

    def __init__(
        self,
        parent: Union[Viewport, VpPath],
        children: VpList,
    ) -> None:
        if not _viewport_or_path(parent):
            raise TypeError(
                "'parent' must be a Viewport or VpPath in VpTree"
            )
        if not isinstance(children, VpList):
            raise TypeError(
                "'children' must be a VpList in VpTree"
            )
        self._parent = parent
        self._children = children

    @property
    def parent(self) -> Union[Viewport, VpPath]:
        """Parent viewport.

        Returns
        -------
        Viewport or VpPath
        """
        return self._parent

    @property
    def children(self) -> VpList:
        """Child viewport list.

        Returns
        -------
        VpList
        """
        return self._children

    # -- string representation -----------------------------------------------

    def __str__(self) -> str:
        """Mirrors R's ``as.character.vpTree``.

        Returns
        -------
        str
        """
        return f"{self._parent}->{self._children}"

    def __repr__(self) -> str:
        return f"VpTree(parent={self._parent!r}, children={self._children!r})"


# ---------------------------------------------------------------------------
# depth() generic
# ---------------------------------------------------------------------------


def depth(x: Any) -> int:
    """Return the depth (number of nesting levels) of a viewport object.

    Parameters
    ----------
    x : Viewport, VpList, VpStack, VpTree, or VpPath
        The viewport object whose depth to compute.

    Returns
    -------
    int
        Number of levels.

    Raises
    ------
    TypeError
        If *x* is not a recognised viewport type.

    Notes
    -----
    This mirrors R's ``depth()`` generic.

    * ``Viewport``: always 1.
    * ``VpList``: depth of the *last* element (since pushing a list leaves
      you wherever the last element leaves you).
    * ``VpStack``: sum of the depths of all elements.
    * ``VpTree``: depth of the parent plus depth of the last child.
    * ``VpPath``: number of path components.
    """
    if isinstance(x, Viewport):
        return 1
    if isinstance(x, VpList):
        if len(x) == 0:
            return 0
        return depth(x[len(x) - 1])
    if isinstance(x, VpStack):
        return sum(depth(v) for v in x)
    if isinstance(x, VpTree):
        if len(x.children) == 0:
            return depth(x.parent)
        return depth(x.parent) + depth(x.children[len(x.children) - 1])
    if isinstance(x, VpPath):
        return x.n
    raise TypeError(
        f"depth() does not support {type(x).__name__}"
    )


# ---------------------------------------------------------------------------
# State-management helpers (late-import pattern)
# ---------------------------------------------------------------------------

def _get_state() -> Any:
    """Lazily import the state manager to avoid circular imports.

    Returns
    -------
    module
        The ``grid_py._state`` module.
    """
    from ._state import get_state  # noqa: WPS433 (late import to break circular dep)
    return get_state()


# ---------------------------------------------------------------------------
# Navigation functions
# ---------------------------------------------------------------------------


def push_viewport(
    *args: Union[Viewport, VpList, VpStack, VpTree, VpPath],
    recording: bool = True,
) -> None:
    """Push one or more viewports onto the viewport stack.

    Each argument is pushed in order.  A :class:`Viewport` is pushed
    directly; container types (:class:`VpList`, :class:`VpStack`,
    :class:`VpTree`) are traversed according to their semantics.

    Parameters
    ----------
    *args : Viewport, VpList, VpStack, VpTree, or VpPath
        Viewports to push.
    recording : bool
        Whether to record the operation on the display list.

    Raises
    ------
    ValueError
        If no viewports are provided.
    """
    if len(args) == 0:
        raise ValueError("must specify at least one viewport")
    state = _get_state()
    for vp in args:
        state.push_viewport(vp)


def pop_viewport(n: int = 1, recording: bool = True) -> None:
    """Pop *n* viewports from the viewport stack.

    Parameters
    ----------
    n : int
        Number of viewports to pop.  If ``0``, pop all viewports down
        to the root.
    recording : bool
        Whether to record the operation on the display list.

    Raises
    ------
    ValueError
        If *n* < 0.
    """
    if n < 0:
        raise ValueError("must pop at least one viewport")
    state = _get_state()
    state.pop_viewport(n)


def up_viewport(n: int = 1, recording: bool = True) -> Optional[VpPath]:
    """Navigate up *n* levels in the viewport tree without removing them.

    Parameters
    ----------
    n : int
        Number of levels to navigate up.  If ``0``, navigate to the
        root viewport.
    recording : bool
        Whether to record the operation on the display list.

    Returns
    -------
    VpPath or None
        The path segment that was navigated.

    Raises
    ------
    ValueError
        If *n* < 0.
    """
    if n < 0:
        raise ValueError("must navigate up at least one viewport")
    state = _get_state()

    # Capture the path segment being navigated (mirrors R grid.R:234-238).
    # R returns the tail of the current path corresponding to the n levels
    # being navigated away from.
    up_path: Optional[VpPath] = None
    path_str = state.current_vp_path()  # e.g. "ROOT/A/B"
    if path_str:
        parts = path_str.split("/")
        # Remove "ROOT" prefix for VpPath (R doesn't include ROOT)
        vp_parts = [p for p in parts if p != "ROOT"]
        if n == 0:
            # Navigate to root: return entire path
            if vp_parts:
                up_path = VpPath("/".join(vp_parts))
        elif len(vp_parts) >= n:
            tail = "/".join(vp_parts[-n:])
            if tail:
                up_path = VpPath(tail)

    state.up_viewport(n)
    return up_path


def down_viewport(
    name: Union[str, VpPath],
    strict: bool = False,
    recording: bool = True,
) -> int:
    """Navigate down to a named viewport that has already been pushed.

    Parameters
    ----------
    name : str or VpPath
        Name or path of the viewport to navigate to.
    strict : bool
        If ``True``, require an exact path match.
    recording : bool
        Whether to record the operation on the display list.

    Returns
    -------
    int
        The depth navigated.
    """
    if isinstance(name, str):
        name = VpPath(name)
    state = _get_state()
    return state.down_viewport(str(name), strict=strict)


def seek_viewport(name: str, recording: bool = True) -> int:
    """Navigate to a named viewport from anywhere in the tree.

    This is equivalent to navigating up to the root and then searching
    downward.

    Parameters
    ----------
    name : str
        Name of the viewport to find.
    recording : bool
        Whether to record the operation on the display list.

    Returns
    -------
    int
        The depth navigated from the root.
    """
    up_viewport(0, recording=recording)
    return down_viewport(name, recording=recording)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def current_viewport() -> Viewport:
    """Return the current viewport.

    Returns
    -------
    Viewport
    """
    state = _get_state()
    return state.current_viewport()


def current_vp_path() -> Optional[VpPath]:
    """Return the full path from the root to the current viewport.

    Returns
    -------
    VpPath or None
        ``None`` if the current viewport is the root.
    """
    state = _get_state()
    return state.current_vp_path()


def current_vp_tree() -> Union[Viewport, VpTree]:
    """Return the full viewport tree starting from the root.

    Returns
    -------
    Viewport or VpTree
    """
    state = _get_state()
    return state.current_vp_tree()


def current_transform() -> np.ndarray:
    """Return the 3x3 transformation matrix of the current viewport.

    The matrix maps normalised parent coordinates (NPC) to device
    coordinates, incorporating position, size, justification, and
    rotation.

    Returns
    -------
    numpy.ndarray
        A 3x3 float array.
    """
    state = _get_state()
    return state.current_transform()


def current_rotation() -> float:
    """Return the cumulative rotation angle (degrees) of the current viewport.

    Returns
    -------
    float
    """
    state = _get_state()
    return state.current_rotation()


def current_parent(n: int = 1) -> Optional[Viewport]:
    """Return the *n*-th generation ancestor of the current viewport.

    Parameters
    ----------
    n : int
        Number of generations to go up (default 1 = immediate parent).

    Returns
    -------
    Viewport or None
        ``None`` if the ancestor is the root (which has no parent).

    Raises
    ------
    ValueError
        If *n* < 1 or exceeds the depth of the viewport stack.
    """
    if n < 1:
        raise ValueError("invalid number of generations")
    state = _get_state()
    return state.current_parent(n)


# ---------------------------------------------------------------------------
# Convenience viewport constructors
# ---------------------------------------------------------------------------


def data_viewport(
    xData: Optional[Sequence[float]] = None,
    yData: Optional[Sequence[float]] = None,
    xscale: Optional[Sequence[float]] = None,
    yscale: Optional[Sequence[float]] = None,
    extension: float = 0.04,
    **kwargs: Any,
) -> Viewport:
    """Create a viewport with scales derived from data ranges.

    If *xscale* is not supplied it is computed from *xData* (and similarly
    for *yscale* / *yData*).  An *extension* factor is applied to expand
    the range slightly beyond the data limits.

    Parameters
    ----------
    xData : array-like or None
        Data for the x-axis.
    yData : array-like or None
        Data for the y-axis.
    xscale : sequence of float or None
        Explicit x-scale.  Overrides *xData* if given.
    yscale : sequence of float or None
        Explicit y-scale.  Overrides *yData* if given.
    extension : float
        Proportional extension of the data range on each side.
    **kwargs
        Additional keyword arguments passed to :class:`Viewport`.

    Returns
    -------
    Viewport

    Raises
    ------
    ValueError
        If neither *xData* nor *xscale* (or *yData* nor *yscale*) is
        supplied.

    Notes
    -----
    Mirrors R's ``dataViewport()``.
    """
    ext = [extension, extension]
    if len(ext) < 2:
        ext = ext * 2

    if xscale is None:
        if xData is None:
            raise ValueError(
                "must specify at least one of 'xData' or 'xscale'"
            )
        xarr = np.asarray(xData, dtype=float)
        rng = float(np.nanmax(xarr) - np.nanmin(xarr))
        xscale = [
            float(np.nanmin(xarr)) - ext[0] * rng,
            float(np.nanmax(xarr)) + ext[0] * rng,
        ]

    if yscale is None:
        if yData is None:
            raise ValueError(
                "must specify at least one of 'yData' or 'yscale'"
            )
        yarr = np.asarray(yData, dtype=float)
        rng = float(np.nanmax(yarr) - np.nanmin(yarr))
        yscale = [
            float(np.nanmin(yarr)) - ext[1] * rng,
            float(np.nanmax(yarr)) + ext[1] * rng,
        ]

    return Viewport(xscale=xscale, yscale=yscale, **kwargs)


def plot_viewport(
    margins: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Viewport:
    """Create a viewport with margins specified in lines.

    This mirrors R's ``plotViewport()``.  The four margins are given in the
    order ``[bottom, left, top, right]``.

    Parameters
    ----------
    margins : sequence of float or None
        Four margin sizes in ``"lines"`` units, ordered
        ``[bottom, left, top, right]``.  Defaults to
        ``[5.1, 4.1, 4.1, 2.1]``.
    **kwargs
        Additional keyword arguments passed to :class:`Viewport`.

    Returns
    -------
    Viewport
    """
    if margins is None:
        margins = [5.1, 4.1, 4.1, 2.1]
    else:
        margins = list(margins)
    # Ensure exactly 4 values by recycling
    while len(margins) < 4:
        margins = margins * 2
    margins = [float(m) for m in margins[:4]]

    bottom, left, top, right = margins

    x = Unit(left, "lines")
    width = Unit(1, "npc") - Unit(left + right, "lines")
    y = Unit(bottom, "lines")
    height = Unit(1, "npc") - Unit(bottom + top, "lines")

    return Viewport(
        x=x,
        width=width,
        y=y,
        height=height,
        just=["left", "bottom"],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# edit_viewport
# ---------------------------------------------------------------------------


def edit_viewport(
    vp: Optional[Viewport] = None,
    **kwargs: Any,
) -> Viewport:
    """Return an edited copy of a viewport.

    Creates a new :class:`Viewport` by taking the fields of *vp* and
    overriding any that are supplied via keyword arguments.

    Parameters
    ----------
    vp : Viewport or None
        The viewport to edit.  If ``None``, uses :func:`current_viewport`.
    **kwargs
        Fields to override (same names as :class:`Viewport` constructor
        parameters).

    Returns
    -------
    Viewport
        A new viewport with the edited fields.

    Notes
    -----
    Mirrors R's ``editViewport()``.
    """
    if vp is None:
        vp = current_viewport()

    base_kwargs = {
        "x": vp.x,
        "y": vp.y,
        "width": vp.width,
        "height": vp.height,
        "default_units": vp.default_units,
        "just": vp.just,
        "gp": vp.gp,
        "clip": vp.clip,
        "mask": vp.mask,
        "xscale": vp.xscale,
        "yscale": vp.yscale,
        "angle": vp.angle,
        "layout": vp.layout,
        "layout_pos_row": vp.layout_pos_row,
        "layout_pos_col": vp.layout_pos_col,
        "name": vp.name,
    }
    # Remap clip from internal representation back to constructor-friendly form
    if "clip" not in kwargs:
        clip_val = base_kwargs["clip"]
        if clip_val is True:
            base_kwargs["clip"] = "on"
        elif clip_val is None:
            base_kwargs["clip"] = "off"
        elif clip_val is False:
            base_kwargs["clip"] = "inherit"
    # Similarly for mask
    if "mask" not in kwargs:
        mask_val = base_kwargs["mask"]
        if mask_val is True:
            base_kwargs["mask"] = "inherit"
        elif mask_val is False:
            base_kwargs["mask"] = "none"

    base_kwargs.update(kwargs)
    return Viewport(**base_kwargs)


# ---------------------------------------------------------------------------
# show_viewport
# ---------------------------------------------------------------------------


def show_viewport(
    vp: Optional[Viewport] = None,
    recurse: bool = True,
    depth_val: int = 0,
    **kwargs: Any,
) -> str:
    """Return a human-readable summary of a viewport (tree).

    Parameters
    ----------
    vp : Viewport or None
        The viewport to display.  If ``None``, uses
        :func:`current_viewport`.
    recurse : bool
        Whether to recurse into children.
    depth_val : int
        Current indentation depth (used internally for recursive calls).
    **kwargs
        Reserved for future use.

    Returns
    -------
    str
        Multi-line summary string.

    Notes
    -----
    Mirrors R's ``showViewport()`` output.
    """
    if vp is None:
        vp = current_viewport()

    indent = "  " * depth_val
    lines: List[str] = []
    lines.append(f"{indent}{vp}")
    lines.append(f"{indent}  x      = {vp.x!r}")
    lines.append(f"{indent}  y      = {vp.y!r}")
    lines.append(f"{indent}  width  = {vp.width!r}")
    lines.append(f"{indent}  height = {vp.height!r}")
    lines.append(f"{indent}  just   = {vp.just!r}")
    lines.append(f"{indent}  xscale = {vp.xscale!r}")
    lines.append(f"{indent}  yscale = {vp.yscale!r}")
    lines.append(f"{indent}  angle  = {vp.angle!r}")

    if vp.layout is not None:
        lines.append(f"{indent}  layout = {vp.layout!r}")
    if vp.layout_pos_row is not None:
        lines.append(f"{indent}  layout.pos.row = {vp.layout_pos_row!r}")
    if vp.layout_pos_col is not None:
        lines.append(f"{indent}  layout.pos.col = {vp.layout_pos_col!r}")

    if recurse and vp.children:
        for child_name, child_vp in vp.children.items():
            lines.append(
                show_viewport(
                    child_vp, recurse=True, depth_val=depth_val + 1
                )
            )

    return "\n".join(lines)
