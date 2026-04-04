"""Global state management for grid_py (port of R's grid C-level state).

This module provides the :class:`GridState` singleton that manages the
viewport tree, display list, graphical-parameter inheritance stack, and
the binding to a matplotlib ``Figure``/``Axes`` pair.  It replaces the
C-level ``GridState`` struct found in R's *grid* package.

.. note::
   Viewport classes are **not** imported here to avoid circular
   dependencies.  Viewport references are stored and manipulated via
   duck typing (any object with ``name``, ``parent``, ``children``, and
   ``layout_pos`` attributes is accepted).
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._gpar import Gpar

__all__ = ["GridState", "get_state"]

# ---------------------------------------------------------------------------
# Default device dimensions (≈ 7 in ≈ 17.78 cm)
# ---------------------------------------------------------------------------

_DEFAULT_DEVICE_WIDTH_CM: float = 17.78
_DEFAULT_DEVICE_HEIGHT_CM: float = 17.78


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_root_viewport() -> Any:
    """Create a minimal root viewport dict (duck-typed).

    A "real" viewport object will replace this once
    :meth:`GridState.reset` or :meth:`GridState.push_viewport` is
    called with an actual viewport instance.  The dict is used only as
    the initial sentinel so that the tree is never ``None``.

    Returns
    -------
    dict
        A mapping that quacks like a viewport for bootstrap purposes.
    """
    return {
        "name": "ROOT",
        "parent": None,
        "children": [],
        "layout_pos": None,
        "gpar": Gpar(),
        "rotation": 0.0,
        "transform": np.eye(3, dtype=np.float64),
    }


def _vp_attr(vp: Any, attr: str, default: Any = None) -> Any:
    """Retrieve an attribute from a viewport, supporting both objects and dicts.

    Parameters
    ----------
    vp : Any
        Viewport object or dict.
    attr : str
        Attribute / key name.
    default : Any, optional
        Fallback value.

    Returns
    -------
    Any
    """
    if isinstance(vp, dict):
        return vp.get(attr, default)
    return getattr(vp, attr, default)


def _vp_set_attr(vp: Any, attr: str, value: Any) -> None:
    """Set an attribute on a viewport, supporting both objects and dicts.

    Parameters
    ----------
    vp : Any
        Viewport object or dict.
    attr : str
        Attribute / key name.
    value : Any
        Value to assign.
    """
    if isinstance(vp, dict):
        vp[attr] = value
    else:
        setattr(vp, attr, value)


def _vp_children(vp: Any) -> list:
    """Return the children list of a viewport.

    Parameters
    ----------
    vp : Any
        Viewport object or dict.

    Returns
    -------
    list
    """
    return _vp_attr(vp, "children", [])


def _vp_name(vp: Any) -> str:
    """Return the name of a viewport.

    Parameters
    ----------
    vp : Any
        Viewport object or dict.

    Returns
    -------
    str
    """
    return _vp_attr(vp, "name", "")


def _vp_parent(vp: Any) -> Any:
    """Return the parent of a viewport.

    Parameters
    ----------
    vp : Any
        Viewport object or dict.

    Returns
    -------
    Any
    """
    return _vp_attr(vp, "parent", None)


# ---------------------------------------------------------------------------
# GridState
# ---------------------------------------------------------------------------

class GridState:
    """Singleton holding the global grid graphics state.

    Manages the viewport tree, display list, graphical-parameter
    inheritance stack, and the connection to a matplotlib device
    (``Figure`` / ``Axes``).

    Attributes
    ----------
    _vp_tree : Any
        The root viewport (pushed viewport representing the device).
    _current_vp : Any
        Reference to the currently active viewport.
    _display_list : list[Any]
        Recorded drawing operations.
    _dl_on : bool
        Whether display-list recording is enabled.
    _gpar_stack : list[Gpar]
        Stack of graphical parameter objects for inheritance.
    _device_width_cm : float
        Device width in centimetres.
    _device_height_cm : float
        Device height in centimetres.
    _figure : Optional[Any]
        Matplotlib ``Figure`` reference (or ``None``).
    _axes : Optional[Any]
        Matplotlib ``Axes`` reference (or ``None``).

    Examples
    --------
    >>> state = GridState()
    >>> state.current_viewport()["name"]
    'ROOT'
    """

    # ---- class-level singleton bookkeeping --------------------------------

    _instance: Optional["GridState"] = None

    def __new__(cls) -> "GridState":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._initialized = False
            cls._instance = inst
        return cls._instance

    # ---- initialisation ---------------------------------------------------

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized: bool = True
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Set every slot to its default value."""
        self._vp_tree: Any = _make_root_viewport()
        self._current_vp: Any = self._vp_tree
        self._display_list: List[Any] = []
        self._dl_on: bool = True
        self._gpar_stack: List[Gpar] = [Gpar()]
        self._device_width_cm: float = _DEFAULT_DEVICE_WIDTH_CM
        self._device_height_cm: float = _DEFAULT_DEVICE_HEIGHT_CM
        self._figure: Optional[Any] = None
        self._axes: Optional[Any] = None

    # ---- reset ------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state and reinitialise to the default root viewport.

        This is the equivalent of ``grid.newpage()`` in R.
        """
        self._init_defaults()

    # ---- viewport tree manipulation ---------------------------------------

    def push_viewport(self, vp: Any) -> None:
        """Push a viewport onto the tree as a child of the current viewport.

        Parameters
        ----------
        vp : Any
            A viewport-like object.  Must expose ``name``, ``parent``,
            and ``children`` attributes (or dict keys).
        """
        _vp_set_attr(vp, "parent", self._current_vp)
        children = _vp_children(self._current_vp)
        if children is None:
            children = []
            _vp_set_attr(self._current_vp, "children", children)
        children.append(vp)
        self._current_vp = vp

    def pop_viewport(self, n: int = 1) -> None:
        """Pop *n* viewports, navigating back toward the root.

        Parameters
        ----------
        n : int, optional
            Number of levels to pop (default ``1``).  If *n* equals ``0``
            the navigation returns to the root viewport.

        Raises
        ------
        ValueError
            If *n* is negative or greater than the current depth.
        """
        if n < 0:
            raise ValueError(f"'n' must be non-negative, got {n}")
        if n == 0:
            # Pop to root.
            self._current_vp = self._vp_tree
            return
        for _ in range(n):
            parent = _vp_parent(self._current_vp)
            if parent is None:
                raise ValueError(
                    "Cannot pop past the root viewport."
                )
            # Remove the current viewport from its parent's children.
            parent_children = _vp_children(parent)
            try:
                parent_children.remove(self._current_vp)
            except ValueError:
                pass
            self._current_vp = parent

    def up_viewport(self, n: int = 1) -> None:
        """Navigate up *n* levels without removing viewports from the tree.

        Parameters
        ----------
        n : int, optional
            Number of levels to ascend (default ``1``).  ``0`` navigates
            to the root.

        Raises
        ------
        ValueError
            If *n* is negative or exceeds the current depth.
        """
        if n < 0:
            raise ValueError(f"'n' must be non-negative, got {n}")
        if n == 0:
            self._current_vp = self._vp_tree
            return
        for _ in range(n):
            parent = _vp_parent(self._current_vp)
            if parent is None:
                raise ValueError(
                    "Cannot navigate above the root viewport."
                )
            self._current_vp = parent

    def down_viewport(self, name: str, strict: bool = False) -> int:
        """Navigate down to a named viewport (breadth-first search).

        The search starts from the **children** of the current viewport.

        Parameters
        ----------
        name : str
            Name of the target viewport.
        strict : bool, optional
            If ``True`` the name must match exactly; otherwise a
            case-insensitive match is attempted after an exact match
            fails.

        Returns
        -------
        int
            The depth (number of levels descended) to reach the target.

        Raises
        ------
        LookupError
            If no matching viewport is found.
        """
        depth = self._search_down(self._current_vp, name, strict)
        if depth is None:
            raise LookupError(
                f"Viewport '{name}' not found below the current viewport."
            )
        return depth

    def _search_down(
        self, start: Any, name: str, strict: bool
    ) -> Optional[int]:
        """BFS helper for :meth:`down_viewport`.

        Returns the depth on success, or ``None`` on failure.  As a
        side-effect, ``_current_vp`` is updated to point to the found
        viewport.
        """
        queue: Deque[Tuple[Any, int]] = deque()
        for child in _vp_children(start):
            queue.append((child, 1))

        while queue:
            vp, d = queue.popleft()
            vp_n = _vp_name(vp)
            if vp_n == name or (not strict and vp_n.lower() == name.lower()):
                self._current_vp = vp
                return d
            for child in _vp_children(vp):
                queue.append((child, d + 1))
        return None

    def seek_viewport(self, name: str) -> int:
        """Global search for a named viewport starting from the root.

        If found, ``_current_vp`` is set to the matching viewport and
        the absolute depth from the root is returned.

        Parameters
        ----------
        name : str
            Viewport name to search for.

        Returns
        -------
        int
            Depth from the root to the found viewport.

        Raises
        ------
        LookupError
            If no matching viewport is found anywhere in the tree.
        """
        result = self._search_down(self._vp_tree, name, strict=False)
        if result is None:
            raise LookupError(
                f"Viewport '{name}' not found in the viewport tree."
            )
        return result

    # ---- viewport queries -------------------------------------------------

    def current_viewport(self) -> Any:
        """Return the currently active viewport.

        Returns
        -------
        Any
            The active viewport object (or dict).
        """
        return self._current_vp

    def current_vp_path(self) -> str:
        """Return the ``/``-separated path from root to the current viewport.

        Returns
        -------
        str
            E.g. ``"ROOT/panel/strip"``.
        """
        parts: List[str] = []
        vp: Any = self._current_vp
        while vp is not None:
            parts.append(_vp_name(vp))
            vp = _vp_parent(vp)
        parts.reverse()
        return "/".join(parts)

    def current_vp_tree(self) -> Any:
        """Return the root of the entire viewport tree.

        Returns
        -------
        Any
        """
        return self._vp_tree

    def current_transform(self) -> np.ndarray:
        """Return the cumulative 3x3 transformation matrix for the current viewport.

        The matrix is accumulated by multiplying transforms from the
        root down to the current viewport.

        Returns
        -------
        numpy.ndarray
            A 3x3 ``float64`` transformation matrix.
        """
        matrices: List[np.ndarray] = []
        vp: Any = self._current_vp
        while vp is not None:
            t = _vp_attr(vp, "transform", None)
            if t is not None:
                matrices.append(np.asarray(t, dtype=np.float64))
            vp = _vp_parent(vp)
        matrices.reverse()

        result = np.eye(3, dtype=np.float64)
        for m in matrices:
            result = result @ m
        return result

    def current_rotation(self) -> float:
        """Return the cumulative rotation angle (degrees) at the current viewport.

        Returns
        -------
        float
            The sum of ``rotation`` attributes from root to current viewport.
        """
        total: float = 0.0
        vp: Any = self._current_vp
        while vp is not None:
            total += float(_vp_attr(vp, "rotation", 0.0))
            vp = _vp_parent(vp)
        return total

    def current_parent(self) -> Any:
        """Return the parent of the current viewport.

        Returns
        -------
        Any
            Parent viewport, or ``None`` if at the root.
        """
        return _vp_parent(self._current_vp)

    # ---- gpar management --------------------------------------------------

    def get_gpar(self) -> Gpar:
        """Return the current (top-of-stack) graphical parameters.

        Returns
        -------
        Gpar
        """
        return self._gpar_stack[-1]

    def set_gpar(self, gp: Gpar) -> None:
        """Push a :class:`Gpar` onto the parameter stack.

        Parameters
        ----------
        gp : Gpar
            Graphical parameters to make current.
        """
        self._gpar_stack.append(gp)

    # ---- display list -----------------------------------------------------

    def record(self, op: Any) -> None:
        """Append an operation to the display list (if recording is on).

        Parameters
        ----------
        op : Any
            A drawable / grob operation.
        """
        if self._dl_on:
            self._display_list.append(op)

    def get_display_list(self) -> List[Any]:
        """Return the current display list.

        Returns
        -------
        list[Any]
        """
        return self._display_list

    def set_display_list_on(self, on: bool) -> None:
        """Enable or disable display-list recording.

        Parameters
        ----------
        on : bool
            ``True`` to enable, ``False`` to disable.
        """
        self._dl_on = bool(on)

    # ---- device binding ---------------------------------------------------

    def init_device(
        self,
        fig: Any,
        ax: Any,
        width_cm: float = _DEFAULT_DEVICE_WIDTH_CM,
        height_cm: float = _DEFAULT_DEVICE_HEIGHT_CM,
    ) -> None:
        """Bind the state to a matplotlib ``Figure`` and ``Axes``.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The matplotlib figure.
        ax : matplotlib.axes.Axes
            The matplotlib axes.
        width_cm : float, optional
            Device width in centimetres (default ~7 in).
        height_cm : float, optional
            Device height in centimetres (default ~7 in).
        """
        self._figure = fig
        self._axes = ax
        self._device_width_cm = float(width_cm)
        self._device_height_cm = float(height_cm)

    def get_device(self) -> Tuple[Any, Any]:
        """Return the matplotlib ``(Figure, Axes)`` tuple.

        Returns
        -------
        tuple[Any, Any]
            ``(fig, ax)`` — either may be ``None`` if
            :meth:`init_device` has not yet been called.
        """
        return (self._figure, self._axes)


# ---------------------------------------------------------------------------
# Module-level singleton & accessor
# ---------------------------------------------------------------------------

_state: GridState = GridState()


def get_state() -> GridState:
    """Return the module-level :class:`GridState` singleton.

    Returns
    -------
    GridState
        The global state instance.
    """
    return _state
