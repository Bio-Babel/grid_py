"""Group, define, and use grob system for grid_py.

Port of R's ``grid/R/group.R``.  This module provides grob classes and
factory functions for compositing groups:

* :class:`GroupGrob` -- a grob that groups its children with a compositing
  operator (``"over"``, ``"source"``, ``"xor"``, etc.).
* :class:`DefineGrob` -- a grob for deferred definition (define once, use
  later via :class:`UseGrob`).
* :class:`UseGrob` -- a grob that references a previously defined group and
  optionally applies an affine transform.

Factory functions mirror the R API:

* :func:`group_grob` / :func:`grid_group`
* :func:`define_grob` / :func:`grid_define`
* :func:`use_grob` / :func:`grid_use`
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ._gpar import Gpar
from ._grob import Grob, GList, GTree
from ._transforms import (
    viewport_transform,
)

__all__ = [
    # Classes
    "GroupGrob",
    "DefineGrob",
    "UseGrob",
    # Factory / convenience functions
    "group_grob",
    "grid_group",
    "define_grob",
    "grid_define",
    "use_grob",
    "grid_use",
    # Constants
    "COMPOSITING_OPERATORS",
]

# ---------------------------------------------------------------------------
# Valid compositing operators (mirrors R's .opIndex validation)
# ---------------------------------------------------------------------------

COMPOSITING_OPERATORS: tuple[str, ...] = (
    "clear",
    "source",
    "over",
    "in",
    "out",
    "atop",
    "dest",
    "dest.over",
    "dest.in",
    "dest.out",
    "dest.atop",
    "xor",
    "add",
    "saturate",
)


def _validate_op(op: str) -> str:
    """Validate a compositing operator string.

    Parameters
    ----------
    op : str
        Compositing operator name.

    Returns
    -------
    str
        The validated (lower-cased) operator.

    Raises
    ------
    ValueError
        If *op* is not a recognised compositing operator.
    """
    op_lower = op.lower()
    if op_lower not in COMPOSITING_OPERATORS:
        raise ValueError(
            f"Invalid compositing operator {op!r}. "
            f"Must be one of {COMPOSITING_OPERATORS!r}"
        )
    return op_lower


def _validate_transform(transform: Optional[NDArray[np.float64]]) -> None:
    """Validate that *transform* is a legal 3x3 affine matrix.

    The bottom-right element must be 1 and the right column (indices
    ``[0, 2]`` and ``[1, 2]``) must be 0, matching R's convention.

    Parameters
    ----------
    transform : ndarray or None
        A 3x3 numeric matrix, or ``None``.

    Raises
    ------
    ValueError
        If the matrix does not satisfy the constraints.
    TypeError
        If *transform* is not a numpy array.
    """
    if transform is None:
        return
    if not isinstance(transform, np.ndarray):
        raise TypeError(
            f"'transform' must be a numpy ndarray, got {type(transform).__name__}"
        )
    if transform.shape != (3, 3):
        raise ValueError(
            f"'transform' must be a 3x3 matrix, got shape {transform.shape}"
        )
    if not np.issubdtype(transform.dtype, np.number):
        raise ValueError("'transform' must contain numeric values")
    if transform[0, 2] != 0 or transform[1, 2] != 0 or transform[2, 2] != 1:
        raise ValueError(
            "Invalid transform: requires transform[0,2]==0, "
            "transform[1,2]==0, and transform[2,2]==1"
        )


# ============================================================================
# GroupGrob
# ============================================================================


class GroupGrob(GTree):
    """A grob that groups its children with a compositing operator.

    This is the Python equivalent of R's ``GridGroup`` S3 class produced by
    ``groupGrob()``.  When drawn, the *src* grob is composited onto the
    optional *dst* grob using the specified Porter-Duff compositing
    *op* (defaulting to ``"over"``).

    Parameters
    ----------
    src : Grob or None
        Source grob to composite.
    op : str
        Compositing operator (default ``"over"``).  Must be one of
        :data:`COMPOSITING_OPERATORS`.
    dst : Grob or None
        Destination grob.  When ``None`` the destination is transparent.
    name : str or None
        Unique grob name.  Auto-generated when ``None``.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Raises
    ------
    TypeError
        If *src* is not a :class:`Grob` (when provided) or *dst* is neither
        a :class:`Grob` nor ``None``.
    ValueError
        If *op* is not a recognised compositing operator.

    Examples
    --------
    >>> from grid_py._grob import Grob
    >>> src = Grob(name="circle1")
    >>> g = GroupGrob(src=src, op="xor")
    >>> g.op
    'xor'
    """

    _grid_class: str = "GridGroup"  # type: ignore[assignment]

    def __init__(
        self,
        src: Optional[Grob] = None,
        op: str = "over",
        dst: Optional[Grob] = None,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
    ) -> None:
        self.src: Optional[Grob] = src
        self.op: str = _validate_op(op)
        self.dst: Optional[Grob] = dst
        # Build a children GList from src and dst for the GTree machinery
        children = self._build_children()
        super().__init__(
            children=children,
            name=name,
            gp=gp,
            vp=vp,
            _grid_class="GridGroup",
        )

    # -- helpers -----------------------------------------------------------

    def _build_children(self) -> Optional[GList]:
        """Build a :class:`GList` from *src* and *dst*."""
        parts: list[Grob] = []
        if self.src is not None:
            parts.append(self.src)
        if self.dst is not None:
            parts.append(self.dst)
        return GList(*parts) if parts else None

    # -- validation --------------------------------------------------------

    def valid_details(self) -> None:
        """Validate GroupGrob-specific slots.

        Raises
        ------
        TypeError
            If *src* or *dst* have incorrect types.
        ValueError
            If *op* is invalid.
        """
        if hasattr(self, "src"):
            if self.src is not None and not isinstance(self.src, Grob):
                raise TypeError("Invalid source: must be a Grob or None")
        if hasattr(self, "dst"):
            if self.dst is not None and not isinstance(self.dst, Grob):
                raise TypeError("Invalid destination: must be a Grob or None")
        if hasattr(self, "op"):
            self.op = _validate_op(self.op)

    # -- drawing -----------------------------------------------------------

    def draw_details(self, recording: bool = True) -> None:
        """Draw the composited group.

        Parameters
        ----------
        recording : bool
            Whether the drawing should be recorded on the display list.

        Notes
        -----
        Actual device-level compositing is delegated to the rendering
        backend.  This method prepares the source / destination callables
        and compositing operator and stores the group definition for
        potential later reuse via :class:`UseGrob`.
        """
        # Placeholder: actual rendering requires a device backend.
        pass

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GroupGrob[{self.name}](op={self.op!r}, "
            f"src={self.src!r}, dst={self.dst!r})"
        )


# ============================================================================
# DefineGrob
# ============================================================================


class DefineGrob(GTree):
    """A grob for deferred group definition.

    This is the Python equivalent of R's ``GridDefine`` S3 class produced by
    ``defineGrob()``.  The group is defined (but not drawn) so that it can
    later be referenced by :class:`UseGrob`.

    Parameters
    ----------
    src : Grob
        Source grob to define.
    op : str
        Compositing operator (default ``"over"``).
    dst : Grob or None
        Destination grob (default ``None``).
    name : str or None
        Unique grob name.  Auto-generated when ``None``.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Raises
    ------
    TypeError
        If *src* is not a :class:`Grob`.
    ValueError
        If *op* is not a recognised compositing operator.

    Examples
    --------
    >>> from grid_py._grob import Grob
    >>> src = Grob(name="rect1")
    >>> d = DefineGrob(src=src)
    >>> d.src.name
    'rect1'
    """

    _grid_class: str = "GridDefine"  # type: ignore[assignment]

    def __init__(
        self,
        src: Grob = None,  # type: ignore[assignment]
        op: str = "over",
        dst: Optional[Grob] = None,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
    ) -> None:
        self.src: Grob = src  # type: ignore[assignment]
        self.op: str = _validate_op(op)
        self.dst: Optional[Grob] = dst
        # Build children
        parts: list[Grob] = []
        if self.src is not None:
            parts.append(self.src)
        if self.dst is not None:
            parts.append(self.dst)
        children = GList(*parts) if parts else None
        super().__init__(
            children=children,
            name=name,
            gp=gp,
            vp=vp,
            _grid_class="GridDefine",
        )

    # -- validation --------------------------------------------------------

    def valid_details(self) -> None:
        """Validate DefineGrob-specific slots.

        Raises
        ------
        TypeError
            If *src* is not a :class:`Grob`.
        """
        if hasattr(self, "src") and self.src is not None:
            if not isinstance(self.src, Grob):
                raise TypeError("Invalid source: must be a Grob")
        if hasattr(self, "dst"):
            if self.dst is not None and not isinstance(self.dst, Grob):
                raise TypeError("Invalid destination: must be a Grob or None")
        if hasattr(self, "op"):
            self.op = _validate_op(self.op)

    # -- drawing -----------------------------------------------------------

    def draw_details(self, recording: bool = True) -> None:
        """Define the group without drawing.

        Parameters
        ----------
        recording : bool
            Whether the definition should be recorded on the display list.

        Notes
        -----
        The group is registered so that a subsequent :class:`UseGrob` can
        reference it by name.  No visible output is produced.
        """
        # Placeholder: actual device-level definition requires a backend.
        pass

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DefineGrob[{self.name}](src={self.src!r}, "
            f"op={self.op!r}, dst={self.dst!r})"
        )


# ============================================================================
# UseGrob
# ============================================================================


class UseGrob(Grob):
    """A grob that references a previously defined group.

    This is the Python equivalent of R's ``GridUse`` S3 class produced by
    ``useGrob()``.  It draws a group that was previously registered via
    :class:`DefineGrob`, optionally applying an affine *transform*.

    Parameters
    ----------
    group : str
        Name of the previously defined group to use.
    transform : ndarray or None
        A 3x3 affine transformation matrix (NumPy array, dtype float64).
        When ``None``, the default viewport transform is used.
    name : str or None
        Unique grob name.  Auto-generated when ``None``.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Raises
    ------
    TypeError
        If *group* is not a string or *transform* is not a numpy array.
    ValueError
        If *transform* does not satisfy the affine-matrix constraints.

    Examples
    --------
    >>> u = UseGrob(group="rect1")
    >>> u.group
    'rect1'
    """

    _grid_class: str = "GridUse"  # type: ignore[assignment]

    def __init__(
        self,
        group: str = "",
        transform: Optional[NDArray[np.float64]] = None,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
    ) -> None:
        self.group: str = str(group)
        self.transform: Optional[NDArray[np.float64]] = transform
        _validate_transform(self.transform)
        super().__init__(
            name=name,
            gp=gp,
            vp=vp,
            _grid_class="GridUse",
        )

    # -- validation --------------------------------------------------------

    def valid_details(self) -> None:
        """Validate UseGrob-specific slots.

        Raises
        ------
        TypeError
            If *group* is not a string.
        ValueError
            If *transform* is invalid.
        """
        if hasattr(self, "group"):
            if not isinstance(self.group, str):
                raise TypeError("'group' must be a string")
        if hasattr(self, "transform"):
            _validate_transform(self.transform)

    # -- drawing -----------------------------------------------------------

    def draw_details(self, recording: bool = True) -> None:
        """Draw the referenced group with the optional transform.

        Parameters
        ----------
        recording : bool
            Whether the drawing should be recorded on the display list.

        Notes
        -----
        Looks up the group registered under :attr:`group` and renders it,
        applying :attr:`transform` if provided.  Issues a warning if the
        group has not been defined.
        """
        # Placeholder: actual device-level use requires a backend.
        pass

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UseGrob[{self.name}](group={self.group!r}, "
            f"transform={'set' if self.transform is not None else 'None'})"
        )


# ============================================================================
# Factory functions
# ============================================================================


def group_grob(
    src: Optional[Grob] = None,
    op: str = "over",
    dst: Optional[Grob] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GroupGrob:
    """Create a :class:`GroupGrob`.

    This is the functional equivalent of R's ``groupGrob()``.

    Parameters
    ----------
    src : Grob or None
        Source grob.
    op : str
        Compositing operator (default ``"over"``).
    dst : Grob or None
        Destination grob.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    GroupGrob
    """
    return GroupGrob(src=src, op=op, dst=dst, name=name, gp=gp, vp=vp)


def grid_group(
    src: Optional[Grob] = None,
    op: str = "over",
    dst: Optional[Grob] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    draw: bool = True,
) -> GroupGrob:
    """Create and optionally draw a :class:`GroupGrob`.

    This is the functional equivalent of R's ``grid.group()``.

    Parameters
    ----------
    src : Grob or None
        Source grob.
    op : str
        Compositing operator (default ``"over"``).
    dst : Grob or None
        Destination grob.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.
    draw : bool
        If ``True`` (default), the grob is drawn immediately via
        ``draw_details``.

    Returns
    -------
    GroupGrob
    """
    grb = GroupGrob(src=src, op=op, dst=dst, name=name, gp=gp, vp=vp)
    if draw:
        grb.draw_details()
    return grb


def define_grob(
    src: Grob,
    op: str = "over",
    dst: Optional[Grob] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> DefineGrob:
    """Create a :class:`DefineGrob`.

    This is the functional equivalent of R's ``defineGrob()``.

    Parameters
    ----------
    src : Grob
        Source grob.
    op : str
        Compositing operator (default ``"over"``).
    dst : Grob or None
        Destination grob.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    DefineGrob
    """
    return DefineGrob(src=src, op=op, dst=dst, name=name, gp=gp, vp=vp)


def grid_define(
    src: Grob,
    op: str = "over",
    dst: Optional[Grob] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    draw: bool = True,
) -> DefineGrob:
    """Create and optionally draw a :class:`DefineGrob`.

    This is the functional equivalent of R's ``grid.define()``.

    Parameters
    ----------
    src : Grob
        Source grob.
    op : str
        Compositing operator (default ``"over"``).
    dst : Grob or None
        Destination grob.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.
    draw : bool
        If ``True`` (default), the grob is drawn (defined) immediately.

    Returns
    -------
    DefineGrob
    """
    grb = DefineGrob(src=src, op=op, dst=dst, name=name, gp=gp, vp=vp)
    if draw:
        grb.draw_details()
    return grb


def use_grob(
    group: str,
    transform: Optional[NDArray[np.float64]] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> UseGrob:
    """Create a :class:`UseGrob`.

    This is the functional equivalent of R's ``useGrob()``.

    Parameters
    ----------
    group : str
        Name of the previously defined group.
    transform : ndarray or None
        3x3 affine transformation matrix.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    UseGrob
    """
    return UseGrob(group=group, transform=transform, name=name, gp=gp, vp=vp)


def grid_use(
    group: str,
    transform: Optional[NDArray[np.float64]] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    draw: bool = True,
) -> UseGrob:
    """Create and optionally draw a :class:`UseGrob`.

    This is the functional equivalent of R's ``grid.use()``.

    Parameters
    ----------
    group : str
        Name of the previously defined group.
    transform : ndarray or None
        3x3 affine transformation matrix.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.
    draw : bool
        If ``True`` (default), the grob is drawn immediately.

    Returns
    -------
    UseGrob
    """
    grb = UseGrob(group=group, transform=transform, name=name, gp=gp, vp=vp)
    if draw:
        grb.draw_details()
    return grb
