"""Scene capture and manipulation for grid_py (port of R's grid grab/force/cap).

This module provides functions for capturing, forcing, reverting, and
reordering the current grid scene:

* :func:`grid_grab` -- grab the current display list as a :class:`~._grob.GTree`.
* :func:`grid_grab_expr` -- evaluate a callable and grab the result.
* :func:`grid_force` -- force delayed grobs on the display list.
* :func:`grid_revert` -- revert previously forced grobs.
* :func:`grid_cap` -- capture the current display as a raster (NumPy array).
* :func:`grid_reorder` -- reorder children of a gTree on the display list.

References
----------
R source: ``src/library/grid/R/grab.R`` (~248 lines)
"""

from __future__ import annotations

import copy
import warnings
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from ._grob import (
    GList,
    GTree,
    Grob,
    force_grob,
    is_grob,
    reorder_grob,
)
from ._path import GPath
from ._display_list import DisplayList, DLDrawGrob
from ._state import GridState, get_state

__all__ = [
    "grid_grab",
    "grid_grab_expr",
    "grid_force",
    "grid_revert",
    "grid_cap",
    "grid_reorder",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_dl_grobs(dl: DisplayList, warn: int = 2) -> Optional[GList]:
    """Collect all grobs from the display list into a :class:`GList`.

    Parameters
    ----------
    dl : DisplayList
        The display list to scan.
    warn : int
        Warning level.  0 = silent, 1 = warn on definite problems,
        2 = warn on possible problems.

    Returns
    -------
    GList or None
        A :class:`GList` containing all grobs found, or ``None`` if the
        display list is empty.
    """
    grobs: list[Grob] = []
    seen_names: set[str] = set()

    for item in dl:
        if isinstance(item, DLDrawGrob) and item.grob is not None:
            grob = item.grob
            if warn >= 1:
                if grob.name in seen_names:
                    warnings.warn(
                        "one or more grobs overwritten "
                        "(grab may not be faithful; try wrap=True)",
                        stacklevel=3,
                    )
            seen_names.add(grob.name)
            grobs.append(grob)

    if not grobs:
        return None
    return GList(*grobs)


def _grab_dl(
    warn: int = 2,
    wrap: bool = False,
    wrap_vps: bool = False,
) -> Optional[GTree]:
    """Grab the current display list as a :class:`GTree`.

    Parameters
    ----------
    warn : int
        Warning level (0, 1, or 2).
    wrap : bool
        If ``True``, wrap all viewport pushes and grobs.
    wrap_vps : bool
        If ``True``, wrap viewport operations inside recorded grobs.

    Returns
    -------
    GTree or None
        A :class:`GTree` encapsulating the scene, or ``None`` if the
        display list is empty.
    """
    state = get_state()
    dl = state.display_list

    if len(dl) == 0:
        return None

    children = _collect_dl_grobs(dl, warn=warn)
    if children is None:
        return None

    if wrap:
        # When wrapping, deep-copy each grob so the grabbed tree is independent
        wrapped: list[Grob] = []
        for child in children:
            wrapped.append(copy.deepcopy(child))
        children = GList(*wrapped)

    return GTree(children=children)


# ---------------------------------------------------------------------------
# Public API -- grid.grab
# ---------------------------------------------------------------------------


def grid_grab(
    warn: int = 2,
    wrap: bool = False,
    wrap_vps: bool = False,
) -> Optional[GTree]:
    """Grab the current display list as a :class:`~._grob.GTree`.

    Collects all grobs from the display list into a single gTree that
    represents the current scene.  This is the Python equivalent of R's
    ``grid.grab()``.

    Parameters
    ----------
    warn : int, optional
        Warning level controlling how aggressively potential problems are
        reported.

        - ``0`` -- no warnings.
        - ``1`` -- warn about situations that are definitely *not* captured
          faithfully (e.g. duplicated top-level grob names).
        - ``2`` (default) -- additionally warn about situations that *may*
          not be captured faithfully (e.g. top-level viewport pushes).
    wrap : bool, optional
        If ``True``, wrap all pushes and grobs in the resulting gTree so
        that the grabbed tree can be replayed independently.
    wrap_vps : bool, optional
        If ``True``, also wrap viewport operations.

    Returns
    -------
    GTree or None
        A :class:`~._grob.GTree` encapsulating the current scene, or
        ``None`` if the display list is empty.

    Examples
    --------
    >>> tree = grid_grab()
    >>> if tree is not None:
    ...     print(tree)
    """
    return _grab_dl(warn=warn, wrap=wrap, wrap_vps=wrap_vps)


# ---------------------------------------------------------------------------
# Public API -- grid.grabExpr
# ---------------------------------------------------------------------------


def grid_grab_expr(
    expr: Callable[[], Any],
    warn: int = 2,
    wrap: bool = False,
    wrap_vps: bool = False,
    width: float = 7.0,
    height: float = 7.0,
) -> Optional[GTree]:
    """Evaluate *expr* and grab the resulting scene.

    The callable *expr* is executed in a temporary graphics context (the
    display list is cleared and restored afterwards).  The scene produced
    by *expr* is then captured via :func:`grid_grab`.  This is the Python
    equivalent of R's ``grid.grabExpr()``.

    Parameters
    ----------
    expr : callable
        A zero-argument callable that performs grid drawing operations.
    warn : int, optional
        Warning level (see :func:`grid_grab`).
    wrap : bool, optional
        Wrap mode (see :func:`grid_grab`).
    wrap_vps : bool, optional
        Wrap viewport operations (see :func:`grid_grab`).
    width : float, optional
        Nominal device width in inches (default ``7.0``).  Stored on the
        state but not used for physical rendering.
    height : float, optional
        Nominal device height in inches (default ``7.0``).

    Returns
    -------
    GTree or None
        A :class:`~._grob.GTree` encapsulating the scene drawn by *expr*,
        or ``None`` if nothing was drawn.
    """
    state = get_state()
    # Save and clear the display list
    saved_items = state.display_list.get_items()
    state.display_list.clear()

    try:
        # Run the user's drawing code
        expr()
        # Grab what was drawn
        result = _grab_dl(warn=warn, wrap=wrap, wrap_vps=wrap_vps)
    finally:
        # Restore the original display list
        state.display_list.clear()
        for item in saved_items:
            state.display_list.record(item)

    return result


# ---------------------------------------------------------------------------
# Public API -- grid.force / grid.revert
# ---------------------------------------------------------------------------


def grid_force(
    x: Optional[Union[Grob, GTree]] = None,
    redraw: bool = True,
) -> Optional[Union[Grob, GTree]]:
    """Force delayed grobs, materialising deferred content.

    If *x* is ``None``, every grob on the current display list is forced
    in place.  Otherwise, *x* is forced and a new (forced) copy is
    returned.  This is the Python equivalent of R's ``grid.force()``.

    Parameters
    ----------
    x : Grob, GTree, or None, optional
        The grob to force.  ``None`` forces the entire display list.
    redraw : bool, optional
        If ``True`` (default) and *x* is ``None``, redraw after forcing.

    Returns
    -------
    Grob, GTree, or None
        When *x* is provided, the forced copy.  When *x* is ``None``,
        ``None`` is returned (the display list is modified in place).

    Examples
    --------
    >>> forced_tree = grid_force(my_gtree)
    """
    if x is not None:
        return force_grob(x)

    # Force every grob on the display list
    state = get_state()
    dl = state.display_list

    for item in dl:
        if isinstance(item, DLDrawGrob) and item.grob is not None:
            forced = force_grob(item.grob)
            item.grob = forced
            item.params["grob"] = forced

    if redraw:
        dl.replay(state)

    return None


def grid_revert(
    x: Optional[Union[Grob, GTree]] = None,
    redraw: bool = True,
) -> Optional[Union[Grob, GTree]]:
    """Revert previously forced grobs to their original (unforced) state.

    If *x* is ``None``, every grob on the display list that carries an
    ``_original`` attribute (set by :func:`grid_force`) is reverted.
    Otherwise, *x* itself is reverted.  This is the Python equivalent of
    R's ``grid.revert()``.

    Parameters
    ----------
    x : Grob, GTree, or None, optional
        The grob to revert.  ``None`` reverts the entire display list.
    redraw : bool, optional
        If ``True`` (default) and *x* is ``None``, redraw after reverting.

    Returns
    -------
    Grob, GTree, or None
        When *x* is provided, the reverted grob (or the unchanged grob if
        it was not previously forced).  When *x* is ``None``, ``None`` is
        returned (the display list is modified in place).
    """
    if x is not None:
        original = getattr(x, "_original", None)
        if original is not None:
            return original
        # For gTrees, try to revert children
        if isinstance(x, GTree):
            result = copy.deepcopy(x)
            reverted_children: list[Grob] = []
            for name in result._children_order:
                child = result._children[name]
                reverted = grid_revert(child)
                reverted_children.append(reverted if reverted is not None else child)
            result._set_children_internal(GList(*reverted_children))
            return result
        return x

    # Revert every grob on the display list
    state = get_state()
    dl = state.display_list

    for item in dl:
        if isinstance(item, DLDrawGrob) and item.grob is not None:
            original = getattr(item.grob, "_original", None)
            if original is not None:
                item.grob = original
                item.params["grob"] = original

    if redraw:
        dl.replay(state)

    return None


# ---------------------------------------------------------------------------
# Public API -- grid.cap
# ---------------------------------------------------------------------------


def grid_cap(native: bool = True) -> Optional[np.ndarray]:
    """Capture the current display as a raster image.

    This attempts to rasterise the current scene.  In the Python port this
    relies on a matplotlib backend (if one is available).  This is the
    Python equivalent of R's ``grid.cap()``.

    Parameters
    ----------
    native : bool, optional
        If ``True`` (default), return the raster in the device's native
        resolution as a NumPy array of shape ``(H, W, 4)`` (RGBA uint8).
        If ``False``, return in normalised [0, 1] float64.

    Returns
    -------
    numpy.ndarray or None
        The raster image, or ``None`` if no rendering backend is available.
    """
    state = get_state()

    # Attempt to get a matplotlib figure from the state
    fig = getattr(state, "_figure", None)
    if fig is None:
        fig = getattr(state, "figure", None)

    if fig is None:
        warnings.warn(
            "no rendering backend available for grid_cap; returning None",
            stacklevel=2,
        )
        return None

    try:
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf, dtype=np.uint8).copy()
        if not native:
            arr = arr.astype(np.float64) / 255.0
        return arr
    except Exception:
        warnings.warn(
            "failed to capture raster from rendering backend; returning None",
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Public API -- grid.reorder
# ---------------------------------------------------------------------------


def grid_reorder(
    gPath: Union[str, GPath],
    order: Union[List[int], List[str]],
    back: bool = True,
    grep: bool = False,
    redraw: bool = True,
) -> None:
    """Reorder the children of a gTree on the display list.

    Locates the gTree identified by *gPath* on the current display list
    and reorders its children according to *order*.  This is the Python
    equivalent of R's ``grid.reorder()``.

    Parameters
    ----------
    gPath : str or GPath
        Path identifying the target gTree.
    order : list[int] or list[str]
        Indices (0-based) or names specifying the new ordering.
    back : bool, optional
        If ``True`` (default), specified children come first (drawn behind);
        unspecified children are appended.  If ``False``, unspecified
        children come first; specified children are appended (drawn in
        front).
    grep : bool, optional
        If ``True``, use regex matching on path components.
    redraw : bool, optional
        If ``True`` (default), redraw after reordering.

    Raises
    ------
    TypeError
        If *gPath* is invalid.
    ValueError
        If no gTree matching *gPath* is found, or if *order* contains
        invalid names or indices.
    """
    if isinstance(gPath, str):
        gpath = GPath(gPath)
    elif isinstance(gPath, GPath):
        gpath = gPath
    else:
        raise TypeError(f"invalid gPath: expected str or GPath, got {type(gPath).__name__}")

    state = get_state()
    dl = state.display_list

    # Find the target gTree -- import the helper from _edit
    from ._edit import _find_dl_grobs

    grep_flags = [grep] * gpath.n
    matches = _find_dl_grobs(dl, gpath, strict=False, grep=grep_flags, global_=False)

    if not matches:
        raise ValueError(f"gPath ({gpath}) does not match any grob on the display list")

    _dl_idx, matched_grob = matches[0]

    if not isinstance(matched_grob, GTree):
        raise TypeError(
            f"gPath matched '{matched_grob.name}' which is not a gTree; "
            "cannot reorder"
        )

    # Perform the reorder in place
    reordered = reorder_grob(matched_grob, order, back=back)
    matched_grob._children_order = reordered._children_order

    if redraw:
        dl.replay(state)
