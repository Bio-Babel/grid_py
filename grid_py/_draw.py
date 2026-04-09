"""Core drawing engine for grid_py -- Python port of R's grid drawing functions.

This module handles rendering grobs via :class:`CairoRenderer`, porting
functionality from R's ``grid/R/grid.R`` and ``grid/R/grob.R``.

The central entry point is :func:`grid_draw`, which performs S3-like dispatch
on grobs, gTrees, gLists, viewports, and viewport paths.

References
----------
R source: ``src/library/grid/R/grid.R``, ``src/library/grid/R/grob.R``
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._gpar import Gpar
from ._grob import Grob, GList, GTree
from ._state import get_state
from ._display_list import DisplayList, DLDrawGrob
from ._units import Unit
from ._utils import grid_pretty as _grid_pretty

__all__ = [
    "grid_draw",
    "grid_newpage",
    "grid_refresh",
    "grid_record",
    "record_grob",
    "grid_delay",
    "delay_grob",
    "grid_dl_apply",
    "grid_locator",
    "grid_pretty",
]


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------


def _unit_to_float(val: Any) -> float:
    """Extract a scalar float from a value that may be a Unit."""
    from ._units import Unit
    if isinstance(val, Unit):
        return float(val._values[0])
    return float(val)


def _unit_to_array(val: Any) -> np.ndarray:
    """Extract a numeric array from a value that may be a Unit."""
    from ._units import Unit
    if isinstance(val, Unit):
        return np.asarray(val._values, dtype=float)
    if isinstance(val, (list, tuple)):
        try:
            return np.asarray(val, dtype=float)
        except (ValueError, TypeError):
            return np.array([_unit_to_float(v) for v in val], dtype=float)
    return np.atleast_1d(np.asarray(val, dtype=float))


def _render_grob(
    grob: Grob,
    renderer: Any,
    gp: Optional[Gpar] = None,
    transform: Optional[np.ndarray] = None,
) -> None:
    """Render a single grob via the :class:`CairoRenderer`.

    Dispatches on ``grob._grid_class`` to call the appropriate renderer
    method.

    Parameters
    ----------
    grob : Grob
        The graphical object to render.
    renderer : CairoRenderer
        The Cairo renderer.
    gp : Gpar or None, optional
        Resolved graphical parameters (merged from context + grob).
    transform : numpy.ndarray or None, optional
        3x3 affine transform matrix; currently reserved for future use.
    """
    if renderer is None:
        return

    cls = getattr(grob, "_grid_class", "grob")

    # ---- rect -----------------------------------------------------------
    if cls == "rect":
        renderer.draw_rect(
            x=_unit_to_float(getattr(grob, "x", 0.0)),
            y=_unit_to_float(getattr(grob, "y", 0.0)),
            w=_unit_to_float(getattr(grob, "width", 1.0)),
            h=_unit_to_float(getattr(grob, "height", 1.0)),
            hjust=float(getattr(grob, "hjust", None) or 0.5),
            vjust=float(getattr(grob, "vjust", None) or 0.5),
            gp=gp,
        )

    # ---- roundrect ------------------------------------------------------
    elif cls == "roundrect":
        renderer.draw_roundrect(
            x=_unit_to_float(getattr(grob, "x", 0.0)),
            y=_unit_to_float(getattr(grob, "y", 0.0)),
            w=_unit_to_float(getattr(grob, "width", 1.0)),
            h=_unit_to_float(getattr(grob, "height", 1.0)),
            r=_unit_to_float(getattr(grob, "r", 0.0)),
            hjust=float(getattr(grob, "hjust", None) or 0.5),
            vjust=float(getattr(grob, "vjust", None) or 0.5),
            gp=gp,
        )

    # ---- circle ---------------------------------------------------------
    elif cls == "circle":
        renderer.draw_circle(
            x=_unit_to_float(getattr(grob, "x", 0.5)),
            y=_unit_to_float(getattr(grob, "y", 0.5)),
            r=_unit_to_float(getattr(grob, "r", 0.5)),
            gp=gp,
        )

    # ---- lines / polyline ------------------------------------------------
    elif cls in ("lines", "polyline"):
        x = _unit_to_array(getattr(grob, "x", [0.0, 1.0]))
        y = _unit_to_array(getattr(grob, "y", [0.0, 1.0]))
        id_ = getattr(grob, "id", None)
        if id_ is not None:
            id_ = np.atleast_1d(np.asarray(id_, dtype=int))
        renderer.draw_polyline(x, y, id_=id_, gp=gp)

    # ---- segments --------------------------------------------------------
    elif cls == "segments":
        renderer.draw_segments(
            x0=_unit_to_array(getattr(grob, "x0", [])),
            y0=_unit_to_array(getattr(grob, "y0", [])),
            x1=_unit_to_array(getattr(grob, "x1", [])),
            y1=_unit_to_array(getattr(grob, "y1", [])),
            gp=gp,
        )

    # ---- polygon ---------------------------------------------------------
    elif cls == "polygon":
        renderer.draw_polygon(
            x=_unit_to_array(getattr(grob, "x", [])),
            y=_unit_to_array(getattr(grob, "y", [])),
            gp=gp,
        )

    # ---- text ------------------------------------------------------------
    elif cls == "text":
        renderer.draw_text(
            x=_unit_to_float(getattr(grob, "x", 0.5)),
            y=_unit_to_float(getattr(grob, "y", 0.5)),
            label=getattr(grob, "label", ""),
            rot=float(getattr(grob, "rot", 0.0)),
            hjust=float(getattr(grob, "hjust", None) or 0.5),
            vjust=float(getattr(grob, "vjust", None) or 0.5),
            gp=gp,
        )

    # ---- points ----------------------------------------------------------
    elif cls == "points":
        pch_raw = getattr(grob, "pch", 19)
        if hasattr(pch_raw, "__len__"):
            pch_raw = pch_raw[0] if len(pch_raw) > 0 else 19
        renderer.draw_points(
            x=_unit_to_array(getattr(grob, "x", [])),
            y=_unit_to_array(getattr(grob, "y", [])),
            size=_unit_to_float(getattr(grob, "size", 1.0)),
            pch=int(pch_raw),
            gp=gp,
        )

    # ---- pathgrob --------------------------------------------------------
    elif cls == "pathgrob":
        x = _unit_to_array(getattr(grob, "x", []))
        y = _unit_to_array(getattr(grob, "y", []))
        path_id = getattr(grob, "pathId", None)
        if path_id is None:
            path_id = np.ones(len(x), dtype=int)
        else:
            path_id = np.atleast_1d(np.asarray(path_id, dtype=int))
        renderer.draw_path(
            x=x, y=y, path_id=path_id,
            rule=getattr(grob, "rule", "winding"),
            gp=gp,
        )

    # ---- rastergrob ------------------------------------------------------
    elif cls == "rastergrob":
        image = getattr(grob, "image", None)
        if image is not None:
            renderer.draw_raster(
                image=image,
                x=_unit_to_float(getattr(grob, "x", 0.0)),
                y=_unit_to_float(getattr(grob, "y", 0.0)),
                w=_unit_to_float(getattr(grob, "width", 1.0)),
                h=_unit_to_float(getattr(grob, "height", 1.0)),
                interpolate=getattr(grob, "interpolate", True),
            )

    # ---- null / no-op ---------------------------------------------------
    elif cls == "null":
        pass

    # ---- move.to / line.to -----------------------------------------------
    elif cls == "move.to":
        renderer.move_to(
            _unit_to_float(getattr(grob, "x", 0.0)),
            _unit_to_float(getattr(grob, "y", 0.0)),
        )

    elif cls == "line.to":
        renderer.line_to(
            _unit_to_float(getattr(grob, "x", 0.0)),
            _unit_to_float(getattr(grob, "y", 0.0)),
            gp=gp,
        )

    else:
        warnings.warn(
            f"_render_grob: unknown grob class '{cls}', skipping",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Viewport / gpar push/pop helpers
# ---------------------------------------------------------------------------


def _push_grob_vp(vp: Any) -> None:
    """Push a grob's viewport (or navigate down for a VpPath).

    Parameters
    ----------
    vp : Viewport or VpPath
        The viewport to push or navigate to.
    """
    from ._viewport import Viewport, push_viewport, down_viewport
    from ._path import VpPath

    if isinstance(vp, VpPath):
        down_viewport(vp, strict=True, recording=False)
    else:
        push_viewport(vp, recording=False)


def _pop_grob_vp(vp: Any) -> None:
    """Pop/navigate up from a grob's viewport.

    Parameters
    ----------
    vp : Viewport or VpPath
        The viewport that was previously pushed.
    """
    from ._viewport import up_viewport
    from ._path import VpPath

    d = _vp_depth(vp)
    up_viewport(d, recording=False)


def _vp_depth(vp: Any) -> int:
    """Return the depth of a viewport (number of levels it adds).

    Parameters
    ----------
    vp : Any
        A viewport, VpPath, VpStack, VpList, or VpTree.

    Returns
    -------
    int
        The depth.
    """
    from ._path import VpPath

    if isinstance(vp, VpPath):
        # VpPath stores the number of path components
        return getattr(vp, "n", 1)
    if hasattr(vp, "depth"):
        return vp.depth()
    # Default single viewport depth
    return 1


def _push_vp_gp(grob: Grob) -> None:
    """Push the grob's viewport and apply its gpar.

    Parameters
    ----------
    grob : Grob
        The grob whose ``vp`` and ``gp`` should be activated.
    """
    state = get_state()
    if grob.vp is not None:
        _push_grob_vp(grob.vp)
    if grob.gp is not None:
        state.set_gpar(grob.gp)


# ---------------------------------------------------------------------------
# Draw-grob dispatcher (mirrors R's drawGrob / drawGTree / drawGList)
# ---------------------------------------------------------------------------


def _draw_grob(x: Grob) -> None:
    """Internal: draw a plain Grob (not a GTree).

    Mirrors R's ``drawGrob``: saves gpar, calls preDraw (makeContext +
    pushvpgp + preDrawDetails), makeContent, drawDetails, postDraw.

    Parameters
    ----------
    x : Grob
        The grob to draw.
    """
    state = get_state()

    # Save current gpar
    saved_gpar = copy.copy(state.get_gpar())

    try:
        # preDraw: makeContext -> push vp/gp -> preDrawDetails
        x = x.make_context()
        _push_vp_gp(x)
        x.pre_draw_details()

        # makeContent -> drawDetails
        x = x.make_content()
        x.draw_details(recording=False)

        # Render to Cairo
        renderer = state.get_renderer()
        if renderer is not None:
            merged_gp = _merge_gpar(state.get_gpar(), x.gp)
            _render_grob(x, renderer, gp=merged_gp)

        # postDraw: postDrawDetails -> pop vp
        x.post_draw_details()
        if x.vp is not None:
            _pop_grob_vp(x.vp)
    finally:
        # Restore gpar
        state.set_gpar(saved_gpar)


def _draw_gtree(x: GTree) -> None:
    """Internal: draw a GTree.

    Mirrors R's ``drawGTree``: saves gpar + current grob context, calls
    preDraw (makeContext + pushvpgp + children vp + preDrawDetails),
    makeContent, drawDetails, draws children in order, then postDraw.

    Parameters
    ----------
    x : GTree
        The gTree to draw.
    """
    state = get_state()

    # Save state
    saved_gpar = copy.copy(state.get_gpar())

    try:
        # preDraw: makeContext -> push vp/gp -> preDrawDetails
        x = x.make_context()
        _push_vp_gp(x)

        # Push children viewport if present, then navigate back up
        children_vp = getattr(x, "childrenvp", None) or getattr(x, "children_vp", None)
        if children_vp is not None:
            from ._viewport import push_viewport, up_viewport
            temp_gp = copy.copy(state.get_gpar())
            push_viewport(children_vp, recording=False)
            up_viewport(_vp_depth(children_vp), recording=False)
            state.set_gpar(temp_gp)

        x.pre_draw_details()

        # makeContent -> drawDetails
        x = x.make_content()
        x.draw_details(recording=False)

        # Render the gTree itself (in case it has direct content)
        renderer = state.get_renderer()
        if renderer is not None:
            merged_gp = _merge_gpar(state.get_gpar(), x.gp)
            _render_grob(x, renderer, gp=merged_gp)

        # Draw children in order
        for child_name in x._children_order:
            child = x._children.get(child_name)
            if child is not None:
                grid_draw(child, recording=False)

        # postDraw
        x.post_draw_details()
        if x.vp is not None:
            _pop_grob_vp(x.vp)
    finally:
        state.set_gpar(saved_gpar)


def _draw_glist(x: GList) -> None:
    """Internal: draw every grob in a GList.

    Each child is drawn individually via :func:`grid_draw`.

    Parameters
    ----------
    x : GList
        The list of grobs to draw.
    """
    for grob in x:
        grid_draw(grob, recording=True)


def _merge_gpar(context_gp: Optional[Gpar], grob_gp: Optional[Gpar]) -> Gpar:
    """Merge context graphical parameters with grob-level overrides.

    Parameters
    ----------
    context_gp : Gpar or None
        The inherited graphical parameters from the viewport stack.
    grob_gp : Gpar or None
        The grob's own graphical parameters.

    Returns
    -------
    Gpar
        A new Gpar with grob settings taking precedence over context.
    """
    if context_gp is None and grob_gp is None:
        return Gpar()
    if context_gp is None:
        return grob_gp  # type: ignore[return-value]
    if grob_gp is None:
        return context_gp

    # Build merged copy: start from context, override with grob
    merged = copy.copy(context_gp)
    for name in ("col", "fill", "alpha", "lty", "lwd", "lex",
                 "lineend", "linejoin", "linemitre",
                 "fontsize", "cex", "fontfamily", "fontface",
                 "lineheight", "font"):
        val = grob_gp.get(name, None)
        if val is not None:
            merged.set(name, val)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grid_draw(
    x: Any,
    recording: bool = True,
) -> None:
    """Draw a grob (or gList, gTree, viewport, vpPath).

    This is the main entry point for rendering grid objects.  It provides
    S3-style dispatch analogous to R's ``grid.draw``:

    * **Grob**: pushes ``vp`` if present, applies ``gp``, calls
      ``pre_draw_details`` / ``draw_details`` / ``post_draw_details``,
      then pops ``vp``.
    * **GTree**: runs ``make_context`` / ``make_content``, then draws
      children in order.
    * **GList**: draws each grob in sequence.
    * **Viewport**: pushes it.
    * **VpPath**: navigates to it.

    Parameters
    ----------
    x : Grob, GTree, GList, Viewport, VpPath, or None
        The object to draw.  ``None`` is silently ignored.
    recording : bool, optional
        Whether to record this operation on the display list
        (default ``True``).

    Notes
    -----
    Mirrors R's ``grid.draw()`` with S3 dispatch on the class of *x*.
    """
    if x is None:
        return

    state = get_state()

    # Late imports to avoid circular dependencies
    from ._path import VpPath

    # -- Viewport dispatch ---------------------------------------------------
    # Import Viewport lazily
    try:
        from ._viewport import Viewport, push_viewport, down_viewport
    except ImportError:
        Viewport = type(None)  # type: ignore[misc,assignment]
        push_viewport = None  # type: ignore[assignment]
        down_viewport = None  # type: ignore[assignment]

    if isinstance(x, VpPath):
        if down_viewport is not None:
            down_viewport(x, strict=False, recording=False)
        if recording:
            state.record(x)
        return

    if Viewport is not None and isinstance(x, Viewport):
        if push_viewport is not None:
            push_viewport(x, recording=False)
        if recording:
            state.record(x)
        return

    # -- GList dispatch (before GTree/Grob since GTree is-a Grob) -----------
    if isinstance(x, GList):
        _draw_glist(x)
        return

    # -- GTree dispatch (must be checked before Grob) -----------------------
    if isinstance(x, GTree):
        _draw_gtree(x)
        if recording:
            state.record(DLDrawGrob(grob=x))
        return

    # -- Grob dispatch ------------------------------------------------------
    if isinstance(x, Grob):
        _draw_grob(x)
        if recording:
            state.record(DLDrawGrob(grob=x))
        return

    # -- Numeric "pop" / "up" dispatches (from R display list replay) -------
    if isinstance(x, (int, float)):
        # In R, a numeric on the display list encodes a pop/up count.
        # We silently ignore it here; replay is handled by grid_refresh.
        return

    warnings.warn(
        f"grid_draw: don't know how to draw object of type {type(x).__name__}",
        stacklevel=2,
    )


def grid_newpage(
    recording: bool = True,
    clear_dl: bool = True,
    width: float = 7.0,
    height: float = 5.0,
    dpi: float = 150.0,
    bg: Any = "white",
) -> None:
    """Clear the surface and start a fresh page.

    This is equivalent to R's ``grid.newpage()``.  If no
    :class:`CairoRenderer` currently exists, a new one is created.
    The viewport stack is reset to the root viewport.

    Parameters
    ----------
    recording : bool, optional
        If ``True`` (default) the display list is initialised for recording
        new operations.
    clear_dl : bool, optional
        If ``True`` (default) the existing display list is cleared.
    width : float, optional
        Device width in inches (default 7.0).
    height : float, optional
        Device height in inches (default 5.0).
    dpi : float, optional
        Resolution in dots per inch (default 150).
    bg : str or tuple, optional
        Background colour (default ``"white"``).
    """
    from .renderer import CairoRenderer

    state = get_state()

    # Reset all state (viewport tree, gpar stack, display list)
    state.reset()

    # Obtain or create a renderer
    renderer = state.get_renderer()
    if renderer is None:
        renderer = CairoRenderer(
            width=width, height=height, dpi=dpi, bg=bg,
        )
        state.init_device(renderer)
    else:
        renderer.new_page(bg=bg)

    if clear_dl:
        dl = state.get_display_list()
        dl.clear()

    if recording:
        state.set_display_list_on(True)


def grid_refresh() -> None:
    """Replay the display list, redrawing the current scene.

    Equivalent to R's ``grid.refresh()``.  This calls ``grid_newpage``
    with ``recording=False`` and then redraws every item on the display
    list.
    """
    state = get_state()
    dl = list(state.get_display_list())  # snapshot

    grid_newpage(recording=False, clear_dl=False)

    for item in dl:
        if hasattr(item, "grob") and item.grob is not None:
            grid_draw(item.grob, recording=False)
        elif hasattr(item, "replay"):
            item.replay(state)
        else:
            grid_draw(item, recording=False)


def grid_record(
    expr: Callable[..., None],
    list_: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> None:
    """Record an expression as a grob and draw it.

    Equivalent to R's ``grid.record()``.  The *expr* callable is wrapped in
    a ``Grob`` with class ``"recordedGrob"`` and drawn immediately.

    Parameters
    ----------
    expr : callable
        A callable that performs drawing operations when called.
    list_ : dict or None, optional
        Additional variables to pass as the evaluation environment.
    name : str or None, optional
        Name for the wrapper grob.
    """
    grob = record_grob(expr, list_=list_, name=name)
    grid_draw(grob)


def record_grob(
    expr: Callable[..., None],
    list_: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> Grob:
    """Create a recorded-expression grob without drawing it.

    Equivalent to R's ``recordGrob()``.  Returns a :class:`Grob` whose
    ``draw_details`` evaluates *expr*.

    Parameters
    ----------
    expr : callable
        A callable that performs drawing operations.
    list_ : dict or None, optional
        Environment mapping made available to *expr*.
    name : str or None, optional
        Name for the grob.

    Returns
    -------
    Grob
        A grob with ``_grid_class="recordedGrob"`` that evaluates *expr*
        in its ``draw_details`` hook.
    """

    class _RecordedGrob(Grob):
        """A grob that evaluates a stored callable when drawn."""

        def __init__(
            self,
            expr_: Callable[..., None],
            env: Optional[Dict[str, Any]],
            name_: Optional[str],
        ) -> None:
            self._expr = expr_
            self._env = env or {}
            super().__init__(name=name_, _grid_class="recordedGrob")

        def draw_details(self, recording: bool = True) -> None:
            self._expr(**self._env)

    return _RecordedGrob(expr_=expr, env=list_, name_=name)


def grid_delay(
    expr: Callable[..., Any],
    list_: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> None:
    """Create a delayed-evaluation grob and draw it.

    Equivalent to R's ``grid.delay()``.  The *expr* callable must return
    a :class:`Grob` or :class:`GList`; evaluation is deferred to
    ``make_content`` time.

    Parameters
    ----------
    expr : callable
        A callable returning a :class:`Grob` or :class:`GList`.
    list_ : dict or None, optional
        Environment mapping available to *expr*.
    name : str or None, optional
        Name for the wrapper gTree.
    """
    grob = delay_grob(expr, list_=list_, name=name)
    grid_draw(grob)


def delay_grob(
    expr: Callable[..., Any],
    list_: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> GTree:
    """Create a delayed-evaluation gTree without drawing it.

    Equivalent to R's ``delayGrob()``.  The returned :class:`GTree`
    evaluates *expr* in its ``make_content`` hook, which must produce a
    :class:`Grob` or :class:`GList`.

    Parameters
    ----------
    expr : callable
        A callable returning a :class:`Grob` or :class:`GList`.
    list_ : dict or None, optional
        Environment mapping available to *expr*.
    name : str or None, optional
        Name for the gTree.

    Returns
    -------
    GTree
        A gTree with ``_grid_class="delayedgrob"`` whose ``make_content``
        evaluates *expr*.
    """

    class _DelayedGrob(GTree):
        """A gTree that lazily evaluates its content."""

        def __init__(
            self,
            expr_: Callable[..., Any],
            env: Optional[Dict[str, Any]],
            name_: Optional[str],
        ) -> None:
            self._expr = expr_
            self._env = env or {}
            super().__init__(name=name_, _grid_class="delayedgrob")

        def make_content(self) -> "GTree":
            result = self._expr(**self._env)
            if isinstance(result, Grob):
                children = GList(result)
            elif isinstance(result, GList):
                children = result
            else:
                raise TypeError("'expr' must return a Grob or GList")
            self.set_children(children)
            return self

    return _DelayedGrob(expr_=expr, env=list_, name_=name)


def grid_dl_apply(
    fn: Callable[[Any], Any],
) -> None:
    """Apply a function to each display-list item, replacing in place.

    Equivalent to R's ``grid.DLapply()``.  The function *fn* is called on
    every display-list entry.  The return value replaces the original entry.
    If *fn* returns ``None`` the entry is kept as ``None``; otherwise the
    return value must be the same type as the original entry.

    Parameters
    ----------
    fn : callable
        A function ``(item) -> new_item``.  *new_item* must be ``None`` or
        of the same class as *item*.

    Raises
    ------
    TypeError
        If *fn* returns a value whose type does not match the original entry.

    Notes
    -----
    This is "blood-curdlingly dangerous" for the display-list state (to
    quote the R source).  Two safety measures are taken:

    1. All new elements are generated first before any assignment, so an
       error during generation does not trash the display list.
    2. Each new element is type-checked against the original.
    """
    state = get_state()
    dl = state.get_display_list()

    # Phase 1: generate replacements
    new_items: List[Any] = []
    for item in dl:
        new_item = fn(item)
        if new_item is not None and type(new_item) is not type(item):
            raise TypeError(
                f"invalid modification of the display list: "
                f"expected {type(item).__name__}, got {type(new_item).__name__}"
            )
        new_items.append(new_item)

    # Phase 2: assign
    dl.clear()
    dl.extend(new_items)


def grid_locator(
    unit: str = "native",
) -> Optional[Dict[str, float]]:
    """Interactive point selection on the current device.

    Equivalent to R's ``grid.locator()``.  The Cairo renderer does not
    support interactive point selection; this function always returns
    ``None``.

    Parameters
    ----------
    unit : str, optional
        The unit in which to return coordinates (default ``"native"``).

    Returns
    -------
    None
        Always ``None`` (interactive selection not supported).
    """
    warnings.warn(
        "grid_locator: interactive point selection is not supported "
        "with the Cairo renderer",
        stacklevel=2,
    )
    return None


def grid_pretty(
    range_val: Sequence[float],
) -> np.ndarray:
    """Return pretty tick positions for a numeric range.

    This is a thin wrapper around :func:`._utils.grid_pretty`.

    Parameters
    ----------
    range_val : sequence of float
        A two-element sequence ``[min, max]`` defining the range.

    Returns
    -------
    numpy.ndarray
        An array of tick positions.
    """
    return _grid_pretty(range_val)
