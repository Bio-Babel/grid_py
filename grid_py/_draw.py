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


_HJUST_MAP = {"left": 0.0, "right": 1.0, "centre": 0.5, "center": 0.5}
_VJUST_MAP = {"bottom": 0.0, "top": 1.0, "centre": 0.5, "center": 0.5}


def _resolve_just(grob: Any) -> Tuple[float, float]:
    """Resolve hjust/vjust from a grob, honouring the ``just`` attribute.

    Matches R's ``valid.just()``: a single string like ``"right"`` sets
    only the horizontal component (vjust defaults to 0.5), and
    ``"top"`` sets only the vertical component (hjust defaults to 0.5).
    A 2-element vector ``("left", "top")`` sets both.
    """
    hjust = getattr(grob, "hjust", None)
    vjust = getattr(grob, "vjust", None)
    if hjust is not None and vjust is not None:
        return float(hjust), float(vjust)
    just = getattr(grob, "just", None)
    if just is not None:
        if isinstance(just, str):
            # Single string: "left"/"right" → hjust only;
            # "top"/"bottom" → vjust only; "centre" → both
            if just in _HJUST_MAP:
                hj = _HJUST_MAP[just]
                vj = _VJUST_MAP.get(just, 0.5)
            elif just in _VJUST_MAP:
                hj = _HJUST_MAP.get(just, 0.5)
                vj = _VJUST_MAP[just]
            else:
                hj, vj = 0.5, 0.5
        elif isinstance(just, (list, tuple)) and len(just) >= 2:
            hj = _HJUST_MAP.get(just[0], 0.5) if isinstance(just[0], str) else float(just[0])
            vj = _VJUST_MAP.get(just[1], 0.5) if isinstance(just[1], str) else float(just[1])
        elif isinstance(just, (list, tuple)) and len(just) == 1:
            hj = _HJUST_MAP.get(just[0], 0.5) if isinstance(just[0], str) else float(just[0])
            vj = 0.5
        else:
            hj, vj = 0.5, 0.5
        if hjust is None:
            hjust = hj
        if vjust is None:
            vjust = vj
    return float(hjust if hjust is not None else 0.5), float(vjust if vjust is not None else 0.5)


def _subset_gpar(gp: Optional[Gpar], i: int) -> Optional[Gpar]:
    """Return a Gpar containing only the *i*-th element of each vectorised param."""
    if gp is None:
        return None
    new_params: Dict[str, Any] = {}
    for key, val in gp.params.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) > 1:
            new_params[key] = val[i % len(val)]
        elif isinstance(val, (list, tuple)) and len(val) > 1:
            new_params[key] = val[i % len(val)]
        else:
            new_params[key] = val
    return Gpar(**new_params)


def _unit_to_float(val: Any) -> float:
    """Extract a scalar float from a value that may be a Unit.

    .. deprecated::
        Use ``renderer.resolve_x/y/w/h`` instead for unit-aware resolution.
        This function is kept only for call sites that genuinely need a raw
        numeric value without unit resolution (e.g. rotation angles).
    """
    from ._units import Unit
    if isinstance(val, Unit):
        return float(val._values[0])
    return float(val)


def _unit_to_array(val: Any) -> np.ndarray:
    """Extract a numeric array from a value that may be a Unit.

    .. deprecated::
        Use ``renderer.resolve_x/y_array`` instead for unit-aware resolution.
    """
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
        xs = renderer.resolve_x_array(getattr(grob, "x", [0.0]), gp=gp)
        ys = renderer.resolve_y_array(getattr(grob, "y", [0.0]), gp=gp)
        ws = renderer.resolve_w_array(getattr(grob, "width", [1.0]), gp=gp)
        hs = renderer.resolve_h_array(getattr(grob, "height", [1.0]), gp=gp)
        hj, vj = _resolve_just(grob)
        n = max(len(xs), len(ys), len(ws), len(hs))
        if len(xs) == 1:
            xs = np.full(n, xs[0])
        if len(ys) == 1:
            ys = np.full(n, ys[0])
        if len(ws) == 1:
            ws = np.full(n, ws[0])
        if len(hs) == 1:
            hs = np.full(n, hs[0])
        for i in range(n):
            gp_i = _subset_gpar(gp, i) if n > 1 else gp
            renderer.draw_rect(
                x=float(xs[i]), y=float(ys[i]),
                w=float(ws[i]), h=float(hs[i]),
                hjust=hj, vjust=vj, gp=gp_i,
            )

    # ---- roundrect ------------------------------------------------------
    elif cls == "roundrect":
        renderer.draw_roundrect(
            x=renderer.resolve_x(getattr(grob, "x", 0.0), gp=gp),
            y=renderer.resolve_y(getattr(grob, "y", 0.0), gp=gp),
            w=renderer.resolve_w(getattr(grob, "width", 1.0), gp=gp),
            h=renderer.resolve_h(getattr(grob, "height", 1.0), gp=gp),
            r=renderer.resolve_w(getattr(grob, "r", 0.0), gp=gp),
            hjust=float(getattr(grob, "hjust", None) or 0.5),
            vjust=float(getattr(grob, "vjust", None) or 0.5),
            gp=gp,
        )

    # ---- circle ---------------------------------------------------------
    elif cls == "circle":
        renderer.draw_circle(
            x=renderer.resolve_x(getattr(grob, "x", 0.5), gp=gp),
            y=renderer.resolve_y(getattr(grob, "y", 0.5), gp=gp),
            r=renderer.resolve_w(getattr(grob, "r", 0.5), gp=gp),
            gp=gp,
        )

    # ---- lines / polyline ------------------------------------------------
    elif cls in ("lines", "polyline"):
        x = renderer.resolve_x_array(getattr(grob, "x", [0.0, 1.0]), gp=gp)
        y = renderer.resolve_y_array(getattr(grob, "y", [0.0, 1.0]), gp=gp)
        id_ = getattr(grob, "id", None)
        if id_ is not None:
            id_ = np.atleast_1d(np.asarray(id_, dtype=int))
        renderer.draw_polyline(x, y, id_=id_, gp=gp)

    # ---- segments --------------------------------------------------------
    elif cls == "segments":
        renderer.draw_segments(
            x0=renderer.resolve_x_array(getattr(grob, "x0", []), gp=gp),
            y0=renderer.resolve_y_array(getattr(grob, "y0", []), gp=gp),
            x1=renderer.resolve_x_array(getattr(grob, "x1", []), gp=gp),
            y1=renderer.resolve_y_array(getattr(grob, "y1", []), gp=gp),
            gp=gp,
        )

    # ---- polygon ---------------------------------------------------------
    elif cls == "polygon":
        px = renderer.resolve_x_array(getattr(grob, "x", []), gp=gp)
        py = renderer.resolve_y_array(getattr(grob, "y", []), gp=gp)
        pid = getattr(grob, "id", None)
        if pid is not None:
            # R semantics: polygonGrob(id=...) draws separate polygons
            # per unique id value, each with its own fill/stroke.
            pid = np.atleast_1d(np.asarray(pid))
            unique_ids = np.unique(pid)
            for idx, uid in enumerate(unique_ids):
                mask = pid == uid
                gp_i = _subset_gpar(gp, idx) if gp else gp
                renderer.draw_polygon(px[mask], py[mask], gp=gp_i)
        else:
            renderer.draw_polygon(px, py, gp=gp)

    # ---- text ------------------------------------------------------------
    elif cls == "text":
        hj, vj = _resolve_just(grob)
        renderer.draw_text(
            x=renderer.resolve_x(getattr(grob, "x", 0.5), gp=gp),
            y=renderer.resolve_y(getattr(grob, "y", 0.5), gp=gp),
            label=getattr(grob, "label", ""),
            rot=float(getattr(grob, "rot", 0.0)),
            hjust=hj,
            vjust=vj,
            gp=gp,
        )

    # ---- points ----------------------------------------------------------
    elif cls == "points":
        pch_raw = getattr(grob, "pch", 19)
        # pch may be a scalar or per-point array — pass through as-is
        if isinstance(pch_raw, (np.ndarray, list, tuple)):
            pch_val = np.asarray(pch_raw, dtype=int)
        elif isinstance(pch_raw, (int, float, np.integer, np.floating)):
            pch_val = int(pch_raw)
        else:
            pch_val = 19
        renderer.draw_points(
            x=renderer.resolve_x_array(getattr(grob, "x", []), gp=gp),
            y=renderer.resolve_y_array(getattr(grob, "y", []), gp=gp),
            size=renderer.resolve_w(getattr(grob, "size", 1.0), gp=gp),
            pch=pch_val,
            gp=gp,
        )

    # ---- pathgrob --------------------------------------------------------
    elif cls == "pathgrob":
        x = renderer.resolve_x_array(getattr(grob, "x", []), gp=gp)
        y = renderer.resolve_y_array(getattr(grob, "y", []), gp=gp)
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
        image = getattr(grob, "raster", None)
        if image is None:
            image = getattr(grob, "image", None)
        if image is not None:
            # Apply justification (same as rect_grob)
            hj, vj = _resolve_just(grob)
            raw_x = renderer.resolve_x(getattr(grob, "x", 0.0), gp=gp)
            raw_y = renderer.resolve_y(getattr(grob, "y", 0.0), gp=gp)
            raw_w = renderer.resolve_w(getattr(grob, "width", 1.0), gp=gp)
            raw_h = renderer.resolve_h(getattr(grob, "height", 1.0), gp=gp)
            # Compute bottom-left corner from anchor + justification
            x0 = raw_x - raw_w * hj
            y0 = raw_y - raw_h * vj
            renderer.draw_raster(
                image=image,
                x=x0,
                y=y0,
                w=raw_w,
                h=raw_h,
                interpolate=getattr(grob, "interpolate", True),
            )

    # ---- GridStroke / GridFill / GridFillStroke (R 4.2+, path.R) ----------
    elif cls == "GridStroke":
        path_grob = getattr(grob, "path", None)
        if path_grob is not None and hasattr(renderer, "begin_path_collect"):
            renderer._ctx.save()
            renderer.begin_path_collect()
            _render_grob(path_grob, renderer, gp=gp)
            renderer.end_path_stroke(gp)
            renderer._ctx.restore()

    elif cls == "GridFill":
        path_grob = getattr(grob, "path", None)
        rule = getattr(grob, "rule", "winding")
        if path_grob is not None and hasattr(renderer, "begin_path_collect"):
            renderer._ctx.save()
            renderer.begin_path_collect(rule=rule)
            _render_grob(path_grob, renderer, gp=gp)
            renderer.end_path_fill(gp)
            renderer._ctx.restore()

    elif cls == "GridFillStroke":
        path_grob = getattr(grob, "path", None)
        rule = getattr(grob, "rule", "winding")
        if path_grob is not None and hasattr(renderer, "begin_path_collect"):
            renderer._ctx.save()
            renderer.begin_path_collect(rule=rule)
            _render_grob(path_grob, renderer, gp=gp)
            renderer.end_path_fill_stroke(gp)
            renderer._ctx.restore()

    # ---- null / gTree / base grob – no-op --------------------------------
    elif cls in ("null", "grob", "gTree"):
        pass

    # ---- move.to / line.to -----------------------------------------------
    elif cls == "move.to":
        renderer.move_to(
            renderer.resolve_x(getattr(grob, "x", 0.0), gp=gp),
            renderer.resolve_y(getattr(grob, "y", 0.0), gp=gp),
        )

    elif cls == "line.to":
        renderer.line_to(
            renderer.resolve_x(getattr(grob, "x", 0.0), gp=gp),
            renderer.resolve_y(getattr(grob, "y", 0.0), gp=gp),
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

    Also updates the renderer's coordinate transform so that NPC [0,1]
    maps to the viewport's sub-region, matching R's grid semantics.

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

    # Update renderer coordinate transform
    state = get_state()
    renderer = state.get_renderer()
    if renderer is not None and hasattr(renderer, "push_viewport"):
        renderer.push_viewport(vp)


def _pop_grob_vp(vp: Any) -> None:
    """Pop/navigate up from a grob's viewport.

    Also restores the renderer's coordinate transform.

    Parameters
    ----------
    vp : Viewport or VpPath
        The viewport that was previously pushed.
    """
    from ._viewport import up_viewport
    from ._path import VpPath

    d = _vp_depth(vp)
    up_viewport(d, recording=False)

    # Restore renderer coordinate transform
    state = get_state()
    renderer = state.get_renderer()
    if renderer is not None and hasattr(renderer, "pop_viewport"):
        for _ in range(d):
            renderer.pop_viewport()


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
    x_device: Optional[float] = None,
    y_device: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Convert device coordinates to grid coordinates in the current viewport.

    Equivalent to R's ``grid.locator(unit)``.  In R, the function waits
    for a mouse click on an interactive device.  In Python, since we use
    file-based Cairo rendering, device coordinates are passed explicitly
    via *x_device* and *y_device*.

    The conversion uses the current viewport's coordinate transform,
    matching R's approach of applying ``solve(current.transform())``
    to the raw device coordinates.

    Mirrors ``grid.locator`` in R (``grid/R/interactive.R``).

    Parameters
    ----------
    unit : str, optional
        Target unit for the returned coordinates (default ``"native"``).
        Common values: ``"native"``, ``"npc"``, ``"cm"``, ``"inches"``,
        ``"points"``.
    x_device : float or None
        X coordinate in device pixels (0 = left edge).
    y_device : float or None
        Y coordinate in device pixels (0 = top edge).

    Returns
    -------
    dict or None
        ``{"x": <value>, "y": <value>}`` in the requested unit,
        or ``None`` if coordinates are not provided.

    Examples
    --------
    >>> grid_locator("npc", x_device=200, y_device=150)
    {"x": 0.45, "y": 0.62}
    """
    if x_device is None or y_device is None:
        import warnings
        warnings.warn(
            "grid.locator() is not supported in non-interactive mode; "
            "pass x_device and y_device explicitly.",
            stacklevel=2,
        )
        return None

    state = get_state()
    renderer = state.get_renderer()
    if renderer is None:
        return None

    # Current viewport bounds in device coords: (x0, y0, pw, ph)
    x0, y0, pw, ph, *_vp_rest = renderer._vp_stack[-1]
    if pw == 0 or ph == 0:
        return None

    # Device coords → NPC within current viewport
    # _x(npc) = x0 + npc * pw   → npc = (x_device - x0) / pw
    # _y(npc) = y0 + (1-npc)*ph → npc = 1 - (y_device - y0) / ph
    npc_x = (float(x_device) - x0) / pw
    npc_y = 1.0 - (float(y_device) - y0) / ph

    if unit == "npc":
        return {"x": npc_x, "y": npc_y}

    # NPC → inches (via viewport device size and DPI)
    x_inches = npc_x * pw / renderer.dpi
    y_inches = npc_y * ph / renderer.dpi

    if unit == "inches":
        return {"x": x_inches, "y": y_inches}
    elif unit == "cm":
        return {"x": x_inches * 2.54, "y": y_inches * 2.54}
    elif unit == "mm":
        return {"x": x_inches * 25.4, "y": y_inches * 25.4}
    elif unit in ("points", "pt"):
        return {"x": x_inches * 72.0, "y": y_inches * 72.0}
    elif unit == "native":
        vp = state.current_viewport()
        if vp is not None:
            xscale = getattr(vp, "xscale", [0, 1])
            yscale = getattr(vp, "yscale", [0, 1])
            if hasattr(xscale, '__len__') and len(xscale) >= 2:
                x_native = xscale[0] + npc_x * (xscale[1] - xscale[0])
            else:
                x_native = npc_x
            if hasattr(yscale, '__len__') and len(yscale) >= 2:
                y_native = yscale[0] + npc_y * (yscale[1] - yscale[0])
            else:
                y_native = npc_y
            return {"x": x_native, "y": y_native}
        return {"x": npc_x, "y": npc_y}
    else:
        return {"x": npc_x, "y": npc_y}


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
