"""Core drawing engine for grid_py -- Python port of R's grid drawing functions.

This module handles rendering grobs to matplotlib, porting functionality from
R's ``grid/R/grid.R`` (``grid.newpage``, ``grid.draw``, ``grid.record``,
``recordGrob``, ``grid.delay``, ``delayGrob``, ``grid.DLapply``,
``grid.refresh``, ``grid.locator``) and ``grid/R/grob.R`` (``drawGrob``,
``drawGTree``, ``drawGList``, ``preDraw``, ``postDraw``).

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
# Gpar -> matplotlib keyword conversion
# ---------------------------------------------------------------------------

# Mapping from R line-type names to matplotlib dash patterns
_LTY_MAP: Dict[str, str] = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dotdash": "dashdot",
    "longdash": (0, (10, 3)),  # type: ignore[dict-item]
    "twodash": (0, (5, 2, 10, 2)),  # type: ignore[dict-item]
}

_LINEEND_MAP: Dict[str, str] = {
    "round": "round",
    "butt": "butt",
    "square": "projecting",
}

_LINEJOIN_MAP: Dict[str, str] = {
    "round": "round",
    "mitre": "miter",
    "bevel": "bevel",
}


def _gpar_to_mpl(gp: Optional[Gpar]) -> Dict[str, Any]:
    """Convert a :class:`Gpar` instance to matplotlib keyword arguments.

    Parameters
    ----------
    gp : Gpar or None
        The graphical parameters to convert.  If ``None`` an empty dict
        is returned.

    Returns
    -------
    dict[str, Any]
        A dictionary of matplotlib-compatible keyword arguments suitable
        for passing to patch constructors, ``ax.plot``, ``ax.text``, etc.
    """
    if gp is None:
        return {}

    kw: Dict[str, Any] = {}

    # -- colour / fill -------------------------------------------------------
    col = gp.get("col", None)
    if col is not None:
        kw["color"] = col[0] if isinstance(col, (list, tuple)) else col

    fill = gp.get("fill", None)
    if fill is not None:
        kw["facecolor"] = fill[0] if isinstance(fill, (list, tuple)) else fill

    # -- alpha ---------------------------------------------------------------
    alpha = gp.get("alpha", None)
    if alpha is not None:
        kw["alpha"] = float(alpha[0] if isinstance(alpha, (list, tuple)) else alpha)

    # -- line properties -----------------------------------------------------
    lwd = gp.get("lwd", None)
    if lwd is not None:
        val = lwd[0] if isinstance(lwd, (list, tuple)) else lwd
        kw["linewidth"] = float(val)

    lty = gp.get("lty", None)
    if lty is not None:
        val = lty[0] if isinstance(lty, (list, tuple)) else lty
        kw["linestyle"] = _LTY_MAP.get(str(val), "solid")

    lineend = gp.get("lineend", None)
    if lineend is not None:
        val = lineend[0] if isinstance(lineend, (list, tuple)) else lineend
        capstyle = _LINEEND_MAP.get(str(val), "butt")
        kw["solid_capstyle"] = capstyle
        kw["dash_capstyle"] = capstyle

    linejoin = gp.get("linejoin", None)
    if linejoin is not None:
        val = linejoin[0] if isinstance(linejoin, (list, tuple)) else linejoin
        joinstyle = _LINEJOIN_MAP.get(str(val), "round")
        kw["solid_joinstyle"] = joinstyle
        kw["dash_joinstyle"] = joinstyle

    # -- font properties -----------------------------------------------------
    fontsize = gp.get("fontsize", None)
    cex = gp.get("cex", None)
    if fontsize is not None:
        fs = float(fontsize[0] if isinstance(fontsize, (list, tuple)) else fontsize)
        if cex is not None:
            cx = float(cex[0] if isinstance(cex, (list, tuple)) else cex)
            fs *= cx
        kw["fontsize"] = fs

    fontfamily = gp.get("fontfamily", None)
    if fontfamily is not None:
        val = fontfamily[0] if isinstance(fontfamily, (list, tuple)) else fontfamily
        kw["fontfamily"] = str(val)

    fontface = gp.get("fontface", None)
    if fontface is not None:
        val = fontface[0] if isinstance(fontface, (list, tuple)) else fontface
        val = int(val) if isinstance(val, (int, float)) else val
        if val in (2, "bold"):
            kw["fontweight"] = "bold"
        if val in (3, "italic", "oblique"):
            kw["fontstyle"] = "italic"
        if val in (4, "bold.italic"):
            kw["fontweight"] = "bold"
            kw["fontstyle"] = "italic"

    return kw


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
    ax: Any,
    gp: Optional[Gpar] = None,
    transform: Optional[np.ndarray] = None,
) -> None:
    """Render a single grob to a matplotlib ``Axes``.

    Dispatches on ``grob._grid_class`` to create the appropriate matplotlib
    artist and add it to *ax*.

    Parameters
    ----------
    grob : Grob
        The graphical object to render.
    ax : matplotlib.axes.Axes
        The target axes.
    gp : Gpar or None, optional
        Resolved graphical parameters (merged from context + grob).
    transform : numpy.ndarray or None, optional
        3x3 affine transform matrix; currently reserved for future use.

    Notes
    -----
    Unrecognised ``_grid_class`` values are silently ignored (a warning is
    emitted at DEBUG level).
    """
    import matplotlib.pyplot as plt  # noqa: F811 – local import
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    if ax is None:
        return

    cls = getattr(grob, "_grid_class", "grob")
    mpl_kw = _gpar_to_mpl(gp)

    # ---- rect -----------------------------------------------------------
    if cls == "rect":
        x = _unit_to_float(getattr(grob, "x", 0.0))
        y = _unit_to_float(getattr(grob, "y", 0.0))
        w = _unit_to_float(getattr(grob, "width", 1.0))
        h = _unit_to_float(getattr(grob, "height", 1.0))
        hjust = float(getattr(grob, "hjust", None) or 0.5)
        vjust = float(getattr(grob, "vjust", None) or 0.5)
        x0 = x - w * hjust
        y0 = y - h * vjust
        fc = mpl_kw.pop("facecolor", mpl_kw.pop("color", "white"))
        ec = mpl_kw.pop("color", "black") if "facecolor" in {} else mpl_kw.pop("color", "black")
        # Re-extract edge colour: col -> edgecolor, fill -> facecolor
        ec_raw = gp.get("col", None) if gp else None
        fc_raw = gp.get("fill", None) if gp else None
        ec_use = (ec_raw[0] if isinstance(ec_raw, (list, tuple)) else ec_raw) if ec_raw is not None else "black"
        fc_use = (fc_raw[0] if isinstance(fc_raw, (list, tuple)) else fc_raw) if fc_raw is not None else "white"
        patch_kw = {k: v for k, v in mpl_kw.items() if k not in ("color", "facecolor")}
        rect = mpatches.Rectangle(
            (x0, y0), w, h,
            facecolor=fc_use,
            edgecolor=ec_use,
            **patch_kw,
        )
        ax.add_patch(rect)

    # ---- circle ---------------------------------------------------------
    elif cls == "circle":
        x = _unit_to_float(getattr(grob, "x", 0.5))
        y = _unit_to_float(getattr(grob, "y", 0.5))
        r = _unit_to_float(getattr(grob, "r", 0.5))
        ec_raw = gp.get("col", None) if gp else None
        fc_raw = gp.get("fill", None) if gp else None
        ec_use = (ec_raw[0] if isinstance(ec_raw, (list, tuple)) else ec_raw) if ec_raw is not None else "black"
        fc_use = (fc_raw[0] if isinstance(fc_raw, (list, tuple)) else fc_raw) if fc_raw is not None else "white"
        patch_kw = {k: v for k, v in mpl_kw.items() if k not in ("color", "facecolor")}
        circ = mpatches.Circle(
            (x, y), r,
            facecolor=fc_use,
            edgecolor=ec_use,
            **patch_kw,
        )
        ax.add_patch(circ)

    # ---- lines / polyline ------------------------------------------------
    elif cls in ("lines", "polyline"):
        x = _unit_to_array(getattr(grob, "x", [0.0, 1.0]))
        y = _unit_to_array(getattr(grob, "y", [0.0, 1.0]))
        line_kw = {k: v for k, v in mpl_kw.items() if k not in ("facecolor",)}
        ax.plot(x, y, **line_kw)

    # ---- segments --------------------------------------------------------
    elif cls == "segments":
        x0 = _unit_to_array(getattr(grob, "x0", []))
        y0 = _unit_to_array(getattr(grob, "y0", []))
        x1 = _unit_to_array(getattr(grob, "x1", []))
        y1 = _unit_to_array(getattr(grob, "y1", []))
        segs = [[(sx0, sy0), (sx1, sy1)] for sx0, sy0, sx1, sy1
                in zip(x0, y0, x1, y1)]
        col = mpl_kw.pop("color", "black")
        lw = mpl_kw.pop("linewidth", 1.0)
        ls = mpl_kw.pop("linestyle", "solid")
        seg_kw = {k: v for k, v in mpl_kw.items()
                  if k not in ("facecolor",)}
        lc = LineCollection(segs, colors=col, linewidths=lw,
                            linestyles=ls, **seg_kw)
        ax.add_collection(lc)

    # ---- polygon ---------------------------------------------------------
    elif cls == "polygon":
        x = _unit_to_array(getattr(grob, "x", []))
        y = _unit_to_array(getattr(grob, "y", []))
        if len(x) > 0:
            verts = np.column_stack([x, y])
            ec_raw = gp.get("col", None) if gp else None
            fc_raw = gp.get("fill", None) if gp else None
            ec_use = (ec_raw[0] if isinstance(ec_raw, (list, tuple)) else ec_raw) if ec_raw is not None else "black"
            fc_use = (fc_raw[0] if isinstance(fc_raw, (list, tuple)) else fc_raw) if fc_raw is not None else "white"
            patch_kw = {k: v for k, v in mpl_kw.items() if k not in ("color", "facecolor")}
            poly = Polygon(verts, closed=True,
                           facecolor=fc_use, edgecolor=ec_use,
                           **patch_kw)
            ax.add_patch(poly)

    # ---- text ------------------------------------------------------------
    elif cls == "text":
        x = _unit_to_float(getattr(grob, "x", 0.5))
        y = _unit_to_float(getattr(grob, "y", 0.5))
        label = getattr(grob, "label", "")
        rot = float(getattr(grob, "rot", 0.0))
        hjust = getattr(grob, "hjust", None) or 0.5
        vjust = getattr(grob, "vjust", None) or 0.5
        ha_map = {0.0: "left", 0.5: "center", 1.0: "right"}
        va_map = {0.0: "bottom", 0.5: "center", 1.0: "top"}
        ha = ha_map.get(float(hjust), "center")
        va = va_map.get(float(vjust), "center")
        text_kw = {k: v for k, v in mpl_kw.items()
                   if k not in ("facecolor", "linewidth", "linestyle")}
        ax.text(x, y, str(label), rotation=rot,
                ha=ha, va=va, **text_kw)

    # ---- points ----------------------------------------------------------
    elif cls == "points":
        x = _unit_to_array(getattr(grob, "x", []))
        y = _unit_to_array(getattr(grob, "y", []))
        s = _unit_to_float(getattr(grob, "size", 1.0)) * 20  # rough pt-to-area scale
        c = mpl_kw.pop("color", "black")
        scatter_kw = {k: v for k, v in mpl_kw.items()
                      if k not in ("facecolor", "linewidth", "linestyle",
                                   "fontsize", "fontfamily")}
        ax.scatter(x, y, s=s, c=c, **scatter_kw)

    # ---- pathgrob --------------------------------------------------------
    elif cls == "pathgrob":
        x = _unit_to_array(getattr(grob, "x", []))
        y = _unit_to_array(getattr(grob, "y", []))
        path_id = getattr(grob, "pathId", None)
        rule = getattr(grob, "rule", "winding")
        if path_id is None:
            path_id = np.ones(len(x), dtype=int)
        else:
            path_id = np.atleast_1d(np.asarray(path_id, dtype=int))
        unique_ids = np.unique(path_id)
        vertices: List[Tuple[float, float]] = []
        codes: List[int] = []
        for pid in unique_ids:
            mask = path_id == pid
            px = x[mask]
            py = y[mask]
            for j, (vx, vy) in enumerate(zip(px, py)):
                vertices.append((float(vx), float(vy)))
                codes.append(MplPath.MOVETO if j == 0 else MplPath.LINETO)
            vertices.append((float(px[0]), float(py[0])))
            codes.append(MplPath.CLOSEPOLY)
        if vertices:
            mpath = MplPath(vertices, codes)
            ec_raw = gp.get("col", None) if gp else None
            fc_raw = gp.get("fill", None) if gp else None
            ec_use = (ec_raw[0] if isinstance(ec_raw, (list, tuple)) else ec_raw) if ec_raw is not None else "black"
            fc_use = (fc_raw[0] if isinstance(fc_raw, (list, tuple)) else fc_raw) if fc_raw is not None else "white"
            patch_kw = {k: v for k, v in mpl_kw.items() if k not in ("color", "facecolor")}
            pp = PathPatch(mpath, facecolor=fc_use, edgecolor=ec_use,
                           **patch_kw)
            ax.add_patch(pp)

    # ---- rastergrob ------------------------------------------------------
    elif cls == "rastergrob":
        image = getattr(grob, "image", None)
        x = _unit_to_float(getattr(grob, "x", 0.0))
        y = _unit_to_float(getattr(grob, "y", 0.0))
        w = _unit_to_float(getattr(grob, "width", 1.0))
        h = _unit_to_float(getattr(grob, "height", 1.0))
        if image is not None:
            extent = [x, x + w, y, y + h]
            interp = getattr(grob, "interpolate", True)
            ax.imshow(image, extent=extent, aspect="auto",
                      interpolation="bilinear" if interp else "nearest")

    # ---- null / no-op ---------------------------------------------------
    elif cls == "null":
        pass

    # ---- move.to / line.to -----------------------------------------------
    elif cls == "move.to":
        # Record the pen position on the axes (stash for a subsequent line.to)
        ax._grid_pen_x = _unit_to_float(getattr(grob, "x", 0.0))
        ax._grid_pen_y = _unit_to_float(getattr(grob, "y", 0.0))

    elif cls == "line.to":
        x1 = _unit_to_float(getattr(grob, "x", 0.0))
        y1 = _unit_to_float(getattr(grob, "y", 0.0))
        x0 = getattr(ax, "_grid_pen_x", 0.0)
        y0 = getattr(ax, "_grid_pen_y", 0.0)
        line_kw = {k: v for k, v in mpl_kw.items()
                   if k not in ("facecolor",)}
        ax.plot([x0, x1], [y0, y1], **line_kw)
        ax._grid_pen_x = x1
        ax._grid_pen_y = y1

    else:
        # Unknown class -- no-op with a debug-level warning.
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

        # Render to matplotlib
        fig, ax = state.get_device()
        if ax is not None:
            merged_gp = _merge_gpar(state.get_gpar(), x.gp)
            _render_grob(x, ax, gp=merged_gp)

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

        # Render the gTree itself to matplotlib (in case it has direct content)
        fig, ax = state.get_device()
        if ax is not None:
            merged_gp = _merge_gpar(state.get_gpar(), x.gp)
            _render_grob(x, ax, gp=merged_gp)

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
) -> None:
    """Clear the figure and start a fresh page.

    This is equivalent to R's ``grid.newpage()``.  If no matplotlib figure
    currently exists, a new one is created.  The viewport stack is reset to
    the root viewport.

    Parameters
    ----------
    recording : bool, optional
        If ``True`` (default) the display list is initialised for recording
        new operations.
    clear_dl : bool, optional
        If ``True`` (default) the existing display list is cleared.

    Notes
    -----
    High-level plotting functions should call this.  If you write a function
    that calls ``grid_newpage``, provide an argument to let users turn it off
    so they can draw into a parent viewport instead of starting a new page.
    """
    import matplotlib.pyplot as plt

    state = get_state()

    # Reset all state (viewport tree, gpar stack, display list)
    state.reset()

    # Obtain or create a matplotlib figure
    fig, ax = state.get_device()
    if fig is None:
        fig, ax = plt.subplots(1, 1)
        # Set up a unit-square axes spanning the full figure
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.axis("off")
        state.init_device(fig, ax)
    else:
        # Clear the existing axes
        ax.cla()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("auto")
        ax.axis("off")

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

    Equivalent to R's ``grid.locator()``.  In this Python port the function
    is a stub that uses ``matplotlib.pyplot.ginput`` when running in an
    interactive backend; otherwise it returns ``None``.

    Parameters
    ----------
    unit : str, optional
        The unit in which to return coordinates (default ``"native"``).
        Currently the raw device coordinates are returned regardless of
        *unit*; full unit conversion is a future enhancement.

    Returns
    -------
    dict[str, float] or None
        A dictionary ``{"x": ..., "y": ...}`` with the selected point's
        coordinates, or ``None`` if no point was selected or the backend
        is non-interactive.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    state = get_state()
    fig, ax = state.get_device()

    if fig is None:
        warnings.warn("grid_locator: no device available", stacklevel=2)
        return None

    backend = matplotlib.get_backend().lower()
    if "agg" in backend and "qt" not in backend and "tk" not in backend:
        # Non-interactive backend
        warnings.warn(
            "grid_locator: interactive point selection is not available "
            f"with the '{matplotlib.get_backend()}' backend",
            stacklevel=2,
        )
        return None

    try:
        pts = fig.ginput(n=1, timeout=0)
        if pts:
            return {"x": pts[0][0], "y": pts[0][1]}
    except Exception:
        pass
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
