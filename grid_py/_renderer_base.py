"""Abstract base class for all grid_py rendering backends.

Provides the shared coordinate system (viewport stack, unit resolution,
layout computation, NPC-to-device coordinate helpers) that every backend
needs.  Subclasses implement the actual drawing primitives and output
methods.

The coordinate convention matches R's grid: the unit square [0, 1] x [0, 1]
with the origin at the **bottom-left**.  Device coordinates use a top-left
origin (Y-flip is applied internally by :meth:`_y`).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["GridRenderer"]


class GridRenderer(ABC):
    """Abstract base for all grid_py rendering backends.

    Parameters
    ----------
    width : float
        Device width in inches.
    height : float
        Device height in inches.
    dpi : float
        Dots per inch.
    device_width : float or None
        Root viewport width in device units.  Defaults to ``width * dpi``
        (appropriate for raster surfaces).  Vector surfaces should pass
        ``width * 72.0``.
    device_height : float or None
        Root viewport height in device units.
    """

    def __init__(
        self,
        width: float = 7.0,
        height: float = 5.0,
        dpi: float = 150.0,
        device_width: Optional[float] = None,
        device_height: Optional[float] = None,
    ) -> None:
        self.width_in: float = width
        self.height_in: float = height
        self.dpi: float = dpi

        dw = float(device_width) if device_width is not None else width * dpi
        dh = float(device_height) if device_height is not None else height * dpi

        # Viewport transform stack.  Each entry is
        # ``(x0, y0, w, h, vp_obj)`` in device-unit space.
        # NPC [0, 1] maps to [x0, x0+w] x [y0, y0+h].
        # ``vp_obj`` is the Viewport object (``None`` for the root entry).
        self._vp_stack: List[Tuple[float, float, float, float, Any]] = [
            (0.0, 0.0, dw, dh, None)
        ]
        self._layout_stack: List[dict] = []
        self._layout_depth_stack: List[int] = []
        self._clip_stack: List[bool] = []
        self._path_collecting: bool = False

        # Pen position for move.to / line.to
        self._pen_x: float = 0.0
        self._pen_y: float = 0.0

        # Grob metadata (tooltip data attachment for web renderers)
        self._current_grob_metadata: Optional[dict] = None

    # ===================================================================== #
    # Grob metadata (data attachment for interactive features)              #
    # ===================================================================== #

    def set_grob_metadata(self, metadata: Optional[dict]) -> None:
        """Set metadata for the next draw_* call (tooltip data, etc.).

        Called by ``_render_grob`` before each drawing primitive.
        Subclasses (e.g. WebRenderer) use this to attach data to scene
        graph nodes.  CairoRenderer ignores it.
        """
        self._current_grob_metadata = metadata

    def clear_grob_metadata(self) -> None:
        """Clear grob metadata after the draw_* call completes."""
        self._current_grob_metadata = None

    # ===================================================================== #
    # Public viewport-bounds API (replaces direct _vp_stack access)         #
    # ===================================================================== #

    def get_viewport_bounds(self) -> Tuple[float, float, float, float]:
        """Return ``(x0, y0, pw, ph)`` of the current viewport in device units."""
        e = self._vp_stack[-1]
        return (e[0], e[1], e[2], e[3])

    def get_viewport_object(self) -> Any:
        """Return the Viewport object of the current viewport, or ``None``."""
        return self._vp_stack[-1][4] if len(self._vp_stack[-1]) > 4 else None

    # ===================================================================== #
    # Viewport management (shared across all backends)                      #
    # ===================================================================== #

    def push_viewport(self, vp: Any) -> None:
        """Push a viewport, updating the coordinate transform.

        Handles three viewport types:
        1. Layout viewport (has ``_layout`` with grid info) -- stores grid
        2. Child viewport with ``layout_pos_row/col`` -- uses parent grid
        3. Simple viewport with x/y/width/height -- direct position
        """
        from ._units import Unit

        x0, y0, pw, ph, *_vp_rest = self._vp_stack[-1]

        layout = getattr(vp, "_layout", None)
        layout_pos_row = getattr(vp, "_layout_pos_row", None)
        layout_pos_col = getattr(vp, "_layout_pos_col", None)

        if layout is not None:
            respect = getattr(layout, "respect", False)
            grid_info = self._compute_grid(layout, pw, ph, respect=bool(respect))
            self._vp_stack.append((x0, y0, pw, ph, vp))
            self._layout_stack.append(grid_info)
            self._clip_stack.append(False)
            self._layout_depth_stack.append(len(self._vp_stack))
            return

        if layout_pos_row is not None and layout_pos_col is not None:
            if self._layout_stack:
                grid = self._layout_stack[-1]
                col_starts = grid["col_starts"]
                col_widths = grid["col_widths"]
                row_starts = grid["row_starts"]
                row_heights = grid["row_heights"]

                if isinstance(layout_pos_row, (list, tuple)):
                    t, b = int(layout_pos_row[0]) - 1, int(layout_pos_row[1]) - 1
                else:
                    t = b = int(layout_pos_row) - 1
                if isinstance(layout_pos_col, (list, tuple)):
                    l, r = int(layout_pos_col[0]) - 1, int(layout_pos_col[1]) - 1
                else:
                    l = r = int(layout_pos_col) - 1

                cell_x0 = x0 + (col_starts[l] if l < len(col_starts) else 0)
                cell_y0 = y0 + (row_starts[t] if t < len(row_starts) else 0)
                cell_w = sum(col_widths[l:r + 1]) if r < len(col_widths) else pw
                cell_h = sum(row_heights[t:b + 1]) if b < len(row_heights) else ph

                self._vp_stack.append((cell_x0, cell_y0, cell_w, cell_h, vp))
                self._do_apply_clip(vp, cell_x0, cell_y0, cell_w, cell_h)
                return

        # Simple viewport with explicit x/y/width/height
        vp_x_raw = getattr(vp, "_x", None)
        vp_y_raw = getattr(vp, "_y", None)
        vp_w_raw = getattr(vp, "_width", None)
        vp_h_raw = getattr(vp, "_height", None)

        vp_x = self.resolve_x(vp_x_raw) if vp_x_raw is not None else 0.5
        vp_y = self.resolve_y(vp_y_raw) if vp_y_raw is not None else 0.5
        vp_w = self.resolve_w(vp_w_raw) if vp_w_raw is not None else 1.0
        vp_h = self.resolve_h(vp_h_raw) if vp_h_raw is not None else 1.0

        just = getattr(vp, "_just", (0.5, 0.5))
        if isinstance(just, (list, tuple)) and len(just) >= 2:
            hjust, vjust = float(just[0]), float(just[1])
        else:
            hjust, vjust = 0.5, 0.5

        new_w = vp_w * pw
        new_h = vp_h * ph
        new_x0 = x0 + vp_x * pw - hjust * new_w
        new_y0 = y0 + vp_y * ph - vjust * new_h

        self._vp_stack.append((new_x0, new_y0, new_w, new_h, vp))
        self._do_apply_clip(vp, new_x0, new_y0, new_w, new_h)

    def _do_apply_clip(self, vp: Any, x0: float, y0: float, w: float, h: float) -> None:
        """Check whether clipping is requested and delegate to the backend."""
        clip = getattr(vp, "_clip", None)
        if clip is True or clip == "on":
            self._apply_clip_rect(x0, y0, w, h)
            self._clip_stack.append(True)
        else:
            self._clip_stack.append(False)

    def pop_viewport(self) -> None:
        """Pop the current viewport and restore clipping/layout state."""
        if len(self._vp_stack) > 1:
            depth_stack = self._layout_depth_stack
            if depth_stack and depth_stack[-1] == len(self._vp_stack):
                depth_stack.pop()
                if self._layout_stack:
                    self._layout_stack.pop()
            self._vp_stack.pop()
            if self._clip_stack:
                had_clip = self._clip_stack.pop()
                if had_clip:
                    self._restore_clip()

    # ===================================================================== #
    # Layout computation (shared)                                           #
    # ===================================================================== #

    def _compute_grid(
        self, layout: Any, parent_w: float, parent_h: float,
        respect: bool = False,
    ) -> dict:
        """Compute row/column positions for a GridLayout within the parent.

        Delegates to :func:`._layout._calc_layout_sizes` which implements
        the full three-phase layout algorithm from R ``layout.c``.
        """
        from ._layout import _calc_layout_sizes, GridLayout

        if isinstance(layout, GridLayout):
            col_widths, row_heights = _calc_layout_sizes(
                layout, parent_w, parent_h, self.dpi,
            )
        else:
            nrow = getattr(layout, "nrow", 1)
            ncol = getattr(layout, "ncol", 1)
            col_widths = self._resolve_sizes(
                getattr(layout, "widths", None), ncol, parent_w,
            )
            row_heights = self._resolve_sizes(
                getattr(layout, "heights", None), nrow, parent_h,
            )

        ncol = len(col_widths)
        nrow = len(row_heights)
        col_starts = [sum(col_widths[:i]) for i in range(ncol)]
        row_starts = [sum(row_heights[:i]) for i in range(nrow)]

        return {
            "col_starts": col_starts, "col_widths": col_widths,
            "row_starts": row_starts, "row_heights": row_heights,
        }

    def _resolve_sizes_with_scale(
        self, unit_obj: Any, n: int, total: float, null_scale: float,
    ) -> list:
        """Resolve sizes using a fixed scale for null units (respect mode)."""
        if unit_obj is None:
            return [null_scale] * n
        from ._units import Unit, _INCHES_PER
        if not isinstance(unit_obj, Unit):
            return [null_scale] * n
        sizes = []
        for v, t in zip(unit_obj._values, unit_obj._units):
            if t == "npc":
                sizes.append(float(v) * total)
            elif t in _INCHES_PER:
                sizes.append(float(v) * _INCHES_PER[t] * self.dpi)
            else:
                sizes.append(float(v) * null_scale)
        return sizes

    def _resolve_sizes(self, unit_obj: Any, n: int, total: float) -> list:
        """Resolve a Unit vector to device sizes, distributing null units.

        Mirrors R's grid unit resolution: absolute units are converted to
        device pixels first; the remaining space is distributed among
        ``"null"`` units proportionally.
        """
        if unit_obj is None:
            return [total / n] * n

        from ._units import Unit, _INCHES_PER
        if not isinstance(unit_obj, Unit):
            return [total / n] * n

        vals = unit_obj._values
        types = (
            unit_obj._units
            if hasattr(unit_obj, "_units")
            else getattr(unit_obj, "_types", ["null"] * len(vals))
        )

        abs_sizes: Dict[int, float] = {}
        abs_total = 0.0
        null_total = 0.0

        for i, (v, t) in enumerate(zip(vals, types)):
            if t == "npc":
                px = float(v) * total
                abs_sizes[i] = px
                abs_total += px
            elif t in _INCHES_PER:
                px = float(v) * _INCHES_PER[t] * self.dpi
                abs_sizes[i] = px
                abs_total += px
            elif t == "null":
                null_total += float(v)
            else:
                null_total += float(v)

        remaining = max(total - abs_total, 0.0)
        if null_total == 0:
            null_total = 1.0

        sizes = []
        for i, (v, t) in enumerate(zip(vals, types)):
            if i in abs_sizes:
                sizes.append(abs_sizes[i])
            else:
                sizes.append(float(v) / null_total * remaining)
        return sizes

    # ===================================================================== #
    # Unit resolution (shared -- NO Cairo dependency)                       #
    # ===================================================================== #

    def _resolve_to_npc(
        self,
        unit_obj: Any,
        axis: str,
        is_dim: bool,
        gp: Optional[Any] = None,
    ) -> float:
        """Resolve a single :class:`Unit` value to an NPC float.

        Mirrors R's two-phase unit resolution (``unit.c:transformX/Y``):
        first converts the unit to inches using the current viewport's
        physical dimensions, then normalises to NPC by dividing by the
        viewport size in inches.
        """
        from ._units import Unit, _INCHES_PER

        if not isinstance(unit_obj, Unit):
            return float(unit_obj)

        value = float(unit_obj._values[0])
        utype = unit_obj._units[0]

        # Viewport physical size (accounting for rotation).
        x0, y0, pw, ph, vp_obj = self._vp_stack[-1]
        angle = float(getattr(vp_obj, "_angle", 0)) if vp_obj is not None else 0.0
        sin_a = abs(math.sin(math.radians(angle)))
        cos_a = abs(math.cos(math.radians(angle)))
        eff_w = pw * cos_a + ph * sin_a
        eff_h = ph * cos_a + pw * sin_a
        vp_px = eff_w if axis == "x" else eff_h
        vp_inches = vp_px / self.dpi
        if vp_inches == 0:
            return 0.0

        if utype == "npc":
            return value

        if utype in _INCHES_PER:
            inches = value * _INCHES_PER[utype]
            return inches / vp_inches

        if utype == "native":
            if vp_obj is not None:
                scale = (
                    getattr(vp_obj, "_xscale", [0, 1])
                    if axis == "x"
                    else getattr(vp_obj, "_yscale", [0, 1])
                )
            else:
                scale = [0, 1]
            smin, smax = float(scale[0]), float(scale[1])
            srange = smax - smin
            if srange == 0:
                return 0.0
            if is_dim:
                return value / srange
            return (value - smin) / srange

        if utype in ("char", "lines"):
            fontsize = 12.0
            cex = 1.0
            lineheight = 1.2
            if gp is not None:
                fs = gp.get("fontsize", None)
                if fs is not None:
                    fontsize = float(fs[0] if isinstance(fs, (list, tuple)) else fs)
                cx = gp.get("cex", None)
                if cx is not None:
                    cex = float(cx[0] if isinstance(cx, (list, tuple)) else cx)
                if utype == "lines":
                    lh = gp.get("lineheight", None)
                    if lh is not None:
                        lineheight = float(lh[0] if isinstance(lh, (list, tuple)) else lh)
            pts = value * fontsize * cex
            if utype == "lines":
                pts *= lineheight
            inches = pts / 72.0
            return inches / vp_inches

        if utype == "null":
            return 0.0

        if utype == "snpc":
            other_px = eff_h if axis == "x" else eff_w
            min_inches = min(vp_px, other_px) / self.dpi
            inches = value * min_inches
            return inches / vp_inches

        if utype in ("strwidth", "strheight", "strascent", "strdescent"):
            from ._size import calc_string_metric
            text = str(unit_obj._data[0]) if unit_obj._data[0] is not None else ""
            m = calc_string_metric(text, gp=gp)
            if utype == "strwidth":
                inches = m["width"]
            elif utype == "strheight":
                inches = m["ascent"] + m["descent"]
            elif utype == "strascent":
                inches = m["ascent"]
            else:
                inches = m["descent"]
            return (inches * value) / vp_inches

        if utype in ("grobwidth", "grobheight", "grobascent", "grobdescent"):
            from ._size import width_details, height_details, ascent_details, descent_details
            grob_ref = unit_obj._data[0]
            if grob_ref is not None:
                dispatch = {
                    "grobwidth": width_details,
                    "grobheight": height_details,
                    "grobascent": ascent_details,
                    "grobdescent": descent_details,
                }
                metric_unit = dispatch[utype](grob_ref)
                if metric_unit is not None and hasattr(metric_unit, "_units"):
                    return self._resolve_to_npc(
                        metric_unit, axis=axis, is_dim=True, gp=gp
                    ) * value
            return 0.0

        if utype == "sum":
            child = unit_obj._data[0]
            if isinstance(child, Unit):
                total = 0.0
                for j in range(len(child)):
                    elem = Unit(child._values[j], child._units[j], data=child._data[j])
                    total += self._resolve_to_npc(elem, axis, is_dim, gp)
                return total * value
            return 0.0

        if utype == "min":
            child = unit_obj._data[0]
            if isinstance(child, Unit):
                best = float("inf")
                for j in range(len(child)):
                    elem = Unit(child._values[j], child._units[j], data=child._data[j])
                    best = min(best, self._resolve_to_npc(elem, axis, is_dim, gp))
                return best * value if best != float("inf") else 0.0
            return 0.0

        if utype == "max":
            child = unit_obj._data[0]
            if isinstance(child, Unit):
                best = float("-inf")
                for j in range(len(child)):
                    elem = Unit(child._values[j], child._units[j], data=child._data[j])
                    best = max(best, self._resolve_to_npc(elem, axis, is_dim, gp))
                return best * value if best != float("-inf") else 0.0
            return 0.0

        # Fallback: treat as NPC
        return value

    # -- public convenience: resolve_to_npc (for external callers) --

    def resolve_to_npc(
        self,
        unit_obj: Any,
        axis: str = "x",
        is_dim: bool = False,
        gp: Optional[Any] = None,
    ) -> float:
        """Public interface to unit resolution.

        Equivalent to ``_resolve_to_npc`` but with a stable public name.
        """
        return self._resolve_to_npc(unit_obj, axis=axis, is_dim=is_dim, gp=gp)

    # -- public convenience: scalar resolvers --

    def resolve_x(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to an NPC x-coordinate."""
        from ._units import Unit
        if not isinstance(val, Unit):
            return float(val)
        return self._resolve_to_npc(val, axis="x", is_dim=False, gp=gp)

    def resolve_y(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to an NPC y-coordinate."""
        from ._units import Unit
        if not isinstance(val, Unit):
            return float(val)
        return self._resolve_to_npc(val, axis="y", is_dim=False, gp=gp)

    def resolve_w(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to an NPC width (fraction of viewport)."""
        from ._units import Unit
        if not isinstance(val, Unit):
            return float(val)
        return self._resolve_to_npc(val, axis="x", is_dim=True, gp=gp)

    def resolve_h(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to an NPC height (fraction of viewport)."""
        from ._units import Unit
        if not isinstance(val, Unit):
            return float(val)
        return self._resolve_to_npc(val, axis="y", is_dim=True, gp=gp)

    # -- public convenience: array resolvers --

    def resolve_x_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of NPC x-coordinates."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                elem = Unit(val._values[i], val._units[i], data=val._data[i])
                out[i] = self._resolve_to_npc(elem, "x", False, gp)
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_x(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def resolve_y_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of NPC y-coordinates."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                elem = Unit(val._values[i], val._units[i], data=val._data[i])
                out[i] = self._resolve_to_npc(elem, "y", False, gp)
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_y(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def resolve_w_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of NPC widths."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                elem = Unit(val._values[i], val._units[i], data=val._data[i])
                out[i] = self._resolve_to_npc(elem, "x", True, gp)
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_w(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def resolve_h_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of NPC heights."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                elem = Unit(val._values[i], val._units[i], data=val._data[i])
                out[i] = self._resolve_to_npc(elem, "y", True, gp)
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_h(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    # ===================================================================== #
    # Coordinate helpers (NPC → device)                                     #
    # ===================================================================== #

    def _x(self, npc: float) -> float:
        """Convert NPC x -> device x (within current viewport)."""
        x0, y0, pw, ph, *_ = self._vp_stack[-1]
        return x0 + npc * pw

    def _y(self, npc: float) -> float:
        """Convert NPC y -> device y (Y-flip: 0=bottom, 1=top).

        In device coords, y=0 is the top and increases downward.
        In NPC, y=0 is the bottom and y=1 is the top.
        """
        x0, y0, pw, ph, *_ = self._vp_stack[-1]
        return y0 + (1.0 - npc) * ph

    def _sx(self, npc: float) -> float:
        """Scale a width from NPC to device units."""
        return npc * self._vp_stack[-1][2]

    def _sy(self, npc: float) -> float:
        """Scale a height from NPC to device units."""
        return npc * self._vp_stack[-1][3]

    # ===================================================================== #
    # Abstract methods: backend-specific clipping                           #
    # ===================================================================== #

    @abstractmethod
    def _apply_clip_rect(self, x0: float, y0: float, w: float, h: float) -> None:
        """Apply a rectangular clip region in device coordinates."""
        ...

    @abstractmethod
    def _restore_clip(self) -> None:
        """Restore from the most recent clip save."""
        ...

    # ===================================================================== #
    # Abstract methods: graphics state save/restore                         #
    # ===================================================================== #

    @abstractmethod
    def save_state(self) -> None:
        """Save the current graphics state (for path collection, etc.)."""
        ...

    @abstractmethod
    def restore_state(self) -> None:
        """Restore the previously saved graphics state."""
        ...

    # ===================================================================== #
    # Abstract methods: path collection (fill/stroke grobs)                 #
    # ===================================================================== #

    @abstractmethod
    def begin_path_collect(self, rule: str = "winding") -> None:
        """Enter path-collecting mode."""
        ...

    @abstractmethod
    def end_path_stroke(self, gp: Optional[Any] = None) -> None:
        """End path collection with stroke only."""
        ...

    @abstractmethod
    def end_path_fill(self, gp: Optional[Any] = None) -> None:
        """End path collection with fill only."""
        ...

    @abstractmethod
    def end_path_fill_stroke(self, gp: Optional[Any] = None) -> None:
        """End path collection with fill then stroke."""
        ...

    # ===================================================================== #
    # Abstract methods: drawing primitives                                  #
    # ===================================================================== #

    @abstractmethod
    def draw_rect(self, x: float, y: float, w: float, h: float,
                  hjust: float = 0.5, vjust: float = 0.5,
                  gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_circle(self, x: float, y: float, r: float,
                    gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_line(self, x: "np.ndarray", y: "np.ndarray",
                  gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_polyline(self, x: "np.ndarray", y: "np.ndarray",
                      id_: Optional["np.ndarray"] = None,
                      gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_segments(self, x0: "np.ndarray", y0: "np.ndarray",
                      x1: "np.ndarray", y1: "np.ndarray",
                      gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_polygon(self, x: "np.ndarray", y: "np.ndarray",
                     gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_path(self, x: "np.ndarray", y: "np.ndarray",
                  path_id: "np.ndarray", rule: str = "winding",
                  gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_text(self, x: float, y: float, label: str,
                  rot: float = 0.0, hjust: float = 0.5, vjust: float = 0.5,
                  gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_points(self, x: "np.ndarray", y: "np.ndarray",
                    size: float = 1.0, pch: Any = 19,
                    gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def draw_raster(self, image: Any, x: float, y: float,
                    w: float, h: float,
                    interpolate: bool = True) -> None: ...

    @abstractmethod
    def draw_roundrect(self, x: float, y: float, w: float, h: float,
                       r: float = 0.0, hjust: float = 0.5, vjust: float = 0.5,
                       gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def move_to(self, x: float, y: float) -> None: ...

    @abstractmethod
    def line_to(self, x: float, y: float,
                gp: Optional[Any] = None) -> None: ...

    # ===================================================================== #
    # Abstract methods: clipping (explicit push/pop)                        #
    # ===================================================================== #

    @abstractmethod
    def push_clip(self, x0: float, y0: float, x1: float, y1: float) -> None: ...

    @abstractmethod
    def pop_clip(self) -> None: ...

    # ===================================================================== #
    # Abstract methods: text metrics                                        #
    # ===================================================================== #

    @abstractmethod
    def text_extents(self, text: str,
                     gp: Optional[Any] = None) -> Dict[str, float]:
        """Return ``{'ascent', 'descent', 'width'}`` in inches."""
        ...

    # ===================================================================== #
    # Abstract methods: masking                                             #
    # ===================================================================== #

    @abstractmethod
    def render_mask(self, mask_grob: Any) -> Any: ...

    @abstractmethod
    def apply_mask(self, mask_surface: Any,
                   mask_type: str = "alpha") -> None: ...

    # ===================================================================== #
    # Abstract methods: output / surface management                         #
    # ===================================================================== #

    @abstractmethod
    def new_page(self, bg: Any = "white") -> None: ...

    @abstractmethod
    def finish(self) -> None: ...
