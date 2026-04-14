"""Abstract base class for all grid_py rendering backends.

Provides the shared coordinate system (viewport transform stack, unit
resolution to **inches**, layout computation, and inches-to-device
coordinate helpers) that every backend needs.  Subclasses implement the
actual drawing primitives and output methods.

The coordinate convention matches R's grid: the unit square [0, 1] × [0, 1]
with the origin at the **bottom-left**.  Device coordinates use a top-left
origin (Y-flip is applied internally by :meth:`_to_dev_x` / :meth:`_to_dev_y`).

Coordinate pipeline (matches R's grid/src/unit.c + viewport.c):
    Unit → _resolve_to_inches() → inches within viewport
         → trans(location, viewport_transform) → absolute inches on device
         → _to_dev_x/_to_dev_y → device coordinates (pixels or points)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vp_calc import (
    ViewportContext,
    ViewportTransformResult,
    calc_root_transform,
    calc_viewport_transform,
    identity,
    location,
    trans,
    transform_x_to_inches,
    transform_y_to_inches,
    transform_width_to_inches,
    transform_height_to_inches,
    _transform_to_inches,
    _INCHES_PER,
)

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
        self._device_width: float = dw
        self._device_height: float = dh

        # Device dimensions in CM (used by calcViewportTransform)
        self._device_width_cm: float = width * 2.54
        self._device_height_cm: float = height * 2.54

        # Scale factor: device units per inch
        # For raster surfaces: dpi.  For vector surfaces (PDF/SVG): 72.
        self._dev_units_per_inch: float = dw / width if width > 0 else dpi

        # Viewport transform stack.  Each entry is a ViewportTransformResult
        # containing width_cm, height_cm, rotation_angle, 3×3 transform matrix,
        # and ViewportContext (xscale/yscale).
        # The root entry represents the device itself.
        root_vtr = calc_root_transform(self._device_width_cm, self._device_height_cm)
        self._vp_transform_stack: List[ViewportTransformResult] = [root_vtr]

        # Keep a parallel list of viewport objects for attribute access
        self._vp_obj_stack: List[Any] = [None]

        self._layout_stack: List[dict] = []
        self._layout_depth_stack: List[int] = []
        self._clip_stack: List[bool] = []
        self._path_collecting: bool = False

        # Pen position for move.to / line.to (in device coords now)
        self._pen_x: float = 0.0
        self._pen_y: float = 0.0

        # Grob metadata (tooltip data attachment for web renderers)
        self._current_grob_metadata: Optional[dict] = None

        # ---- Backward compatibility: old _vp_stack API ----
        # Some external code may still access _vp_stack.  We provide a
        # property that synthesises the old (x0, y0, pw, ph, vp_obj) tuples
        # from the new transform stack.

    @property
    def _vp_stack(self) -> List[Tuple[float, float, float, float, Any]]:
        """Backward-compatible viewport stack (device-unit tuples).

        Synthesised from the new transform stack.  Each entry is
        ``(x0, y0, pw, ph, vp_obj)`` where (x0, y0) is the bottom-left
        corner in device units and (pw, ph) are dimensions in device units.
        """
        result = []
        for i, vtr in enumerate(self._vp_transform_stack):
            vp_obj = self._vp_obj_stack[i] if i < len(self._vp_obj_stack) else None
            # Bottom-left corner in inches (origin of viewport)
            bl = trans(location(0.0, 0.0), vtr.transform)
            w_in = vtr.width_cm / 2.54
            h_in = vtr.height_cm / 2.54
            x0 = bl[0] * self._dev_units_per_inch
            y0_bottom = bl[1] * self._dev_units_per_inch
            pw = w_in * self._dev_units_per_inch
            ph = h_in * self._dev_units_per_inch
            # Convert to top-left origin for device coords
            y0_device = self._device_height - y0_bottom - ph
            result.append((x0, y0_device, pw, ph, vp_obj))
        return result

    # ===================================================================== #
    # Grob metadata (data attachment for interactive features)              #
    # ===================================================================== #

    def set_grob_metadata(self, metadata: Optional[dict]) -> None:
        self._current_grob_metadata = metadata

    def clear_grob_metadata(self) -> None:
        self._current_grob_metadata = None

    # ===================================================================== #
    # Public viewport-bounds API                                            #
    # ===================================================================== #

    def get_viewport_bounds(self) -> Tuple[float, float, float, float]:
        """Return ``(x0, y0, pw, ph)`` of the current viewport in device units.

        Uses the backward-compatible synthesised bounds.
        """
        stack = self._vp_stack
        e = stack[-1]
        return (e[0], e[1], e[2], e[3])

    def get_viewport_object(self) -> Any:
        """Return the Viewport object of the current viewport, or ``None``."""
        return self._vp_obj_stack[-1] if self._vp_obj_stack else None

    def get_current_vtr(self) -> ViewportTransformResult:
        """Return the current viewport's transform result."""
        return self._vp_transform_stack[-1]

    # ===================================================================== #
    # Gpar extraction helpers                                               #
    # ===================================================================== #

    def _gpar_font_params(self, gp: Optional[Any] = None) -> Tuple[float, float, float]:
        """Extract (fontsize, cex, lineheight) from gpar for unit resolution."""
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
            lh = gp.get("lineheight", None)
            if lh is not None:
                lineheight = float(lh[0] if isinstance(lh, (list, tuple)) else lh)
        return fontsize, cex, lineheight

    # ===================================================================== #
    # Viewport management (shared across all backends)                      #
    # ===================================================================== #

    def push_viewport(self, vp: Any) -> None:
        """Push a viewport, computing its 3×3 transform via calcViewportTransform.

        Handles three viewport types:
        1. Layout viewport (has ``_layout``) -- stores grid, same transform
        2. Child viewport with ``layout_pos_row/col`` -- uses parent grid
        3. Simple viewport with x/y/width/height -- full transform calc
        """
        from ._units import Unit

        parent_vtr = self._vp_transform_stack[-1]

        layout = getattr(vp, "_layout", None)
        layout_pos_row = getattr(vp, "_layout_pos_row", None)
        layout_pos_col = getattr(vp, "_layout_pos_col", None)

        # --- Case 1: Layout viewport ---
        if layout is not None:
            # Layout viewport uses same bounds as parent but stores grid info.
            # Compute grid in device units for layout children.
            w_dev = parent_vtr.width_cm / 2.54 * self._dev_units_per_inch
            h_dev = parent_vtr.height_cm / 2.54 * self._dev_units_per_inch
            respect = getattr(layout, "respect", False)
            grid_info = self._compute_grid(layout, w_dev, h_dev, respect=bool(respect))

            # The layout viewport itself has the same transform as parent
            # but we create a new VTR with the vp's xscale/yscale
            xscale = getattr(vp, "_xscale", [0.0, 1.0])
            yscale = getattr(vp, "_yscale", [0.0, 1.0])
            vtr = ViewportTransformResult(
                width_cm=parent_vtr.width_cm,
                height_cm=parent_vtr.height_cm,
                rotation_angle=parent_vtr.rotation_angle,
                transform=parent_vtr.transform.copy(),
                vpc=ViewportContext(
                    xscale=(float(xscale[0]), float(xscale[1])),
                    yscale=(float(yscale[0]), float(yscale[1])),
                ),
            )
            self._vp_transform_stack.append(vtr)
            self._vp_obj_stack.append(vp)
            self._layout_stack.append(grid_info)
            self._clip_stack.append(False)
            self._layout_depth_stack.append(len(self._vp_transform_stack))
            return

        # --- Case 2: Layout-positioned child ---
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

                cell_x0_dev = col_starts[l] if l < len(col_starts) else 0
                cell_y0_dev = row_starts[t] if t < len(row_starts) else 0
                cell_w_dev = sum(col_widths[l:r + 1]) if r < len(col_widths) else 0
                cell_h_dev = sum(row_heights[t:b + 1]) if b < len(row_heights) else 0

                # Convert device units to inches for the transform
                cell_w_in = cell_w_dev / self._dev_units_per_inch
                cell_h_in = cell_h_dev / self._dev_units_per_inch

                # The cell's bottom-left in the parent's coordinate system
                # Layout grid uses device coords with top-left origin;
                # we need to convert to the parent's inches system.
                parent_h_in = parent_vtr.height_cm / 2.54

                # Cell position in parent's NPC then inches
                parent_w_dev = parent_vtr.width_cm / 2.54 * self._dev_units_per_inch
                parent_h_dev = parent_vtr.height_cm / 2.54 * self._dev_units_per_inch
                cell_x_in = cell_x0_dev / self._dev_units_per_inch
                # Device y is top-down; convert to bottom-up inches
                cell_y_in = parent_h_in - (cell_y0_dev + cell_h_dev) / self._dev_units_per_inch

                # Build a simple translation transform for the cell
                from ._vp_calc import translation, multiply
                cell_translation = translation(cell_x_in, cell_y_in)
                cell_transform = multiply(cell_translation, parent_vtr.transform)

                xscale = getattr(vp, "_xscale", [0.0, 1.0])
                yscale = getattr(vp, "_yscale", [0.0, 1.0])
                vtr = ViewportTransformResult(
                    width_cm=cell_w_in * 2.54,
                    height_cm=cell_h_in * 2.54,
                    rotation_angle=parent_vtr.rotation_angle,
                    transform=cell_transform,
                    vpc=ViewportContext(
                        xscale=(float(xscale[0]), float(xscale[1])),
                        yscale=(float(yscale[0]), float(yscale[1])),
                    ),
                )
                self._vp_transform_stack.append(vtr)
                self._vp_obj_stack.append(vp)
                self._do_apply_clip_vtr(vp, vtr)
                return

        # --- Case 3: Simple viewport with x/y/width/height ---
        # Use calc_viewport_transform (port of R's calcViewportTransform)
        fontsize, cex, lineheight = self._gpar_font_params(None)

        vtr = calc_viewport_transform(
            vp,
            parent_vtr.transform,
            parent_vtr.width_cm,
            parent_vtr.height_cm,
            parent_vtr.rotation_angle,
            parent_vtr.vpc,
            gc_fontsize=fontsize,
            gc_cex=cex,
            gc_lineheight=lineheight,
            str_metric_fn=self._str_metric_fn,
            grob_metric_fn=None,
        )
        self._vp_transform_stack.append(vtr)
        self._vp_obj_stack.append(vp)
        self._do_apply_clip_vtr(vp, vtr)

    def _str_metric_fn(self, text: str, gp: Any) -> Dict[str, float]:
        """String metric callback for unit resolution."""
        return self.text_extents(text, gp=gp)

    def _do_apply_clip_vtr(self, vp: Any, vtr: ViewportTransformResult) -> None:
        """Apply clipping for a viewport using its transform."""
        clip = getattr(vp, "_clip", None)
        if clip is True or clip == "on":
            # Compute clip rect in device coords from the viewport bounds
            bl = trans(location(0.0, 0.0), vtr.transform)
            w_in = vtr.width_cm / 2.54
            h_in = vtr.height_cm / 2.54
            x0 = bl[0] * self._dev_units_per_inch
            y0_bottom = bl[1] * self._dev_units_per_inch
            pw = w_in * self._dev_units_per_inch
            ph = h_in * self._dev_units_per_inch
            # Convert to device top-left origin
            y0_device = self._device_height - y0_bottom - ph
            self._apply_clip_rect(x0, y0_device, pw, ph)
            self._clip_stack.append(True)
        else:
            self._clip_stack.append(False)

    def pop_viewport(self) -> None:
        """Pop the current viewport and restore clipping/layout state."""
        if len(self._vp_transform_stack) > 1:
            depth_stack = self._layout_depth_stack
            if depth_stack and depth_stack[-1] == len(self._vp_transform_stack):
                depth_stack.pop()
                if self._layout_stack:
                    self._layout_stack.pop()
            self._vp_transform_stack.pop()
            self._vp_obj_stack.pop()
            if self._clip_stack:
                had_clip = self._clip_stack.pop()
                if had_clip:
                    self._restore_clip()

    def pop_viewport_to_root(self) -> None:
        """Pop all viewports back to the root (device-level) entry."""
        while len(self._vp_transform_stack) > 1:
            self.pop_viewport()

    # ===================================================================== #
    # Layout computation (shared)                                           #
    # ===================================================================== #

    def _compute_grid(
        self, layout: Any, parent_w: float, parent_h: float,
        respect: bool = False,
    ) -> dict:
        """Compute row/column positions for a GridLayout within the parent."""
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

    def _resolve_sizes(self, unit_obj: Any, n: int, total: float) -> list:
        """Resolve a Unit vector to device sizes, distributing null units."""
        if unit_obj is None:
            return [total / n] * n

        from ._units import Unit
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
                px = float(v) * _INCHES_PER[t] * self._dev_units_per_inch
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
    # Unit resolution: to INCHES (port of unit.c:transform)                 #
    # ===================================================================== #

    def _get_scale(self) -> float:
        """Return the current GSS_SCALE zoom factor (default 1.0)."""
        try:
            from ._state import get_state
            return get_state()._scale
        except Exception:
            return 1.0

    def _resolve_to_inches(
        self,
        unit_obj: Any,
        axis: str,
        is_dim: bool,
        gp: Optional[Any] = None,
    ) -> float:
        """Resolve a single :class:`Unit` value to inches.

        Port of R's unit.c transformXtoINCHES / transformYtoINCHES.
        Uses the current viewport's transform context (widthCM, heightCM,
        ViewportContext) for the conversion.
        """
        from ._units import Unit

        if not isinstance(unit_obj, Unit):
            return float(unit_obj)

        vtr = self._vp_transform_stack[-1]
        fontsize, cex, lineheight = self._gpar_font_params(gp)

        return _transform_to_inches(
            unit_obj, 0, vtr.vpc,
            fontsize, cex, lineheight,
            this_cm=vtr.width_cm if axis == "x" else vtr.height_cm,
            other_cm=vtr.height_cm if axis == "x" else vtr.width_cm,
            axis=axis, is_dim=is_dim,
            str_metric_fn=self._str_metric_fn,
            grob_metric_fn=None,
            scale=self._get_scale(),
        )

    def _resolve_to_inches_idx(
        self,
        unit_obj: Any,
        index: int,
        axis: str,
        is_dim: bool,
        gp: Optional[Any] = None,
    ) -> float:
        """Resolve element *index* of a Unit to inches."""
        from ._units import Unit
        if not isinstance(unit_obj, Unit):
            return float(unit_obj)

        vtr = self._vp_transform_stack[-1]
        fontsize, cex, lineheight = self._gpar_font_params(gp)

        return _transform_to_inches(
            unit_obj, index, vtr.vpc,
            fontsize, cex, lineheight,
            this_cm=vtr.width_cm if axis == "x" else vtr.height_cm,
            other_cm=vtr.height_cm if axis == "x" else vtr.width_cm,
            axis=axis, is_dim=is_dim,
            str_metric_fn=self._str_metric_fn,
            grob_metric_fn=None,
            scale=self._get_scale(),
        )

    # ===================================================================== #
    # Inches → device coordinate conversion                                 #
    # ===================================================================== #

    def inches_to_dev_x(self, x_inches: float) -> float:
        """Convert absolute x in inches to device x coordinate."""
        return x_inches * self._dev_units_per_inch

    def inches_to_dev_y(self, y_inches: float) -> float:
        """Convert absolute y in inches to device y coordinate.

        Applies Y-flip: in grid, y=0 is bottom; in device, y=0 is top.
        """
        return self._device_height - y_inches * self._dev_units_per_inch

    def inches_to_dev_w(self, w_inches: float) -> float:
        """Convert width in inches to device width."""
        return w_inches * self._dev_units_per_inch

    def inches_to_dev_h(self, h_inches: float) -> float:
        """Convert height in inches to device height."""
        return h_inches * self._dev_units_per_inch

    def transform_loc_to_device(
        self, x_inches: float, y_inches: float,
    ) -> Tuple[float, float]:
        """Transform a location from viewport inches to device coordinates.

        Port of R's transformLocn() + toDeviceX/Y():
        1. Apply the current viewport's 3×3 transform to get absolute inches
        2. Convert absolute inches to device coordinates
        """
        vtr = self._vp_transform_stack[-1]
        loc = location(x_inches, y_inches)
        abs_loc = trans(loc, vtr.transform)
        dev_x = self.inches_to_dev_x(abs_loc[0])
        dev_y = self.inches_to_dev_y(abs_loc[1])
        return dev_x, dev_y

    def transform_dim_to_device(
        self, w_inches: float, h_inches: float,
    ) -> Tuple[float, float]:
        """Transform dimensions from viewport inches to device units.

        For dimensions (widths/heights), we apply only the scaling/rotation
        part of the transform (no translation).  For now, without rotation
        we simply convert inches to device units.  When rotation is present,
        the dimension scaling depends on the rotation angle.
        """
        vtr = self._vp_transform_stack[-1]
        angle = vtr.rotation_angle
        if abs(angle % 360) < 1e-10:
            # No rotation: simple scaling
            return (w_inches * self._dev_units_per_inch,
                    h_inches * self._dev_units_per_inch)
        else:
            # With rotation, the effective device dimensions change.
            # For a rotated viewport, widths and heights in viewport-local
            # inches map to device units through the rotation.
            rad = math.radians(angle)
            cos_a = abs(math.cos(rad))
            sin_a = abs(math.sin(rad))
            dev_w = (w_inches * cos_a + h_inches * sin_a) * self._dev_units_per_inch
            dev_h = (h_inches * cos_a + w_inches * sin_a) * self._dev_units_per_inch
            return dev_w, dev_h

    # ===================================================================== #
    # Public convenience: resolve + transform (Unit → device coords)        #
    # ===================================================================== #

    def resolve_x(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to a device x-coordinate."""
        inches = self._resolve_to_inches(val, axis="x", is_dim=False, gp=gp)
        dev_x, _ = self.transform_loc_to_device(inches, 0.0)
        return dev_x

    def resolve_y(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to a device y-coordinate."""
        inches = self._resolve_to_inches(val, axis="y", is_dim=False, gp=gp)
        _, dev_y = self.transform_loc_to_device(0.0, inches)
        return dev_y

    def resolve_w(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to a device width."""
        inches = self._resolve_to_inches(val, axis="x", is_dim=True, gp=gp)
        return self.inches_to_dev_w(inches)

    def resolve_h(self, val: Any, gp: Optional[Any] = None) -> float:
        """Resolve *val* to a device height."""
        inches = self._resolve_to_inches(val, axis="y", is_dim=True, gp=gp)
        return self.inches_to_dev_h(inches)

    def resolve_x_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of device x-coordinates."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                inches = self._resolve_to_inches_idx(val, i, "x", False, gp)
                dev_x, _ = self.transform_loc_to_device(inches, 0.0)
                out[i] = dev_x
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_x(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def resolve_y_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of device y-coordinates."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                inches = self._resolve_to_inches_idx(val, i, "y", False, gp)
                _, dev_y = self.transform_loc_to_device(0.0, inches)
                out[i] = dev_y
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_y(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def resolve_w_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of device widths."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                inches = self._resolve_to_inches_idx(val, i, "x", True, gp)
                out[i] = self.inches_to_dev_w(inches)
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_w(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    def resolve_h_array(self, val: Any, gp: Optional[Any] = None) -> "np.ndarray":
        """Resolve *val* to an array of device heights."""
        from ._units import Unit
        if isinstance(val, Unit):
            out = np.empty(len(val), dtype=float)
            for i in range(len(val)):
                inches = self._resolve_to_inches_idx(val, i, "y", True, gp)
                out[i] = self.inches_to_dev_h(inches)
            return out
        if isinstance(val, (list, tuple)):
            return np.asarray([self.resolve_h(v, gp) for v in val], dtype=float)
        return np.atleast_1d(np.asarray(val, dtype=float))

    # ===================================================================== #
    # Backward-compatible NPC resolution (for code not yet migrated)        #
    # ===================================================================== #

    def _resolve_to_npc(
        self, unit_obj: Any, axis: str, is_dim: bool, gp: Optional[Any] = None,
    ) -> float:
        """Backward-compatible NPC resolution.

        Converts to inches first (new pipeline), then normalises to NPC
        by dividing by the viewport size in inches.
        """
        inches = self._resolve_to_inches(unit_obj, axis, is_dim, gp)
        vtr = self._vp_transform_stack[-1]
        vp_inches = (vtr.width_cm if axis == "x" else vtr.height_cm) / 2.54
        if vp_inches == 0:
            return 0.0
        return inches / vp_inches

    def resolve_to_npc(
        self, unit_obj: Any, axis: str = "x",
        is_dim: bool = False, gp: Optional[Any] = None,
    ) -> float:
        """Public backward-compatible NPC resolution."""
        return self._resolve_to_npc(unit_obj, axis=axis, is_dim=is_dim, gp=gp)

    # ===================================================================== #
    # Coordinate helpers: NPC → device (backward compatibility)             #
    # ===================================================================== #
    # These are still used by CairoRenderer draw_* methods that receive
    # NPC values.  After full migration they can be removed.

    def _x(self, npc: float) -> float:
        """Convert NPC x -> device x (within current viewport).

        DEPRECATED: use resolve_x() or transform_loc_to_device() instead.
        """
        vtr = self._vp_transform_stack[-1]
        x_inches = npc * vtr.width_cm / 2.54
        dev_x, _ = self.transform_loc_to_device(x_inches, 0.0)
        return dev_x

    def _y(self, npc: float) -> float:
        """Convert NPC y -> device y (Y-flip).

        DEPRECATED: use resolve_y() or transform_loc_to_device() instead.
        """
        vtr = self._vp_transform_stack[-1]
        y_inches = npc * vtr.height_cm / 2.54
        _, dev_y = self.transform_loc_to_device(0.0, y_inches)
        return dev_y

    def _sx(self, npc: float) -> float:
        """Scale a width from NPC to device units.

        DEPRECATED: use resolve_w() instead.
        """
        vtr = self._vp_transform_stack[-1]
        w_inches = npc * vtr.width_cm / 2.54
        return self.inches_to_dev_w(w_inches)

    def _sy(self, npc: float) -> float:
        """Scale a height from NPC to device units.

        DEPRECATED: use resolve_h() instead.
        """
        vtr = self._vp_transform_stack[-1]
        h_inches = npc * vtr.height_cm / 2.54
        return self.inches_to_dev_h(h_inches)

    # ===================================================================== #
    # Abstract methods: backend-specific clipping                           #
    # ===================================================================== #

    @abstractmethod
    def _apply_clip_rect(self, x0: float, y0: float, w: float, h: float) -> None:
        ...

    @abstractmethod
    def _restore_clip(self) -> None:
        ...

    # ===================================================================== #
    # Abstract methods: graphics state save/restore                         #
    # ===================================================================== #

    @abstractmethod
    def save_state(self) -> None: ...

    @abstractmethod
    def restore_state(self) -> None: ...

    # ===================================================================== #
    # Abstract methods: path collection (fill/stroke grobs)                 #
    # ===================================================================== #

    @abstractmethod
    def begin_path_collect(self, rule: str = "winding") -> None: ...

    @abstractmethod
    def end_path_stroke(self, gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def end_path_fill(self, gp: Optional[Any] = None) -> None: ...

    @abstractmethod
    def end_path_fill_stroke(self, gp: Optional[Any] = None) -> None: ...

    # ===================================================================== #
    # Abstract methods: drawing primitives                                  #
    # ===================================================================== #
    # All coordinates are now in DEVICE units (pixels for raster, points
    # for vector).  The resolve_* methods handle the full pipeline:
    # Unit → inches → transform → device.

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
