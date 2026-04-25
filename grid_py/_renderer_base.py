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

        # --- Case 2 (check first): Layout-positioned child ---
        # Must be checked BEFORE Case 1: a viewport can have BOTH
        # layout_pos (its position in the parent's layout) AND its own
        # layout (for its children).  In R, layout_pos determines the
        # viewport's own size/position first, then the layout applies
        # within that region.
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

                # Apply layout-level hjust / vjust (R: layout.c::subRegion
                # lines 447-450). When the cells' total dimensions don't
                # fill the parent vp (common for chrome-merge slices in
                # patchwork), the leftover space is distributed per the
                # layout's justification (default 0.5 → centred). The
                # prior implementation always pinned cells to top-left
                # (vjust=1, hjust=0), which placed merged chrome at the
                # top of the parent panel cell rather than just outside
                # it, breaking simplify_fixed renders.
                hjust = float(grid.get("hjust", 0.0))
                vjust = float(grid.get("vjust", 1.0))
                total_w_dev = sum(col_widths)
                total_h_dev = sum(row_heights)
                hjust_offset_dev = hjust * (parent_w_dev - total_w_dev)
                vjust_offset_dev = (1.0 - vjust) * (parent_h_dev - total_h_dev)

                cell_x_in = (cell_x0_dev + hjust_offset_dev) / self._dev_units_per_inch
                # Device y is top-down; convert to bottom-up inches.
                # vjust_offset_dev shifts the block downward in device-y
                # (i.e. upward in bottom-up coords) by the leftover * (1-vjust).
                cell_y_in = parent_h_in - (
                    cell_y0_dev + cell_h_dev + vjust_offset_dev
                ) / self._dev_units_per_inch

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

                # If this layout-positioned viewport ALSO has its own
                # layout, compute the grid within the cell's bounds so
                # that its children can use layout_pos_row/col.
                if layout is not None:
                    w_dev = cell_w_dev
                    h_dev = cell_h_dev
                    # R: ``calcViewportLayout`` (layout.c:492-590) reads
                    # ``layoutRespect(layout)`` / ``layoutRespectMat(layout)``
                    # straight off the layout object — there is no caller-side
                    # respect argument. Mirror that: ``_calc_layout_sizes``
                    # consumes ``layout._valid_respect`` directly.
                    grid_info = self._compute_grid(layout, w_dev, h_dev)
                    self._layout_stack.append(grid_info)
                    self._layout_depth_stack.append(
                        len(self._vp_transform_stack))
                return

        # --- Case 1: Layout viewport (no layout_pos) ---
        if layout is not None:
            # Layout viewport uses same bounds as parent but stores grid info.
            # Compute grid in device units for layout children.
            w_dev = parent_vtr.width_cm / 2.54 * self._dev_units_per_inch
            h_dev = parent_vtr.height_cm / 2.54 * self._dev_units_per_inch
            # See R layout.c:492-590 — respect lives on the layout object;
            # callers don't pass it down.
            grid_info = self._compute_grid(layout, w_dev, h_dev)

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
            grob_metric_fn=self._grob_metric_fn,
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
    ) -> dict:
        """Compute row/column positions for a GridLayout within the parent.

        R reference: ``layout.c:calcViewportLayout`` (lines 492-590).
        Respect (full or matrix-form) lives on the layout object itself —
        ``_calc_layout_sizes`` reads ``layout._valid_respect`` directly,
        so this signature has no ``respect`` argument (matches R, where
        there is no caller-side respect parameter either).
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
                getattr(layout, "widths", None), ncol, parent_w, axis="x",
            )
            row_heights = self._resolve_sizes(
                getattr(layout, "heights", None), nrow, parent_h, axis="y",
            )

        ncol = len(col_widths)
        nrow = len(row_heights)
        col_starts = [sum(col_widths[:i]) for i in range(ncol)]
        row_starts = [sum(row_heights[:i]) for i in range(nrow)]

        # Layout justification (R: layout.c::subRegion lines 433-450).
        # When the cells' total size is less than the parent vp, hjust /
        # vjust shift the layout block within the parent (default 0.5 →
        # centred). ``GridLayout`` carries this on ``_valid_just`` as
        # (hjust, vjust); fall back to (0, 1) for non-GridLayout
        # containers (matching the prior top-left alignment).
        valid_just = getattr(layout, "_valid_just", None)
        if valid_just is not None:
            hjust = float(valid_just[0])
            vjust = float(valid_just[1])
        else:
            hjust, vjust = 0.0, 1.0

        return {
            "col_starts": col_starts, "col_widths": col_widths,
            "row_starts": row_starts, "row_heights": row_heights,
            "parent_w": parent_w, "parent_h": parent_h,
            "hjust": hjust, "vjust": vjust,
        }

    def _resolve_sizes(self, unit_obj: Any, n: int, total: float,
                        axis: str = "x") -> list:
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
            elif t in ("sum", "min", "max", "lines", "char", "snpc",
                        "strwidth", "strheight", "strascent", "strdescent",
                        "grobwidth", "grobheight"):
                # Context-dependent or compound units: resolve to inches
                # via the full pipeline, then convert to device pixels.
                elem = Unit(float(v), t,
                            data=unit_obj._data[i] if unit_obj._data else None)
                inches = self._resolve_to_inches(elem, axis, True)
                px = inches * self._dev_units_per_inch
                abs_sizes[i] = px
                abs_total += px
            else:
                # Unknown type — treat as null
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

    # ===================================================================== #
    # evaluateGrobUnit -- port of R unit.c:325-590                          #
    # ===================================================================== #

    # Per-renderer cycle guard — populated with id(grob) for every grob
    # currently being measured. ``_evaluate_grob_unit`` consults this
    # before pushing the grob's vp; re-entry for the same grob means
    # the vp itself references its own grob*Details (e.g.
    # ``viewport(width = grobWidth(self))``), which would otherwise
    # infinite-recurse through calcViewportTransform → grob_metric_fn
    # → _evaluate_grob_unit → push vp → ...
    #
    # R bounds the same recursion implicitly because its
    # widthDetails.gtable returns an absolute mm sum, and R's
    # setviewport/calcViewportTransform doesn't re-trigger the metric
    # path during the recursive pushViewport (verified empirically:
    # preDraw.gtable runs exactly 3 times for one toplevel grid.draw —
    # not unbounded). We mirror the bound explicitly.
    @property
    def _measuring_grobs(self) -> set:
        s = getattr(self, "_measuring_grobs_set", None)
        if s is None:
            s = set()
            self._measuring_grobs_set = s
        return s

    def _evaluate_grob_unit(
        self,
        grob: Any,
        unit_type: str,
        value: float = 1.0,
    ) -> Optional[float]:
        """Evaluate a grobwidth/grobheight/etc. unit, returning inches.

        Port of R's ``evaluateGrobUnit()`` (unit.c:325-590).
        Performs the full cycle:
          1. Save state (gpar, current grob, DL recording)
          2. If *grob* is a gPath (string), resolve to actual grob
          3. ``preDraw(grob)`` — pushes grob's vp/gp
          4. ``widthDetails(grob)``/``heightDetails(grob)`` — get result Unit
          5. Convert result Unit to inches *within grob's viewport context*
          6. ``postDraw(grob)`` — pops grob's vp
          7. Restore state

        Parameters
        ----------
        grob : Grob or str
            The grob (or gPath name) to measure.
        unit_type : str
            One of ``"grobwidth"``, ``"grobheight"``, ``"grobascent"``,
            ``"grobdescent"``, ``"grobx"``, ``"groby"``.
        value : float
            The numeric value of the unit (angle for grobx/groby).

        Returns
        -------
        float or None
            Size in inches, or None on failure.
        """
        import copy
        from ._state import get_state
        from ._grob import Grob, GTree
        from ._path import GPath
        from ._size import (
            width_details, height_details,
            ascent_details, descent_details,
        )

        state = get_state()

        # --- Resolve gPath to actual grob (R unit.c:405-431) ---
        if isinstance(grob, (str, GPath)):
            grob = self._find_grob_for_metric(grob, state)
            if grob is None:
                return 0.0

        if not isinstance(grob, Grob):
            return 0.0

        # --- Cycle break for self-referential grobwidth/grobheight ---
        # When we're already measuring this grob and the grob's vp
        # references its own width/height (e.g. patchwork's
        # ``as_patch.GT`` builds ``viewport(width=grobWidth(grob))``),
        # the recursive vp push would trigger _evaluate_grob_unit for
        # the same grob again, etc. Resolve widthDetails/heightDetails
        # directly without the recursive push — for non-context-
        # dependent grobs (gtable widthDetails returns absolute_size of
        # the widths sum) this gives the same answer as the full path,
        # matching R's behaviour where preDraw.gtable runs a bounded
        # number of times rather than unboundedly.
        if id(grob) in self._measuring_grobs:
            if unit_type == "grobwidth":
                ru = width_details(grob)
                axis = "x"
            elif unit_type == "grobheight":
                ru = height_details(grob)
                axis = "y"
            elif unit_type == "grobascent":
                ru = ascent_details(grob)
                axis = "y"
            elif unit_type == "grobdescent":
                ru = descent_details(grob)
                axis = "y"
            else:
                return 0.0
            if ru is None:
                return 0.0
            from ._units import Unit as _U
            if not isinstance(ru, _U):
                return 0.0
            if len(ru) == 1 and ru._units[0] == "null":
                return 0.0
            return float(self._resolve_to_inches(ru, axis, True))

        # Capture id BEFORE make_context() rebinds ``grob`` to the
        # context-wrapped copy — we need to release the same id we added.
        _measuring_id = id(grob)
        self._measuring_grobs.add(_measuring_id)

        # --- Save state (R unit.c:355-377) ---
        saved_dl_on = state._dl_on
        state.set_display_list_on(False)
        saved_gpar = copy.copy(state.get_gpar())
        saved_current_grob = getattr(state, "_current_grob", None)

        try:
            # --- preDraw(grob) (R unit.c:434-435) ---
            # This may push viewports and set gpar
            from ._draw import _push_vp_gp, _pop_grob_vp
            grob = grob.make_context()
            if isinstance(grob, GTree):
                state._current_grob = grob
            _push_vp_gp(grob)
            grob.pre_draw_details()

            # --- After preDraw, re-establish viewport context ---
            # (R unit.c:451-456)
            vtr = self._vp_transform_stack[-1]
            gp = state.get_gpar()
            fontsize, cex, lineheight = self._gpar_font_params(gp)

            if unit_type in ("grobx", "groby"):
                # Compute the x/y coordinate on the grob's bounding box at
                # the requested angle (encoded in ``value``; 0=east, 90=north,
                # 180=west, 270=south).  Mirrors ``xDetails.text`` /
                # ``yDetails.text`` in R grid (primitives.R:1406-1428).
                result = self._grob_xy_inches_at_theta(
                    grob, unit_type, float(value), gp,
                )
            else:
                if unit_type == "grobwidth":
                    result_unit = width_details(grob)
                elif unit_type == "grobheight":
                    result_unit = height_details(grob)
                elif unit_type == "grobascent":
                    result_unit = ascent_details(grob)
                elif unit_type == "grobdescent":
                    result_unit = descent_details(grob)
                else:
                    result_unit = None

                if result_unit is None:
                    result = 0.0
                else:
                    from ._units import Unit
                    if not isinstance(result_unit, Unit):
                        result = 0.0
                    elif (len(result_unit) == 1
                          and result_unit._units[0] == "null"):
                        # "null" units evaluate to 0 (R unit.c:530-531)
                        result = 0.0
                    else:
                        if unit_type in ("grobwidth",):
                            result = self._resolve_to_inches(
                                result_unit, "x", True, gp)
                        elif unit_type in ("grobheight", "grobascent",
                                           "grobdescent"):
                            result = self._resolve_to_inches(
                                result_unit, "y", True, gp)
                        else:
                            result = 0.0

            # --- postDraw(grob) (R unit.c:556-557) ---
            grob.post_draw_details()
            if grob.vp is not None:
                _pop_grob_vp(grob.vp)

        except Exception:
            result = 0.0
        finally:
            # --- Restore state (R unit.c:561-562) ---
            state.replace_gpar(saved_gpar)
            state._current_grob = saved_current_grob
            state.set_display_list_on(saved_dl_on)
            self._measuring_grobs.discard(_measuring_id)

        return result

    def _grob_xy_inches_at_theta(
        self,
        grob: Any,
        unit_type: str,
        theta_deg: float,
        gp: Optional[Any] = None,
    ) -> float:
        """Return the inches x- or y-coordinate at angle ``theta_deg`` on a
        grob's bounding box.

        Used to resolve ``grobx`` / ``groby`` units (e.g. those produced by
        ``grob_x(text_grob, "west")``).  The angle convention: 0 = east,
        90 = north, 180 = west, 270 = south.

        Only the axis-aligned rectangle defined by width/height + hjust/vjust
        is considered; rotated text is approximated by its upright box (good
        enough for the common ``rot=0`` path that dominates ggrepel output).
        """
        import math
        from ._units import Unit
        from ._size import width_details, height_details

        # Grob anchor (grob.x, grob.y) — default to center of viewport if absent.
        x_unit = getattr(grob, "x", None)
        y_unit = getattr(grob, "y", None)
        if x_unit is None:
            x_unit = Unit(0.5, "npc")
        if y_unit is None:
            y_unit = Unit(0.5, "npc")
        try:
            x_inches = self._resolve_to_inches(x_unit, "x", False, gp)
        except Exception:
            x_inches = 0.0
        try:
            y_inches = self._resolve_to_inches(y_unit, "y", False, gp)
        except Exception:
            y_inches = 0.0

        # Width / height of the grob's bounding box, in inches.
        def _details_inches(fn, axis: str) -> float:
            try:
                u = fn(grob)
            except Exception:
                return 0.0
            if u is None:
                return 0.0
            if not isinstance(u, Unit):
                return 0.0
            if len(u) == 1 and u._units[0] == "null":
                return 0.0
            try:
                return float(self._resolve_to_inches(u, axis, True, gp))
            except Exception:
                return 0.0

        w_in = _details_inches(width_details, "x")
        h_in = _details_inches(height_details, "y")

        # hjust / vjust control which corner of the box is anchored at (x, y).
        def _just_to_float(v: Any, default: float) -> float:
            if v is None:
                return default
            if isinstance(v, (int, float)):
                return float(v)
            _H = {"left": 0.0, "right": 1.0, "centre": 0.5, "center": 0.5}
            _V = {"bottom": 0.0, "top": 1.0, "centre": 0.5, "center": 0.5}
            s = str(v).lower()
            return _H.get(s, _V.get(s, default))

        hjust = _just_to_float(getattr(grob, "hjust", 0.5), 0.5)
        vjust = _just_to_float(getattr(grob, "vjust", 0.5), 0.5)

        # Centre of the bounding box in inches.
        cx = x_inches + (0.5 - hjust) * w_in
        cy = y_inches + (0.5 - vjust) * h_in

        # Point on the box at direction theta (from centre).  Ray hits the
        # nearest axis-aligned edge.
        rad = math.radians(theta_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        dx = w_in / 2.0
        dy = h_in / 2.0
        eps = 1e-12
        if abs(cos_t) < eps:
            t = dy / max(abs(sin_t), eps)
        elif abs(sin_t) < eps:
            t = dx / max(abs(cos_t), eps)
        else:
            t = min(dx / abs(cos_t), dy / abs(sin_t))

        px = cx + t * cos_t
        py = cy + t * sin_t
        return float(px if unit_type == "grobx" else py)

    def _find_grob_for_metric(self, grob_ref: Any, state: Any) -> Any:
        """Resolve a gPath/string to an actual grob for metric evaluation.

        Port of R unit.c:405-431: if current grob is NULL, search the
        display list; otherwise search the current grob's children.
        """
        from ._grob import Grob, GTree
        from ._path import GPath
        from ._display_list import DLDrawGrob

        name = str(grob_ref)

        # Check current grob's children first (R unit.c:420-425)
        current_grob = getattr(state, "_current_grob", None)
        if current_grob is not None and isinstance(current_grob, GTree):
            child = current_grob._children.get(name)
            if child is not None:
                return child

        # Search display list (R unit.c:413-418)
        dl = state.get_display_list()
        for item in dl:
            if isinstance(item, DLDrawGrob) and item.grob is not None:
                if getattr(item.grob, "name", None) == name:
                    return item.grob
                # Search inside GTrees
                if isinstance(item.grob, GTree):
                    child = item.grob._children.get(name)
                    if child is not None:
                        return child

        return None

    def _grob_metric_fn(self, grob: Any, unit_type: str, value: float) -> Optional[float]:
        """Callback for _transform_to_inches grob_metric_fn parameter.

        Delegates to _evaluate_grob_unit which does the full
        preDraw/widthDetails/postDraw cycle.
        """
        return self._evaluate_grob_unit(grob, unit_type, value)

    # ===================================================================== #
    # Unit → inches resolution (core pipeline)                              #
    # ===================================================================== #

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
            grob_metric_fn=self._grob_metric_fn,
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
            grob_metric_fn=self._grob_metric_fn,
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
