"""Cairo-based renderer for grid_py.

Replaces the former matplotlib rendering backend.  All grob primitives
are drawn via pycairo, and output can be written as PNG, PDF, SVG, or
PostScript without any matplotlib dependency.

The coordinate convention matches R's grid: the unit square [0, 1] x [0, 1]
with the origin at the **bottom-left**.  Cairo's native origin is top-left,
so a Y-flip is applied internally.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cairo
except ImportError:
    raise ImportError(
        "pycairo is required for grid_py rendering.  "
        "Install it with:  conda install -c conda-forge pycairo"
    )

from ._gpar import Gpar

__all__ = ["CairoRenderer"]

# ---------------------------------------------------------------------------
# R colour helpers
# ---------------------------------------------------------------------------

import re as _re

_GREY_RE = _re.compile(r"^gr[ae]y(\d{1,3})$")

# Minimal named-colour table (R's grDevices::colors subset most commonly
# encountered in ggplot2 / grid themes).
_NAMED_COLOURS: Dict[str, Tuple[float, float, float]] = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.6471, 0.0),
    "purple": (0.6275, 0.1255, 0.9412),
    "pink": (1.0, 0.7529, 0.7961),
    "brown": (0.6471, 0.1647, 0.1647),
    "grey": (0.7451, 0.7451, 0.7451),
    "gray": (0.7451, 0.7451, 0.7451),
    "transparent": (0.0, 0.0, 0.0),  # handled specially via alpha
    "NA": (0.0, 0.0, 0.0),
}


def _parse_colour(c: Any) -> Tuple[float, float, float, float]:
    """Convert an R-style colour specification to (r, g, b, a) in [0, 1]."""
    if c is None:
        return (0.0, 0.0, 0.0, 1.0)

    if isinstance(c, (list, tuple)):
        if len(c) >= 4:
            return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
        if len(c) == 3:
            return (float(c[0]), float(c[1]), float(c[2]), 1.0)
        # Single-element list
        c = c[0]

    if isinstance(c, str):
        s = c.strip()
        if s.lower() in ("transparent", "na", "none", ""):
            return (0.0, 0.0, 0.0, 0.0)
        # grey<N> / gray<N>
        m = _GREY_RE.match(s)
        if m:
            v = int(m.group(1)) / 100.0
            return (v, v, v, 1.0)
        # Hex colour  #RRGGBB or #RRGGBBAA
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 6:
                r = int(h[0:2], 16) / 255.0
                g = int(h[2:4], 16) / 255.0
                b = int(h[4:6], 16) / 255.0
                return (r, g, b, 1.0)
            if len(h) == 8:
                r = int(h[0:2], 16) / 255.0
                g = int(h[2:4], 16) / 255.0
                b = int(h[4:6], 16) / 255.0
                a = int(h[6:8], 16) / 255.0
                return (r, g, b, a)
        # Named colour
        low = s.lower()
        if low in _NAMED_COLOURS:
            rgb = _NAMED_COLOURS[low]
            return (rgb[0], rgb[1], rgb[2], 1.0)

    # Already numeric (float grey level, etc.)
    if isinstance(c, (int, float)):
        v = float(c)
        return (v, v, v, 1.0)

    # Fallback: black
    return (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Line-type mapping
# ---------------------------------------------------------------------------

_LTY_DASHES: Dict[str, Optional[Sequence[float]]] = {
    "solid": None,
    "dashed": [6.0, 4.0],
    "dotted": [2.0, 2.0],
    "dotdash": [2.0, 2.0, 6.0, 2.0],
    "longdash": [10.0, 3.0],
    "twodash": [5.0, 2.0, 10.0, 2.0],
    "blank": [0.0, 100.0],
}

_LINEEND_MAP = {
    "round": cairo.LINE_CAP_ROUND,
    "butt": cairo.LINE_CAP_BUTT,
    "square": cairo.LINE_CAP_SQUARE,
}

_LINEJOIN_MAP = {
    "round": cairo.LINE_JOIN_ROUND,
    "mitre": cairo.LINE_JOIN_MITER,
    "miter": cairo.LINE_JOIN_MITER,
    "bevel": cairo.LINE_JOIN_BEVEL,
}


# ---------------------------------------------------------------------------
# CairoRenderer
# ---------------------------------------------------------------------------

class CairoRenderer:
    """Render grid grobs to a Cairo surface.

    Parameters
    ----------
    width : float
        Device width in inches.
    height : float
        Device height in inches.
    dpi : float
        Dots per inch (default 150).
    surface_type : str
        ``"image"`` (default, raster PNG), ``"pdf"``, ``"svg"``, ``"ps"``.
    filename : str or None
        Output file path.  Required for ``"pdf"``, ``"svg"``, ``"ps"``;
        ignored for ``"image"`` (use :meth:`write_to_png` or
        :meth:`to_png_bytes` instead).
    bg : str or tuple or None
        Background colour (default ``"white"``).
    """

    def __init__(
        self,
        width: float = 7.0,
        height: float = 5.0,
        dpi: float = 150.0,
        surface_type: str = "image",
        filename: Optional[str] = None,
        bg: Any = "white",
    ) -> None:
        self.width_in = width
        self.height_in = height
        self.dpi = dpi
        self._surface_type = surface_type
        self._width_px = int(width * dpi)
        self._height_px = int(height * dpi)

        # Points for vector surfaces (1 pt = 1/72 inch)
        width_pt = width * 72.0
        height_pt = height * 72.0

        if surface_type == "image":
            self._surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, self._width_px, self._height_px
            )
        elif surface_type == "pdf":
            if filename is None:
                raise ValueError("filename is required for PDF surface")
            self._surface = cairo.PDFSurface(filename, width_pt, height_pt)
        elif surface_type == "svg":
            if filename is None:
                raise ValueError("filename is required for SVG surface")
            self._surface = cairo.SVGSurface(filename, width_pt, height_pt)
        elif surface_type == "ps":
            if filename is None:
                raise ValueError("filename is required for PS surface")
            self._surface = cairo.PSSurface(filename, width_pt, height_pt)
        else:
            raise ValueError(f"Unknown surface_type: {surface_type!r}")

        self._ctx = cairo.Context(self._surface)

        # Fill background
        bg_rgba = _parse_colour(bg)
        self._ctx.set_source_rgba(*bg_rgba)
        self._ctx.paint()

        # Viewport transform stack.  Each entry is (x0, y0, w, h) in
        # device-unit space.  NPC [0,1] maps to [x0, x0+w] x [y0, y0+h].
        if self._surface_type == "image":
            self._vp_stack: list = [(0.0, 0.0, float(self._width_px), float(self._height_px))]
        else:
            self._vp_stack = [(0.0, 0.0, self.width_in * 72.0, self.height_in * 72.0)]
        self._layout_stack: list = []  # GridLayout info stack
        self._clip_stack: list = []   # Track which viewport pushes had clipping

    # ---- viewport management -----------------------------------------------

    def push_viewport(self, vp: Any) -> None:
        """Push a viewport, updating the coordinate transform.

        Handles three viewport types:
        1. Layout viewport (has ``_layout`` with grid info) — stores grid
        2. Child viewport with ``layout_pos_row/col`` — uses parent grid
        3. Simple viewport with x/y/width/height — direct position
        """
        from ._units import Unit

        x0, y0, pw, ph = self._vp_stack[-1]

        layout = getattr(vp, "_layout", None)
        layout_pos_row = getattr(vp, "_layout_pos_row", None)
        layout_pos_col = getattr(vp, "_layout_pos_col", None)

        if layout is not None:
            # Layout viewport: same size as parent, but store grid info
            grid_info = self._compute_grid(layout, pw, ph)
            self._vp_stack.append((x0, y0, pw, ph))
            self._layout_stack.append(grid_info)
            self._clip_stack.append(False)
            return

        if layout_pos_row is not None and layout_pos_col is not None:
            # Child of a layout viewport: compute cell position
            if self._layout_stack:
                grid = self._layout_stack[-1]
                col_starts, col_widths = grid["col_starts"], grid["col_widths"]
                row_starts, row_heights = grid["row_starts"], grid["row_heights"]

                # layout_pos is 1-based (t, b) or single int
                if isinstance(layout_pos_row, (list, tuple)):
                    t, b = int(layout_pos_row[0]) - 1, int(layout_pos_row[1]) - 1
                else:
                    t = b = int(layout_pos_row) - 1
                if isinstance(layout_pos_col, (list, tuple)):
                    l, r = int(layout_pos_col[0]) - 1, int(layout_pos_col[1]) - 1
                else:
                    l = r = int(layout_pos_col) - 1

                # Compute cell region
                cell_x0 = x0 + (col_starts[l] if l < len(col_starts) else 0)
                cell_y0 = y0 + (row_starts[t] if t < len(row_starts) else 0)
                cell_w = sum(col_widths[l:r + 1]) if r < len(col_widths) else pw
                cell_h = sum(row_heights[t:b + 1]) if b < len(row_heights) else ph

                self._vp_stack.append((cell_x0, cell_y0, cell_w, cell_h))
                self._apply_clip(vp, cell_x0, cell_y0, cell_w, cell_h)
                return

        # Simple viewport with explicit x/y/width/height
        vp_x_raw = getattr(vp, "_x", None)
        vp_y_raw = getattr(vp, "_y", None)
        vp_w_raw = getattr(vp, "_width", None)
        vp_h_raw = getattr(vp, "_height", None)

        vp_x = float(vp_x_raw._values[0]) if vp_x_raw is not None and hasattr(vp_x_raw, "_values") else 0.5
        vp_y = float(vp_y_raw._values[0]) if vp_y_raw is not None and hasattr(vp_y_raw, "_values") else 0.5
        vp_w = float(vp_w_raw._values[0]) if vp_w_raw is not None and hasattr(vp_w_raw, "_values") else 1.0
        vp_h = float(vp_h_raw._values[0]) if vp_h_raw is not None and hasattr(vp_h_raw, "_values") else 1.0

        just = getattr(vp, "_just", (0.5, 0.5))
        if isinstance(just, (list, tuple)) and len(just) >= 2:
            hjust, vjust = float(just[0]), float(just[1])
        else:
            hjust, vjust = 0.5, 0.5

        new_w = vp_w * pw
        new_h = vp_h * ph
        new_x0 = x0 + vp_x * pw - hjust * new_w
        new_y0 = y0 + vp_y * ph - vjust * new_h

        self._vp_stack.append((new_x0, new_y0, new_w, new_h))
        self._apply_clip(vp, new_x0, new_y0, new_w, new_h)

    def _apply_clip(self, vp: Any, x0: float, y0: float, w: float, h: float) -> None:
        """Apply Cairo clipping rectangle if the viewport has clip=True/'on'."""
        clip = getattr(vp, "_clip", None)
        if clip is True or clip == "on":
            self._ctx.save()
            self._ctx.rectangle(x0, y0, w, h)
            self._ctx.clip()
            self._clip_stack.append(True)
        else:
            self._clip_stack.append(False)

    def pop_viewport(self) -> None:
        """Pop the current viewport and restore clipping state."""
        if len(self._vp_stack) > 1:
            self._vp_stack.pop()
            if self._clip_stack:
                had_clip = self._clip_stack.pop()
                if had_clip:
                    self._ctx.restore()

    def _compute_grid(self, layout: Any, parent_w: float, parent_h: float) -> dict:
        """Compute row/column positions for a GridLayout within the parent."""
        nrow = getattr(layout, "nrow", 1)
        ncol = getattr(layout, "ncol", 1)
        widths = getattr(layout, "widths", None)
        heights = getattr(layout, "heights", None)

        col_widths = self._resolve_sizes(widths, ncol, parent_w)
        row_heights = self._resolve_sizes(heights, nrow, parent_h)

        col_starts = [sum(col_widths[:i]) for i in range(ncol)]
        row_starts = [sum(row_heights[:i]) for i in range(nrow)]

        return {
            "col_starts": col_starts, "col_widths": col_widths,
            "row_starts": row_starts, "row_heights": row_heights,
        }

    def _resolve_sizes(self, unit_obj: Any, n: int, total: float) -> list:
        """Resolve a Unit vector to device sizes, distributing null units.

        Mirrors R's grid unit resolution: absolute units (cm, inches, pt,
        etc.) are converted to device pixels first; the remaining space is
        then distributed among ``"null"`` units proportionally.
        """
        if unit_obj is None:
            return [total / n] * n

        from ._units import Unit, _INCHES_PER
        if not isinstance(unit_obj, Unit):
            return [total / n] * n

        vals = unit_obj._values
        types = (unit_obj._units
                 if hasattr(unit_obj, "_units")
                 else getattr(unit_obj, "_types", ["null"] * len(vals)))

        # --- Pass 1: resolve absolute units to device pixels ---------------
        dpi = getattr(self, "dpi", 150.0)
        abs_sizes = {}  # index → device pixels
        abs_total = 0.0
        null_total = 0.0

        for i, (v, t) in enumerate(zip(vals, types)):
            if t == "npc":
                px = float(v) * total
                abs_sizes[i] = px
                abs_total += px
            elif t in _INCHES_PER:
                # absolute unit → inches → device pixels
                px = float(v) * _INCHES_PER[t] * dpi
                abs_sizes[i] = px
                abs_total += px
            elif t == "null":
                null_total += float(v)
            else:
                # Unknown type: treat as null
                null_total += float(v)

        # --- Pass 2: distribute remaining space among null units -----------
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

    # ---- coordinate helpers ------------------------------------------------

    def _x(self, npc: float) -> float:
        """Convert NPC x → device x (within current viewport)."""
        x0, y0, pw, ph = self._vp_stack[-1]
        return x0 + npc * pw

    def _y(self, npc: float) -> float:
        """Convert NPC y → device y (Y-flip: 0=bottom, 1=top).

        In Cairo device coords, y=0 is the top and increases downward.
        In NPC, y=0 is the bottom and y=1 is the top.  Within a viewport
        whose device-top is ``y0`` and device-height is ``ph``:
          NPC 0 (bottom) → device ``y0 + ph``
          NPC 1 (top)    → device ``y0``
        """
        x0, y0, pw, ph = self._vp_stack[-1]
        return y0 + (1.0 - npc) * ph

    def _sx(self, npc: float) -> float:
        """Scale a width from NPC to device units."""
        return npc * self._vp_stack[-1][2]

    def _sy(self, npc: float) -> float:
        """Scale a height from NPC to device units."""
        return npc * self._vp_stack[-1][3]

    # ---- gpar application --------------------------------------------------

    def _apply_stroke(self, gp: Optional[Gpar]) -> Tuple[float, float, float, float]:
        """Set stroke colour, line width, dash, caps, joins from Gpar.

        Returns the stroke RGBA so caller can decide whether to actually
        stroke (transparent == skip).
        """
        ctx = self._ctx
        if gp is None:
            ctx.set_source_rgba(0, 0, 0, 1)
            ctx.set_line_width(1.0)
            return (0.0, 0.0, 0.0, 1.0)

        col = gp.get("col", None)
        col_val = col[0] if isinstance(col, (list, tuple)) else col
        # R semantics: col=NA (None) means "no border" → transparent
        if col_val is None:
            return (0.0, 0.0, 0.0, 0.0)
        rgba = _parse_colour(col_val)

        alpha = gp.get("alpha", None)
        if alpha is not None:
            a = float(alpha[0] if isinstance(alpha, (list, tuple)) else alpha)
            rgba = (rgba[0], rgba[1], rgba[2], rgba[3] * a)

        ctx.set_source_rgba(*rgba)

        lwd = gp.get("lwd", None)
        lw = float((lwd[0] if isinstance(lwd, (list, tuple)) else lwd) if lwd is not None else 1.0)
        # R semantics: lwd=0 means invisible line
        if lw <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        ctx.set_line_width(lw)

        lty = gp.get("lty", None)
        if lty is not None:
            lty_val = str(lty[0] if isinstance(lty, (list, tuple)) else lty)
            dashes = _LTY_DASHES.get(lty_val)
            if dashes is not None:
                ctx.set_dash(dashes)
            else:
                ctx.set_dash([])
        else:
            ctx.set_dash([])

        lineend = gp.get("lineend", None)
        if lineend is not None:
            le = str(lineend[0] if isinstance(lineend, (list, tuple)) else lineend)
            ctx.set_line_cap(_LINEEND_MAP.get(le, cairo.LINE_CAP_BUTT))
        else:
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)

        linejoin = gp.get("linejoin", None)
        if linejoin is not None:
            lj = str(linejoin[0] if isinstance(linejoin, (list, tuple)) else linejoin)
            ctx.set_line_join(_LINEJOIN_MAP.get(lj, cairo.LINE_JOIN_ROUND))
        else:
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        return rgba

    def _fill_rgba(self, gp: Optional[Gpar]) -> Tuple[float, float, float, float]:
        """Extract fill colour from Gpar."""
        if gp is None:
            return (1.0, 1.0, 1.0, 1.0)
        fill = gp.get("fill", None)
        fill_val = fill[0] if isinstance(fill, (list, tuple)) else fill
        rgba = _parse_colour(fill_val if fill is not None else "white")

        alpha = gp.get("alpha", None)
        if alpha is not None:
            a = float(alpha[0] if isinstance(alpha, (list, tuple)) else alpha)
            rgba = (rgba[0], rgba[1], rgba[2], rgba[3] * a)
        return rgba

    def _set_font(self, gp: Optional[Gpar]) -> float:
        """Configure font on context from Gpar.  Returns font size in device units."""
        ctx = self._ctx
        family = "sans-serif"
        slant = cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_NORMAL
        fontsize = 12.0  # points

        if gp is not None:
            ff = gp.get("fontfamily", None)
            if ff is not None:
                family = str(ff[0] if isinstance(ff, (list, tuple)) else ff)

            fs = gp.get("fontsize", None)
            if fs is not None:
                fontsize = float(fs[0] if isinstance(fs, (list, tuple)) else fs)

            cex = gp.get("cex", None)
            if cex is not None:
                fontsize *= float(cex[0] if isinstance(cex, (list, tuple)) else cex)

            face = gp.get("fontface", None)
            if face is not None:
                val = face[0] if isinstance(face, (list, tuple)) else face
                if isinstance(val, str):
                    val = val.lower()
                if val in (2, "bold"):
                    weight = cairo.FONT_WEIGHT_BOLD
                elif val in (3, "italic", "oblique"):
                    slant = cairo.FONT_SLANT_ITALIC
                elif val in (4, "bold.italic"):
                    weight = cairo.FONT_WEIGHT_BOLD
                    slant = cairo.FONT_SLANT_ITALIC

        ctx.select_font_face(family, slant, weight)
        # Font size in device units (points for vector, pixels for raster).
        # Cairo interprets set_font_size as "user-space units".
        # For image surfaces we scale points → pixels.
        if self._surface_type == "image":
            device_fs = fontsize * self.dpi / 72.0
        else:
            device_fs = fontsize
        ctx.set_font_size(device_fs)
        return device_fs

    # ---- drawing primitives ------------------------------------------------

    def draw_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        hjust: float = 0.5,
        vjust: float = 0.5,
        gp: Optional[Gpar] = None,
    ) -> None:
        ctx = self._ctx
        ctx.save()

        x0 = x - w * hjust
        y0 = y - h * vjust
        dx = self._x(x0)
        # top-left corner in device coords (Y-flipped):
        dy = self._y(y0 + h)
        dw = self._sx(w)
        dh = self._sy(h)

        ctx.rectangle(dx, dy, dw, dh)

        fill = self._fill_rgba(gp)
        if fill[3] > 0:
            ctx.set_source_rgba(*fill)
            ctx.fill_preserve()

        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()

        ctx.restore()

    def draw_circle(
        self,
        x: float,
        y: float,
        r: float,
        gp: Optional[Gpar] = None,
    ) -> None:
        ctx = self._ctx
        ctx.save()

        cx = self._x(x)
        cy = self._y(y)
        # Radius: average of x/y scale
        dr = (self._sx(r) + self._sy(r)) / 2.0

        ctx.arc(cx, cy, dr, 0, 2 * math.pi)

        fill = self._fill_rgba(gp)
        if fill[3] > 0:
            ctx.set_source_rgba(*fill)
            ctx.fill_preserve()

        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()

        ctx.restore()

    def draw_line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        gp: Optional[Gpar] = None,
    ) -> None:
        n = max(len(x), len(y))
        if n < 2:
            return
        # R-style recycling: repeat shorter array to match the longer one
        if len(x) < n:
            x = np.resize(x, n)
        if len(y) < n:
            y = np.resize(y, n)

        ctx = self._ctx
        ctx.save()
        stroke = self._apply_stroke(gp)

        ctx.move_to(self._x(x[0]), self._y(y[0]))
        for i in range(1, n):
            ctx.line_to(self._x(x[i]), self._y(y[i]))

        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()
        ctx.restore()

    def draw_polyline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        id_: Optional[np.ndarray] = None,
        gp: Optional[Gpar] = None,
    ) -> None:
        ctx = self._ctx
        ctx.save()
        stroke = self._apply_stroke(gp)

        if id_ is None:
            self.draw_line(x, y, gp)
            ctx.restore()
            return

        for uid in np.unique(id_):
            mask = id_ == uid
            px = x[mask]
            py = y[mask]
            if len(px) < 2:
                continue
            ctx.move_to(self._x(px[0]), self._y(py[0]))
            for i in range(1, len(px)):
                ctx.line_to(self._x(px[i]), self._y(py[i]))
            if stroke[3] > 0:
                ctx.stroke()
            else:
                ctx.new_path()

        ctx.restore()

    def draw_segments(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        gp: Optional[Gpar] = None,
    ) -> None:
        ctx = self._ctx
        ctx.save()
        stroke = self._apply_stroke(gp)

        n = min(len(x0), len(y0), len(x1), len(y1))
        for i in range(n):
            ctx.move_to(self._x(x0[i]), self._y(y0[i]))
            ctx.line_to(self._x(x1[i]), self._y(y1[i]))

        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()
        ctx.restore()

    def draw_polygon(
        self,
        x: np.ndarray,
        y: np.ndarray,
        gp: Optional[Gpar] = None,
    ) -> None:
        if len(x) < 3:
            return
        ctx = self._ctx
        ctx.save()

        ctx.move_to(self._x(x[0]), self._y(y[0]))
        for i in range(1, len(x)):
            ctx.line_to(self._x(x[i]), self._y(y[i]))
        ctx.close_path()

        fill = self._fill_rgba(gp)
        if fill[3] > 0:
            ctx.set_source_rgba(*fill)
            ctx.fill_preserve()

        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()
        ctx.restore()

    def draw_path(
        self,
        x: np.ndarray,
        y: np.ndarray,
        path_id: np.ndarray,
        rule: str = "winding",
        gp: Optional[Gpar] = None,
    ) -> None:
        ctx = self._ctx
        ctx.save()

        fill_rule = (
            cairo.FILL_RULE_EVEN_ODD
            if rule == "evenodd"
            else cairo.FILL_RULE_WINDING
        )
        ctx.set_fill_rule(fill_rule)

        for pid in np.unique(path_id):
            mask = path_id == pid
            px = x[mask]
            py = y[mask]
            if len(px) < 2:
                continue
            ctx.move_to(self._x(px[0]), self._y(py[0]))
            for i in range(1, len(px)):
                ctx.line_to(self._x(px[i]), self._y(py[i]))
            ctx.close_path()

        fill = self._fill_rgba(gp)
        if fill[3] > 0:
            ctx.set_source_rgba(*fill)
            ctx.fill_preserve()

        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()
        ctx.restore()

    def draw_text(
        self,
        x: float,
        y: float,
        label: str,
        rot: float = 0.0,
        hjust: float = 0.5,
        vjust: float = 0.5,
        gp: Optional[Gpar] = None,
    ) -> None:
        ctx = self._ctx
        ctx.save()

        self._set_font(gp)
        stroke = self._apply_stroke(gp)

        ext = ctx.text_extents(str(label))
        tw = ext.width
        th = ext.height

        # Anchor position in device coords
        dx = self._x(x)
        dy = self._y(y)

        # Justification offsets
        off_x = -tw * hjust
        off_y = th * vjust

        if rot != 0.0:
            ctx.translate(dx, dy)
            ctx.rotate(-math.radians(rot))  # negative: grid uses CCW
            ctx.move_to(off_x, off_y)
        else:
            ctx.move_to(dx + off_x, dy + off_y)

        ctx.show_text(str(label))
        ctx.restore()

    def draw_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        size: float = 1.0,
        pch: int = 19,
        gp: Optional[Gpar] = None,
    ) -> None:
        """Draw point markers.  Currently renders filled circles (pch 19)."""
        ctx = self._ctx
        ctx.save()

        # Size: approximate pt-to-device scaling
        if self._surface_type == "image":
            r = size * self.dpi / 72.0 * 0.5
        else:
            r = size * 0.5

        fill = self._fill_rgba(gp)
        stroke_rgba = self._apply_stroke(gp)

        # Handle per-point colours
        col = gp.get("col", None) if gp else None
        col_list: Optional[list] = None
        if isinstance(col, (list, tuple, np.ndarray)) and len(col) == len(x):
            col_list = [_parse_colour(c) for c in col]

        for i in range(len(x)):
            cx = self._x(x[i])
            cy = self._y(y[i])
            ctx.arc(cx, cy, r, 0, 2 * math.pi)

            if col_list is not None:
                c = col_list[i]
                ctx.set_source_rgba(*c)
                ctx.fill_preserve()
                ctx.set_source_rgba(*c)
                ctx.stroke()
            else:
                if fill[3] > 0:
                    ctx.set_source_rgba(*fill)
                    ctx.fill_preserve()
                if stroke_rgba[3] > 0:
                    ctx.set_source_rgba(*stroke_rgba)
                    ctx.stroke()
                else:
                    ctx.new_path()

        ctx.restore()

    def draw_raster(
        self,
        image: Any,
        x: float,
        y: float,
        w: float,
        h: float,
        interpolate: bool = True,
    ) -> None:
        ctx = self._ctx
        ctx.save()

        img_array = np.asarray(image, dtype=np.uint8)
        if img_array.ndim == 2:
            # Greyscale → RGBA
            rgba = np.stack([img_array] * 3 + [np.full_like(img_array, 255)], axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            # RGB → RGBA
            rgba = np.concatenate(
                [img_array, np.full((*img_array.shape[:2], 1), 255, dtype=np.uint8)],
                axis=2,
            )
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            rgba = img_array
        else:
            ctx.restore()
            return

        # Cairo expects BGRA premultiplied in native byte order
        img_h, img_w = rgba.shape[:2]
        bgra = np.empty((img_h, img_w, 4), dtype=np.uint8)
        bgra[:, :, 0] = rgba[:, :, 2]  # B
        bgra[:, :, 1] = rgba[:, :, 1]  # G
        bgra[:, :, 2] = rgba[:, :, 0]  # R
        bgra[:, :, 3] = rgba[:, :, 3]  # A

        stride = cairo.ImageSurface.format_stride_for_width(
            cairo.FORMAT_ARGB32, img_w
        )
        img_surface = cairo.ImageSurface.create_for_data(
            bytearray(bgra.tobytes()), cairo.FORMAT_ARGB32, img_w, img_h, stride
        )

        dx = self._x(x)
        dy = self._y(y + h)
        dw = self._sx(w)
        dh = self._sy(h)

        ctx.translate(dx, dy)
        ctx.scale(dw / img_w, dh / img_h)
        ctx.set_source_surface(img_surface, 0, 0)
        pattern = ctx.get_source()
        if interpolate:
            pattern.set_filter(cairo.FILTER_BILINEAR)
        else:
            pattern.set_filter(cairo.FILTER_NEAREST)
        ctx.paint()

        ctx.restore()

    def draw_roundrect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        r: float = 0.0,
        hjust: float = 0.5,
        vjust: float = 0.5,
        gp: Optional[Gpar] = None,
    ) -> None:
        """Draw a rounded rectangle."""
        ctx = self._ctx
        ctx.save()

        x0 = x - w * hjust
        y0 = y - h * vjust
        dx = self._x(x0)
        dy = self._y(y0 + h)
        dw = self._sx(w)
        dh = self._sy(h)
        dr = min(self._sx(r), dw / 2, dh / 2)

        if dr <= 0:
            ctx.rectangle(dx, dy, dw, dh)
        else:
            ctx.new_path()
            ctx.arc(dx + dw - dr, dy + dr, dr, -math.pi / 2, 0)
            ctx.arc(dx + dw - dr, dy + dh - dr, dr, 0, math.pi / 2)
            ctx.arc(dx + dr, dy + dh - dr, dr, math.pi / 2, math.pi)
            ctx.arc(dx + dr, dy + dr, dr, math.pi, 3 * math.pi / 2)
            ctx.close_path()

        fill = self._fill_rgba(gp)
        if fill[3] > 0:
            ctx.set_source_rgba(*fill)
            ctx.fill_preserve()

        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()

        ctx.restore()

    # ---- pen-position drawing (move.to / line.to) --------------------------

    def move_to(self, x: float, y: float) -> None:
        self._pen_x = x
        self._pen_y = y

    def line_to(self, x: float, y: float, gp: Optional[Gpar] = None) -> None:
        ctx = self._ctx
        ctx.save()
        stroke = self._apply_stroke(gp)
        x0 = getattr(self, "_pen_x", 0.0)
        y0 = getattr(self, "_pen_y", 0.0)
        ctx.move_to(self._x(x0), self._y(y0))
        ctx.line_to(self._x(x), self._y(y))
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()
        self._pen_x = x
        self._pen_y = y
        ctx.restore()

    # ---- clipping ----------------------------------------------------------

    def push_clip(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._ctx.save()
        dx0 = self._x(min(x0, x1))
        dy0 = self._y(max(y0, y1))
        dw = self._sx(abs(x1 - x0))
        dh = self._sy(abs(y1 - y0))
        self._ctx.rectangle(dx0, dy0, dw, dh)
        self._ctx.clip()

    def pop_clip(self) -> None:
        self._ctx.restore()

    # ---- page / output -----------------------------------------------------

    def new_page(self, bg: Any = "white") -> None:
        """Clear the surface and start a fresh page."""
        if self._surface_type == "image":
            # Clear and repaint background
            self._ctx.set_operator(cairo.OPERATOR_SOURCE)
            bg_rgba = _parse_colour(bg)
            self._ctx.set_source_rgba(*bg_rgba)
            self._ctx.paint()
            self._ctx.set_operator(cairo.OPERATOR_OVER)
        else:
            # Vector surfaces: show_page starts a new page
            self._ctx.show_page()

    def write_to_png(self, filename: str) -> None:
        """Write the current surface to a PNG file."""
        self._surface.write_to_png(filename)

    def to_png_bytes(self) -> bytes:
        """Return the current surface as PNG bytes."""
        buf = io.BytesIO()
        self._surface.write_to_png(buf)
        buf.seek(0)
        return buf.read()

    def finish(self) -> None:
        """Finalise the surface (required for PDF/SVG/PS)."""
        self._surface.finish()

    # ---- text metrics (for _size.py) ---------------------------------------

    def text_extents(
        self, text: str, gp: Optional[Gpar] = None
    ) -> Dict[str, float]:
        """Measure text dimensions in inches.

        Returns dict with ``ascent``, ``descent``, ``width`` in inches.
        """
        ctx = self._ctx
        ctx.save()
        self._set_font(gp)

        fe = ctx.font_extents()
        te = ctx.text_extents(text)

        # Convert from device units back to inches
        if self._surface_type == "image":
            scale = 1.0 / self.dpi
        else:
            scale = 1.0 / 72.0

        ascent = fe[0] * scale
        descent = fe[1] * scale
        width = te.x_advance * scale

        ctx.restore()
        return {"ascent": ascent, "descent": descent, "width": width}
