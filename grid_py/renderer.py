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

from ._colour import parse_r_colour as _parse_colour
from ._gpar import Gpar
from ._patterns import LinearGradient, RadialGradient, Pattern
from ._renderer_base import GridRenderer

__all__ = ["CairoRenderer"]


# _parse_colour is imported from ._colour (shared R colour table)


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

class CairoRenderer(GridRenderer):
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
        self._surface_type = surface_type
        self._width_px = int(width * dpi)
        self._height_px = int(height * dpi)

        # Compute device dimensions for the base class
        if surface_type == "image":
            dw = float(self._width_px)
            dh = float(self._height_px)
        else:
            dw = width * 72.0
            dh = height * 72.0

        super().__init__(width, height, dpi, device_width=dw, device_height=dh)

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

    # ---- abstract method implementations: state save/restore ----------------

    def save_state(self) -> None:
        self._ctx.save()

    def restore_state(self) -> None:
        self._ctx.restore()

    # ---- abstract method implementations: clipping -------------------------

    def _apply_clip_rect(self, x0: float, y0: float, w: float, h: float) -> None:
        self._ctx.save()
        self._ctx.rectangle(x0, y0, w, h)
        self._ctx.clip()

    def _restore_clip(self) -> None:
        self._ctx.restore()

    # ---- path collection mode (R 4.2+ fill/stroke grobs) -------------------

    def begin_path_collect(self, rule: str = "winding") -> None:
        """Enter path-collecting mode.

        While active, draw_* methods build the Cairo path without
        filling or stroking.  Call one of the ``end_path_*`` methods
        to finalise.

        Mirrors R's ``C_stroke``/``C_fill``/``C_fillStroke`` pattern
        (``grid/src/path.c``).
        """
        self._ctx.new_path()
        self._path_collecting = True
        self._ctx.set_fill_rule(
            cairo.FILL_RULE_EVEN_ODD if rule == "evenodd"
            else cairo.FILL_RULE_WINDING
        )

    def end_path_stroke(self, gp: Optional[Any] = None) -> None:
        """End path collection with stroke only (no fill)."""
        self._path_collecting = False
        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            self._ctx.stroke()
        else:
            self._ctx.new_path()

    def end_path_fill(self, gp: Optional[Any] = None) -> None:
        """End path collection with fill only (no stroke)."""
        self._path_collecting = False
        bbox = self._ctx.path_extents()
        self._apply_fill(gp, bbox=bbox)
        self._ctx.new_path()

    def end_path_fill_stroke(self, gp: Optional[Any] = None) -> None:
        """End path collection with fill then stroke."""
        self._path_collecting = False
        bbox = self._ctx.path_extents()
        self._apply_fill(gp, bbox=bbox)
        stroke = self._apply_stroke(gp)
        if stroke[3] > 0:
            self._ctx.stroke()
        else:
            self._ctx.new_path()

    def render_mask(self, mask_grob: Any) -> Optional["cairo.ImageSurface"]:
        """Render a mask grob to an off-screen alpha surface.

        Mirrors R's ``resolveMask.GridMask`` which draws the mask grob
        and extracts the alpha channel for compositing.

        Parameters
        ----------
        mask_grob : grob
            A grob to render as the mask.

        Returns
        -------
        cairo.ImageSurface or None
            An ARGB32 surface whose alpha channel is the mask, or
            ``None`` on failure.
        """
        x0, y0, pw, ph = self.get_viewport_bounds()
        dw = max(int(round(pw)), 1)
        dh = max(int(round(ph)), 1)

        mask_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, dw, dh)
        mask_renderer = CairoRenderer.__new__(CairoRenderer)
        # Initialise base class state directly (bypass __init__ for perf)
        mask_renderer.width_in = float(dw) / self.dpi
        mask_renderer.height_in = float(dh) / self.dpi
        mask_renderer.dpi = self.dpi
        # Initialize the new transform stack for the mask renderer
        from ._vp_calc import calc_root_transform
        mask_w_in = float(dw) / self.dpi
        mask_h_in = float(dh) / self.dpi
        root_vtr = calc_root_transform(mask_w_in * 2.54, mask_h_in * 2.54)
        mask_renderer._vp_transform_stack = [root_vtr]
        mask_renderer._vp_obj_stack = [None]
        mask_renderer._layout_stack = []
        mask_renderer._layout_depth_stack = []
        mask_renderer._clip_stack = []
        mask_renderer._path_collecting = False
        mask_renderer._pen_x = 0.0
        mask_renderer._pen_y = 0.0
        mask_renderer._device_width = float(dw)
        mask_renderer._device_height = float(dh)
        mask_renderer._device_width_cm = mask_w_in * 2.54
        mask_renderer._device_height_cm = mask_h_in * 2.54
        mask_renderer._dev_units_per_inch = self.dpi
        # Cairo-specific state
        mask_renderer._surface = mask_surface
        mask_renderer._ctx = cairo.Context(mask_surface)
        mask_renderer._surface_type = "image"
        mask_renderer._width_px = dw
        mask_renderer._height_px = dh

        # Clear to transparent
        mask_renderer._ctx.set_source_rgba(0, 0, 0, 0)
        mask_renderer._ctx.paint()

        try:
            from ._draw import grid_draw
            from ._state import get_state
            state = get_state()
            orig_renderer = state._renderer
            state._renderer = mask_renderer
            try:
                grid_draw(mask_grob, recording=False)
            finally:
                state._renderer = orig_renderer
        except Exception:
            return None

        return mask_surface

    def apply_mask(
        self,
        mask_surface: "cairo.ImageSurface",
        mask_type: str = "alpha",
    ) -> None:
        """Apply a pre-rendered mask surface to the current drawing.

        For ``type="alpha"``, the mask's alpha channel is used directly.
        For ``type="luminance"``, the mask's luminance (brightness) is
        converted to alpha.

        Parameters
        ----------
        mask_surface : cairo.ImageSurface
            The rendered mask surface.
        mask_type : str
            ``"alpha"`` or ``"luminance"``.
        """
        x0, y0, pw, ph = self.get_viewport_bounds()

        if mask_type == "luminance":
            # Convert luminance to alpha: iterate pixels and set alpha
            # based on brightness. This is approximate; full implementation
            # would need per-pixel manipulation.
            pass  # Cairo doesn't natively support luminance masks;
            # the alpha channel is used as-is for now.

        # Apply mask at the current viewport position
        self._ctx.mask_surface(mask_surface, x0, y0)

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
        # R semantics: col=NA means "no border" → transparent.
        # But col=None (unset) should fall back to the default black,
        # matching R's get.gpar()$col default of "black".
        if col_val is None:
            col_val = "black"
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

    def _fill_rgba(
        self,
        gp: Optional[Gpar],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Union[Tuple[float, float, float, float], "cairo.Pattern"]:
        """Extract fill colour or gradient pattern from Gpar.

        Returns either an RGBA tuple for solid colours, or a
        ``cairo.LinearGradient`` / ``cairo.RadialGradient`` pattern
        for gradient fills.

        Parameters
        ----------
        gp : Gpar or None
            Graphical parameters.
        bbox : tuple or None
            Shape bounding box ``(x, y, w, h)`` in NPC, passed to
            ``_setup_gradient`` for ``group=False`` resolution.
        """
        if gp is None:
            return (1.0, 1.0, 1.0, 1.0)
        fill = gp.get("fill", None)

        # Handle gradient objects (LinearGradient / RadialGradient)
        if isinstance(fill, (LinearGradient, RadialGradient)):
            return self._setup_gradient(fill, gp, bbox=bbox)
        # Handle tiling pattern
        if isinstance(fill, Pattern):
            return self._setup_pattern(fill, gp, bbox=bbox)

        # Distinguish RGBA colour tuples (r,g,b,a) from lists of colour values.
        # An RGBA tuple has 3-4 numeric elements all in [0,1]; a colour list
        # contains strings, None, or gradient/pattern objects.
        if isinstance(fill, tuple) and len(fill) in (3, 4) and all(
            isinstance(c, (int, float)) for c in fill
        ):
            # Direct RGBA/RGB tuple — treat as a single colour
            fill_val = fill
        elif isinstance(fill, (list, tuple)):
            fill_val = fill[0]
        else:
            fill_val = fill

        # R semantics: fill=NA (None) means "no fill" → transparent
        if fill_val is None:
            return (0.0, 0.0, 0.0, 0.0)

        # Check if a scalar fill_val is a gradient or pattern object
        if isinstance(fill_val, (LinearGradient, RadialGradient)):
            return self._setup_gradient(fill_val, gp, bbox=bbox)
        if isinstance(fill_val, Pattern):
            return self._setup_pattern(fill_val, gp, bbox=bbox)

        rgba = _parse_colour(fill_val)

        alpha = gp.get("alpha", None)
        if alpha is not None:
            a = float(alpha[0] if isinstance(alpha, (list, tuple)) else alpha)
            rgba = (rgba[0], rgba[1], rgba[2], rgba[3] * a)
        return rgba

    def _setup_gradient(
        self,
        gradient: Union[LinearGradient, RadialGradient],
        gp: Optional[Gpar] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> "cairo.Pattern":
        """Convert a grid gradient object to a cairo Pattern.

        When ``gradient.group`` is ``True`` (default), coordinates are
        resolved relative to the current viewport.  When ``False``,
        coordinates are resolved relative to the shape's bounding box
        given by *bbox* ``(x, y, w, h)`` in NPC.

        Mirrors R's ``resolvePattern()`` in ``patterns.R:391-418``.

        Parameters
        ----------
        gradient : LinearGradient or RadialGradient
            The gradient specification.
        gp : Gpar or None
            Graphical parameters (for alpha).
        bbox : tuple of float or None
            Shape bounding box ``(x, y, w, h)`` in NPC, used when
            ``gradient.group is False``.

        Returns
        -------
        cairo.Pattern
            A cairo linear or radial gradient pattern.
        """
        # Get alpha from gpar
        gp_alpha = 1.0
        if gp is not None:
            a = gp.get("alpha", None)
            if a is not None:
                gp_alpha = float(a[0] if isinstance(a, (list, tuple)) else a)

        # Determine coordinate mapping: viewport (group=True) or bbox (group=False)
        use_bbox = (not gradient.group) and (bbox is not None)

        if isinstance(gradient, LinearGradient):
            gx1 = float(gradient.x1.values[0])
            gy1 = float(gradient.y1.values[0])
            gx2 = float(gradient.x2.values[0])
            gy2 = float(gradient.y2.values[0])

            if use_bbox:
                # Map gradient NPC coords to shape bbox
                bx, by, bw, bh = bbox
                x1_npc = bx + gx1 * bw
                y1_npc = by + gy1 * bh
                x2_npc = bx + gx2 * bw
                y2_npc = by + gy2 * bh
            else:
                x1_npc, y1_npc = gx1, gy1
                x2_npc, y2_npc = gx2, gy2

            pattern = cairo.LinearGradient(
                self._x(x1_npc), self._y(y1_npc),
                self._x(x2_npc), self._y(y2_npc),
            )
        elif isinstance(gradient, RadialGradient):
            gcx1 = float(gradient.cx1.values[0])
            gcy1 = float(gradient.cy1.values[0])
            gr1 = float(gradient.r1.values[0])
            gcx2 = float(gradient.cx2.values[0])
            gcy2 = float(gradient.cy2.values[0])
            gr2 = float(gradient.r2.values[0])

            if use_bbox:
                bx, by, bw, bh = bbox
                cx1_npc = bx + gcx1 * bw
                cy1_npc = by + gcy1 * bh
                cx2_npc = bx + gcx2 * bw
                cy2_npc = by + gcy2 * bh
                # Scale radius by bbox size
                r1_npc = gr1 * min(bw, bh)
                r2_npc = gr2 * min(bw, bh)
            else:
                cx1_npc, cy1_npc = gcx1, gcy1
                cx2_npc, cy2_npc = gcx2, gcy2
                r1_npc, r2_npc = gr1, gr2

            r1_dev = (self._sx(r1_npc) + self._sy(r1_npc)) / 2.0
            r2_dev = (self._sx(r2_npc) + self._sy(r2_npc)) / 2.0

            pattern = cairo.RadialGradient(
                self._x(cx1_npc), self._y(cy1_npc), r1_dev,
                self._x(cx2_npc), self._y(cy2_npc), r2_dev,
            )
        else:
            return (0.0, 0.0, 0.0, 0.0)

        # Add colour stops
        for colour_str, stop in zip(gradient.colours, gradient.stops):
            r, g, b, a = _parse_colour(colour_str)
            pattern.add_color_stop_rgba(stop, r, g, b, a * gp_alpha)

        # Set extend mode
        extend_map = {
            "pad": cairo.EXTEND_PAD,
            "repeat": cairo.EXTEND_REPEAT,
            "reflect": cairo.EXTEND_REFLECT,
            "none": cairo.EXTEND_NONE,
        }
        pattern.set_extend(extend_map.get(gradient.extend, cairo.EXTEND_PAD))

        return pattern

    def _setup_pattern(
        self,
        pat: "Pattern",
        gp: Optional[Gpar] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> "cairo.Pattern":
        """Convert a grid tiling Pattern to a cairo SurfacePattern.

        Renders the pattern's grob to an off-screen Cairo surface, then
        creates a ``cairo.SurfacePattern`` with the appropriate extend mode.

        Mirrors R's ``resolvePattern.GridTilingPattern`` (patterns.R:420-429).

        Parameters
        ----------
        pat : Pattern
            The tiling pattern specification.
        gp : Gpar or None
            Graphical parameters (for alpha).
        bbox : tuple or None
            Shape bounding box in NPC (for group=False resolution).

        Returns
        -------
        cairo.SurfacePattern
            A cairo surface pattern for tiling.
        """
        # Resolve tile position and dimensions in NPC
        px = float(pat.x.values[0])
        py = float(pat.y.values[0])
        pw = float(pat.width.values[0])
        ph = float(pat.height.values[0])

        if not pat.group and bbox is not None:
            bx, by, bw, bh = bbox
            px = bx + px * bw
            py = by + py * bh
            pw = pw * bw
            ph = ph * bh

        # Compute tile origin (left, bottom) using justification
        tile_left = px - pat.hjust * pw
        tile_bottom = py - pat.vjust * ph

        # Convert to device coordinates
        tile_dx = self._x(tile_left)
        tile_dy = self._y(tile_bottom + ph)  # Y-flip: top in device coords
        tile_dw = max(int(round(self._sx(pw))), 1)
        tile_dh = max(int(round(self._sy(ph))), 1)

        # Render the pattern grob to an off-screen surface
        tile_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, tile_dw, tile_dh)
        tile_ctx = tile_surface

        # Create a temporary renderer for the tile
        tile_renderer = CairoRenderer(
            width=pw * 2.54,  # convert NPC width to approximate inches
            height=ph * 2.54,
            dpi=self.dpi,
            surface_type="image",
        )

        # Draw the grob into the tile using state swap
        try:
            from ._draw import grid_draw
            from ._state import get_state
            state = get_state()
            orig_renderer = state._renderer
            state._renderer = tile_renderer
            # Reset viewport to root for clean drawing
            tile_renderer.push_viewport(None)
            try:
                grid_draw(pat.grob, recording=False)
            finally:
                state._renderer = orig_renderer
        except Exception:
            # If grob rendering fails, return transparent
            return (0.0, 0.0, 0.0, 0.0)

        tile_surface = tile_renderer._surface

        # Create a surface pattern from the rendered tile
        surface_pattern = cairo.SurfacePattern(tile_surface)

        # Set extend mode
        extend_map = {
            "pad": cairo.EXTEND_PAD,
            "repeat": cairo.EXTEND_REPEAT,
            "reflect": cairo.EXTEND_REFLECT,
            "none": cairo.EXTEND_NONE,
        }
        surface_pattern.set_extend(extend_map.get(pat.extend, cairo.EXTEND_REPEAT))

        # Set the pattern matrix to position the tile correctly
        # The pattern needs to be translated to the tile's position in device space
        matrix = cairo.Matrix()
        matrix.translate(-tile_dx, -tile_dy)
        surface_pattern.set_matrix(matrix)

        return surface_pattern

    def _apply_fill(
        self,
        gp: Optional[Gpar],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        """Apply fill (solid colour or gradient) to the current path.

        Parameters
        ----------
        gp : Gpar or None
            Graphical parameters.
        bbox : tuple or None
            Shape bounding box ``(x, y, w, h)`` in NPC for
            ``group=False`` gradient resolution.

        Returns ``True`` if a fill was applied, ``False`` if transparent.
        """
        ctx = self._ctx
        fill = self._fill_rgba(gp, bbox=bbox)
        if isinstance(fill, cairo.Pattern):
            ctx.set_source(fill)
            ctx.fill_preserve()
            return True
        elif isinstance(fill, tuple) and fill[3] > 0:
            ctx.set_source_rgba(*fill)
            ctx.fill_preserve()
            return True
        return False

    def _set_font(self, gp: Optional[Gpar]) -> float:
        """Configure font on context from Gpar.  Returns font size in device units.

        Mirrors R's gpar.c:391-398 where ``gc->ps = fontsize * GSS_SCALE``
        and the device uses ``gc->ps * gc->cex`` for the effective size.
        """
        ctx = self._ctx
        family = "sans-serif"
        slant = cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_NORMAL
        fontsize = 12.0  # points
        cex_val = 1.0

        if gp is not None:
            ff = gp.get("fontfamily", None)
            if ff is not None:
                family = str(ff[0] if isinstance(ff, (list, tuple)) else ff)

            fs = gp.get("fontsize", None)
            if fs is not None:
                fontsize = float(fs[0] if isinstance(fs, (list, tuple)) else fs)

            cex = gp.get("cex", None)
            if cex is not None:
                cex_val = float(cex[0] if isinstance(cex, (list, tuple)) else cex)

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

        # R: gc->ps = fontsize * GSS_SCALE;  effective = gc->ps * gc->cex
        scale = self._get_scale()
        fontsize = fontsize * scale * cex_val

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
        """Draw a rectangle.  x, y, w, h are in device coordinates."""
        ctx = self._ctx
        ctx.save()

        # x,y is the anchor point; apply justification to get top-left
        dx = x - w * hjust
        dy = y - h * (1.0 - vjust)  # device y increases downward

        ctx.rectangle(dx, dy, w, h)

        if self._path_collecting:
            ctx.restore()
            return

        bbox = (dx, dy, w, h)
        self._apply_fill(gp, bbox=bbox)

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
        """Draw a circle.  x, y, r are in device coordinates."""
        ctx = self._ctx
        ctx.save()

        ctx.arc(x, y, r, 0, 2 * math.pi)

        if self._path_collecting:
            ctx.restore()
            return

        bbox = (x - r, y - r, 2 * r, 2 * r)
        self._apply_fill(gp, bbox=bbox)

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
        """Draw connected lines.  x, y are in device coordinates."""
        n = max(len(x), len(y))
        if n < 2:
            return
        if len(x) < n:
            x = np.resize(x, n)
        if len(y) < n:
            y = np.resize(y, n)

        ctx = self._ctx
        ctx.save()

        ctx.move_to(x[0], y[0])
        for i in range(1, n):
            ctx.line_to(x[i], y[i])

        if self._path_collecting:
            ctx.restore()
            return

        stroke = self._apply_stroke(gp)
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
            ctx.move_to(px[0], py[0])
            for i in range(1, len(px)):
                ctx.line_to(px[i], py[i])

        if self._path_collecting:
            ctx.restore()
            return

        stroke = self._apply_stroke(gp)
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

        n = min(len(x0), len(y0), len(x1), len(y1))
        for i in range(n):
            ctx.move_to(x0[i], y0[i])
            ctx.line_to(x1[i], y1[i])

        if self._path_collecting:
            ctx.restore()
            return

        stroke = self._apply_stroke(gp)
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

        ctx.move_to(x[0], y[0])
        for i in range(1, len(x)):
            ctx.line_to(x[i], y[i])
        ctx.close_path()

        if self._path_collecting:
            ctx.restore()
            return

        bbox = (float(np.min(x)), float(np.min(y)),
                float(np.ptp(x)), float(np.ptp(y)))
        self._apply_fill(gp, bbox=bbox)

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
            ctx.move_to(px[0], py[0])
            for i in range(1, len(px)):
                ctx.line_to(px[i], py[i])
            ctx.close_path()

        if self._path_collecting:
            ctx.restore()
            return

        bbox = (float(np.min(x)), float(np.min(y)),
                float(np.ptp(x)), float(np.ptp(y)))
        self._apply_fill(gp, bbox=bbox)

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

        # x, y are already in device coords
        off_x = -tw * hjust
        off_y = th * vjust

        if rot != 0.0:
            ctx.translate(x, y)
            ctx.rotate(-math.radians(rot))  # negative: grid uses CCW
            ctx.move_to(off_x, off_y)
        else:
            ctx.move_to(x + off_x, y + off_y)

        if self._path_collecting:
            # Build text outline path without rendering pixels
            # (mirrors R's use of cairo_text_path in C_stroke/C_fill)
            ctx.text_path(str(label))
        else:
            ctx.show_text(str(label))
        ctx.restore()

    # ------------------------------------------------------------------ #
    #  pch shape path helpers (R pch 0-25)                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pch_path_circle(ctx: Any, cx: float, cy: float, r: float) -> None:
        ctx.arc(cx, cy, r, 0, 2 * math.pi)

    @staticmethod
    def _pch_path_square(ctx: Any, cx: float, cy: float, r: float) -> None:
        ctx.rectangle(cx - r, cy - r, 2 * r, 2 * r)

    @staticmethod
    def _pch_path_triangle_up(ctx: Any, cx: float, cy: float, r: float) -> None:
        h = r * 1.2  # slightly taller for visual balance
        ctx.move_to(cx, cy - h)
        ctx.line_to(cx - h, cy + h * 0.7)
        ctx.line_to(cx + h, cy + h * 0.7)
        ctx.close_path()

    @staticmethod
    def _pch_path_triangle_down(ctx: Any, cx: float, cy: float, r: float) -> None:
        h = r * 1.2
        ctx.move_to(cx, cy + h)
        ctx.line_to(cx - h, cy - h * 0.7)
        ctx.line_to(cx + h, cy - h * 0.7)
        ctx.close_path()

    @staticmethod
    def _pch_path_diamond(ctx: Any, cx: float, cy: float, r: float) -> None:
        ctx.move_to(cx, cy - r)
        ctx.line_to(cx + r, cy)
        ctx.line_to(cx, cy + r)
        ctx.line_to(cx - r, cy)
        ctx.close_path()

    @staticmethod
    def _pch_path_plus(ctx: Any, cx: float, cy: float, r: float) -> None:
        ctx.move_to(cx - r, cy)
        ctx.line_to(cx + r, cy)
        ctx.move_to(cx, cy - r)
        ctx.line_to(cx, cy + r)

    @staticmethod
    def _pch_path_cross(ctx: Any, cx: float, cy: float, r: float) -> None:
        d = r * 0.707  # r / sqrt(2)
        ctx.move_to(cx - d, cy - d)
        ctx.line_to(cx + d, cy + d)
        ctx.move_to(cx - d, cy + d)
        ctx.line_to(cx + d, cy - d)

    @staticmethod
    def _pch_path_asterisk(ctx: Any, cx: float, cy: float, r: float) -> None:
        """6-armed asterisk (pch 8)."""
        for angle_deg in (0, 60, 120):
            a = math.radians(angle_deg)
            dx = r * math.cos(a)
            dy = r * math.sin(a)
            ctx.move_to(cx - dx, cy - dy)
            ctx.line_to(cx + dx, cy + dy)

    @staticmethod
    def _pch_path_star(ctx: Any, cx: float, cy: float, r: float) -> None:
        """5-pointed star outline (pch 11) – two overlaid triangles."""
        h = r * 1.2
        # up triangle
        ctx.move_to(cx, cy - h)
        ctx.line_to(cx - h, cy + h * 0.7)
        ctx.line_to(cx + h, cy + h * 0.7)
        ctx.close_path()
        # down triangle
        ctx.move_to(cx, cy + h)
        ctx.line_to(cx - h, cy - h * 0.7)
        ctx.line_to(cx + h, cy - h * 0.7)
        ctx.close_path()

    def _draw_pch_shape(
        self,
        ctx: Any,
        pch_val: int,
        cx: float,
        cy: float,
        r: float,
        col_rgba: Tuple[float, float, float, float],
        fill_rgba: Tuple[float, float, float, float],
        lwd: float,
    ) -> None:
        """Draw a single R pch shape at (*cx*, *cy*) with radius *r*.

        Follows R semantics for pch groups:
        - 0-14 : open / line-only shapes — stroke with *col*, no fill
        - 15-18: solid filled shapes — both stroke and fill use *col*
        - 19-20: filled circles — fill and stroke use *col*
        - 21-25: filled shapes with separate fill and col
        """
        ctx.save()
        ctx.new_path()  # clear any residual path from prior draws
        ctx.set_line_width(lwd)

        if pch_val <= 14:
            # --- Group 0-14: stroke-only (use col for outline, no fill) ---
            if pch_val == 0:                       # square open
                self._pch_path_square(ctx, cx, cy, r)
            elif pch_val == 1:                     # circle open
                self._pch_path_circle(ctx, cx, cy, r)
            elif pch_val == 2:                     # triangle open
                self._pch_path_triangle_up(ctx, cx, cy, r)
            elif pch_val == 3:                     # plus
                self._pch_path_plus(ctx, cx, cy, r)
            elif pch_val == 4:                     # cross (×)
                self._pch_path_cross(ctx, cx, cy, r)
            elif pch_val == 5:                     # diamond open
                self._pch_path_diamond(ctx, cx, cy, r)
            elif pch_val == 6:                     # triangle down open
                self._pch_path_triangle_down(ctx, cx, cy, r)
            elif pch_val == 7:                     # square cross
                self._pch_path_square(ctx, cx, cy, r)
                self._pch_path_cross(ctx, cx, cy, r)
            elif pch_val == 8:                     # asterisk
                self._pch_path_asterisk(ctx, cx, cy, r)
            elif pch_val == 9:                     # diamond plus
                self._pch_path_diamond(ctx, cx, cy, r)
                self._pch_path_plus(ctx, cx, cy, r)
            elif pch_val == 10:                    # circle plus
                self._pch_path_circle(ctx, cx, cy, r)
                self._pch_path_plus(ctx, cx, cy, r)
            elif pch_val == 11:                    # star (two triangles)
                self._pch_path_star(ctx, cx, cy, r)
            elif pch_val == 12:                    # square plus
                self._pch_path_square(ctx, cx, cy, r)
                self._pch_path_plus(ctx, cx, cy, r)
            elif pch_val == 13:                    # circle cross
                self._pch_path_circle(ctx, cx, cy, r)
                self._pch_path_cross(ctx, cx, cy, r)
            elif pch_val == 14:                    # square triangle
                self._pch_path_square(ctx, cx, cy, r)
                self._pch_path_triangle_up(ctx, cx, cy, r)

            if col_rgba[3] > 0:
                ctx.set_source_rgba(*col_rgba)
                ctx.stroke()
            else:
                ctx.new_path()

        elif pch_val <= 20:
            # --- Group 15-20: solid filled — col used for both fill & stroke ---
            if pch_val == 15:                      # square
                self._pch_path_square(ctx, cx, cy, r)
            elif pch_val == 16:                    # circle small
                self._pch_path_circle(ctx, cx, cy, r * 0.75)
            elif pch_val == 17:                    # triangle
                self._pch_path_triangle_up(ctx, cx, cy, r)
            elif pch_val == 18:                    # diamond
                self._pch_path_diamond(ctx, cx, cy, r)
            elif pch_val == 19:                    # circle (default)
                self._pch_path_circle(ctx, cx, cy, r)
            elif pch_val == 20:                    # bullet (small)
                self._pch_path_circle(ctx, cx, cy, r * 0.6)

            if col_rgba[3] > 0:
                ctx.set_source_rgba(*col_rgba)
                ctx.fill_preserve()
                ctx.set_source_rgba(*col_rgba)
                ctx.stroke()
            else:
                ctx.new_path()

        else:
            # --- Group 21-25: separate fill and col (stroke) ---
            if pch_val == 21:                      # circle filled
                self._pch_path_circle(ctx, cx, cy, r)
            elif pch_val == 22:                    # square filled
                self._pch_path_square(ctx, cx, cy, r)
            elif pch_val == 23:                    # diamond filled
                self._pch_path_diamond(ctx, cx, cy, r)
            elif pch_val == 24:                    # triangle filled
                self._pch_path_triangle_up(ctx, cx, cy, r)
            elif pch_val == 25:                    # triangle down filled
                self._pch_path_triangle_down(ctx, cx, cy, r)
            else:
                # fallback: circle
                self._pch_path_circle(ctx, cx, cy, r)

            if fill_rgba[3] > 0:
                ctx.set_source_rgba(*fill_rgba)
                ctx.fill_preserve()
            else:
                ctx.new_path()
                # re-draw path for stroke
                if pch_val == 21:
                    self._pch_path_circle(ctx, cx, cy, r)
                elif pch_val == 22:
                    self._pch_path_square(ctx, cx, cy, r)
                elif pch_val == 23:
                    self._pch_path_diamond(ctx, cx, cy, r)
                elif pch_val == 24:
                    self._pch_path_triangle_up(ctx, cx, cy, r)
                elif pch_val == 25:
                    self._pch_path_triangle_down(ctx, cx, cy, r)
                else:
                    self._pch_path_circle(ctx, cx, cy, r)

            if col_rgba[3] > 0:
                ctx.set_source_rgba(*col_rgba)
                ctx.stroke()
            else:
                ctx.new_path()

        ctx.restore()

    def draw_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        size: float = 1.0,
        pch: Any = 19,
        gp: Optional[Gpar] = None,
    ) -> None:
        """Draw point markers with full R pch 0-25 support.

        Parameters
        ----------
        x, y : array-like
            Point coordinates in NPC.
        size : float
            Fallback symbol size (used when ``gp.fontsize`` is absent).
        pch : int, array-like of int
            Plotting character code(s).  Scalar → same shape for all points;
            array → per-point shapes.
        gp : Gpar or None
            Graphical parameters (col, fill, fontsize, lwd, …).
        """
        ctx = self._ctx
        ctx.save()

        n = len(x)
        if n == 0:
            ctx.restore()
            return

        # --- per-point pch array ---
        if isinstance(pch, (list, tuple, np.ndarray)):
            pch_arr = np.asarray(pch, dtype=int)
        else:
            pch_arr = np.full(n, int(pch), dtype=int)
        if len(pch_arr) < n:
            pch_arr = np.resize(pch_arr, n)

        # --- per-point sizes from gpar.fontsize (R: cex * fontsize) ---
        fs = gp.get("fontsize", None) if gp else None
        if isinstance(fs, (list, tuple, np.ndarray)):
            size_arr = np.asarray(fs, dtype=float)
        elif fs is not None:
            size_arr = np.full(n, float(fs))
        else:
            size_arr = np.full(n, float(size))

        # Radius conversion: fontsize (pt) → device pixels
        scale = self.dpi / 72.0 * 0.5 if self._surface_type == "image" else 0.5

        # --- per-point colours (col) ---
        col_raw = gp.get("col", None) if gp else None
        if isinstance(col_raw, (list, tuple, np.ndarray)) and len(col_raw) >= n:
            col_list = [_parse_colour(c) for c in col_raw[:n]]
        elif col_raw is not None:
            c0 = _parse_colour(col_raw[0] if isinstance(col_raw, (list, tuple)) else col_raw)
            col_list = [c0] * n
        else:
            col_list = [(0.0, 0.0, 0.0, 1.0)] * n

        # --- per-point fill colours ---
        fill_raw = gp.get("fill", None) if gp else None
        if isinstance(fill_raw, (list, tuple, np.ndarray)) and len(fill_raw) >= n:
            fill_list = [_parse_colour(c) for c in fill_raw[:n]]
        elif fill_raw is not None:
            f0 = _parse_colour(fill_raw[0] if isinstance(fill_raw, (list, tuple)) else fill_raw)
            fill_list = [f0] * n
        else:
            fill_list = [(0.0, 0.0, 0.0, 0.0)] * n

        # --- per-point lwd ---
        lwd_raw = gp.get("lwd", None) if gp else None
        if isinstance(lwd_raw, (list, tuple, np.ndarray)):
            lwd_arr = np.asarray(lwd_raw, dtype=float)
        elif lwd_raw is not None:
            lwd_arr = np.full(n, float(lwd_raw))
        else:
            lwd_arr = np.full(n, 1.0)

        for i in range(n):
            cx = x[i]
            cy = y[i]
            r = size_arr[i] * scale if i < len(size_arr) else size * scale
            lwd_i = float(lwd_arr[i % len(lwd_arr)])
            self._draw_pch_shape(
                ctx, int(pch_arr[i]), cx, cy, r,
                col_rgba=col_list[i],
                fill_rgba=fill_list[i],
                lwd=lwd_i,
            )

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

        img_array = np.asarray(image)

        # Handle colour string arrays (e.g. from colourbar raster)
        # Convert colour strings to uint8 RGBA
        if img_array.dtype.kind in ("U", "S", "O"):
            h_img, w_img = img_array.shape[:2]
            rgba = np.zeros((h_img, w_img, 4), dtype=np.uint8)
            for r in range(h_img):
                for c in range(w_img):
                    colour = img_array[r, c] if img_array.ndim >= 2 else img_array[r]
                    cr, cg, cb, ca = _parse_colour(str(colour))
                    rgba[r, c] = [int(cr * 255), int(cg * 255),
                                  int(cb * 255), int(ca * 255)]
            img_array = rgba
        else:
            img_array = img_array.astype(np.uint8)
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

        # x, y, w, h are in device coords; y is top-left origin
        ctx.translate(x, y)
        ctx.scale(w / img_w, h / img_h)
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
        """Draw a rounded rectangle.  All coords in device units."""
        ctx = self._ctx
        ctx.save()

        dx = x - w * hjust
        dy = y - h * (1.0 - vjust)
        dr = min(r, w / 2, h / 2)

        if dr <= 0:
            ctx.rectangle(dx, dy, w, h)
        else:
            ctx.new_path()
            ctx.arc(dx + w - dr, dy + dr, dr, -math.pi / 2, 0)
            ctx.arc(dx + w - dr, dy + h - dr, dr, 0, math.pi / 2)
            ctx.arc(dx + dr, dy + h - dr, dr, math.pi / 2, math.pi)
            ctx.arc(dx + dr, dy + dr, dr, math.pi, 3 * math.pi / 2)
            ctx.close_path()

        if self._path_collecting:
            ctx.restore()
            return

        bbox = (dx, dy, w, h)
        self._apply_fill(gp, bbox=bbox)

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
        """Draw line from pen to (x,y).  Coords in device units."""
        ctx = self._ctx
        ctx.save()
        stroke = self._apply_stroke(gp)
        x0 = getattr(self, "_pen_x", 0.0)
        y0 = getattr(self, "_pen_y", 0.0)
        ctx.move_to(x0, y0)
        ctx.line_to(x, y)
        if stroke[3] > 0:
            ctx.stroke()
        else:
            ctx.new_path()
        self._pen_x = x
        self._pen_y = y
        ctx.restore()

    # ---- clipping ----------------------------------------------------------

    def push_clip(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Push clip rectangle.  Coords in device units."""
        self._ctx.save()
        cx = min(x0, x1)
        cy = min(y0, y1)
        cw = abs(x1 - x0)
        ch = abs(y1 - y0)
        self._ctx.rectangle(cx, cy, cw, ch)
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

    def get_surface(self) -> Any:
        """Return the underlying Cairo surface (for raster capture)."""
        return self._surface

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

    # ---- group compositing (R 4.1+ / group.R) -------------------------------

    # Porter-Duff + PDF blend mode → Cairo operator mapping
    # R group.R:272-287 lists all valid operators.
    _CAIRO_OPERATOR_MAP = {
        "clear": cairo.OPERATOR_CLEAR,
        "source": cairo.OPERATOR_SOURCE,
        "over": cairo.OPERATOR_OVER,
        "in": cairo.OPERATOR_IN,
        "out": cairo.OPERATOR_OUT,
        "atop": cairo.OPERATOR_ATOP,
        "dest": cairo.OPERATOR_DEST,
        "dest.over": cairo.OPERATOR_DEST_OVER,
        "dest.in": cairo.OPERATOR_DEST_IN,
        "dest.out": cairo.OPERATOR_DEST_OUT,
        "dest.atop": cairo.OPERATOR_DEST_ATOP,
        "xor": cairo.OPERATOR_XOR,
        "add": cairo.OPERATOR_ADD,
        "saturate": cairo.OPERATOR_SATURATE,
        # PDF blend modes (pycairo ≥ 1.12)
        "multiply": getattr(cairo, "OPERATOR_MULTIPLY", cairo.OPERATOR_OVER),
        "screen": getattr(cairo, "OPERATOR_SCREEN", cairo.OPERATOR_OVER),
        "overlay": getattr(cairo, "OPERATOR_OVERLAY", cairo.OPERATOR_OVER),
        "darken": getattr(cairo, "OPERATOR_DARKEN", cairo.OPERATOR_OVER),
        "lighten": getattr(cairo, "OPERATOR_LIGHTEN", cairo.OPERATOR_OVER),
        "color.dodge": getattr(cairo, "OPERATOR_COLOR_DODGE", cairo.OPERATOR_OVER),
        "color.burn": getattr(cairo, "OPERATOR_COLOR_BURN", cairo.OPERATOR_OVER),
        "hard.light": getattr(cairo, "OPERATOR_HARD_LIGHT", cairo.OPERATOR_OVER),
        "soft.light": getattr(cairo, "OPERATOR_SOFT_LIGHT", cairo.OPERATOR_OVER),
        "difference": getattr(cairo, "OPERATOR_DIFFERENCE", cairo.OPERATOR_OVER),
        "exclusion": getattr(cairo, "OPERATOR_EXCLUSION", cairo.OPERATOR_OVER),
    }

    def define_group(
        self, src_fn: Any, op: str = "over", dst_fn: Any = None,
    ) -> Any:
        """Define a compositing group on the Cairo surface.

        Port of R ``.defineGroup(src, op, dst)`` (group.R:263,302).
        Uses ``cairo.Context.push_group()`` / ``pop_group()`` to capture
        source and destination content, then composites them.

        Parameters
        ----------
        src_fn : callable
            Function that draws the source content.
        op : str
            Compositing operator name.
        dst_fn : callable or None
            Function that draws the destination content (None = transparent).

        Returns
        -------
        cairo.Pattern or None
            The composited group as a pattern, or None on failure.
        """
        ctx = self._ctx
        cairo_op = self._CAIRO_OPERATOR_MAP.get(op, cairo.OPERATOR_OVER)

        try:
            # Draw destination first (if any)
            if dst_fn is not None:
                ctx.push_group()
                dst_fn()
                dst_pattern = ctx.pop_group()
            else:
                dst_pattern = None

            # Draw source
            ctx.push_group()
            src_fn()
            src_pattern = ctx.pop_group()

            # Composite: dst first, then src with operator
            ctx.push_group()

            if dst_pattern is not None:
                ctx.set_source(dst_pattern)
                ctx.paint()

            ctx.set_operator(cairo_op)
            ctx.set_source(src_pattern)
            ctx.paint()
            ctx.set_operator(cairo.OPERATOR_OVER)

            result = ctx.pop_group()
            return result

        except Exception:
            return None

    def use_group(self, ref: Any, transform: Any = None) -> None:
        """Draw a previously defined group, optionally with a transform.

        Port of R ``.useGroup(ref, transform)`` (group.R:269,345).

        Parameters
        ----------
        ref : cairo.Pattern
            The group pattern returned by :meth:`define_group`.
        transform : ndarray or None
            A 3x3 affine transformation matrix.  When ``None``, the
            group is drawn as-is.
        """
        if ref is None:
            return

        ctx = self._ctx
        ctx.save()

        if transform is not None:
            # Apply the 3x3 affine transform.
            # R uses row-vector convention: point @ matrix
            # Cairo uses column-vector: matrix * point
            # So we need to transpose the matrix for Cairo.
            import numpy as np
            m = np.asarray(transform, dtype=float)
            # Cairo Matrix(xx, yx, xy, yy, x0, y0) = column-major
            # From row-vector convention: m[0,0]=xx, m[1,0]=yx, m[0,1]=xy,
            # m[1,1]=yy, m[2,0]=x0, m[2,1]=y0
            cairo_matrix = cairo.Matrix(
                m[0, 0], m[1, 0],  # xx, yx
                m[0, 1], m[1, 1],  # xy, yy
                m[2, 0], m[2, 1],  # x0, y0
            )
            ctx.transform(cairo_matrix)

        ctx.set_source(ref)
        ctx.paint()
        ctx.restore()
