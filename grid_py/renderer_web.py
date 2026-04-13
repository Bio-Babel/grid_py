"""WebRenderer — scene-graph renderer that outputs interactive HTML.

Builds a JSON-serialisable scene graph during ``draw_*()`` calls.  The
browser-side runtime (``gridpy.js``) reads the scene graph and renders
it to layered SVG + Canvas with D3.js-powered interactions.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._font_metrics import get_font_backend, FontMetricsBackend
from ._gpar import Gpar
from ._patterns import LinearGradient, RadialGradient, Pattern
from ._renderer_base import GridRenderer
from ._scene_graph import (
    DefsCollection,
    GrobNode,
    SceneGraph,
    SceneNode,
    ViewportNode,
    _IdGenerator,
)

__all__ = ["WebRenderer"]

# ---------------------------------------------------------------------------
# Colour / Gpar serialisation helpers
# ---------------------------------------------------------------------------

_NAMED_COLOURS = {
    "transparent", "na", "none", "NA", "",
}


def _parse_colour_str(c: Any) -> Optional[str]:
    """Normalise a colour value to a CSS-compatible string, or ``None``."""
    if c is None:
        return None
    if isinstance(c, (list, tuple)):
        if len(c) >= 4:
            return f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]:.3f})"
        if len(c) == 3:
            return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
        c = c[0]
    if isinstance(c, str):
        s = c.strip().lower()
        if s in _NAMED_COLOURS:
            return "transparent"
        return c
    if isinstance(c, (int, float)):
        v = int(float(c) * 255)
        return f"rgb({v},{v},{v})"
    return str(c)


def _serialise_gpar(gp: Optional[Gpar], defs: DefsCollection, id_gen: _IdGenerator) -> dict:
    """Convert Gpar to a JSON-safe dict.  Gradient/pattern fills become def refs."""
    if gp is None:
        return {}
    result: Dict[str, Any] = {}
    for key in ("col", "fill", "lwd", "lty", "lineend", "linejoin",
                "fontsize", "fontfamily", "fontface", "alpha"):
        val = gp.get(key, None)
        if val is None:
            continue
        # Unwrap single-element lists/arrays to scalar
        if isinstance(val, (list, tuple, np.ndarray)) and not isinstance(val, str):
            if len(val) == 0:
                continue
            if len(val) == 1:
                val = val[0]
            else:
                # Multi-element: keep as list for per-element rendering
                if key in ("col", "fill"):
                    result[key] = [_parse_colour_str(v) for v in val]
                elif key in ("lwd", "fontsize", "alpha"):
                    result[key] = [float(v) for v in val]
                else:
                    result[key] = [str(v) for v in val]
                continue
        if key in ("col", "fill"):
            if isinstance(val, LinearGradient):
                grad_id = _register_gradient(val, defs, id_gen)
                result[key] = f"url(#{grad_id})"
                continue
            if isinstance(val, RadialGradient):
                grad_id = _register_gradient(val, defs, id_gen)
                result[key] = f"url(#{grad_id})"
                continue
            if isinstance(val, Pattern):
                pat_id = _register_pattern(val, defs, id_gen)
                result[key] = f"url(#{pat_id})"
                continue
            result[key] = _parse_colour_str(val)
        elif key in ("lwd", "fontsize", "alpha"):
            result[key] = float(val)
        else:
            result[key] = str(val) if not isinstance(val, (int, float)) else val
    return result


def _register_gradient(grad: Any, defs: DefsCollection, id_gen: _IdGenerator) -> str:
    grad_id = id_gen.next("grad")
    entry: Dict[str, Any] = {"id": grad_id}
    if isinstance(grad, LinearGradient):
        entry["type"] = "linear"
        entry["x1"] = float(grad.x1.values[0])
        entry["y1"] = float(grad.y1.values[0])
        entry["x2"] = float(grad.x2.values[0])
        entry["y2"] = float(grad.y2.values[0])
    else:
        entry["type"] = "radial"
        entry["cx1"] = float(grad.cx1.values[0])
        entry["cy1"] = float(grad.cy1.values[0])
        entry["r1"] = float(grad.r1.values[0])
        entry["cx2"] = float(grad.cx2.values[0])
        entry["cy2"] = float(grad.cy2.values[0])
        entry["r2"] = float(grad.r2.values[0])
    entry["colours"] = list(grad.colours)
    entry["stops"] = [float(s) for s in grad.stops]
    entry["extend"] = getattr(grad, "extend", "pad")
    defs.gradients.append(entry)
    return grad_id


def _register_pattern(pat: Any, defs: DefsCollection, id_gen: _IdGenerator) -> str:
    pat_id = id_gen.next("pat")
    entry: Dict[str, Any] = {
        "id": pat_id,
        "width": float(pat.width.values[0]) if hasattr(pat.width, "values") else float(pat.width),
        "height": float(pat.height.values[0]) if hasattr(pat.height, "values") else float(pat.height),
    }
    defs.patterns.append(entry)
    return pat_id


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_D3_CDN_URL = "https://d3js.org/d3.v7.min.js"

_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")


def _load_resource(name: str) -> str:
    path = os.path.join(_RESOURCES_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
<div id="gridpy-plot"></div>
<script src="{d3_src}"></script>
<script>
{runtime_js}
</script>
<script>
gridpy.render(
    document.getElementById("gridpy-plot"),
    {scene_json},
    {{interactive: {interactive}, theme: "{theme}"}}
);
</script>
</body>
</html>
"""


def _array_to_data_uri(image: Any) -> str:
    """Convert a numpy image array to a base64 data URI (PNG)."""
    img_array = np.asarray(image)
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    try:
        from PIL import Image
        if img_array.ndim == 2:
            pil_img = Image.fromarray(img_array, mode="L")
        elif img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, mode="RGB")
        else:
            pil_img = Image.fromarray(img_array, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except ImportError:
        # Fallback: manually encode a minimal valid PNG from raw RGBA data
        import struct, zlib
        if img_array.ndim == 2:
            h_img, w_img = img_array.shape
            rgba = np.stack([img_array]*3 + [np.full_like(img_array, 255)], axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            h_img, w_img = img_array.shape[:2]
            rgba = np.concatenate(
                [img_array, np.full((*img_array.shape[:2], 1), 255, dtype=np.uint8)],
                axis=2)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            h_img, w_img = img_array.shape[:2]
            rgba = img_array
        else:
            return "data:image/png;base64,"
        # Build minimal PNG (uncompressed)
        def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", w_img, h_img, 8, 6, 0, 0, 0)
        raw_rows = b""
        for row in range(h_img):
            raw_rows += b"\x00" + rgba[row].tobytes()
        idat = zlib.compress(raw_rows)
        png_bytes = sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# WebRenderer
# ---------------------------------------------------------------------------


class WebRenderer(GridRenderer):
    """Scene-graph renderer producing interactive HTML output.

    Parameters
    ----------
    width : float
        Device width in inches.
    height : float
        Device height in inches.
    dpi : float
        Dots per inch (default 150).
    default_hint : str
        Default ``render_hint`` for grob nodes (``"auto"`` | ``"svg"`` | ``"canvas"``).
    theme : str
        CSS theme class (``"light"`` or ``"dark"``).
    """

    def __init__(
        self,
        width: float = 7.0,
        height: float = 5.0,
        dpi: float = 150.0,
        default_hint: str = "auto",
        theme: str = "light",
    ) -> None:
        device_w = width * dpi
        device_h = height * dpi
        super().__init__(width, height, dpi,
                         device_width=device_w, device_height=device_h)
        self._default_hint = default_hint
        self._theme = theme

        # Font metrics backend
        self._font_backend: FontMetricsBackend = get_font_backend()

        # Scene graph
        self._id_gen = _IdGenerator()
        self._defs = DefsCollection()
        self._scene_root = ViewportNode(
            node_id=self._id_gen.next("vp"),
            node_type="viewport",
            name="ROOT",
            transform={"x0": 0.0, "y0": 0.0, "w": device_w, "h": device_h},
            clip=False,
        )
        self._node_stack: List[SceneNode] = [self._scene_root]

        # Path collection buffer
        self._path_buffer: List[SceneNode] = []

    @property
    def _current_parent(self) -> SceneNode:
        return self._node_stack[-1]

    # ===================================================================== #
    # Viewport management (augments base class)                             #
    # ===================================================================== #

    def push_viewport(self, vp: Any) -> None:
        super().push_viewport(vp)
        x0, y0, pw, ph = self.get_viewport_bounds()
        clip_active = bool(self._clip_stack and self._clip_stack[-1])

        clip_id: Optional[str] = None
        if clip_active:
            clip_id = self._id_gen.next("clip")
            self._defs.clip_paths.append({
                "id": clip_id,
                "x": x0, "y": y0, "w": pw, "h": ph,
            })

        vp_node = ViewportNode(
            node_id=self._id_gen.next("vp"),
            node_type="viewport",
            name=getattr(vp, "name", "") or getattr(vp, "_name", "") or "",
            transform={"x0": x0, "y0": y0, "w": pw, "h": ph},
            clip=clip_active,
            clip_id=clip_id,
        )
        self._current_parent.children.append(vp_node)
        self._node_stack.append(vp_node)

    def pop_viewport(self) -> None:
        super().pop_viewport()
        if len(self._node_stack) > 1:
            self._node_stack.pop()

    # ===================================================================== #
    # Clipping (abstract implementations)                                   #
    # ===================================================================== #

    def _apply_clip_rect(self, x0: float, y0: float, w: float, h: float) -> None:
        pass  # Clip is recorded via ViewportNode.clip_id in push_viewport

    def _restore_clip(self) -> None:
        pass  # SVG clip-path / Canvas ctx.restore handled by JS runtime

    # ===================================================================== #
    # State save/restore                                                    #
    # ===================================================================== #

    def save_state(self) -> None:
        pass  # No graphics state to save in scene graph mode

    def restore_state(self) -> None:
        pass

    # ===================================================================== #
    # Path collection                                                       #
    # ===================================================================== #

    def begin_path_collect(self, rule: str = "winding") -> None:
        self._path_collecting = True
        self._path_buffer = []
        self._path_rule = rule

    def end_path_stroke(self, gp: Optional[Any] = None) -> None:
        self._path_collecting = False
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="compound_stroke",
            props={"rule": getattr(self, "_path_rule", "winding"),
                   "sub_paths": [n.to_dict() for n in self._path_buffer]},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint="svg",
        )
        self._current_parent.children.append(node)
        self._path_buffer = []

    def end_path_fill(self, gp: Optional[Any] = None) -> None:
        self._path_collecting = False
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="compound_fill",
            props={"rule": getattr(self, "_path_rule", "winding"),
                   "sub_paths": [n.to_dict() for n in self._path_buffer]},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint="svg",
        )
        self._current_parent.children.append(node)
        self._path_buffer = []

    def end_path_fill_stroke(self, gp: Optional[Any] = None) -> None:
        self._path_collecting = False
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="compound_fill_stroke",
            props={"rule": getattr(self, "_path_rule", "winding"),
                   "sub_paths": [n.to_dict() for n in self._path_buffer]},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint="svg",
        )
        self._current_parent.children.append(node)
        self._path_buffer = []

    # ===================================================================== #
    # Drawing primitives — append to scene graph                            #
    # ===================================================================== #

    def _append_node(self, node: SceneNode) -> None:
        # Attach grob metadata if available (for tooltip data)
        if self._current_grob_metadata and isinstance(node, GrobNode):
            node.metadata = dict(self._current_grob_metadata)
        if self._path_collecting:
            self._path_buffer.append(node)
        else:
            self._current_parent.children.append(node)

    def draw_rect(self, x: float, y: float, w: float, h: float,
                  hjust: float = 0.5, vjust: float = 0.5,
                  gp: Optional[Any] = None) -> None:
        x0 = x - w * hjust
        y0 = y - h * vjust
        dx = self._x(x0)
        dy = self._y(y0 + h)
        dw = self._sx(w)
        dh = self._sy(h)
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="rect",
            props={"x": dx, "y": dy, "w": dw, "h": dh},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_circle(self, x: float, y: float, r: float,
                    gp: Optional[Any] = None) -> None:
        cx = self._x(x)
        cy = self._y(y)
        dr = (self._sx(r) + self._sy(r)) / 2.0
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="circle",
            props={"x": cx, "y": cy, "r": dr},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_line(self, x: "np.ndarray", y: "np.ndarray",
                  gp: Optional[Any] = None) -> None:
        n = max(len(x), len(y))
        if n < 2:
            return
        if len(x) < n:
            x = np.resize(x, n)
        if len(y) < n:
            y = np.resize(y, n)
        dx = [self._x(float(v)) for v in x]
        dy = [self._y(float(v)) for v in y]
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="polyline",
            props={"x": dx, "y": dy},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_polyline(self, x: "np.ndarray", y: "np.ndarray",
                      id_: Optional["np.ndarray"] = None,
                      gp: Optional[Any] = None) -> None:
        if id_ is None:
            self.draw_line(x, y, gp)
            return
        groups = []
        for uid in np.unique(id_):
            mask = id_ == uid
            px = x[mask]
            py = y[mask]
            if len(px) < 2:
                continue
            groups.append({
                "x": [self._x(float(v)) for v in px],
                "y": [self._y(float(v)) for v in py],
            })
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="polyline",
            props={"groups": groups},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_segments(self, x0: "np.ndarray", y0: "np.ndarray",
                      x1: "np.ndarray", y1: "np.ndarray",
                      gp: Optional[Any] = None) -> None:
        n = min(len(x0), len(y0), len(x1), len(y1))
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="segments",
            props={
                "x0": [self._x(float(x0[i])) for i in range(n)],
                "y0": [self._y(float(y0[i])) for i in range(n)],
                "x1": [self._x(float(x1[i])) for i in range(n)],
                "y1": [self._y(float(y1[i])) for i in range(n)],
            },
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_polygon(self, x: "np.ndarray", y: "np.ndarray",
                     gp: Optional[Any] = None) -> None:
        if len(x) < 3:
            return
        dx = [self._x(float(v)) for v in x]
        dy = [self._y(float(v)) for v in y]
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="polygon",
            props={"x": dx, "y": dy},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_path(self, x: "np.ndarray", y: "np.ndarray",
                  path_id: "np.ndarray", rule: str = "winding",
                  gp: Optional[Any] = None) -> None:
        groups = []
        for pid in np.unique(path_id):
            mask = path_id == pid
            px = x[mask]
            py = y[mask]
            if len(px) < 2:
                continue
            groups.append({
                "x": [self._x(float(v)) for v in px],
                "y": [self._y(float(v)) for v in py],
            })
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="path",
            props={"groups": groups, "rule": rule},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_text(self, x: float, y: float, label: str,
                  rot: float = 0.0, hjust: float = 0.5, vjust: float = 0.5,
                  gp: Optional[Any] = None) -> None:
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="text",
            props={
                "x": self._x(x), "y": self._y(y),
                "label": str(label),
                "rot": rot, "hjust": hjust, "vjust": vjust,
            },
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint="svg",  # Text always SVG
        )
        self._append_node(node)

    def draw_points(self, x: "np.ndarray", y: "np.ndarray",
                    size: float = 1.0, pch: Any = 19,
                    gp: Optional[Any] = None) -> None:
        dx = [self._x(float(v)) for v in x]
        dy = [self._y(float(v)) for v in y]
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="points",
            props={
                "x": dx, "y": dy,
                "size": float(size),
                "pch": int(pch) if not isinstance(pch, (list, tuple, np.ndarray)) else [int(p) for p in pch],
            },
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def draw_raster(self, image: Any, x: float, y: float,
                    w: float, h: float, interpolate: bool = True) -> None:
        data_uri = _array_to_data_uri(image)
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="raster",
            props={
                "x": self._x(x), "y": self._y(y + h),
                "w": self._sx(w), "h": self._sy(h),
                "src": data_uri,
                "interpolate": interpolate,
            },
            gpar={},
            render_hint="svg",
        )
        self._append_node(node)

    def draw_roundrect(self, x: float, y: float, w: float, h: float,
                       r: float = 0.0, hjust: float = 0.5, vjust: float = 0.5,
                       gp: Optional[Any] = None) -> None:
        x0 = x - w * hjust
        y0 = y - h * vjust
        dx = self._x(x0)
        dy = self._y(y0 + h)
        dw = self._sx(w)
        dh = self._sy(h)
        dr = min(self._sx(r), dw / 2, dh / 2) if r > 0 else 0.0
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="roundrect",
            props={"x": dx, "y": dy, "w": dw, "h": dh, "r": dr},
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)

    def move_to(self, x: float, y: float) -> None:
        self._pen_x = x
        self._pen_y = y

    def line_to(self, x: float, y: float,
                gp: Optional[Any] = None) -> None:
        x0 = getattr(self, "_pen_x", 0.0)
        y0 = getattr(self, "_pen_y", 0.0)
        node = GrobNode(
            node_id=self._id_gen.next("grob"),
            node_type="polyline",
            props={
                "x": [self._x(x0), self._x(x)],
                "y": [self._y(y0), self._y(y)],
            },
            gpar=_serialise_gpar(gp, self._defs, self._id_gen),
            render_hint=self._default_hint,
        )
        self._append_node(node)
        self._pen_x = x
        self._pen_y = y

    # ===================================================================== #
    # Clipping (explicit push/pop)                                          #
    # ===================================================================== #

    def push_clip(self, x0: float, y0: float, x1: float, y1: float) -> None:
        clip_id = self._id_gen.next("clip")
        dx0 = self._x(min(x0, x1))
        dy0 = self._y(max(y0, y1))
        dw = self._sx(abs(x1 - x0))
        dh = self._sy(abs(y1 - y0))
        self._defs.clip_paths.append({
            "id": clip_id, "x": dx0, "y": dy0, "w": dw, "h": dh,
        })
        # Wrap subsequent children in a clipped group
        clip_group = ViewportNode(
            node_id=self._id_gen.next("vp"),
            node_type="viewport",
            name="__clip__",
            transform=self._current_parent.transform
                if isinstance(self._current_parent, ViewportNode) else {},
            clip=True,
            clip_id=clip_id,
        )
        self._current_parent.children.append(clip_group)
        self._node_stack.append(clip_group)

    def pop_clip(self) -> None:
        if len(self._node_stack) > 1:
            self._node_stack.pop()

    # ===================================================================== #
    # Text metrics                                                          #
    # ===================================================================== #

    def text_extents(self, text: str, gp: Optional[Any] = None) -> Dict[str, float]:
        return self._font_backend.measure(text, gp)

    # ===================================================================== #
    # Masking                                                               #
    # ===================================================================== #

    def render_mask(self, mask_grob: Any) -> Any:
        mask_id = self._id_gen.next("mask")
        # Render mask grob into a sub scene graph
        sub = WebRenderer(
            width=self.width_in, height=self.height_in,
            dpi=self.dpi, default_hint="svg",
        )
        try:
            from ._draw import grid_draw
            from ._state import get_state
            state = get_state()
            orig = state._renderer
            state._renderer = sub
            try:
                grid_draw(mask_grob, recording=False)
            finally:
                state._renderer = orig
        except Exception:
            pass
        self._defs.masks.append({
            "id": mask_id,
            "content": sub._scene_root.to_dict(),
        })
        return mask_id

    def apply_mask(self, mask_surface: Any, mask_type: str = "alpha") -> None:
        if isinstance(mask_surface, str):
            # mask_surface is actually a mask_id
            if isinstance(self._current_parent, ViewportNode):
                self._current_parent.mask_id = mask_surface
                self._current_parent.mask_type = mask_type

    # ===================================================================== #
    # Output                                                                #
    # ===================================================================== #

    def to_scene_dict(self) -> dict:
        """Return the scene graph as a plain dict (JSON-serialisable)."""
        return SceneGraph(
            width=self.width_in * self.dpi,
            height=self.height_in * self.dpi,
            dpi=self.dpi,
            root=self._scene_root,
            defs=self._defs,
        ).to_dict()

    def to_scene_json(self, indent: Optional[int] = None) -> str:
        """Return the scene graph as a JSON string."""
        return json.dumps(self.to_scene_dict(), indent=indent)

    def to_html(self, interactive: bool = True, cdn: bool = True) -> str:
        """Generate a self-contained HTML file."""
        scene_json = self.to_scene_json()
        css = _load_resource("gridpy.css")
        runtime_js = _load_resource("gridpy.js")
        d3_src = _D3_CDN_URL if cdn else ""
        return _HTML_TEMPLATE.format(
            css=css,
            d3_src=d3_src,
            runtime_js=runtime_js,
            scene_json=scene_json,
            interactive="true" if interactive else "false",
            theme=self._theme,
        )

    def _repr_html_(self) -> str:
        """Jupyter notebook display integration."""
        return self.to_html(interactive=True, cdn=True)

    def save(self, filename: str) -> None:
        """Save to an HTML file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.to_html())

    def new_page(self, bg: Any = "white") -> None:
        """Reset the scene graph for a new page."""
        self._id_gen.reset()
        self._defs = DefsCollection()
        dw = self.width_in * self.dpi
        dh = self.height_in * self.dpi
        self._scene_root = ViewportNode(
            node_id=self._id_gen.next("vp"),
            node_type="viewport",
            name="ROOT",
            transform={"x0": 0.0, "y0": 0.0, "w": dw, "h": dh},
            clip=False,
        )
        self._node_stack = [self._scene_root]
        # Re-init base class viewport stack
        self._vp_stack = [(0.0, 0.0, dw, dh, None)]
        self._layout_stack = []
        self._layout_depth_stack = []
        self._clip_stack = []
        self._path_collecting = False

    def finish(self) -> None:
        pass  # No resources to release

    def get_surface(self) -> Any:
        """WebRenderer has no Cairo surface; return ``None``."""
        return None

    def write_to_png(self, filename: str) -> None:
        """Not supported — WebRenderer produces HTML, not PNG.

        Use :meth:`save` for HTML output.
        """
        raise NotImplementedError(
            "WebRenderer produces HTML, not PNG. Use save() or to_html()."
        )

    def to_png_bytes(self) -> bytes:
        """Not supported — WebRenderer produces HTML, not PNG.

        Use :meth:`to_scene_json` or :meth:`to_html`.
        """
        raise NotImplementedError(
            "WebRenderer produces HTML, not PNG. Use to_scene_json() or to_html()."
        )
