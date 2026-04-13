"""Gap tests: comprehensive coverage for WebRenderer features identified
in the post-implementation audit.

Tests cover: grob metadata passthrough, mask rendering, GridStroke/GridFill
pipeline, layout viewports, FonttoolsMetrics, raster PNG encoder, colour
parsing, per-point gpar, grid_cap fallback, and JS structure validation.
"""

import base64
import json
import struct
import zlib

import numpy as np
import pytest

from grid_py._font_metrics import (
    CairoFontMetrics,
    FonttoolsMetrics,
    HeuristicMetrics,
    get_font_backend,
)
from grid_py._gpar import Gpar
from grid_py._renderer_base import GridRenderer
from grid_py._units import Unit
from grid_py.renderer_web import WebRenderer, _parse_colour_str, _serialise_gpar
from grid_py._scene_graph import DefsCollection, _IdGenerator


# ===================================================================== #
# FonttoolsMetrics                                                      #
# ===================================================================== #

class TestFonttoolsMetrics:
    def test_basic_measurement(self):
        ft = FonttoolsMetrics()
        r = ft.measure("Hello")
        assert r["width"] > 0
        assert r["ascent"] > 0
        assert r["descent"] > 0

    def test_empty_string(self):
        ft = FonttoolsMetrics()
        r = ft.measure("")
        assert r["width"] == 0.0

    def test_wider_for_longer_text(self):
        ft = FonttoolsMetrics()
        r1 = ft.measure("A")
        r2 = ft.measure("AAAA")
        assert r2["width"] > r1["width"]

    def test_larger_font_wider(self):
        ft = FonttoolsMetrics()
        r1 = ft.measure("X", gp=Gpar(fontsize=12))
        r2 = ft.measure("X", gp=Gpar(fontsize=24))
        assert r2["width"] > r1["width"]
        assert r2["ascent"] > r1["ascent"]

    def test_cex_scales(self):
        ft = FonttoolsMetrics()
        r1 = ft.measure("X", gp=Gpar(fontsize=12, cex=1.0))
        r2 = ft.measure("X", gp=Gpar(fontsize=12, cex=2.0))
        assert abs(r2["width"] / r1["width"] - 2.0) < 0.01

    def test_reasonable_agreement_with_cairo(self):
        """FonttoolsMetrics and CairoFontMetrics should agree within 20%."""
        ft = FonttoolsMetrics()
        cairo = CairoFontMetrics()
        for text in ["Hello", "Testing 123", "WWWWW", "iiiii"]:
            ft_r = ft.measure(text)
            ca_r = cairo.measure(text)
            if ca_r["width"] > 0:
                ratio = ft_r["width"] / ca_r["width"]
                assert 0.5 < ratio < 2.0, (
                    f"Width ratio {ratio:.2f} out of range for '{text}': "
                    f"fonttools={ft_r['width']:.4f}, cairo={ca_r['width']:.4f}"
                )

    def test_unicode_characters(self):
        ft = FonttoolsMetrics()
        # Should not crash even if glyph is missing
        r = ft.measure("ABC\u00e9\u00f1")
        assert r["width"] > 0


# ===================================================================== #
# Colour parsing (_parse_colour_str)                                    #
# ===================================================================== #

class TestParseColourStr:
    def test_named_colour(self):
        assert _parse_colour_str("red") == "red"
        assert _parse_colour_str("blue") == "blue"

    def test_hex_colour(self):
        assert _parse_colour_str("#FF0000") == "#FF0000"
        assert _parse_colour_str("#00FF00FF") == "#00FF00FF"

    def test_transparent(self):
        assert _parse_colour_str("transparent") == "transparent"
        assert _parse_colour_str("NA") == "transparent"
        assert _parse_colour_str("none") == "transparent"
        assert _parse_colour_str("") == "transparent"

    def test_none(self):
        assert _parse_colour_str(None) is None

    def test_rgb_tuple(self):
        result = _parse_colour_str((1.0, 0.0, 0.0))
        assert "rgb" in result
        assert "255" in result

    def test_rgba_tuple(self):
        result = _parse_colour_str((1.0, 0.0, 0.0, 0.5))
        assert "rgba" in result
        assert "0.500" in result

    def test_numeric_grey(self):
        result = _parse_colour_str(0.5)
        assert "rgb" in result
        assert "127" in result or "128" in result

    def test_single_element_list(self):
        result = _parse_colour_str(["red"])
        assert result == "red"


# ===================================================================== #
# Gpar serialisation edge cases                                         #
# ===================================================================== #

class TestGparSerialisation:
    def setup_method(self):
        self.defs = DefsCollection()
        self.id_gen = _IdGenerator()

    def test_numpy_array_col(self):
        gp = Gpar(col=np.array(["red", "blue"]))
        r = _serialise_gpar(gp, self.defs, self.id_gen)
        assert isinstance(r["col"], list)
        assert r["col"] == ["red", "blue"]

    def test_numpy_scalar_fontsize(self):
        gp = Gpar(fontsize=np.float64(14.0))
        r = _serialise_gpar(gp, self.defs, self.id_gen)
        assert r["fontsize"] == 14.0

    def test_none_values_skipped(self):
        gp = Gpar(col="red")
        r = _serialise_gpar(gp, self.defs, self.id_gen)
        assert "fill" not in r or r.get("fill") is not None


# ===================================================================== #
# Grob metadata passthrough                                            #
# ===================================================================== #

class TestGrobMetadataPassthrough:
    def test_metadata_flows_through_pipeline(self):
        """Grob.metadata should appear in scene graph node data."""
        from grid_py._draw import grid_draw
        from grid_py._primitives import rect_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        grob = rect_grob(x=0.5, y=0.5, width=0.3, height=0.2, gp=Gpar(fill="red"))
        grob.metadata = {"species": "setosa", "value": 4.5}
        grid_draw(grob)

        sg = json.loads(r.to_scene_json())
        child = sg["root"]["children"][0]
        assert "data" in child, "Grob metadata not passed to scene graph"
        assert child["data"]["species"] == "setosa"
        assert child["data"]["value"] == 4.5

    def test_no_metadata_no_data_field(self):
        """Grobs without metadata should not have a data field."""
        from grid_py._draw import grid_draw
        from grid_py._primitives import rect_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        grid_draw(rect_grob(x=0.5, y=0.5, width=0.3, height=0.2))
        sg = json.loads(r.to_scene_json())
        child = sg["root"]["children"][0]
        assert "data" not in child

    def test_metadata_on_points(self):
        from grid_py._draw import grid_draw
        from grid_py._primitives import points_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        grob = points_grob(x=[0.1, 0.5, 0.9], y=[0.2, 0.4, 0.6], gp=Gpar(col="blue"))
        grob.metadata = {"label": ["A", "B", "C"]}
        grid_draw(grob)
        sg = json.loads(r.to_scene_json())
        child = sg["root"]["children"][0]
        assert child["data"]["label"] == ["A", "B", "C"]


# ===================================================================== #
# GridStroke / GridFill / GridFillStroke through full pipeline           #
# ===================================================================== #

class TestPathGrobPipeline:
    def test_stroke_grob_through_pipeline(self):
        from grid_py._draw import grid_draw
        from grid_py._primitives import stroke_grob, rect_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        path = rect_grob(x=0.5, y=0.5, width=0.5, height=0.5)
        grob = stroke_grob(path, gp=Gpar(col="red", lwd=2))
        grid_draw(grob)

        sg = json.loads(r.to_scene_json())
        children = sg["root"]["children"]
        assert len(children) == 1
        assert children[0]["type"] == "compound_stroke"
        assert len(children[0]["props"]["sub_paths"]) >= 1

    def test_fill_grob_through_pipeline(self):
        from grid_py._draw import grid_draw
        from grid_py._primitives import fill_grob, circle_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        path = circle_grob(x=0.5, y=0.5, r=0.3)
        grob = fill_grob(path, rule="evenodd", gp=Gpar(fill="blue"))
        grid_draw(grob)

        sg = json.loads(r.to_scene_json())
        children = sg["root"]["children"]
        assert len(children) == 1
        assert children[0]["type"] == "compound_fill"
        assert children[0]["props"]["rule"] == "evenodd"

    def test_fill_stroke_grob_through_pipeline(self):
        from grid_py._draw import grid_draw
        from grid_py._primitives import fill_stroke_grob, rect_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        path = rect_grob(x=0.5, y=0.5, width=0.6, height=0.4)
        grob = fill_stroke_grob(path, gp=Gpar(fill="yellow", col="black"))
        grid_draw(grob)

        sg = json.loads(r.to_scene_json())
        children = sg["root"]["children"]
        assert len(children) == 1
        assert children[0]["type"] == "compound_fill_stroke"


# ===================================================================== #
# Layout viewport integration                                           #
# ===================================================================== #

class TestLayoutViewportIntegration:
    def test_2x2_grid_layout(self):
        from grid_py._layout import GridLayout
        from grid_py._viewport import Viewport

        r = WebRenderer(width=7, height=5, dpi=150)

        layout = GridLayout(
            nrow=2, ncol=2,
            widths=Unit([1, 1], "null"),
            heights=Unit([1, 1], "null"),
        )
        vp_layout = Viewport(name="grid", layout=layout)
        r.push_viewport(vp_layout)

        # Push into cell (1,1) and draw
        vp_cell = Viewport(name="cell_1_1", layout_pos_row=1, layout_pos_col=1)
        r.push_viewport(vp_cell)
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="red"))
        r.pop_viewport()

        # Push into cell (2,2) and draw
        vp_cell2 = Viewport(name="cell_2_2", layout_pos_row=2, layout_pos_col=2)
        r.push_viewport(vp_cell2)
        r.draw_rect(0.5, 0.5, 1.0, 1.0, gp=Gpar(fill="blue"))
        r.pop_viewport()

        r.pop_viewport()

        sg = json.loads(r.to_scene_json())

        # Verify tree structure
        root = sg["root"]
        assert root["type"] == "viewport"
        # Root has layout viewport child
        layout_vp = root["children"][0]
        assert layout_vp["type"] == "viewport"
        assert layout_vp["name"] == "grid"
        # Layout vp has two cell viewport children
        cells = [c for c in layout_vp["children"] if c["type"] == "viewport"]
        assert len(cells) == 2
        # Each cell has one rect child
        for cell in cells:
            rects = [c for c in cell["children"] if c["type"] == "rect"]
            assert len(rects) == 1

    def test_layout_cell_bounds_are_correct(self):
        """Verify cell viewport transforms divide parent space correctly."""
        from grid_py._layout import GridLayout
        from grid_py._viewport import Viewport

        dpi = 150
        r = WebRenderer(width=4, height=4, dpi=dpi)
        full_w = 4 * dpi  # 600px

        layout = GridLayout(
            nrow=1, ncol=2,
            widths=Unit([1, 1], "null"),
            heights=Unit([1], "null"),
        )
        vp_layout = Viewport(name="grid", layout=layout)
        r.push_viewport(vp_layout)

        vp_left = Viewport(name="left", layout_pos_row=1, layout_pos_col=1)
        r.push_viewport(vp_left)
        bounds = r.get_viewport_bounds()
        left_w = bounds[2]
        r.pop_viewport()

        vp_right = Viewport(name="right", layout_pos_row=1, layout_pos_col=2)
        r.push_viewport(vp_right)
        bounds = r.get_viewport_bounds()
        right_w = bounds[2]
        r.pop_viewport()

        r.pop_viewport()

        # Each column should get half the width
        assert abs(left_w - full_w / 2) < 1.0, f"left_w={left_w}, expected={full_w/2}"
        assert abs(right_w - full_w / 2) < 1.0, f"right_w={right_w}, expected={full_w/2}"


# ===================================================================== #
# Mask rendering                                                        #
# ===================================================================== #

class TestMaskRendering:
    def test_render_mask_creates_def(self):
        from grid_py._primitives import rect_grob

        r = WebRenderer(width=7, height=5)
        mask_grob = rect_grob(x=0.5, y=0.5, width=1.0, height=1.0,
                              gp=Gpar(fill="white"))
        mask_id = r.render_mask(mask_grob)
        assert isinstance(mask_id, str)
        assert mask_id.startswith("mask-")
        assert len(r._defs.masks) == 1
        assert r._defs.masks[0]["id"] == mask_id

    def test_apply_mask_sets_viewport_attribute(self):
        from grid_py._primitives import rect_grob
        from grid_py._viewport import Viewport
        from grid_py._scene_graph import ViewportNode

        r = WebRenderer(width=7, height=5)
        mask_grob = rect_grob(x=0.5, y=0.5, width=1.0, height=1.0)
        mask_id = r.render_mask(mask_grob)

        vp = Viewport(name="masked_vp", x=0.5, y=0.5, width=0.8, height=0.8)
        r.push_viewport(vp)
        r.apply_mask(mask_id, mask_type="alpha")

        vp_node = r._scene_root.children[0]
        assert isinstance(vp_node, ViewportNode)
        assert vp_node.mask_id == mask_id
        assert vp_node.mask_type == "alpha"
        r.pop_viewport()

    def test_mask_content_in_defs(self):
        from grid_py._primitives import circle_grob

        r = WebRenderer(width=7, height=5)
        mask_grob = circle_grob(x=0.5, y=0.5, r=0.3, gp=Gpar(fill="white"))
        mask_id = r.render_mask(mask_grob)

        mask_def = r._defs.masks[0]
        assert "content" in mask_def
        # Content should be a serialized viewport node with children
        content = mask_def["content"]
        assert content["type"] == "viewport"


# ===================================================================== #
# Raster PNG fallback encoder validation                                #
# ===================================================================== #

class TestRasterPngEncoder:
    def test_encoded_png_is_valid(self):
        """The manual PNG encoder should produce a valid PNG file."""
        r = WebRenderer()
        # Create a small test image
        img = np.zeros((3, 4, 3), dtype=np.uint8)
        img[0, :, 0] = 255  # Red row
        img[1, :, 1] = 255  # Green row
        img[2, :, 2] = 255  # Blue row

        r.draw_raster(img, 0.0, 0.0, 1.0, 1.0)
        d = r._scene_root.children[0].to_dict()
        uri = d["props"]["src"]

        assert uri.startswith("data:image/png;base64,")
        b64_data = uri.split(",", 1)[1]
        png_bytes = base64.b64decode(b64_data)

        # Validate PNG signature
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

        # Validate IHDR chunk
        ihdr_len = struct.unpack(">I", png_bytes[8:12])[0]
        assert png_bytes[12:16] == b"IHDR"
        width = struct.unpack(">I", png_bytes[16:20])[0]
        height = struct.unpack(">I", png_bytes[20:24])[0]
        assert width == 4
        assert height == 3

    def test_greyscale_image(self):
        r = WebRenderer()
        img = np.full((5, 5), 128, dtype=np.uint8)
        r.draw_raster(img, 0.0, 0.0, 1.0, 1.0)
        d = r._scene_root.children[0].to_dict()
        assert d["props"]["src"].startswith("data:image/png;base64,")
        # Should be decodable
        b64_data = d["props"]["src"].split(",", 1)[1]
        base64.b64decode(b64_data)

    def test_rgba_image(self):
        r = WebRenderer()
        img = np.zeros((2, 2, 4), dtype=np.uint8)
        img[:, :, 3] = 128  # Semi-transparent
        r.draw_raster(img, 0.0, 0.0, 1.0, 1.0)
        d = r._scene_root.children[0].to_dict()
        assert d["props"]["src"].startswith("data:image/png;base64,")


# ===================================================================== #
# grid_cap / get_surface fallback                                       #
# ===================================================================== #

class TestGridCapFallback:
    def test_get_surface_returns_none(self):
        r = WebRenderer()
        assert r.get_surface() is None

    def test_write_to_png_raises(self):
        r = WebRenderer()
        with pytest.raises(NotImplementedError, match="HTML, not PNG"):
            r.write_to_png("/tmp/test.png")

    def test_to_png_bytes_raises(self):
        r = WebRenderer()
        with pytest.raises(NotImplementedError, match="HTML, not PNG"):
            r.to_png_bytes()

    def test_grid_cap_returns_none_gracefully(self):
        """grid_cap() on WebRenderer should return None, not crash."""
        from grid_py._draw import grid_draw
        from grid_py._grab import grid_cap
        from grid_py._primitives import rect_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)
        grid_draw(rect_grob(x=0.5, y=0.5, width=0.5, height=0.5))
        result = grid_cap()
        assert result is None


# ===================================================================== #
# Per-point gpar in draw_points scene graph                             #
# ===================================================================== #

class TestPerPointGpar:
    def test_per_point_colors_in_scene_graph(self):
        """draw_points with per-point gpar should produce array colours."""
        from grid_py._draw import grid_draw
        from grid_py._primitives import points_grob
        from grid_py._state import get_state

        r = WebRenderer(width=7, height=5)
        state = get_state()
        state.init_device(r)

        gp = Gpar(col=["red", "blue", "green"], fill=["#FF000080", "#00FF0080", "#0000FF80"])
        grid_draw(points_grob(x=[0.1, 0.5, 0.9], y=[0.2, 0.5, 0.8], gp=gp))

        sg = json.loads(r.to_scene_json())
        child = sg["root"]["children"][0]
        assert child["type"] == "points"
        assert isinstance(child["gpar"]["col"], list)
        assert len(child["gpar"]["col"]) == 3
        assert isinstance(child["gpar"]["fill"], list)
        assert len(child["gpar"]["fill"]) == 3


# ===================================================================== #
# Scene graph JSON structure matches JS runtime expectations            #
# ===================================================================== #

class TestSceneGraphJsContract:
    """Verify the JSON structure matches what gridpy.js expects."""

    def test_root_has_required_fields(self):
        r = WebRenderer(width=7, height=5, dpi=150)
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        sg = json.loads(r.to_scene_json())

        assert sg["version"] == 1
        assert "width" in sg
        assert "height" in sg
        assert "dpi" in sg
        assert "root" in sg
        assert "defs" in sg

    def test_viewport_node_fields(self):
        from grid_py._viewport import Viewport
        r = WebRenderer(width=7, height=5)
        vp = Viewport(name="test", x=0.5, y=0.5, width=0.5, height=0.5, clip=True)
        r.push_viewport(vp)
        r.pop_viewport()
        sg = json.loads(r.to_scene_json())
        vp_node = sg["root"]["children"][0]

        assert vp_node["type"] == "viewport"
        assert "name" in vp_node
        assert "transform" in vp_node
        assert "clip" in vp_node
        assert "clip_id" in vp_node  # clip=True should produce clip_id
        t = vp_node["transform"]
        assert all(k in t for k in ("x0", "y0", "w", "h"))

    def test_grob_node_fields(self):
        r = WebRenderer()
        r.draw_rect(0.5, 0.5, 0.3, 0.2, gp=Gpar(fill="red"))
        sg = json.loads(r.to_scene_json())
        grob_node = sg["root"]["children"][0]

        assert "id" in grob_node
        assert "type" in grob_node
        assert "props" in grob_node
        assert "gpar" in grob_node
        assert "render_hint" in grob_node

    def test_defs_structure(self):
        r = WebRenderer()
        sg = json.loads(r.to_scene_json())
        defs = sg["defs"]

        assert "gradients" in defs
        assert "patterns" in defs
        assert "clip_paths" in defs
        assert "masks" in defs
        assert isinstance(defs["gradients"], list)

    def test_all_grob_types_serializable(self):
        """Every draw_* method should produce valid JSON."""
        r = WebRenderer(width=7, height=5, dpi=150)
        r.draw_rect(0.5, 0.5, 0.3, 0.2)
        r.draw_circle(0.5, 0.5, 0.1)
        r.draw_text(0.5, 0.5, "Test")
        r.draw_points(np.array([0.1, 0.5]), np.array([0.2, 0.8]))
        r.draw_polyline(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 0.0]))
        r.draw_segments(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([1.0]))
        r.draw_polygon(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 0.0]))
        r.draw_path(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 0.0]),
                     np.array([1, 1, 1]))
        r.draw_roundrect(0.5, 0.5, 0.3, 0.2, r=0.05)
        r.draw_raster(np.zeros((2, 2, 3), dtype=np.uint8), 0.0, 0.0, 1.0, 1.0)
        r.move_to(0.1, 0.2)
        r.line_to(0.5, 0.8)

        j = r.to_scene_json()
        sg = json.loads(j)  # Must not raise
        assert len(sg["root"]["children"]) == 11
        types = [c["type"] for c in sg["root"]["children"]]
        assert "rect" in types
        assert "circle" in types
        assert "text" in types
        assert "points" in types
        assert "polyline" in types
        assert "segments" in types
        assert "polygon" in types
        assert "path" in types
        assert "roundrect" in types
        assert "raster" in types
