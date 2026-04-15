"""
grid_py — Python port of the R grid graphics package.

Provides a complete reimplementation of R's grid graphics system including
units, viewports, grobs (graphical objects), layouts, and rendering via
Cairo (pycairo).
"""

__version__ = "4.5.3-R4.5.3"

# --- Utilities ---
from grid_py._utils import depth, explode, grid_pretty, n2mfrow

# --- Justification ---
from grid_py._just import valid_just, resolve_hjust, resolve_vjust, resolve_raster_size

# --- Units ---
from grid_py._units import (
    Unit, is_unit, unit_type, unit_c, unit_length,
    unit_pmax, unit_pmin, unit_psum, unit_rep,
    string_width, string_height, string_ascent, string_descent,
    absolute_size,
    convert_unit, convert_x, convert_y, convert_width, convert_height,
)

# --- Graphical Parameters ---
from grid_py._gpar import Gpar, get_gpar

# --- Arrow ---
from grid_py._arrow import Arrow, arrow

# --- Paths ---
from grid_py._path import GPath, VpPath, GridPath, as_path, is_closed, PATH_SEP

# --- Layout ---
from grid_py._layout import (
    GridLayout, layout_region,
)

# --- Patterns, Masks, Clip Paths ---
from grid_py._patterns import (
    LinearGradient, RadialGradient, Pattern,
    linear_gradient, radial_gradient, pattern,
)
from grid_py._mask import GridMask, as_mask, is_mask
from grid_py._clippath import GridClipPath, as_clip_path, is_clip_path

# --- Viewport ---
from grid_py._viewport import (
    Viewport, VpList, VpStack, VpTree,
    push_viewport, pop_viewport, down_viewport, up_viewport, seek_viewport,
    current_viewport, current_vp_path, current_vp_tree,
    current_transform, current_rotation, current_parent,
    data_viewport, plot_viewport, edit_viewport, show_viewport,
)

# --- State ---
from grid_py._state import GridState, get_state

# --- Display List ---
from grid_py._display_list import DisplayList

# --- Transforms ---
from grid_py._transforms import (
    group_translate, group_rotate, group_scale, group_shear, group_flip,
    defn_translate, defn_rotate, defn_scale,
    use_translate, use_rotate, use_scale,
    viewport_translate, viewport_rotate, viewport_scale, viewport_transform,
)

# --- Grob ---
from grid_py._grob import (
    Grob, GTree, GList, GEdit, GEditList,
    grob_tree, grob_name, is_grob,
    get_grob, set_grob, add_grob, remove_grob, edit_grob,
    force_grob, set_children, reorder_grob,
    apply_edit, apply_edits,
)

# --- Primitives ---
from grid_py._primitives import (
    move_to_grob, grid_move_to,
    line_to_grob, grid_line_to,
    lines_grob, grid_lines,
    polyline_grob, grid_polyline,
    segments_grob, grid_segments,
    arrows_grob, grid_arrows,
    points_grob, grid_points,
    rect_grob, grid_rect,
    roundrect_grob, grid_roundrect,
    circle_grob, grid_circle,
    polygon_grob, grid_polygon,
    path_grob, grid_path,
    text_grob, grid_text,
    raster_grob, grid_raster,
    clip_grob, grid_clip,
    null_grob, grid_null,
    function_grob, grid_function,
    as_path,
    stroke_grob, grid_stroke,
    fill_grob, grid_fill,
    fill_stroke_grob, grid_fill_stroke,
)

# --- Coordinates ---
from grid_py._coords import (
    GridCoords, GridGrobCoords, GridGTreeCoords,
    grob_coords, grob_points,
    grid_coords, grid_grob_coords, grid_gtree_coords,
    empty_coords, empty_grob_coords, empty_gtree_coords,
    is_empty_coords,
)

# --- Curves ---
from grid_py._curve import (
    curve_grob, grid_curve,
    xspline_grob, grid_xspline,
    bezier_grob, grid_bezier,
    xspline_points, bezier_points,
    arc_curvature,
)

# --- Groups ---
from grid_py._group import (
    GroupGrob, DefineGrob, UseGrob,
    group_grob, grid_group,
    define_grob, grid_define,
    use_grob, grid_use,
)

# --- Renderer ---
from grid_py._renderer_base import GridRenderer
from grid_py.renderer import CairoRenderer
from grid_py.renderer_web import WebRenderer

# --- Drawing ---
from grid_py._draw import (
    grid_draw, grid_newpage, grid_refresh,
    grid_record, record_grob,
    grid_delay, delay_grob,
    grid_dl_apply, grid_locator,
)

# --- Edit (display list) ---
from grid_py._edit import (
    grid_edit, grid_get, grid_set, grid_add, grid_remove,
    grid_gedit, grid_gget, grid_gremove,
)

# --- Listing & Search ---
from grid_py._ls import (
    grid_ls, grid_grep,
    nested_listing, path_listing, grob_path_listing,
    show_grob, get_names, child_names,
)

# --- Grab ---
from grid_py._grab import (
    grid_grab, grid_grab_expr,
    grid_force, grid_revert,
    grid_cap, grid_reorder,
)

# --- High-level ---
from grid_py._highlevel import (
    grid_grill, grid_show_layout, grid_show_viewport,
    grid_abline, grid_plot_and_legend, layout_torture,
    frame_grob, grid_frame, pack_grob, grid_pack, place_grob, grid_place,
    xaxis_grob, grid_xaxis, yaxis_grob, grid_yaxis,
    legend_grob, grid_legend,
    grid_multipanel, grid_panel, grid_strip,
    grid_top_level_vp,
)

# --- Size / Metrics ---
from grid_py._size import (
    calc_string_metric,
    width_details, height_details, ascent_details, descent_details,
    grob_width, grob_height, grob_x, grob_y,
    grob_ascent, grob_descent,
)

# --- Typeset ---
from grid_py._typeset import glyph_grob, grid_glyph

# --- Deprecated aliases (R compatibility) ---
convert_native = convert_unit
grid_convert = convert_unit
grid_convert_x = convert_x
grid_convert_y = convert_y
grid_convert_width = convert_width
grid_convert_height = convert_height
from grid_py._units import device_loc, device_dim

# R names grid.collection and grid.copy were undocumented stubs
grid_collection = grid_draw
grid_copy = grid_draw

# grid.display.list / engine.display.list
def grid_display_list(on: bool = True) -> bool:
    """Enable or disable the display list.

    Parameters
    ----------
    on : bool
        Whether to enable recording.

    Returns
    -------
    bool
        Previous state.
    """
    state = get_state()
    prev = state._dl_on
    state._dl_on = on
    return prev

engine_display_list = grid_display_list


__all__ = [
    # Utils
    "depth", "explode", "grid_pretty", "n2mfrow",
    # Just
    "valid_just", "resolve_hjust", "resolve_vjust", "resolve_raster_size",
    # Units
    "Unit", "is_unit", "unit_type", "unit_c", "unit_length",
    "unit_pmax", "unit_pmin", "unit_psum", "unit_rep",
    "string_width", "string_height", "string_ascent", "string_descent",
    "absolute_size",
    "convert_unit", "convert_x", "convert_y", "convert_width", "convert_height",
    # Gpar
    "Gpar", "get_gpar",
    # Arrow
    "Arrow", "arrow",
    # Path
    "GPath", "VpPath", "GridPath", "as_path", "is_closed", "PATH_SEP",
    # Layout
    "GridLayout", "layout_region",
    # Patterns
    "LinearGradient", "RadialGradient", "Pattern",
    "linear_gradient", "radial_gradient", "pattern",
    # Mask
    "GridMask", "as_mask", "is_mask",
    # Clip path
    "GridClipPath", "as_clip_path", "is_clip_path",
    # Viewport
    "Viewport", "VpList", "VpStack", "VpTree",
    "push_viewport", "pop_viewport", "down_viewport", "up_viewport", "seek_viewport",
    "current_viewport", "current_vp_path", "current_vp_tree",
    "current_transform", "current_rotation", "current_parent",
    "data_viewport", "plot_viewport", "edit_viewport", "show_viewport",
    # State
    "GridState", "get_state",
    # Display list
    "DisplayList",
    # Transforms
    "group_translate", "group_rotate", "group_scale", "group_shear", "group_flip",
    "defn_translate", "defn_rotate", "defn_scale",
    "use_translate", "use_rotate", "use_scale",
    "viewport_translate", "viewport_rotate", "viewport_scale", "viewport_transform",
    # Grob
    "Grob", "GTree", "GList", "GEdit", "GEditList",
    "grob_tree", "grob_name", "is_grob",
    "get_grob", "set_grob", "add_grob", "remove_grob", "edit_grob",
    "force_grob", "set_children", "reorder_grob",
    "apply_edit", "apply_edits",
    # Primitives
    "move_to_grob", "grid_move_to",
    "line_to_grob", "grid_line_to",
    "lines_grob", "grid_lines",
    "polyline_grob", "grid_polyline",
    "segments_grob", "grid_segments",
    "arrows_grob", "grid_arrows",
    "points_grob", "grid_points",
    "rect_grob", "grid_rect",
    "roundrect_grob", "grid_roundrect",
    "circle_grob", "grid_circle",
    "polygon_grob", "grid_polygon",
    "path_grob", "grid_path",
    "text_grob", "grid_text",
    "raster_grob", "grid_raster",
    "clip_grob", "grid_clip",
    "null_grob", "grid_null",
    "function_grob", "grid_function",
    # Coords
    "GridCoords", "GridGrobCoords", "GridGTreeCoords",
    "grob_coords", "grob_points",
    "grid_coords", "grid_grob_coords", "grid_gtree_coords",
    "empty_coords", "empty_grob_coords", "empty_gtree_coords",
    "is_empty_coords",
    # Curves
    "curve_grob", "grid_curve",
    "xspline_grob", "grid_xspline",
    "bezier_grob", "grid_bezier",
    "xspline_points", "bezier_points",
    "arc_curvature",
    # Groups
    "GroupGrob", "DefineGrob", "UseGrob",
    "group_grob", "grid_group",
    "define_grob", "grid_define",
    "use_grob", "grid_use",
    # Draw
    "grid_draw", "grid_newpage", "grid_refresh",
    "grid_record", "record_grob",
    "grid_delay", "delay_grob",
    "grid_dl_apply", "grid_locator",
    # Edit
    "grid_edit", "grid_get", "grid_set", "grid_add", "grid_remove",
    "grid_gedit", "grid_gget", "grid_gremove",
    # LS
    "grid_ls", "grid_grep",
    "nested_listing", "path_listing", "grob_path_listing",
    "show_grob", "get_names", "child_names",
    # Grab
    "grid_grab", "grid_grab_expr",
    "grid_force", "grid_revert",
    "grid_cap", "grid_reorder",
    # High-level
    "grid_grill", "grid_show_layout", "grid_show_viewport",
    "grid_abline", "grid_plot_and_legend", "layout_torture",
    "frame_grob", "grid_frame", "pack_grob", "grid_pack", "place_grob", "grid_place",
    "xaxis_grob", "grid_xaxis", "yaxis_grob", "grid_yaxis",
    "legend_grob", "grid_legend",
    "grid_multipanel", "grid_panel", "grid_strip",
    "grid_top_level_vp",
    # Size
    "calc_string_metric",
    "grob_width", "grob_height", "grob_x", "grob_y",
    "grob_ascent", "grob_descent",
    # Typeset
    "glyph_grob", "grid_glyph",
    # Deprecated
    "convert_native", "grid_convert",
    "grid_convert_x", "grid_convert_y",
    "grid_convert_width", "grid_convert_height",
    "device_loc", "device_dim",
    "grid_collection", "grid_copy",
    "grid_display_list", "engine_display_list",
]
