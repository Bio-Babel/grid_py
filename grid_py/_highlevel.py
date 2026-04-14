"""High-level grid functions -- Python port of R's grid high-level API.

This module ports functionality from three R source files:

* ``grid/R/highlevel.R`` -- grid.grill, grid.show.layout, grid.show.viewport,
  grid.plot.and.legend, grid.abline, layoutTorture, grid.multipanel, etc.
* ``grid/R/frames.R`` -- frameGrob, grid.frame, packGrob, grid.pack,
  placeGrob, grid.place, and internal helpers.
* ``grid/R/components.R`` -- xaxisGrob, grid.xaxis, yaxisGrob, grid.yaxis,
  legendGrob, grid.legend, and related helpers.

References
----------
R source: ``src/library/grid/R/highlevel.R``, ``src/library/grid/R/frames.R``,
``src/library/grid/R/components.R``
"""

from __future__ import annotations

import copy
import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ._gpar import Gpar
from ._grob import (
    GEdit,
    GEditList,
    GList,
    GTree,
    Grob,
    add_grob,
    apply_edits,
    edit_grob,
    get_grob,
    grob_tree,
    is_grob,
    remove_grob,
    set_grob,
)
from ._just import valid_just
from ._layout import (
    GridLayout,
    layout_heights,
    layout_ncol,
    layout_nrow,
    layout_widths,
)
from ._primitives import (
    grid_lines,
    grid_points,
    grid_rect,
    grid_segments,
    grid_text,
    lines_grob,
    null_grob,
    points_grob,
    rect_grob,
    segments_grob,
    text_grob,
)
from ._units import Unit, is_unit, unit_c
from ._viewport import (
    Viewport,
    VpStack,
    current_viewport,
    pop_viewport,
    push_viewport,
)
from ._draw import grid_draw, grid_newpage, grid_pretty

__all__ = [
    # high-level (highlevel.R)
    "grid_grill",
    "grid_plot_and_legend",
    "grid_show_layout",
    "grid_show_viewport",
    "grid_abline",
    "layout_torture",
    # frames (frames.R)
    "frame_grob",
    "grid_frame",
    "pack_grob",
    "grid_pack",
    "place_grob",
    "grid_place",
    # components (components.R)
    "xaxis_grob",
    "yaxis_grob",
    "grid_xaxis",
    "grid_yaxis",
    "legend_grob",
    "grid_legend",
    "grid_multipanel",
    "grid_panel",
    "grid_strip",
    "grid_top_level_vp",
]


# =========================================================================
# Internal helpers
# =========================================================================


def _ensure_unit(value: Any, default_units: str = "npc") -> Unit:
    """Coerce *value* to a :class:`Unit` if it is not one already."""
    if is_unit(value):
        return value
    return Unit(value, default_units)


def _is_even(n: int) -> bool:
    """Return ``True`` if *n* is even."""
    return n % 2 == 0


def _is_odd(n: int) -> bool:
    """Return ``True`` if *n* is odd."""
    return n % 2 == 1


def _extend_range(x: Sequence[float], f: float = 0.05) -> Tuple[float, float]:
    """Extend range of *x* by fraction *f* on each side (like R extendrange)."""
    mn, mx = float(min(x)), float(max(x))
    rng = mx - mn
    if rng == 0:
        rng = 1.0
    return (mn - f * rng, mx + f * rng)


# =========================================================================
# Frame / Pack / Place  (frames.R)
# =========================================================================


def frame_grob(
    layout: Optional[GridLayout] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GTree:
    """Create a frame grob -- a GTree intended for packing child grobs.

    Parameters
    ----------
    layout : GridLayout or None
        Optional initial layout for the frame.
    name : str or None
        Grob name (auto-generated if ``None``).
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
        A GTree with ``_grid_class="frame"`` and a *framevp* attribute.
    """
    framevp: Optional[Viewport] = None
    if layout is not None:
        framevp = Viewport(layout=layout)
    return GTree(
        name=name,
        gp=gp,
        vp=vp,
        _grid_class="frame",
        framevp=framevp,
    )


def grid_frame(
    layout: Optional[GridLayout] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    draw: bool = True,
) -> GTree:
    """Create (and optionally draw) a frame grob.

    Parameters
    ----------
    layout : GridLayout or None
        Optional initial layout.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.
    draw : bool
        If ``True``, draw the frame immediately.

    Returns
    -------
    GTree
        The frame grob.
    """
    fg = frame_grob(layout=layout, name=name, gp=gp, vp=vp)
    if draw:
        grid_draw(fg)
    return fg


# ---------------------------------------------------------------------------
# Internal helpers for cell grobs and packing
# ---------------------------------------------------------------------------


def _cell_viewport(
    col: Any,
    row: Any,
    border: Optional[Sequence[Unit]],
) -> Any:
    """Build a viewport (or VpStack) for a cell, optionally with border insets.

    Parameters
    ----------
    col : int or list
        Column index or range.
    row : int or list
        Row index or range.
    border : list of Unit or None
        Four-element border ``[bottom, left, top, right]``.

    Returns
    -------
    Viewport or VpStack
    """
    vp = Viewport(layout_pos_col=col, layout_pos_row=row)
    if border is not None:
        inner = Viewport(
            x=border[1],
            y=border[0],
            width=Unit(1, "npc") - (border[1] + border[3]),
            height=Unit(1, "npc") - (border[0] + border[2]),
            just=["left", "bottom"],
        )
        return VpStack(vp, inner)
    return vp


def _cell_grob(
    col: Any,
    row: Any,
    border: Optional[Sequence[Unit]],
    grob: Grob,
    dynamic: bool,
    vp: Any,
) -> GTree:
    """Wrap a grob in a cellGrob container.

    Parameters
    ----------
    col : int or list
        Column position.
    row : int or list
        Row position.
    border : list of Unit or None
        Border insets.
    grob : Grob
        The child grob.
    dynamic : bool
        Whether to use dynamic sizing.
    vp : object
        Cell viewport.

    Returns
    -------
    GTree
        A GTree with ``_grid_class="cellGrob"``.
    """
    return GTree(
        children=GList(grob),
        _grid_class="cellGrob",
        col=col,
        row=row,
        border=border,
        dynamic=dynamic,
        cellvp=vp,
    )


def _frame_dim(frame: GTree) -> Tuple[int, int]:
    """Return ``(nrow, ncol)`` of *frame*'s layout, or ``(0, 0)``."""
    framevp = getattr(frame, "framevp", None)
    if framevp is None:
        return (0, 0)
    lay = getattr(framevp, "layout", None)
    if lay is None:
        return (0, 0)
    return (layout_nrow(lay), layout_ncol(lay))


# -- column / row specification helpers ------------------------------------


def _num_col_specs(
    side: Optional[str],
    col: Optional[Any],
    col_before: Optional[int],
    col_after: Optional[int],
) -> int:
    side_counts = 0 if (side is None or side in ("top", "bottom")) else 1
    return side_counts + (col is not None) + (col_before is not None) + (col_after is not None)


def _col_spec(
    side: Optional[str],
    col: Optional[int],
    col_before: Optional[int],
    col_after: Optional[int],
    ncol: int,
) -> int:
    if side is not None:
        if side == "left":
            return 1
        if side == "right":
            return ncol + 1
    if col_before is not None:
        return col_before
    if col_after is not None:
        return col_after + 1
    return col  # type: ignore[return-value]


def _new_col(
    side: Optional[str],
    col: Optional[Any],
    col_before: Optional[int],
    col_after: Optional[int],
    ncol: int,
) -> bool:
    result = True
    if col is not None:
        if isinstance(col, (list, tuple)) and len(col) == 2:
            if col[0] < 1 or col[1] > ncol:
                raise ValueError("'col' can only be a range of existing columns")
            result = False
        else:
            c = col if isinstance(col, int) else col
            if c < 1 or c > ncol + 1:
                raise ValueError("invalid 'col' specification")
            result = c == ncol + 1
    return result


def _num_row_specs(
    side: Optional[str],
    row: Optional[Any],
    row_before: Optional[int],
    row_after: Optional[int],
) -> int:
    side_counts = 0 if (side is None or side in ("left", "right")) else 1
    return side_counts + (row is not None) + (row_before is not None) + (row_after is not None)


def _row_spec(
    side: Optional[str],
    row: Optional[int],
    row_before: Optional[int],
    row_after: Optional[int],
    nrow: int,
) -> int:
    if side is not None:
        if side == "top":
            return 1
        if side == "bottom":
            return nrow + 1
    if row_before is not None:
        return row_before
    if row_after is not None:
        return row_after + 1
    return row  # type: ignore[return-value]


def _new_row(
    side: Optional[str],
    row: Optional[Any],
    row_before: Optional[int],
    row_after: Optional[int],
    nrow: int,
) -> bool:
    result = True
    if row is not None:
        if isinstance(row, (list, tuple)) and len(row) == 2:
            if row[0] < 1 or row[1] > nrow:
                raise ValueError("'row' can only be a range of existing rows")
            result = False
        else:
            r = row if isinstance(row, int) else row
            if r < 1 or r > nrow + 1:
                raise ValueError("invalid 'row' specification")
            result = r == nrow + 1
    return result


def _mod_dims(
    dim: Unit,
    dims: Unit,
    index: int,
    is_new: bool,
    nindex: int,
    force: bool,
) -> Unit:
    """Update dimension list when packing a new grob.

    Parameters
    ----------
    dim : Unit
        Width or height of the new grob.
    dims : Unit
        Current list of widths or heights.
    index : int
        1-based index for the new grob's row/column.
    is_new : bool
        Whether a new row/column is being added.
    nindex : int
        Current number of rows/columns (before adding).
    force : bool
        If ``True``, override existing dimension; otherwise take the max.

    Returns
    -------
    Unit
    """
    if is_new:
        if index == 1:
            return unit_c(dim, dims)
        elif index == nindex:
            return unit_c(dims, dim)
        else:
            # Insert before the existing index
            before = dims[0 : index - 1] if index > 1 else None
            after = dims[index - 1 :] if index <= nindex else None
            parts: list[Unit] = []
            if before is not None:
                parts.append(before)
            parts.append(dim)
            if after is not None:
                parts.append(after)
            result = parts[0]
            for p in parts[1:]:
                result = unit_c(result, p)
            return result
    else:
        # Existing row/col: take max or force
        if not force:
            # Use the larger of existing and new
            # For simplicity, just keep existing (proper max requires context)
            pass
        # Replace the dimension at *index* (1-based)
        idx0 = index - 1
        parts_list: list[Unit] = []
        if idx0 > 0:
            parts_list.append(dims[0:idx0])
        parts_list.append(dim)
        if idx0 + 1 < nindex:
            parts_list.append(dims[idx0 + 1 :])
        if not parts_list:
            return dim
        result2 = parts_list[0]
        for p in parts_list[1:]:
            result2 = unit_c(result2, p)
        return result2


def _update_col(col: Any, added_col: int) -> Any:
    """Shift *col* if a new column was inserted before or at it."""
    if isinstance(col, (list, tuple)) and len(col) == 2:
        if added_col <= col[1]:
            return [col[0], col[1] + 1]
        return col
    if added_col <= col:
        return col + 1
    return col


def _update_row(row: Any, added_row: int) -> Any:
    """Shift *row* if a new row was inserted before or at it."""
    if isinstance(row, (list, tuple)) and len(row) == 2:
        if added_row <= row[1]:
            return [row[0], row[1] + 1]
        return row
    if added_row <= row:
        return row + 1
    return row


# ---------------------------------------------------------------------------
# pack_grob / grid_pack
# ---------------------------------------------------------------------------


def pack_grob(
    frame: GTree,
    grob: Grob,
    side: Optional[str] = None,
    row: Optional[Any] = None,
    row_before: Optional[int] = None,
    row_after: Optional[int] = None,
    col: Optional[Any] = None,
    col_before: Optional[int] = None,
    col_after: Optional[int] = None,
    width: Optional[Unit] = None,
    height: Optional[Unit] = None,
    force_width: bool = False,
    force_height: bool = False,
    border: Optional[Sequence[Unit]] = None,
    dynamic: bool = False,
) -> GTree:
    """Pack a grob into a frame, returning a new frame.

    This function is the Python equivalent of R's ``packGrob``.  It manages
    the frame's internal layout, adding rows/columns as needed, and wraps
    the child grob in a ``cellGrob``.

    Parameters
    ----------
    frame : GTree
        The frame grob (must have ``_grid_class="frame"``).
    grob : Grob
        The child grob to pack.
    side : str or None
        One of ``"left"``, ``"right"``, ``"top"``, ``"bottom"``.
    row : int, list of int, or None
        Row position or range.
    row_before : int or None
        Insert before this row.
    row_after : int or None
        Insert after this row.
    col : int, list of int, or None
        Column position or range.
    col_before : int or None
        Insert before this column.
    col_after : int or None
        Insert after this column.
    width : Unit or None
        Explicit width (derived from *grob* if ``None``).
    height : Unit or None
        Explicit height (derived from *grob* if ``None``).
    force_width : bool
        If ``True``, override the existing column width.
    force_height : bool
        If ``True``, override the existing row height.
    border : list of Unit or None
        Four-element border insets ``[bottom, left, top, right]``.
    dynamic : bool
        If ``True``, use a grob path for deferred sizing.

    Returns
    -------
    GTree
        A new frame with the child packed in.

    Raises
    ------
    TypeError
        If *frame* is not a frame grob or *grob* is not a grob.
    ValueError
        If the row/column specification is invalid.
    """
    if not isinstance(frame, GTree) or getattr(frame, "_grid_class", None) != "frame":
        raise TypeError("invalid 'frame'")
    if not is_grob(grob):
        raise TypeError("invalid 'grob'")

    # Normalise col/row ranges
    col_range = False
    row_range = False
    if col is not None and isinstance(col, (list, tuple)) and len(col) > 1:
        col = [min(col), max(col)]
        col_range = True
    if row is not None and isinstance(row, (list, tuple)) and len(row) > 1:
        row = [min(row), max(row)]
        row_range = True

    # Get current layout dimensions
    frame = copy.deepcopy(frame)
    frame_vp = getattr(frame, "framevp", None)
    if frame_vp is None:
        frame_vp = Viewport()
    lay = getattr(frame_vp, "layout", None)
    if lay is None:
        ncol = 0
        nrow = 0
    else:
        ncol = layout_ncol(lay)
        nrow = layout_nrow(lay)

    # (i) Validate location specifications
    ncs = _num_col_specs(side, col, col_before, col_after)
    if ncs == 0:
        if ncol > 0:
            col = [1, ncol]
            col_range = True
        else:
            col = 1
        ncs = 1
    if ncs != 1:
        raise ValueError(
            "cannot specify more than one of 'side=[left/right]', "
            "'col', 'col_before', or 'col_after'"
        )

    nrs = _num_row_specs(side, row, row_before, row_after)
    if nrs == 0:
        if nrow > 0:
            row = [1, nrow]
            row_range = True
        else:
            row = 1
        nrs = 1
    if nrs != 1:
        raise ValueError(
            "must specify exactly one of 'side=[top/bottom]', "
            "'row', 'row_before', or 'row_after'"
        )

    # (ii) Determine location
    is_new_col = _new_col(side, col, col_before, col_after, ncol)
    col = _col_spec(side, col, col_before, col_after, ncol)
    is_new_row = _new_row(side, row, row_before, row_after, nrow)
    row = _row_spec(side, row, row_before, row_after, nrow)

    # Build cell grob
    cgrob: Optional[GTree] = None
    if grob is not None:
        cgrob = _cell_grob(col, row, border, grob, dynamic,
                           _cell_viewport(col, row, border))

    # (iii) Default width/height from the grob
    if width is None:
        width = Unit(1, "null") if grob is None else Unit(1, "null")
    if height is None:
        height = Unit(1, "null") if grob is None else Unit(1, "null")

    # Include border in width/height
    if border is not None:
        width = border[1] + width + border[3]
        height = border[0] + height + border[2]

    # (iv) Update layout
    if is_new_col:
        ncol += 1
    if is_new_row:
        nrow += 1

    if lay is None:
        widths = width
        heights = height
    else:
        if col_range:
            widths = layout_widths(lay)
        else:
            widths = _mod_dims(width, layout_widths(lay), col, is_new_col,
                               ncol, force_width)
        if row_range:
            heights = layout_heights(lay)
        else:
            heights = _mod_dims(height, layout_heights(lay), row, is_new_row,
                                nrow, force_height)

    frame_vp._layout = GridLayout(nrow=nrow, ncol=ncol,
                                  widths=widths, heights=heights)

    # Shift existing children if new row/col was added
    if is_new_col or is_new_row:
        for child_name in list(frame._children_order):
            child = frame._children[child_name]
            if is_new_col:
                new_c = _update_col(getattr(child, "col", 1), col)
                child.col = new_c
                child.cellvp = _cell_viewport(new_c, getattr(child, "row", 1),
                                              getattr(child, "border", None))
            if is_new_row:
                new_r = _update_row(getattr(child, "row", 1), row)
                child.row = new_r
                child.cellvp = _cell_viewport(getattr(child, "col", 1), new_r,
                                              getattr(child, "border", None))

    # Add the new grob
    if cgrob is not None:
        frame.add_child(cgrob)

    frame.framevp = frame_vp
    return frame


def grid_pack(
    frame: GTree,
    grob: Grob,
    redraw: bool = True,
    side: Optional[str] = None,
    row: Optional[Any] = None,
    row_before: Optional[int] = None,
    row_after: Optional[int] = None,
    col: Optional[Any] = None,
    col_before: Optional[int] = None,
    col_after: Optional[int] = None,
    width: Optional[Unit] = None,
    height: Optional[Unit] = None,
    force_width: bool = False,
    force_height: bool = False,
    border: Optional[Sequence[Unit]] = None,
    dynamic: bool = False,
) -> GTree:
    """Pack a grob into a frame and optionally redraw.

    This is the display-list-aware wrapper around :func:`pack_grob`.

    Parameters
    ----------
    frame : GTree
        The frame grob.
    grob : Grob
        The child grob to pack.
    redraw : bool
        If ``True``, redraw after packing.
    side : str or None
        Side specification.
    row, row_before, row_after : int or None
        Row specification.
    col, col_before, col_after : int or None
        Column specification.
    width : Unit or None
        Explicit width.
    height : Unit or None
        Explicit height.
    force_width : bool
        Override existing column width.
    force_height : bool
        Override existing row height.
    border : list of Unit or None
        Border insets.
    dynamic : bool
        Use dynamic sizing.

    Returns
    -------
    GTree
        The updated frame.
    """
    result = pack_grob(
        frame, grob,
        side=side,
        row=row, row_before=row_before, row_after=row_after,
        col=col, col_before=col_before, col_after=col_after,
        width=width, height=height,
        force_width=force_width, force_height=force_height,
        border=border, dynamic=dynamic,
    )
    if redraw:
        grid_draw(result)
    return result


# ---------------------------------------------------------------------------
# place_grob / grid_place
# ---------------------------------------------------------------------------


def place_grob(
    frame: GTree,
    grob: Grob,
    row: Optional[Any] = None,
    col: Optional[Any] = None,
) -> GTree:
    """Place a grob into an existing cell of a frame.

    Unlike :func:`pack_grob`, this does **not** create new rows or columns;
    it only places the grob into an already-existing cell.

    Parameters
    ----------
    frame : GTree
        The frame grob (must have ``_grid_class="frame"``).
    grob : Grob
        The child grob.
    row : int, list of int, or None
        Row position (defaults to full range).
    col : int, list of int, or None
        Column position (defaults to full range).

    Returns
    -------
    GTree
        A new frame with the grob placed.

    Raises
    ------
    TypeError
        If *frame* is not a frame grob or *grob* is not a grob.
    ValueError
        If *row*/*col* are out of range.
    """
    if not isinstance(frame, GTree) or getattr(frame, "_grid_class", None) != "frame":
        raise TypeError("invalid 'frame'")
    if not is_grob(grob):
        raise TypeError("invalid 'grob'")

    dim = _frame_dim(frame)
    if row is None:
        row = [1, dim[0]]
    if col is None:
        col = [1, dim[1]]

    # Validate
    row_vals = row if isinstance(row, (list, tuple)) else [row]
    col_vals = col if isinstance(col, (list, tuple)) else [col]
    if min(row_vals) < 1 or max(row_vals) > dim[0]:
        raise ValueError("invalid 'row' (no such row in frame layout)")
    if min(col_vals) < 1 or max(col_vals) > dim[1]:
        raise ValueError("invalid 'col' (no such col in frame layout)")

    cgrob = _cell_grob(col, row, None, grob, False,
                        _cell_viewport(col, row, None))
    return add_grob(frame, cgrob)


def grid_place(
    frame: GTree,
    grob: Grob,
    row: int = 1,
    col: int = 1,
    redraw: bool = True,
) -> GTree:
    """Place a grob into an existing cell of a frame and optionally redraw.

    Parameters
    ----------
    frame : GTree
        The frame grob.
    grob : Grob
        The child grob.
    row : int
        Row position.
    col : int
        Column position.
    redraw : bool
        If ``True``, redraw after placing.

    Returns
    -------
    GTree
        The updated frame.
    """
    result = place_grob(frame, grob, row=row, col=col)
    if redraw:
        grid_draw(result)
    return result


# =========================================================================
# Components  (components.R)
# =========================================================================


# ---------------------------------------------------------------------------
# X-axis internals
# ---------------------------------------------------------------------------


def _make_xaxis_major(at: Sequence[float], main: bool) -> Grob:
    """Create the major line for an x-axis."""
    y = [0, 0] if main else [1, 1]
    return lines_grob(
        x=Unit([min(at), max(at)], "native"),
        y=Unit(y, "npc"),
        name="major",
    )


def _make_xaxis_ticks(at: Sequence[float], main: bool) -> Grob:
    """Create tick marks for an x-axis."""
    if main:
        tick_y0 = Unit(0, "npc")
        tick_y1 = Unit(-0.5, "lines")
    else:
        tick_y0 = Unit(1, "npc")
        tick_y1 = Unit(1, "npc") + Unit(0.5, "lines")
    return segments_grob(
        x0=Unit(list(at), "native"),
        y0=tick_y0,
        x1=Unit(list(at), "native"),
        y1=tick_y1,
        name="ticks",
    )


def _make_xaxis_labels(
    at: Sequence[float],
    label: Any,
    main: bool,
) -> Grob:
    """Create tick labels for an x-axis."""
    label_y = Unit(-1.5, "lines") if main else Unit(1, "npc") + Unit(1.5, "lines")
    labels = [str(a) for a in at] if isinstance(label, bool) else list(label)
    return text_grob(
        label=labels,
        x=Unit(list(at), "native"),
        y=label_y,
        just="centre",
        rot=0,
        name="labels",
    )


def _update_xlabels(x: GTree) -> GTree:
    """Add or remove x-axis labels depending on label specification."""
    lab = getattr(x, "label", True)
    if isinstance(lab, bool) and not lab:
        try:
            return remove_grob(x, "labels")
        except KeyError:
            return x
    return add_grob(x, _make_xaxis_labels(x.at, x.label, x.main))


# ---------------------------------------------------------------------------
# Y-axis internals
# ---------------------------------------------------------------------------


def _make_yaxis_major(at: Sequence[float], main: bool) -> Grob:
    """Create the major line for a y-axis."""
    x = [0, 0] if main else [1, 1]
    return lines_grob(
        x=Unit(x, "npc"),
        y=Unit([min(at), max(at)], "native"),
        name="major",
    )


def _make_yaxis_ticks(at: Sequence[float], main: bool) -> Grob:
    """Create tick marks for a y-axis."""
    if main:
        tick_x0 = Unit(0, "npc")
        tick_x1 = Unit(-0.5, "lines")
    else:
        tick_x0 = Unit(1, "npc")
        tick_x1 = Unit(1, "npc") + Unit(0.5, "lines")
    return segments_grob(
        x0=tick_x0,
        y0=Unit(list(at), "native"),
        x1=tick_x1,
        y1=Unit(list(at), "native"),
        name="ticks",
    )


def _make_yaxis_labels(
    at: Sequence[float],
    label: Any,
    main: bool,
) -> Grob:
    """Create tick labels for a y-axis."""
    if main:
        hjust = "right"
        label_x = Unit(-1, "lines")
    else:
        hjust = "left"
        label_x = Unit(1, "npc") + Unit(1, "lines")
    labels = [str(a) for a in at] if isinstance(label, bool) else list(label)
    return text_grob(
        label=labels,
        x=label_x,
        y=Unit(list(at), "native"),
        just=[hjust, "centre"],
        rot=0,
        name="labels",
    )


def _update_ylabels(x: GTree) -> GTree:
    """Add or remove y-axis labels depending on label specification."""
    lab = getattr(x, "label", True)
    if isinstance(lab, bool) and not lab:
        try:
            return remove_grob(x, "labels")
        except KeyError:
            return x
    return add_grob(x, _make_yaxis_labels(x.at, x.label, x.main))


class _XAxisGTree(GTree):
    """GTree subclass for x-axis with on-the-fly tick generation."""

    def __init__(self, at, label, main, edits, **kwargs):
        super().__init__(_grid_class="xaxis", at=at, label=label,
                         main=main, edits=edits, **kwargs)

    def make_content(self):
        at = getattr(self, "at", None)
        if at is None:
            from ._viewport import current_viewport
            vp = current_viewport()
            xscale = getattr(vp, "_xscale", None) or getattr(vp, "xscale", [0, 1])
            at = grid_pretty(xscale)
            self.at = at
            main = getattr(self, "main", True)
            label = getattr(self, "label", True)
            self = add_grob(self, _make_xaxis_major(at, main))
            self = add_grob(self, _make_xaxis_ticks(at, main))
            self = _update_xlabels(self)
            edits = getattr(self, "edits", None)
            if edits is not None:
                self = apply_edits(self, edits)
        return self


class _YAxisGTree(GTree):
    """GTree subclass for y-axis with on-the-fly tick generation."""

    def __init__(self, at, label, main, edits, **kwargs):
        super().__init__(_grid_class="yaxis", at=at, label=label,
                         main=main, edits=edits, **kwargs)

    def make_content(self):
        at = getattr(self, "at", None)
        if at is None:
            from ._viewport import current_viewport
            vp = current_viewport()
            yscale = getattr(vp, "_yscale", None) or getattr(vp, "yscale", [0, 1])
            at = grid_pretty(yscale)
            self.at = at
            main = getattr(self, "main", True)
            label = getattr(self, "label", True)
            self = add_grob(self, _make_yaxis_major(at, main))
            self = add_grob(self, _make_yaxis_ticks(at, main))
            self = _update_ylabels(self)
            edits = getattr(self, "edits", None)
            if edits is not None:
                self = apply_edits(self, edits)
        return self


# ---------------------------------------------------------------------------
# Public axis constructors
# ---------------------------------------------------------------------------


def xaxis_grob(
    at: Optional[Sequence[float]] = None,
    label: Any = True,
    main: bool = True,
    edits: Optional[Any] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GTree:
    """Create an x-axis grob.

    Parameters
    ----------
    at : sequence of float or None
        Tick positions in native coordinates.  If ``None``, tick positions
        are calculated on-the-fly when the grob is drawn.
    label : bool or sequence of str
        If ``True`` (default), labels are derived from *at*.  If ``False``,
        no labels are drawn.  A sequence provides explicit labels.
    main : bool
        If ``True`` (default), the axis is drawn on the bottom.
    edits : GEdit, GEditList, or None
        Edits to apply to child grobs.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
        A GTree with ``_grid_class="xaxis"``.
    """
    return grid_xaxis(at=at, label=label, main=main, edits=edits,
                      name=name, gp=gp, draw=False, vp=vp)


def grid_xaxis(
    at: Optional[Sequence[float]] = None,
    label: Any = True,
    main: bool = True,
    edits: Optional[Any] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> GTree:
    """Create and optionally draw an x-axis.

    Parameters
    ----------
    at : sequence of float or None
        Tick positions in native coordinates.
    label : bool or sequence of str
        Label specification.
    main : bool
        If ``True``, the axis is on the bottom.
    edits : GEdit, GEditList, or None
        Edits to apply to children.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True``, draw immediately.
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
        A GTree with ``_grid_class="xaxis"``.
    """
    if at is None:
        major = None
        ticks = None
        labels = None
    else:
        at = [float(a) for a in at]
        major = _make_xaxis_major(at, main)
        ticks = _make_xaxis_ticks(at, main)
        if isinstance(label, bool) and not label:
            labels = None
        else:
            labels = _make_xaxis_labels(at, label, main)

    children_list = [g for g in (major, ticks, labels) if g is not None]
    xg = _XAxisGTree(
        at=at, label=label, main=main, edits=edits,
        children=GList(*children_list) if children_list else None,
        name=name, gp=gp, vp=vp,
    )
    if edits is not None:
        xg = apply_edits(xg, edits)
    if draw:
        grid_draw(xg)
    return xg


def yaxis_grob(
    at: Optional[Sequence[float]] = None,
    label: Any = True,
    main: bool = True,
    edits: Optional[Any] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GTree:
    """Create a y-axis grob.

    Parameters
    ----------
    at : sequence of float or None
        Tick positions in native coordinates.  If ``None``, tick positions
        are calculated on-the-fly when the grob is drawn.
    label : bool or sequence of str
        Label specification.
    main : bool
        If ``True`` (default), the axis is drawn on the left.
    edits : GEdit, GEditList, or None
        Edits to apply to children.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
        A GTree with ``_grid_class="yaxis"``.
    """
    return grid_yaxis(at=at, label=label, main=main, edits=edits,
                      name=name, gp=gp, draw=False, vp=vp)


def grid_yaxis(
    at: Optional[Sequence[float]] = None,
    label: Any = True,
    main: bool = True,
    edits: Optional[Any] = None,
    name: Optional[str] = None,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    vp: Optional[Any] = None,
) -> GTree:
    """Create and optionally draw a y-axis.

    Parameters
    ----------
    at : sequence of float or None
        Tick positions in native coordinates.
    label : bool or sequence of str
        Label specification.
    main : bool
        If ``True``, the axis is on the left.
    edits : GEdit, GEditList, or None
        Edits to apply to children.
    name : str or None
        Grob name.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True``, draw immediately.
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
        A GTree with ``_grid_class="yaxis"``.
    """
    if at is None:
        major = None
        ticks = None
        labels = None
    else:
        at = [float(a) for a in at]
        major = _make_yaxis_major(at, main)
        ticks = _make_yaxis_ticks(at, main)
        if isinstance(label, bool) and not label:
            labels = None
        else:
            labels = _make_yaxis_labels(at, label, main)

    children_list = [g for g in (major, ticks, labels) if g is not None]
    yg = _YAxisGTree(
        at=at, label=label, main=main, edits=edits,
        children=GList(*children_list) if children_list else None,
        name=name, gp=gp, vp=vp,
    )
    if edits is not None:
        yg = apply_edits(yg, edits)
    if draw:
        grid_draw(yg)
    return yg


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def legend_grob(
    labels: Sequence[str],
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    byrow: bool = False,
    do_lines: bool = True,
    do_points: bool = True,
    lines_first: bool = True,
    pch: Optional[Sequence[int]] = None,
    hgap: Any = None,
    vgap: Any = None,
    default_units: str = "lines",
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> GTree:
    """Create a legend grob.

    This is the Python port of R's ``legendGrob``.  It builds a frame
    grob containing symbol and text entries arranged in a grid.

    Parameters
    ----------
    labels : sequence of str
        Legend entry labels.
    nrow : int or None
        Number of rows.  If ``None`` and *ncol* is also ``None``, defaults
        to ``len(labels)`` with ``ncol=1``.
    ncol : int or None
        Number of columns.
    byrow : bool
        If ``True``, fill by row; otherwise by column.
    do_lines : bool
        If ``True``, draw lines in the symbol column.
    do_points : bool
        If ``True``, draw points in the symbol column.
    lines_first : bool
        If ``True``, draw lines before points.
    pch : sequence of int or None
        Point characters.
    hgap : Unit or numeric or None
        Horizontal gap between columns.  Defaults to ``Unit(1, "lines")``.
    vgap : Unit or numeric or None
        Vertical gap between rows.  Defaults to ``Unit(1, "lines")``.
    default_units : str
        Unit type for bare numerics in *hgap* / *vgap*.
    gp : Gpar or None
        Graphical parameters (may contain ``col``, ``lty``, ``lwd``, ``fill``).
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
        A frame grob containing the legend entries.
    """
    labels = [str(lb) for lb in labels]
    nkeys = len(labels)
    if nkeys == 0:
        return null_grob(vp=vp)  # type: ignore[return-value]

    # Defaults
    if hgap is None:
        hgap = Unit(1, "lines")
    elif not is_unit(hgap):
        hgap = Unit(hgap, default_units)
    if vgap is None:
        vgap = Unit(1, "lines")
    elif not is_unit(vgap):
        vgap = Unit(vgap, default_units)

    # nrow / ncol defaults
    if nrow is not None and nrow < 1:
        raise ValueError("'nrow' must be >= 1")
    if ncol is not None and ncol < 1:
        raise ValueError("'ncol' must be >= 1")

    if nrow is None and ncol is None:
        ncol = 1
        nrow = nkeys
    elif nrow is None:
        nrow = math.ceil(nkeys / ncol)  # type: ignore[arg-type]
    elif ncol is None:
        ncol = math.ceil(nkeys / nrow)
    if nrow * ncol < nkeys:  # type: ignore[operator]
        raise ValueError("nrow * ncol < number of legend labels")

    # Recycle pch
    has_pch = pch is not None and len(pch) > 0
    if has_pch:
        pch_list = list(pch)  # type: ignore[arg-type]
        while len(pch_list) < nkeys:
            pch_list = pch_list * (nkeys // len(pch_list) + 1)
        pch_list = pch_list[:nkeys]
    else:
        pch_list = []

    # Extract per-key gp components
    gp_dict: Dict[str, Any] = {}
    if gp is not None:
        for attr in ("lty", "lwd", "col", "fill"):
            val = getattr(gp, attr, None)
            if val is not None:
                if isinstance(val, (list, tuple, np.ndarray)):
                    lst = list(val)
                    while len(lst) < nkeys:
                        lst = lst * (nkeys // len(lst) + 1)
                    gp_dict[attr] = lst[:nkeys]
                else:
                    gp_dict[attr] = [val] * nkeys

    u0 = Unit(0, "npc")
    u1 = Unit(1, "char")

    fg = frame_grob(vp=vp)

    for i in range(nkeys):
        if byrow:
            ci = 1 + (i % ncol)  # type: ignore[operator]
            ri = 1 + (i // ncol)  # type: ignore[operator]
        else:
            ci = 1 + (i // nrow)
            ri = 1 + (i % nrow)

        # Build per-key gp
        gpi_kwargs: Dict[str, Any] = {}
        for attr in ("lty", "lwd", "col", "fill"):
            if attr in gp_dict:
                gpi_kwargs[attr] = gp_dict[attr][i]
        gpi = Gpar(**gpi_kwargs) if gpi_kwargs else (gp if gp is not None else Gpar())

        # Borders
        vg = vgap if ri != nrow else u0
        symbol_border = [vg, u0, u0, hgap * 0.5]
        text_border = [vg, u0, u0, hgap if ci != ncol else u0]  # type: ignore[operator]

        # Points/lines grob
        if has_pch and do_lines:
            line_g = lines_grob(x=Unit([0, 1], "npc"), y=Unit(0.5, "npc"), gp=gpi)
            point_g = points_grob(
                x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                pch=pch_list[i], gp=gpi,
            )
            if lines_first:
                pl_grob: Grob = GTree(children=GList(line_g, point_g))
            else:
                pl_grob = GTree(children=GList(point_g, line_g))
        elif has_pch:
            pl_grob = points_grob(
                x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                pch=pch_list[i], gp=gpi,
            )
        elif do_lines:
            pl_grob = lines_grob(x=Unit([0, 1], "npc"), y=Unit(0.5, "npc"), gp=gpi)
        else:
            pl_grob = null_grob()

        fg = pack_grob(
            fg, pl_grob,
            col=2 * ci - 1, row=ri,
            border=symbol_border,
            width=u1, height=u1,
            force_width=True,
        )

        # Text grob
        gpi_text = Gpar(col="black")
        fg = pack_grob(
            fg,
            text_grob(label=labels[i], x=Unit(0, "npc"), y=Unit(0.5, "npc"),
                       just=["left", "centre"], gp=gpi_text),
            col=2 * ci, row=ri,
            border=text_border,
        )

    return fg


def grid_legend(
    labels: Sequence[str],
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    byrow: bool = False,
    do_lines: bool = True,
    do_points: bool = True,
    lines_first: bool = True,
    pch: Optional[Sequence[int]] = None,
    hgap: Any = None,
    vgap: Any = None,
    default_units: str = "lines",
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    draw: bool = True,
) -> GTree:
    """Create and optionally draw a legend.

    Parameters
    ----------
    labels : sequence of str
        Legend entry labels.
    nrow : int or None
        Number of rows.
    ncol : int or None
        Number of columns.
    byrow : bool
        Fill by row.
    do_lines : bool
        Draw lines in symbol column.
    do_points : bool
        Draw points in symbol column.
    lines_first : bool
        Draw lines before points.
    pch : sequence of int or None
        Point characters.
    hgap : Unit or numeric or None
        Horizontal gap.
    vgap : Unit or numeric or None
        Vertical gap.
    default_units : str
        Default unit type.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.
    draw : bool
        If ``True``, draw immediately.

    Returns
    -------
    GTree
        The legend grob.
    """
    g = legend_grob(
        labels, nrow=nrow, ncol=ncol, byrow=byrow,
        do_lines=do_lines, do_points=do_points,
        lines_first=lines_first, pch=pch,
        hgap=hgap, vgap=vgap, default_units=default_units,
        gp=gp, vp=vp,
    )
    if draw:
        grid_draw(g)
    return g


# =========================================================================
# High-level functions  (highlevel.R)
# =========================================================================


def grid_grill(
    h: Optional[Any] = None,
    v: Optional[Any] = None,
    default_units: str = "npc",
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
) -> None:
    """Draw a grid of horizontal and vertical lines (background grid).

    Parameters
    ----------
    h : Unit, numeric, or None
        Horizontal line positions.  Defaults to ``[0.25, 0.5, 0.75]`` in
        NPC coordinates.
    v : Unit, numeric, or None
        Vertical line positions.  Defaults to ``[0.25, 0.5, 0.75]`` in
        NPC coordinates.
    default_units : str
        Unit type for bare numerics.
    gp : Gpar or None
        Graphical parameters.  Defaults to ``Gpar(col="grey")``.
    vp : object or None
        Viewport.
    """
    if h is None:
        h = Unit([0.25, 0.50, 0.75], "npc")
    if not is_unit(h):
        h = Unit(h, default_units)
    if v is None:
        v = Unit([0.25, 0.50, 0.75], "npc")
    if not is_unit(v):
        v = Unit(v, default_units)
    if gp is None:
        gp = Gpar(col="grey")

    if vp is not None:
        push_viewport(vp)

    # Vertical lines
    grid_segments(
        x0=v, y0=Unit(0, "npc"),
        x1=v, y1=Unit(1, "npc"),
        gp=gp,
    )
    # Horizontal lines
    grid_segments(
        x0=Unit(0, "npc"), y0=h,
        x1=Unit(1, "npc"), y1=h,
        gp=gp,
    )

    if vp is not None:
        pop_viewport()


def grid_show_layout(
    layout: GridLayout,
    newpage: bool = True,
    vp_ex: float = 0.8,
    bg: str = "light grey",
    cell_border: str = "blue",
    cell_fill: str = "light blue",
    cell_label: bool = True,
    label_col: str = "blue",
    unit_col: str = "red",
    vp: Optional[Any] = None,
) -> Viewport:
    """Visualize a :class:`GridLayout`.

    Draws a representation of the layout on the current device, showing
    cell boundaries, labels, and dimension annotations.

    Parameters
    ----------
    layout : GridLayout
        The layout to visualize.
    newpage : bool
        If ``True``, start a new page before drawing.
    vp_ex : float
        Fraction of the page used for the viewport (0--1).
    bg : str
        Background colour.
    cell_border : str
        Cell border colour.
    cell_fill : str
        Cell fill colour.
    cell_label : bool
        If ``True``, label each cell with ``(row, col)``.
    label_col : str
        Colour for cell labels.
    unit_col : str
        Colour for dimension annotations.
    vp : object or None
        Viewport to push before drawing.

    Returns
    -------
    Viewport
        The viewport used to represent the parent region.
    """
    if newpage:
        grid_newpage()
    if vp is not None:
        push_viewport(vp)

    grid_rect(gp=Gpar(col=None, fill=bg))

    vp_mid = Viewport(
        x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
        width=Unit(vp_ex, "npc"), height=Unit(vp_ex, "npc"),
        layout=layout,
    )
    push_viewport(vp_mid)
    grid_rect(gp=Gpar(fill="white"))

    gp_red = Gpar(col=unit_col)
    nr = layout_nrow(layout)
    nc = layout_ncol(layout)

    for i in range(1, nr + 1):
        for j in range(1, nc + 1):
            vp_inner = Viewport(layout_pos_row=i, layout_pos_col=j)
            push_viewport(vp_inner)
            grid_rect(gp=Gpar(col=cell_border, fill=cell_fill))
            if cell_label:
                grid_text(label=f"({i}, {j})", gp=Gpar(col=label_col))
            # Dimension annotations on the edges
            if j == 1:
                grid_text(
                    label=str(layout_heights(layout)),
                    gp=gp_red,
                    just=["right", "centre"],
                    x=Unit(-0.05, "inches"),
                    y=Unit(0.5, "npc"),
                    rot=0,
                )
            if i == nr:
                grid_text(
                    label=str(layout_widths(layout)),
                    gp=gp_red,
                    just=["centre", "top"],
                    x=Unit(0.5, "npc"),
                    y=Unit(-0.05, "inches"),
                    rot=0,
                )
            if j == nc:
                grid_text(
                    label=str(layout_heights(layout)),
                    gp=gp_red,
                    just=["left", "centre"],
                    x=Unit(1, "npc") + Unit(0.05, "inches"),
                    y=Unit(0.5, "npc"),
                    rot=0,
                )
            if i == 1:
                grid_text(
                    label=str(layout_widths(layout)),
                    gp=gp_red,
                    just=["centre", "bottom"],
                    x=Unit(0.5, "npc"),
                    y=Unit(1, "npc") + Unit(0.05, "inches"),
                    rot=0,
                )
            pop_viewport()

    pop_viewport()
    if vp is not None:
        pop_viewport()
    return vp_mid


def grid_show_viewport(
    v: Optional[Viewport] = None,
    parent_layout: Optional[GridLayout] = None,
    newpage: bool = True,
    vp_ex: float = 0.8,
    border_fill: str = "light grey",
    vp_col: str = "blue",
    vp_fill: str = "light blue",
    scale_col: str = "red",
    vp: Optional[Any] = None,
    recurse: bool = True,
    depth: int = 0,
) -> None:
    """Visualize a viewport (or viewport tree).

    Draws a representation of the viewport on the current device, showing
    its position, size, and native scale.

    Parameters
    ----------
    v : Viewport or None
        The viewport to visualize.  If ``None``, uses the current viewport.
    parent_layout : GridLayout or None
        The parent viewport's layout (used when *v* has layout position).
    newpage : bool
        If ``True``, start a new page.
    vp_ex : float
        Fraction of the page used for the outer viewport.
    border_fill : str
        Fill colour for the outer border area.
    vp_col : str
        Border colour for the viewport rectangle.
    vp_fill : str
        Fill colour for the viewport rectangle.
    scale_col : str
        Colour for scale annotations.
    vp : object or None
        Viewport to push before drawing.
    recurse : bool
        If ``True``, recurse into child viewports (not yet implemented).
    depth : int
        Current recursion depth.
    """
    if v is None:
        v = Viewport()

    # Check if viewport has layout position and parent layout
    has_pos = (getattr(v, "layout_pos_row", None) is not None or
               getattr(v, "layout_pos_col", None) is not None)
    if has_pos and parent_layout is not None:
        # Show within parent layout context
        if vp is not None:
            push_viewport(vp)
        vp_mid = grid_show_layout(
            parent_layout, vp_ex=vp_ex,
            cell_border="grey", cell_fill="white",
            cell_label=False, newpage=newpage,
        )
        push_viewport(vp_mid)
        push_viewport(v)
        gp_red = Gpar(col=scale_col)
        grid_rect(gp=Gpar(col="blue", fill="light blue"))
        xscale = getattr(v, "xscale", [0, 1])
        at = grid_pretty(xscale)
        if len(at) >= 2:
            grid_xaxis(at=[min(at), max(at)], gp=gp_red)
        yscale = getattr(v, "yscale", [0, 1])
        at = grid_pretty(yscale)
        if len(at) >= 2:
            grid_yaxis(at=[min(at), max(at)], gp=gp_red)
        pop_viewport(2)
        if vp is not None:
            pop_viewport()
    else:
        # Standard display
        if newpage:
            grid_newpage()
        if vp is not None:
            push_viewport(vp)
        grid_rect(gp=Gpar(col=None, fill=border_fill))
        vp_mid = Viewport(
            x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
            width=Unit(vp_ex, "npc"), height=Unit(vp_ex, "npc"),
        )
        push_viewport(vp_mid)
        grid_rect(gp=Gpar(fill="white"))
        push_viewport(v)
        grid_rect(gp=Gpar(col=vp_col, fill=vp_fill))
        gp_red = Gpar(col=scale_col)
        xscale = getattr(v, "xscale", [0, 1])
        at = grid_pretty(xscale)
        if len(at) >= 2:
            grid_xaxis(at=[min(at), max(at)], gp=gp_red)
        yscale = getattr(v, "yscale", [0, 1])
        at = grid_pretty(yscale)
        if len(at) >= 2:
            grid_yaxis(at=[min(at), max(at)], gp=gp_red)
        pop_viewport(2)
        if vp is not None:
            pop_viewport()


def grid_abline(
    intercept: float = 0,
    slope: float = 1,
    gp: Optional[Gpar] = None,
    draw: bool = True,
    name: Optional[str] = None,
    vp: Optional[Any] = None,
) -> Grob:
    """Draw a line from the equation ``y = intercept + slope * x``.

    The line is drawn across the full NPC range ``[0, 1]``, mapping the
    x-values 0 and 1 through the linear equation to obtain y-values.

    Parameters
    ----------
    intercept : float
        Y-intercept of the line.
    slope : float
        Slope of the line.
    gp : Gpar or None
        Graphical parameters.
    draw : bool
        If ``True``, draw immediately.
    name : str or None
        Grob name.
    vp : object or None
        Viewport.

    Returns
    -------
    Grob
        A lines grob representing the line.
    """
    x = [0, 1]
    y = [intercept + slope * xi for xi in x]
    g = lines_grob(
        x=Unit(x, "npc"),
        y=Unit(y, "npc"),
        gp=gp,
        name=name,
        vp=vp,
    )
    if draw:
        grid_draw(g)
    return g


def grid_plot_and_legend(
    plot_expr: Optional[Grob] = None,
    legend_expr: Optional[Grob] = None,
    widths: Optional[Any] = None,
    heights: Optional[Any] = None,
) -> None:
    """Layout a plot and legend side by side.

    This is a simple demonstration function that creates a frame with
    the plot on the left and the legend on the right.

    Parameters
    ----------
    plot_expr : Grob or None
        A grob (or GTree) for the main plot area.
    legend_expr : Grob or None
        A grob (or GTree) for the legend.
    widths : Unit or None
        Column widths (unused in simple version).
    heights : Unit or None
        Row heights (unused in simple version).
    """
    grid_newpage()
    top_vp = Viewport(width=Unit(0.8, "npc"), height=Unit(0.8, "npc"))
    push_viewport(top_vp)

    lf = frame_grob()
    if plot_expr is not None:
        lf = pack_grob(lf, plot_expr)
    if legend_expr is not None:
        lf = pack_grob(lf, legend_expr,
                        height=Unit(1, "null"), side="right")
    grid_draw(lf)


def layout_torture(
    n_row: int = 2,
    n_col: int = 2,
) -> None:
    """Stress-test the layout system with a simple grid of cells.

    Creates a layout with *n_row* rows and *n_col* columns, populates
    each cell with a labelled rectangle, and draws the result.

    Parameters
    ----------
    n_row : int
        Number of rows.
    n_col : int
        Number of columns.
    """
    grid_newpage()
    lay = GridLayout(nrow=n_row, ncol=n_col)
    top_vp = Viewport(layout=lay)
    push_viewport(top_vp)

    for i in range(1, n_row + 1):
        for j in range(1, n_col + 1):
            cell_vp = Viewport(layout_pos_row=i, layout_pos_col=j)
            push_viewport(cell_vp)
            grid_rect(gp=Gpar(col="blue", fill="light blue"))
            grid_text(label=f"({i}, {j})")
            pop_viewport()

    pop_viewport()


# ---------------------------------------------------------------------------
# Panel / strip / multipanel stubs (highlevel.R)
# ---------------------------------------------------------------------------


def grid_strip(
    label: str = "whatever",
    range_full: Sequence[float] = (0, 1),
    range_thumb: Sequence[float] = (0.3, 0.6),
    fill: str = "#FFBF00",
    thumb: str = "#FF8000",
    vp: Optional[Any] = None,
) -> None:
    """Draw a strip indicator (simple stub).

    Parameters
    ----------
    label : str
        Label text.
    range_full : sequence of float
        Full range.
    range_thumb : sequence of float
        Thumb (selected) range.
    fill : str
        Background fill colour.
    thumb : str
        Thumb fill colour.
    vp : object or None
        Viewport.
    """
    diff_full = range_full[1] - range_full[0]
    diff_thumb = range_thumb[1] - range_thumb[0]
    if vp is not None:
        push_viewport(vp)
    grid_rect(gp=Gpar(col=None, fill=fill))
    grid_rect(
        x=Unit((range_thumb[0] - range_full[0]) / diff_full, "npc"),
        y=Unit(0, "npc"),
        width=Unit(diff_thumb / diff_full, "npc"),
        height=Unit(1, "npc"),
        just=["left", "bottom"],
        gp=Gpar(col=None, fill=thumb),
    )
    grid_text(label=label)
    if vp is not None:
        pop_viewport()


def grid_panel(
    x: Optional[Sequence[float]] = None,
    y: Optional[Sequence[float]] = None,
    zrange: Sequence[float] = (0, 1),
    zbin: Optional[Sequence[float]] = None,
    xscale: Optional[Sequence[float]] = None,
    yscale: Optional[Sequence[float]] = None,
    axis_left: bool = True,
    axis_left_label: bool = True,
    axis_right: bool = False,
    axis_right_label: bool = True,
    axis_bottom: bool = True,
    axis_bottom_label: bool = True,
    axis_top: bool = False,
    axis_top_label: bool = True,
    vp: Optional[Any] = None,
) -> Dict[str, Viewport]:
    """Draw a panel with optional axes and strip (simple stub).

    Parameters
    ----------
    x : sequence of float or None
        X data values.
    y : sequence of float or None
        Y data values.
    zrange : sequence of float
        Full z-range for the strip.
    zbin : sequence of float or None
        Z-bin for the strip.
    xscale : sequence of float or None
        X-axis scale.
    yscale : sequence of float or None
        Y-axis scale.
    axis_left : bool
        Show left axis.
    axis_left_label : bool
        Show left axis labels.
    axis_right : bool
        Show right axis.
    axis_right_label : bool
        Show right axis labels.
    axis_bottom : bool
        Show bottom axis.
    axis_bottom_label : bool
        Show bottom axis labels.
    axis_top : bool
        Show top axis.
    axis_top_label : bool
        Show top axis labels.
    vp : object or None
        Viewport.

    Returns
    -------
    dict
        A dictionary with ``"strip_vp"`` and ``"plot_vp"`` keys.
    """
    if x is None:
        x = list(np.random.uniform(size=10))
    if y is None:
        y = list(np.random.uniform(size=10))
    if zbin is None:
        zbin = list(np.random.uniform(size=2))
    if xscale is None:
        xscale = list(_extend_range(x))
    if yscale is None:
        yscale = list(_extend_range(y))

    if vp is not None:
        push_viewport(vp)

    temp_vp = Viewport(
        layout=GridLayout(
            nrow=2, ncol=1,
            heights=Unit([1, 1], ["lines", "null"]),
        )
    )
    push_viewport(temp_vp)

    strip_vp = Viewport(layout_pos_row=1, layout_pos_col=1, xscale=xscale)
    push_viewport(strip_vp)
    grid_strip(range_full=zrange, range_thumb=zbin)
    grid_rect()
    if axis_top:
        grid_xaxis(main=False, label=axis_top_label)
    pop_viewport()

    plot_vp = Viewport(
        layout_pos_row=2, layout_pos_col=1,
        xscale=xscale, yscale=yscale,
    )
    push_viewport(plot_vp)
    grid_grill()
    grid_points(x=x, y=y, gp=Gpar(col="blue"))
    grid_rect()
    if axis_left:
        grid_yaxis(label=axis_left_label)
    if axis_right:
        grid_yaxis(main=False, label=axis_right_label)
    if axis_bottom:
        grid_xaxis(label=axis_bottom_label)
    pop_viewport(2)

    if vp is not None:
        pop_viewport()

    return {"strip_vp": strip_vp, "plot_vp": plot_vp}


def grid_multipanel(
    x: Optional[Sequence[float]] = None,
    y: Optional[Sequence[float]] = None,
    z: Optional[Sequence[float]] = None,
    nplots: int = 9,
    nrow: Optional[int] = None,
    ncol: Optional[int] = None,
    newpage: bool = True,
    vp: Optional[Any] = None,
) -> None:
    """Draw a multi-panel layout (simple stub).

    Parameters
    ----------
    x : sequence of float or None
        X data values.
    y : sequence of float or None
        Y data values.
    z : sequence of float or None
        Z data values used to split into panels.
    nplots : int
        Number of panels.
    nrow : int or None
        Number of rows (computed from *nplots* if ``None``).
    ncol : int or None
        Number of columns (computed from *nplots* if ``None``).
    newpage : bool
        If ``True``, start a new page.
    vp : object or None
        Viewport.
    """
    if x is None:
        x = list(np.random.uniform(size=90))
    if y is None:
        y = list(np.random.uniform(size=90))
    if z is None:
        z = list(np.random.uniform(size=90))

    if nplots < 1:
        raise ValueError("'nplots' must be >= 1")

    # Smart defaults for nrow/ncol
    if nrow is None or ncol is None:
        ncol_auto = max(1, math.ceil(math.sqrt(nplots)))
        nrow_auto = math.ceil(nplots / ncol_auto)
        if nrow is None:
            nrow = nrow_auto
        if ncol is None:
            ncol = ncol_auto

    if newpage:
        grid_newpage()
    if vp is not None:
        push_viewport(vp)

    temp_vp = Viewport(layout=GridLayout(nrow=nrow, ncol=ncol))
    push_viewport(temp_vp)

    xscale = list(_extend_range(x))
    yscale = list(_extend_range(y))
    breaks = list(np.linspace(min(z), max(z), nplots + 1))

    for i in range(nplots):
        col_idx = i % ncol + 1
        row_idx = i // ncol + 1
        panel_vp = Viewport(layout_pos_row=row_idx, layout_pos_col=col_idx)

        # Subset data
        mask = [(zv >= breaks[i] and zv <= breaks[i + 1]) for zv in z]
        panelx = [xv for xv, m in zip(x, mask) if m]
        panely = [yv for yv, m in zip(y, mask) if m]

        if len(panelx) == 0:
            panelx = [0.5]
            panely = [0.5]

        grid_panel(
            x=panelx, y=panely,
            zrange=[min(z), max(z)],
            zbin=[breaks[i], breaks[i + 1]],
            xscale=xscale, yscale=yscale,
            axis_left=(col_idx == 1),
            axis_right=(col_idx == ncol or i == nplots - 1),
            axis_bottom=(row_idx == nrow),
            axis_top=(row_idx == 1),
            axis_left_label=_is_even(row_idx),
            axis_right_label=_is_odd(row_idx),
            axis_bottom_label=_is_even(col_idx),
            axis_top_label=_is_odd(col_idx),
            vp=panel_vp,
        )

    pop_viewport()
    if vp is not None:
        pop_viewport()


# ---------------------------------------------------------------------------
# Top-level viewport helper
# ---------------------------------------------------------------------------


def grid_top_level_vp() -> Viewport:
    """Return a top-level viewport suitable for a standard multi-panel layout.

    This viewport occupies 80% of the device centred in the page, matching
    the common R pattern for demonstration plots.

    Returns
    -------
    Viewport
        A viewport with width and height of ``0.8 npc``.
    """
    return Viewport(
        width=Unit(0.8, "npc"),
        height=Unit(0.8, "npc"),
    )
