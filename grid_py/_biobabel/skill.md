---
name: use-grid-py
description: How to write idiomatic grid_py — Python port of R's grid graphics — when building plotting packages or custom multi-panel layouts.
---

# grid_py — R grid for Python

`grid_py` is a 1:1 port of R's `grid` package, the low-level graphics layer that underpins `ggplot2`, `pheatmap`, `ComplexHeatmap`, and friends.

## Mental model in 30 seconds

1. **Viewport** — a rectangular region on the device with its own coordinate system. Viewports form a *stack*. The topmost viewport is the "current" one. All units of `"npc"` (normalized parent coordinates) are resolved against it.

2. **Grob** — a graphical object. `rect_grob(...)`, `text_grob(...)`, `circle_grob(...)`, `lines_grob(...)`, `polygon_grob(...)`. A `GTree` is a grob containing children. You compose grobs with `grob_tree(child1, child2, ...)`.

3. **Unit** — a value plus a unit kind (`"npc"`, `"native"`, `"cm"`, `"inches"`, `"mm"`, ...). Construct with `Unit(0.5, "npc")` (no lowercase `unit()` factory; this is the only R-API gap).

4. **Gpar** — graphical parameters (color, fill, lwd, fontsize, ...). Build via `gpar(col="red", fill="lightblue", lwd=2)` and attach to grobs.

5. **Drawing** = push viewport → emit grobs → pop viewport. The act of *drawing* (`grid_draw`) renders a grob; the act of *building* a grob (`rect_grob`) does not.

## The two cardinal sins

1. **Building grobs inside a tight loop and `grid_draw`-ing each one separately** — every `grid_draw` triggers a device flush. Build a list of grobs, wrap with `grob_tree(*grobs)`, draw once. See anti-pattern `grid_py.grob_in_loop`.

2. **Imbalanced `push_viewport` / `pop_viewport`** — you must pop everything you push. If you raise mid-block, wrap in `try/finally` (idiom `grid_py.try_finally_pop`).

## The two cardinal idioms

- **Push-Draw-Pop**: `push_viewport(vp); grid_draw(grob); pop_viewport()` is the basic motif for drawing inside a specific region.
- **Build-then-draw**: assemble a `grob_tree(...)` first, draw last. This keeps your code referentially transparent and makes it trivially testable.

## When to reach for grid_py vs alternatives

| You want                                | Reach for                       |
|-----------------------------------------|---------------------------------|
| A one-liner statistical plot            | `ggplot2_py`                    |
| A heatmap                               | `pheatmap_py` / `ComplexHeatmap_py` |
| A new plot *type* not in ggplot2_py     | extend ggplot2_py geom (use `scaffold(kind=geom)`) |
| A new plotting *package* end-to-end     | `grid_py` + `scaffold(kind=package, base=grid_py)` |
| Sub-region annotation on existing plots | `grid_py` directly              |

## Quick reference

```python
import grid_py
from grid_py import Unit, gpar, push_viewport, pop_viewport, Viewport

# Idiomatic: build a grob tree, then draw once.
rects = [
    grid_py.rect_grob(x=Unit(i * 0.25 + 0.125, "npc"),
                      width=Unit(0.2, "npc"),
                      gp=gpar(fill="lightblue"))
    for i in range(4)
]
grid_py.grid_draw(grid_py.grob_tree(*rects))
```

For more, run `biobabel.list_idioms(package="grid_py")`.
