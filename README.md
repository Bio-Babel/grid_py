# grid_py

Python port of the R **grid** package.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from grid_py import (
    CairoRenderer, Gpar, Unit, Viewport, GridLayout,
    get_state, grid_draw, grid_newpage,
    push_viewport, pop_viewport,
    rect_grob, text_grob, points_grob, circle_grob,
    unit_c, string_width,
)
```

## The Unit System — Why Layout "Just Works"

The central idea of grid is that **sizes are expressions, not numbers**.
A `Unit` carries both a value and a *strategy* for resolving that value.
Resolution is deferred until a viewport is pushed — at that point the parent
dimensions, font metrics, and device DPI are all known, so every unit can
evaluate itself in context.

This means the same layout specification produces correct results on a 72 dpi
screen, a 300 dpi PDF, a 7-inch plot or a 14-inch poster — with no manual
tweaking.

### Unit types at a glance

| Category | Units | Resolved from |
|----------|-------|---------------|
| **Absolute** | `"cm"`, `"inches"`, `"mm"`, `"points"` | Fixed physical conversion |
| **Relative** | `"npc"` (0-1 fraction of parent) | Parent viewport dimensions |
| **Font-relative** | `"lines"` (line height), `"char"` (char width) | Current `fontsize` × `lineheight` |
| **Content-measuring** | `"strwidth"`, `"strheight"` | Cairo text measurement |
| **Grob-measuring** | `"grobwidth"`, `"grobheight"` | Grob bounding box query |
| **Flex** | `"null"` | Remaining space (layout only) |
| **Data** | `"native"` | Viewport `xscale` / `yscale` mapping |

### Mixing units with arithmetic

Units of different types can be freely combined.  The result is a **compound
unit** that is evaluated recursively at resolve time:

```python
# "fill the parent, but leave 2 line-heights of margin on each side"
width = Unit(1, "npc") - Unit(4, "lines")

# "start 1 cm from the right edge"
x = Unit(1, "npc") - Unit(1, "cm")

# also supports min / max across types
from grid_py import unit_pmin
safe_width = unit_pmin(Unit(10, "cm"), Unit(1, "npc"))  # whichever is smaller
```

Internally `Unit(1,"npc") - Unit(4,"lines")` is stored as a tree:

```
sum ──┬── 1.0 npc
      └── -4.0 lines
```

When the viewport is pushed, each leaf is resolved to inches and summed.
Changing the font or the device size automatically changes the result.

## Layout Patterns

### Pattern 1 — Adaptive margins with `"lines"`

The most common pattern: margins that scale with the font.

```python
r = CairoRenderer(width=7, height=5, dpi=150)
get_state().init_device(r)
grid_newpage()

# Title: 2 line-heights tall, pinned to the top
title_vp = Viewport(
    name="title",
    x=Unit(0.5, "npc"),
    y=Unit(1, "npc") - Unit(1, "lines"),
    width=Unit(1, "npc"),
    height=Unit(2, "lines"),
)

# Plot area: fills the rest, with room for axis labels
plot_vp = Viewport(
    name="plot",
    x=Unit(0.5, "npc") + Unit(1, "lines"),
    y=Unit(0.5, "npc") - Unit(0.5, "lines"),
    width=Unit(1, "npc") - Unit(4, "lines"),
    height=Unit(1, "npc") - Unit(5, "lines"),
)
```

If you later change `Gpar(fontsize=14)` to `fontsize=20`, the margins grow
proportionally — no constants to update.

### Pattern 2 — Content-driven margins with `string_width`

Let the label measure itself:

```python
label = "Sepal Length (cm)"
margin = string_width(label)          # Unit whose value = rendered width of that string

plot_vp = Viewport(
    x=margin + Unit(0.5, "cm"),       # left edge = label width + gap
    width=Unit(1, "npc") - margin - Unit(1, "cm"),
    y=Unit(0.5, "npc"),
    height=Unit(1, "npc") - Unit(3, "lines"),
)
```

Change the label text → the margin updates automatically.

### Pattern 3 — Flexible grid with `"null"` units

`"null"` units divide **remaining space** proportionally, after all absolute
and relative units have been allocated.  This is how `GridLayout` works:

```python
# Column 1: fixed 3 cm (e.g. y-axis labels)
# Columns 2-3: split remaining space 2:1
layout = GridLayout(
    nrow=1, ncol=3,
    widths=unit_c(Unit(3, "cm"), Unit(2, "null"), Unit(1, "null")),
    heights=Unit([1], "null"),
)

main_vp = Viewport(name="main", layout=layout,
                   x=Unit(0.5, "npc"), y=Unit(0.5, "npc"),
                   width=Unit(1, "npc"), height=Unit(1, "npc"))
push_viewport(main_vp)

# Place children into cells
for col in [1, 2, 3]:
    cell_vp = Viewport(name=f"cell_{col}",
                       layout_pos_row=1, layout_pos_col=col)
    push_viewport(cell_vp)
    grid_draw(rect_grob(x=0.5, y=0.5, width=1, height=1,
                        gp=Gpar(fill="grey90", col="grey50")))
    pop_viewport(1)
```

Resize the device → column 1 stays 3 cm, the rest reflows.

### Pattern 4 — Nested viewports for complex figures

Viewports nest.  Each child resolves its units against its parent, so you
can build deeply structured layouts compositionally:

```python
# Outer: 2-row layout (title + body)
outer = GridLayout(nrow=2, ncol=1,
                   heights=unit_c(Unit(2, "lines"), Unit(1, "null")))

# Body: 1×3 panel grid
inner = GridLayout(nrow=1, ncol=3,
                   widths=Unit([1, 1, 1], "null"))

push_viewport(Viewport(name="page", layout=outer, ...))

# Row 1 — title
push_viewport(Viewport(layout_pos_row=1, layout_pos_col=1))
grid_draw(text_grob("My Title", x=0.5, y=0.5, gp=Gpar(fontsize=16, fontface="bold")))
pop_viewport(1)

# Row 2 — panels
push_viewport(Viewport(layout_pos_row=2, layout_pos_col=1, layout=inner))
for col in [1, 2, 3]:
    push_viewport(Viewport(layout_pos_row=1, layout_pos_col=col))
    # ... draw panel content using npc coordinates (0-1 within this cell)
    pop_viewport(1)
pop_viewport(1)

pop_viewport(1)
```

### The three-layer coordinate pipeline

Every coordinate in grid passes through three distinct transformations before
reaching the device.  Understanding these layers is the key to reasoning about
why layouts are portable and how viewports compose.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1:  Unit  →  Viewport-local inches                   │
│                                                             │
│  _transform_to_inches()  (_vp_calc.py)                      │
│                                                             │
│  Each unit type has its own rule:                           │
│    "npc"       →  value × parent_width_inches               │
│    "cm"        →  value / 2.54                              │
│    "lines"     →  value × fontsize × cex × lineheight / 72 │
│    "native"    →  map [scalemin, scalemax] → [0, parent]    │
│    "strwidth"  →  Cairo text_extents(string).width          │
│    "grobwidth" →  grob.width_details() recursive query      │
│    "sum"       →  Σ recursive resolve of child units        │
│    "null"      →  0  (only meaningful inside GridLayout)    │
│                                                             │
│  Result: a position or dimension in inches, local to the    │
│  current viewport's own coordinate system.                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2:  Viewport-local inches  →  Absolute inches        │
│                                                             │
│  transform_loc_to_device()  (_renderer_base.py)             │
│                                                             │
│  Applies the viewport's accumulated 3×3 affine transform:   │
│    T = Justification × Rotation × Translation × Parent_T   │
│                                                             │
│  Built at push_viewport() time by calc_viewport_transform() │
│  Each push multiplies into the parent's matrix, so nested   │
│  viewports compose naturally:                               │
│                                                             │
│    abs_loc = [x_inches, y_inches, 1] @ vp_transform_3x3    │
│                                                             │
│  Result: inches from device origin (bottom-left).           │
├─────────────────────────────────────────────────────────────┤
│  Layer 3:  Absolute inches  →  Device pixels                │
│                                                             │
│  inches_to_dev_x/y()  (_renderer_base.py)                   │
│                                                             │
│    dev_x = abs_inches_x × DPI                              │
│    dev_y = device_height - abs_inches_y × DPI   (Y-flip)   │
│                                                             │
│  Grid uses bottom-left origin; devices use top-left.        │
│  This layer bridges the two.                                │
└─────────────────────────────────────────────────────────────┘
```

A concrete example — drawing a point at `x = Unit(1,"npc") - Unit(2,"lines")`
inside a child viewport:

```
User writes:   Unit(1, "npc") - Unit(2, "lines") + Unit(0.5, "cm")
                         │
                         ▼  arithmetic builds a compound tree
Stored as:     sum ──┬── 1.0  npc
                     ├── -2.0 lines
                     └── 0.5  cm

  ── Layer 1 ─────────────────────────────────────────
                         │
                         ▼  _transform_to_inches(parent_context)
Resolve:       npc    → 1.0 × parent_width_inches       = 5.000"
               lines  → -2 × fontsize × lineheight / 72 = -0.333"
               cm     → 0.5 / 2.54                      = 0.197"
               sum    → 5.000 - 0.333 + 0.197           = 4.864"  (viewport-local)

  ── Layer 2 ─────────────────────────────────────────
                         │
                         ▼  [4.864, y, 1] @ viewport_transform_3x3
Absolute:      account for viewport position, rotation  = 5.214"  (from device origin)

  ── Layer 3 ─────────────────────────────────────────
                         │
                         ▼  × DPI, Y-flip
Device:        5.214 × 150 = 782 px
```

**Why this matters for users:**

- **Layer 1** is where your Unit expressions are evaluated.  Because each unit
  type knows how to measure itself (text metrics, parent size, font size),
  your layout adapts to context automatically.
- **Layer 2** is where viewport nesting works.  You never compute global
  positions — you work in local coordinates and the transform stack composes
  them.
- **Layer 3** is invisible to you.  It just makes sure the same inches
  produce correct pixels on any device.

The net effect: you describe layout in meaningful terms (`"lines"`, `"cm"`,
`"npc"`, `"strwidth"`), nest viewports freely, and the three-layer pipeline
ensures the result is correct on every device and at every DPI.

## Backend Architecture

grid\_py ships with a pluggable rendering backend system built on a single
abstract base class (`GridRenderer`). User code is identical across backends —
swap the renderer instance to switch output formats.

```
                      GridRenderer (ABC)
                 ┌──────────────────────────┐
                 │  viewport transform stack │
                 │  unit resolution          │
                 │  coordinate system mgmt   │
                 │  29 abstract methods      │
                 └────────────┬─────────────┘
                        ┌─────┴──────┐
                        │            │
                CairoRenderer   WebRenderer
                (immediate)     (scene graph)
                ┌──────────┐   ┌──────────────┐
                │ PNG/PDF/ │   │ JSON Scene → │
                │ SVG/PS   │   │ HTML + D3.js │
                └──────────┘   └──────────────┘
```

### Renderers

| Backend | Module | Output Formats | Rendering Mode |
|---------|--------|----------------|----------------|
| **CairoRenderer** | `renderer.py` | PNG, PDF, SVG, PS | Immediate — draws directly to a pycairo surface |
| **WebRenderer** | `renderer_web.py` | Standalone HTML (SVG + Canvas + D3.js) | Deferred — builds a JSON scene graph, rendered browser-side |

### Supporting Components

| Component | Module | Role |
|-----------|--------|------|
| GridRenderer (ABC) | `_renderer_base.py` | Abstract base class defining the renderer interface (29 abstract methods) |
| Scene Graph | `_scene_graph.py` | `SceneNode` / `ViewportNode` / `GrobNode` tree used by WebRenderer |
| Font Metrics | `_font_metrics.py` | Pluggable text measurement (Cairo, fonttools, or heuristic backends) |
| GridState | `_state.py` | Global singleton that binds the active renderer to the drawing API |

### Backend Selection

There is no registry or factory — instantiate the renderer you need and bind it:

```python
from grid_py import WebRenderer, get_state, grid_draw

r = WebRenderer(width=7, height=5, dpi=100)
state = get_state()
state.init_device(r)

grid_draw(my_grob)          # draws into the scene graph
html = r.to_html()          # export interactive HTML
```

`grid_newpage()` creates a `CairoRenderer` by default when no renderer is bound.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```
