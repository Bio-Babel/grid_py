# grid_py

Python port of the R **grid** graphics package.

`grid_py` provides a complete reimplementation of R's grid graphics system in Python, using matplotlib as the rendering backend. It includes the full grid API: units, viewports, grobs (graphical objects), layouts, graphical parameters, coordinate queries, patterns/gradients, masking, clipping, grouping/compositing, and more.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import matplotlib
matplotlib.use("Agg")
import grid_py as gp

# Create a new page
gp.grid_newpage()

# Draw a rectangle
gp.grid_rect(gp=gp.Gpar(col="blue", fill="lightblue"))

# Draw text
gp.grid_text("Hello grid_py!", gp=gp.Gpar(fontsize=20))
```

## Key Concepts

### Units
Units are the foundation of grid's coordinate system:
```python
u1 = gp.Unit(1, "cm")
u2 = gp.Unit(0.5, "npc")  # normalized parent coordinates
u3 = u1 + u2               # compound unit
```

### Viewports
Viewports define rectangular regions for drawing:
```python
vp = gp.Viewport(x=0.5, y=0.5, width=0.8, height=0.8,
                  xscale=[0, 10], yscale=[0, 100])
gp.push_viewport(vp)
```

### Grobs
Graphical objects (grobs) represent drawing elements:
```python
r = gp.rect_grob(gp=gp.Gpar(fill="red"))
gp.grid_draw(r)
```

### Layouts
Layouts divide viewports into rows and columns:
```python
lay = gp.GridLayout(nrow=2, ncol=3)
vp = gp.Viewport(layout=lay)
```

## Tutorials

- [Introduction to grid_py](tutorials/grid.ipynb)
- [Working with Grobs](tutorials/grobs.ipynb)
- [Working with Viewports](tutorials/viewports.ipynb)
- [Locations vs Dimensions](tutorials/locndimn.ipynb)
- [Frames and Packing](tutorials/frame.ipynb)
- [The Display List](tutorials/displaylist.ipynb)
- [Building a Plot](tutorials/plotexample.ipynb)
- [Interactive Editing](tutorials/interactive.ipynb)
- [Move-To and Line-To](tutorials/moveline.ipynb)
- [Non-Finite Values](tutorials/nonfinite.ipynb)
- [Rotated Viewports](tutorials/rotated.ipynb)
- [Saving and Loading](tutorials/saveload.ipynb)
- [Sharing Parameters](tutorials/sharing.ipynb)

## API Reference

See the [API Reference](api.md) for full documentation of all 244 public symbols.
