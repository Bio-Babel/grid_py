"""Recipe: scatter plot with manual axes and threshold lines.

A canonical example of building a custom plot directly on grid primitives:
no ggplot2_py, no pheatmap, just viewports + Unit("native") + grobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grid_py import (
    CairoRenderer,
    Unit,
    Viewport,
    get_state,
    gpar,
    grid_lines,
    grid_newpage,
    grid_points,
    grid_xaxis,
    grid_yaxis,
    pop_viewport,
    push_viewport,
)


def main(out_path: Path = Path("scatter_with_thresholds.png")) -> Path:
    renderer = CairoRenderer(width=5, height=4, dpi=150, bg="white")
    get_state().init_device(renderer)
    grid_newpage()

    rng = np.random.default_rng(0)
    x = rng.normal(loc=0, scale=1.5, size=300)
    y = -np.log10(rng.uniform(1e-6, 1, size=300))

    plot_vp = Viewport(
        xscale=(float(x.min()) - 0.5, float(x.max()) + 0.5),
        yscale=(0, float(y.max()) + 0.5),
        x=Unit(0.55, "npc"),
        y=Unit(0.55, "npc"),
        width=Unit(0.78, "npc"),
        height=Unit(0.78, "npc"),
    )
    push_viewport(plot_vp)
    try:
        grid_points(
            x=Unit(x.tolist(), "native"),
            y=Unit(y.tolist(), "native"),
            gp=gpar(col="grey40"),
        )
        # Horizontal threshold at -log10(p) = 1.3 (i.e. p = 0.05)
        grid_lines(
            x=Unit([float(x.min()) - 0.5, float(x.max()) + 0.5], "native"),
            y=Unit([1.3, 1.3], "native"),
            gp=gpar(col="red", lwd=1.5, lty="dashed"),
        )
        grid_xaxis()
        grid_yaxis()
    finally:
        pop_viewport()

    renderer.write_to_png(str(out_path))
    return out_path


if __name__ == "__main__":
    print(f"wrote {main().resolve()}")
