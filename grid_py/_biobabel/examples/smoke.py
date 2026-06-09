"""Smoke test for grid_py._biobabel.

Run by CI as `biobabel run-example`. This script must succeed without errors
and produce a smoke.png artifact.
"""

from __future__ import annotations

from pathlib import Path

from grid_py import (
    CairoRenderer,
    GridLayout,
    Unit,
    Viewport,
    get_state,
    gpar,
    grid_newpage,
    grid_rect,
    grid_text,
    pop_viewport,
    push_viewport,
)


def main(out_path: Path = Path("smoke.png")) -> Path:
    renderer = CairoRenderer(width=4, height=4, dpi=150, bg="white")
    get_state().init_device(renderer)
    grid_newpage()

    parent = Viewport(layout=GridLayout(nrow=2, ncol=2))
    push_viewport(parent)
    try:
        for row in (1, 2):
            for col in (1, 2):
                push_viewport(Viewport(layout_pos_row=row, layout_pos_col=col))
                try:
                    grid_rect(gp=gpar(fill="white", col="black"))
                    grid_text(
                        f"panel {row},{col}",
                        x=Unit(0.5, "npc"),
                        y=Unit(0.5, "npc"),
                    )
                finally:
                    pop_viewport()
    finally:
        pop_viewport()

    renderer.write_to_png(str(out_path))
    return out_path


if __name__ == "__main__":
    out = main()
    print(f"wrote {out.resolve()}")
