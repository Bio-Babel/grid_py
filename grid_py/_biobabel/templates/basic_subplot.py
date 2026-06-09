"""Recipe: draw a rectangle and a label inside a centered sub-viewport.

Demonstrates the canonical push_draw_pop idiom (`grid_py.push_draw_pop`).
"""

from __future__ import annotations

from pathlib import Path

from grid_py import (
    CairoRenderer,
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


def main(out_path: Path = Path("basic_subplot.png")) -> Path:
    renderer = CairoRenderer(width=4, height=4, dpi=150, bg="white")
    get_state().init_device(renderer)
    grid_newpage()

    push_viewport(Viewport(
        x=Unit(0.5, "npc"),
        y=Unit(0.5, "npc"),
        width=Unit(0.5, "npc"),
        height=Unit(0.5, "npc"),
    ))
    try:
        grid_rect(gp=gpar(fill="lightblue", col="navy"))
        grid_text("inside sub-viewport", x=Unit(0.5, "npc"), y=Unit(0.5, "npc"))
    finally:
        pop_viewport()

    renderer.write_to_png(str(out_path))
    return out_path


if __name__ == "__main__":
    print(f"wrote {main().resolve()}")
