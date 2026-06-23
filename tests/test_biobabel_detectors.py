"""Unit tests for the grid_py AST anti-pattern detectors.

These import the REAL detector callables registered into biobabel via the
``biobabel.detectors`` entry-point group (see this package's pyproject.toml)
and exercise each one with positive (anti-pattern present) and negative
(clean / suppressed) cases. The ``args`` passed to every detector mirror the
exact config its corresponding ``grid_py/_biobabel/anti_patterns/*.yaml`` rule
declares, so the tests reflect real production configuration.
"""

from __future__ import annotations

import ast

from grid_py._biobabel.detectors import (
    captured_stale_renderer,
    for_loop_calls,
    unbalanced,
    unit_kw,
)


# ---------------------------------------------------------------------------
# captured_stale_renderer  (rule: grid_py.stale_renderer_write)
# ---------------------------------------------------------------------------

_STALE_ARGS = {
    "ctors": ["CairoRenderer", "WebRenderer"],
    "newpage": "grid_newpage",
    "write_methods": ["write_to_png", "to_png_bytes", "save", "finish"],
}


def test_captured_stale_renderer_pos_with_sizing_args():
    src = (
        "r = CairoRenderer(width=8, height=6)\n"
        "get_state().init_device(r)\n"
        "grid_newpage(width=8, height=6)\n"
        "grid_draw(build_layout())\n"
        'r.write_to_png("p.png")\n'
    )
    hits = captured_stale_renderer(ast.parse(src), _STALE_ARGS)
    assert len(hits) == 1
    assert hits[0].detail["renderer_var"] == "r"
    assert hits[0].detail["written_after_newpage"] is True


def test_captured_stale_renderer_pos_bare_newpage():
    src = (
        "r = CairoRenderer(width=8, height=6)\n"
        "get_state().init_device(r)\n"
        "grid_newpage()\n"
        "grid_draw(build_layout())\n"
        'r.write_to_png("p.png")\n'
    )
    hits = captured_stale_renderer(ast.parse(src), _STALE_ARGS)
    assert len(hits) == 1
    assert hits[0].detail["renderer_var"] == "r"


def test_captured_stale_renderer_neg_finalize_bound():
    src = (
        "r = CairoRenderer(width=8, height=6)\n"
        "get_state().init_device(r)\n"
        "grid_newpage()\n"
        "grid_draw(build_layout())\n"
        'get_state().get_renderer().write_to_png("p.png")\n'
    )
    assert captured_stale_renderer(ast.parse(src), _STALE_ARGS) == []


def test_captured_stale_renderer_neg_write_before_newpage():
    src = (
        "r = CairoRenderer(width=8, height=6)\n"
        "get_state().init_device(r)\n"
        'r.write_to_png("p.png")\n'
        "grid_newpage()\n"
    )
    assert captured_stale_renderer(ast.parse(src), _STALE_ARGS) == []


def test_captured_stale_renderer_neg_no_newpage_between():
    src = (
        "r = CairoRenderer(width=8, height=6)\n"
        "get_state().init_device(r)\n"
        "grid_draw(build_layout())\n"
        'r.write_to_png("p.png")\n'
    )
    assert captured_stale_renderer(ast.parse(src), _STALE_ARGS) == []


# ---------------------------------------------------------------------------
# for_loop_calls  (rule: grid_py.grob_in_loop)
# ---------------------------------------------------------------------------

_FOR_ARGS = {"calls": ["rect_grob"], "unless_calls": ["push_viewport"]}


def test_for_loop_calls_pos():
    src = (
        "for i in range(100):\n"
        '    grid_draw(rect_grob(x=Unit(i * 0.01, "npc")))\n'
    )
    hits = for_loop_calls(ast.parse(src), _FOR_ARGS)
    assert len(hits) == 1
    assert hits[0].detail["target_call"] == "rect_grob"


def test_for_loop_calls_neg_push_viewport_suppresses():
    src = (
        "for i in range(100):\n"
        "    push_viewport(vp)\n"
        "    grid_draw(rect_grob())\n"
        "    pop_viewport()\n"
    )
    assert for_loop_calls(ast.parse(src), _FOR_ARGS) == []


def test_for_loop_calls_neg_list_comprehension():
    src = "rects = [rect_grob(x=i) for i in range(100)]\n"
    assert for_loop_calls(ast.parse(src), _FOR_ARGS) == []


# ---------------------------------------------------------------------------
# unbalanced  (rule: grid_py.unbalanced_push_pop)
# ---------------------------------------------------------------------------

_UNBALANCED_ARGS = {"push": "push_viewport", "pop": "pop_viewport"}


def test_unbalanced_pos():
    src = (
        "push_viewport(a)\n"
        "push_viewport(b)\n"
        "grid_rect()\n"
        "pop_viewport()\n"
    )
    hits = unbalanced(ast.parse(src), _UNBALANCED_ARGS)
    assert len(hits) == 1
    assert hits[0].detail["push_count"] == 2
    assert hits[0].detail["pop_count"] == 1
    assert hits[0].detail["diff"] == 1


def test_unbalanced_neg_balanced_try_finally():
    src = (
        "push_viewport(vp)\n"
        "try:\n"
        "    grid_rect()\n"
        "finally:\n"
        "    pop_viewport()\n"
    )
    assert unbalanced(ast.parse(src), _UNBALANCED_ARGS) == []


# ---------------------------------------------------------------------------
# unit_kw  (rule: grid_py.npc_units_for_data)
# ---------------------------------------------------------------------------

_UNIT_ARGS = {"bad_units": ["npc", "snpc"], "data_hints": ["df", "data", "values", "obs"]}


def test_unit_kw_pos_data_hint():
    src = 'grid_points(x=Unit(df["x"].tolist(), "npc"))\n'
    hits = unit_kw(ast.parse(src), _UNIT_ARGS)
    assert len(hits) == 1
    assert hits[0].detail["units"] == "npc"
    assert "df" in hits[0].detail["referenced"]


def test_unit_kw_neg_literal_value_with_npc():
    src = 'grid_rect(x=Unit(0.5, "npc"))\n'
    assert unit_kw(ast.parse(src), _UNIT_ARGS) == []


def test_unit_kw_neg_good_units():
    src = 'grid_points(x=Unit(df["x"].tolist(), "native"))\n'
    assert unit_kw(ast.parse(src), _UNIT_ARGS) == []
