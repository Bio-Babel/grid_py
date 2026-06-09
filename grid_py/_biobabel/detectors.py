"""AST detectors registered into biobabel via the ``biobabel.detectors``
entry-point group (declared in this package's pyproject.toml).

These three detectors used to live inside biobabel core as a hard-coded
``_DETECTORS`` dispatch table driven by a mini-DSL in YAML. Schema v2 /
ADR-0008 moved them here so the package that owns the domain knowledge
(``grid_py``) also owns the detection code. biobabel core is now generic.

Each detector follows the :data:`biobabel.detector_api.DetectorFn`
signature: ``(tree: ast.AST, args: dict[str, Any]) -> list[DetectorMatch]``.
"""

from __future__ import annotations

import ast
from typing import Any

from biobabel.detector_api import DetectorMatch


def for_loop_calls(tree: ast.AST, args: dict[str, Any]) -> list[DetectorMatch]:
    """Flag every ``for`` body that directly calls one of ``args["calls"]``.

    Used by ``grid_py.grob_in_loop``: drawing one grob at a time inside a
    loop incurs one device round-trip per grob, vs building a grob tree
    and drawing once. A loop whose body also calls any name in
    ``args["unless_calls"]`` is skipped: pushing a viewport per iteration is
    the legitimate multi-panel / nested-viewport idiom, not the anti-pattern.
    """
    targets = set(args.get("calls", []))
    unless = set(args.get("unless_calls", []))
    hits: list[DetectorMatch] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            body_calls = {
                _call_name(c.func)
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
            }
            # A loop that pushes its own viewport per iteration (multi-panel /
            # nested-viewport idiom) is legitimate, not the anti-pattern.
            if unless & body_calls:
                continue
            hit = targets & body_calls
            if hit:
                hits.append(
                    DetectorMatch(line=node.lineno, detail={"target_call": sorted(hit)[0]})
                )
    return hits


def unbalanced(tree: ast.AST, args: dict[str, Any]) -> list[DetectorMatch]:
    """Flag module/function scopes where push-call count != pop-call count.

    Used by ``grid_py.unbalanced_push_pop``: every ``push_viewport`` must
    be matched by exactly one ``pop_viewport``; counts diverging at the
    static level is a strong indicator of a leaked viewport stack.
    """
    push_fn = args.get("push", "")
    pop_fn = args.get("pop", "")
    if not push_fn or not pop_fn:
        return []

    push_count = 0
    pop_count = 0
    first_push_line = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == push_fn:
                push_count += 1
                if first_push_line == 0:
                    first_push_line = node.lineno
            elif name == pop_fn:
                pop_count += 1

    if push_count != pop_count:
        return [
            DetectorMatch(
                line=first_push_line or 1,
                detail={
                    "push_count": push_count,
                    "pop_count": pop_count,
                    "diff": push_count - pop_count,
                },
            )
        ]
    return []


def unit_kw(tree: ast.AST, args: dict[str, Any]) -> list[DetectorMatch]:
    """Flag ``Unit(value, "npc"/"snpc")`` used with a data-like argument.

    Heuristic: positional or keyword ``units="npc"|"snpc"`` to a call to
    ``Unit`` whose value-arg names one of the data-context hint names
    (``df``, ``data``, ``values``, ``obs``). Used by
    ``grid_py.npc_units_for_data``.
    """
    bad_units = set(args.get("bad_units", ["npc", "snpc"]))
    data_hints = set(args.get("data_hints", ["df", "data", "values", "obs"]))
    hits: list[DetectorMatch] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == "Unit":
            units_value = _extract_unit_kw(node)
            if units_value not in bad_units:
                continue
            first_arg = node.args[0] if node.args else None
            if first_arg is None:
                continue
            referenced = _names_in(first_arg)
            if referenced & data_hints:
                hits.append(
                    DetectorMatch(
                        line=node.lineno,
                        detail={"units": units_value, "referenced": sorted(referenced)},
                    )
                )
    return hits


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _extract_unit_kw(call: ast.Call) -> str:
    for kw in call.keywords:
        if (
            kw.arg == "units"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    if len(call.args) >= 2:
        a = call.args[1]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
    return ""


def _names_in(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
    return out
