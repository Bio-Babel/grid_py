"""Display-list edit operations for grid_py (port of R's grid edit/get/set/add/remove).

This module provides the ``grid_*`` family of functions that operate on the
current display list, mirroring the R functions ``grid.edit``, ``grid.get``,
``grid.set``, ``grid.add``, ``grid.remove``, and their ``grep``-defaulting
convenience aliases ``grid.gedit``, ``grid.gget``, ``grid.gremove``.

These functions locate grobs on the display list by :class:`~._path.GPath`
and apply modifications (edit attributes, replace, add children, or remove
grobs) with optional regex matching.

References
----------
R source: ``src/library/grid/R/edit.R``, ``src/library/grid/R/grob.R``
"""

from __future__ import annotations

import copy
import re
import warnings
from typing import Any, Dict, List, Optional, Union

from ._grob import (
    GEdit,
    GEditList,
    GList,
    GTree,
    Grob,
    apply_edit,
    apply_edits,
    edit_grob,
    is_grob,
)
from ._path import GPath
from ._display_list import DisplayList, DLDrawGrob
from ._state import get_state

__all__ = [
    "grid_edit",
    "grid_get",
    "grid_set",
    "grid_add",
    "grid_remove",
    "grid_gedit",
    "grid_gget",
    "grid_gremove",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_gpath(gpath: Union[str, GPath]) -> GPath:
    """Convert *gpath* to a :class:`GPath` if it is a string.

    Parameters
    ----------
    gpath : str or GPath
        A grob path specification.

    Returns
    -------
    GPath

    Raises
    ------
    TypeError
        If *gpath* is neither a string nor a :class:`GPath`.
    """
    if isinstance(gpath, str):
        return GPath(gpath)
    if isinstance(gpath, GPath):
        return gpath
    raise TypeError(f"invalid gPath: expected str or GPath, got {type(gpath).__name__}")


def _name_match(pattern: str, candidates: List[str], grep: bool) -> Optional[int]:
    """Return the index (0-based) of the first matching candidate, or ``None``.

    Parameters
    ----------
    pattern : str
        The name or regex pattern to match.
    candidates : list[str]
        The candidate names to search.
    grep : bool
        If ``True``, use regex matching; otherwise, exact matching.

    Returns
    -------
    int or None
        The 0-based index of the first match, or ``None`` if no match.
    """
    for i, name in enumerate(candidates):
        if grep:
            if re.search(pattern, name):
                return i
        else:
            if pattern == name:
                return i
    return None


def _find_dl_grobs(
    dl: DisplayList,
    gpath: GPath,
    strict: bool,
    grep: Union[bool, List[bool]],
    global_: bool,
) -> List[tuple]:
    """Find grob(s) on the display list matching *gpath*.

    Parameters
    ----------
    dl : DisplayList
        The display list to search.
    gpath : GPath
        Path identifying the target grob(s).
    strict : bool
        If ``True``, the path must match exactly from the root of each grob.
    grep : bool or list[bool]
        Whether to use regex matching at each path level.
    global_ : bool
        If ``True``, find all matches; otherwise, stop at first match.

    Returns
    -------
    list[tuple]
        Each tuple is ``(dl_index, grob_reference)`` for a matching grob.
    """
    if isinstance(grep, bool):
        grep_flags = [grep] * gpath.n
    else:
        grep_flags = list(grep)
        while len(grep_flags) < gpath.n:
            grep_flags.append(grep_flags[-1] if grep_flags else False)

    matches: list[tuple] = []

    for idx, item in enumerate(dl):
        if not isinstance(item, DLDrawGrob) or item.grob is None:
            continue
        grob = item.grob
        found = _match_grob_path(grob, gpath, strict, grep_flags)
        if found is not None:
            matches.append((idx, found))
            if not global_:
                break

    return matches


def _match_grob_path(
    grob: Any,
    gpath: GPath,
    strict: bool,
    grep_flags: List[bool],
) -> Optional[Any]:
    """Attempt to match *gpath* against *grob* and its children.

    Parameters
    ----------
    grob : Grob
        The grob to match against.
    gpath : GPath
        The target path.
    strict : bool
        Require exact depth matching.
    grep_flags : list[bool]
        Per-level regex flags.

    Returns
    -------
    Grob or None
        The matched grob, or ``None`` if no match.
    """
    if not is_grob(grob):
        return None

    components = gpath.components

    # Single-level path: match grob name directly
    if len(components) == 1:
        pattern = components[0]
        use_grep = grep_flags[0]
        if use_grep:
            if re.search(pattern, grob.name):
                return grob
        else:
            if pattern == grob.name:
                return grob
        # If not strict, search children recursively
        if not strict and isinstance(grob, GTree):
            for child_name in grob._children_order:
                child = grob._children[child_name]
                result = _match_grob_path(child, gpath, strict, grep_flags)
                if result is not None:
                    return result
        return None

    # Multi-level path: first component must match this grob (or a descendant if not strict)
    pattern = components[0]
    use_grep = grep_flags[0]
    rest_path = GPath(*components[1:])
    rest_flags = grep_flags[1:]

    matched_here = False
    if use_grep:
        matched_here = bool(re.search(pattern, grob.name))
    else:
        matched_here = pattern == grob.name

    if matched_here and isinstance(grob, GTree):
        for child_name in grob._children_order:
            child = grob._children[child_name]
            result = _match_grob_path(child, rest_path, strict, rest_flags)
            if result is not None:
                return result

    # If not strict, skip this level and try children with the full path
    if not strict and isinstance(grob, GTree):
        for child_name in grob._children_order:
            child = grob._children[child_name]
            result = _match_grob_path(child, gpath, strict, grep_flags)
            if result is not None:
                return result

    return None


def _redraw_display_list() -> None:
    """Replay the display list to refresh the output.

    Calls :meth:`DisplayList.replay` on the current state's display list.
    """
    state = get_state()
    dl = state.display_list
    if hasattr(dl, "replay"):
        dl.replay(state)


# ---------------------------------------------------------------------------
# Public API -- grid.edit / grid.gedit
# ---------------------------------------------------------------------------


def grid_edit(
    gPath: Union[str, GPath],
    *,
    strict: bool = False,
    grep: bool = False,
    global_: bool = False,
    allDevices: bool = False,
    redraw: bool = True,
    **kwargs: Any,
) -> None:
    """Edit grob(s) on the display list matching *gPath*.

    Locates grobs whose names match *gPath* on the current display list and
    applies the attribute changes specified in *kwargs*.  This is the Python
    equivalent of R's ``grid.edit()``.

    Parameters
    ----------
    gPath : str or GPath
        A grob-path specification identifying the grob(s) to edit.
    strict : bool, optional
        If ``True``, the path must match exactly from the root.
    grep : bool, optional
        If ``True``, path components are treated as regex patterns.
    global_ : bool, optional
        If ``True``, edit *all* matching grobs; otherwise only the first.
    allDevices : bool, optional
        If ``True``, apply the edit to all open devices.  Not yet
        implemented -- raises :class:`NotImplementedError`.
    redraw : bool, optional
        If ``True`` (default), redraw after the edit.
    **kwargs
        Attribute name-value pairs to apply to the matched grob(s).

    Raises
    ------
    NotImplementedError
        If *allDevices* is ``True``.
    TypeError
        If *gPath* is invalid.
    """
    if allDevices:
        raise NotImplementedError("allDevices is not yet implemented")

    gpath = _ensure_gpath(gPath)

    if not isinstance(grep, bool):
        raise TypeError("invalid 'grep' value")
    grep_flags = [grep] * gpath.n

    state = get_state()
    dl = state.display_list

    matches = _find_dl_grobs(dl, gpath, strict, grep_flags, global_)
    for dl_idx, matched_grob in matches:
        for key, value in kwargs.items():
            if hasattr(matched_grob, key) or hasattr(matched_grob, f"_{key}"):
                setattr(matched_grob, key, value)
            elif key == "gp":
                matched_grob._gp = value
            elif key == "name":
                matched_grob._name = str(value) if value is not None else None
            elif key == "vp":
                matched_grob._vp = Grob._check_vp(value)
            else:
                warnings.warn(f"slot '{key}' not found on grob", stacklevel=2)

    if redraw and matches:
        _redraw_display_list()


def grid_gedit(
    gPath: Union[str, GPath],
    *,
    strict: bool = False,
    global_: bool = True,
    allDevices: bool = False,
    redraw: bool = True,
    **kwargs: Any,
) -> None:
    """Edit grobs on the display list with ``grep=True`` (convenience alias).

    This is the Python equivalent of R's ``grid.gedit()``, which is
    ``grid.edit()`` with ``grep=True`` and ``global_=True`` as defaults.

    Parameters
    ----------
    gPath : str or GPath
        A grob-path specification (regex matching enabled).
    strict : bool, optional
        If ``True``, require exact depth matching.
    global_ : bool, optional
        If ``True`` (default), edit all matching grobs.
    allDevices : bool, optional
        Not yet implemented.
    redraw : bool, optional
        If ``True`` (default), redraw after the edit.
    **kwargs
        Attribute name-value pairs to apply.
    """
    grid_edit(
        gPath,
        strict=strict,
        grep=True,
        global_=global_,
        allDevices=allDevices,
        redraw=redraw,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public API -- grid.get / grid.gget
# ---------------------------------------------------------------------------


def grid_get(
    gPath: Union[str, GPath],
    strict: bool = False,
    grep: bool = False,
    global_: bool = False,
    allDevices: bool = False,
) -> Union[Grob, List[Grob], None]:
    """Get grob(s) from the display list matching *gPath*.

    Returns the grob(s) whose names match *gPath*.  This is the Python
    equivalent of R's ``grid.get()``.

    Parameters
    ----------
    gPath : str or GPath
        A grob-path specification.
    strict : bool, optional
        If ``True``, require exact depth matching.
    grep : bool, optional
        If ``True``, use regex matching on path components.
    global_ : bool, optional
        If ``True``, return a list of all matching grobs; otherwise,
        return the first match (or ``None``).
    allDevices : bool, optional
        Not yet implemented.

    Returns
    -------
    Grob or list[Grob] or None
        The matched grob(s).  If *global_* is ``False``, a single
        :class:`~._grob.Grob` or ``None``.  If *global_* is ``True``,
        a list (possibly empty).

    Raises
    ------
    NotImplementedError
        If *allDevices* is ``True``.
    TypeError
        If *gPath* is invalid or *grep* is not boolean.
    """
    if allDevices:
        raise NotImplementedError("allDevices is not yet implemented")

    gpath = _ensure_gpath(gPath)

    if not isinstance(grep, bool):
        raise TypeError("invalid 'grep' value")
    grep_flags = [grep] * gpath.n

    state = get_state()
    dl = state.display_list

    matches = _find_dl_grobs(dl, gpath, strict, grep_flags, global_)

    if global_:
        return [copy.deepcopy(m[1]) for m in matches]
    if matches:
        return copy.deepcopy(matches[0][1])
    return None


def grid_gget(
    gPath: Union[str, GPath],
    strict: bool = False,
    allDevices: bool = False,
) -> Union[Grob, List[Grob], None]:
    """Get grobs from the display list with ``grep=True`` (convenience alias).

    This is the Python equivalent of R's ``grid.gget()``, which is
    ``grid.get()`` with ``grep=True`` and ``global_=True`` as defaults.

    Parameters
    ----------
    gPath : str or GPath
        A grob-path specification (regex matching enabled).
    strict : bool, optional
        If ``True``, require exact depth matching.
    allDevices : bool, optional
        Not yet implemented.

    Returns
    -------
    Grob or list[Grob] or None
        The matched grob(s).
    """
    return grid_get(
        gPath,
        strict=strict,
        grep=True,
        global_=True,
        allDevices=allDevices,
    )


# ---------------------------------------------------------------------------
# Public API -- grid.set
# ---------------------------------------------------------------------------


def grid_set(
    gPath: Union[str, GPath],
    newGrob: Grob,
    strict: bool = False,
    grep: bool = False,
    redraw: bool = True,
) -> None:
    """Set (replace) a grob on the display list.

    Locates the first grob matching *gPath* on the display list and replaces
    it with *newGrob*.  This is the Python equivalent of R's ``grid.set()``.

    Parameters
    ----------
    gPath : str or GPath
        A grob-path specification identifying the grob to replace.
    newGrob : Grob
        The replacement grob.
    strict : bool, optional
        If ``True``, require exact depth matching.
    grep : bool, optional
        If ``True``, use regex matching on path components.
    redraw : bool, optional
        If ``True`` (default), redraw after the replacement.

    Raises
    ------
    TypeError
        If *gPath* is invalid or *grep* is not boolean.
    ValueError
        If *gPath* does not match any grob on the display list.
    """
    gpath = _ensure_gpath(gPath)

    if not isinstance(grep, bool):
        raise TypeError("invalid 'grep' value")
    grep_flags = [grep] * gpath.n

    state = get_state()
    dl = state.display_list

    matches = _find_dl_grobs(dl, gpath, strict, grep_flags, False)
    if not matches:
        raise ValueError("gPath does not specify a valid child")

    dl_idx, _old = matches[0]
    # Replace the grob in the display list item
    item = dl[dl_idx]
    if isinstance(item, DLDrawGrob):
        item.grob = newGrob
        item.params["grob"] = newGrob

    if redraw:
        _redraw_display_list()


# ---------------------------------------------------------------------------
# Public API -- grid.add
# ---------------------------------------------------------------------------


def grid_add(
    grob: Grob,
    gPath: Optional[Union[str, GPath]] = None,
    strict: bool = False,
    grep: bool = False,
    global_: bool = False,
    allDevices: bool = False,
    redraw: bool = True,
) -> None:
    """Add a grob to a gTree on the display list.

    If *gPath* is ``None``, *grob* is appended as a new top-level entry on
    the display list.  Otherwise, the grob is added as a child of the
    gTree identified by *gPath*.  This is the Python equivalent of R's
    ``grid.add()``.

    Parameters
    ----------
    grob : Grob
        The grob to add.
    gPath : str, GPath, or None, optional
        Path to the parent gTree.  ``None`` adds to the top level.
    strict : bool, optional
        If ``True``, require exact depth matching.
    grep : bool, optional
        If ``True``, use regex matching on path components.
    global_ : bool, optional
        If ``True``, add to all matching gTrees; otherwise only the first.
    allDevices : bool, optional
        Not yet implemented.
    redraw : bool, optional
        If ``True`` (default), redraw after the addition.

    Raises
    ------
    NotImplementedError
        If *allDevices* is ``True``.
    TypeError
        If *gPath* is invalid or *grep* is not boolean.
    """
    if allDevices:
        raise NotImplementedError("allDevices is not yet implemented")

    state = get_state()
    dl = state.display_list

    if gPath is None:
        # Add as a new top-level entry
        dl.record(DLDrawGrob(grob=grob))
    else:
        gpath = _ensure_gpath(gPath)
        if not isinstance(grep, bool):
            raise TypeError("invalid 'grep' value")
        grep_flags = [grep] * gpath.n

        matches = _find_dl_grobs(dl, gpath, strict, grep_flags, global_)
        for _dl_idx, matched_grob in matches:
            if isinstance(matched_grob, GTree):
                matched_grob.add_child(grob)
            else:
                warnings.warn(
                    f"gPath matched a non-gTree grob '{matched_grob.name}'; "
                    "cannot add child",
                    stacklevel=2,
                )

    if redraw:
        _redraw_display_list()


# ---------------------------------------------------------------------------
# Public API -- grid.remove / grid.gremove
# ---------------------------------------------------------------------------


def grid_remove(
    gPath: Union[str, GPath],
    warn: bool = True,
    strict: bool = False,
    grep: bool = False,
    global_: bool = False,
    allDevices: bool = False,
    redraw: bool = True,
) -> None:
    """Remove grob(s) from the display list.

    Locates grobs matching *gPath* and removes them.  For single-component
    paths the grob is removed directly from the display list.  For
    multi-component paths the leaf grob is removed from its parent gTree.
    This is the Python equivalent of R's ``grid.remove()``.

    Parameters
    ----------
    gPath : str or GPath
        Path identifying the grob(s) to remove.
    warn : bool, optional
        If ``True`` (default), warn when the path does not match.
    strict : bool, optional
        If ``True``, require exact depth matching.
    grep : bool, optional
        If ``True``, use regex matching on path components.
    global_ : bool, optional
        If ``True``, remove all matching grobs; otherwise only the first.
    allDevices : bool, optional
        Not yet implemented.
    redraw : bool, optional
        If ``True`` (default), redraw after the removal.

    Raises
    ------
    NotImplementedError
        If *allDevices* is ``True``.
    TypeError
        If *gPath* is invalid or *grep* is not boolean.
    """
    if allDevices:
        raise NotImplementedError("allDevices is not yet implemented")

    gpath = _ensure_gpath(gPath)

    if not isinstance(grep, bool):
        raise TypeError("invalid 'grep' value")
    grep_flags = [grep] * gpath.n

    state = get_state()
    dl = state.display_list

    if gpath.n == 1:
        # Remove directly from the display list
        pattern = gpath.name
        use_grep = grep_flags[0]
        removed_any = False
        items = dl.get_items()
        indices_to_remove: list[int] = []

        for idx, item in enumerate(items):
            if not isinstance(item, DLDrawGrob) or item.grob is None:
                continue
            matched = False
            if use_grep:
                matched = bool(re.search(pattern, item.grob.name))
            else:
                matched = pattern == item.grob.name
            if matched:
                indices_to_remove.append(idx)
                removed_any = True
                if not global_:
                    break

        # Remove in reverse order to preserve indices
        for idx in reversed(indices_to_remove):
            dl._items.pop(idx)

        if not removed_any and warn:
            warnings.warn(
                f"gPath ({gpath}) not found on the display list",
                stacklevel=2,
            )
    else:
        # Multi-level: remove leaf from parent gTree
        parent_path = GPath(*gpath.components[:-1])
        leaf_name = gpath.name
        leaf_grep = grep_flags[-1]
        parent_grep_flags = grep_flags[:-1]

        matches = _find_dl_grobs(dl, parent_path, strict, parent_grep_flags, global_)
        removed_any = False

        for _dl_idx, parent_grob in matches:
            if not isinstance(parent_grob, GTree):
                continue
            # Find matching child(ren) to remove
            names_to_remove: list[str] = []
            for child_name in parent_grob._children_order:
                if leaf_grep:
                    if re.search(leaf_name, child_name):
                        names_to_remove.append(child_name)
                else:
                    if leaf_name == child_name:
                        names_to_remove.append(child_name)
                if not global_ and names_to_remove:
                    break

            for name in names_to_remove:
                parent_grob.remove_child(name)
                removed_any = True

        if not removed_any and warn:
            warnings.warn(
                f"gPath ({gpath}) not found",
                stacklevel=2,
            )

    if redraw:
        _redraw_display_list()


def grid_gremove(
    gPath: Union[str, GPath],
    warn: bool = True,
    strict: bool = False,
    allDevices: bool = False,
    redraw: bool = True,
) -> None:
    """Remove grobs with ``grep=True`` (convenience alias).

    This is the Python equivalent of R's ``grid.gremove()``, which is
    ``grid.remove()`` with ``grep=True`` and ``global_=True`` as defaults.

    Parameters
    ----------
    gPath : str or GPath
        Path identifying the grob(s) to remove (regex matching enabled).
    warn : bool, optional
        If ``True`` (default), warn when the path does not match.
    strict : bool, optional
        If ``True``, require exact depth matching.
    allDevices : bool, optional
        Not yet implemented.
    redraw : bool, optional
        If ``True`` (default), redraw after the removal.
    """
    grid_remove(
        gPath,
        warn=warn,
        strict=strict,
        grep=True,
        global_=True,
        allDevices=allDevices,
        redraw=redraw,
    )
