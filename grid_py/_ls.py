"""Listing and searching grobs/viewports for grid_py (port of R's grid ls/grep).

This module provides functions for inspecting the grid scene graph:

* :func:`grid_ls` -- list grobs and/or viewports on the display list or
  within a given grob/viewport tree.
* :func:`grid_grep` -- search for grobs (or viewports) by name pattern.
* Formatting helpers: :func:`nested_listing`, :func:`path_listing`,
  :func:`grob_path_listing`.
* Introspection: :func:`show_grob`, :func:`get_names`, :func:`child_names`.

References
----------
R source: ``src/library/grid/R/ls.R`` (~908 lines)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from ._grob import GList, GTree, Grob, is_grob
from ._path import GPath, VpPath, PATH_SEP
from ._display_list import DisplayList, DLDrawGrob, DLPushViewport, DLPopViewport, DLUpViewport, DLDownViewport
from ._state import get_state

__all__ = [
    "grid_ls",
    "grid_grep",
    "nested_listing",
    "path_listing",
    "grob_path_listing",
    "show_grob",
    "get_names",
    "child_names",
]


# ---------------------------------------------------------------------------
# GridListing -- internal data structure
# ---------------------------------------------------------------------------


@dataclass
class GridListingEntry:
    """A single entry in a grid listing.

    Parameters
    ----------
    name : str
        Display name of the grob or viewport.
    g_depth : int
        Grob nesting depth.
    vp_depth : int
        Viewport nesting depth.
    g_path : str
        Grob path accumulated so far.
    vp_path : str
        Viewport path accumulated so far.
    entry_type : str
        One of ``"grobListing"``, ``"gTreeListing"``, ``"vpListing"``,
        ``"vpPopListing"``, ``"vpUpListing"``, ``"vpNameListing"``.
    """

    name: str
    g_depth: int = 0
    vp_depth: int = 0
    g_path: str = ""
    vp_path: str = ""
    entry_type: str = "grobListing"


@dataclass
class FlatGridListing:
    """Flattened listing of grobs and viewports.

    This is the Python equivalent of R's ``flatGridListing`` object, storing
    parallel vectors of metadata for each entry.

    Parameters
    ----------
    names : list[str]
        Names of the listed elements.
    g_depths : list[int]
        Grob depths.
    vp_depths : list[int]
        Viewport depths.
    g_paths : list[str]
        Grob paths.
    vp_paths : list[str]
        Viewport paths.
    types : list[str]
        Entry type tags.
    """

    names: List[str] = field(default_factory=list)
    g_depths: List[int] = field(default_factory=list)
    vp_depths: List[int] = field(default_factory=list)
    g_paths: List[str] = field(default_factory=list)
    vp_paths: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    def extend(self, other: "FlatGridListing") -> None:
        """Append all entries from *other* to this listing.

        Parameters
        ----------
        other : FlatGridListing
            The listing to merge in.
        """
        self.names.extend(other.names)
        self.g_depths.extend(other.g_depths)
        self.vp_depths.extend(other.vp_depths)
        self.g_paths.extend(other.g_paths)
        self.vp_paths.extend(other.vp_paths)
        self.types.extend(other.types)

    def append_entry(self, entry: GridListingEntry) -> None:
        """Append a single entry.

        Parameters
        ----------
        entry : GridListingEntry
            The entry to append.
        """
        self.names.append(entry.name)
        self.g_depths.append(entry.g_depth)
        self.vp_depths.append(entry.vp_depth)
        self.g_paths.append(entry.g_path)
        self.vp_paths.append(entry.vp_path)
        self.types.append(entry.entry_type)

    def subset(self, indices: List[int]) -> "FlatGridListing":
        """Return a new listing containing only the entries at *indices*.

        Parameters
        ----------
        indices : list[int]
            The 0-based indices to keep.

        Returns
        -------
        FlatGridListing
        """
        return FlatGridListing(
            names=[self.names[i] for i in indices],
            g_depths=[self.g_depths[i] for i in indices],
            vp_depths=[self.vp_depths[i] for i in indices],
            g_paths=[self.g_paths[i] for i in indices],
            vp_paths=[self.vp_paths[i] for i in indices],
            types=[self.types[i] for i in indices],
        )

    def __repr__(self) -> str:
        return f"FlatGridListing(n={len(self.names)})"


# ---------------------------------------------------------------------------
# Internal listing builder
# ---------------------------------------------------------------------------


def _inc_path(old_path: str, addition: str) -> str:
    """Append *addition* to *old_path* with the grid path separator.

    Parameters
    ----------
    old_path : str
        Existing accumulated path.
    addition : str
        Component to append.

    Returns
    -------
    str
    """
    if old_path:
        return f"{old_path}{PATH_SEP}{addition}"
    return addition


def _list_grob(
    x: Any,
    grobs: bool,
    viewports: bool,
    full_names: bool,
    recursive: bool,
    g_depth: int = 0,
    vp_depth: int = 0,
    g_path: str = "",
    vp_path: str = "",
) -> FlatGridListing:
    """Recursively build a flat listing for a grob or gTree.

    Parameters
    ----------
    x : Grob or GTree or GList or None
        The object to list.
    grobs : bool
        Include grob entries.
    viewports : bool
        Include viewport entries.
    full_names : bool
        Use full class-qualified names.
    recursive : bool
        Recurse into children.
    g_depth : int
        Current grob depth.
    vp_depth : int
        Current viewport depth.
    g_path : str
        Accumulated grob path.
    vp_path : str
        Accumulated viewport path.

    Returns
    -------
    FlatGridListing
    """
    listing = FlatGridListing()

    if x is None:
        return listing

    # Handle GList
    if isinstance(x, GList):
        for child in x:
            child_listing = _list_grob(
                child, grobs, viewports, full_names, recursive,
                g_depth, vp_depth, g_path, vp_path,
            )
            listing.extend(child_listing)
        return listing

    if not is_grob(x):
        return listing

    # Determine display name
    if full_names:
        display_name = repr(x)
    else:
        display_name = x.name

    # gTree case
    if isinstance(x, GTree):
        if grobs:
            listing.append_entry(GridListingEntry(
                name=display_name,
                g_depth=g_depth,
                vp_depth=vp_depth,
                g_path=g_path,
                vp_path=vp_path,
                entry_type="gTreeListing",
            ))

        if recursive:
            child_g_path = _inc_path(g_path, display_name) if grobs else g_path
            child_g_depth = g_depth + 1 if grobs else g_depth
            for child_name in x._children_order:
                child = x._children[child_name]
                child_listing = _list_grob(
                    child, grobs, viewports, full_names, recursive,
                    child_g_depth, vp_depth, child_g_path, vp_path,
                )
                listing.extend(child_listing)

        return listing

    # Plain grob case
    if grobs:
        listing.append_entry(GridListingEntry(
            name=display_name,
            g_depth=g_depth,
            vp_depth=vp_depth,
            g_path=g_path,
            vp_path=vp_path,
            entry_type="grobListing",
        ))

    return listing


def _list_display_list(
    grobs: bool,
    viewports: bool,
    full_names: bool,
    recursive: bool,
) -> FlatGridListing:
    """Build a flat listing from the current display list.

    Parameters
    ----------
    grobs : bool
        Include grob entries.
    viewports : bool
        Include viewport entries.
    full_names : bool
        Use full class-qualified names.
    recursive : bool
        Recurse into gTree children.

    Returns
    -------
    FlatGridListing
    """
    state = get_state()
    dl = state.display_list
    listing = FlatGridListing()
    vp_depth = 0
    vp_path = ""

    for item in dl:
        if isinstance(item, DLDrawGrob) and item.grob is not None:
            grob_listing = _list_grob(
                item.grob, grobs, viewports, full_names, recursive,
                g_depth=0, vp_depth=vp_depth, g_path="", vp_path=vp_path,
            )
            listing.extend(grob_listing)

        elif isinstance(item, DLPushViewport) and viewports:
            vp = item.viewport
            vp_name = getattr(vp, "name", str(vp)) if vp is not None else "?"
            if full_names:
                display_name = f"viewport[{vp_name}]"
            else:
                display_name = vp_name
            listing.append_entry(GridListingEntry(
                name=display_name,
                g_depth=0,
                vp_depth=vp_depth,
                g_path="",
                vp_path=vp_path,
                entry_type="vpListing",
            ))
            vp_depth += 1
            vp_path = _inc_path(vp_path, vp_name)

        elif isinstance(item, DLPopViewport) and viewports:
            n = item.n
            if full_names:
                display_name = f"popViewport[{n}]"
            else:
                display_name = str(n)
            listing.append_entry(GridListingEntry(
                name=display_name,
                g_depth=0,
                vp_depth=vp_depth,
                g_path="",
                vp_path=vp_path,
                entry_type="vpPopListing",
            ))
            # Adjust depth and path
            vp_depth = max(0, vp_depth - n)
            parts = vp_path.split(PATH_SEP) if vp_path else []
            remaining = max(0, len(parts) - n)
            vp_path = PATH_SEP.join(parts[:remaining]) if remaining > 0 else ""

        elif isinstance(item, DLUpViewport) and viewports:
            n = item.n
            if full_names:
                display_name = f"upViewport[{n}]"
            else:
                display_name = str(n)
            listing.append_entry(GridListingEntry(
                name=display_name,
                g_depth=0,
                vp_depth=vp_depth,
                g_path="",
                vp_path=vp_path,
                entry_type="vpUpListing",
            ))
            vp_depth = max(0, vp_depth - n)
            parts = vp_path.split(PATH_SEP) if vp_path else []
            remaining = max(0, len(parts) - n)
            vp_path = PATH_SEP.join(parts[:remaining]) if remaining > 0 else ""

        elif isinstance(item, DLDownViewport) and viewports:
            path_obj = item.params.get("path")
            if path_obj is not None:
                vp_name = getattr(path_obj, "name", str(path_obj))
            else:
                vp_name = "?"
            if full_names:
                display_name = f"downViewport[{vp_name}]"
            else:
                display_name = vp_name
            listing.append_entry(GridListingEntry(
                name=display_name,
                g_depth=0,
                vp_depth=vp_depth,
                g_path="",
                vp_path=vp_path,
                entry_type="vpNameListing",
            ))
            vp_depth += 1
            vp_path = _inc_path(vp_path, vp_name)

    return listing


# ---------------------------------------------------------------------------
# Public API -- grid.ls
# ---------------------------------------------------------------------------


def grid_ls(
    x: Any = None,
    grobs: bool = True,
    viewports: bool = False,
    fullNames: bool = False,
    recursive: bool = True,
    print_: Union[bool, Callable[..., Any]] = True,
) -> FlatGridListing:
    """List grobs and/or viewports.

    When *x* is ``None``, the current display list is listed.  Otherwise,
    *x* should be a :class:`~._grob.Grob`, :class:`~._grob.GTree`, or
    :class:`~._grob.GList` to inspect.  This is the Python equivalent of
    R's ``grid.ls()``.

    Parameters
    ----------
    x : Grob, GTree, GList, or None, optional
        The object to list.  ``None`` lists the display list.
    grobs : bool, optional
        Include grob entries (default ``True``).
    viewports : bool, optional
        Include viewport entries (default ``False``).
    fullNames : bool, optional
        Use full class-qualified names (default ``False``).
    recursive : bool, optional
        Recurse into gTree children (default ``True``).
    print_ : bool or callable, optional
        If ``True`` (default), print the listing to stdout.  If a callable,
        call it with the listing.  If ``False``, suppress output.

    Returns
    -------
    FlatGridListing
        The generated listing object.

    Raises
    ------
    TypeError
        If *print_* is not a bool or callable.

    Examples
    --------
    >>> listing = grid_ls(print_=False)  # capture without printing
    """
    if x is None:
        listing = _list_display_list(
            grobs=grobs,
            viewports=viewports,
            full_names=fullNames,
            recursive=recursive,
        )
    else:
        listing = _list_grob(
            x,
            grobs=grobs,
            viewports=viewports,
            full_names=fullNames,
            recursive=recursive,
        )

    if isinstance(print_, bool):
        if print_:
            nested_listing(listing)
    elif callable(print_):
        print_(listing)
    else:
        raise TypeError("invalid 'print_' argument")

    return listing


# ---------------------------------------------------------------------------
# Public API -- grid.grep
# ---------------------------------------------------------------------------


def grid_grep(
    path: Union[str, GPath],
    x: Any = None,
    grep: bool = True,
    global_: bool = True,
    allDevices: bool = False,
    viewports: bool = False,
    strict: bool = False,
) -> Union[GPath, List[GPath], List[str]]:
    """Search for grobs (or viewports) whose names match *path*.

    Returns the full grob/viewport path(s) that match the given *path*
    pattern.  This is the Python equivalent of R's ``grid.grep()``.

    Parameters
    ----------
    path : str or GPath
        The name or pattern to search for.
    x : Grob, GTree, or None, optional
        Object to search within.  ``None`` searches the display list.
    grep : bool, optional
        If ``True`` (default), use regex matching on path components.
    global_ : bool, optional
        If ``True`` (default), return all matches; otherwise return
        the first match only.
    allDevices : bool, optional
        Not yet implemented.
    viewports : bool, optional
        If ``True``, also search viewport names.
    strict : bool, optional
        If ``True``, require exact depth matching.

    Returns
    -------
    GPath or list[GPath] or list[str]
        Matching path(s).  Returns an empty list when no matches are found
        and *global_* is ``True``, or ``None``-like empty list otherwise.

    Raises
    ------
    NotImplementedError
        If *allDevices* is ``True``.
    """
    if allDevices:
        raise NotImplementedError("allDevices is not yet implemented")

    if isinstance(path, str):
        gpath = GPath(path)
    elif isinstance(path, GPath):
        gpath = path
    else:
        raise TypeError(f"invalid path: expected str or GPath, got {type(path).__name__}")

    depth = gpath.n
    path_pieces = list(gpath.components)

    # Normalise grep to a per-component list
    if isinstance(grep, bool):
        grep_flags = [grep] * depth
    else:
        grep_flags = list(grep)
        while len(grep_flags) < depth:
            grep_flags.append(grep_flags[-1] if grep_flags else False)

    # Build the flat listing
    listing = grid_ls(
        x,
        grobs=True,
        viewports=viewports,
        fullNames=False,
        recursive=True,
        print_=False,
    )

    if not listing.names:
        return []

    # Filter to grob/gTree/vp listings only
    keep_types = {"grobListing", "gTreeListing"}
    if viewports:
        keep_types.add("vpListing")

    keep_indices = [
        i for i, t in enumerate(listing.types) if t in keep_types
    ]
    if not keep_indices:
        return []

    matches: list[GPath] = []

    for i in keep_indices:
        entry_name = listing.names[i]
        entry_g_path = listing.g_paths[i]
        entry_type = listing.types[i]

        # Build the full path components for this entry
        if entry_type.startswith("vp"):
            entry_path_str = listing.vp_paths[i]
            entry_depth = listing.vp_depths[i]
        else:
            entry_path_str = entry_g_path
            entry_depth = listing.g_depths[i]

        if entry_path_str:
            dl_path_pieces = entry_path_str.split(PATH_SEP) + [entry_name]
        else:
            dl_path_pieces = [entry_name]

        dl_depth = len(dl_path_pieces)

        # Filter by depth
        if strict:
            if dl_depth != depth:
                continue
        else:
            if dl_depth < depth:
                continue

        # Attempt match
        matched = False

        if strict:
            # All path pieces must match at same position
            all_match = True
            for j in range(depth):
                if grep_flags[j]:
                    if not re.search(path_pieces[j], dl_path_pieces[j]):
                        all_match = False
                        break
                else:
                    if path_pieces[j] != dl_path_pieces[j]:
                        all_match = False
                        break
            matched = all_match
        else:
            # Sliding window match
            offset = 0
            while offset + depth <= dl_depth:
                all_match = True
                for j in range(depth):
                    if grep_flags[j]:
                        if not re.search(path_pieces[j], dl_path_pieces[offset + j]):
                            all_match = False
                            break
                    else:
                        if path_pieces[j] != dl_path_pieces[offset + j]:
                            all_match = False
                            break
                if all_match:
                    matched = True
                    break
                offset += 1

        if matched:
            result_path = GPath(*dl_path_pieces)
            if not global_:
                return result_path
            matches.append(result_path)

    return matches


# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------


def nested_listing(
    x: FlatGridListing,
    gindent: str = "  ",
    vpindent: Optional[str] = None,
) -> None:
    """Print a :class:`FlatGridListing` with nested indentation.

    Parameters
    ----------
    x : FlatGridListing
        The listing to print.
    gindent : str, optional
        String to repeat for each level of grob depth (default ``"  "``).
    vpindent : str or None, optional
        String to repeat for each level of viewport depth.  Defaults to
        *gindent*.

    Raises
    ------
    TypeError
        If *x* is not a :class:`FlatGridListing`.
    """
    if not isinstance(x, FlatGridListing):
        raise TypeError("invalid listing: expected FlatGridListing")

    if vpindent is None:
        vpindent = gindent

    for i in range(len(x.names)):
        prefix = gindent * x.g_depths[i] + vpindent * x.vp_depths[i]
        print(f"{prefix}{x.names[i]}")


def path_listing(
    x: FlatGridListing,
    gvpSep: str = " | ",
    gAlign: bool = True,
) -> None:
    """Print a :class:`FlatGridListing` with full paths.

    Viewport entries show their accumulated path; grob entries show
    ``vpPath | grobPath``.

    Parameters
    ----------
    x : FlatGridListing
        The listing to print.
    gvpSep : str, optional
        Separator between viewport path and grob path (default ``" | "``).
    gAlign : bool, optional
        If ``True`` (default), pad viewport paths so grob paths align.

    Raises
    ------
    TypeError
        If *x* is not a :class:`FlatGridListing`.
    """
    if not isinstance(x, FlatGridListing):
        raise TypeError("invalid listing: expected FlatGridListing")

    n = len(x.names)
    if n == 0:
        return

    vp_listings = [t.startswith("vp") for t in x.types]
    paths: list[str] = list(x.vp_paths)

    # Build viewport display paths
    max_len = 0
    for i in range(n):
        if vp_listings[i]:
            paths[i] = _inc_path(paths[i], x.names[i])
            max_len = max(max_len, len(paths[i]))

    if not any(vp_listings):
        max_len = max((len(p) for p in paths), default=0)

    # Build grob display paths
    for i in range(n):
        if not vp_listings[i]:
            grob_full = _inc_path(x.g_paths[i], x.names[i])
            if gAlign:
                padded = paths[i] + " " * max(0, max_len - len(paths[i]))
            else:
                padded = paths[i]
            paths[i] = f"{padded}{gvpSep}{grob_full}"

    for p in paths:
        print(p)


def grob_path_listing(x: FlatGridListing, **kwargs: Any) -> None:
    """Print only the grob entries from a :class:`FlatGridListing`.

    Parameters
    ----------
    x : FlatGridListing
        The listing to filter and print.
    **kwargs
        Additional keyword arguments forwarded to :func:`path_listing`.
    """
    grob_indices = [
        i for i, t in enumerate(x.types) if t.startswith("g")
    ]
    if grob_indices:
        sub = x.subset(grob_indices)
        path_listing(sub, **kwargs)


# ---------------------------------------------------------------------------
# show_grob
# ---------------------------------------------------------------------------


def show_grob(
    x: Any = None,
    gPath: Optional[Union[str, GPath]] = None,
    strict: bool = False,
    grep: bool = False,
) -> Optional[Grob]:
    """Display information about a grob, optionally navigating via *gPath*.

    If *x* is ``None``, the display list is searched.  Returns the
    located grob (or ``None`` if not found).

    Parameters
    ----------
    x : Grob, GTree, or None, optional
        The grob to inspect.  ``None`` searches the display list.
    gPath : str, GPath, or None, optional
        Path to a specific child within *x* or the display list.
    strict : bool, optional
        Require exact depth matching when resolving *gPath*.
    grep : bool, optional
        Use regex matching when resolving *gPath*.

    Returns
    -------
    Grob or None
        The located grob, or ``None`` if not found.
    """
    if x is None and gPath is None:
        # List the display list
        listing = grid_ls(print_=True)
        return None

    if x is None:
        # Search display list by gPath
        from ._edit import grid_get
        result = grid_get(gPath, strict=strict, grep=grep, global_=False)
        if result is not None and is_grob(result):
            print(repr(result))
        return result

    if gPath is not None:
        # Navigate into x
        if isinstance(gPath, str):
            gPath = GPath(gPath)
        if isinstance(x, GTree):
            try:
                from ._grob import get_grob
                child = get_grob(x, gPath)
                print(repr(child))
                return child
            except (KeyError, TypeError):
                return None
        return None

    # Just show x
    print(repr(x))
    return x


# ---------------------------------------------------------------------------
# get_names / child_names (deprecated helpers)
# ---------------------------------------------------------------------------


def get_names(x: Any = None) -> List[str]:
    """Return names of grobs on the display list or children of *x*.

    .. deprecated::
        Use ``grid_ls(print_=False)`` instead.

    Parameters
    ----------
    x : GTree or None, optional
        A gTree whose children to list.  ``None`` lists the display list.

    Returns
    -------
    list[str]
        Names of grobs or children.
    """
    warnings.warn(
        "get_names is deprecated; use grid_ls(print_=False) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if x is None:
        listing = grid_ls(print_=False)
        return list(listing.names)
    if isinstance(x, GTree):
        return list(x._children_order)
    if is_grob(x):
        return [x.name]
    return []


def child_names(x: Any) -> List[str]:
    """Return the names of the children of gTree *x*.

    .. deprecated::
        Use ``x._children_order`` directly or ``grid_ls(x, print_=False)``.

    Parameters
    ----------
    x : GTree
        The gTree to inspect.

    Returns
    -------
    list[str]
        Names of children.
    """
    warnings.warn(
        "child_names is deprecated; use grid_ls(x, print_=False) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(x, GTree):
        return list(x._children_order)
    return []
