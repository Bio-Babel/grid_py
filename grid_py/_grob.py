"""Grob base classes for grid_py (port of R's grid ``grob`` system).

This module provides the core graphical-object hierarchy:

* :class:`Grob` -- base class for all graphical objects.
* :class:`GTree` -- a grob that contains child grobs.
* :class:`GList` -- a flat container of grobs.
* :class:`GEdit` / :class:`GEditList` -- edit specifications.

Together with a suite of free functions for constructing, querying, and
mutating grob trees, these classes form the backbone of the grid_py scene
graph, closely mirroring R's *grid* package.
"""

from __future__ import annotations

import copy
import itertools
import warnings
from collections import OrderedDict
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from ._gpar import Gpar
from ._path import GPath

__all__ = [
    "Grob",
    "GTree",
    "GList",
    "GEdit",
    "GEditList",
    "grob_tree",
    "grob_name",
    "is_grob",
    "get_grob",
    "set_grob",
    "add_grob",
    "remove_grob",
    "edit_grob",
    "force_grob",
    "set_children",
    "reorder_grob",
    "apply_edit",
    "apply_edits",
]

# ---------------------------------------------------------------------------
# Auto-name counter (mirrors R's ``grobAutoName`` closure)
# ---------------------------------------------------------------------------

_auto_name_counter: int = 0


def _reset_auto_name() -> None:
    """Reset the global auto-name counter (useful for testing)."""
    global _auto_name_counter
    _auto_name_counter = 0


def _auto_name(prefix: str = "GRID", suffix: str = "GROB") -> str:
    """Generate a unique grob name like ``GRID.GROB.1``.

    Parameters
    ----------
    prefix : str
        Leading part of the name.
    suffix : str
        Middle part of the name (typically the grob class).

    Returns
    -------
    str
    """
    global _auto_name_counter
    _auto_name_counter += 1
    return f"{prefix}.{suffix}.{_auto_name_counter}"


# ---------------------------------------------------------------------------
# grob_name (public helper)
# ---------------------------------------------------------------------------


def grob_name(grob: Optional["Grob"] = None, prefix: str = "GRID") -> str:
    """Return an auto-generated grob name.

    Parameters
    ----------
    grob : Grob or None
        If supplied, the grob's class name is used as the suffix.
    prefix : str
        Leading part of the generated name.

    Returns
    -------
    str
        A unique name such as ``"GRID.rect.3"``.

    Raises
    ------
    TypeError
        If *grob* is not ``None`` and not a :class:`Grob`.
    """
    if grob is None:
        return _auto_name(prefix)
    if not isinstance(grob, Grob):
        raise TypeError("invalid 'grob' argument")
    suffix = grob._grid_class if grob._grid_class else type(grob).__name__
    return _auto_name(prefix, suffix)


# ---------------------------------------------------------------------------
# is_grob
# ---------------------------------------------------------------------------


def is_grob(x: Any) -> bool:
    """Return ``True`` if *x* is a :class:`Grob` (or subclass).

    Parameters
    ----------
    x : Any

    Returns
    -------
    bool
    """
    return isinstance(x, Grob)


# ---------------------------------------------------------------------------
# Grob
# ---------------------------------------------------------------------------


class Grob:
    """Base class for all graphical objects.

    This is the Python equivalent of R's ``grob`` S3 class.  Every grob has
    a *name* (auto-generated when omitted), an optional *gp* (:class:`Gpar`),
    and an optional *vp* (viewport or viewport path).  Subclasses add
    domain-specific fields and override the various ``*_details`` hooks.

    Parameters
    ----------
    name : str or None
        Unique name.  Auto-generated (e.g. ``"GRID.grob.1"``) when ``None``.
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport (duck-typed to avoid circular imports).
    _grid_class : str or None
        R-style class tag (e.g. ``"rect"``, ``"circle"``).
    **kwargs
        Arbitrary grob-specific fields stored as instance attributes.
    """

    # We allow arbitrary attributes via __dict__, but declare the core
    # slots here for documentation.

    def __init__(
        self,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
        _grid_class: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._grid_class: str = _grid_class or "grob"
        # Store extra fields first so checkNameSlot can use _grid_class
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._name: Optional[str] = name
        self._gp: Optional[Gpar] = gp
        self._vp: Optional[Any] = vp
        # Validate and fill defaults
        self._validate()

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        """Run validation (mirrors R ``validGrob.grob``)."""
        self.valid_details()
        # Auto-name
        if self._name is None:
            self._name = _auto_name(suffix=self._grid_class)
        else:
            self._name = str(self._name)
        # Check gp
        if self._gp is not None and not isinstance(self._gp, Gpar):
            raise TypeError("invalid 'gp' slot: expected Gpar or None")
        # Check vp (duck typed -- accept anything with a reasonable interface)
        if self._vp is not None:
            self._vp = self._check_vp(self._vp)

    @staticmethod
    def _check_vp(vp: Any) -> Any:
        """Validate the vp slot (duck-typed viewport check).

        Parameters
        ----------
        vp : Any

        Returns
        -------
        object
            The validated viewport (possibly wrapped as a VpPath for strings).
        """
        if vp is None:
            return None
        # Accept anything that quacks like a viewport or vpPath.
        # If a plain string is given, try to wrap it in a VpPath (mirror R
        # behaviour).  Import lazily to avoid circular imports.
        if isinstance(vp, str):
            from ._path import VpPath
            return VpPath(vp)
        return vp

    # -- properties --------------------------------------------------------

    @property
    def name(self) -> str:
        """The grob's unique name.

        Returns
        -------
        str
        """
        return self._name  # type: ignore[return-value]

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def gp(self) -> Optional[Gpar]:
        """Graphical parameters.

        Returns
        -------
        Gpar or None
        """
        return self._gp

    @gp.setter
    def gp(self, value: Optional[Gpar]) -> None:
        if value is not None and not isinstance(value, Gpar):
            raise TypeError("invalid 'gp' slot: expected Gpar or None")
        self._gp = value

    @property
    def vp(self) -> Optional[Any]:
        """Viewport associated with this grob.

        Returns
        -------
        object or None
        """
        return self._vp

    @vp.setter
    def vp(self, value: Any) -> None:
        self._vp = self._check_vp(value)

    # -- hook methods (override in subclasses) -----------------------------

    def draw_details(self, recording: bool = True) -> None:
        """Perform class-specific drawing.

        Override in subclasses to implement actual rendering.

        Parameters
        ----------
        recording : bool
            Whether the drawing should be recorded on the display list.
        """

    def pre_draw_details(self) -> None:
        """Pre-draw hook (called before ``draw_details``).

        Override to set up any state that ``draw_details`` requires.
        """

    def post_draw_details(self) -> None:
        """Post-draw hook (called after ``draw_details``).

        Override to tear down state created by ``pre_draw_details``.
        """

    def valid_details(self) -> None:
        """Validate class-specific slots.

        Override in subclasses to perform additional validation.  Called
        during construction and after editing.  Modify ``self`` in place or
        raise on invalid state.
        """

    def make_content(self) -> "Grob":
        """Create or transform the drawing content.

        Override to lazily materialise content just before drawing.

        Returns
        -------
        Grob
            Typically ``self``, possibly modified.
        """
        return self

    def make_context(self) -> "Grob":
        """Create or transform the drawing context (viewport, etc.).

        Override to lazily adjust the viewport or gp before drawing.

        Returns
        -------
        Grob
            Typically ``self``, possibly modified.
        """
        return self

    def edit_details(self, **kwargs: Any) -> "Grob":
        """Hook called after attributes are updated via :func:`edit_grob`.

        Override to perform additional bookkeeping when edits occur.

        Parameters
        ----------
        **kwargs
            The name-value pairs that were applied.

        Returns
        -------
        Grob
            ``self`` (possibly modified).
        """
        return self

    def width_details(self) -> Any:
        """Return the width of this grob in its own coordinate system.

        Returns
        -------
        object
            Implementation-dependent width representation.
        """
        return None

    def height_details(self) -> Any:
        """Return the height of this grob in its own coordinate system.

        Returns
        -------
        object
            Implementation-dependent height representation.
        """
        return None

    def x_details(self, theta: float = 0.0) -> Any:
        """Return the x-location at angle *theta*.

        Parameters
        ----------
        theta : float
            Angle in degrees.

        Returns
        -------
        object
        """
        return None

    def y_details(self, theta: float = 0.0) -> Any:
        """Return the y-location at angle *theta*.

        Parameters
        ----------
        theta : float
            Angle in degrees.

        Returns
        -------
        object
        """
        return None

    def ascent_details(self) -> Any:
        """Return the typographic ascent of this grob.

        Returns
        -------
        object
        """
        return None

    def descent_details(self) -> Any:
        """Return the typographic descent of this grob.

        Returns
        -------
        object
        """
        return None

    def grob_coords(self, closed: bool = True) -> Any:
        """Return coordinates for this grob.

        Parameters
        ----------
        closed : bool
            Whether to return coordinates for a closed shape.

        Returns
        -------
        object
        """
        return None

    def grob_points(self, closed: bool = True) -> Any:
        """Return points for this grob.

        Parameters
        ----------
        closed : bool
            Whether to return points for a closed shape.

        Returns
        -------
        object
        """
        return None

    # -- dunder methods ----------------------------------------------------

    def __repr__(self) -> str:
        return f"{self._grid_class}[{self._name}]"

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# GList
# ---------------------------------------------------------------------------


class GList:
    """A flat container of :class:`Grob` objects.

    Mirrors R's ``gList`` class.  Duplicate names are allowed at this level
    but will be disambiguated when the list is attached to a :class:`GTree`.

    Parameters
    ----------
    *grobs : Grob
        Zero or more grob instances.

    Raises
    ------
    TypeError
        If any element is not a :class:`Grob` (or ``None`` / ``GList``,
        which are flattened/ignored).
    """

    __slots__ = ("_grobs",)

    def __init__(self, *grobs: Union["Grob", "GList", None]) -> None:
        flat: list[Grob] = []
        for g in grobs:
            if g is None:
                continue
            if isinstance(g, GList):
                flat.extend(g._grobs)
            elif isinstance(g, Grob):
                flat.append(g)
            else:
                raise TypeError(f"only Grob instances allowed in GList, got {type(g).__name__}")
        self._grobs: list[Grob] = flat

    # -- sequence protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._grobs)

    def __iter__(self) -> Iterator[Grob]:
        return iter(self._grobs)

    def __getitem__(self, index: Union[int, slice]) -> Union[Grob, "GList"]:
        result = self._grobs[index]
        if isinstance(index, slice):
            gl = GList.__new__(GList)
            gl._grobs = list(result)
            return gl
        return result

    def __setitem__(self, index: int, value: Grob) -> None:
        if not isinstance(value, Grob):
            raise TypeError(f"only Grob instances allowed in GList, got {type(value).__name__}")
        self._grobs[index] = value

    def append(self, grob: Grob) -> None:
        """Append a grob to this list.

        Parameters
        ----------
        grob : Grob
        """
        if not isinstance(grob, Grob):
            raise TypeError(f"only Grob instances allowed in GList, got {type(grob).__name__}")
        self._grobs.append(grob)

    # -- dunder methods ----------------------------------------------------

    def __repr__(self) -> str:
        inner = ", ".join(str(g) for g in self._grobs)
        return f"({inner})"


# ---------------------------------------------------------------------------
# GTree
# ---------------------------------------------------------------------------


class GTree(Grob):
    """A grob that contains child grobs.

    This is the Python equivalent of R's ``gTree`` S3 class.  Children are
    stored in an ordered dictionary keyed by name, with a separate
    ``children_order`` list controlling the draw order.

    Parameters
    ----------
    children : GList or None
        Initial set of child grobs.
    name : str or None
        Unique name (auto-generated if ``None``).
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.
    children_order : list[str] or None
        Explicit draw order; derived from *children* if ``None``.
    _grid_class : str or None
        R-style class tag.
    **kwargs
        Extra grob-specific fields.
    """

    def __init__(
        self,
        children: Optional[GList] = None,
        name: Optional[str] = None,
        gp: Optional[Gpar] = None,
        vp: Optional[Any] = None,
        children_order: Optional[List[str]] = None,
        _grid_class: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Initialise children storage *before* parent __init__ so that
        # _validate can inspect it if needed.
        self._children: OrderedDict[str, Grob] = OrderedDict()
        self._children_order: list[str] = []
        super().__init__(
            name=name,
            gp=gp,
            vp=vp,
            _grid_class=_grid_class or "gTree",
            **kwargs,
        )
        # Populate children using the canonical setter
        self._set_children_internal(children)
        # Override order if explicitly provided
        if children_order is not None:
            self._children_order = list(children_order)

    # -- internal helpers --------------------------------------------------

    def _set_children_internal(self, children: Optional[GList]) -> None:
        """Populate children dict and order from a GList."""
        if children is not None and not isinstance(children, GList):
            raise TypeError("'children' must be a GList or None")
        self._children = OrderedDict()
        self._children_order = []
        if children is not None:
            # Filter out Nones (shouldn't be in GList, but be safe)
            for child in children:
                if child is not None:
                    self._children[child.name] = child
                    self._children_order.append(child.name)

    # -- child accessors (mirror R API) ------------------------------------

    def get_children(self) -> GList:
        """Return a :class:`GList` of this tree's children.

        Returns
        -------
        GList
        """
        gl = GList.__new__(GList)
        gl._grobs = [self._children[n] for n in self._children_order]
        return gl

    def set_children(self, gl: GList) -> None:
        """Replace all children with *gl*.

        Parameters
        ----------
        gl : GList
        """
        self._set_children_internal(gl)

    def n_children(self) -> int:
        """Return the number of children.

        Returns
        -------
        int
        """
        return len(self._children_order)

    def add_child(self, child: Grob) -> None:
        """Add or replace a child grob.

        If a child with the same name already exists it is replaced in-place
        (preserving draw order).

        Parameters
        ----------
        child : Grob
        """
        if not isinstance(child, Grob):
            raise TypeError("can only add a Grob to a GTree")
        cname = child.name
        self._children[cname] = child
        # If already in order list, leave it; otherwise append
        if cname not in self._children_order:
            self._children_order.append(cname)

    def remove_child(self, name: str) -> None:
        """Remove the child with the given *name*.

        Parameters
        ----------
        name : str

        Raises
        ------
        KeyError
            If no child with *name* exists.
        """
        if name not in self._children:
            raise KeyError(f"child '{name}' not found")
        del self._children[name]
        self._children_order = [n for n in self._children_order if n != name]

    def get_child(self, name: str) -> Grob:
        """Return the child with the given *name*.

        Parameters
        ----------
        name : str

        Returns
        -------
        Grob

        Raises
        ------
        KeyError
            If no child with *name* exists.
        """
        if name not in self._children:
            raise KeyError(f"child '{name}' not found")
        return self._children[name]

    def set_child(self, name: str, child: Grob) -> None:
        """Replace an existing child by name.

        Parameters
        ----------
        name : str
            Name of the child to replace.  Must exist.
        child : Grob
            Replacement grob.  Its name should match *name*.

        Raises
        ------
        KeyError
            If no child with *name* exists.
        ValueError
            If ``child.name != name``.
        """
        if name not in self._children:
            raise KeyError(f"child '{name}' not found")
        if child.name != name:
            raise ValueError(
                f"new grob name ('{child.name}') does not match existing name ('{name}')"
            )
        self._children[name] = child

    # -- override draw details for gTree -----------------------------------

    def edit_details(self, **kwargs: Any) -> "GTree":
        """Disallow direct editing of ``children`` or ``children_order``.

        Parameters
        ----------
        **kwargs
            The name-value pairs that were applied.

        Returns
        -------
        GTree

        Raises
        ------
        ValueError
            If ``"children"`` or ``"children_order"`` appear in *kwargs*.
        """
        forbidden = {"children", "children_order", "_children", "_children_order"}
        if forbidden.intersection(kwargs):
            raise ValueError(
                "it is invalid to directly edit the 'children' or "
                "'children_order' slot; use add_child / remove_child instead"
            )
        return self

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        child_strs = ", ".join(self._children_order)
        return f"{self._grid_class}[{self._name}]({child_strs})"


# ---------------------------------------------------------------------------
# GEdit / GEditList
# ---------------------------------------------------------------------------


class GEdit:
    """An edit specification storing parameter name-value pairs.

    Mirrors R's ``gEdit()`` constructor.

    Parameters
    ----------
    **kwargs
        Attribute names and their new values.
    """

    __slots__ = ("_specs",)

    def __init__(self, **kwargs: Any) -> None:
        self._specs: dict[str, Any] = dict(kwargs)

    @property
    def specs(self) -> dict[str, Any]:
        """The stored name-value pairs.

        Returns
        -------
        dict[str, Any]
        """
        return dict(self._specs)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._specs.items())
        return f"GEdit({items})"


class GEditList:
    """A list of :class:`GEdit` objects.

    Parameters
    ----------
    *edits : GEdit
        One or more edit specifications.

    Raises
    ------
    TypeError
        If any element is not a :class:`GEdit`.
    """

    __slots__ = ("_edits",)

    def __init__(self, *edits: GEdit) -> None:
        for e in edits:
            if not isinstance(e, GEdit):
                raise TypeError(f"GEditList can only contain GEdit objects, got {type(e).__name__}")
        self._edits: tuple[GEdit, ...] = tuple(edits)

    def __len__(self) -> int:
        return len(self._edits)

    def __iter__(self) -> Iterator[GEdit]:
        return iter(self._edits)

    def __repr__(self) -> str:
        inner = ", ".join(repr(e) for e in self._edits)
        return f"GEditList({inner})"


# ---------------------------------------------------------------------------
# Free functions -- construction helpers
# ---------------------------------------------------------------------------


def grob_tree(*args: Grob, name: Optional[str] = None,
              gp: Optional[Gpar] = None,
              vp: Optional[Any] = None) -> GTree:
    """Convenience constructor for a :class:`GTree` wrapping *args*.

    Parameters
    ----------
    *args : Grob
        Child grobs.
    name : str or None
        Name for the tree (auto-generated if ``None``).
    gp : Gpar or None
        Graphical parameters.
    vp : object or None
        Viewport.

    Returns
    -------
    GTree
    """
    return GTree(children=GList(*args), name=name, gp=gp, vp=vp)


# ---------------------------------------------------------------------------
# get / set / add / remove / edit
# ---------------------------------------------------------------------------


def _resolve_path(path: Union[str, GPath]) -> GPath:
    """Ensure *path* is a :class:`GPath`."""
    if isinstance(path, str):
        return GPath(path)
    if isinstance(path, GPath):
        return path
    raise TypeError(f"invalid path type: {type(path).__name__}")


def get_grob(gtree: GTree, path: Union[str, GPath]) -> Grob:
    """Retrieve a child grob from *gtree* by *path*.

    Currently only supports single-depth (direct child) lookup.

    Parameters
    ----------
    gtree : GTree
        The tree to search.
    path : str or GPath
        Name or path to the desired child.

    Returns
    -------
    Grob

    Raises
    ------
    TypeError
        If *gtree* is not a :class:`GTree`.
    KeyError
        If the child cannot be found.
    """
    if not isinstance(gtree, GTree):
        raise TypeError("can only get a child from a GTree")
    gpath = _resolve_path(path)
    if gpath.n == 1:
        return gtree.get_child(gpath.name)
    # Multi-depth: walk the tree
    current: Grob = gtree
    for component in gpath.components[:-1]:
        if not isinstance(current, GTree):
            raise KeyError(f"'{component}' is not a GTree; cannot descend further")
        current = current.get_child(component)
    if not isinstance(current, GTree):
        raise KeyError(f"cannot get child '{gpath.name}' from a non-GTree grob")
    return current.get_child(gpath.name)


def set_grob(gtree: GTree, path: Union[str, GPath], value: Grob) -> GTree:
    """Return a copy of *gtree* with the child at *path* replaced by *value*.

    Parameters
    ----------
    gtree : GTree
        The tree to modify.
    path : str or GPath
        Name or path identifying the child to replace.
    value : Grob
        The replacement grob.

    Returns
    -------
    GTree
        A shallow copy with the replacement applied.

    Raises
    ------
    TypeError
        If *gtree* is not a :class:`GTree` or *value* is not a :class:`Grob`.
    KeyError
        If the path does not identify an existing child.
    """
    if not isinstance(gtree, GTree):
        raise TypeError("can only set a child on a GTree")
    if not isinstance(value, Grob):
        raise TypeError("replacement must be a Grob")
    gpath = _resolve_path(path)
    result = copy.copy(gtree)
    result._children = OrderedDict(gtree._children)
    result._children_order = list(gtree._children_order)

    if gpath.n == 1:
        if gpath.name not in result._children:
            raise KeyError(f"child '{gpath.name}' not found")
        if value.name != gpath.name:
            raise ValueError(
                f"new grob name ('{value.name}') does not match path ('{gpath.name}')"
            )
        result._children[gpath.name] = value
        return result
    # Multi-depth: recursively copy inner gTrees
    first = gpath.components[0]
    rest = GPath(*gpath.components[1:])
    child = result._children.get(first)
    if child is None:
        raise KeyError(f"child '{first}' not found")
    if not isinstance(child, GTree):
        raise TypeError(f"child '{first}' is not a GTree; cannot descend further")
    result._children[first] = set_grob(child, rest, value)
    return result


def add_grob(gtree: GTree, child: Grob, name: Optional[str] = None) -> GTree:
    """Return a copy of *gtree* with *child* added.

    Parameters
    ----------
    gtree : GTree
        The tree to add to.
    child : Grob
        The child to add.
    name : str or None
        Override name for the child (uses ``child.name`` if ``None``).

    Returns
    -------
    GTree
        A shallow copy with the new child appended.
    """
    if not isinstance(gtree, GTree):
        raise TypeError("can only add a child to a GTree")
    if not isinstance(child, Grob):
        raise TypeError("child must be a Grob")
    result = copy.copy(gtree)
    result._children = OrderedDict(gtree._children)
    result._children_order = list(gtree._children_order)
    if name is not None:
        child = copy.copy(child)
        child.name = name
    result.add_child(child)
    return result


def remove_grob(gtree: GTree, name: str) -> GTree:
    """Return a copy of *gtree* with the named child removed.

    Parameters
    ----------
    gtree : GTree
        The tree to modify.
    name : str
        Name of the child to remove.

    Returns
    -------
    GTree
        A shallow copy with the child removed.
    """
    if not isinstance(gtree, GTree):
        raise TypeError("can only remove a child from a GTree")
    result = copy.copy(gtree)
    result._children = OrderedDict(gtree._children)
    result._children_order = list(gtree._children_order)
    result.remove_child(name)
    return result


def _edit_this_grob(grob: Grob, specs: dict[str, Any]) -> Grob:
    """Apply *specs* to *grob* in place and revalidate (internal).

    Parameters
    ----------
    grob : Grob
    specs : dict

    Returns
    -------
    Grob
    """
    for key, value in specs.items():
        if not key:
            continue
        if key == "gp":
            # Special handling: merge gpar
            if value is None:
                grob._gp = None
            elif grob._gp is not None:
                # Merge new gp on top of existing
                grob._gp = grob._gp.merge(value) if hasattr(grob._gp, "merge") else value
            else:
                grob._gp = value
        elif key == "name":
            grob._name = str(value) if value is not None else None
        elif key == "vp":
            grob._vp = Grob._check_vp(value)
        elif hasattr(grob, key) or hasattr(grob, f"_{key}"):
            setattr(grob, key, value)
        else:
            warnings.warn(f"slot '{key}' not found", stacklevel=3)
    # Re-validate
    grob.valid_details()
    grob.edit_details(**specs)
    return grob


def edit_grob(grob: Grob, **kwargs: Any) -> Grob:
    """Return an edited copy of *grob*.

    Parameters
    ----------
    grob : Grob
        The grob to edit.
    **kwargs
        Attribute name-value pairs to update.

    Returns
    -------
    Grob
        A (deep) copy with the edits applied.
    """
    result = copy.deepcopy(grob)
    return _edit_this_grob(result, kwargs)


def force_grob(grob: Grob) -> Grob:
    """Force evaluation of a (possibly delayed) grob.

    This calls :meth:`~Grob.make_context` and :meth:`~Grob.make_content`
    without actually drawing, capturing any modifications those hooks make.
    If the grob is unchanged, the original object is returned.

    Parameters
    ----------
    grob : Grob
        The grob to force.

    Returns
    -------
    Grob
        The forced grob, possibly with a ``_original`` attribute storing
        the pre-force state.
    """
    original = grob
    x = copy.deepcopy(grob)
    x = x.make_context()
    x = x.make_content()
    # For gTree, also force children
    if isinstance(x, GTree):
        forced_children: list[Grob] = []
        for name in x._children_order:
            forced_children.append(force_grob(x._children[name]))
        x._set_children_internal(GList(*forced_children))
    # If anything changed, stash the original
    # (We can't do an identity check after deepcopy, so we always store.)
    x._original = original  # type: ignore[attr-defined]
    return x


def set_children(gtree: GTree, children: GList) -> GTree:
    """Return a copy of *gtree* with its children replaced.

    Parameters
    ----------
    gtree : GTree
        The tree to modify.
    children : GList
        The new children.

    Returns
    -------
    GTree
        A shallow copy with the new children set.
    """
    if not isinstance(gtree, GTree):
        raise TypeError("can only set children on a GTree")
    result = copy.copy(gtree)
    result._set_children_internal(children)
    return result


def reorder_grob(gtree: GTree, order: Union[List[int], List[str]],
                 back: bool = True) -> GTree:
    """Return a copy of *gtree* with children reordered.

    Parameters
    ----------
    gtree : GTree
        The tree to reorder.
    order : list[int] or list[str]
        Indices (0-based) or names specifying the new front-of-order.
    back : bool
        If ``True`` (default), the specified children come first (back,
        i.e. drawn first / behind); unspecified children are appended.
        If ``False``, unspecified children come first; specified are appended
        (drawn last / in front).

    Returns
    -------
    GTree
        A shallow copy with reordered ``children_order``.

    Raises
    ------
    ValueError
        If *order* contains invalid names or indices.
    """
    if not isinstance(gtree, GTree):
        raise TypeError("can only reorder children of a GTree")
    result = copy.copy(gtree)
    result._children = OrderedDict(gtree._children)
    old_order = list(gtree._children_order)
    n = len(old_order)

    # Deduplicate while preserving order
    seen: set[Any] = set()
    unique_order: list[Any] = []
    for o in order:
        if o not in seen:
            unique_order.append(o)
            seen.add(o)

    # Convert to integer indices
    int_indices: list[int] = []
    for o in unique_order:
        if isinstance(o, str):
            try:
                idx = old_order.index(o)
            except ValueError:
                raise ValueError(f"child name '{o}' not found in children_order")
            int_indices.append(idx)
        elif isinstance(o, int):
            if o < 0 or o >= n:
                raise ValueError(f"index {o} out of range [0, {n})")
            int_indices.append(o)
        else:
            raise TypeError(f"order elements must be int or str, got {type(o).__name__}")

    specified = [old_order[i] for i in int_indices]
    rest = [old_order[i] for i in range(n) if i not in set(int_indices)]

    if back:
        new_order = specified + rest
    else:
        new_order = rest + specified

    result._children_order = new_order
    return result


# ---------------------------------------------------------------------------
# apply_edit / apply_edits
# ---------------------------------------------------------------------------


def apply_edit(grob: Grob, edit: Optional[GEdit]) -> Grob:
    """Apply a single :class:`GEdit` to *grob*.

    Parameters
    ----------
    grob : Grob
        The target grob.
    edit : GEdit or None
        The edit to apply.  ``None`` is a no-op.

    Returns
    -------
    Grob
        An edited copy (or the original if *edit* is ``None``).
    """
    if edit is None:
        return grob
    if not isinstance(edit, GEdit):
        raise TypeError("invalid edit: expected GEdit")
    return edit_grob(grob, **edit._specs)


def apply_edits(grob: Grob, edits: Optional[Union[GEdit, GEditList]]) -> Grob:
    """Apply one or more edits to *grob*.

    Parameters
    ----------
    grob : Grob
        The target grob.
    edits : GEdit, GEditList, or None
        The edit(s) to apply.  ``None`` is a no-op.

    Returns
    -------
    Grob
        An edited copy (or the original if *edits* is ``None``).
    """
    if edits is None:
        return grob
    if isinstance(edits, GEdit):
        return apply_edit(grob, edits)
    if isinstance(edits, GEditList):
        result = grob
        for e in edits:
            result = apply_edits(result, e)
        return result
    raise TypeError("invalid edits: expected GEdit, GEditList, or None")
