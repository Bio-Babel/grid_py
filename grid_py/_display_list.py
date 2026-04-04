"""Display-list management for grid_py -- Python port of R's ``grid`` display list.

In R's *grid* package the display list records every drawing operation so that
the scene can be replayed when a device is resized, copied, or printed.  This
module provides the equivalent bookkeeping for grid_py:

* :class:`DisplayList` -- the mutable, iterable list that records and replays
  operations (analogous to R's ``grid.display.list``).
* :class:`DLOperation` -- abstract base class for a single recorded operation.
* Concrete operation subclasses: :class:`DLDrawGrob`, :class:`DLPushViewport`,
  :class:`DLPopViewport`, :class:`DLUpViewport`, :class:`DLDownViewport`,
  :class:`DLEditGrob`, :class:`DLSetGpar`.

References
----------
R source: ``src/library/grid/R/displaylist.R``, ``src/library/grid/R/grid.R``
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

__all__ = [
    "DisplayList",
    "DLOperation",
    "DLDrawGrob",
    "DLPushViewport",
    "DLPopViewport",
    "DLUpViewport",
    "DLDownViewport",
    "DLEditGrob",
    "DLSetGpar",
]


# ---------------------------------------------------------------------------
# DisplayList
# ---------------------------------------------------------------------------


class DisplayList:
    """A mutable list of recorded drawing operations.

    The display list captures every graphical operation so it can be replayed
    later (e.g. when a device is resized).  Recording can be temporarily
    disabled via :meth:`set_enabled`.

    Parameters
    ----------
    None

    Attributes
    ----------
    _items : list[DLOperation]
        Recorded operations in chronological order.
    _enabled : bool
        Whether new operations are currently being recorded.

    Examples
    --------
    >>> dl = DisplayList()
    >>> dl.record(DLDrawGrob(grob=some_grob))
    >>> len(dl)
    1
    >>> dl.clear()
    >>> len(dl)
    0
    """

    def __init__(self) -> None:
        self._items: List[DLOperation] = []
        self._enabled: bool = True

    # -- recording -----------------------------------------------------------

    def record(self, item: DLOperation) -> None:
        """Append *item* to the display list if recording is enabled.

        Parameters
        ----------
        item : DLOperation
            The operation to record.
        """
        if self._enabled:
            self._items.append(item)

    # -- query / access ------------------------------------------------------

    def get_items(self) -> List[DLOperation]:
        """Return a shallow copy of the recorded operations.

        Returns
        -------
        list[DLOperation]
            A copy of the internal list.
        """
        return list(self._items)

    def is_enabled(self) -> bool:
        """Return whether recording is currently enabled.

        Returns
        -------
        bool
            ``True`` if recording is on.
        """
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        """Enable or disable recording.

        Parameters
        ----------
        on : bool
            ``True`` to enable recording, ``False`` to disable.
        """
        self._enabled = on

    # -- mutation ------------------------------------------------------------

    def clear(self) -> None:
        """Remove all recorded operations."""
        self._items.clear()

    # -- replay / apply ------------------------------------------------------

    def replay(self, state: Any) -> None:
        """Replay every recorded operation against *state*.

        Each item's :meth:`~DLOperation.replay` method is called in order if
        the item exposes one.

        Parameters
        ----------
        state : Any
            An opaque state object (typically the current graphics state)
            passed to each operation's ``replay`` method.
        """
        for item in self._items:
            if hasattr(item, "replay"):
                item.replay(state)

    def apply(self, fn: Callable[[DLOperation], Any]) -> List[Any]:
        """Apply *fn* to every recorded item (``grid.DLapply`` equivalent).

        Parameters
        ----------
        fn : callable
            A function that accepts a single :class:`DLOperation` argument.

        Returns
        -------
        list[Any]
            The collected return values of *fn* for each item.
        """
        return [fn(item) for item in self._items]

    # -- dunder protocols ----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of recorded operations."""
        return len(self._items)

    def __iter__(self) -> Iterator[DLOperation]:
        """Iterate over recorded operations in order."""
        return iter(self._items)

    def __getitem__(self, i: int) -> DLOperation:
        """Return the *i*-th recorded operation.

        Parameters
        ----------
        i : int
            Zero-based index.

        Returns
        -------
        DLOperation
        """
        return self._items[i]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DisplayList(n_items={len(self._items)}, "
            f"enabled={self._enabled})"
        )


# ---------------------------------------------------------------------------
# DLOperation base class
# ---------------------------------------------------------------------------


class DLOperation:
    """Base class for a single display-list entry.

    Subclasses should override :meth:`replay` to perform their specific
    action when the display list is replayed.

    Parameters
    ----------
    op_type : str
        A short human-readable tag describing the operation kind
        (e.g. ``"draw_grob"``, ``"push_vp"``).
    **kwargs : Any
        Arbitrary keyword arguments stored in :attr:`params`.

    Attributes
    ----------
    op_type : str
        The operation tag.
    params : dict[str, Any]
        Additional data captured at recording time.
    """

    def __init__(self, op_type: str, **kwargs: Any) -> None:
        self.op_type: str = op_type
        self.params: Dict[str, Any] = kwargs

    def replay(self, state: Any) -> None:
        """Replay this operation.

        The default implementation is a no-op.  Subclasses should override
        this to perform their specific action.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """

    def __repr__(self) -> str:  # pragma: no cover
        kw = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        extra = f", {kw}" if kw else ""
        return f"{type(self).__name__}(op_type={self.op_type!r}{extra})"


# ---------------------------------------------------------------------------
# Concrete DLOperation subclasses
# ---------------------------------------------------------------------------


class DLDrawGrob(DLOperation):
    """Records a grob drawing operation.

    Parameters
    ----------
    grob : Any
        The graphical object that was drawn.
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    grob : Any
        Reference to the drawn grob.
    """

    def __init__(self, grob: Any = None, **kwargs: Any) -> None:
        super().__init__(op_type="draw_grob", grob=grob, **kwargs)
        self.grob: Any = grob

    def replay(self, state: Any) -> None:
        """Replay the grob drawing.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if self.grob is not None and hasattr(self.grob, "draw"):
            self.grob.draw(state)


class DLPushViewport(DLOperation):
    """Records a viewport push operation.

    Parameters
    ----------
    viewport : Any
        The viewport that was pushed.
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    viewport : Any
        Reference to the pushed viewport.
    """

    def __init__(self, viewport: Any = None, **kwargs: Any) -> None:
        super().__init__(op_type="push_vp", viewport=viewport, **kwargs)
        self.viewport: Any = viewport

    def replay(self, state: Any) -> None:
        """Replay the viewport push.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if state is not None and hasattr(state, "push_viewport"):
            state.push_viewport(self.viewport)


class DLPopViewport(DLOperation):
    """Records a viewport pop operation.

    Parameters
    ----------
    n : int
        Number of viewports to pop (default ``1``).
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    n : int
        Number of viewports to pop.
    """

    def __init__(self, n: int = 1, **kwargs: Any) -> None:
        super().__init__(op_type="pop_vp", n=n, **kwargs)
        self.n: int = n

    def replay(self, state: Any) -> None:
        """Replay the viewport pop.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if state is not None and hasattr(state, "pop_viewport"):
            state.pop_viewport(self.n)


class DLUpViewport(DLOperation):
    """Records an up-viewport navigation.

    Parameters
    ----------
    n : int
        Number of levels to go up (default ``1``).
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    n : int
        Number of levels to navigate up.
    """

    def __init__(self, n: int = 1, **kwargs: Any) -> None:
        super().__init__(op_type="up_vp", n=n, **kwargs)
        self.n: int = n

    def replay(self, state: Any) -> None:
        """Replay the up-viewport navigation.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if state is not None and hasattr(state, "up_viewport"):
            state.up_viewport(self.n)


class DLDownViewport(DLOperation):
    """Records a down-viewport navigation.

    Parameters
    ----------
    path : Any
        Viewport name or path to navigate down to.
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    path : Any
        The viewport path used for navigation.
    """

    def __init__(self, path: Any = None, **kwargs: Any) -> None:
        super().__init__(op_type="down_vp", path=path, **kwargs)
        self.path: Any = path

    def replay(self, state: Any) -> None:
        """Replay the down-viewport navigation.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if state is not None and hasattr(state, "down_viewport"):
            state.down_viewport(self.path)


class DLEditGrob(DLOperation):
    """Records a grob edit operation.

    Parameters
    ----------
    grob_name : Optional[str]
        Name of the grob being edited.
    specs : Any
        Edit specifications (e.g. a :class:`GEdit` or dict of new values).
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    grob_name : Optional[str]
        Name of the target grob.
    specs : Any
        Edit specifications.
    """

    def __init__(
        self,
        grob_name: Optional[str] = None,
        specs: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            op_type="edit_grob", grob_name=grob_name, specs=specs, **kwargs
        )
        self.grob_name: Optional[str] = grob_name
        self.specs: Any = specs

    def replay(self, state: Any) -> None:
        """Replay the grob edit.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if state is not None and hasattr(state, "edit_grob"):
            state.edit_grob(self.grob_name, self.specs)


class DLSetGpar(DLOperation):
    """Records a graphical-parameter change.

    Parameters
    ----------
    gpar : Any
        The ``Gpar`` instance or dict of graphical parameters that was set.
    **kwargs : Any
        Additional parameters forwarded to :class:`DLOperation`.

    Attributes
    ----------
    gpar : Any
        The graphical parameters.
    """

    def __init__(self, gpar: Any = None, **kwargs: Any) -> None:
        super().__init__(op_type="set_gpar", gpar=gpar, **kwargs)
        self.gpar: Any = gpar

    def replay(self, state: Any) -> None:
        """Replay the gpar change.

        Parameters
        ----------
        state : Any
            The current graphics state.
        """
        if state is not None and hasattr(state, "set_gpar"):
            state.set_gpar(self.gpar)
