"""Graphical parameters for grid_py (port of R's grid ``gpar`` system).

This module provides the :class:`Gpar` class, which encapsulates a set of
graphical parameters analogous to R's ``gpar()`` objects.  Individual
parameters may be scalars **or** vectors (lists); when vectorised the
recycling / subscripting semantics of R are preserved.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["Gpar", "get_gpar"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_LTY: set[str] = {
    "solid",
    "dashed",
    "dotted",
    "dotdash",
    "longdash",
    "twodash",
}

_VALID_LINEEND: set[str] = {"round", "butt", "square"}

_VALID_LINEJOIN: set[str] = {"round", "mitre", "bevel"}

_FONTFACE_MAP: dict[str, int] = {
    "plain": 1,
    "bold": 2,
    "italic": 3,
    "oblique": 3,
    "bold.italic": 4,
    "symbol": 5,
    "cyrillic": 5,
    "cyrillic.oblique": 6,
    "EUC": 7,
}

# The set of parameter names that the Gpar constructor accepts.
_GPAR_NAMES: set[str] = {
    "col",
    "fill",
    "alpha",
    "lty",
    "lwd",
    "lex",
    "lineend",
    "linejoin",
    "linemitre",
    "fontsize",
    "cex",
    "fontfamily",
    "fontface",
    "lineheight",
    "font",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list:
    """Wrap scalars in a list; pass through sequences unchanged."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _is_hex_lty(s: str) -> bool:
    """Return True when *s* looks like a valid hex-string line-type spec."""
    return all(c in "0123456789abcdefABCDEF" for c in s) and len(s) > 0


def _resolve_fontface(value: Any) -> int:
    """Convert a fontface specification to an integer code.

    Parameters
    ----------
    value : int, str, or float
        A fontface specification.  Strings are mapped through
        ``_FONTFACE_MAP``; numeric values are cast to ``int``.

    Returns
    -------
    int
        The integer font-face code.

    Raises
    ------
    ValueError
        If the string is not a recognised face name.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    if isinstance(value, str):
        if value in _FONTFACE_MAP:
            return _FONTFACE_MAP[value]
        raise ValueError(f"invalid fontface '{value}'")
    raise TypeError(f"fontface must be int or str, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Gpar class
# ---------------------------------------------------------------------------


class Gpar:
    """A set of graphical parameters (port of R's ``gpar``).

    Parameters
    ----------
    col : str or list of str, optional
        Line / border colour(s).
    fill : str or list of str, optional
        Fill colour(s) or pattern(s).
    alpha : float or list of float, optional
        Transparency value(s) in the range ``[0, 1]``.
    lty : str, int, or list, optional
        Line type.  One of ``"solid"``, ``"dashed"``, ``"dotted"``,
        ``"dotdash"``, ``"longdash"``, ``"twodash"``, a hex string, or an
        integer code.
    lwd : float or list of float, optional
        Line width(s).
    lex : float or list of float, optional
        Line-width expansion multiplier(s).
    lineend : str or list of str, optional
        Line end style: ``"round"``, ``"butt"``, or ``"square"``.
    linejoin : str or list of str, optional
        Line join style: ``"round"``, ``"mitre"``, or ``"bevel"``.
    linemitre : float or list of float, optional
        Mitre limit (must be >= 1).
    fontsize : float or list of float, optional
        Font size in points.
    cex : float or list of float, optional
        Character expansion factor.
    fontfamily : str or list of str, optional
        Font family name(s).
    fontface : int, str, or list, optional
        Font face specification.  Mapped to an integer code internally.
        Cannot be specified together with *font*.
    lineheight : float or list of float, optional
        Line-height multiplier.
    font : int or list of int, optional
        Integer font-face code (alias for *fontface*).  Cannot be specified
        together with *fontface*.

    Raises
    ------
    TypeError
        If a parameter has an inappropriate type.
    ValueError
        If a parameter value is out of range or not among valid choices.

    Examples
    --------
    >>> gp = Gpar(col="red", lwd=2, alpha=0.5)
    >>> gp
    Gpar(col='red', lwd=2, alpha=0.5)
    """

    # Slots keep instances lightweight.
    __slots__ = ("_params",)

    def __init__(self, **kwargs: Any) -> None:
        # Reject unknown parameter names early.
        unknown = set(kwargs) - _GPAR_NAMES
        if unknown:
            raise TypeError(
                f"unknown graphical parameter(s): {', '.join(sorted(unknown))}"
            )

        params: Dict[str, Any] = {}

        # -- fontface / font mutual exclusion (mirrors R) ------------------
        if "fontface" in kwargs and "font" in kwargs:
            raise ValueError("must specify only one of 'font' and 'fontface'")

        if "fontface" in kwargs:
            ff = kwargs.pop("fontface")
            if ff is not None:
                ff_list = _as_list(ff)
                if len(ff_list) == 0:
                    raise ValueError("'gpar' element 'fontface' must not be length 0")
                resolved = [_resolve_fontface(v) for v in ff_list]
                params["font"] = resolved[0] if len(resolved) == 1 else resolved
            # fontface is consumed; do not store it directly.

        # -- process remaining parameters ----------------------------------
        for name, value in kwargs.items():
            if value is None:
                continue

            vals = _as_list(value)

            if len(vals) == 0:
                raise ValueError(
                    f"'gpar' element '{name}' must not be length 0"
                )

            # --- per-parameter validation ---------------------------------
            if name in ("fontsize", "lineheight", "cex", "lwd", "lex"):
                try:
                    vals = [float(v) for v in vals]
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"'{name}' must be numeric, got {type(value).__name__}"
                    ) from exc

            elif name == "alpha":
                try:
                    vals = [float(v) for v in vals]
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"'alpha' must be numeric, got {type(value).__name__}"
                    ) from exc
                if any(v < 0 or v > 1 for v in vals):
                    raise ValueError("invalid 'alpha' value (must be 0-1)")

            elif name == "linemitre":
                try:
                    vals = [float(v) for v in vals]
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"'linemitre' must be numeric, got {type(value).__name__}"
                    ) from exc
                if any(v < 1 for v in vals):
                    raise ValueError("invalid 'linemitre' value (must be >= 1)")

            elif name == "font":
                try:
                    vals = [int(v) for v in vals]
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"'font' must be integer, got {type(value).__name__}"
                    ) from exc

            elif name == "lty":
                for v in vals:
                    if isinstance(v, str):
                        if v not in _VALID_LTY and not _is_hex_lty(v):
                            raise ValueError(
                                f"invalid line type '{v}'; must be one of "
                                f"{sorted(_VALID_LTY)} or a hex string"
                            )
                    elif not isinstance(v, (int, float, np.integer, np.floating)):
                        raise TypeError(
                            f"'lty' must be str or numeric, got {type(v).__name__}"
                        )

            elif name == "lineend":
                for v in vals:
                    if v not in _VALID_LINEEND:
                        raise ValueError(
                            f"invalid 'lineend' value '{v}'; "
                            f"must be one of {sorted(_VALID_LINEEND)}"
                        )

            elif name == "linejoin":
                for v in vals:
                    if v not in _VALID_LINEJOIN:
                        raise ValueError(
                            f"invalid 'linejoin' value '{v}'; "
                            f"must be one of {sorted(_VALID_LINEJOIN)}"
                        )

            elif name == "fontfamily":
                vals = [str(v) for v in vals]

            elif name in ("col", "fill"):
                # Accept strings or lists of strings; no further validation
                # here (colour resolution is deferred to the rendering
                # backend, matching R's behaviour).
                pass

            # Store single-element lists as scalars for cleaner repr.
            params[name] = vals[0] if len(vals) == 1 else vals

        self._params = params

    # -- dict-like access --------------------------------------------------

    @property
    def params(self) -> Dict[str, Any]:
        """Return a **copy** of the underlying parameter dictionary.

        Returns
        -------
        dict
            Mapping of parameter names to their values.
        """
        return dict(self._params)

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a single parameter value.

        Parameters
        ----------
        name : str
            Parameter name.
        default : object, optional
            Value returned when *name* is not set.

        Returns
        -------
        object
            The parameter value, or *default*.
        """
        return self._params.get(name, default)

    def set(self, name: str, value: Any) -> None:
        """Set a single parameter value.

        Parameters
        ----------
        name : str
            Parameter name.
        value : object
            The value to set.
        """
        self._params[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def names(self) -> List[str]:
        """Return the names of parameters currently set.

        Returns
        -------
        list of str
        """
        return list(self._params.keys())

    # -- length & subscripting ---------------------------------------------

    def __len__(self) -> int:
        """Return the maximum length across all vectorised parameters.

        Returns
        -------
        int
            0 if no parameters are set; otherwise ``max(len(v))`` over all
            parameters (scalars count as length 1).
        """
        if not self._params:
            return 0
        return max(
            len(v) if isinstance(v, list) else 1
            for v in self._params.values()
        )

    def __getitem__(self, index: int) -> "Gpar":
        """Subscript the Gpar, returning a new Gpar with the *index*-th element.

        Vector parameters are recycled to the maximum length (matching R's
        ``[.gpar`` method) before the element is selected.

        Parameters
        ----------
        index : int
            Zero-based index into the (recycled) parameter vectors.

        Returns
        -------
        Gpar
            A new ``Gpar`` containing scalar values for each parameter.

        Raises
        ------
        IndexError
            If *index* is out of range after recycling.
        """
        n = len(self)
        if n == 0:
            return Gpar()

        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Gpar index {index} out of range [0, {n})")

        new_params: Dict[str, Any] = {}
        for name, value in self._params.items():
            if isinstance(value, list):
                # Recycle to length n, then pick element.
                recycled = (value * math.ceil(n / len(value)))[:n]
                new_params[name] = recycled[index]
            else:
                new_params[name] = value
        # Build via internal path to skip re-validation.
        gp = object.__new__(Gpar)
        gp._params = new_params
        return gp

    # -- merge -------------------------------------------------------------

    def _merge(self, parent: "Gpar") -> "Gpar":
        """Merge with a *parent* ``Gpar`` (child overrides parent).

        Parameters that are present in *self* take precedence.  Parameters
        only present in *parent* are inherited.  ``cex``, ``alpha``, and
        ``lex`` are **cumulative** — the child value is multiplied by the
        parent value, matching R's ``set.gpar`` semantics.

        Parameters
        ----------
        parent : Gpar
            The parent graphical parameters to merge with.

        Returns
        -------
        Gpar
            A new ``Gpar`` containing the merged parameters.
        """
        merged = copy.deepcopy(parent._params)
        merged.update(copy.deepcopy(self._params))

        # Cumulative parameters
        for cum_name in ("cex", "alpha", "lex"):
            if cum_name in self._params and cum_name in parent._params:
                child_val = self._params[cum_name]
                parent_val = parent._params[cum_name]
                if isinstance(child_val, list) or isinstance(parent_val, list):
                    c_list = _as_list(child_val)
                    p_list = _as_list(parent_val)
                    maxn = max(len(c_list), len(p_list))
                    c_cyc = (c_list * math.ceil(maxn / len(c_list)))[:maxn]
                    p_cyc = (p_list * math.ceil(maxn / len(p_list)))[:maxn]
                    result = [c * p for c, p in zip(c_cyc, p_cyc)]
                    merged[cum_name] = result[0] if len(result) == 1 else result
                else:
                    merged[cum_name] = child_val * parent_val

        gp = object.__new__(Gpar)
        gp._params = merged
        return gp

    # -- display -----------------------------------------------------------

    def __repr__(self) -> str:
        if not self._params:
            return "Gpar()"
        items = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"Gpar({items})"

    def __str__(self) -> str:
        return self.__repr__()

    # -- equality (useful for testing) -------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gpar):
            return NotImplemented
        return self._params == other._params


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _default_gpar() -> Gpar:
    """Return a ``Gpar`` populated with R's default graphical parameters.

    Returns
    -------
    Gpar
        Default graphical parameters matching R's internal defaults:
        ``fontsize=12``, ``cex=1``, ``fontfamily=""``, ``fontface=1``
        (plain), ``lineheight=1.2``, ``col="black"``, ``fill="transparent"``,
        ``alpha=1``, ``lwd=1``, ``lex=1``, ``lty="solid"``,
        ``lineend="round"``, ``linejoin="round"``, ``linemitre=10``.
    """
    return Gpar(
        fontsize=12,
        cex=1,
        fontfamily="",
        fontface=1,
        lineheight=1.2,
        col="black",
        fill="transparent",
        alpha=1.0,
        lwd=1.0,
        lex=1.0,
        lty="solid",
        lineend="round",
        linejoin="round",
        linemitre=10.0,
    )


def get_gpar(names: Optional[Sequence[str]] = None) -> Gpar:
    """Return the current graphical parameters.

    This is the Python equivalent of R's ``get.gpar()``.  In the current
    implementation no graphics device state is tracked, so the function
    simply returns the default ``Gpar``.

    Parameters
    ----------
    names : sequence of str, optional
        If provided, only the listed parameter names are returned.  All
        names must be valid ``Gpar`` parameter names.

    Returns
    -------
    Gpar
        A ``Gpar`` instance with the requested (or all default) parameters.

    Raises
    ------
    ValueError
        If any element of *names* is not a valid gpar name.

    Examples
    --------
    >>> gp = get_gpar()
    >>> gp.get("fontsize")
    12.0
    >>> get_gpar(names=["col", "lwd"])
    Gpar(col='black', lwd=1.0)
    """
    default = _default_gpar()
    if names is None:
        return default

    invalid = set(names) - _GPAR_NAMES
    if invalid:
        raise ValueError(
            f"invalid gpar name(s): {', '.join(sorted(invalid))}"
        )

    subset: Dict[str, Any] = {}
    for n in names:
        if n in default._params:
            subset[n] = default._params[n]
    gp = object.__new__(Gpar)
    gp._params = subset
    return gp
