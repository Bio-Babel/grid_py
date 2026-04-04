"""Glyph/typesetting support for grid_py (port of R's grid ``typeset.R``).

This module provides grob constructors for rendering pre-typeset glyph
information, mirroring R's ``glyphGrob()`` and ``grid.glyph()`` functions.
A *glyph grob* wraps a ``GlyphInfo`` object together with position,
justification, and graphical parameters, allowing the grid drawing
pipeline to render individual glyphs at specified locations.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

from ._gpar import Gpar
from ._grob import Grob, grob_name
from ._units import Unit, is_unit

__all__ = [
    "GlyphJust",
    "glyph_just",
    "GlyphInfo",
    "glyph_grob",
    "grid_glyph",
]


# ---------------------------------------------------------------------------
# Glyph justification
# ---------------------------------------------------------------------------


class GlyphJust:
    """Wrapper for glyph justification values.

    In R, glyph justification can be a numeric proportion (0 = left/bottom,
    1 = right/top) or a named anchor string (e.g. ``"left"``, ``"centre"``).
    This class tags the value so that downstream code can distinguish between
    numeric proportional justification and named-anchor justification.

    Parameters
    ----------
    value : float, int, or str
        The justification value.  Strings such as ``"left"``, ``"centre"``,
        ``"center"``, ``"right"``, ``"top"``, ``"bottom"`` are accepted as
        named anchors.  Numeric values are interpreted as proportional
        offsets (0.0 to 1.0).
    name : str or None, optional
        An optional width/height name to associate with a numeric
        justification (mirrors R's ``names(hjust)``).

    Attributes
    ----------
    value : float or str
        The justification value.
    name : str or None
        Optional name qualifier.
    """

    __slots__ = ("value", "name")

    def __init__(
        self,
        value: Union[float, int, str],
        name: Optional[str] = None,
    ) -> None:
        self.value: Union[float, str] = value
        self.name: Optional[str] = name

    def __repr__(self) -> str:
        if self.name is not None:
            return f"GlyphJust(value={self.value!r}, name={self.name!r})"
        return f"GlyphJust(value={self.value!r})"

    @property
    def is_numeric(self) -> bool:
        """Return ``True`` if this justification is a numeric proportion."""
        return isinstance(self.value, (int, float))


def glyph_just(value: Union[float, int, str, "GlyphJust"]) -> GlyphJust:
    """Normalise a justification value into a :class:`GlyphJust`.

    Parameters
    ----------
    value : float, int, str, or GlyphJust
        If already a ``GlyphJust``, return as-is.  Strings are mapped to
        canonical anchor names; numeric values are wrapped directly.

    Returns
    -------
    GlyphJust
        A validated glyph-justification object.

    Raises
    ------
    TypeError
        If *value* is not a recognised type.
    """
    if isinstance(value, GlyphJust):
        return value

    if isinstance(value, str):
        canonical = _JUST_ALIASES.get(value.lower(), value.lower())
        return GlyphJust(canonical)

    if isinstance(value, (int, float)):
        return GlyphJust(float(value))

    raise TypeError(
        f"'hjust'/'vjust' must be numeric or a string, got {type(value).__name__}"
    )


_JUST_ALIASES: Dict[str, str] = {
    "center": "centre",
    "left": "left",
    "right": "right",
    "top": "top",
    "bottom": "bottom",
    "centre": "centre",
}


# ---------------------------------------------------------------------------
# GlyphInfo (minimal container)
# ---------------------------------------------------------------------------


class GlyphInfo:
    """Container for pre-typeset glyph information.

    This is a lightweight Python analogue of R's ``RGlyphInfo`` objects
    produced by typesetting engines (e.g. ``systemfonts::shape_string``).
    It carries per-glyph positions, font information, and overall bounding
    metrics.

    Parameters
    ----------
    glyphs : dict
        Per-glyph data.  Expected keys include ``"x"``, ``"y"``, and
        optionally ``"font"``, ``"size"``, ``"rot"``, ``"colour"``.
    width : dict or float
        Overall width(s) of the typeset block in big points.  May be a
        dict mapping names to widths.
    height : dict or float
        Overall height(s) of the typeset block in big points.
    h_anchor : dict or None, optional
        Named horizontal anchors (e.g. ``{"left": 0, "right": 100}``).
    v_anchor : dict or None, optional
        Named vertical anchors (e.g. ``{"bottom": 0, "top": 80}``).

    Attributes
    ----------
    glyphs : dict
    width : dict or float
    height : dict or float
    h_anchor : dict
    v_anchor : dict
    """

    def __init__(
        self,
        glyphs: Dict[str, Any],
        width: Union[Dict[str, float], float],
        height: Union[Dict[str, float], float],
        h_anchor: Optional[Dict[str, float]] = None,
        v_anchor: Optional[Dict[str, float]] = None,
    ) -> None:
        self.glyphs = glyphs
        self.width = width
        self.height = height
        self.h_anchor = h_anchor if h_anchor is not None else {"left": 0.0}
        self.v_anchor = v_anchor if v_anchor is not None else {"bottom": 0.0}

    def __repr__(self) -> str:
        n = len(self.glyphs.get("x", []))
        return f"GlyphInfo(n_glyphs={n})"


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _valid_glyph_grob(x: Grob) -> Grob:
    """Validate a glyph grob (mirrors R's ``validDetails.glyphgrob``).

    Parameters
    ----------
    x : Grob
        The grob to validate.

    Returns
    -------
    Grob
        The validated grob (unchanged if valid).

    Raises
    ------
    TypeError
        If *glyphInfo* is not a :class:`GlyphInfo`, or if *x*/*y* are not
        units, or if justification values are invalid.
    ValueError
        If *x*/*y* have length < 1, or justification values have length != 1.
    """
    glyph_info = getattr(x, "glyphInfo", None)
    if not isinstance(glyph_info, GlyphInfo):
        raise TypeError("Invalid glyph info; expected a GlyphInfo instance")

    grob_x = getattr(x, "x", None)
    grob_y = getattr(x, "y", None)
    if not is_unit(grob_x) or not is_unit(grob_y):
        raise TypeError("'x' and 'y' must be Unit objects")

    if len(grob_x) < 1 or len(grob_y) < 1:
        raise ValueError("'x' and 'y' must have length > 0")

    hjust = getattr(x, "hjust", None)
    vjust = getattr(x, "vjust", None)
    if not isinstance(hjust, GlyphJust) or not isinstance(vjust, GlyphJust):
        raise TypeError("'hjust' and 'vjust' must be GlyphJust values")

    return x


# ---------------------------------------------------------------------------
# Glyph grob constructor
# ---------------------------------------------------------------------------


def glyph_grob(
    glyphInfo: GlyphInfo,
    x: Union[float, Unit] = 0.5,
    y: Union[float, Unit] = 0.5,
    default_units: str = "npc",
    hjust: Union[float, str, GlyphJust] = "centre",
    vjust: Union[float, str, GlyphJust] = "centre",
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    name: Optional[str] = None,
) -> Grob:
    """Create a glyph grob for rendering pre-typeset glyph information.

    This mirrors R's ``glyphGrob()`` function.  The resulting :class:`Grob`
    has ``_grid_class="glyphgrob"`` and carries the glyph data, position,
    and justification as attributes.

    Parameters
    ----------
    glyphInfo : GlyphInfo
        The pre-typeset glyph information to render.
    x : float or Unit, optional
        Horizontal position of the glyph block (default ``0.5``).
    y : float or Unit, optional
        Vertical position of the glyph block (default ``0.5``).
    default_units : str, optional
        Unit type for *x* and *y* when they are plain numbers
        (default ``"npc"``).
    hjust : float, str, or GlyphJust, optional
        Horizontal justification (default ``"centre"``).
    vjust : float, str, or GlyphJust, optional
        Vertical justification (default ``"centre"``).
    gp : Gpar or None, optional
        Graphical parameters.
    vp : object or None, optional
        Viewport.
    name : str or None, optional
        Grob name.  Auto-generated when ``None``.

    Returns
    -------
    Grob
        A grob with ``_grid_class="glyphgrob"`` ready for drawing.

    Raises
    ------
    TypeError
        If *glyphInfo* is not a :class:`GlyphInfo`.

    Examples
    --------
    >>> info = GlyphInfo({"x": [0], "y": [0]}, width=10.0, height=12.0)
    >>> g = glyph_grob(info)
    >>> g._grid_class
    'glyphgrob'
    """
    # Coerce x/y to Unit if needed
    if not is_unit(x):
        x = Unit(x, default_units)
    if not is_unit(y):
        y = Unit(y, default_units)

    # Normalise justification
    hjust_val = glyph_just(hjust)
    vjust_val = glyph_just(vjust)

    grob_obj = Grob(
        name=name,
        gp=gp if gp is not None else Gpar(),
        vp=vp,
        _grid_class="glyphgrob",
        glyphInfo=glyphInfo,
        x=x,
        y=y,
        hjust=hjust_val,
        vjust=vjust_val,
    )

    # Validate
    _valid_glyph_grob(grob_obj)

    return grob_obj


# ---------------------------------------------------------------------------
# grid.glyph equivalent
# ---------------------------------------------------------------------------


def grid_glyph(
    glyphInfo: GlyphInfo,
    x: Union[float, Unit] = 0.5,
    y: Union[float, Unit] = 0.5,
    default_units: str = "npc",
    hjust: Union[float, str, GlyphJust] = "centre",
    vjust: Union[float, str, GlyphJust] = "centre",
    gp: Optional[Gpar] = None,
    vp: Optional[Any] = None,
    name: Optional[str] = None,
    draw: bool = True,
) -> Grob:
    """Create and optionally draw a glyph grob.

    This is the high-level interface mirroring R's ``grid.glyph()``.  It
    constructs a glyph grob via :func:`glyph_grob` and, when *draw* is
    ``True``, immediately renders it by calling ``grid_draw``.

    Parameters
    ----------
    glyphInfo : GlyphInfo
        The pre-typeset glyph information.
    x : float or Unit, optional
        Horizontal position (default ``0.5``).
    y : float or Unit, optional
        Vertical position (default ``0.5``).
    default_units : str, optional
        Unit type when *x*/*y* are plain numbers (default ``"npc"``).
    hjust : float, str, or GlyphJust, optional
        Horizontal justification (default ``"centre"``).
    vjust : float, str, or GlyphJust, optional
        Vertical justification (default ``"centre"``).
    gp : Gpar or None, optional
        Graphical parameters.
    vp : object or None, optional
        Viewport.
    name : str or None, optional
        Grob name.
    draw : bool, optional
        If ``True`` (default), the grob is drawn immediately.

    Returns
    -------
    Grob
        The glyph grob (returned invisibly in R; here simply returned).
    """
    g = glyph_grob(
        glyphInfo=glyphInfo,
        x=x,
        y=y,
        default_units=default_units,
        hjust=hjust,
        vjust=vjust,
        gp=gp,
        vp=vp,
        name=name,
    )

    if draw:
        from ._draw import grid_draw

        grid_draw(g)

    return g
