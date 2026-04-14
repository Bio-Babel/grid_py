"""Pluggable text measurement backends for grid_py.

Provides font metrics implementations that renderers can use when they
lack native text measurement capabilities (e.g. the WebRenderer).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

__all__ = [
    "FontMetricsBackend",
    "CairoFontMetrics",
    "FonttoolsMetrics",
    "HeuristicMetrics",
    "get_font_backend",
]

# Module-level cache for the default backend.
_cached_backend: Optional["FontMetricsBackend"] = None


class FontMetricsBackend(ABC):
    """Abstract interface for text measurement."""

    @abstractmethod
    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        """Return ``{'ascent', 'descent', 'width'}`` in inches."""


# ---------------------------------------------------------------------------
# Helpers shared by concrete backends
# ---------------------------------------------------------------------------

def _extract_font_params(gp: Any) -> tuple:
    """Pull fontfamily, fontsize, fontface, and cex from a Gpar-like object.

    Returns (family, fontsize, fontface, cex).
    """
    family: Optional[str] = None
    fontsize: float = 12.0
    fontface: Any = None
    cex: float = 1.0

    if gp is not None:
        val = gp.get("fontfamily", None)
        if val is not None:
            family = str(val)
        val = gp.get("fontsize", None)
        if val is not None:
            fontsize = float(val)
        val = gp.get("fontface", None)
        if val is not None:
            fontface = val
        val = gp.get("cex", None)
        if val is not None:
            cex = float(val)

    return family, fontsize, fontface, cex


# ---------------------------------------------------------------------------
# Cairo backend
# ---------------------------------------------------------------------------

class CairoFontMetrics(FontMetricsBackend):
    """Text measurement using a Cairo context."""

    def __init__(self) -> None:
        import cairo
        self._cairo = cairo
        self._surface = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
        self._ctx = cairo.Context(self._surface)

    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        cairo = self._cairo
        ctx = self._ctx

        family, fontsize, fontface, cex = _extract_font_params(gp)

        slant = cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_NORMAL

        if fontface is not None:
            if fontface in (2, "bold"):
                weight = cairo.FONT_WEIGHT_BOLD
            elif fontface in (3, "italic"):
                slant = cairo.FONT_SLANT_ITALIC
            elif fontface in (4, "bold.italic"):
                weight = cairo.FONT_WEIGHT_BOLD
                slant = cairo.FONT_SLANT_ITALIC

        ctx.select_font_face(family or "sans-serif", slant, weight)
        ctx.set_font_size(fontsize * cex)

        fe = ctx.font_extents()
        te = ctx.text_extents(text)

        # te = (x_bearing, y_bearing, width, height, x_advance, y_advance)
        # R's GEStrWidth returns advance width (te[4]), not ink bbox (te[2]).
        return {
            "ascent": fe[0] / 72.0,
            "descent": fe[1] / 72.0,
            "width": te[4] / 72.0,
        }


# ---------------------------------------------------------------------------
# fonttools backend
# ---------------------------------------------------------------------------

class FonttoolsMetrics(FontMetricsBackend):
    """Text measurement using fontTools to read glyph advances from font files.

    More accurate than heuristic estimation when pycairo is unavailable.
    Raises ``ImportError`` if fontTools is not installed, or ``RuntimeError``
    if no font files can be found on the system.
    """

    def __init__(self) -> None:
        from fontTools.ttLib import TTFont  # noqa: F401 — validate import
        self._font_cache: dict = {}
        self._system_fonts: Optional[dict] = None

    def _find_system_fonts(self) -> dict:
        """Build a map of family -> [(path, bold, italic), ...]."""
        if self._system_fonts is not None:
            return self._system_fonts

        import glob
        import os
        from fontTools.ttLib import TTFont

        self._system_fonts = {}
        search_paths = []

        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            search_paths.append(os.path.join(conda_prefix, "fonts"))
            search_paths.append(os.path.join(conda_prefix, "lib", "fonts"))
        search_paths.extend([
            "/usr/share/fonts", "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ])
        # matplotlib bundled fonts (optional)
        try:
            import matplotlib
            search_paths.append(os.path.join(
                os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"))
        except ImportError:
            pass

        for base in search_paths:
            if not os.path.isdir(base):
                continue
            for pattern in ("**/*.ttf", "**/*.otf"):
                for path in glob.glob(os.path.join(base, pattern), recursive=True):
                    font = None
                    try:
                        font = TTFont(path, lazy=True)
                        name_table = font["name"]
                        family = None
                        for record in name_table.names:
                            if record.nameID == 1:
                                family = record.toUnicode().lower()
                                break
                        if family:
                            os2 = font.get("OS/2")
                            bold = bool(os2.fsSelection & 0x20) if os2 else False
                            italic = bool(os2.fsSelection & 0x01) if os2 else False
                            self._system_fonts.setdefault(family, []).append(
                                (path, bold, italic))
                    except Exception:
                        # Corrupt/unreadable font file — skip it
                        pass
                    finally:
                        if font is not None:
                            font.close()

        if not self._system_fonts:
            raise RuntimeError("FonttoolsMetrics: no font files found on this system")

        return self._system_fonts

    def _get_font(self, family: Optional[str], bold: bool, italic: bool):
        """Load and cache a TTFont for the given style."""
        from fontTools.ttLib import TTFont

        key = (family or "sans-serif", bold, italic)
        if key in self._font_cache:
            return self._font_cache[key]

        fonts = self._find_system_fonts()
        family_lower = (family or "sans-serif").lower()

        # Generic family aliases
        _GENERIC = {
            "sans-serif": ["dejavu sans", "liberation sans", "arial",
                           "helvetica", "source sans", "noto sans"],
            "serif": ["dejavu serif", "liberation serif", "times", "noto serif"],
            "mono": ["dejavu sans mono", "liberation mono", "courier",
                     "source code pro", "noto mono"],
            "monospace": ["dejavu sans mono", "liberation mono", "courier"],
        }

        candidates = fonts.get(family_lower)
        if candidates is None:
            for alias in _GENERIC.get(family_lower, []):
                candidates = fonts.get(alias)
                if candidates:
                    break
        if candidates is None:
            # Use any available font
            candidates = next(iter(fonts.values()))

        # Best style match
        best = candidates[0]
        for path, b, i in candidates:
            if b == bold and i == italic:
                best = (path, b, i)
                break

        font = TTFont(best[0])
        self._font_cache[key] = font
        return font

    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        family, fontsize, fontface, cex = _extract_font_params(gp)
        effective_size = fontsize * cex

        bold = fontface in (2, "bold", 4, "bold.italic")
        italic = fontface in (3, "italic", "oblique", 4, "bold.italic")

        font = self._get_font(family, bold, italic)
        cmap = font.getBestCmap()
        hmtx = font["hmtx"]
        units_per_em = font["head"].unitsPerEm

        total_advance = 0
        for char in text:
            code = ord(char)
            if code in cmap:
                advance, _ = hmtx[cmap[code]]
                total_advance += advance
            else:
                total_advance += units_per_em // 2

        width_inches = (total_advance / units_per_em) * effective_size / 72.0

        if "OS/2" in font:
            os2 = font["OS/2"]
            ascent = os2.sTypoAscender / units_per_em * effective_size / 72.0
            descent = abs(os2.sTypoDescender) / units_per_em * effective_size / 72.0
        else:
            hhea = font["hhea"]
            ascent = hhea.ascent / units_per_em * effective_size / 72.0
            descent = abs(hhea.descent) / units_per_em * effective_size / 72.0

        return {"ascent": ascent, "descent": descent, "width": width_inches}


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

class HeuristicMetrics(FontMetricsBackend):
    """Rough estimates based on character count and font size."""

    def measure(self, text: str, gp: Any = None) -> Dict[str, float]:
        _family, fontsize, _fontface, cex = _extract_font_params(gp)
        effective = fontsize * cex
        avg_char_width = effective * 0.6 / 72.0

        return {
            "ascent": effective * 0.75 / 72.0,
            "descent": effective * 0.25 / 72.0,
            "width": len(text) * avg_char_width,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_font_backend() -> FontMetricsBackend:
    """Return a cached :class:`FontMetricsBackend` instance.

    Tries CairoFontMetrics first, then FonttoolsMetrics, then HeuristicMetrics.
    """
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend

    try:
        _cached_backend = CairoFontMetrics()
        return _cached_backend
    except ImportError:
        pass

    try:
        _cached_backend = FonttoolsMetrics()
        return _cached_backend
    except (ImportError, RuntimeError):
        pass

    _cached_backend = HeuristicMetrics()
    return _cached_backend
