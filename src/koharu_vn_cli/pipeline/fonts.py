"""Font discovery and Vietnamese glyph-safe font resolution."""

from __future__ import annotations

from pathlib import Path

ASSETS_FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
WINDOWS_FONTS_DIR = Path("C:/Windows/Fonts")

FONT_ALIASES: dict[str, str] = {
    "animeace": "animeace2_reg.ttf",
    "animeace2": "animeace2_reg.ttf",
    "animeace3": "animeace2_reg.ttf",
    "teddybear": "animeace2_reg.ttf",
    "pangolin": "Pangolin-Regular.ttf",
    "patrick": "PatrickHand-Regular.ttf",
    "patrick-hand": "PatrickHand-Regular.ttf",
    "mali": "Mali-SemiBold.ttf",
    "mali-semibold": "Mali-SemiBold.ttf",
    "mali-bold": "Mali-Bold.ttf",
    "bangers": "Bangers-Regular.ttf",
    "comic": "comicbd.ttf",
    "arial": "arial.ttf",
    "tahoma": "tahoma.ttf",
    "segoe": "segoeui.ttf",
}

DEFAULT_BUBBLE_FONT = "animeace2_reg.ttf"
DEFAULT_FLOATING_FONT = "Pangolin-Regular.ttf"
DEFAULT_SFX_FONT = "Bangers-Regular.ttf"


def available_fonts() -> list[str]:
    """Return names of bundled Vietnamese-safe comic fonts."""
    if not ASSETS_FONTS_DIR.exists():
        return []
    return sorted(p.name for p in ASSETS_FONTS_DIR.glob("*.ttf"))


def resolve_font_path(font_name: str | None = None, kind: str = "bubble") -> Path:
    """Resolve a font alias, filename, or filesystem path to an existing TTF/OTF file."""
    if font_name:
        direct = Path(font_name)
        if direct.is_file():
            return direct

        key = font_name.strip().lower()
        mapped = FONT_ALIASES.get(key, font_name)

        for directory in (ASSETS_FONTS_DIR, WINDOWS_FONTS_DIR):
            candidate = directory / mapped
            if candidate.is_file():
                return candidate
            candidate_ttf = directory / f"{mapped}.ttf"
            if candidate_ttf.is_file():
                return candidate_ttf

    default_file = DEFAULT_FLOATING_FONT if kind == "floating" else DEFAULT_BUBBLE_FONT
    for candidate in (
        ASSETS_FONTS_DIR / default_file,
        ASSETS_FONTS_DIR / DEFAULT_BUBBLE_FONT,
        ASSETS_FONTS_DIR / "PatrickHand-Regular.ttf",
        WINDOWS_FONTS_DIR / "comicbd.ttf",
        WINDOWS_FONTS_DIR / "arial.ttf",
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("No usable TrueType font found for Vietnamese rendering.")
