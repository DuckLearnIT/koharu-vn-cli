"""Data schemas for koharu-vn-cli translation, CG composition, and JSON sidecar caching."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ComposePreset(str, Enum):
    """Preset styles for AI dialogue composition on textless CG artwork."""

    CG_OVAL = "cg-oval"
    MANGA_TAIL = "manga-tail"
    MIXED = "mixed"
    FLOATING_ONLY = "floating-only"


class BubbleStyle(str, Enum):
    """Visual bubble / text container styles."""

    WHITE_OVAL = "white_oval"
    DARK_TRANSLUCENT = "dark_translucent"
    SPEECH_TAIL = "speech_tail"
    NONE = "none"


class TextRegionItem(BaseModel):
    """One detected or composed dialogue / SFX region on a page."""

    id: int
    kind: Literal["bubble", "floating"] = "bubble"
    bubble_style: BubbleStyle = BubbleStyle.NONE
    text_bbox: tuple[int, int, int, int]
    bubble_bbox: tuple[int, int, int, int] | None = None
    bubble_contour: list[tuple[float, float]] | None = None
    tail_target: tuple[int, int] | None = None
    source_text: str = ""
    translated_text: str = ""
    text_color: tuple[int, int, int] | None = None
    stroke_color: tuple[int, int, int] | None = None
    stroke_width: float | None = None
    font_name: str | None = None
    font_size: float | None = None
    angle: float = 0.0
    alignment: Literal["center", "left", "right"] = "center"
    is_dark_bubble: bool = False
    draw_bubble: bool = False
    confidence: float = 1.0


class CharacterProfile(BaseModel):
    """Character visual traits and locked Vietnamese pronoun rules across a chapter/CG set."""

    name: str
    appearance: str = ""
    role_and_tone: str = ""
    pronoun_rules: str = ""


class StoryBible(BaseModel):
    """Global story context built automatically from full image(s) before page-by-page generation."""

    title: str = ""
    source_game: str = ""
    genre_and_tone: str = ""
    pre_event_backstory: str = ""
    during_event_plot: str = ""
    full_plot_summary: str = ""
    characters: list[CharacterProfile] = Field(default_factory=list)
    page_beats: dict[str, str] = Field(default_factory=dict)

    def to_prompt_block(self, page_index: int | None = None) -> str:
        lines: list[str] = []
        if self.title or self.genre_and_tone:
            lines.append(f"Tác phẩm / Thể loại: {self.title} ({self.genre_and_tone})".strip())
        if self.source_game:
            lines.append(f"Game / Series gốc: {self.source_game}")
        if self.pre_event_backstory:
            lines.append(f"Cốt truyện TRƯỚC sự kiện ảnh: {self.pre_event_backstory}")
        if self.during_event_plot:
            lines.append(f"Diễn biến TRONG sự kiện ảnh: {self.during_event_plot}")
        if self.full_plot_summary:
            lines.append(f"Mạch truyện tổng thể: {self.full_plot_summary}")
        if self.characters:
            lines.append("Nhân vật & Quy tắc xưng hô BẮT BUỘC giữ nhất quán:")
            for c in self.characters:
                traits = f" [{c.appearance}]" if c.appearance else ""
                tone = f" — {c.role_and_tone}" if c.role_and_tone else ""
                pron = f" => Xưng hô: {c.pronoun_rules}" if c.pronoun_rules else ""
                lines.append(f"  * {c.name}{traits}{tone}{pron}")
        if page_index is not None and self.page_beats:
            nearby = []
            for offset in (-1, 0, 1):
                k = str(page_index + offset)
                if k in self.page_beats:
                    tag = "Trang trước" if offset == -1 else ("Trang NÀY" if offset == 0 else "Trang kế")
                    nearby.append(f"  - {tag} ({k}): {self.page_beats[k]}")
            if nearby:
                lines.append("Tình huống kề cận:")
                lines.extend(nearby)
        return "\n".join(lines)


class PageDocument(BaseModel):
    """Serializable page state stored as *.koharu.json for instant rerendering."""

    version: int = 1
    image_path: str
    cleaned_image_path: str | None = None
    width: int
    height: int
    mode: Literal["translate", "compose"]
    source_language: str = "auto"
    page_summary: str = ""
    regions: list[TextRegionItem] = Field(default_factory=list)
