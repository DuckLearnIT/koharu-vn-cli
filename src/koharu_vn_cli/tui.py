"""OpenCode / Claude Code style interactive AI CLI UI for Koharu-VN."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, ProgressBar, RichLog, Static
from textual.widgets.option_list import Option

from koharu_vn_cli import __version__
from koharu_vn_cli.config import CONFIG_FILE, AppConfig, load_config, parse_api_keys, save_config
from koharu_vn_cli.models import CharacterProfile, ComposePreset, PageDocument, StoryBible
from koharu_vn_cli.pipeline.detector import active_ort_provider_label
from koharu_vn_cli.pipeline.fonts import available_fonts
from koharu_vn_cli.pipeline.gemini_client import VisionLLMClient
from koharu_vn_cli.pipeline.runner import (
    IMAGE_EXTS,
    ModelCache,
    PipelineCancelledError,
    PipelineHooks,
    pack_images_to_webp,
    run_compose_pipeline,
    run_rerender_pipeline,
    run_translate_pipeline,
)


@dataclass(frozen=True)
class SlashSpec:
    name: str
    args: str
    desc: str


SLASH_COMMANDS: tuple[SlashSpec, ...] = (
    SlashSpec("/translate", "<đường_dẫn> [--resume]", "Dịch truyện tranh (Nhật/Hàn/Trung/Anh → Tiếng Việt)"),
    SlashSpec("/compose", "<đường_dẫn> [--resume]", "Tự sáng tác thoại cho Artist CG không lời"),
    SlashSpec("/resume", "[đường_dẫn]", "Chạy tiếp tác vụ bị dừng/lỗi từ trang chưa xong (bỏ qua trang đã dịch)"),
    SlashSpec("/rerender", "<thư_mục_output>", "Render lại ảnh từ .koharu.json (0 giây API)"),
    SlashSpec("/rename", "<thư_mục> [--in-place]", "Đổi tên theo thứ tự đọc (001.webp...) & nén WebP max quality (100)"),
    SlashSpec("/format", "", "Chọn định dạng xuất (webp 100% / original) & đánh số 001.webp"),
    SlashSpec("/chars", "[on|off]", "Bật/tắt bước kiểm tra & sửa tên nhân vật game sau Pass 1"),
    SlashSpec("/vision", "[on|off]", "Bật/tắt gửi ảnh cho model con (tắt = OCR cục bộ + dịch theo Story Bible)"),
    SlashSpec("/gpu", "[on|off]", "Bật/tắt tăng tốc GPU (CUDA) cho Detector, LaMa Inpaint & OCR"),
    SlashSpec("/style", "", "Chọn 1 trong 2/4 kiểu sáng tác thoại CG (cg-oval / manga-tail)"),
    SlashSpec("/lang", "", "Chọn ngôn ngữ nguồn ưu tiên (auto / ja / ko / zh / en)"),
    SlashSpec("/font", "", "Chọn font Tiếng Việt mặc định"),
    SlashSpec("/density", "", "Chọn mật độ thoại khi sáng tác CG (low / medium / high)"),
    SlashSpec("/model", "", "Chọn Model PHỤ (dịch & sáng tác thoại nhanh từ Story Bible)"),
    SlashSpec("/story-model", "[tên_model|off]", "Chọn Model CHÍNH (mạnh) cho Pass 1 cốt truyện & nhân vật"),
    SlashSpec("/key", "", "Cài đặt GEMINI_API_KEY & --base-url tùy chọn"),
    SlashSpec("/prompt", "[nội_dung]", "Đặt gợi ý xưng hô / văn phong mặc định"),
    SlashSpec("/tags", "[vanilla, NTR, ...]", "Đặt tag thể loại cho sáng tác CG & dịch (vd: vanilla, NTR)"),
    SlashSpec("/story", "[chữ_hoặc_file.txt]", "Nạp cốt truyện / tên nhân vật / lore trước khi dịch"),
    SlashSpec("/models", "", "Kiểm tra & tải trước các model ONNX vào cache"),
    SlashSpec("/open", "", "Mở thư mục ảnh kết quả gần nhất trong Explorer"),
    SlashSpec("/theme", "", "Đổi giao diện màu (tokyo-night / catppuccin-mocha / ...)"),
    SlashSpec("/clear", "", "Xóa lịch sử khung hiển thị"),
    SlashSpec("/help", "", "Xem hướng dẫn lệnh và phím tắt"),
    SlashSpec("/exit", "", "Thoát Koharu-VN CLI"),
)

STYLE_CHOICES: list[tuple[str, str]] = [
    ("cg-oval", "CG Oval (Recommended) — Bubble oval trắng / đen mờ + chữ thường"),
    ("manga-tail", "Manga Tail — Bubble có đuôi nhọn chỉ nhân vật + oval + chữ thường"),
    ("mixed", "Mixed — Tự do kết hợp oval, đuôi nhọn & chữ thường"),
    ("floating-only", "Floating Only — Chỉ chữ thường & SFX trực tiếp lên tranh"),
]

LANG_CHOICES: list[tuple[str, str]] = [
    ("auto", "Tự động nhận diện (Nhật / Hàn / Trung / Anh)"),
    ("ja", "Tiếng Nhật (Manga / Doujinshi)"),
    ("ko", "Tiếng Hàn (Manhwa / Webtoon)"),
    ("zh", "Tiếng Trung (Manhua)"),
    ("en", "Tiếng Anh"),
]

DENSITY_CHOICES: list[tuple[str, str]] = [
    ("low", "Ít (1–3 câu thoại, chừa nhiều tranh)"),
    ("medium", "Vừa phải (3–5 câu cân đối, Recommended)"),
    ("high", "Nhiều (5–8 câu gồm thoại + suy nghĩ + SFX)"),
]

MODEL_CHOICES: list[tuple[str, str]] = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash (Ổn định mới nhất, OCR & dịch/CG cực chuẩn — Recommended)"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite (Siêu nhanh, tiết kiệm)"),
    ("gemini-3.6-flash", "Gemini 3.6 Flash (Thông minh, độ trễ thấp)"),
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview (Lập luận & văn phong sâu)"),
    ("gemini-4-hydrogen", "Gemini 4 Hydrogen (Frontier reasoning đa phương thức)"),
    ("gemini-4-argon", "Gemini 4 Argon (Frontier hiệu năng cao)"),
    ("gemini-4-carbon", "Gemini 4 Carbon (Frontier flagship)"),
]

THEME_CHOICES: list[tuple[str, str]] = [
    ("tokyo-night", "Tokyo Night (OpenCode default dark)"),
    ("catppuccin-mocha", "Catppuccin Mocha (Dịu mắt)"),
    ("dracula", "Dracula"),
    ("nord", "Nord"),
    ("textual-dark", "Textual Dark"),
]


def strip_path_quotes(raw: str) -> str:
    """Clean dragged-and-dropped Windows Terminal / Explorer path syntax."""
    s = raw.strip()
    if s.startswith("& "):
        s = s[2:].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1]
    return s.strip()


def extract_path_and_remainder(text: str) -> tuple[str | None, str]:
    """
    Find a Windows/Unix file or folder path (quoted, `@prefixed`, or drive-letter) inside user text
    and return `(clean_path, remaining_instructions)`.
    """
    cleaned = text.strip()
    # 1. Explicit quoted path "..." or '...' or & '...'
    m_quoted = re.search(r'(?:&\s*)?(["\'])([^"\']+)\1', cleaned)
    if m_quoted:
        cand = m_quoted.group(2).strip().lstrip("@")
        if cand:
            rem = (cleaned[: m_quoted.start()] + " " + cleaned[m_quoted.end() :]).strip()
            return cand, rem

    # 2. Windows drive path (possibly containing spaces up to known keyword or EOL)
    m_win = re.search(r'@?([A-Za-z]:[\\/][^\n"\']+)', cleaned)
    if m_win:
        raw_p = m_win.group(1).strip()
        # Try shrinking from right by space-separated tokens until an existing path matches
        tokens = raw_p.split()
        best_p: str | None = None
        best_idx = len(tokens)
        for i in range(len(tokens), 0, -1):
            probe = " ".join(tokens[:i])
            if Path(probe).exists():
                best_p = probe
                best_idx = i
                break
        if best_p is None:
            best_p = tokens[0]
            best_idx = 1
        rem_left = cleaned[: m_win.start()].strip()
        rem_right = " ".join(tokens[best_idx:]).strip()
        return best_p, f"{rem_left} {rem_right}".strip()

    # 3. @path or relative token containing / or image/archive suffix
    for tok in cleaned.split():
        cand = tok.lstrip("@")
        if (
            cand.startswith((".", "~", "/", "\\"))
            or any(cand.lower().endswith(ext) for ext in (*IMAGE_EXTS, ".zip", ".cbz"))
            or Path(cand).exists()
        ):
            rem = cleaned.replace(tok, "", 1).strip()
            return str(Path(cand).expanduser()), rem

    return None, cleaned


def parse_natural_intent(text: str) -> tuple[str, str | None, dict[str, Any]]:
    """
    Interpret natural Vietnamese prompt into `(action, path, extra_opts)`.
    `action` is one of: 'translate', 'compose', 'rerender', 'pick_action', 'chat'.
    """
    path_str, rem = extract_path_and_remainder(text)
    low = rem.lower() + " " + text.lower()

    if any(k in low for k in ("rerender", "render lại", "vẽ lại")):
        return "rerender", path_str, {"prompt": rem}

    style_override: ComposePreset | None = None
    if any(k in low for k in ("đuôi nhọn", "manga-tail", "speech tail", "có đuôi")):
        style_override = ComposePreset.MANGA_TAIL
    elif any(k in low for k in ("floating-only", "chỉ chữ thường", "không bubble")):
        style_override = ComposePreset.FLOATING_ONLY
    elif any(k in low for k in ("cg-oval", "oval", "bong bóng tròn")):
        style_override = ComposePreset.CG_OVAL

    is_compose = any(
        k in low
        for k in (
            "sáng tác",
            "tự nghĩ thoại",
            "tạo thoại",
            "chế thoại",
            "không lời",
            "artist cg",
            "compose",
        )
    )
    is_translate = any(k in low for k in ("dịch", "translate", "raw", "manga", "manhwa", "manhua"))

    if path_str:
        # Clean common command filler words to preserve meaningful story/tone hint
        hint = re.sub(
            r"\b(hãy|giúp|tôi|dịch|truyện|sáng tác|thoại|cg|compose|translate|cho|ảnh|thư mục|với|kiểu|phong cách|đuôi nhọn|cg-oval|manga-tail)\b",
            " ",
            rem,
            flags=re.IGNORECASE,
        )
        hint = re.sub(r"\s+", " ", hint).strip(" ,.-:")

        if is_compose and not is_translate:
            return "compose", path_str, {"prompt": hint, "style": style_override}
        if is_translate and not is_compose:
            return "translate", path_str, {"prompt": hint}
        if style_override is not None:
            return "compose", path_str, {"prompt": hint, "style": style_override}
        return "pick_action", path_str, {"prompt": hint}

    return "chat", None, {"text": text}


def split_cmd_and_hint(arg_str: str) -> tuple[str | None, str]:
    """Parse `/translate "C:\\path\\to\\dir" optional extra prompt...` cleanly."""
    s = arg_str.strip()
    if not s:
        return None, ""
    if s.startswith("& "):
        s = s[2:].strip()

    # Check quoted leading path first
    if s[0] in {'"', "'"}:
        q = s[0]
        end = s.find(q, 1)
        if end > 1:
            return s[1:end].strip().lstrip("@"), s[end + 1 :].strip()

    # Handle unquoted Windows path containing spaces if it resolves on disk
    tokens = s.split()
    for i in range(len(tokens), 0, -1):
        probe = " ".join(tokens[:i]).lstrip("@")
        if Path(probe).exists():
            return probe, " ".join(tokens[i:]).strip()

    try:
        parts = shlex.split(s, posix=False)
    except ValueError:
        parts = s.split()
    first = parts[0].strip("\"'").lstrip("@")
    rest = s[len(parts[0]) :].strip() if parts else ""
    return first, rest


class PickerModal(ModalScreen[str | None]):
    """Arrow-key modal selector matching Claude Code / OpenCode interactive menus."""

    BINDINGS = [Binding("escape", "dismiss_none", "Đóng")]

    def __init__(self, title: str, subtitle: str, choices: list[tuple[str, str]], current: str = "") -> None:
        super().__init__()
        self.title_text = title
        self.subtitle_text = subtitle
        self.choices = choices
        self.current = current

    def compose(self) -> ComposeResult:
        opts: list[Option] = []
        for val, desc in self.choices:
            marker = "● " if val == self.current else "  "
            opts.append(Option(f"{marker}[bold]{val}[/bold]  —  {desc}", id=val))
        with Vertical(id="modal-card"):
            yield Label(self.title_text, id="modal-title")
            yield Label(self.subtitle_text, id="modal-subtitle")
            yield OptionList(*opts, id="picker-list")
            yield Label("[dim]↑/↓ Chọn  •  Enter Xác nhận  •  Esc Hủy[/dim]", id="modal-hint")

    def on_mount(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        for idx, (val, _) in enumerate(self.choices):
            if val == self.current:
                ol.highlighted = idx
                break
        ol.focus()

    @on(OptionList.OptionSelected, "#picker-list")
    def _chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class KeySetupModal(ModalScreen[tuple[str, str] | None]):
    """Input dialog supporting single or multiple rotating Gemini API keys and optional base URL."""

    BINDINGS = [Binding("escape", "cancel", "Hủy")]

    def __init__(self, current_key: str, current_base_url: str | None) -> None:
        super().__init__()
        self.current_key = current_key
        self.current_base_url = current_base_url or ""

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-card"):
            yield Label("Cấu hình Gemini API & Endpoint", id="modal-title")
            yield Label(
                "Có thể nhập NHIỀU API key cách nhau bởi dấu phẩy (,) hoặc chấm phẩy (;) để tự động đổi key khi hết quota/lỗi.",
                id="modal-subtitle",
            )
            yield Label("GEMINI_API_KEY (1 hoặc nhiều key):")
            yield Input(
                value=self.current_key,
                password=False,
                placeholder="AIzaSyKey1..., AIzaSyKey2..., AIzaSyKey3...",
                id="key-input",
            )
            yield Label("Base URL tùy chọn (OpenAI-compatible — không bị bộ lọc Google khi dịch R18):")
            yield Input(
                value=self.current_base_url,
                placeholder="https://openrouter.ai/api/v1 | https://api.x.ai/v1 | http://127.0.0.1:11434/v1",
                id="url-input",
            )
            with Horizontal(id="modal-actions"):
                yield Button("Lưu cấu hình", variant="primary", id="save-btn")
                yield Button("Hủy (Esc)", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#key-input", Input).focus()

    @on(Button.Pressed, "#save-btn")
    def _save(self) -> None:
        k = self.query_one("#key-input", Input).value.strip()
        u = self.query_one("#url-input", Input).value.strip()
        self.dismiss((k, u))

    @on(Button.Pressed, "#cancel-btn")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class RunJobModal(ModalScreen[tuple[str, str, str] | None]):
    """Interactive prompt asking for path, tags + custom instructions when `/translate` or `/compose` lacks args."""

    BINDINGS = [Binding("escape", "cancel", "Hủy")]

    def __init__(
        self,
        mode: str,
        initial_path: str = "",
        initial_prompt: str = "",
        style_name: str = "",
        initial_tags: str = "",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.initial_path = initial_path
        self.initial_prompt = initial_prompt
        self.style_name = style_name
        self.initial_tags = initial_tags

    def compose(self) -> ComposeResult:
        title = (
            "Chạy Pipeline Dịch Truyện"
            if self.mode == "translate"
            else f"Sáng Tác Thoại Artist CG (style: {self.style_name})"
        )
        hint = (
            "Gợi ý xưng hô / bối cảnh (vd: anh - em, kiếm hiệp):"
            if self.mode == "translate"
            else "Gợi ý cốt truyện / tính cách nhân vật (vd: trêu chọc, ngọt ngào):"
        )
        with Vertical(id="modal-card"):
            yield Label(title, id="modal-title")
            yield Label("Kéo thả hoặc dán đường dẫn ảnh / thư mục / file .cbz / .zip:", id="modal-subtitle")
            yield Input(value=self.initial_path, placeholder=r"C:\Users\Duck\Manga\Chap_01", id="job-path")
            yield Label("Tag thể loại (vd: vanilla, NTR, tsundere, comedy):")
            yield Input(value=self.initial_tags, placeholder="Tùy chọn (vd: vanilla, ngọt ngào)...", id="job-tags")
            yield Label(hint)
            yield Input(value=self.initial_prompt, placeholder="Tùy chọn (có thể bỏ trống)...", id="job-prompt")
            with Horizontal(id="modal-actions"):
                yield Button("Bắt đầu chạy", variant="primary", id="run-btn")
                yield Button("Hủy (Esc)", id="cancel-btn")

    def on_mount(self) -> None:
        self.query_one("#job-path", Input).focus()

    @on(Input.Submitted, "#job-path")
    def _next_tags(self) -> None:
        self.query_one("#job-tags", Input).focus()

    @on(Input.Submitted, "#job-tags")
    def _next_or_submit(self) -> None:
        self.query_one("#job-prompt", Input).focus()

    @on(Input.Submitted, "#job-prompt")
    @on(Button.Pressed, "#run-btn")
    def _submit(self) -> None:
        p = strip_path_quotes(self.query_one("#job-path", Input).value)
        tg = self.query_one("#job-tags", Input).value.strip()
        pr = self.query_one("#job-prompt", Input).value.strip()
        if p:
            self.dismiss((p, pr, tg))

    @on(Button.Pressed, "#cancel-btn")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PromptEditModal(ModalScreen[str | None]):
    """Interactive editor for default pronoun / tone / story instructions (`/prompt`)."""

    BINDINGS = [Binding("escape", "cancel", "Hủy")]

    def __init__(self, current_prompt: str) -> None:
        super().__init__()
        self.current_prompt = current_prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-card"):
            yield Label("Chỉnh Prompt / Ngữ cảnh mặc định", id="modal-title")
            yield Label("Gợi ý xưng hô, phong cách dịch hoặc thể loại sáng tác CG (để trống để xóa):", id="modal-subtitle")
            yield Input(
                value=self.current_prompt,
                placeholder="vd: xưng hô anh - em, văn phong tự nhiên dí dỏm",
                id="global-prompt-input",
            )
            with Horizontal(id="modal-actions"):
                yield Button("Lưu prompt", variant="primary", id="save-prompt-btn")
                yield Button("Hủy (Esc)", variant="default", id="cancel-prompt-btn")

    def on_mount(self) -> None:
        self.query_one("#global-prompt-input", Input).focus()

    @on(Input.Submitted, "#global-prompt-input")
    @on(Button.Pressed, "#save-prompt-btn")
    def _save(self) -> None:
        self.dismiss(self.query_one("#global-prompt-input", Input).value.strip())

    @on(Button.Pressed, "#cancel-prompt-btn")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


TAG_PRESETS: tuple[str, ...] = (
    "vanilla",
    "NTR",
    "ecchi/NSFW",
    "romantic/sensual",
    "dirty-talk",
    "sweet/fluff",
    "tsundere",
    "yandere",
    "comedy",
    "angst",
    "wholesome",
    "dark",
    "slice-of-life",
)


class TagEditModal(ModalScreen[str | None]):
    """Interactive genre/theme tag editor (`/tags`) with quick-toggle presets and free-text tags."""

    BINDINGS = [Binding("escape", "cancel", "Hủy")]

    def __init__(self, current_tags: str) -> None:
        super().__init__()
        self.current_tags = current_tags

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-card"):
            yield Label("Đặt Tag Thể Loại (/tags)", id="modal-title")
            yield Label(
                "Chọn nhanh từ danh sách hoặc nhập tay tự do (cách nhau bởi dấu phẩy):",
                id="modal-subtitle",
            )
            yield OptionList(*TAG_PRESETS, id="picker-list")
            yield Input(
                value=self.current_tags,
                placeholder="vd: vanilla, sweet/fluff hoặc NTR, angst",
                id="tag-edit-input",
            )
            with Horizontal(id="modal-actions"):
                yield Button("Lưu tags", variant="primary", id="save-tags-btn")
                yield Button("Xóa hết", variant="warning", id="clear-tags-btn")
                yield Button("Hủy (Esc)", variant="default", id="cancel-tags-btn")

    def on_mount(self) -> None:
        self.query_one("#tag-edit-input", Input).focus()

    @on(OptionList.OptionSelected, "#picker-list")
    def _toggle_preset(self, event: OptionList.OptionSelected) -> None:
        tag = str(event.option.prompt)
        inp = self.query_one("#tag-edit-input", Input)
        existing = [t.strip() for t in inp.value.split(",") if t.strip()]
        low_map = {t.lower(): i for i, t in enumerate(existing)}
        if tag.lower() in low_map:
            existing.pop(low_map[tag.lower()])
        else:
            existing.append(tag)
        inp.value = ", ".join(existing)
        inp.focus()

    @on(Input.Submitted, "#tag-edit-input")
    @on(Button.Pressed, "#save-tags-btn")
    def _save(self) -> None:
        self.dismiss(self.query_one("#tag-edit-input", Input).value.strip())

    @on(Button.Pressed, "#clear-tags-btn")
    def _clear(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#cancel-tags-btn")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class StoryBibleReviewModal(ModalScreen[StoryBible | None]):
    """Interactive modal after Pass 1 so the user can fix detected game & character names/pronouns."""

    BINDINGS = [Binding("escape", "cancel", "Giữ nguyên")]

    def __init__(self, bible: StoryBible) -> None:
        super().__init__()
        self.bible = bible

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-card"):
            yield Label("Kiểm tra Game & Nhân vật (Pass 1)", id="modal-title")
            yield Static(
                "Model chính đã nhận diện bên dưới. Nếu chưa chuẩn, bạn có thể sửa lại trước khi dịch/sáng tác:",
                id="modal-subtitle",
            )
            yield Label("Game / Series gốc:", classes="field-label")
            yield Input(value=self.bible.source_game, placeholder="vd: Honkai: Star Rail / Blue Archive...", id="rev-game-input")
            with VerticalScroll(id="review-scroll"):
                for idx, ch in enumerate(self.bible.characters):
                    yield Label(
                        f"Nhân vật #{idx + 1} ({ch.appearance or 'chưa rõ ngoại hình'}):",
                        classes="field-label",
                    )
                    with Horizontal(classes="char-row"):
                        yield Input(value=ch.name, placeholder="Tên nhân vật", id=f"rev-name-{idx}")
                        yield Input(value=ch.pronoun_rules, placeholder="Xưng hô (vd: anh - em)", id=f"rev-pron-{idx}")
                # Extra blank row so user can add a missed character
                add_idx = len(self.bible.characters)
                yield Label("Thêm nhân vật mới (nếu model nhận thiếu):", classes="field-label")
                with Horizontal(classes="char-row"):
                    yield Input(value="", placeholder="Tên nhân vật thêm (bỏ trống nếu không)", id=f"rev-name-{add_idx}")
                    yield Input(value="", placeholder="Xưng hô (vd: tôi - cô ấy)", id=f"rev-pron-{add_idx}")
            with Horizontal(id="modal-actions"):
                yield Button("Giữ nguyên AI", id="skip-review-btn")
                yield Button("Xác nhận & Tiếp tục", variant="primary", id="save-review-btn")

    @on(Button.Pressed, "#save-review-btn")
    def _save(self) -> None:
        new_game = self.query_one("#rev-game-input", Input).value.strip()
        updated_chars: list[CharacterProfile] = []
        for idx, ch in enumerate(self.bible.characters):
            name = self.query_one(f"#rev-name-{idx}", Input).value.strip()
            pron = self.query_one(f"#rev-pron-{idx}", Input).value.strip()
            if name:
                updated_chars.append(ch.model_copy(update={"name": name, "pronoun_rules": pron}))
        add_idx = len(self.bible.characters)
        extra_name = self.query_one(f"#rev-name-{add_idx}", Input).value.strip()
        extra_pron = self.query_one(f"#rev-pron-{add_idx}", Input).value.strip()
        if extra_name:
            updated_chars.append(CharacterProfile(name=extra_name, pronoun_rules=extra_pron))
        self.dismiss(self.bible.model_copy(update={"source_game": new_game, "characters": updated_chars}))

    @on(Button.Pressed, "#skip-review-btn")
    def _skip(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class KoharuCommandProvider(Provider):
    """Textual Command Palette (`Ctrl+P`) entries matching OpenCode shortcuts."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        tui = self.app
        assert isinstance(tui, KoharuTUI)
        for spec in SLASH_COMMANDS:
            score = matcher.match(f"{spec.name} {spec.desc}")
            if score > 0:
                cmd_text = spec.name
                yield Hit(
                    score,
                    matcher.highlight(f"{spec.name} — {spec.desc}"),
                    lambda c=cmd_text: tui.dispatch_command(c),
                    help=spec.args or None,
                )


class KoharuTUI(App[None]):
    """Interactive AI CLI TUI inspired by Claude Code & OpenCode."""

    TITLE = "Koharu-VN CLI"
    COMMANDS = {KoharuCommandProvider}

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #topbar {
        height: 1;
        dock: top;
        background: $panel;
        color: $text;
        padding: 0 2;
    }

    #workspace {
        height: 1fr;
        layout: horizontal;
    }

    #main-col {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    #stream {
        height: 1fr;
        border: round $primary-darken-1;
        background: $surface;
        padding: 1 2;
        scrollbar-gutter: stable;
    }

    #sidebar {
        width: 32;
        height: 1fr;
        border: round $secondary-darken-1;
        background: $panel;
        padding: 1 1;
        margin-left: 1;
    }

    .sidebar-hidden #sidebar {
        display: none;
    }

    .sb-section {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    .sb-kv {
        color: $text-muted;
        margin-bottom: 0;
    }

    #job-progress {
        margin: 1 0;
    }

    #slash-popup {
        display: none;
        max-height: 11;
        border: round $accent;
        background: $panel;
        margin: 0 1;
    }

    #slash-popup.visible {
        display: block;
    }

    #composer-box {
        height: 3;
        border: round $accent;
        background: $panel;
        margin: 0 1;
        padding: 0 1;
        align-vertical: middle;
    }

    #prompt-caret {
        width: 3;
        height: 1;
        color: $accent;
        text-style: bold;
        padding-top: 0;
    }

    #prompt-input {
        width: 1fr;
        height: 1;
        border: none;
        background: transparent;
        padding: 0;
    }

    #prompt-input:focus {
        border: none;
    }

    #status-strip {
        height: 1;
        color: $text-muted;
        padding: 0 2;
    }

    /* Modals */
    PickerModal, KeySetupModal, RunJobModal, PromptEditModal, TagEditModal {
        align: center middle;
    }

    #modal-card {
        width: 74;
        max-width: 90%;
        height: auto;
        max-height: 85%;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }

    #modal-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 0;
    }

    #modal-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }

    #picker-list {
        height: auto;
        max-height: 14;
        border: round $primary-darken-2;
        margin-bottom: 1;
    }

    #modal-hint {
        color: $text-muted;
    }

    #modal-actions {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }

    #modal-actions Button {
        margin-left: 1;
    }

    #review-scroll {
        max-height: 18;
        margin: 1 0;
    }

    .char-row {
        height: auto;
        margin-bottom: 1;
    }

    .char-row Input {
        width: 1fr;
        margin-right: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True),
        Binding("f2", "choose_style", "Kiểu CG", show=True),
        Binding("f3", "choose_model", "Model", show=True),
        Binding("escape", "cancel_or_close", "Hủy/Đóng", show=True),
        Binding("ctrl+q", "quit", "Thoát", show=True),
    ]

    def __init__(self, initial_theme: str | None = None) -> None:
        super().__init__()
        self.cfg: AppConfig = load_config()
        if initial_theme:
            self.cfg.theme = initial_theme
        self.model_cache = ModelCache()
        self.cancel_event = threading.Event()
        self.is_running_job = False
        self._job_started_at: float = 0.0
        self._spinner_idx: int = 0
        self._last_job_status: str = "Đang rảnh"
        self.history: list[str] = []
        self.history_idx: int = -1
        self._popup_mode: str = ""  # "slash" | "path" | ""
        self._popup_items: list[tuple[str, str]] = []  # (insert_text, kind)
        self._popup_navigated = False

    def compose(self) -> ComposeResult:
        yield Static("", id="topbar", markup=True)
        with Horizontal(id="workspace"):
            with Vertical(id="main-col"):
                yield RichLog(id="stream", wrap=True, highlight=True, markup=True)
            with Vertical(id="sidebar"):
                yield Label("⚙ CẤU HÌNH PHIÊN", classes="sb-section")
                yield Static("", id="sb-config", classes="sb-kv", markup=True)
                yield Label("⚡ TIẾN TRÌNH TÁC VỤ", classes="sb-section")
                yield Static("Đang rảnh", id="sb-job-status", classes="sb-kv", markup=True)
                yield ProgressBar(total=100, show_eta=False, id="job-progress")
                yield Label("📦 TÀI NGUYÊN & OUTPUT", classes="sb-section")
                yield Static("", id="sb-resources", classes="sb-kv", markup=True)
        yield OptionList(id="slash-popup")
        with Horizontal(id="composer-box"):
            yield Label("❯ ", id="prompt-caret")
            yield Input(
                placeholder="Gõ / lệnh, @đường_dẫn, kéo thả thư mục truyện/CG, hoặc yêu cầu tự nhiên...",
                id="prompt-input",
            )
        yield Static("", id="status-strip", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.theme = self.cfg.theme
        except Exception:
            self.theme = "textual-dark"

        # Responsive auto-hide sidebar on narrow terminals (<112 cols)
        if self.size.width and self.size.width < 112:
            self.add_class("sidebar-hidden")

        self._refresh_chrome()
        self._write_welcome()
        self.set_interval(0.5, self._tick_job_timer)
        self.query_one("#prompt-input", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width < 108 and not self.has_class("sidebar-hidden"):
            self.add_class("sidebar-hidden")

    def _refresh_chrome(self) -> None:
        key_ok = bool(self.cfg.effective_api_key())
        key_badge = (
            f"[green]● {self.cfg.masked_api_key()}[/green]"
            if key_ok
            else ("[cyan]● Proxy[/cyan]" if self.cfg.base_url else "[yellow]○ Chưa có key (/key)[/yellow]")
        )
        model_badge = (
            f"Chính: [bold magenta]{self.cfg.story_model}[/bold magenta] → Phụ: [bold]{self.cfg.model}[/bold]"
            if self.cfg.story_model and self.cfg.story_model != self.cfg.model
            else f"Model: [bold]{self.cfg.model}[/bold]"
        )
        topbar = self.query_one("#topbar", Static)
        topbar.update(
            f"[bold cyan]◈ KOHARU-VN CLI[/bold cyan] [dim]v{__version__}[/dim]  │  "
            f"{model_badge}  │  "
            f"API: {key_badge}  │  "
            f"CG Style: [magenta]{self.cfg.compose_style.value}[/magenta]  │  "
            f"Lang: [cyan]{self.cfg.lang}[/cyan]"
        )

        sb_cfg = self.query_one("#sb-config", Static)
        sb_cfg.update(
            f"Model chính: [bold magenta]{self.cfg.story_model or f'chung ({self.cfg.model})'}[/bold magenta]\n"
            f"Model phụ: [bold white]{self.cfg.model}[/bold white]\n"
            f"Endpoint: [dim]{self.cfg.base_url or 'Gemini Official'}[/dim]\n"
            f"Key: {key_badge}\n"
            f"Ngôn ngữ: [cyan]{self.cfg.lang}[/cyan]\n"
            f"CG Style: [magenta]{self.cfg.compose_style.value}[/magenta]\n"
            f"Mật độ CG: [cyan]{self.cfg.density}[/cyan]\n"
            f"Tags: [magenta]{(self.cfg.tags[:30] + '...') if len(self.cfg.tags) > 30 else (self.cfg.tags or 'chưa đặt')}[/magenta]\n"
            f"Font: [white]{self.cfg.font or 'auto (animeace3/pangolin)'}[/white]\n"
            f"Prompt: [dim]{(self.cfg.custom_prompt[:46] + '...') if len(self.cfg.custom_prompt) > 46 else (self.cfg.custom_prompt or 'không')}[/dim]"
        )

        ctd_cached = (Path.home() / ".cache" / "koharu-vn" / "comictextdetector.pt.onnx").is_file()
        out_str = self.cfg.last_output_dir or "Chưa có"
        if len(out_str) > 32:
            out_str = "..." + out_str[-29:]
        self.query_one("#sb-resources", Static).update(
            f"ONNX Mask: {'[green]Đã cache[/green]' if ctd_cached else '[yellow]Tự tải khi chạy[/yellow]'}\n"
            f"Fonts: [green]{len(available_fonts())} font Việt[/green]\n"
            f"Config: [dim]~/.koharu-vn/config.json[/dim]\n"
            f"Output gần nhất:\n[cyan]{out_str}[/cyan]"
        )

        status_strip = self.query_one("#status-strip", Static)
        if self.is_running_job:
            spin = self._SPINNER_FRAMES[self._spinner_idx % len(self._SPINNER_FRAMES)]
            elapsed = self._format_elapsed()
            job_txt = f"[yellow]{spin} Đang chạy ({elapsed}) — Esc để hủy[/yellow]"
        else:
            job_txt = "[green]Sẵn sàng[/green]"
        status_strip.update(
            f"{job_txt}  │  Gõ [bold]/[/bold] xem lệnh  •  [bold]@[/bold] duyệt file  •  [bold]F2[/bold] đổi kiểu CG  •  [bold]Ctrl+P[/bold] bảng lệnh"
        )

    def _write_welcome(self) -> None:
        stream = self.query_one("#stream", RichLog)
        banner = Text()
        banner.append("  █▄▀ █▀█ █ █ ▄▀█ █▀█ █ █   █ █ █▄ █\n", style="bold cyan")
        banner.append("  █ █ █▄█ █▀█ █▀█ █▀▄ █▄█   ▀▄▀ █ ▀█\n\n", style="bold magenta")
        banner.append("  • ", style="cyan")
        banner.append("Dịch truyện Nhật/Hàn/Trung/Anh → Tiếng Việt", style="bold white")
        banner.append(" (ONNX + LaMa + Gemini Vision + Sắp chữ Việt siêu elip)\n", style="dim")
        banner.append("  • ", style="magenta")
        banner.append("Sáng tác thoại Artist CG không lời", style="bold white")
        banner.append(" (Hỗ trợ cả Bubble oval/đuôi nhọn lẫn chữ thường)\n\n", style="dim")
        banner.append("  Lệnh nhanh: ", style="bold yellow")
        banner.append("/translate  /compose  /style  /rerender  /key  /model  /models  /help", style="cyan")
        stream.write(
            Panel(
                banner,
                border_style="cyan",
                box=box.ROUNDED,
                title=f"[bold white]Chào mừng đến Koharu-VN CLI v{__version__}[/bold white]",
                subtitle="[dim]Kéo thả ảnh/thư mục vào khung dưới hoặc gõ / để bắt đầu[/dim]",
            )
        )

    def _log_user(self, message: str) -> None:
        stream = self.query_one("#stream", RichLog)
        stream.write(Text.assemble(("❯ ", "bold cyan"), (message, "bold white")))

    def _log_ai(self, message: str, style: str = "white") -> None:
        if not self.is_mounted:
            return
        try:
            stream = self.query_one("#stream", RichLog)
        except Exception:
            return
        stream.write(Text.from_markup(f"[cyan]●[/cyan] [{style}]{message}[/{style}]"))

    def _log_page_table(self, idx: int, total: int, img_name: str, doc: PageDocument, out_path: Path) -> None:
        stream = self.query_one("#stream", RichLog)
        if not doc.regions:
            stream.write(
                Text.from_markup(f"  [dim]└─ [{idx}/{total}] {img_name}: 0 vùng thoại → lưu {out_path.name}[/dim]")
            )
            return

        tbl = Table(
            title=f"[{idx}/{total}] {img_name} ({len(doc.regions)} thoại) → {out_path.name}",
            box=box.SIMPLE_HEAVY,
            expand=True,
            show_lines=False,
        )
        tbl.add_column("ID", style="cyan", width=4, justify="right")
        tbl.add_column("Kiểu", style="magenta", width=15)
        tbl.add_column("Gốc / Tóm tắt", style="dim", ratio=2)
        tbl.add_column("Tiếng Việt", style="bold white", ratio=3)

        for r in doc.regions:
            kind_lbl = (
                r.bubble_style.value
                if doc.mode == "compose"
                else ("bubble" if r.kind == "bubble" else "floating")
            )
            src_txt = (r.source_text or "").replace("\n", " ")[:48]
            vn_txt = (r.translated_text or "").replace("\n", " ")
            tbl.add_row(str(r.id), kind_lbl, src_txt or "—", vn_txt)
        stream.write(tbl)
        if doc.page_summary:
            stream.write(Text.from_markup(f"  [dim italic]Ngữ cảnh trang: {doc.page_summary}[/dim italic]"))

    def _log_story_bible(self, bible: StoryBible) -> None:
        stream = self.query_one("#stream", RichLog)
        lines: list[str] = []
        if bible.title or bible.genre_and_tone:
            lines.append(f"[bold cyan]{bible.title or 'Story Bible'}[/bold cyan] [dim]({bible.genre_and_tone})[/dim]")
        if bible.source_game:
            lines.append(f"[bold magenta]Game / Series gốc:[/bold magenta] [bold white]{bible.source_game}[/bold white]")
        if bible.pre_event_backstory:
            lines.append(f"[bold yellow]Trước sự kiện ảnh:[/bold yellow] [white]{bible.pre_event_backstory}[/white]")
        if bible.during_event_plot:
            lines.append(f"[bold green]Trong sự kiện ảnh:[/bold green] [white]{bible.during_event_plot}[/white]")
        if bible.full_plot_summary:
            lines.append(f"[dim]{bible.full_plot_summary}[/dim]")
        if bible.characters:
            lines.append("\n[bold yellow]Nhân vật & Quy tắc xưng hô đã khóa:[/bold yellow]")
            for c in bible.characters:
                lines.append(
                    f"  • [bold]{c.name}[/bold] [dim]({c.appearance})[/dim] → [green]{c.pronoun_rules or c.role_and_tone}[/green]"
                )
        stream.write(
            Panel(
                Text.from_markup("\n".join(lines) or "Đã phân tích ngữ cảnh toàn tập."),
                title="[bold magenta]Pass 1 · Cốt truyện & Nhân vật toàn tập[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )

    # -------------------- Slash & @ Autocomplete Popup --------------------

    def _hide_popup(self) -> None:
        popup = self.query_one("#slash-popup", OptionList)
        popup.remove_class("visible")
        self._popup_mode = ""
        self._popup_items.clear()
        self._popup_navigated = False

    def _show_slash_popup(self, query: str) -> None:
        q = query.lower()
        matches = [s for s in SLASH_COMMANDS if s.name.startswith(q) or q.lstrip("/") in s.desc.lower()]
        popup = self.query_one("#slash-popup", OptionList)
        popup.clear_options()
        self._popup_items.clear()
        if not matches:
            self._hide_popup()
            return
        for s in matches:
            arg = f" {s.args}" if s.args else ""
            popup.add_option(Option(f"[bold cyan]{s.name}[/bold cyan][yellow]{arg}[/yellow]  —  [dim]{s.desc}[/dim]"))
            self._popup_items.append((s.name, "slash"))
        self._popup_mode = "slash"
        self._popup_navigated = False
        popup.highlighted = 0
        popup.add_class("visible")

    def _show_path_popup(self, token: str) -> None:
        raw_prefix = token.lstrip("@").strip("\"'")
        expanded = Path(raw_prefix).expanduser() if raw_prefix else Path.cwd()
        if raw_prefix.endswith(("/", "\\")) and expanded.is_dir():
            search_dir = expanded
            stem_prefix = ""
        elif expanded.parent.is_dir() and raw_prefix:
            search_dir = expanded.parent
            stem_prefix = expanded.name.lower()
        else:
            search_dir = Path.cwd()
            stem_prefix = raw_prefix.lower()

        entries: list[Path] = []
        try:
            for item in sorted(search_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if item.name.startswith("."):
                    continue
                if stem_prefix and not item.name.lower().startswith(stem_prefix):
                    continue
                if item.is_dir() or item.suffix.lower() in (*IMAGE_EXTS, ".cbz", ".zip"):
                    entries.append(item)
                if len(entries) >= 10:
                    break
        except OSError:
            pass

        popup = self.query_one("#slash-popup", OptionList)
        popup.clear_options()
        self._popup_items.clear()
        if not entries:
            self._hide_popup()
            return

        for p in entries:
            kind_tag = "[DIR]" if p.is_dir() else "[IMG/CBZ]"
            p_str = str(p)
            if " " in p_str:
                p_str = f'"{p_str}"'
            popup.add_option(Option(f"[cyan]{kind_tag}[/cyan] {p.name}"))
            self._popup_items.append((p_str + (" " if not p.is_dir() else ""), "path"))
        self._popup_mode = "path"
        self._popup_navigated = False
        popup.highlighted = 0
        popup.add_class("visible")

    @on(Input.Changed, "#prompt-input")
    def _on_prompt_changed(self, event: Input.Changed) -> None:
        val = event.value
        if val.startswith("/") and " " not in val:
            self._show_slash_popup(val)
            return

        # Path suggestion on `@...` or after `/translate ` / `/compose ` / `/rerender `
        last_tok = val.split()[-1] if val.split() else ""
        if "@" in last_tok and last_tok.startswith("@"):
            self._show_path_popup(last_tok)
            return
        for cmd in ("/translate ", "/compose ", "/rerender "):
            if val.startswith(cmd) and len(val) >= len(cmd):
                rest = val[len(cmd) :].strip()
                if not rest or rest.endswith(("/", "\\", "@")):
                    self._show_path_popup(rest)
                    return
                if len(rest.split()) == 1:
                    self._show_path_popup(rest)
                    return

        self._hide_popup()

    def _apply_highlighted_popup(self) -> str | None:
        popup = self.query_one("#slash-popup", OptionList)
        if not self._popup_items or popup.highlighted is None:
            return None
        idx = max(0, min(len(self._popup_items) - 1, popup.highlighted))
        insert_text, kind = self._popup_items[idx]
        inp = self.query_one("#prompt-input", Input)
        if kind == "slash":
            spec = next((s for s in SLASH_COMMANDS if s.name == insert_text), None)
            needs_arg = bool(spec and "<" in spec.args)
            inp.value = f"{insert_text} " if needs_arg else insert_text
            inp.cursor_position = len(inp.value)
            self._hide_popup()
            return None if needs_arg else insert_text
        # Replace trailing token with chosen path
        cur = inp.value
        for prefix in ("/translate ", "/compose ", "/rerender "):
            if cur.startswith(prefix):
                inp.value = prefix + insert_text
                inp.cursor_position = len(inp.value)
                self._hide_popup()
                return None
        parts = cur.rsplit(" ", 1)
        inp.value = (f"{parts[0]} " if len(parts) == 2 else "") + insert_text
        inp.cursor_position = len(inp.value)
        self._hide_popup()
        return None

    def on_key(self, event: events.Key) -> None:
        inp = self.query_one("#prompt-input", Input)
        if not inp.has_focus:
            return
        popup = self.query_one("#slash-popup", OptionList)
        popup_visible = popup.has_class("visible") and bool(self._popup_items)

        if popup_visible:
            if event.key == "down":
                popup.highlighted = ((popup.highlighted or 0) + 1) % len(self._popup_items)
                self._popup_navigated = True
                event.prevent_default()
                event.stop()
                return
            if event.key == "up":
                popup.highlighted = ((popup.highlighted or 0) - 1) % len(self._popup_items)
                self._popup_navigated = True
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab":
                self._apply_highlighted_popup()
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter" and self._popup_navigated:
                chosen_immediate = self._apply_highlighted_popup()
                event.prevent_default()
                event.stop()
                if chosen_immediate:
                    inp.value = ""
                    self._handle_submission(chosen_immediate)
                return

        # Command history when popup closed
        if not popup_visible and self.history:
            if event.key == "up":
                if self.history_idx == -1:
                    self.history_idx = len(self.history) - 1
                elif self.history_idx > 0:
                    self.history_idx -= 1
                inp.value = self.history[self.history_idx]
                inp.cursor_position = len(inp.value)
                event.prevent_default()
                event.stop()
            elif event.key == "down" and self.history_idx != -1:
                if self.history_idx < len(self.history) - 1:
                    self.history_idx += 1
                    inp.value = self.history[self.history_idx]
                else:
                    self.history_idx = -1
                    inp.value = ""
                inp.cursor_position = len(inp.value)
                event.prevent_default()
                event.stop()

    @on(OptionList.OptionSelected, "#slash-popup")
    def _popup_clicked(self) -> None:
        chosen_immediate = self._apply_highlighted_popup()
        inp = self.query_one("#prompt-input", Input)
        inp.focus()
        if chosen_immediate:
            inp.value = ""
            self._handle_submission(chosen_immediate)

    @on(Input.Submitted, "#prompt-input")
    def _on_submit(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return
        event.input.value = ""
        self._hide_popup()
        self.history.append(raw)
        self.history_idx = -1
        self._handle_submission(raw)

    # -------------------- Dispatch & Actions --------------------

    def _handle_submission(self, raw: str) -> None:
        self._log_user(raw)
        if raw.startswith("/"):
            self.dispatch_command(raw)
            return

        action, path_str, extra = parse_natural_intent(raw)
        if action == "translate" and path_str:
            self._launch_translate(Path(path_str), extra.get("prompt") or "")
        elif action == "compose" and path_str:
            if extra.get("style") is not None:
                self.cfg.compose_style = extra["style"]
                save_config(self.cfg)
                self._refresh_chrome()
            self._launch_compose(Path(path_str), extra.get("prompt") or "")
        elif action == "rerender" and path_str:
            self._launch_rerender(Path(path_str))
        elif action == "pick_action" and path_str:
            self._ask_path_action(Path(path_str), extra.get("prompt") or "")
        else:
            self._run_chat_assistant(raw)

    def _ask_path_action(self, target: Path, hint: str) -> None:
        def _after(choice: str | None) -> None:
            if choice == "translate":
                self._launch_translate(target, hint)
            elif choice == "compose":
                self._launch_compose(target, hint)
            elif choice == "rerender":
                self._launch_rerender(target)

        opts = [
            ("translate", "Dịch truyện (Nhật/Hàn/Trung/Anh → Tiếng Việt)"),
            ("compose", f"Sáng tác thoại Artist CG không lời (kiểu: {self.cfg.compose_style.value})"),
            ("rerender", "Render lại từ .koharu_cache/*.koharu.json (0 giây API)"),
        ]
        self.push_screen(
            PickerModal("Chọn tác vụ cho đường dẫn", str(target), opts, current="translate"),
            _after,
        )

    def dispatch_command(self, line: str) -> None:
        parts = line.strip().split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/exit", "/quit"}:
            self.exit()
        elif cmd == "/clear":
            self.query_one("#stream", RichLog).clear()
            self._write_welcome()
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/style":
            self.action_choose_style()
        elif cmd == "/lang":
            self._open_picker(
                "Chọn ngôn ngữ gốc",
                "Ưu tiên nhận diện ngôn ngữ trong bong bóng thoại",
                LANG_CHOICES,
                self.cfg.lang,
                self._set_lang,
            )
        elif cmd == "/density":
            self._open_picker(
                "Chọn mật độ thoại Artist CG",
                "Số lượng câu thoại Gemini sáng tác trên mỗi tranh",
                DENSITY_CHOICES,
                self.cfg.density,
                self._set_density,
            )
        elif cmd == "/font":
            f_choices = [("auto", "Tự chọn (AnimeAce3 [TeddyBear] cho bubble, Pangolin cho chữ bay)")] + [
                (f, f"Font đóng gói hỗ trợ đầy đủ dấu Tiếng Việt ({f})") for f in available_fonts()
            ]
            self._open_picker(
                "Chọn Font Tiếng Việt",
                "Áp dụng cho cả dịch truyện lẫn sáng tác thoại CG",
                f_choices,
                self.cfg.font or "auto",
                self._set_font,
            )
        elif cmd == "/model":
            if rest:
                self._set_model(rest)
            else:
                self.action_choose_model()
        elif cmd == "/story-model":
            if rest:
                cleaned = "" if rest.lower() in {"none", "off", "clear", "same"} else rest.strip()
                self._set_story_model(cleaned)
            else:
                self._choose_story_model()
        elif cmd == "/theme":
            if rest:
                self._set_theme(rest)
            else:
                self._open_picker("Đổi Theme TUI", "Giao diện OpenCode / Claude Code", THEME_CHOICES, self.cfg.theme, self._set_theme)
        elif cmd == "/key":
            if rest:
                keys = parse_api_keys(rest)
                self.cfg.api_keys = keys
                self.cfg.api_key = keys[0] if keys else ""
                save_config(self.cfg)
                self._refresh_chrome()
                self._log_ai(f"Đã lưu {len(keys)} API key ({self.cfg.masked_api_key()}).")
            else:
                saved_raw = ", ".join(self.cfg.api_keys) if self.cfg.api_keys else self.cfg.api_key
                self.push_screen(
                    KeySetupModal(saved_raw, self.cfg.base_url),
                    self._on_key_saved,
                )
        elif cmd == "/prompt":
            if rest:
                self._set_default_prompt("" if rest in {"none", "off", "clear"} else rest)
            else:
                self.push_screen(
                    PromptEditModal(self.cfg.custom_prompt),
                    lambda val: self._set_default_prompt(val) if val is not None else None,
                )
        elif cmd == "/tags":
            if rest:
                cleaned = "" if rest.lower() in {"none", "off", "clear"} else rest.strip()
                self._set_tags(cleaned)
            else:
                self.push_screen(
                    TagEditModal(self.cfg.tags),
                    lambda val: self._set_tags(val) if val is not None else None,
                )
        elif cmd == "/story":
            if rest:
                cleaned = "" if rest in {"none", "off", "clear"} else strip_path_quotes(rest)
                self._set_story_context(cleaned)
            else:
                self.push_screen(
                    PromptEditModal(self.cfg.story_context),
                    lambda val: self._set_story_context(val) if val is not None else None,
                )
        elif cmd == "/models":
            if not self.is_running_job:
                self._warmup_models_worker()
        elif cmd == "/open":
            self._open_last_output()
        elif cmd == "/translate":
            resume_flag = "--resume" in rest
            cleaned_rest = rest.replace("--resume", "").strip()
            p_str, hint = split_cmd_and_hint(cleaned_rest)
            if p_str:
                self._launch_translate(Path(p_str), hint, resume=resume_flag)
            else:
                self.push_screen(
                    RunJobModal("translate", "", self.cfg.custom_prompt, initial_tags=self.cfg.tags),
                    lambda res: self._launch_translate(Path(res[0]), res[1], res[2], resume=resume_flag) if res else None,
                )
        elif cmd == "/compose":
            resume_flag = "--resume" in rest
            cleaned_rest = rest.replace("--resume", "").strip()
            p_str, hint = split_cmd_and_hint(cleaned_rest)
            if p_str:
                self._launch_compose(Path(p_str), hint, resume=resume_flag)
            else:
                self.push_screen(
                    RunJobModal("compose", "", self.cfg.custom_prompt, self.cfg.compose_style.value, self.cfg.tags),
                    lambda res: self._launch_compose(Path(res[0]), res[1], res[2], resume=resume_flag) if res else None,
                )
        elif cmd == "/resume":
            p_str, hint = split_cmd_and_hint(rest)
            target_str = p_str or self.cfg.last_input_path or ""
            if not target_str:
                self._log_ai("Chưa có tác vụ trước đó. Hãy chạy `/translate <đường_dẫn> --resume`.", style="yellow")
            elif self.cfg.last_job_mode == "compose":
                self._launch_compose(Path(target_str), hint, resume=True)
            else:
                self._launch_translate(Path(target_str), hint, resume=True)
        elif cmd == "/rerender":
            p_str, _ = split_cmd_and_hint(rest)
            target = p_str or self.cfg.last_output_dir
            if target:
                self._launch_rerender(Path(target))
            else:
                self._log_ai("Chưa có thư mục output nào. Hãy truyền đường dẫn: `/rerender <thư_mục_output>`", style="yellow")
        elif cmd == "/rename":
            in_place = "--in-place" in rest
            cleaned = rest.replace("--in-place", "").strip()
            p_str, _ = split_cmd_and_hint(cleaned)
            target = p_str or self.cfg.last_output_dir
            if target:
                self._launch_rename(Path(target), in_place=in_place)
            else:
                self._log_ai("Hãy truyền thư mục cần đổi tên & chuyển WebP: `/rename <thư_mục> [--in-place]`", style="yellow")
        elif cmd == "/format":
            val = rest.strip().lower()
            if val in {"webp", "original", "png", "jpg"}:
                self.cfg.output_format = val
                save_config(self.cfg)
                self._log_ai(f"Đã đặt định dạng xuất mặc định: [bold cyan]{val}[/bold cyan] (WebP = max quality 100).")
            elif val in {"seq-on", "seq", "on"}:
                self.cfg.sequential_names = True
                save_config(self.cfg)
                self._log_ai("Đã BẬT tự động đánh số thứ tự đọc (`001.webp`, `002.webp`...).")
            elif val in {"seq-off", "off"}:
                self.cfg.sequential_names = False
                save_config(self.cfg)
                self._log_ai("Đã TẮT đánh số tuần tự (giữ tên file gốc).")
            else:
                self._open_picker(
                    "Định dạng & đặt tên ảnh xuất",
                    f"Hiện tại: format={self.cfg.output_format}, đánh số 001..={'BẬT' if self.cfg.sequential_names else 'TẮT'}",
                    [
                        ("webp_seq", "WebP Max Quality (100) + Đánh số 001.webp, 002.webp... (Recommended)"),
                        ("webp_orig", "WebP Max Quality (100) + Giữ tên gốc"),
                        ("original_seq", "Giữ định dạng gốc + Đánh số 001, 002..."),
                        ("original_orig", "Giữ định dạng & tên gốc"),
                    ],
                    f"{self.cfg.output_format}_{'seq' if self.cfg.sequential_names else 'orig'}",
                    self._set_format_combo,
                )
        elif cmd == "/chars":
            val = rest.strip().lower()
            if val in {"on", "1", "true"}:
                self.cfg.review_characters = True
                save_config(self.cfg)
                self._log_ai("Đã BẬT kiểm tra & sửa tên nhân vật game sau Pass 1.")
            elif val in {"off", "0", "false"}:
                self.cfg.review_characters = False
                save_config(self.cfg)
                self._log_ai("Đã TẮT hộp thoại kiểm tra nhân vật sau Pass 1.")
            else:
                self.cfg.review_characters = not self.cfg.review_characters
                save_config(self.cfg)
                state = "BẬT" if self.cfg.review_characters else "TẮT"
                self._log_ai(f"Kiểm tra & sửa tên nhân vật game sau Pass 1: [bold cyan]{state}[/bold cyan].")
        elif cmd == "/vision":
            val = rest.strip().lower()
            if val in {"on", "1", "true"}:
                self.cfg.vision_translate = True
            elif val in {"off", "0", "false"}:
                self.cfg.vision_translate = False
            else:
                self.cfg.vision_translate = not self.cfg.vision_translate
            save_config(self.cfg)
            if self.cfg.vision_translate:
                self._log_ai(
                    "Đã [bold cyan]BẬT Vision[/bold cyan] cho Pass 2 (Model con vừa xem ảnh vừa dịch)."
                )
            else:
                self._log_ai(
                    "Đã [bold yellow]TẮT Vision[/bold yellow] cho Pass 2: Model lớn xem ảnh dựng Story Bible theo tuyến -> OCR cục bộ CPU (manga-ocr-onnx) -> Model con chỉ dịch text theo tuyến (siêu nhanh, né lọc ảnh)."
                )
        elif cmd == "/gpu":
            val = rest.strip().lower()
            if val in {"on", "1", "true"}:
                self.cfg.use_gpu = True
            elif val in {"off", "0", "false", "cpu"}:
                self.cfg.use_gpu = False
            else:
                self.cfg.use_gpu = not self.cfg.use_gpu
            save_config(self.cfg)
            if self.cfg.use_gpu:
                self._log_ai(
                    "Đã [bold cyan]BẬT tăng tốc GPU[/bold cyan] cho Detector + LaMa Inpaint + OCR (tự dùng CUDA/DirectML nếu khả dụng, tự lùi về CPU nếu lỗi)."
                )
            else:
                self._log_ai("Đã chuyển Detector + LaMa Inpaint + OCR sang chạy [bold yellow]CPU[/bold yellow].")
        else:
            self._log_ai(f"Không rõ lệnh `[bold]{cmd}[/bold]`. Gõ `/help` hoặc `/` để xem các lệnh hỗ trợ.", style="yellow")

    def _open_picker(self, title: str, subtitle: str, choices: list[tuple[str, str]], current: str, cb) -> None:
        self.push_screen(PickerModal(title, subtitle, choices, current=current), lambda v: cb(v) if v else None)

    def action_toggle_sidebar(self) -> None:
        self.toggle_class("sidebar-hidden")

    def action_choose_style(self) -> None:
        self._open_picker(
            "Chọn kiểu sáng tác thoại CG (--style)",
            "Áp dụng khi chạy /compose cho ảnh Artist CG không lời",
            STYLE_CHOICES,
            self.cfg.compose_style.value,
            self._set_style,
        )

    def action_choose_model(self) -> None:
        if self.cfg.effective_api_key() or self.cfg.base_url:
            self._log_ai("Đang truy vấn API lấy danh sách model mới nhất...", "dim")
            self._fetch_remote_models_and_pick("model")
            return
        self._open_picker(
            "Chọn Model PHỤ (dịch & sáng tác thoại nhanh)",
            "Gõ /key để kết nối API và tự động cập nhật model mới từ Google",
            MODEL_CHOICES,
            self.cfg.model,
            self._set_model,
        )

    def _choose_story_model(self) -> None:
        if self.cfg.effective_api_key() or self.cfg.base_url:
            self._log_ai("Đang truy vấn API lấy danh sách model cho Model CHÍNH (Pass 1)...", "dim")
            self._fetch_remote_models_and_pick("story_model")
            return
        choices = [("", "Dùng chung với Model phụ")] + list(MODEL_CHOICES)
        self._open_picker(
            "Chọn Model CHÍNH (mạnh) cho Pass 1 cốt truyện & nhân vật",
            "Gõ /key để kết nối API và quét model mới nhất",
            choices,
            self.cfg.story_model,
            self._set_story_model,
        )

    @work(thread=True)
    def _fetch_remote_models_and_pick(self, target: str = "model") -> None:
        choices = list(MODEL_CHOICES)
        sub = "Danh sách mặc định (không lấy được danh sách online)"
        try:
            client = VisionLLMClient(
                api_key=self.cfg.effective_api_key() or None,
                api_keys=self.cfg.effective_api_keys(),
                model=self.cfg.model,
                base_url=self.cfg.base_url,
            )
            remote = client.list_available_models(limit=25)
            if remote:
                seen = {m[0] for m in remote}
                choices = remote + [c for c in MODEL_CHOICES if c[0] not in seen]
                sub = f"Đã đồng bộ trực tiếp {len(remote)} model mới nhất từ API"
        except Exception as exc:
            sub = f"Fallback mặc định (lỗi truy vấn API: {str(exc)[:60]})"

        if target == "story_model":
            choices = [("__same__", "Dùng chung với Model phụ (1 model)")] + choices
            self.call_from_thread(
                self._open_picker,
                "Chọn Model CHÍNH (Pass 1: cốt truyện, nhân vật & tình huống)",
                sub,
                choices,
                self.cfg.story_model or "__same__",
                lambda v: self._set_story_model("" if v == "__same__" else v),
            )
        else:
            self.call_from_thread(
                self._open_picker,
                "Chọn Model PHỤ (Pass 2: dịch & sáng tác thoại nhanh)",
                sub,
                choices,
                self.cfg.model,
                self._set_model,
            )

    def action_cancel_or_close(self) -> None:
        popup = self.query_one("#slash-popup", OptionList)
        if popup.has_class("visible"):
            self._hide_popup()
            return
        if self.is_running_job:
            self.cancel_event.set()
            self._log_ai("Đang gửi tín hiệu hủy tác vụ sau bước hiện tại...", style="yellow")

    def _set_style(self, value: str) -> None:
        self.cfg.compose_style = ComposePreset(value)
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã chọn kiểu thoại CG: [magenta bold]{value}[/magenta bold]")

    def _set_lang(self, value: str) -> None:
        self.cfg.lang = value
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã đặt ngôn ngữ gốc: [cyan bold]{value}[/cyan bold]")

    def _set_density(self, value: str) -> None:
        self.cfg.density = value
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã đặt mật độ thoại CG: [cyan bold]{value}[/cyan bold]")

    def _set_font(self, value: str) -> None:
        self.cfg.font = None if value == "auto" else value
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã đặt font Tiếng Việt: [bold]{self.cfg.font or 'auto'}[/bold]")

    def _set_model(self, value: str) -> None:
        self.cfg.model = value
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã chuyển Model PHỤ sang: [bold cyan]{value}[/bold cyan]")

    def _set_story_model(self, value: str) -> None:
        self.cfg.story_model = value.strip()
        save_config(self.cfg)
        self._refresh_chrome()
        label = self.cfg.story_model or f"dùng chung ({self.cfg.model})"
        self._log_ai(f"Đã cập nhật Model CHÍNH (Pass 1): [bold magenta]{label}[/bold magenta]")

    def _set_default_prompt(self, value: str) -> None:
        self.cfg.custom_prompt = value
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã cập nhật prompt mặc định: [bold]{self.cfg.custom_prompt or '(trống)'}[/bold]")

    def _set_story_context(self, value: str) -> None:
        self.cfg.story_context = value
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã cập nhật cốt truyện/nhân vật (`/story`): [bold]{self.cfg.story_context or '(trống)'}[/bold]")

    def _set_tags(self, value: str) -> None:
        self.cfg.tags = value.strip()
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(f"Đã cập nhật tag thể loại (`/tags`): [bold magenta]{self.cfg.tags or '(trống)'}[/bold magenta]")

    def _set_theme(self, value: str) -> None:
        try:
            self.theme = value
            self.cfg.theme = value
            save_config(self.cfg)
            self._log_ai(f"Đã đổi theme sang: [bold]{value}[/bold]")
        except Exception as exc:
            self._log_ai(f"Không áp dụng được theme '{value}': {exc}", style="red")

    def _on_key_saved(self, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        raw_keys, url = result
        keys = parse_api_keys(raw_keys)
        self.cfg.api_keys = keys
        self.cfg.api_key = keys[0] if keys else ""
        self.cfg.base_url = url or None
        save_config(self.cfg)
        self._refresh_chrome()
        self._log_ai(
            f"Đã lưu {len(keys)} API key vào [dim]{CONFIG_FILE}[/dim] (Key: {self.cfg.masked_api_key()})."
        )

    def _open_last_output(self) -> None:
        target = self.cfg.last_output_dir
        if not target or not Path(target).exists():
            self._log_ai("Chưa có thư mục output hợp lệ để mở.", style="yellow")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            self._log_ai(f"Đã mở thư mục: [cyan]{target}[/cyan]")
        except Exception as exc:
            self._log_ai(f"Không mở được Explorer: {exc}", style="red")

    def _show_help(self) -> None:
        tbl = Table(title="Danh sách lệnh Koharu-VN CLI", box=box.ROUNDED, expand=True)
        tbl.add_column("Lệnh / Phím", style="bold cyan", width=24)
        tbl.add_column("Mô tả", style="white")
        for s in SLASH_COMMANDS:
            arg = f" {s.args}" if s.args else ""
            tbl.add_row(f"{s.name}{arg}", s.desc)
        tbl.add_row("F2 / F3", "Mở nhanh menu chọn Kiểu CG / Model Gemini")
        tbl.add_row("Ctrl+P / Ctrl+B", "Mở Command Palette / Ẩn-hiện Sidebar")
        tbl.add_row("Esc", "Hủy tác vụ đang chạy hoặc đóng popup")
        self.query_one("#stream", RichLog).write(tbl)

    # -------------------- Background Workers --------------------

    def _require_job_idle_and_key(self, needs_llm: bool = True) -> bool:
        if self.is_running_job:
            self._log_ai("Đang có một tác vụ chạy. Nhấn `Esc` nếu muốn hủy trước khi chạy mới.", style="yellow")
            return False
        if needs_llm and not self.cfg.effective_api_key() and not self.cfg.base_url:
            self._log_ai(
                "Chưa có `GEMINI_API_KEY`! Mở hộp thoại `/key` để dán API key (hoặc base URL proxy) trước nhé.",
                style="yellow",
            )
            self.push_screen(KeySetupModal(self.cfg.api_key, self.cfg.base_url), self._on_key_saved)
            return False
        return True

    _SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def _format_elapsed(self) -> str:
        if not self._job_started_at:
            return "00:00"
        secs = max(0, int(time.monotonic() - self._job_started_at))
        mins, rem = divmod(secs, 60)
        return f"{mins:02d}:{rem:02d}"

    def _tick_job_timer(self) -> None:
        if not self.is_running_job:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNER_FRAMES)
        spin = self._SPINNER_FRAMES[self._spinner_idx]
        elapsed = self._format_elapsed()
        self.query_one("#sb-job-status", Static).update(
            f"[yellow]{spin} {elapsed}[/yellow]\n{self._last_job_status}"
        )
        self.query_one("#status-strip", Static).update(
            f"[yellow]{spin} Đang chạy ({elapsed}) — Esc để hủy[/yellow]  │  "
            "Gõ [bold]/[/bold] xem lệnh  •  [bold]@[/bold] duyệt file  •  [bold]F2[/bold] đổi kiểu CG  •  [bold]Ctrl+P[/bold] bảng lệnh"
        )

    def _set_job_ui(self, running: bool, status_text: str = "Đang rảnh", pct: float = 0.0) -> None:
        was_running = self.is_running_job
        self.is_running_job = running
        self._last_job_status = status_text
        if running and not was_running:
            self._job_started_at = time.monotonic()
            self._spinner_idx = 0
        elif not running:
            self._job_started_at = 0.0
        if not self.is_mounted:
            return
        try:
            sb_status = self.query_one("#sb-job-status", Static)
            pb = self.query_one("#job-progress", ProgressBar)
        except Exception:
            return
        if running:
            spin = self._SPINNER_FRAMES[self._spinner_idx % len(self._SPINNER_FRAMES)]
            elapsed = self._format_elapsed()
            sb_status.update(f"[yellow]{spin} {elapsed}[/yellow]\n{status_text}")
        else:
            sb_status.update(status_text)
        pb.update(progress=pct)
        self._refresh_chrome()

    def _set_format_combo(self, val: str) -> None:
        fmt, seq = val.split("_", 1)
        self.cfg.output_format = fmt
        self.cfg.sequential_names = (seq == "seq")
        save_config(self.cfg)
        self._log_ai(
            f"Đã cập nhật xuất ảnh: [bold cyan]{fmt}[/bold cyan] · "
            f"Đánh số 001..={'[green]BẬT[/green]' if self.cfg.sequential_names else '[dim]TẮT[/dim]'}"
        )

    def _open_story_review_modal(
        self,
        bible: StoryBible,
        done_evt: threading.Event,
        result_holder: list[StoryBible | None],
    ) -> None:
        def _on_done(edited: StoryBible | None) -> None:
            result_holder.append(edited)
            done_evt.set()

        self.push_screen(StoryBibleReviewModal(bible), _on_done)

    def _build_hooks(self) -> PipelineHooks:
        def on_step(idx: int, total: int, name: str, stage: str, detail: str) -> None:
            pct = (idx / max(1, total)) * 100.0 if idx > 0 else 5.0
            stage_tag = f"[{idx}/{total}] {name}" if idx > 0 else name
            self.call_from_thread(self._set_job_ui, True, f"[cyan]{stage_tag}[/cyan]\n{detail}", pct)
            self.call_from_thread(self._log_ai, f"[dim]{stage_tag}[/dim] · {detail}")

        def on_page_done(idx: int, total: int, name: str, doc: PageDocument, out_path: Path) -> None:
            self.call_from_thread(self._log_page_table, idx, total, name, doc, out_path)

        def on_story_bible(bible: StoryBible) -> None:
            self.call_from_thread(self._log_story_bible, bible)

        def on_confirm_story_bible(bible: StoryBible) -> StoryBible | None:
            if not self.cfg.review_characters:
                return None
            done_evt = threading.Event()
            result_holder: list[StoryBible | None] = []
            self.call_from_thread(self._set_job_ui, True, "Đang chờ bạn xác nhận nhân vật game...", 15.0)
            self.call_from_thread(self._open_story_review_modal, bible, done_evt, result_holder)
            while not done_evt.wait(timeout=0.25):
                if self.cancel_event.is_set():
                    return None
            return result_holder[0] if result_holder else None

        return PipelineHooks(
            on_step=on_step,
            on_page_done=on_page_done,
            on_story_bible=on_story_bible,
            on_confirm_story_bible=on_confirm_story_bible,
            is_cancelled=self.cancel_event.is_set,
        )

    def _launch_translate(
        self, input_path: Path, extra_prompt: str = "", extra_tags: str = "", *, resume: bool = False
    ) -> None:
        if not self._require_job_idle_and_key(needs_llm=True):
            return
        if not input_path.exists():
            self._log_ai(f"Không tìm thấy đường dẫn: [red]{input_path}[/red]", style="red")
            return
        self.cfg.last_input_path = str(input_path)
        self.cfg.last_job_mode = "translate"
        save_config(self.cfg)
        self._start_translate(input_path, extra_prompt, extra_tags, resume=resume)

    def _launch_compose(
        self, input_path: Path, extra_prompt: str = "", extra_tags: str = "", *, resume: bool = False
    ) -> None:
        if not self._require_job_idle_and_key(needs_llm=True):
            return
        if not input_path.exists():
            self._log_ai(f"Không tìm thấy đường dẫn: [red]{input_path}[/red]", style="red")
            return
        self.cfg.last_input_path = str(input_path)
        self.cfg.last_job_mode = "compose"
        save_config(self.cfg)
        self._start_compose(input_path, extra_prompt, extra_tags, resume=resume)

    def _launch_rerender(self, output_dir: Path) -> None:
        if not self._require_job_idle_and_key(needs_llm=False):
            return
        self._start_rerender(output_dir)

    @work(thread=True)
    def _start_translate(
        self, input_path: Path, extra_prompt: str = "", extra_tags: str = "", *, resume: bool = False
    ) -> None:
        self.cancel_event.clear()
        mode_lbl = "Tiếp tục dịch" if resume else "Dịch truyện"
        self.call_from_thread(self._set_job_ui, True, f"{mode_lbl}: {input_path.name}", 2.0)
        merged_prompt = " | ".join(x for x in (self.cfg.custom_prompt, extra_prompt) if x) or None
        eff_tags = (extra_tags or self.cfg.tags).strip() or None

        try:
            out_dir, docs = run_translate_pipeline(
                input_path,
                lang=self.cfg.lang,
                api_key=self.cfg.effective_api_key() or None,
                api_keys=self.cfg.effective_api_keys(),
                base_url=self.cfg.base_url,
                model=self.cfg.model,
                story_model=self.cfg.story_model or None,
                prompt=merged_prompt,
                story=self.cfg.story_context or None,
                tags=eff_tags,
                font=self.cfg.font,
                conf=self.cfg.conf_thresh,
                accurate_detector=self.cfg.accurate_detector,
                save_cbz=self.cfg.save_cbz,
                output_format=self.cfg.output_format,
                sequential_names=self.cfg.sequential_names,
                vision_translate=self.cfg.vision_translate,
                use_gpu=self.cfg.use_gpu,
                resume=resume,
                model_cache=self.model_cache,
                hooks=self._build_hooks(),
            )
            self.cfg.last_output_dir = str(out_dir)
            save_config(self.cfg)
            self.call_from_thread(
                self._set_job_ui, False, f"[green]Xong {len(docs)} trang![/green]", 100.0
            )
            self.call_from_thread(
                self._log_ai,
                f"[bold green]Hoàn tất dịch {len(docs)} trang![/bold green] Lưu tại: [cyan]{out_dir}[/cyan] (gõ `/open` để mở)",
            )
        except PipelineCancelledError:
            self.call_from_thread(self._set_job_ui, False, "[yellow]Đã hủy[/yellow]", 0.0)
            self.call_from_thread(self._log_ai, "Tác vụ đã được hủy theo yêu cầu. Gõ `/resume` để chạy tiếp.", "yellow")
        except Exception as exc:
            self.call_from_thread(self._set_job_ui, False, "[red]Lỗi[/red]", 0.0)
            self.call_from_thread(
                self._log_ai,
                f"Lỗi khi dịch truyện: [red]{exc}[/red]\n[dim]Mẹo: Gõ `/resume` để dịch tiếp từ trang đang dừng mà không tốn công dịch lại các trang đã xong.[/dim]",
                "red",
            )

    @work(thread=True)
    def _start_compose(
        self, input_path: Path, extra_prompt: str = "", extra_tags: str = "", *, resume: bool = False
    ) -> None:
        self.cancel_event.clear()
        mode_lbl = "Tiếp tục CG" if resume else f"Sáng tác CG ({self.cfg.compose_style.value})"
        self.call_from_thread(
            self._set_job_ui, True, f"{mode_lbl}: {input_path.name}", 2.0
        )
        merged_prompt = " | ".join(x for x in (self.cfg.custom_prompt, extra_prompt) if x) or None
        eff_tags = (extra_tags or self.cfg.tags).strip() or None

        try:
            out_dir, docs = run_compose_pipeline(
                input_path,
                style=self.cfg.compose_style,
                density=self.cfg.density,
                prompt=merged_prompt,
                story=self.cfg.story_context or None,
                tags=eff_tags,
                api_key=self.cfg.effective_api_key() or None,
                api_keys=self.cfg.effective_api_keys(),
                base_url=self.cfg.base_url,
                model=self.cfg.model,
                story_model=self.cfg.story_model or None,
                font=self.cfg.font,
                save_cbz=self.cfg.save_cbz,
                output_format=self.cfg.output_format,
                sequential_names=self.cfg.sequential_names,
                vision_translate=self.cfg.vision_translate,
                use_gpu=self.cfg.use_gpu,
                resume=resume,
                hooks=self._build_hooks(),
            )
            self.cfg.last_output_dir = str(out_dir)
            save_config(self.cfg)
            self.call_from_thread(
                self._set_job_ui, False, f"[green]Xong {len(docs)} ảnh CG![/green]", 100.0
            )
            self.call_from_thread(
                self._log_ai,
                f"[bold green]Hoàn tất sáng tác thoại CG ({self.cfg.compose_style.value})![/bold green] Lưu tại: [cyan]{out_dir}[/cyan]",
            )
        except PipelineCancelledError:
            self.call_from_thread(self._set_job_ui, False, "[yellow]Đã hủy[/yellow]", 0.0)
            self.call_from_thread(self._log_ai, "Tác vụ đã được hủy. Gõ `/resume` để chạy tiếp.", "yellow")
        except Exception as exc:
            self.call_from_thread(self._set_job_ui, False, "[red]Lỗi[/red]", 0.0)
            self.call_from_thread(
                self._log_ai,
                f"Lỗi khi sáng tác CG: [red]{exc}[/red]\n[dim]Mẹo: Gõ `/resume` để chạy tiếp từ trang đang dừng.[/dim]",
                "red",
            )

    def _launch_rename(self, input_path: Path, in_place: bool = False) -> None:
        if self.is_running_job:
            self._log_ai("Đang có tác vụ khác chạy. Hãy chờ xong hoặc nhấn ESC để hủy.", style="yellow")
            return
        if not input_path.exists():
            self._log_ai(f"Không tìm thấy đường dẫn: [red]{input_path}[/red]", style="red")
            return
        self._start_rename(input_path, in_place=in_place)

    @work(thread=True)
    def _start_rename(self, input_path: Path, in_place: bool = False) -> None:
        self.cancel_event.clear()
        self.call_from_thread(self._set_job_ui, True, f"Đổi tên & nén WebP 100%: {input_path.name}", 10.0)
        try:
            out_dir, files = pack_images_to_webp(
                input_path,
                in_place=in_place,
                hooks=self._build_hooks(),
            )
            self.cfg.last_output_dir = str(out_dir)
            save_config(self.cfg)
            self.call_from_thread(self._set_job_ui, False, f"[green]Xong {len(files)} ảnh WebP![/green]", 100.0)
            self.call_from_thread(
                self._log_ai,
                f"[bold green]Đã đổi tên 001..{len(files):03d}.webp (max quality 100)![/bold green] Lưu tại: [cyan]{out_dir}[/cyan]",
            )
        except Exception as exc:
            self.call_from_thread(self._set_job_ui, False, "[red]Lỗi[/red]", 0.0)
            self.call_from_thread(self._log_ai, f"Lỗi khi đổi tên & chuyển WebP: [red]{exc}[/red]", "red")

    @work(thread=True)
    def _start_rerender(self, output_dir: Path) -> None:
        self.cancel_event.clear()
        self.call_from_thread(self._set_job_ui, True, f"Rerender: {output_dir.name}", 10.0)
        try:
            docs = run_rerender_pipeline(
                output_dir,
                font=self.cfg.font,
                output_format="original",
                sequential_names=False,
                hooks=self._build_hooks(),
            )
            self.call_from_thread(self._set_job_ui, False, f"[green]Rerender {len(docs)} trang![/green]", 100.0)
            self.call_from_thread(
                self._log_ai,
                f"[bold green]Đã render lại {len(docs)} trang (0 giây API)![/bold green] Tại: [cyan]{output_dir}[/cyan]",
            )
        except Exception as exc:
            self.call_from_thread(self._set_job_ui, False, "[red]Lỗi[/red]", 0.0)
            self.call_from_thread(self._log_ai, f"Lỗi khi rerender: [red]{exc}[/red]", "red")

    @work(thread=True)
    def _warmup_models_worker(self) -> None:
        if self.is_running_job:
            return
        self.call_from_thread(self._set_job_ui, True, "Kiểm tra & tải cache model ONNX...", 20.0)
        try:
            det, inp = self.model_cache.warmup_all(accurate=self.cfg.accurate_detector, use_gpu=self.cfg.use_gpu)
            det_dev = active_ort_provider_label(getattr(det, "_rtdetr", None))
            inp_dev = active_ort_provider_label(getattr(inp, "_lama", None))
            self.call_from_thread(self._set_job_ui, False, "[green]ONNX sẵn sàng![/green]", 100.0)
            self.call_from_thread(
                self._log_ai,
                f"[bold green]Đã nạp sẵn 3 model ONNX[/bold green] (Detector={det_dev}, LaMa={inp_dev}) & "
                f"{len(available_fonts())} font Tiếng Việt.",
            )
        except Exception as exc:
            self.call_from_thread(self._set_job_ui, False, "[red]Lỗi model[/red]", 0.0)
            self.call_from_thread(self._log_ai, f"Lỗi khi kiểm tra ONNX: [red]{exc}[/red]", "red")

    @work(thread=True)
    def _run_chat_assistant(self, text: str) -> None:
        key = self.cfg.effective_api_key()
        if not key and not self.cfg.base_url:
            self.call_from_thread(
                self._log_ai,
                "Bạn chưa đặt `GEMINI_API_KEY` (gõ `/key` để cài) và câu vừa rồi chưa chứa đường dẫn ảnh/thư mục.\n"
                "  Mẹo nhanh: Kéo thả thư mục truyện hoặc gõ `/translate <đường_dẫn>`, `/compose <đường_dẫn>`, `/style`, `/help`.",
            )
            return

        self.call_from_thread(self._set_job_ui, True, "Đang hỏi trợ lý AI...", 30.0)
        try:
            client = VisionLLMClient(
                api_key=key or None,
                api_keys=self.cfg.effective_api_keys(),
                model=self.cfg.model,
                base_url=self.cfg.base_url,
            )
            summary = f"model={self.cfg.model}, lang={self.cfg.lang}, compose_style={self.cfg.compose_style.value}"
            reply = client.ask_assistant(text, session_summary=summary)
            self.call_from_thread(self._set_job_ui, False, "Sẵn sàng", 0.0)
            self.call_from_thread(self._log_ai, reply)
        except Exception as exc:
            self.call_from_thread(self._set_job_ui, False, "[red]Lỗi API[/red]", 0.0)
            self.call_from_thread(self._log_ai, f"Không gọi được Gemini: [red]{exc}[/red]", "red")


def launch_tui(theme: str | None = None) -> None:
    """Entry point launching the OpenCode / Claude Code style TUI."""
    app = KoharuTUI(initial_theme=theme)
    app.run()
