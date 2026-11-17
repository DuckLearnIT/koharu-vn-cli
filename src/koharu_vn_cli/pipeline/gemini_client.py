"""Gemini Vision API & OpenAI-compatible Vision client for Manga/Manhwa/Manhua translation and CG dialogue composition."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from koharu_vn_cli.models import (
    BubbleStyle,
    CharacterProfile,
    ComposePreset,
    SceneArc,
    StoryBible,
    TextRegionItem,
)
from koharu_vn_cli.pipeline.bubble_renderer import resolve_box_overlaps
from koharu_vn_cli.pipeline.fonts import resolve_font_path
from koharu_vn_cli.pipeline.scene_cluster import SceneGroup

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ("deepseek-chat", "deepseek-reasoner")


def is_deepseek_model(model: str | None) -> bool:
    """Return True when the model ID targets DeepSeek's API."""
    return (model or "").strip().lower().startswith("deepseek-")


def is_deepseek_endpoint(base_url: str | None) -> bool:
    """Return True when base_url points to official DeepSeek API."""
    return "api.deepseek.com" in (base_url or "").lower()


def is_text_only_model(model: str | None, base_url: str | None = None) -> bool:
    """Return True when the target model/endpoint cannot accept image_url vision inputs."""
    if is_deepseek_endpoint(base_url):
        return True
    return is_deepseek_model(model) and not base_url


class ContentBlockedError(RuntimeError):
    """Raised when the upstream API hard-blocks content (e.g. PROHIBITED_CONTENT / SAFETY)."""


class QuotaExhaustedError(RuntimeError):
    """Raised when all API keys/retries hit quota or rate-limit exhaustion (e.g. HTTP 429 / RESOURCE_EXHAUSTED)."""


_QUOTA_BODY_RE = re.compile(
    r"RESOURCE_EXHAUSTED|insufficient_quota|exceeded.*quota|rate.?limit|too many requests|\bquota\b",
    re.IGNORECASE,
)


def _is_quota_error(status_code: int, body_text: str) -> bool:
    if status_code in (429, 402):
        return True
    if status_code in (400, 403) and _QUOTA_BODY_RE.search(body_text or ""):
        return True
    return False


def _pil_to_jpeg_b64(img: Image.Image, max_side: int = 1600, quality: int = 90) -> str:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before `}` or `]` outside quoted strings."""
    out: list[str] = []
    in_str = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_truncated_json(text: str) -> str:
    """Best-effort close open string/arrays/objects when output was truncated near the tail."""
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]") and stack and stack[-1] == ch:
            stack.pop()
    repaired = text.rstrip()
    if in_str:
        if repaired.endswith("\\"):
            repaired = repaired[:-1]
        repaired += '"'
    repaired = re.sub(r"[:,]\s*$", "", repaired)
    repaired = _strip_trailing_commas(repaired)
    return repaired + "".join(reversed(stack))


def _looks_like_soft_refusal(text: str) -> bool:
    low = text.lower()
    markers = (
        "tôi không thể",
        "tôi bị giới hạn",
        "không được phép tạo",
        "nguyên tắc an toàn",
        "giới hạn an toàn",
        "i cannot",
        "i can't",
        "i am unable",
        "i'm unable",
        "cannot assist",
        "cannot fulfill",
        "safety guidelines",
        "against my",
    )
    return any(m in low for m in markers)


def _extract_json_block(raw: str) -> Any:
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response (no JSON content)")
    text = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", raw, flags=re.IGNORECASE).strip()
    if not text:
        # Some reasoning models occasionally place output inside <think>; inspect raw before failing
        text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        open_fence = re.search(r"```(?:json)?\s*([\s\S]+)$", text, flags=re.IGNORECASE)
        if open_fence:
            text = open_fence.group(1).strip()

    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        for candidate in (stripped, _strip_trailing_commas(stripped)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        decoder = json.JSONDecoder()
        for candidate in (stripped, _strip_trailing_commas(stripped)):
            try:
                obj, _ = decoder.raw_decode(candidate)
                if isinstance(obj, (dict, list)):
                    return obj
            except json.JSONDecodeError:
                continue
        for trim_last in (stripped, stripped.rsplit(",", 1)[0]):
            repaired = _repair_truncated_json(trim_last)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        snippet = text[idx:]
        for attempt in (snippet, _strip_trailing_commas(snippet)):
            try:
                obj, _ = decoder.raw_decode(attempt)
                if isinstance(obj, (dict, list)):
                    return obj
            except json.JSONDecodeError:
                continue

    # Last-resort repair for truncated JSON responses with leading prose
    start_candidates = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if start_candidates:
        sliced = text[min(start_candidates) :]
        for trim_last in (sliced, sliced.rsplit(",", 1)[0]):
            repaired = _repair_truncated_json(trim_last)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue

    preview = text.strip().replace("\n", " ")[:220]
    if _looks_like_soft_refusal(text):
        raise ContentBlockedError(
            f"Model từ chối trả lời (không xuất JSON): \"{preview}\" — "
            "Mẹo: Thử xóa/đổi các từ nhạy cảm trong `/tags` (gõ `/tags clear`), chọn `/story-model` khác hoặc dùng `--base-url` phù hợp."
        )
    raise ValueError(f"Model trả về văn bản không phải JSON: {preview!r}")


def _build_annotated_page_and_collage(
    rgb: np.ndarray,
    regions: list[TextRegionItem],
) -> tuple[Image.Image, Image.Image]:
    """Create a numbered full-page context image + high-res numbered crop sheet for ultra-accurate OCR."""
    page_pil = Image.fromarray(rgb)
    annotated = page_pil.copy()
    draw = ImageDraw.Draw(annotated)
    badge_font = ImageFont.truetype(str(resolve_font_path("mali-bold")), size=max(14, min(28, page_pil.height // 35)))

    crops: list[tuple[int, Image.Image]] = []
    h, w = rgb.shape[:2]

    for reg in regions:
        x1, y1, x2, y2 = reg.text_bbox
        draw.rectangle((x1, y1, x2, y2), outline=(255, 50, 50), width=2)
        tag = f"[{reg.id}]"
        tb = draw.textbbox((0, 0), tag, font=badge_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bx = max(0, min(w - tw - 6, x1))
        by = max(0, y1 - th - 6)
        draw.rectangle((bx, by, bx + tw + 6, by + th + 4), fill=(220, 30, 30))
        draw.text((bx + 3, by + 1), tag, font=badge_font, fill=(255, 255, 255))

        # Crop with generous margin and upscale small crops so tiny kana/kanji/hangul are sharp
        bw, bh = x2 - x1, y2 - y1
        pad_x = max(6, int(bw * 0.12))
        pad_y = max(6, int(bh * 0.12))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = page_pil.crop((cx1, cy1, cx2, cy2))
        if max(crop.size) < 180:
            scale = min(3.0, 220.0 / max(1, max(crop.size)))
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.BICUBIC)
        elif max(crop.size) > 480:
            scale = 480.0 / max(crop.size)
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
        crops.append((reg.id, crop))

    # Pack crops neatly into a 2- or 3-column grid
    cols = 1 if len(crops) == 1 else (2 if len(crops) <= 6 else 3)
    cell_pad = 14
    header_h = 28
    max_cw = max((c.width for _, c in crops), default=120)
    col_w = max_cw + cell_pad * 2

    rows = [crops[i : i + cols] for i in range(0, len(crops), cols)]
    row_heights = [max(c.height for _, c in r) + header_h + cell_pad for r in rows]
    sheet_w = cols * col_w
    sheet_h = max(64, sum(row_heights) + cell_pad)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (242, 242, 246))
    sdraw = ImageDraw.Draw(sheet)
    lbl_font = ImageFont.truetype(str(resolve_font_path("mali-bold")), size=18)

    y_cur = cell_pad
    for r_idx, row in enumerate(rows):
        for c_idx, (rid, crop) in enumerate(row):
            x_cur = c_idx * col_w + cell_pad
            sdraw.rectangle((x_cur, y_cur, x_cur + crop.width, y_cur + header_h), fill=(35, 38, 48))
            sdraw.text((x_cur + 8, y_cur + 3), f"ID {rid}", font=lbl_font, fill=(255, 230, 120))
            sheet.paste(crop, (x_cur, y_cur + header_h))
            sdraw.rectangle(
                (x_cur, y_cur, x_cur + crop.width, y_cur + header_h + crop.height),
                outline=(100, 110, 130),
                width=1,
            )
        y_cur += row_heights[r_idx]

    return annotated, sheet


def _build_story_contact_sheets(
    image_paths: list[Path],
    max_pages: int = 36,
    *,
    pages_per_sheet: int | None = None,
    include_keyframes: bool = False,
    page_numbers: list[int] | None = None,
    total_pages: int | None = None,
) -> list[tuple[str, str]]:
    """
    Pack chapter/CG pages into crisp numbered contact sheets (plus optional high-res character
    keyframes for 50-100+ page chapters) so Gemini sees clear character traits & plot arc.
    """
    total = total_pages or len(image_paths)
    count = len(image_paths)
    if count == 0:
        return []

    if count == 1 and page_numbers is None:
        try:
            with Image.open(image_paths[0]) as im:
                return [
                    (
                        "Ảnh đầy đủ cần phân tích toàn diện (nhân vật, hoàn cảnh trước đó & diễn biến trong ảnh):",
                        _pil_to_jpeg_b64(im.convert("RGB"), max_side=1400, quality=90),
                    )
                ]
        except Exception:
            return []

    sheets: list[tuple[str, str]] = []
    if include_keyframes and count > 4:
        kf_indices = sorted({0, max(1, count // 3), min(count - 1, (2 * count) // 3)})
        for ki in kf_indices:
            try:
                p_label = page_numbers[ki] if page_numbers else (ki + 1)
                with Image.open(image_paths[ki]) as im:
                    sheets.append(
                        (
                            f"Ảnh cận/rõ nét Trang {p_label}/{total} (dùng để nhìn rõ mặt, tóc, phụ kiện & nhận diện chính xác Game/Nhân vật):",
                            _pil_to_jpeg_b64(im.convert("RGB"), max_side=1150, quality=86),
                        )
                    )
            except Exception:
                continue

    if count > max_pages:
        indices = sorted({int(round(i * (count - 1) / (max_pages - 1))) for i in range(max_pages)})
    else:
        indices = list(range(count))

    selected: list[tuple[int, Image.Image]] = []
    for idx in indices:
        try:
            p_no = page_numbers[idx] if page_numbers else (idx + 1)
            with Image.open(image_paths[idx]) as im:
                selected.append((p_no, im.convert("RGB")))
        except Exception:
            continue
    if not selected:
        return sheets

    total_sel = len(selected)
    per_sheet = max(1, pages_per_sheet) if pages_per_sheet else max(1, total_sel)
    if per_sheet <= 9:
        tile_w, tile_h = 340, 440
    elif per_sheet <= 12:
        tile_w, tile_h = 300, 380
    elif per_sheet <= 16:
        tile_w, tile_h = 280, 360
    elif per_sheet <= 25:
        tile_w, tile_h = 230, 300
    else:
        tile_w, tile_h = 190, 250

    resized: list[tuple[int, Image.Image]] = []
    for page_no, rgb in selected:
        copy = rgb.copy()
        copy.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        resized.append((page_no, copy))

    lbl_font = ImageFont.truetype(str(resolve_font_path("mali-bold")), size=16 if per_sheet > 16 else 18)
    for start in range(0, len(resized), per_sheet):
        chunk = resized[start : start + per_sheet]
        cols = min(6 if per_sheet > 12 else 4, max(1, math.ceil(math.sqrt(len(chunk)))))
        rows = (len(chunk) + cols - 1) // cols
        cell_w, cell_h, hdr_h, pad = tile_w + 10, tile_h + 32, 24, 10
        canvas = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (24, 26, 34))
        draw = ImageDraw.Draw(canvas)

        for j, (page_no, thumb) in enumerate(chunk):
            r, c = divmod(j, cols)
            x0 = pad + c * cell_w
            y0 = pad + r * cell_h
            draw.rectangle((x0, y0, x0 + tile_w, y0 + hdr_h), fill=(46, 52, 72))
            draw.text((x0 + 8, y0 + 2), f"Trang {page_no}/{total}", font=lbl_font, fill=(130, 230, 255))
            canvas.paste(thumb, (x0 + (tile_w - thumb.width) // 2, y0 + hdr_h + 4))

        first_no = chunk[0][0]
        last_no = chunk[-1][0]
        sheets.append(
            (
                f"Sheet tổng hợp {len(chunk)} trang (từ Trang {first_no} đến Trang {last_no} / tổng {total}):",
                _pil_to_jpeg_b64(canvas, max_side=1800 if per_sheet > 9 else 1500, quality=86),
            )
        )
    return sheets


def _model_sort_key(model_id: str) -> tuple[int, tuple[int, ...], int, str]:
    """Rank newer Gemini models highest (4.x > 3.8 > 3.6 > 3.5 > 2.5, stable before preview/exp)."""
    low = model_id.lower()
    m_ver = re.search(r"gemini[-_](\d+(?:\.\d+)*)", low)
    ver = tuple(int(x) for x in m_ver.group(1).split(".")) if m_ver else (0,)
    is_latest_alias = 1 if "latest" in low else 0
    stability = 0 if any(t in low for t in ("-exp", "experimental")) else (1 if "preview" in low else 2)
    tier = (
        3
        if any(t in low for t in ("hydrogen", "carbon", "argon"))
        else (2 if "pro" in low else (1 if "flash" in low else 0))
    )
    return (is_latest_alias, ver, stability * 10 + tier, low)


def _format_tags_block(tags: str | None) -> str:
    cleaned = (tags or "").strip()
    if not cleaned:
        return ""
    return (
        f"\n[THỂ LOẠI / TAGS]: {cleaned}\n"
        "- Hãy bám sát sắc thái, nhịp tâm lý, cách xưng hô và không khí đặc trưng của các tag trên "
        "(vd: vanilla -> ngọt ngào, ấm áp, tình cảm tự nhiên; NTR -> căng thẳng tâm lý, lén lút/giằng xé/mỉa mai đúng bối cảnh; "
        "tsundere -> ngoài gắt trong mềm; comedy -> lầy lội, duyên dáng đời thường; angst -> day dứt).\n"
        "- Không nhồi chữ sáo rỗng hay lặp tên tag vào lời thoại.\n"
    )


_CJK_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_KANA_HANGUL_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af]")
_PURE_KATAKANA_RE = re.compile(r"^[\u30a0-\u30ffーッンヴ]+$")
_SFX_STEMS_RE = re.compile(
    r"^(?:"
    r"ドキ|ドク|ビク|ピタ|ツル|ニュル|ヌル|ヌチ|グチュ|グチョ|ズチュ|ズボ|ジュポ|ピチャ|"
    r"ドピュ|ビュル|ペチ|パチ|パキ|バキ|ガタ|カチ|ギシ|ゴゴ|ザワ|ゾク|シク|キラ|"
    r"ド[ンー]|バ[ンー]|パ[ンー]|ポ[ンヨ]|ギ[ュヤ]|ズ[ンボ]"
    r")",
)
_VN_TEXT_KEYS = ("vietnamese", "vi", "viet", "translated", "translated_text", "translation", "text_vi", "text")
_PLACEHOLDER_TEXTS = {"", "...", "…", "(trống)", "n/a", "none", "null", "—", "-"}


def _parse_bool_flag(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val or "").strip().lower() in {"true", "1", "yes"}


def looks_like_floating_sfx(text: str) -> bool:
    """Return True only when non-bubble OCR text is unambiguously a sound-effect glyph."""
    cleaned = re.sub(r"[\s\u3000!！?？…・〜~\-ー]+", "", (text or "").strip())
    if not cleaned or len(cleaned) > 10:
        return False
    if not _PURE_KATAKANA_RE.match(cleaned):
        return False
    return bool(_SFX_STEMS_RE.search(cleaned))


def _sfx_rule_block(skip_sfx: bool) -> str:
    if not skip_sfx:
        return ""
    return (
        "\nQUY TẮC BỎ QUA SFX (BẮT BUỘC):\n"
        "- Nếu khung [ID] là **SFX / từ tượng thanh / tiếng động hiệu ứng vẽ trực tiếp trên tranh** "
        "(vd: ドン, ドキッ, ビクン, グチュ, ズチュ, バキッ, ゴゴゴ, カチッ, パンッ, ぽよん, ピチャ, ゾクッ... "
        "dạng chữ hiệu ứng mô phỏng âm thanh hành động/cơ thể/môi trường): "
        "BẮT BUỘC đặt `\"is_sfx\": true` và `\"vietnamese\": \"\"` (để TRỐNG hoàn toàn, KHÔNG dịch và KHÔNG chèn chữ đè tranh).\n"
        "- NGƯỢC LẠI, với MỌI lời thoại trong bubble, suy nghĩ, lời dẫn, chữ chú thích hay tiếng rên/thở phát ra từ miệng nhân vật "
        "(vd: 'あっ…ぐっ！！', 'んっ…', 'はぁ…', 'やめて…'): đặt `\"is_sfx\": false` và dịch sang Tiếng Việt tự nhiên.\n"
    )


def normalize_source_key(text: str) -> str:
    """Normalize source dialogue for reliable translation-memory reuse across frames."""
    return re.sub(r"[\s\u3000]+", "", (text or "")).strip()


def region_needs_translation(reg: TextRegionItem) -> bool:
    """Return True if a source CJK/Kana/Hangul non-SFX region is still empty or untranslated."""
    if reg.is_sfx:
        return False
    src = (reg.source_text or "").strip()
    vn = (reg.translated_text or "").strip()
    if not src:
        return False
    if not vn or vn.lower() in _PLACEHOLDER_TEXTS:
        return True
    if _CJK_KANA_RE.search(src):
        if normalize_source_key(vn) == normalize_source_key(src):
            return True
        if _KANA_HANGUL_RE.search(vn):
            return True
    return False


def _parse_int_id(val: Any) -> int | None:
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    m = re.search(r"\d+", str(val or ""))
    return int(m.group(0)) if m else None


def _extract_vn_from_item(item: dict[str, Any]) -> str:
    for key in _VN_TEXT_KEYS:
        val = item.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if txt and txt.lower() not in _PLACEHOLDER_TEXTS:
            return txt
    return ""


class VisionLLMClient:
    """Unified Gemini API & OpenAI-compatible Vision client."""

    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        fallback_model: str | list[str] | None = None,
        base_url: str | None = None,
        timeout: float = 90.0,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        eff_base_url = base_url.rstrip("/") if base_url else None
        if not eff_base_url and is_deepseek_model(model):
            eff_base_url = DEEPSEEK_BASE_URL
        use_deepseek_keys = is_deepseek_model(model) or is_deepseek_endpoint(eff_base_url)

        raw_sources: list[str] = []
        if api_keys:
            raw_sources.extend(api_keys)
        env_sources = (
            (
                api_key,
                os.environ.get("DEEPSEEK_API_KEYS"),
                os.environ.get("DEEPSEEK_API_KEY"),
                os.environ.get("OPENAI_API_KEY"),
            )
            if use_deepseek_keys
            else (
                api_key,
                os.environ.get("GEMINI_API_KEYS"),
                os.environ.get("GEMINI_API_KEY"),
                os.environ.get("OPENAI_API_KEY"),
            )
        )
        for src in env_sources:
            if src:
                raw_sources.extend(re.split(r"[\s,;]+", src.strip()))

        seen: set[str] = set()
        self.api_keys: list[str] = []
        for k in raw_sources:
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                self.api_keys.append(k)

        self._key_idx = 0
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model = model
        self.fallback_models: list[str] = []
        raw_fb = (
            fallback_model
            if isinstance(fallback_model, list)
            else re.split(r"[\s,;]+", fallback_model.strip())
            if isinstance(fallback_model, str) and fallback_model.strip()
            else []
        )
        fb_seen: set[str] = {model}
        for fb in raw_fb:
            fb_clean = fb.strip()
            if fb_clean and fb_clean not in fb_seen:
                fb_seen.add(fb_clean)
                self.fallback_models.append(fb_clean)
        self._fallback_idx = 0
        self.base_url = eff_base_url
        self.timeout = timeout
        self.on_status = on_status
        if not self.api_key and not self.base_url:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY or DEEPSEEK_API_KEY env var, or pass --api-key (or --base-url for a local proxy)."
            )
        if is_deepseek_endpoint(self.base_url) and not self.api_key:
            raise ValueError(
                "Missing DeepSeek API key. Set DEEPSEEK_API_KEY env var or pass --api-key."
            )

    def _has_fallback_model(self) -> bool:
        return self._fallback_idx < len(self.fallback_models)

    def _switch_to_fallback_model(self, reason: str) -> bool:
        """Advance to the next configured fallback model when the current model exhausts quota."""
        if not self._has_fallback_model():
            return False
        old_model = self.model
        self.model = self.fallback_models[self._fallback_idx]
        self._fallback_idx += 1
        self._key_idx = 0
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self._emit_status(
            f"⚠ Model '{old_model}' {reason} -> tự động chuyển sang model fallback '{self.model}'!"
        )
        return True

    def _rotate_key(self, reason: str) -> bool:
        """Advance to the next API key when multiple keys are configured."""
        if len(self.api_keys) <= 1:
            return False
        self._key_idx = (self._key_idx + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self._key_idx]
        self._emit_status(
            f"Đổi sang API key #{self._key_idx + 1}/{len(self.api_keys)} ({reason})..."
        )
        return True

    def _emit_status(self, message: str) -> None:
        if self.on_status is not None:
            try:
                self.on_status(message)
            except Exception:
                pass

    def _call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[tuple[str, str]],
        temperature: float = 0.3,
    ) -> Any:
        """Send multimodal prompt + base64 JPEGs and parse JSON response with automatic key & quota-fallback failover."""
        img_kb = max(1, sum(len(b64) for _, b64 in images) * 3 // 4 // 1024) if images else 0
        last_err: Exception | None = None
        last_is_quota = False
        keys_total = max(1, len(self.api_keys))
        keys_tried = 1
        transient_attempt = 0
        max_transient = 3

        while True:
            try:
                if transient_attempt == 0 and keys_tried == 1:
                    key_tag = f" [key {self._key_idx + 1}/{keys_total}]" if keys_total > 1 else ""
                    if images:
                        self._emit_status(
                            f"Đang gửi {len(images)} ảnh (~{img_kb} KB) tới {self.model}{key_tag} (timeout {int(self.timeout)}s)..."
                        )
                    else:
                        self._emit_status(
                            f"Đang gửi text-only (OCR + Story Bible) tới {self.model}{key_tag} (timeout {int(self.timeout)}s)..."
                        )
                if self.base_url:
                    raw = self._call_openai_compatible(system_prompt, user_prompt, images, temperature)
                else:
                    raw = self._call_gemini_rest(system_prompt, user_prompt, images, temperature)
                return _extract_json_block(raw)
            except ContentBlockedError:
                raise
            except httpx.HTTPStatusError as exc:
                last_err = exc
                code = exc.response.status_code
                body_preview = exc.response.text.strip().replace("\n", " ")[:200]
                is_quota = _is_quota_error(code, body_preview)
                last_is_quota = is_quota
                if code == 404:
                    if self._fallback_idx > 0 and self._switch_to_fallback_model("không tồn tại (HTTP 404)"):
                        keys_tried = 1
                        transient_attempt = 0
                        last_is_quota = False
                        continue
                    hint = f" (Model '{self.model}' không tồn tại hoặc key chưa mở — bấm F3 hoặc gõ /model để chọn model hỗ trợ)"
                    raise RuntimeError(f"HTTP {code} từ {self.model}: {body_preview}{hint}") from exc
                if code in (400, 401, 403, 429) or code >= 500 or is_quota:
                    if keys_tried < keys_total and self._rotate_key(f"HTTP {code}"):
                        keys_tried += 1
                        continue
                    if code in (400, 401, 403) and not is_quota:
                        hint = " (API key không hợp lệ hoặc hết quyền — kiểm tra /key)" if code in (401, 403) else ""
                        raise RuntimeError(f"HTTP {code} từ {self.model}: {body_preview}{hint}") from exc
                if is_quota and self._has_fallback_model():
                    if transient_attempt < 1:
                        wait_s = 1.5
                        transient_attempt += 1
                        self._emit_status(f"API trả HTTP {code} (quota), thử lại nhanh {transient_attempt}/1 sau {wait_s:.1f}s...")
                        time.sleep(wait_s)
                        continue
                    if self._switch_to_fallback_model(f"hết quota (HTTP {code})"):
                        keys_tried = 1
                        transient_attempt = 0
                        last_is_quota = False
                        continue
                if transient_attempt < max_transient:
                    wait_s = 1.5 * (2**transient_attempt)
                    transient_attempt += 1
                    self._emit_status(f"API trả HTTP {code}, thử lại lần {transient_attempt}/{max_transient} sau {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                break
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_err = exc
                last_is_quota = False
                if isinstance(exc, httpx.HTTPError) and keys_tried < keys_total and self._rotate_key(type(exc).__name__):
                    keys_tried += 1
                    continue
                if transient_attempt < max_transient:
                    wait_s = 1.5 * (2**transient_attempt)
                    transient_attempt += 1
                    detail = str(exc).replace("\n", " ")[:120]
                    self._emit_status(
                        f"Lỗi tạm thời ({type(exc).__name__}: {detail}), thử lại lần {transient_attempt}/{max_transient} sau {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)
                    continue
                break
        if last_is_quota:
            raise QuotaExhaustedError(f"Model {self.model} hết quota sau khi thử lại: {last_err}") from last_err
        raise RuntimeError(f"LLM vision request failed after retries: {last_err}") from last_err

    def _call_gemini_rest(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[tuple[str, str]],
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        parts: list[dict[str, Any]] = []
        for label, b64 in images:
            if label:
                parts.append({"text": label})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        parts.append({"text": user_prompt})

        gen_cfg: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": 16384}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen_cfg,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"Gemini API HTTP {resp.status_code}: {resp.text[:500]}",
                    request=resp.request,
                    response=resp,
                )
            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise ValueError(f"API HTTP {resp.status_code} trả về non-JSON: {resp.text[:180]!r}") from exc

        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = (data.get("promptFeedback") or {}).get("blockReason", "")
            if block_reason:
                raise ContentBlockedError(
                    f"Gemini API chặn nội dung ({block_reason}, model={self.model}). "
                    "Đây là bộ lọc cứng phía máy chủ Google đối với ảnh nhạy cảm/R18; "
                    "hãy dùng /key với endpoint OpenAI-compatible (--base-url) chạy model Vision không bị chặn hoặc thử ảnh nhẹ hơn."
                )
            raise ValueError(f"Gemini returned no candidates: {data}")
        finish_reason = str(candidates[0].get("finishReason", ""))
        if finish_reason in ("PROHIBITED_CONTENT", "SAFETY"):
            raise ContentBlockedError(
                f"Gemini API chặn phản hồi ({finish_reason}, model={self.model}). "
                "Hãy dùng /key với endpoint --base-url phù hợp nếu xử lý nội dung R18."
            )
        parts_out = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts_out if not p.get("thought") and p.get("text")]
        if not text_parts:
            text_parts = [p.get("text", "") for p in parts_out if p.get("text")]
        joined = "\n".join(text_parts)
        if not joined.strip():
            raise ValueError(f"Gemini returned empty text (finishReason={finish_reason or 'unknown'})")
        return joined

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[tuple[str, str]],
        temperature: float,
        json_mode: bool = True,
    ) -> str:
        assert self.base_url is not None
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
        if images and is_deepseek_endpoint(self.base_url):
            raise ValueError(
                "DeepSeek API chính thức chỉ hỗ trợ text (không nhận ảnh trực tiếp). "
                "Hãy bật OCR cục bộ (--no-vision) khi dịch, hoặc dùng Gemini / OpenAI-Vision proxy."
            )
        if images:
            user_content: str | list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for label, b64 in images:
                if label:
                    user_content.append({"type": "text", "text": label})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        else:
            user_content = user_prompt

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if self.model.strip().lower() != "deepseek-reasoner":
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"OpenAI-compatible API HTTP {resp.status_code}: {resp.text[:500]}",
                    request=resp.request,
                    response=resp,
                )
            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise ValueError(f"API HTTP {resp.status_code} trả về non-JSON: {resp.text[:180]!r}") from exc
        msg_content = data["choices"][0]["message"].get("content", "")
        if isinstance(msg_content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in msg_content
                if isinstance(part, dict) and part.get("type", "text") == "text"
            )
        return str(msg_content or "")

    def ask_assistant(self, user_prompt: str, session_summary: str = "") -> str:
        """Short conversational helper inside Koharu-VN TUI (translation tips, prompts, usage)."""
        system_prompt = (
            "Bạn là trợ lý AI tích hợp trong Koharu-VN CLI (công cụ dịch manga/manhwa/manhua Nhật-Hàn-Trung->Việt "
            "và tự sáng tác thoại CG bằng Gemini).\n"
            "Các lệnh TUI hiện có: /translate <path>, /compose <path>, /rerender <out_dir>, "
            "/style (cg-oval|manga-tail|mixed|floating-only), /lang, /font, /density, /model, /key, /prompt, /models.\n"
            f"Trạng thái phiên hiện tại: {session_summary or 'chưa có'}.\n"
            "Hãy trả lời ngắn gọn, thực tế bằng Tiếng Việt."
        )
        if self.base_url:
            return self._call_openai_compatible(system_prompt, user_prompt, [], temperature=0.4, json_mode=False)
        return self._call_gemini_rest(system_prompt, user_prompt, [], temperature=0.4, json_mode=False)

    def list_available_models(self, limit: int = 30) -> list[tuple[str, str]]:
        """
        Query official Gemini API (`/v1beta/models`) or OpenAI-compatible `/models`
        to dynamically fetch the latest vision/text generation models.
        """
        if self.base_url:
            base = re.sub(r"/chat/completions$", "", self.base_url)
            endpoint = f"{base}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(endpoint, headers=headers)
                resp.raise_for_status()
                raw_items = resp.json().get("data", [])
            models: list[tuple[str, str]] = []
            ds_desc = {
                "deepseek-chat": "DeepSeek V3 Chat (text/JSON, OCR cục bộ)",
                "deepseek-reasoner": "DeepSeek R1 Reasoner (suy luận sâu, OCR cục bộ)",
            }
            for m in raw_items:
                mid = str(m.get("id", "")).strip()
                if not mid:
                    continue
                owner = str(m.get("owned_by", "")).strip()
                label = ds_desc.get(mid) or (f"Proxy model ({owner})" if owner else "Proxy model")
                models.append((mid, label))
            models.sort(key=lambda x: _model_sort_key(x[0]), reverse=True)
            return models[:limit]

        endpoint = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(
                endpoint,
                headers={"x-goog-api-key": self.api_key},
            )
            resp.raise_for_status()
            raw_models = resp.json().get("models", [])

        excluded = ("embedding", "aqa", "-tts", "imagen", "veo", "-live", "robotics")
        found: list[tuple[str, str]] = []
        for m in raw_models:
            methods = m.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            raw_name = str(m.get("name", ""))
            mid = raw_name.removeprefix("models/")
            low = mid.lower()
            if "gemini" not in low or any(x in low for x in excluded):
                continue
            disp = str(m.get("displayName", mid)).strip()
            desc = str(m.get("description", "")).strip().splitlines()[0] if m.get("description") else ""
            summary = f"{disp} — {desc[:68]}" if desc else disp
            found.append((mid, summary))

        found.sort(key=lambda x: _model_sort_key(x[0]), reverse=True)
        return found[:limit]

    _VI_LOCALIZATION_RULES = (
        "NGUYÊN TẮC VIỆT HÓA TỰ NHIÊN & GỢI CẢM ĐÚNG CONTEXT (CỰC KỲ QUAN TRỌNG):\n"
        "- Viết như người Việt ngoài đời đang nói chuyện/thì thầm/rên rỉ thật; TUYỆT ĐỐI KHÔNG dịch word-by-word, không văn viết, không lối diễn văn sách vở hay từ y khoa khô khan.\n"
        "- Khi cảnh mang sắc thái thân mật/18+/trêu chọc, hãy dùng ngôn từ mềm mại, ướt át, gợi cảm hoặc táo bạo đúng với biểu cảm & ngữ cảnh gốc (vd: 'nhạy cảm quá...', 'đừng chạm chỗ đó mà...', 'sâu quá... chậm chút đi...', 'cứng hết cả lên rồi kìa', 'chịu hết nổi rồi...', 'bên trong khít chặt quá...').\n"
        "- TUYỆT ĐỐI TRÁNH từ y khoa khô khan ('âm đạo', 'dương vật', 'giao hợp') và tiếng rên kiểu Anh ('UGH', 'GUH', 'BUH', 'NNGH', 'MMN', 'FUH'): đổi sang khẩu ngữ Việt tự nhiên theo sắc thái cảnh (nhẹ/vanilla: 'bên trong', 'chỗ đó', 'nóng hổi', 'ướt nhẹp'; thô/hardcore/NTR: 'nứng', 'chịch', 'bắn vào trong', 'dính bầu'...) và tiếng thở Việt ('Ư...', 'Ưm...', 'Á...', 'Aaa...', 'Hộc...', 'Khựa...').\n"
        "- CẤM dùng từ cứng, sáo rỗng ít ai nói ngoài đời như: 'hắn ta', 'ả', 'gã', 'cô nàng', 'thực sự là', 'vô cùng', 'cái đó', 'vậy sao', 'thèm thuồng lắm chứ', 'kẻ hầu', 'ấn ký' (trừ khi đúng bối cảnh kỳ ảo cổ trang: mới dùng 'ta - ngươi' hoặc danh xưng cổ trang).\n"
        "- Thay bằng khẩu ngữ ngắn gọn, có nhịp thở ngắt quãng (`...`, `!?`) và tiểu từ tự nhiên (à, ừ, nhỉ, đấy, chứ, đâu, cơ mà, mất rồi, luôn ấy, nha...): vd 'Cứ tưởng cô không thích cơ đấy', 'Ủa sao tự dưng...', 'Đừng có mà...', 'Nhói quá...'.\n"
        "- Câu ngắn gọn vừa khung thoại, giữ đúng sắc thái ngại ngùng / trêu chọc / đê mê / hoảng hốt / áp đảo của tuyến cảnh.\n"
        "- Không giải thích dài dòng, không thêm chú thích ngoài lời thoại.\n"
        "- Tiếng thở/rên thoại của nhân vật (khi là lời nói/suy nghĩ, không phải SFX nền) dịch tự nhiên đúng chất doujinshi (vd: 'Ư...', 'Ưm...', 'Á...', 'Hộc...', 'Hả?!').\n"
        "Bắt buộc trả về đúng JSON hợp lệ."
    )


    _DEEPSEEK_R18_LOCALIZATION_ADDON = (
        "\nPHONG CÁCH DOUJINSHI / R18 TỰ NHIÊN (DEEPSEEK UNCENSORED):\n"
        "- KHÔNG tự kiểm duyệt, KHÔNG làm nhạt hay né tránh sắc thái 18+ của thoại gốc.\n"
        "- Khớp đúng độ gợi cảm theo từng câu gốc: đoạn mở đầu/đời thường giữ giọng tự nhiên; khi vào cảnh thân mật hoặc 18+, dịch ướt át, kích thích, đúng chất doujinshi Việt.\n"
        "- Dùng đúng từ vựng cộng đồng dịch doujinshi/R18 Việt quen dùng theo sắc thái cảnh:\n"
        "  + Ngọt/e thẹn/vanilla: 'chỗ đó nhạy cảm quá...', 'bên trong nóng hổi', 'ướt nhẹp hết rồi', 'sâu quá... chạm tới rồi', 'sắp ra mất...♡', 'đừng liếm chỗ đó mà...'.\n"
        "  + Táo bạo/dirty-talk/hardcore (khớp từ gốc kiểu ちんぽ/まんこ/イく/中出し/気持ちいい): dùng thẳng thắn, tự nhiên, gây kích thích ('nứng', 'cặc', 'lồn', 'bú', 'liếm', 'đụ', 'chịch', 'bắn tinh vào trong', 'sướng phát điên', 'dâm quá đi mất') thay vì né bằng từ y khoa hay từ văn viết.\n"
        "- Giữ nhịp thở ngắt quãng (`...`, `♡`, `!?`) và tiếng rên Việt tự nhiên ('Ưm...', 'Á...♡', 'Hộc...', 'Khựa...').\n"
    )

    def _localization_rules(self) -> str:
        if is_deepseek_model(self.model) or is_deepseek_endpoint(self.base_url):
            return self._VI_LOCALIZATION_RULES + self._DEEPSEEK_R18_LOCALIZATION_ADDON
        return self._VI_LOCALIZATION_RULES

    def build_story_bible(
        self,
        image_paths: list[Path],
        mode: str = "translate",
        user_story: str | None = None,
        extra_instructions: str | None = None,
        tags: str | None = None,
        scenes: list[SceneGroup] | None = None,
    ) -> StoryBible:
        """
        Pass 1 global analysis: scan the whole chapter / CG set (or 1 crisp representative per
        clustered visual scene when many similar frames/oneshots are stacked) before Pass 2.
        """
        multi_scene = bool(scenes and len(scenes) >= 2)
        if multi_scene and scenes:
            rep_paths = [image_paths[sc.rep_index] for sc in scenes]
            rep_nums = [sc.rep_index + 1 for sc in scenes]
            sheets = _build_story_contact_sheets(
                rep_paths,
                max_pages=60,
                pages_per_sheet=9,
                include_keyframes=True,
                page_numbers=rep_nums,
                total_pages=len(image_paths),
            )
            scene_lines = "\n".join(
                f"- Tuyến {sc.scene_id}: Trang {sc.start_page}-{sc.end_page} ({len(sc.page_indices)} ảnh na ná nhau, ảnh đại diện #{sc.rep_index + 1})"
                for sc in scenes
            )
            scenes_block = (
                "\nPHÂN TUYẾN TỰ ĐỘNG (nhiều ảnh na ná nhau đã gom thành từng tuyến/oneshot riêng):\n"
                f"{scene_lines}\n"
                "Hãy điền mảng `scene_arcs` tương ứng TỪNG TUYẾN để Pass 2 dịch riêng biệt, KHÔNG trộn ngữ cảnh tuyến này sang tuyến khác!\n"
            )
        else:
            sheets = _build_story_contact_sheets(
                image_paths,
                max_pages=60,
                pages_per_sheet=12,
                include_keyframes=True,
            )
            scenes_block = ""

        if not sheets and not user_story:
            return StoryBible()

        task_desc = (
            "Xem FULL toàn bộ ảnh/chương truyện để TỰ ĐỘNG suy luận đầy đủ ngữ cảnh: "
            "cốt truyện TRƯỚC khi xảy ra sự kiện trong ảnh, diễn biến đang xảy ra TRONG ảnh, "
            "nhận diện từng nhân vật theo ngoại hình/biểu cảm/trang phục và tự khóa bảng xưng hô Tiếng Việt thật tự nhiên ngoài đời."
            if mode == "translate"
            else "Xem FULL toàn bộ bức/bộ ảnh Artist CG không lời để TỰ ĐỘNG sáng tạo khung kịch bản hoàn chỉnh 100%: "
            "(1) Cốt truyện TRƯỚC khi xảy ra sự kiện trong ảnh (nguyên nhân dẫn đến cảnh này), "
            "(2) Diễn biến TRONG khi sự kiện ảnh diễn ra (tâm lý, hành động — làm sườn cho lời thoại thêm), "
            "(3) Nhận diện nhân vật + tính cách + khóa quy tắc xưng hô Tiếng Việt tự nhiên như người thật nói chuyện."
        )
        user_hint = f"\nThông tin bổ sung từ người dùng (nếu có): {user_story}\n" if user_story else ""
        tags_hint = _format_tags_block(tags)
        extra_hint = f"\nYêu cầu thêm: {extra_instructions}\n" if extra_instructions else ""

        system_prompt = (
            "Bạn là tổng biên kịch & tổng biên tập manga/Artist CG Tiếng Việt.\n"
            f"{task_desc}\n"
            "Phải khai thác tối đa chi tiết thị giác trong full ảnh (màu tóc, mắt, halo/cánh/sừng, trang phục, vũ khí, đạo cụ, bối cảnh) "
            "để NHẬN DIỆN CHÍNH XÁC TÊN GAME / SERIES (vd: Honkai: Star Rail, Blue Archive, Genshin Impact, Zenless Zone Zero, Arknights, Nikke, Azur Lane, Fate, Wuthering Waves...) "
            "và TÊN NHÂN VẬT GỐC TRONG GAME đó (nếu là nhân vật game/anime hãy ghi đúng tên chuẩn của nhân vật; nếu không chắc hoặc là OC hãy ghi rõ 'Chưa rõ / OC').\n"
            "Ưu tiên cặp xưng hô tự nhiên theo đúng quan hệ đời thực & lore nhân vật (anh/em, cậu/tớ, tao/mày, thầy/em, cô/cháu, ta/ngươi chỉ dùng cho cổ trang/fantasy).\n"
            "Nếu có nhiều Tuyến/Oneshot tách biệt, hãy tóm tắt từng tuyến vào `scene_arcs` để tránh trộn lẫn bối cảnh.\n"
            "Trả về đúng JSON hợp lệ."
        )
        user_prompt = (
            f"Tổng số ảnh: {len(image_paths)}.{scenes_block}{user_hint}{tags_hint}{extra_hint}\n"
            "Hãy trả về JSON Story Bible đúng cấu trúc:\n"
            "{\n"
            '  "title": "Tên gợi ý hoặc chủ đề chính của cảnh/chương",\n'
            '  "source_game": "Tên Game / Anime / Series gốc nhận diện được (hoặc \'Original / Chưa rõ\')",\n'
            '  "genre_and_tone": "Thể loại & không khí (vd: lãng mạn trêu chọc, kịch tính, học đường...)",\n'
            '  "pre_event_backstory": "Cốt truyện xảy ra TRƯỚC sự kiện trong ảnh (2-3 câu: chuyện gì đã dẫn nhân vật tới tình huống này)",\n'
            '  "during_event_plot": "Diễn biến xảy ra TRONG khi sự kiện ảnh diễn ra (2-4 câu: hành động, cảm xúc, đối đáp để làm nền cho lời thoại)",\n'
            '  "full_plot_summary": "Tóm tắt mạch truyện hoàn chỉnh từ trước -> trong -> hướng diễn biến (3-5 câu)",\n'
            '  "characters": [\n'
            '    {"name": "Tên chuẩn nhân vật trong game (hoặc danh xưng nếu OC)", "appearance": "đặc điểm nhận diện (tóc, mắt, halo/tai/trang phục...)", "role_and_tone": "vai trò, tính cách", "pronoun_rules": "cặp xưng hô Tiếng Việt bắt buộc (vd: anh - em / chị - em)"}\n'
            "  ],\n"
            '  "scene_arcs": [\n'
            '    {"scene_id": 1, "start_page": 1, "end_page": 7, "title": "Tiêu đề ngắn tuyến này", "plot": "Diễn biến & hoàn cảnh riêng tuyến 1", "pronoun_notes": "Xưng hô riêng tuyến này"}\n'
            "  ],\n"
            '  "page_beats": {"1": "Tình huống cụ thể ảnh 1"}\n'
            "}"
        )
        data = self._call_json(
            system_prompt,
            user_prompt,
            images=sheets,
            temperature=0.4 if mode == "compose" else 0.3,
        )
        chars: list[CharacterProfile] = []
        for raw_c in data.get("characters", []):
            if isinstance(raw_c, dict) and raw_c.get("name"):
                chars.append(
                    CharacterProfile(
                        name=str(raw_c.get("name", "")).strip(),
                        appearance=str(raw_c.get("appearance", "")).strip(),
                        role_and_tone=str(raw_c.get("role_and_tone", "")).strip(),
                        pronoun_rules=str(raw_c.get("pronoun_rules", "")).strip(),
                    )
                )
        arcs: list[SceneArc] = []
        for raw_a in data.get("scene_arcs", []):
            if not isinstance(raw_a, dict):
                continue
            try:
                sid = int(raw_a.get("scene_id", len(arcs) + 1))
                sp = int(raw_a.get("start_page", 1))
                ep = int(raw_a.get("end_page", sp))
            except (TypeError, ValueError):
                continue
            arcs.append(
                SceneArc(
                    scene_id=sid,
                    start_page=min(sp, ep),
                    end_page=max(sp, ep),
                    title=str(raw_a.get("title", "")).strip(),
                    plot=str(raw_a.get("plot", "")).strip(),
                    pronoun_notes=str(raw_a.get("pronoun_notes", "")).strip(),
                )
            )
        if not arcs and multi_scene and scenes:
            arcs = [
                SceneArc(scene_id=sc.scene_id, start_page=sc.start_page, end_page=sc.end_page)
                for sc in scenes
            ]
        raw_beats = data.get("page_beats") or {}
        beats = {str(k): str(v).strip() for k, v in raw_beats.items() if str(v).strip()}
        return StoryBible(
            title=str(data.get("title", "")).strip(),
            source_game=str(data.get("source_game", "")).strip(),
            genre_and_tone=str(data.get("genre_and_tone", "")).strip(),
            pre_event_backstory=str(data.get("pre_event_backstory", "")).strip(),
            during_event_plot=str(data.get("during_event_plot", "")).strip(),
            full_plot_summary=str(data.get("full_plot_summary", "")).strip(),
            characters=chars,
            page_beats=beats,
            scene_arcs=arcs,
        )

    def translate_page(
        self,
        rgb: np.ndarray,
        regions: list[TextRegionItem],
        source_lang: str = "auto",
        previous_summaries: list[str] | None = None,
        extra_instructions: str | None = None,
        story_bible: StoryBible | None = None,
        page_index: int | None = None,
        prev_rgb: np.ndarray | None = None,
        next_rgb: np.ndarray | None = None,
        tags: str | None = None,
        skip_sfx: bool = True,
    ) -> tuple[list[TextRegionItem], str]:
        """OCR and translate all numbered regions on a comic page into natural Vietnamese."""
        if not regions:
            return regions, ""

        annotated, collage = _build_annotated_page_and_collage(rgb, regions)
        page_b64 = _pil_to_jpeg_b64(annotated, max_side=1400, quality=88)
        collage_b64 = _pil_to_jpeg_b64(collage, max_side=1600, quality=92)

        bible_block = ""
        if story_bible:
            bible_txt = story_bible.to_prompt_block(page_index=page_index)
            if bible_txt:
                bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT TOÀN TẬP ===\n{bible_txt}\n==========================================\n"

        context_block = ""
        if previous_summaries:
            joined = "\n".join(f"- Trang trước gần {i + 1}: {s}" for i, s in enumerate(previous_summaries[-6:]))
            context_block = f"\nDiễn biến chi tiết vừa diễn ra:\n{joined}\n"

        tags_block = _format_tags_block(tags)
        sfx_block = _sfx_rule_block(skip_sfx)
        extra_block = f"\nYêu cầu thêm từ người dùng: {extra_instructions}\n" if extra_instructions else ""

        system_prompt = (
            "Bạn là biên dịch viên truyện tranh & doujinshi kỳ cựu (Manga Nhật, Manhwa Hàn, Manhua Trung) chuyên bản địa hóa sang Tiếng Việt.\n"
            "Nhiệm vụ: Đọc chính xác chữ gốc trong từng khung [ID] từ Bảng Crop chi tiết kết hợp Ảnh toàn trang, "
            "rồi dịch thoát ý sang KHẨU NGỮ TIẾNG VIỆT ĐỜI THƯỜNG thật tự nhiên, gợi cảm/gợi tình đúng mức theo đúng sắc thái cảnh truyện và xưng hô đã khóa.\n"
            + sfx_block
            + self._localization_rules()
        )

        lang_hint = (
            "Tự nhận diện Nhật/Hàn/Trung/Anh" if source_lang == "auto" else f"Ngôn ngữ gốc ưu tiên: {source_lang}"
        )
        ids_list = [r.id for r in regions]
        page_label = f" (Trang {page_index})" if page_index else ""
        user_prompt = (
            f"{lang_hint}{page_label}.{bible_block}{context_block}{tags_block}{extra_block}\n"
            f"Hãy đọc và phân loại/dịch toàn bộ các ID sau: {ids_list}.\n"
            "Trả về JSON theo đúng cấu trúc:\n"
            "{\n"
            '  "detected_language": "ja | ko | zh | en | mixed",\n'
            '  "page_summary": "Tóm tắt 1-2 câu diễn biến & xưng hô của trang này bằng tiếng Việt",\n'
            '  "items": [\n'
            '    {"id": 1, "source_text": "...", "vietnamese": "...", "is_sfx": false}\n'
            "  ]\n"
            "}"
        )

        images_payload: list[tuple[str, str]] = [
            ("Ảnh toàn trang CẦN DỊCH có đánh số thứ tự [ID]:", page_b64),
            ("Bảng phóng đại chi tiết từng vùng chữ theo ID:", collage_b64),
        ]
        if prev_rgb is not None:
            images_payload.append(
                ("(Tham chiếu cảnh kề trước để nối mạch thoại):", _pil_to_jpeg_b64(Image.fromarray(prev_rgb), max_side=720, quality=76))
            )
        if next_rgb is not None:
            images_payload.append(
                ("(Tham chiếu cảnh kề sau để biết trước ngữ cảnh):", _pil_to_jpeg_b64(Image.fromarray(next_rgb), max_side=720, quality=76))
            )

        data = self._call_json(
            system_prompt,
            user_prompt,
            images=images_payload,
            temperature=0.25,
        )
        return (
            self._merge_translated_regions(regions, data.get("items", []), skip_sfx=skip_sfx),
            str(data.get("page_summary", "")).strip(),
        )

    @staticmethod
    def _merge_translated_regions(
        regions: list[TextRegionItem],
        raw_items: list[Any],
        ocr_texts: dict[int, str] | None = None,
        skip_sfx: bool = True,
    ) -> list[TextRegionItem]:
        dict_items = [it for it in (raw_items or []) if isinstance(it, dict)]
        by_id: dict[int, dict[str, Any]] = {}
        for item in dict_items:
            parsed = _parse_int_id(item.get("id"))
            if parsed is not None:
                by_id[parsed] = item

        positional_ok = len(dict_items) == len(regions)
        updated: list[TextRegionItem] = []
        for idx, reg in enumerate(regions):
            copy = reg.model_copy()
            match = by_id.get(copy.id)
            if match is None and positional_ok:
                match = dict_items[idx]
            fallback_src = (ocr_texts or {}).get(copy.id, copy.source_text)
            if match:
                copy.source_text = str(match.get("source_text", fallback_src)).strip() or fallback_src
                raw_is_sfx = _parse_bool_flag(match.get("is_sfx"))
                norm_src = normalize_source_key(copy.source_text)
                bubble_has_spoken = bool(re.search(r"[\u3040-\u309f\uac00-\ud7af]", norm_src))
                # Protect spoken dialogue inside speech bubbles from accidental SFX mis-tagging
                if raw_is_sfx and (
                    copy.kind != "bubble"
                    or (not bubble_has_spoken and len(norm_src) <= 8)
                    or looks_like_floating_sfx(copy.source_text)
                ):
                    copy.is_sfx = True
                elif copy.kind == "floating" and looks_like_floating_sfx(copy.source_text):
                    copy.is_sfx = True

                if skip_sfx and copy.is_sfx:
                    copy.translated_text = ""
                else:
                    vn = _extract_vn_from_item(match)
                    if vn:
                        copy.translated_text = vn
            else:
                if fallback_src:
                    copy.source_text = fallback_src
                if copy.kind == "floating" and looks_like_floating_sfx(copy.source_text):
                    copy.is_sfx = True
                    if skip_sfx:
                        copy.translated_text = ""
            updated.append(copy)
        return updated

    def _build_ocr_context_blocks(
        self,
        *,
        source_lang: str,
        previous_summaries: list[str] | None,
        extra_instructions: str | None,
        story_bible: StoryBible | None,
        page_index: int | None,
        tags: str | None,
    ) -> tuple[str, str]:
        bible_block = ""
        if story_bible:
            bible_txt = story_bible.to_prompt_block(page_index=page_index)
            if bible_txt:
                bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT ===\n{bible_txt}\n===================================\n"
        context_block = ""
        if previous_summaries:
            joined = "\n".join(f"- Trang trước gần {i + 1}: {s}" for i, s in enumerate(previous_summaries[-6:]))
            context_block = f"\nDiễn biến vừa xảy ra trong cùng tuyến:\n{joined}\n"
        tags_block = _format_tags_block(tags)
        extra_block = f"\nYêu cầu thêm từ người dùng: {extra_instructions}\n" if extra_instructions else ""
        lang_hint = (
            "Tự nhận diện Nhật/Hàn/Trung/Anh" if source_lang == "auto" else f"Ngôn ngữ gốc ưu tiên: {source_lang}"
        )
        return lang_hint, f"{bible_block}{context_block}{tags_block}{extra_block}"

    def translate_page_from_ocr(
        self,
        regions: list[TextRegionItem],
        ocr_texts: dict[int, str],
        source_lang: str = "auto",
        previous_summaries: list[str] | None = None,
        extra_instructions: str | None = None,
        story_bible: StoryBible | None = None,
        page_index: int | None = None,
        tags: str | None = None,
        translation_memory: dict[str, str] | None = None,
        skip_sfx: bool = True,
        *,
        _allow_retry: bool = True,
    ) -> tuple[list[TextRegionItem], str]:
        """Text-only Pass 2 translation using local ONNX OCR output + Pass 1 StoryBible (sends zero images)."""
        if not regions:
            return regions, ""

        working = [r.model_copy() for r in regions]
        for copy in working:
            raw_ocr = (ocr_texts.get(copy.id, "") or copy.source_text or "").strip()
            if raw_ocr:
                copy.source_text = raw_ocr
            if skip_sfx and copy.kind == "floating" and looks_like_floating_sfx(copy.source_text):
                copy.is_sfx = True
                copy.translated_text = ""

        to_translate = [r for r in working if not (skip_sfx and r.is_sfx)]
        if not to_translate:
            return working, ""

        lang_hint, ctx_block = self._build_ocr_context_blocks(
            source_lang=source_lang,
            previous_summaries=previous_summaries,
            extra_instructions=extra_instructions,
            story_bible=story_bible,
            page_index=page_index,
            tags=tags,
        )
        lines = []
        for r in to_translate:
            raw_txt = (ocr_texts.get(r.id, "") or r.source_text or "").strip() or "(trống / tiếng thở ngắn)"
            x1, y1, x2, y2 = r.text_bbox
            kind = "Dọc" if (y2 - y1) > (x2 - x1) * 1.25 else "Ngang"
            area_tag = "Trong bóng" if r.kind == "bubble" else "Ngoài khung"
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            lines.append(f"- ID {r.id} ({area_tag}, {kind}, tâm=({cx},{cy})): {raw_txt}")
        ocr_block = "\n".join(lines)

        sfx_block = _sfx_rule_block(skip_sfx)
        system_prompt = (
            "Bạn là biên dịch viên truyện tranh & doujinshi kỳ cựu (Manga Nhật, Manhwa Hàn, Manhua Trung) chuyên bản địa hóa sang Tiếng Việt.\n"
            "Chế độ NO-VISION: Chữ gốc từng khung đã được OCR cục bộ trích sẵn bên dưới. "
            "Hãy tự sửa lỗi nhận dạng OCR nhỏ nếu hiển nhiên dựa vào Story Bible & tuyến cảnh hiện tại, "
            "rồi dịch TẤT CẢ các ID lời thoại sang KHẨU NGỮ TIẾNG VIỆT ĐỜI THƯỜNG thật tự nhiên, đúng xưng hô đã khóa.\n"
            "CẤM để trống `vietnamese` đối với lời thoại (trừ khi `is_sfx: true`) hoặc giữ nguyên chữ Nhật/Hàn/Trung!\n"
            + sfx_block
            + self._localization_rules()
        )
        page_label = f" (Trang {page_index})" if page_index else ""
        user_prompt = (
            f"{lang_hint}{page_label}.{ctx_block}\n"
            f"Danh sách chữ OCR cục bộ theo khung thoại:\n{ocr_block}\n\n"
            "Trả về JSON theo đúng cấu trúc:\n"
            "{\n"
            '  "detected_language": "ja | ko | zh | en | mixed",\n'
            '  "page_summary": "Tóm tắt 1-2 câu diễn biến & xưng hô của trang này bằng tiếng Việt",\n'
            '  "items": [\n'
            '    {"id": 1, "source_text": "...", "vietnamese": "Lời thoại Tiếng Việt", "is_sfx": false}\n'
            "  ]\n"
            "}"
        )
        ocr_temp = 0.45 if (is_deepseek_model(self.model) or is_deepseek_endpoint(self.base_url)) else 0.25
        data = self._call_json(system_prompt, user_prompt, images=[], temperature=ocr_temp)
        raw_items = data.get("items") or data.get("regions") or []
        merged_sub = self._merge_translated_regions(to_translate, raw_items, ocr_texts=ocr_texts, skip_sfx=skip_sfx)
        by_sub_id = {r.id: r for r in merged_sub}
        updated = [by_sub_id.get(r.id, r) for r in working]
        summary = str(data.get("page_summary", "")).strip()

        # Guard against small models echoing Japanese or omitting an ID
        missing = [r for r in updated if region_needs_translation(r)]
        if _allow_retry and missing:
            retry_note = (
                (f"{extra_instructions}\n" if extra_instructions else "")
                + "CHÚ Ý: Lần trước vẫn còn ID chưa dịch hoặc còn nguyên tiếng Nhật/Hàn/Trung. "
                "Hãy dịch 100% các ID thoại này sang Tiếng Việt (hoặc đặt is_sfx=true nếu là SFX thuần túy), KHÔNG giữ nguyên tiếng gốc!"
            )
            retried, retry_sum = self.translate_page_from_ocr(
                missing,
                ocr_texts=ocr_texts,
                source_lang=source_lang,
                previous_summaries=previous_summaries,
                extra_instructions=retry_note,
                story_bible=story_bible,
                page_index=page_index,
                tags=tags,
                translation_memory=translation_memory,
                skip_sfx=skip_sfx,
                _allow_retry=False,
            )
            retry_map = {r.id: r for r in retried}
            updated = [
                retry_map[r.id]
                if (r.id in retry_map and not region_needs_translation(retry_map[r.id]))
                else r
                for r in updated
            ]
            if not summary and retry_sum:
                summary = retry_sum

        # Last-resort fallback: if any region still lacks a valid Vietnamese translation after retry,
        # reuse an existing verified translation from translation_memory rather than leaving Japanese!
        if translation_memory:
            for r in updated:
                if region_needs_translation(r):
                    key = normalize_source_key(ocr_texts.get(r.id, "") or r.source_text)
                    cached_vn = translation_memory.get(key, "").strip()
                    if cached_vn and not _KANA_HANGUL_RE.search(cached_vn):
                        r.translated_text = cached_vn

        return updated, summary

    def translate_scene_from_ocr(
        self,
        scene_pages: list[tuple[int, list[TextRegionItem], dict[int, str]]],
        source_lang: str = "auto",
        previous_summaries: list[str] | None = None,
        extra_instructions: str | None = None,
        story_bible: StoryBible | None = None,
        tags: str | None = None,
        translation_memory: dict[str, str] | None = None,
        skip_sfx: bool = True,
        *,
        scene_id: int | None = None,
    ) -> dict[int, tuple[list[TextRegionItem], str]]:
        """
        Batch text-only Pass 2 translation for all text pages in ONE clustered scene/oneshot.
        Sends ZERO images and falls back per-page if any page or region remains untranslated.
        """
        if not scene_pages:
            return {}
        if len(scene_pages) == 1:
            pno, regs, ocrs = scene_pages[0]
            return {
                pno: self.translate_page_from_ocr(
                    regs,
                    ocrs,
                    source_lang=source_lang,
                    previous_summaries=previous_summaries,
                    extra_instructions=extra_instructions,
                    story_bible=story_bible,
                    page_index=pno,
                    tags=tags,
                    translation_memory=translation_memory,
                    skip_sfx=skip_sfx,
                )
            }

        rep_page_no = scene_pages[0][0]
        lang_hint, ctx_block = self._build_ocr_context_blocks(
            source_lang=source_lang,
            previous_summaries=previous_summaries,
            extra_instructions=extra_instructions,
            story_bible=story_bible,
            page_index=rep_page_no,
            tags=tags,
        )

        page_blocks: list[str] = []
        for pno, regs, ocrs in scene_pages:
            r_lines = [
                f"  - ID {r.id} ({'Trong bóng' if r.kind == 'bubble' else 'Ngoài khung'}): {ocrs.get(r.id, '').strip() or '(trống / tiếng thở ngắn)'}"
                for r in regs
            ]
            page_blocks.append(f"[Trang {pno}]\n" + "\n".join(r_lines))
        joined_pages = "\n\n".join(page_blocks)

        sfx_block = _sfx_rule_block(skip_sfx)
        system_prompt = (
            "Bạn là biên dịch viên truyện tranh & doujinshi kỳ cựu chuyên bản địa hóa sang Tiếng Việt.\n"
            "Chế độ NO-VISION THEO TUYẾN (Batch OCR): Dưới đây là toàn bộ lời thoại OCR của các trang thuộc CÙNG MỘT TUYẾN/ONESHOT. "
            "Hãy dịch liền mạch, nối nhịp cảm xúc & xưng hô nhất quán xuyên suốt các trang trong tuyến này.\n"
            "CẤM để trống `vietnamese` (trừ khi `is_sfx: true`) hoặc giữ nguyên chữ Nhật/Hàn/Trung!\n"
            + sfx_block
            + self._localization_rules()
        )
        scene_hdr = f"Tuyến {scene_id}" if scene_id is not None else "Cùng 1 tuyến"
        user_prompt = (
            f"{lang_hint} — {scene_hdr} ({len(scene_pages)} trang có thoại).{ctx_block}\n"
            f"Nội dung OCR từng trang:\n{joined_pages}\n\n"
            "Trả về đúng JSON cho TẤT CẢ các trang trên:\n"
            "{\n"
            '  "pages": [\n'
            "    {\n"
            f'      "page": {rep_page_no},\n'
            '      "page_summary": "Tóm tắt ngắn trang này",\n'
            '      "items": [{"id": 1, "source_text": "...", "vietnamese": "Lời thoại Tiếng Việt", "is_sfx": false}]\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        ocr_temp = 0.45 if (is_deepseek_model(self.model) or is_deepseek_endpoint(self.base_url)) else 0.25
        data = self._call_json(system_prompt, user_prompt, images=[], temperature=ocr_temp)

        by_page: dict[int, dict[str, Any]] = {}
        for p_entry in data.get("pages", []):
            if not isinstance(p_entry, dict):
                continue
            p_raw = p_entry.get("page") or p_entry.get("page_number") or p_entry.get("page_index") or p_entry.get("id")
            parsed_p = _parse_int_id(p_raw)
            if parsed_p is not None:
                by_page[parsed_p] = p_entry

        mem = dict(translation_memory or {})
        out: dict[int, tuple[list[TextRegionItem], str]] = {}
        rolling_summaries = list(previous_summaries or [])
        for pno, regs, ocrs in scene_pages:
            entry = by_page.get(pno)
            raw_page_items = (
                (entry.get("items") or entry.get("regions")) if isinstance(entry, dict) else None
            )
            if entry and isinstance(raw_page_items, list):
                merged = self._merge_translated_regions(regs, raw_page_items, ocr_texts=ocrs, skip_sfx=skip_sfx)
                summary = str(entry.get("page_summary", "")).strip()
                untranslated = [r for r in merged if region_needs_translation(r)]
                if untranslated:
                    fixed_missing, fix_sum = self.translate_page_from_ocr(
                        untranslated,
                        ocrs,
                        source_lang=source_lang,
                        previous_summaries=rolling_summaries,
                        extra_instructions=extra_instructions,
                        story_bible=story_bible,
                        page_index=pno,
                        tags=tags,
                        translation_memory=mem,
                        skip_sfx=skip_sfx,
                    )
                    fix_map = {r.id: r for r in fixed_missing}
                    merged = [fix_map.get(r.id, r) for r in merged]
                    if not summary and fix_sum:
                        summary = fix_sum
                for r in merged:
                    if not r.is_sfx and not region_needs_translation(r) and r.source_text and r.translated_text:
                        mem[normalize_source_key(r.source_text)] = r.translated_text.strip()
                out[pno] = (merged, summary)
                if summary:
                    rolling_summaries.append(summary)
            else:
                res = self.translate_page_from_ocr(
                    regs,
                    ocrs,
                    source_lang=source_lang,
                    previous_summaries=rolling_summaries,
                    extra_instructions=extra_instructions,
                    story_bible=story_bible,
                    page_index=pno,
                    tags=tags,
                    translation_memory=mem,
                    skip_sfx=skip_sfx,
                )
                for r in res[0]:
                    if not r.is_sfx and not region_needs_translation(r) and r.source_text and r.translated_text:
                        mem[normalize_source_key(r.source_text)] = r.translated_text.strip()
                out[pno] = res
                if res[1]:
                    rolling_summaries.append(res[1])
        return out

    def compose_cg_page(
        self,
        rgb: np.ndarray,
        preset: ComposePreset = ComposePreset.CG_OVAL,
        density: str = "medium",
        previous_summaries: list[str] | None = None,
        story_prompt: str | None = None,
        story_bible: StoryBible | None = None,
        page_index: int | None = None,
        prev_rgb: np.ndarray | None = None,
        next_rgb: np.ndarray | None = None,
        tags: str | None = None,
    ) -> tuple[list[TextRegionItem], str]:
        """
        Analyze a textless Artist CG illustration and compose natural Vietnamese dialogue
        with spatial placement avoiding faces/focal areas, supporting both bubble & floating text.
        """
        h, w = rgb.shape[:2]
        img_b64 = _pil_to_jpeg_b64(Image.fromarray(rgb), max_side=1400, quality=90)

        density_guide = {
            "low": "1 đến 3 câu thoại ngắn gọn, chừa nhiều khoảng trống cho tranh.",
            "medium": "3 đến 5 câu thoại/biểu cảm cân đối, nhịp truyện tự nhiên.",
            "high": "5 đến 8 câu thoại gồm cả thoại chính, suy nghĩ thầm và tiếng thốt/SFX.",
        }.get(density, "3 đến 5 câu thoại cân đối.")
        style_rule = _compose_style_rule(preset)

        bible_block = ""
        if story_bible:
            bible_txt = story_bible.to_prompt_block(page_index=page_index)
            if bible_txt:
                bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT TOÀN BỘ CG ===\n{bible_txt}\n==========================================\n"

        context_block = ""
        if previous_summaries:
            joined = "\n".join(f"- Ảnh trước {i + 1}: {s}" for i, s in enumerate(previous_summaries[-6:]))
            context_block = (
                "\nMạch truyện của các bức CG trước (hãy nối tiếp tình tiết, giữ đúng tên & xưng hô nhân vật):\n"
                f"{joined}\n"
            )

        tags_block = _format_tags_block(tags)
        prompt_block = f"\nChỉ dẫn kịch bản / phong cách từ người dùng: {story_prompt}\n" if story_prompt else ""

        system_prompt = (
            "Bạn là biên kịch & đạo diễn dàn trang truyện tranh/CG chuyên nghiệp.\n"
            "Nhiệm vụ: Quan sát kỹ bức tranh Artist CG (biểu cảm khuôn mặt, ánh mắt, cử chỉ, trang phục, bối cảnh), "
            "sáng tác lời thoại Tiếng Việt theo KHẨU NGỮ ĐỜI THƯỜNG thật tự nhiên, có cảm xúc và đúng thể loại/tag, đồng thời xác định tọa độ đặt chữ thẩm mỹ nhất.\n"
            "NGUYÊN TẮC VIẾT THOẠI TỰ NHIÊN & GỢI CẢM ĐÚNG CONTEXT:\n"
            "- Viết như người Việt ngoài đời thật nói chuyện, thì thầm hoặc nghĩ thầm; tránh văn viết dài dòng, tránh từ y khoa khô khan ('âm đạo', 'dương vật') hay tiếng rên kiểu Anh ('UGH', 'GUH', 'NNGH').\n"
            "- Nếu tranh mang không khí thân mật/quyến rũ/gợi cảm, hãy viết thoại thật lôi cuốn, gợi tình và tự nhiên theo đúng ánh mắt, tư thế & tag thể loại (nhẹ/vanilla dùng từ ngọt ngào ướt át; thô/NTR dùng khẩu ngữ táo bạo đúng tâm lý).\n"
            "- Câu thoại ngắn gọn, có nhịp thở ngắt quãng (`...`), tiếng thở Việt tự nhiên (`Ư...`, `Ưm...`, `Á...`), hợp biểu cảm gương mặt & bối cảnh.\n"
            "QUY TẮC ĐẶT TỌA ĐỘ CỰC KỲ QUAN TRỌNG:\n"
            "- Hệ tọa độ chuẩn hóa [ymin, xmin, ymax, xmax] từ 0 đến 1000 (0=trên/trái, 1000=dưới/phải).\n"
            "- TUYỆT ĐỐI KHÔNG đặt hộp thoại che mất khuôn mặt, mắt, miệng hoặc chi tiết hành động chính của nhân vật.\n"
            "- Đặt các hộp thoại ở khoảng trống nền, vùng rìa, hoặc vùng ít chi tiết quanh nhân vật.\n"
            "- Kích thước `box_2d` cho bubble (`white_oval`, `dark_translucent`, `speech_tail`) PHẢI TO RÕ RÀNG để dễ đọc trên điện thoại: "
            "chiều rộng khoảng 120..250/1000 và chiều cao khoảng 120..250/1000 (tránh tạo bóng oval bé xíu dưới 100/1000).\n"
            "- Các hộp thoại tuyệt đối không được chồng chéo lên nhau.\n"
            "Bắt buộc trả về đúng JSON hợp lệ."
        )

        page_label = f" (Ảnh {page_index})" if page_index else ""
        user_prompt = (
            f"Kích thước ảnh gốc{page_label}: {w}x{h}.{bible_block}{context_block}{tags_block}{prompt_block}\n"
            f"Mật độ thoại mong muốn: {density_guide}\n"
            f"Quy tắc loại thoại:\n{style_rule}\n\n"
            "Hãy phân tích bức ảnh và trả về JSON đúng cấu trúc:\n"
            "{\n"
            '  "page_summary": "Tóm tắt 1-2 câu tình huống & xưng hô để nối tiếp cho ảnh sau",\n'
            '  "items": [\n'
            "    {\n"
            '      "id": 1,\n'
            '      "kind": "bubble | floating",\n'
            '      "bubble_style": "white_oval | dark_translucent | speech_tail | none",\n'
            '      "box_2d": [ymin, xmin, ymax, xmax],\n'
            '      "tail_target": [y, x] hoặc null,\n'
            '      "text": "Nội dung lời thoại Tiếng Việt tự nhiên",\n'
            '      "text_color": [R, G, B],\n'
            '      "stroke_color": [R, G, B] hoặc null,\n'
            '      "angle": 0\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        images_payload: list[tuple[str, str]] = [("Bức tranh CG cần sáng tác thoại:", img_b64)]
        if not story_bible:
            if prev_rgb is not None:
                images_payload.append(
                    ("(Tham chiếu cảnh CG trước để nối tiếp cảm xúc):", _pil_to_jpeg_b64(Image.fromarray(prev_rgb), max_side=720, quality=76))
                )
            if next_rgb is not None:
                images_payload.append(
                    ("(Tham chiếu cảnh CG sau để dẫn dắt hợp lý):", _pil_to_jpeg_b64(Image.fromarray(next_rgb), max_side=720, quality=76))
                )

        data = self._call_json(
            system_prompt,
            user_prompt,
            images=images_payload,
            temperature=0.65,
        )

        items = _parse_composed_items(data.get("items", []), w=w, h=h, preset=preset)
        return items, str(data.get("page_summary", "")).strip()

    def compose_cg_sheet(
        self,
        image_paths: list[Path],
        image_sizes: list[tuple[int, int]],
        preset: ComposePreset = ComposePreset.CG_OVAL,
        density: str = "medium",
        story_prompt: str | None = None,
        story_bible: StoryBible | None = None,
        tags: str | None = None,
    ) -> dict[int, tuple[list[TextRegionItem], str]]:
        """
        Batch CG composition on numbered contact sheets:
        - Up to 18 pages -> 1 single full sheet call.
        - > 18 pages (e.g. 50-100+ CGs) -> automatically splits into crisp 12-page sheets
          preserving plot continuity across batches so characters & details stay sharp.
        """
        total = len(image_paths)
        if total == 0:
            return {}
        if total > 18:
            results: dict[int, tuple[list[TextRegionItem], str]] = {}
            running_summaries: list[str] = []
            batch_size = 12
            for start in range(0, total, batch_size):
                end = min(total, start + batch_size)
                page_nums = list(range(start + 1, end + 1))
                self._emit_status(
                    f"Sáng tác batch Trang {page_nums[0]}–{page_nums[-1]}/{total} trên sheet nét cao..."
                )
                batch_res = self._compose_cg_sheet_batch(
                    image_paths=image_paths[start:end],
                    image_sizes=image_sizes[start:end],
                    page_numbers=page_nums,
                    total_pages=total,
                    preset=preset,
                    density=density,
                    story_prompt=story_prompt,
                    story_bible=story_bible,
                    previous_summaries=running_summaries,
                    tags=tags,
                )
                for p_no, val in sorted(batch_res.items()):
                    results[p_no] = val
                    if val[1]:
                        running_summaries.append(val[1])
            return results

        return self._compose_cg_sheet_batch(
            image_paths=image_paths,
            image_sizes=image_sizes,
            page_numbers=list(range(1, total + 1)),
            total_pages=total,
            preset=preset,
            density=density,
            story_prompt=story_prompt,
            story_bible=story_bible,
            previous_summaries=None,
            tags=tags,
        )

    def _compose_cg_sheet_batch(
        self,
        image_paths: list[Path],
        image_sizes: list[tuple[int, int]],
        page_numbers: list[int],
        total_pages: int,
        preset: ComposePreset = ComposePreset.CG_OVAL,
        density: str = "medium",
        story_prompt: str | None = None,
        story_bible: StoryBible | None = None,
        previous_summaries: list[str] | None = None,
        tags: str | None = None,
    ) -> dict[int, tuple[list[TextRegionItem], str]]:
        sheets = _build_story_contact_sheets(
            image_paths,
            max_pages=len(image_paths),
            page_numbers=page_numbers,
            total_pages=total_pages,
        )
        if not sheets:
            return {}

        density_guide = {
            "low": "1 đến 3 câu thoại ngắn gọn mỗi ảnh, chừa nhiều khoảng trống cho tranh.",
            "medium": "3 đến 5 câu thoại/biểu cảm cân đối mỗi ảnh, nhịp truyện tự nhiên.",
            "high": "5 đến 8 câu thoại mỗi ảnh gồm thoại chính, suy nghĩ thầm và tiếng thốt/SFX.",
        }.get(density, "3 đến 5 câu thoại cân đối mỗi ảnh.")
        style_rule = _compose_style_rule(preset)
        bible_block = ""
        if story_bible:
            bible_txt = story_bible.to_prompt_block()
            if bible_txt:
                bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT TOÀN BỘ CG ===\n{bible_txt}\n==========================================\n"
        prev_block = ""
        if previous_summaries:
            joined = "\n".join(f"- Cảnh trước: {s}" for s in previous_summaries[-6:])
            prev_block = f"\nDiễn biến liền trước cụm ảnh này (giữ mạch nối tiếp):\n{joined}\n"
        tags_block = _format_tags_block(tags)
        prompt_block = f"\nChỉ dẫn kịch bản / phong cách từ người dùng: {story_prompt}\n" if story_prompt else ""

        system_prompt = (
            "Bạn là biên kịch & đạo diễn dàn trang truyện tranh/CG chuyên nghiệp.\n"
            "Nhiệm vụ: Quan sát toàn bộ các ô ảnh được đánh số 'Trang X/Y' trên sheet tổng hợp, "
            "sáng tác lời thoại Tiếng Việt theo KHẨU NGỮ ĐỜI THƯỜNG thật tự nhiên, mạch lạc theo đúng số trang, "
            "và trả về tọa độ đặt chữ riêng cho TỪNG TRANG.\n"
            "QUY TẮC TỌA ĐỘ CỰC KỲ QUAN TRỌNG:\n"
            "- Với mỗi trang, `box_2d: [ymin, xmin, ymax, xmax]` tính từ 0 đến 1000 TRONG PHẠM VI Ô ẢNH CỦA TRANG ĐÓ (không phải tọa độ cả sheet).\n"
            "- Kích thước `box_2d` cho bubble (`white_oval`, `dark_translucent`, `speech_tail`) PHẢI TO RÕ RÀNG (rộng ~120..250/1000, cao ~120..250/1000) để chữ dễ đọc trên điện thoại.\n"
            "- TUYỆT ĐỐI KHÔNG che khuôn mặt, mắt, miệng hoặc hành động chính; các hộp không được chồng chéo.\n"
            "Bắt buộc trả về đúng JSON hợp lệ."
        )
        page_range_label = f"Trang {page_numbers[0]} đến {page_numbers[-1]}"
        user_prompt = (
            f"Các trang cần sáng tác trên sheet này ({len(image_paths)} ảnh: {page_range_label} / tổng {total_pages}):"
            f"{bible_block}{prev_block}{tags_block}{prompt_block}\n"
            f"Mật độ thoại mỗi ảnh: {density_guide}\n"
            f"Quy tắc loại thoại:\n{style_rule}\n\n"
            f"Hãy trả về JSON cho ĐỦ TẤT CẢ {len(image_paths)} trang ({page_range_label}) theo đúng số `page` ghi trên đầu mỗi ô ảnh:\n"
            "{\n"
            '  "pages": [\n'
            "    {\n"
            f'      "page": {page_numbers[0]},\n'
            '      "page_summary": "Tóm tắt 1 câu diễn biến trang này",\n'
            '      "items": [\n'
            "        {\n"
            '          "id": 1,\n'
            '          "kind": "bubble | floating",\n'
            '          "bubble_style": "white_oval | dark_translucent | speech_tail | none",\n'
            '          "box_2d": [ymin, xmin, ymax, xmax],\n'
            '          "tail_target": [y, x] hoặc null,\n'
            '          "text": "Nội dung lời thoại Tiếng Việt tự nhiên",\n'
            '          "text_color": [R, G, B],\n'
            '          "stroke_color": [R, G, B] hoặc null,\n'
            '          "angle": 0\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}"
        )
        data = self._call_json(system_prompt, user_prompt, images=sheets, temperature=0.65)
        page_map = {p_no: sz for p_no, sz in zip(page_numbers, image_sizes)}
        results: dict[int, tuple[list[TextRegionItem], str]] = {}
        for p_obj in data.get("pages", []):
            try:
                p_no = int(p_obj.get("page", 0))
            except (TypeError, ValueError):
                continue
            if p_no in page_map:
                w, h = page_map[p_no]
                items = _parse_composed_items(p_obj.get("items", []), w=w, h=h, preset=preset)
                results[p_no] = (items, str(p_obj.get("page_summary", "")).strip())
        return results


def _compose_style_rule(preset: ComposePreset) -> str:
    if preset == ComposePreset.CG_OVAL:
        return (
            "- BẮT BUỘC dùng kết hợp giữa `kind='bubble'` với `bubble_style` CHỈ ĐƯỢC CHỌN "
            "`white_oval` (oval dọc trắng cho thoại nói ra) hoặc `dark_translucent` (oval dọc đen bán trong suốt cho suy nghĩ thầm/lời trêu chọc), "
            "VÀ `kind='floating'` (`bubble_style='none'`) cho chữ thường ngắn/âm thanh đặt trực tiếp lên tranh gần hành động.\n"
            "- Không dùng đuôi nhọn (tail). Các khung `white_oval` và `dark_translucent` thường có dáng thon đứng hoặc bầu dục gọn gàng đặt ở vùng nền trống."
        )
    if preset == ComposePreset.MANGA_TAIL:
        return (
            "- Dùng kết hợp `kind='bubble'` với `bubble_style='speech_tail'` (bóng thoại trắng có đuôi trỏ về phía miệng nhân vật nói, kèm `tail_target: [y, x]` trong hệ 0..1000), "
            "`white_oval` / `dark_translucent` cho suy nghĩ thầm, VÀ `kind='floating'` (`bubble_style='none'`) cho chữ thường/thì thầm/SFX trực tiếp trên tranh."
        )
    if preset == ComposePreset.FLOATING_ONLY:
        return (
            "- CHỈ DÙNG `kind='floating'` (`bubble_style='none'`): toàn bộ thoại viết trực tiếp lên tranh với màu chữ + viền mảnh tương phản, không vẽ khung bubble."
        )
    return (
        "- Tự do kết hợp hài hòa cả 2 loại:\n"
        "  1) `kind='bubble'`: chọn `speech_tail` (khi cần chỉ rõ người nói kèm `tail_target: [y, x]`), `white_oval`, hoặc `dark_translucent` (suy nghĩ thầm / lẩm bẩm).\n"
        "  2) `kind='floating'` (`bubble_style='none'`): chữ thường đặt trực tiếp gần miệng/hành động/góc tranh cho tiếng thốt, thì thầm hoặc SFX."
    )


def _parse_composed_items(
    raw_items: list[dict[str, Any]],
    w: int,
    h: int,
    preset: ComposePreset,
) -> list[TextRegionItem]:
    items: list[TextRegionItem] = []
    for idx, raw in enumerate(raw_items or [], start=1):
        text = str(raw.get("text", "")).strip()
        box = raw.get("box_2d")
        if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) for v in box]
        if ymax <= ymin or xmax <= xmin:
            continue
        x1 = int(round(xmin / 1000.0 * w))
        y1 = int(round(ymin / 1000.0 * h))
        x2 = int(round(xmax / 1000.0 * w))
        y2 = int(round(ymax / 1000.0 * h))
        if (x2 - x1) < 28 or (y2 - y1) < 20:
            continue

        kind = "floating" if raw.get("kind") == "floating" or preset == ComposePreset.FLOATING_ONLY else "bubble"
        raw_style = str(raw.get("bubble_style", "white_oval")).lower()
        if kind == "floating":
            b_style = BubbleStyle.NONE
        else:
            if preset == ComposePreset.CG_OVAL and raw_style == "speech_tail":
                raw_style = "white_oval"
            try:
                b_style = BubbleStyle(raw_style)
            except ValueError:
                b_style = BubbleStyle.WHITE_OVAL
            if b_style == BubbleStyle.NONE:
                b_style = BubbleStyle.WHITE_OVAL

        is_dark = b_style == BubbleStyle.DARK_TRANSLUCENT

        tail_pt: tuple[int, int] | None = None
        raw_tail = raw.get("tail_target")
        if b_style == BubbleStyle.SPEECH_TAIL and isinstance(raw_tail, (list, tuple)) and len(raw_tail) == 2:
            ty, tx = float(raw_tail[0]), float(raw_tail[1])
            tail_pt = (
                int(round(max(0.0, min(1000.0, tx)) / 1000.0 * w)),
                int(round(max(0.0, min(1000.0, ty)) / 1000.0 * h)),
            )

        if kind == "bubble":
            t_col = (255, 255, 255) if is_dark else (18, 18, 18)
            s_col = None
        else:
            rc = raw.get("text_color")
            sc = raw.get("stroke_color")
            t_col = (
                (int(rc[0]), int(rc[1]), int(rc[2]))
                if isinstance(rc, (list, tuple)) and len(rc) == 3
                else (255, 255, 255)
            )
            s_col = (
                (int(sc[0]), int(sc[1]), int(sc[2]))
                if isinstance(sc, (list, tuple)) and len(sc) == 3
                else (18, 18, 18)
            )
            t_luma = 0.299 * t_col[0] + 0.587 * t_col[1] + 0.114 * t_col[2]
            s_luma = 0.299 * s_col[0] + 0.587 * s_col[1] + 0.114 * s_col[2]
            if abs(t_luma - s_luma) < 90.0:
                s_col = (18, 18, 18) if t_luma >= 135.0 else (255, 255, 255)

        angle = float(raw.get("angle") or 0.0)
        angle = max(-22.0, min(22.0, angle)) if kind == "floating" else 0.0

        items.append(
            TextRegionItem(
                id=int(raw.get("id", idx)),
                kind=kind,
                bubble_style=b_style,
                text_bbox=(x1, y1, x2, y2),
                bubble_bbox=(x1, y1, x2, y2) if kind == "bubble" else None,
                tail_target=tail_pt,
                source_text="",
                translated_text=text,
                text_color=t_col,
                stroke_color=s_col,
                stroke_width=None,
                font_name="pangolin" if kind == "floating" else "mali-bold",
                angle=angle,
                is_dark_bubble=is_dark,
                draw_bubble=(kind == "bubble"),
            )
        )

    if items:
        items = resolve_box_overlaps(items, w, h)
    return items
