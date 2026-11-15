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
    StoryBible,
    TextRegionItem,
)
from koharu_vn_cli.pipeline.bubble_renderer import resolve_box_overlaps
from koharu_vn_cli.pipeline.fonts import resolve_font_path

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class ContentBlockedError(RuntimeError):
    """Raised when the upstream API hard-blocks content (e.g. PROHIBITED_CONTENT / SAFETY)."""


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


def _extract_json_block(raw: str) -> Any:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Recover first top-level JSON object or array
        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [idx for idx in (start_obj, start_arr) if idx >= 0]
        if not starts:
            raise
        start = min(starts)
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        end = text.rfind(closer)
        if end > start:
            return json.loads(text[start : end + 1])
        raise


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
) -> list[tuple[str, str]]:
    """
    Pack chapter/CG pages into compact numbered contact sheets (or full-res single image)
    so Gemini sees the entire artwork/story before generating dialogue or translations.
    """
    total = len(image_paths)
    if total == 0:
        return []

    if total == 1:
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

    if total > max_pages:
        indices = sorted({int(round(i * (total - 1) / (max_pages - 1))) for i in range(max_pages)})
    else:
        indices = list(range(total))

    selected: list[tuple[int, Image.Image]] = []
    for idx in indices:
        try:
            with Image.open(image_paths[idx]) as im:
                selected.append((idx + 1, im.convert("RGB")))
        except Exception:
            continue
    if not selected:
        return []

    total_sel = len(selected)
    if total_sel <= 9:
        tile_w, tile_h = 340, 440
    elif total_sel <= 16:
        tile_w, tile_h = 280, 360
    elif total_sel <= 25:
        tile_w, tile_h = 230, 300
    else:
        tile_w, tile_h = 190, 250

    resized: list[tuple[int, Image.Image]] = []
    for page_no, rgb in selected:
        copy = rgb.copy()
        copy.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        resized.append((page_no, copy))

    # Pack all selected pages onto 1 single full contact sheet for speed & lower token cost
    sheets: list[tuple[str, str]] = []
    lbl_font = ImageFont.truetype(str(resolve_font_path("mali-bold")), size=16 if total_sel > 16 else 18)
    per_sheet = max(1, total_sel)
    for start in range(0, len(resized), per_sheet):
        chunk = resized[start : start + per_sheet]
        cols = min(6, max(1, math.ceil(math.sqrt(len(chunk)))))
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
                f"Sheet tổng hợp full {len(chunk)} trang (từ Trang {first_no} đến Trang {last_no} / tổng {total}):",
                _pil_to_jpeg_b64(canvas, max_side=1800 if total_sel > 9 else 1500, quality=84),
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


class VisionLLMClient:
    """Unified Gemini API & OpenAI-compatible Vision client."""

    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        base_url: str | None = None,
        timeout: float = 90.0,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        raw_sources: list[str] = []
        if api_keys:
            raw_sources.extend(api_keys)
        for src in (
            api_key,
            os.environ.get("GEMINI_API_KEYS"),
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        ):
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
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.on_status = on_status
        if not self.api_key and not self.base_url:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY env var or pass --api-key (or --base-url for a local proxy)."
            )

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
        """Send multimodal prompt + base64 JPEGs and parse JSON response with automatic key failover & retries."""
        img_kb = max(1, sum(len(b64) for _, b64 in images) * 3 // 4 // 1024) if images else 0
        last_err: Exception | None = None
        keys_total = max(1, len(self.api_keys))
        keys_tried = 1
        transient_attempt = 0
        max_transient = 3

        while True:
            try:
                if transient_attempt == 0 and keys_tried == 1:
                    key_tag = f" [key {self._key_idx + 1}/{keys_total}]" if keys_total > 1 else ""
                    self._emit_status(
                        f"Đang gửi {len(images)} ảnh (~{img_kb} KB) tới {self.model}{key_tag} (timeout {int(self.timeout)}s)..."
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
                if code == 404:
                    hint = f" (Model '{self.model}' không tồn tại hoặc key chưa mở — bấm F3 hoặc gõ /model để chọn model hỗ trợ)"
                    raise RuntimeError(f"HTTP {code} từ {self.model}: {body_preview}{hint}") from exc
                if code in (400, 401, 403, 429) or code >= 500:
                    if keys_tried < keys_total and self._rotate_key(f"HTTP {code}"):
                        keys_tried += 1
                        continue
                    if code in (400, 401, 403):
                        hint = " (API key không hợp lệ hoặc hết quyền — kiểm tra /key)" if code in (401, 403) else ""
                        raise RuntimeError(f"HTTP {code} từ {self.model}: {body_preview}{hint}") from exc
                if transient_attempt < max_transient:
                    wait_s = 1.5 * (2**transient_attempt)
                    transient_attempt += 1
                    self._emit_status(f"API trả HTTP {code}, thử lại lần {transient_attempt}/{max_transient} sau {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                break
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_err = exc
                if isinstance(exc, httpx.HTTPError) and keys_tried < keys_total and self._rotate_key(type(exc).__name__):
                    keys_tried += 1
                    continue
                if transient_attempt < max_transient:
                    wait_s = 1.5 * (2**transient_attempt)
                    transient_attempt += 1
                    self._emit_status(
                        f"Lỗi tạm thời ({type(exc).__name__}), thử lại lần {transient_attempt}/{max_transient} sau {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)
                    continue
                break
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

        gen_cfg: dict[str, Any] = {"temperature": temperature}
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
            data = resp.json()

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
        text_parts = [p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])]
        return "\n".join(text_parts)

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
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for label, b64 in images:
            if label:
                content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
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
            data = resp.json()
        return data["choices"][0]["message"]["content"]

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
            for m in raw_items:
                mid = str(m.get("id", "")).strip()
                if not mid:
                    continue
                owner = str(m.get("owned_by", "")).strip()
                models.append((mid, f"Proxy model ({owner})" if owner else "Proxy model"))
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

    def build_story_bible(
        self,
        image_paths: list[Path],
        mode: str = "translate",
        user_story: str | None = None,
        extra_instructions: str | None = None,
        tags: str | None = None,
    ) -> StoryBible:
        """
        Pass 1 global analysis: scan the whole chapter / CG set before translating Page 1
        to lock characters, visual appearance, Vietnamese pronoun rules, and full plot arc.
        """
        sheets = _build_story_contact_sheets(image_paths)
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
            "Phải khai thác tối đa chi tiết thị giác trong full ảnh (ánh mắt, cử chỉ, trang phục, đạo cụ, bối cảnh) "
            "để tự suy luận toàn bộ ngữ cảnh mà không cần người dùng giải thích.\n"
            "Ưu tiên cặp xưng hô tự nhiên theo đúng quan hệ đời thực (anh/em, cậu/tớ, tao/mày, thầy/em, cô/cháu, ta/ngươi chỉ dùng cho cổ trang/fantasy).\n"
            "Trả về đúng JSON hợp lệ."
        )
        user_prompt = (
            f"Tổng số ảnh: {len(image_paths)}.{user_hint}{tags_hint}{extra_hint}\n"
            "Hãy trả về JSON Story Bible đúng cấu trúc:\n"
            "{\n"
            '  "title": "Tên gợi ý hoặc chủ đề chính của cảnh/chương",\n'
            '  "genre_and_tone": "Thể loại & không khí (vd: lãng mạn trêu chọc, kịch tính, học đường...)",\n'
            '  "pre_event_backstory": "Cốt truyện xảy ra TRƯỚC sự kiện trong ảnh (2-3 câu: chuyện gì đã dẫn nhân vật tới tình huống này)",\n'
            '  "during_event_plot": "Diễn biến xảy ra TRONG khi sự kiện ảnh diễn ra (2-4 câu: hành động, cảm xúc, đối đáp để làm nền cho lời thoại)",\n'
            '  "full_plot_summary": "Tóm tắt mạch truyện hoàn chỉnh từ trước -> trong -> hướng diễn biến (3-5 câu)",\n'
            '  "characters": [\n'
            '    {"name": "Tên hoặc danh xưng tự đặt", "appearance": "ngoại hình nhận diện...", "role_and_tone": "vai trò, tính cách", "pronoun_rules": "cặp xưng hô Tiếng Việt bắt buộc (vd: anh - em / chị - em)"}\n'
            "  ],\n"
            '  "page_beats": {"1": "Tình huống cụ thể ảnh 1", "2": "Tình huống ảnh 2"}\n'
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
        raw_beats = data.get("page_beats") or {}
        beats = {str(k): str(v).strip() for k, v in raw_beats.items() if str(v).strip()}
        return StoryBible(
            title=str(data.get("title", "")).strip(),
            genre_and_tone=str(data.get("genre_and_tone", "")).strip(),
            pre_event_backstory=str(data.get("pre_event_backstory", "")).strip(),
            during_event_plot=str(data.get("during_event_plot", "")).strip(),
            full_plot_summary=str(data.get("full_plot_summary", "")).strip(),
            characters=chars,
            page_beats=beats,
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
        extra_block = f"\nYêu cầu thêm từ người dùng: {extra_instructions}\n" if extra_instructions else ""

        system_prompt = (
            "Bạn là biên dịch viên truyện tranh & doujinshi kỳ cựu (Manga Nhật, Manhwa Hàn, Manhua Trung) chuyên bản địa hóa sang Tiếng Việt.\n"
            "Nhiệm vụ: Đọc chính xác chữ gốc trong từng khung [ID] từ Bảng Crop chi tiết kết hợp Ảnh toàn trang, "
            "rồi dịch thoát ý sang KHẨU NGỮ TIẾNG VIỆT ĐỜI THƯỜNG thật tự nhiên, gợi cảm/gợi tình đúng mức theo đúng sắc thái cảnh truyện và xưng hô đã khóa.\n"
            "NGUYÊN TẮC VIỆT HÓA TỰ NHIÊN & GỢI CẢM ĐÚNG CONTEXT (CỰC KỲ QUAN TRỌNG):\n"
            "- Viết như người Việt ngoài đời đang nói chuyện/thì thầm/rên rỉ thật; TUYỆT ĐỐI KHÔNG dịch word-by-word, không văn viết, không lối diễn văn sách vở hay từ y khoa khô khan.\n"
            "- Khi cảnh mang sắc thái thân mật/18+/trêu chọc, hãy dùng ngôn từ mềm mại, ướt át, gợi cảm hoặc táo bạo đúng với biểu cảm & ngữ cảnh gốc (vd: 'nhạy cảm quá...', 'đừng chạm chỗ đó mà...', 'sâu quá... chậm chút đi...', 'cứng hết cả lên rồi kìa', 'chịu hết nổi rồi...', 'bên trong khít chặt quá...').\n"
            "- TUYỆT ĐỐI TRÁNH từ y khoa khô khan ('âm đạo', 'dương vật', 'giao hợp') và tiếng rên kiểu Anh ('UGH', 'GUH', 'BUH', 'NNGH', 'MMN', 'FUH'): đổi sang khẩu ngữ Việt tự nhiên theo sắc thái cảnh (nhẹ/vanilla: 'bên trong', 'chỗ đó', 'nóng hổi', 'ướt nhẹp'; thô/hardcore/NTR: 'nứng', 'chịch', 'bắn vào trong', 'dính bầu'...) và tiếng thở Việt ('Ư...', 'Ưm...', 'Á...', 'Aaa...', 'Hộc...', 'Khựa...').\n"
            "- CẤM dùng từ cứng, sáo rỗng ít ai nói ngoài đời như: 'hắn ta', 'ả', 'gã', 'cô nàng', 'thực sự là', 'vô cùng', 'cái đó', 'vậy sao', 'thèm thuồng lắm chứ', 'kẻ hầu', 'ấn ký' (trừ khi đúng bối cảnh kỳ ảo cổ trang: mới dùng 'ta - ngươi' hoặc danh xưng cổ trang).\n"
            "- Thay bằng khẩu ngữ ngắn gọn, có nhịp thở ngắt quãng (`...`, `!?`) và tiểu từ tự nhiên (à, ừ, nhỉ, đấy, chứ, đâu, cơ mà, mất rồi, luôn ấy, nha...): vd 'Cứ tưởng cô không thích cơ đấy', 'Ủa sao tự dưng...', 'Đừng có mà...', 'Nhói quá...'.\n"
            "- Câu ngắn gọn vừa khung thoại, giữ đúng sắc thái ngại ngùng / trêu chọc / đê mê / hoảng hốt / áp đảo trên nét mặt nhân vật.\n"
            "- Không giải thích dài dòng, không thêm chú thích ngoài lời thoại.\n"
            "- Tiếng động / SFX & tiếng thở dịch tự nhiên đúng chất doujinshi (vd: 'Ư...', 'Á...', 'Hộc...', 'CHỤT', 'NHÓP NHÉP', 'PHỤT', 'HẢ?!').\n"
            "Bắt buộc trả về đúng JSON hợp lệ."
        )

        lang_hint = (
            "Tự nhận diện Nhật/Hàn/Trung/Anh" if source_lang == "auto" else f"Ngôn ngữ gốc ưu tiên: {source_lang}"
        )
        ids_list = [r.id for r in regions]
        page_label = f" (Trang {page_index})" if page_index else ""
        user_prompt = (
            f"{lang_hint}{page_label}.{bible_block}{context_block}{tags_block}{extra_block}\n"
            f"Hãy đọc và dịch toàn bộ các ID sau: {ids_list}.\n"
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

        by_id: dict[int, dict[str, Any]] = {}
        for item in data.get("items", []):
            try:
                by_id[int(item["id"])] = item
            except (KeyError, TypeError, ValueError):
                continue

        updated: list[TextRegionItem] = []
        for reg in regions:
            copy = reg.model_copy()
            match = by_id.get(copy.id)
            if match:
                copy.source_text = str(match.get("source_text", "")).strip()
                copy.translated_text = str(match.get("vietnamese", "")).strip()
            updated.append(copy)

        return updated, str(data.get("page_summary", "")).strip()

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
            "- Kích thước `box_2d` phải đủ rộng rãi để chứa vừa câu thoại (chiều rộng/cao hợp lý, không quá bé).\n"
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
        Fast 1-sheet batch CG composition: pack all images onto 1 numbered contact sheet
        and compose Vietnamese dialogue + per-panel normalized coordinates in a single API call.
        """
        sheets = _build_story_contact_sheets(image_paths, max_pages=36)
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
        tags_block = _format_tags_block(tags)
        prompt_block = f"\nChỉ dẫn kịch bản / phong cách từ người dùng: {story_prompt}\n" if story_prompt else ""

        system_prompt = (
            "Bạn là biên kịch & đạo diễn dàn trang truyện tranh/CG chuyên nghiệp.\n"
            "Nhiệm vụ: Quan sát toàn bộ các ô ảnh được đánh số 'Trang 1..N' trên sheet tổng hợp, "
            "sáng tác lời thoại Tiếng Việt theo KHẨU NGỮ ĐỜI THƯỜNG thật tự nhiên, mạch lạc từ Trang 1 đến Trang cuối, "
            "và trả về tọa độ đặt chữ riêng cho TỪNG TRANG.\n"
            "QUY TẮC TỌA ĐỘ CỰC KỲ QUAN TRỌNG:\n"
            "- Với mỗi trang, `box_2d: [ymin, xmin, ymax, xmax]` tính từ 0 đến 1000 TRONG PHẠM VI Ô ẢNH CỦA TRANG ĐÓ (không phải tọa độ cả sheet).\n"
            "- TUYỆT ĐỐI KHÔNG che khuôn mặt, mắt, miệng hoặc hành động chính; các hộp không được chồng chéo.\n"
            "Bắt buộc trả về đúng JSON hợp lệ."
        )
        user_prompt = (
            f"Tổng số ảnh cần sáng tác trên sheet: {len(image_paths)}.{bible_block}{tags_block}{prompt_block}\n"
            f"Mật độ thoại mỗi ảnh: {density_guide}\n"
            f"Quy tắc loại thoại:\n{style_rule}\n\n"
            f"Hãy trả về JSON cho ĐỦ TẤT CẢ {len(image_paths)} trang theo đúng cấu trúc:\n"
            "{\n"
            '  "pages": [\n'
            "    {\n"
            '      "page": 1,\n'
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
        results: dict[int, tuple[list[TextRegionItem], str]] = {}
        for p_obj in data.get("pages", []):
            try:
                p_no = int(p_obj.get("page", 0))
            except (TypeError, ValueError):
                continue
            if 1 <= p_no <= len(image_sizes):
                w, h = image_sizes[p_no - 1]
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
