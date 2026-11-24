"""Gemini Vision API & OpenAI-compatible Vision client for Manga/Manhwa/Manhua translation and CG dialogue composition."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODELS = (
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
)
GROQ_MODELS = (
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
)
_GROQ_MODEL_PREFIXES = (
    "meta-llama/llama-4-",
    "llama-3.3-",
    "llama-3.1-",
    "llama3-",
    "qwen/qwen3-",
    "qwen-qwq-",
    "moonshotai/kimi-k2",
    "openai/gpt-oss-",
    "compound-beta",
    "groq/",
    "gemma2-9b-it",
    "mixtral-8x7b-",
)


def is_deepseek_model(model: str | None) -> bool:
    """Return True when the model ID targets DeepSeek's API."""
    return (model or "").strip().lower().startswith("deepseek-")


def is_deepseek_endpoint(base_url: str | None) -> bool:
    """Return True when base_url points to official DeepSeek API."""
    return "api.deepseek.com" in (base_url or "").lower()


def is_groq_endpoint(base_url: str | None) -> bool:
    """Return True when base_url points to Groq's OpenAI-compatible API."""
    return "api.groq.com" in (base_url or "").lower()


def is_groq_vision_model(model: str | None) -> bool:
    """Return True when a Groq model supports multimodal `image_url` vision inputs."""
    low = (model or "").strip().lower()
    return any(
        tok in low
        for tok in (
            "llama-4-scout",
            "llama-4-maverick",
            "llama-3.2-11b-vision",
            "llama-3.2-90b-vision",
            "-vision",
        )
    )


def is_groq_model(model: str | None) -> bool:
    """Return True when the model ID targets Groq's API."""
    low = (model or "").strip().lower()
    if not low:
        return False
    if low in {m.lower() for m in GROQ_MODELS}:
        return True
    return low.startswith(_GROQ_MODEL_PREFIXES)


def is_text_only_model(model: str | None, base_url: str | None = None) -> bool:
    """Return True when the target model/endpoint cannot accept image_url vision inputs."""
    if is_deepseek_endpoint(base_url):
        return True
    if is_deepseek_model(model) and not base_url:
        return True
    if is_groq_endpoint(base_url) or (is_groq_model(model) and not base_url):
        return not is_groq_vision_model(model)
    return False


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


def _is_auth_error(status_code: int, body_text: str) -> bool:
    if status_code in (401, 403) and not _is_quota_error(status_code, body_text):
        return True
    if status_code == 400 and re.search(
        r"API_KEY_INVALID|invalid api key|api key not valid|expired|permission_denied",
        body_text or "",
        re.IGNORECASE,
    ):
        return True
    return False


_DEFAULT_KEY_HEALTH_PATH = Path.home() / ".koharu-vn" / "key_health.json"
_KEY_HEALTH_RETENTION_SECONDS = 7 * 24 * 3600
_COOLDOWN_SCHEDULE: dict[str, tuple[float, ...]] = {
    "auth": (1800.0, 3600.0, 7200.0),
    "quota": (300.0, 900.0, 1800.0),
    "transient": (60.0, 180.0, 300.0),
}


@dataclass
class _KeyStat:
    use_count: int = 0
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_used_at: float = 0.0
    last_success_at: float = 0.0
    last_error_at: float = 0.0
    cooldown_until: float = 0.0
    last_error_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_count": self.use_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_used_at": self.last_used_at,
            "last_success_at": self.last_success_at,
            "last_error_at": self.last_error_at,
            "cooldown_until": self.cooldown_until,
            "last_error_kind": self.last_error_kind,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> _KeyStat:
        return cls(
            use_count=int(raw.get("use_count", 0) or 0),
            success_count=int(raw.get("success_count", 0) or 0),
            error_count=int(raw.get("error_count", 0) or 0),
            consecutive_errors=int(raw.get("consecutive_errors", 0) or 0),
            last_used_at=float(raw.get("last_used_at", 0.0) or 0.0),
            last_success_at=float(raw.get("last_success_at", 0.0) or 0.0),
            last_error_at=float(raw.get("last_error_at", 0.0) or 0.0),
            cooldown_until=float(raw.get("cooldown_until", 0.0) or 0.0),
            last_error_kind=str(raw.get("last_error_kind", "") or ""),
        )


def _endpoint_scope(base_url: str | None, model: str) -> str:
    if base_url:
        return base_url.strip().rstrip("/").lower()
    if is_deepseek_model(model):
        return DEEPSEEK_BASE_URL.lower()
    if is_groq_model(model):
        return GROQ_BASE_URL.lower()
    return "gemini-official"


def _key_hash(namespace: str, api_key: str) -> str:
    return hashlib.sha256(f"{namespace}::{api_key}".encode("utf-8")).hexdigest()[:24]


class SmartKeyHealthTracker:
    """
    Thread-safe, persistent API key health & smart rotation scheduler.

    Prioritizes:
      0) Healthy & never-used keys (e.g., if keys 1..4 previously errored, jump straight to 5, 6, 7...)
      1) Healthy & previously succeeded keys (least-recently-used + fewest in-flight calls)
      2) Previously errored keys whose cooldown has already expired
      3) Keys still in active cooldown (last resort only, soonest-recovering non-auth-dead first)
    """

    def __init__(self, state_path: Path | None = None, *, persist: bool = True) -> None:
        self._lock = threading.Lock()
        self._persist = persist
        self._state_path = state_path
        self._stats: dict[str, _KeyStat] = {}
        self._in_flight: dict[str, int] = {}
        self._loaded = False

    def _effective_state_path(self) -> Path | None:
        if not self._persist:
            return None
        if self._state_path is not None:
            return self._state_path
        env_p = (os.environ.get("KOHARU_KEY_HEALTH_FILE") or "").strip()
        if env_p:
            return Path(env_p)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return None
        return _DEFAULT_KEY_HEALTH_PATH

    def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._effective_state_path()
        if path is None or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload.get("entries") if isinstance(payload, dict) else None
            if not isinstance(entries, dict):
                return
            cutoff = time.time() - _KEY_HEALTH_RETENTION_SECONDS
            for k_id, raw in entries.items():
                if not isinstance(raw, dict):
                    continue
                st = _KeyStat.from_dict(raw)
                latest_ts = max(st.last_used_at, st.last_success_at, st.last_error_at, st.cooldown_until)
                if latest_ts <= 0.0 or latest_ts >= cutoff:
                    self._stats[str(k_id)] = st
        except Exception:
            pass

    def _save_locked(self) -> None:
        path = self._effective_state_path()
        if path is None:
            return
        try:
            cutoff = time.time() - _KEY_HEALTH_RETENTION_SECONDS
            pruned = {
                k_id: st.to_dict()
                for k_id, st in self._stats.items()
                if max(st.last_used_at, st.last_success_at, st.last_error_at, st.cooldown_until) >= cutoff
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "entries": pruned}, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            pass

    def _ids(self, api_key: str, model: str, base_url: str | None) -> tuple[str, str]:
        ep = _endpoint_scope(base_url, model)
        model_id = _key_hash(f"m::{ep}::{(model or '').strip().lower()}", api_key)
        auth_id = _key_hash(f"auth::{ep}", api_key)
        return model_id, auth_id

    def _eval_rank_locked(
        self,
        idx: int,
        n_keys: int,
        api_key: str,
        model: str,
        base_url: str | None,
        now: float,
        preferred_slot: int,
    ) -> tuple[tuple[Any, ...], bool]:
        model_id, auth_id = self._ids(api_key, model, base_url)
        ms = self._stats.get(model_id) or _KeyStat()
        as_ = self._stats.get(auth_id) or _KeyStat()

        eff_cd = max(ms.cooldown_until, as_.cooldown_until)
        in_cooldown = now < eff_cd
        eff_consec = max(ms.consecutive_errors, as_.consecutive_errors)
        ever_errored = (ms.error_count > 0) or (as_.error_count > 0)
        is_auth_dead = (
            (as_.last_error_kind == "auth" and now < as_.cooldown_until)
            or (ms.last_error_kind == "auth" and now < ms.cooldown_until)
        )
        flight = self._in_flight.get(model_id, 0)
        slot_dist = (idx - (preferred_slot % n_keys)) % n_keys

        if not in_cooldown and not ever_errored and ms.use_count == 0:
            # Tier 0: fresh, completely unused healthy key!
            return (0, flight, slot_dist, idx), False
        if not in_cooldown and eff_consec == 0:
            # Tier 1: proven healthy key — spread by least-in-flight & least-recently-used
            return (1, flight, ms.last_used_at, ms.use_count, slot_dist, idx), False
        if not in_cooldown:
            # Tier 2: previously errored key whose cooldown has expired
            return (2, flight, eff_consec, ms.last_error_at, slot_dist, idx), False
        # Tier 3: currently in active cooldown
        return (3, 1 if is_auth_dead else 0, eff_cd, eff_consec, flight, slot_dist, idx), True

    def pick_initial_index(
        self,
        api_keys: list[str],
        *,
        model: str,
        base_url: str | None,
        preferred_slot: int = 0,
    ) -> int:
        """Select the highest-priority key index without mutating usage counters."""
        if not api_keys:
            return 0
        with self._lock:
            self._ensure_loaded_locked()
            now = time.time()
            n = len(api_keys)
            best_idx = 0
            best_rank: tuple[Any, ...] | None = None
            for i, key in enumerate(api_keys):
                rank, _ = self._eval_rank_locked(i, n, key, model, base_url, now, preferred_slot)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i
            return best_idx

    def acquire_key(
        self,
        api_keys: list[str],
        *,
        model: str,
        base_url: str | None,
        exclude_indices: set[int] | None = None,
        preferred_slot: int = 0,
        skip_cooldown_if_tried: bool = False,
    ) -> tuple[int | None, int]:
        """
        Atomically pick and reserve the best available key index.

        Returns `(chosen_index_or_None, cooled_down_count)`.
        When `skip_cooldown_if_tried=True` and `exclude_indices` is non-empty, returns `None`
        instead of hammering keys that are still inside their active error cooldown.
        """
        if not api_keys:
            return None, 0
        excluded = exclude_indices or set()
        with self._lock:
            self._ensure_loaded_locked()
            now = time.time()
            n = len(api_keys)
            cooled_down_total = 0
            best_idx: int | None = None
            best_rank: tuple[Any, ...] | None = None
            best_in_cd = False

            for i, key in enumerate(api_keys):
                rank, in_cd = self._eval_rank_locked(i, n, key, model, base_url, now, preferred_slot)
                if in_cd:
                    cooled_down_total += 1
                if i in excluded:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i
                    best_in_cd = in_cd

            if best_idx is None:
                return None, cooled_down_total
            if skip_cooldown_if_tried and excluded and best_in_cd:
                return None, cooled_down_total

            chosen_key = api_keys[best_idx]
            model_id, _ = self._ids(chosen_key, model, base_url)
            st = self._stats.setdefault(model_id, _KeyStat())
            st.use_count += 1
            st.last_used_at = now
            self._in_flight[model_id] = self._in_flight.get(model_id, 0) + 1
            self._save_locked()
            return best_idx, cooled_down_total

    def release_key(self, api_key: str, *, model: str, base_url: str | None) -> None:
        if not api_key:
            return
        with self._lock:
            model_id, _ = self._ids(api_key, model, base_url)
            cur = self._in_flight.get(model_id, 0)
            if cur <= 1:
                self._in_flight.pop(model_id, None)
            else:
                self._in_flight[model_id] = cur - 1

    def record_success(self, api_key: str, *, model: str, base_url: str | None) -> None:
        if not api_key:
            return
        with self._lock:
            self._ensure_loaded_locked()
            now = time.time()
            model_id, auth_id = self._ids(api_key, model, base_url)
            st = self._stats.setdefault(model_id, _KeyStat())
            st.success_count += 1
            st.consecutive_errors = 0
            st.cooldown_until = 0.0
            st.last_error_kind = ""
            st.last_success_at = now
            if auth_id in self._stats:
                ast = self._stats[auth_id]
                ast.consecutive_errors = 0
                ast.cooldown_until = 0.0
                ast.last_error_kind = ""
                ast.last_success_at = now
            self._save_locked()

    def record_error(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str | None,
        kind: str = "transient",
        cooldown_override: float | None = None,
    ) -> float:
        """Record an error on `api_key`, apply escalating cooldown, and return cooldown seconds."""
        if not api_key:
            return 0.0
        eff_kind = kind if kind in _COOLDOWN_SCHEDULE else "transient"
        with self._lock:
            self._ensure_loaded_locked()
            now = time.time()
            model_id, auth_id = self._ids(api_key, model, base_url)
            st = self._stats.setdefault(model_id, _KeyStat())
            st.error_count += 1
            st.consecutive_errors += 1
            st.last_error_at = now
            st.last_error_kind = eff_kind
            schedule = _COOLDOWN_SCHEDULE[eff_kind]
            cd_secs = (
                float(cooldown_override)
                if cooldown_override is not None
                else schedule[min(st.consecutive_errors - 1, len(schedule) - 1)]
            )
            st.cooldown_until = max(st.cooldown_until, now + cd_secs)

            if eff_kind == "auth":
                ast = self._stats.setdefault(auth_id, _KeyStat())
                ast.error_count += 1
                ast.consecutive_errors += 1
                ast.last_error_at = now
                ast.last_error_kind = "auth"
                ast.cooldown_until = max(ast.cooldown_until, now + cd_secs)

            self._save_locked()
            return cd_secs

    def reset_keys(self, api_keys: list[str] | None = None) -> None:
        """Clear saved health/cooldown state (all keys if `api_keys` is None, else matching keys)."""
        with self._lock:
            self._ensure_loaded_locked()
            if not api_keys:
                self._stats.clear()
                self._in_flight.clear()
            else:
                # Re-hash is scoped; wipe all entries matching any known provider/model only by clearing all
                # when user replaces key config so newly fixed keys start fresh immediately.
                self._stats.clear()
            self._save_locked()


_GLOBAL_KEY_HEALTH = SmartKeyHealthTracker()


def get_key_health_tracker() -> SmartKeyHealthTracker:
    return _GLOBAL_KEY_HEALTH


def configure_key_health_tracker(
    state_path: Path | None = None,
    *,
    persist: bool = True,
    clear: bool = True,
) -> SmartKeyHealthTracker:
    """Configure or reset the shared smart API key health tracker (primarily for tests & `/key` updates)."""
    global _GLOBAL_KEY_HEALTH
    if clear:
        _GLOBAL_KEY_HEALTH = SmartKeyHealthTracker(state_path=state_path, persist=persist)
    else:
        _GLOBAL_KEY_HEALTH._state_path = state_path
        _GLOBAL_KEY_HEALTH._persist = persist
        _GLOBAL_KEY_HEALTH._loaded = False
    return _GLOBAL_KEY_HEALTH


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
    if per_sheet <= 4:
        tile_w, tile_h = 520, 680
    elif per_sheet <= 6:
        tile_w, tile_h = 440, 580
    elif per_sheet <= 9:
        tile_w, tile_h = 380, 490
    elif per_sheet <= 12:
        tile_w, tile_h = 310, 400
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
                _pil_to_jpeg_b64(canvas, max_side=1800, quality=88),
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
_PURE_KATAKANA_RE = re.compile(r"^[\u3040-\u30ffーッンヴ]+$")
_STRICT_KATAKANA_RE = re.compile(r"^[\u30a0-\u30ffーッンヴ]+$")
_REDUPLICATED_KANA_SFX_RE = re.compile(r"^([\u3040-\u30ff]{2,3})\1+[ッっーーンん]*$")
_PURE_HANGUL_RE = re.compile(r"^[\uac00-\ud7af]+$")
_SFX_STEMS_RE = re.compile(
    r"^(?:"
    r"ドキ|ドク|ビク|ピク|ピタ|ツル|ニュル|ヌル|ヌチ|ヌプ|グチュ|グチョ|クチュ|クチョ|"
    r"ズチュ|ズボ|ズブ|ジュポ|ジュプ|ジュル|ズル|ピチャ|ビチャ|ヒチャ|ペチャ|ベチャ|ピシャ|ビシャ|"
    r"プチュ|ブチュ|チュパ|ペロ|ベロ|レロ|フル|プル|ブル|プニ|ムニュ|ムギュ|ムチ|グニ|プク|ボイン|ユサ|"
    r"メキ|ギシ|ギュ|キュ|ガシ|ガク|グラ|フラ|クラ|モゾ|ゴク|ボタ|ポタ|タラ|トロ|ジワ|"
    r"ドピュ|ビュル|ドバ|ビュ|ピュ|ブシュ|プシュ|シュ|コキ|シコ|ズコ|"
    r"ペチ|パチ|バチ|パキ|バキ|ピキ|ガタ|カチ|ゴゴ|ザワ|ゾク|ゾワ|シク|キラ|ヒク|チク|コツ|カサ|ガサ|ボコ|"
    r"ンン|ハァ|ゼェ|フゥ|"
    r"どき|どく|びく|ぴく|ぴた|つる|にゅる|ぬる|ぬち|ぬぷ|ぐちゅ|ぐちょ|くちゅ|くちょ|"
    r"ずちゅ|ずぼ|ずぶ|じゅぽ|じゅぷ|じゅる|ずる|ぴちゃ|びちゃ|ひちゃ|ぺちゃ|べちゃ|ぴしゃ|びしゃ|"
    r"ぷちゅ|ぶちゅ|ちゅぱ|ぺろ|べろ|れろ|ふる|ぷる|ぶる|ぷに|むにゅ|むぎゅ|むち|ぐに|ぷく|ぼいん|ゆさ|"
    r"めき|ぎし|ぎゅ|きゅ|がし|がく|ぐら|ふら|くら|もぞ|ごく|ぼた|ぽた|たら|とろ|じわ|"
    r"どぴゅ|びゅる|どば|びゅ|ぴゅ|ぶしゅ|ぷしゅ|しゅ|こき|しこ|ずこ|"
    r"ぺち|ぱち|ばち|ぱき|ばき|ぴき|がた|かち|ごご|ざわ|ぞく|ぞわ|しく|きら|ひく|ちく|こつ|かさ|がさ|ぼこ|"
    r"ぽよん|ずっぽ|んん|はぁ|ぜぇ|ふぅ|"
    r"ド[ンー]|バ[ンー]|パ[ンァー]|ポ[ンヨ]|ギ[ュヤ]|ズ[ンボ]|ゴ[ッン]|ザ[ッー]|サ[ッー]|ス[ッー]|グ[ッイ]|"
    r"ど[んー]|ば[んー]|ぱ[んぁー]|ぽ[んよ]|ぎ[ゅや]|ず[んぼ]|ご[っん]|ざ[っー]|さ[っー]|す[っー]|ぐ[っい]"
    r")",
)
_KO_SFX_RE = re.compile(
    r"^(?:"
    r"하+아*앙*|으+아*앙*|흐+윽*|아+앙+|옥|푹|쩍|짝|탁|쿵|퍽|철퍽|질척|찔꺽|주륵|츄+릅*|쪽|"
    r"두근|움찔|부르르|덜컥|스윽|슥|휘익|달칵|찰칵|꿀꺽|사각|바스락|짜릿"
    r")+$"
)
_ASTERISK_CENSOR_RE = re.compile(
    r"(?:(?<!\w)[A-Za-zÀ-ỹ]{1,4}\*+[A-Za-zÀ-ỹ]{0,4}(?!\w)|\*{2,})",
    re.UNICODE,
)


def has_asterisk_censor(text: str | None) -> bool:
    """Detect self-censored Vietnamese profanity/NSFW tokens like `C* L*`, `l*n`, `c*c`, `đ*t`, `***`."""
    return bool(_ASTERISK_CENSOR_RE.search((text or "").strip()))


_VN_CENSOR_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bc\*\s*l\*", re.IGNORECASE), "cái lồn"),
    (re.compile(r"\bl\*n\b", re.IGNORECASE), "lồn"),
    (re.compile(r"\bl\*\*\b", re.IGNORECASE), "lồn"),
    (re.compile(r"\bc\*c\b", re.IGNORECASE), "cặc"),
    (re.compile(r"\bc\*\*\b", re.IGNORECASE), "cặc"),
    (re.compile(r"\bđ\*t\b", re.IGNORECASE), "địt"),
    (re.compile(r"\bd\*t\b", re.IGNORECASE), "địt"),
    (re.compile(r"\bđ\*o\b", re.IGNORECASE), "đéo"),
    (re.compile(r"\bv\*i\b", re.IGNORECASE), "vãi"),
    (re.compile(r"\bch\*ch\b", re.IGNORECASE), "chịch"),
    (re.compile(r"\bb\*ồi\b", re.IGNORECASE), "buồi"),
    (re.compile(r"\bđ\*ụ\b|\bđ\*\b", re.IGNORECASE), "đụ"),
]


def _preserve_case_sub(match: re.Match[str], repl: str) -> str:
    src = match.group(0)
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def uncensor_vietnamese_asterisks(text: str) -> str:
    """Restore common self-censored Vietnamese tokens (`C* L*`, `l*n`, `c*c`, `đ*t`) into natural full words."""
    out = text
    for pat, repl in _VN_CENSOR_FIXES:
        out = pat.sub(lambda m, r=repl: _preserve_case_sub(m, r), out)
    return out
_VN_TEXT_KEYS = ("vietnamese", "vi", "viet", "translated", "translated_text", "translation", "text_vi", "text")
_HARD_PLACEHOLDER_TEXTS = {"", "(trống)", "n/a", "none", "null"}
_ELLIPSIS_PLACEHOLDER_TEXTS = {"...", "…", "—", "-"}
_PLACEHOLDER_TEXTS = _HARD_PLACEHOLDER_TEXTS | _ELLIPSIS_PLACEHOLDER_TEXTS
_LATIN_WORD_RE = re.compile(r"[A-Za-z]")


def _source_has_substantive_words(source_text: str | None) -> bool:
    src = (source_text or "").strip()
    if not src:
        return False
    return bool(_CJK_KANA_RE.search(src) or len(_LATIN_WORD_RE.findall(src)) >= 2)


def _parse_bool_flag(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val or "").strip().lower() in {"true", "1", "yes"}


def looks_like_floating_sfx(text: str) -> bool:
    """Return True when non-bubble OCR text is unambiguously a Japanese/Korean sound-effect glyph."""
    cleaned = re.sub(r"[\s\u3000!！?？…・〜~\-ー♡♥\.。、]+", "", (text or "").strip())
    if not cleaned or len(cleaned) > 16:
        return False
    if _PURE_HANGUL_RE.match(cleaned):
        if _KO_SFX_RE.match(cleaned):
            return True
        # Single isolated Hangul syllable (e.g. '옥', '푹', '탁') floating outside bubbles without parentheses
        return len(cleaned) == 1 and "(" not in (text or "") and ")" not in (text or "")
    if not _PURE_KATAKANA_RE.match(cleaned):
        return False
    if _SFX_STEMS_RE.search(cleaned):
        return True
    # Reduplicated Kana onomatopoeia outside bubbles (e.g. フルフル, レロレロ, ぷるぷる, ちゅぱちゅぱ)
    if _REDUPLICATED_KANA_SFX_RE.match(cleaned):
        return True
    # Short pure Katakana sound effect ending with sokuon/prolongation/n (e.g. メキッ, ギシッ, プルン, ンンッ)
    if _STRICT_KATAKANA_RE.match(cleaned) and 2 <= len(cleaned) <= 6 and cleaned.endswith(("ッ", "ー", "ン")):
        return True
    return False


def _sfx_rule_block(skip_sfx: bool) -> str:
    if not skip_sfx:
        return ""
    return (
        "\nQUY TẮC GIỮ NGUYÊN SFX TIẾNG NHẬT/HÀN GỐC (BẮT BUỘC TUYỆT ĐỐI):\n"
        "- Người dùng muốn **GIỮ NGUYÊN 100% các hiệu ứng SFX vẽ trực tiếp trên tranh** (không dịch, không tẩy, không chèn chữ Việt đè lên tranh!).\n"
        "- BẤT KỲ khung [ID] nào là **SFX / từ tượng thanh / tiếng động / tiếng cọ xát / tiếng ướt át / tiếng run rẩy / tiếng thở vẽ tay bên ngoài bóng thoại** "
        "(Nhật: ドン, ドキッ, ビクン, グチュ, ズチュ, バキッ, カチッ, ギシ, ギシッ, フルフル, ふるふる, プルン, ぷるぷる, ビチャ, ひちゃ, レロレロ, じゅぷ, メキ, ぱんっ, ぽよん, ンンッ, ハァハァ, ゼェゼェ...; "
        "Hàn: 하앙, 옥, 푹, 찔꺽, 철퍽, 주륵, 두근, 움찔, 짝; Trung: 咻, 啪...) "
        "hoặc watermark/copyright (`©...`, `http://...`): "
        "BẮT BUỘC đặt `\"is_sfx\": true` và `\"vietnamese\": \"\"` (để TRỐNG hoàn toàn!).\n"
        "- KHÔNG dịch các SFX trên tranh thành 'Hộc...!', 'Lép nhép', 'Rung rẩy', 'Kẽo kẹt', 'Bạch bạch'...\n"
        "- NGƯỢC LẠI, với MỌI lời thoại trong bubble, suy nghĩ trong ngoặc `(...)`, lời dẫn hay câu nói rõ ràng của nhân vật: "
        "đặt `\"is_sfx\": false` và dịch sang Tiếng Việt tự nhiên.\n"
    )


def normalize_source_key(text: str) -> str:
    """Normalize source dialogue for reliable translation-memory reuse across frames."""
    return re.sub(r"[\s\u3000]+", "", (text or "")).strip()


def region_needs_translation(reg: TextRegionItem) -> bool:
    """Return True if a non-SFX region is still empty, placeholder, asterisk-censored, or untranslated CJK/Kana/Hangul."""
    if reg.is_sfx:
        return False
    src = (reg.source_text or "").strip()
    vn = (reg.translated_text or "").strip()
    if not vn or vn.lower() in _HARD_PLACEHOLDER_TEXTS:
        return True
    if vn in _ELLIPSIS_PLACEHOLDER_TEXTS:
        return _source_has_substantive_words(src)
    if _KANA_HANGUL_RE.search(vn):
        return True
    if has_asterisk_censor(vn):
        return True
    if src and _CJK_KANA_RE.search(src):
        if normalize_source_key(vn) == normalize_source_key(src):
            return True
    return False


def _extract_items_list(data: Any) -> list[Any]:
    """Extract region items list from any JSON envelope shape returned by Gemini/DeepSeek/Proxy."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("items", "regions", "translations", "lines", "results", "data"):
        val = data.get(key)
        if isinstance(val, list):
            return val
    pages = data.get("pages")
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        for key in ("items", "regions", "translations"):
            val = pages[0].get(key)
            if isinstance(val, list):
                return val
    return []


def _parse_int_id(val: Any) -> int | None:
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    m = re.search(r"\d+", str(val or ""))
    return int(m.group(0)) if m else None


def _extract_vn_from_item(item: dict[str, Any], source_text: str = "") -> str:
    has_words = _source_has_substantive_words(source_text)
    for key in _VN_TEXT_KEYS:
        val = item.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        if not txt or txt.lower() in _HARD_PLACEHOLDER_TEXTS:
            continue
        if txt in _ELLIPSIS_PLACEHOLDER_TEXTS and has_words:
            continue
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
        preferred_slot: int = 0,
    ) -> None:
        eff_base_url = base_url.rstrip("/") if base_url else None
        if not eff_base_url and is_deepseek_model(model):
            eff_base_url = DEEPSEEK_BASE_URL
        elif not eff_base_url and is_groq_model(model):
            eff_base_url = GROQ_BASE_URL
        use_groq_keys = is_groq_model(model) or is_groq_endpoint(eff_base_url)
        use_deepseek_keys = is_deepseek_model(model) or is_deepseek_endpoint(eff_base_url)

        raw_sources: list[str] = []
        if api_keys:
            raw_sources.extend(api_keys)
        env_sources = (
            (
                api_key,
                os.environ.get("GROQ_API_KEYS"),
                os.environ.get("GROQ_API_KEY"),
                os.environ.get("OPENAI_API_KEY"),
            )
            if use_groq_keys
            else (
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
            if not k or k in seen:
                continue
            if use_groq_keys and (k.startswith("AIza") or k.startswith("sk-")):
                continue
            if use_deepseek_keys and (k.startswith("AIza") or k.startswith("gsk_")):
                continue
            if not eff_base_url and (k.startswith("sk-") or k.startswith("gsk_")):
                continue
            seen.add(k)
            self.api_keys.append(k)

        if not self.api_keys:
            try:
                from koharu_vn_cli.config import load_config

                cfg_keys = load_config().effective_api_keys(model=model, base_url=eff_base_url)
                for k in cfg_keys:
                    if k and k not in seen:
                        seen.add(k)
                        self.api_keys.append(k)
            except Exception:
                pass

        self._preferred_slot = max(0, int(preferred_slot))
        self.model = model
        self.base_url = eff_base_url
        if self.api_keys:
            self._key_idx = get_key_health_tracker().pick_initial_index(
                self.api_keys,
                model=self.model,
                base_url=self.base_url,
                preferred_slot=self._preferred_slot,
            )
            self.api_key = self.api_keys[self._key_idx]
        else:
            self._key_idx = 0
            self.api_key = ""
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
        self.timeout = timeout
        self.on_status = on_status
        if not self.api_key and not self.base_url:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY, DEEPSEEK_API_KEY, or GROQ_API_KEY env var, or pass --api-key (or --base-url for a local proxy)."
            )
        if is_deepseek_endpoint(self.base_url) and not self.api_key:
            raise ValueError(
                "Missing DeepSeek API key. Set DEEPSEEK_API_KEY env var or pass --api-key."
            )
        if is_groq_endpoint(self.base_url) and not self.api_key:
            raise ValueError(
                "Missing Groq API key. Set GROQ_API_KEY env var or pass --api-key."
            )

    def _has_fallback_model(self) -> bool:
        return self._fallback_idx < len(self.fallback_models)

    def _switch_to_fallback_model(self, reason: str) -> bool:
        """Advance to the next configured fallback model when the current model exhausts quota or is blocked."""
        if not self._has_fallback_model():
            return False
        old_model = self.model
        new_model = self.fallback_models[self._fallback_idx]
        self.model = new_model
        self._fallback_idx += 1
        if is_deepseek_model(new_model) and not is_deepseek_endpoint(self.base_url):
            self.base_url = DEEPSEEK_BASE_URL
            try:
                from koharu_vn_cli.config import load_config

                ds_keys = load_config().effective_api_keys(model=new_model, base_url=DEEPSEEK_BASE_URL)
                if ds_keys:
                    self.api_keys = ds_keys
            except Exception:
                pass
        elif is_groq_model(new_model) and not is_groq_endpoint(self.base_url):
            self.base_url = GROQ_BASE_URL
            try:
                from koharu_vn_cli.config import load_config

                groq_keys = load_config().effective_api_keys(model=new_model, base_url=GROQ_BASE_URL)
                if groq_keys:
                    self.api_keys = groq_keys
            except Exception:
                pass
        elif not is_deepseek_model(new_model) and not is_groq_model(new_model) and (
            is_deepseek_endpoint(self.base_url) or is_groq_endpoint(self.base_url)
        ):
            self.base_url = None
            try:
                from koharu_vn_cli.config import load_config

                gem_keys = load_config().effective_api_keys(model=new_model, base_url="")
                if gem_keys:
                    self.api_keys = gem_keys
            except Exception:
                pass
        if self.api_keys:
            self._key_idx = get_key_health_tracker().pick_initial_index(
                self.api_keys,
                model=self.model,
                base_url=self.base_url,
                preferred_slot=self._preferred_slot,
            )
            self.api_key = self.api_keys[self._key_idx]
        else:
            self._key_idx = 0
            self.api_key = ""
        self._emit_status(
            f"⚠ Model '{old_model}' {reason} -> tự động chuyển sang model fallback '{self.model}'!"
        )
        return True

    def _rotate_key(self, reason: str, *, exclude_indices: set[int] | None = None) -> bool:
        """Advance to the smartest next healthy/unused API key when multiple keys are configured."""
        if len(self.api_keys) <= 1:
            return False
        tracker = get_key_health_tracker()
        excluded = set(exclude_indices) if exclude_indices is not None else {self._key_idx}
        next_idx, cooled_down_cnt = tracker.acquire_key(
            self.api_keys,
            model=self.model,
            base_url=self.base_url,
            exclude_indices=excluded,
            preferred_slot=self._preferred_slot,
            skip_cooldown_if_tried=True,
        )
        if next_idx is None:
            return False
        self._key_idx = next_idx
        self.api_key = self.api_keys[self._key_idx]
        skip_note = f", né {cooled_down_cnt} key lỗi" if cooled_down_cnt > 0 else ""
        self._emit_status(
            f"Đổi sang API key #{self._key_idx + 1}/{len(self.api_keys)} ({reason}{skip_note})..."
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
        """Send multimodal prompt + base64 JPEGs and parse JSON response with smart key health & quota-fallback failover."""
        img_kb = max(1, sum(len(b64) for _, b64 in images) * 3 // 4 // 1024) if images else 0
        tracker = get_key_health_tracker()
        last_err: Exception | None = None
        last_is_quota = False
        tried_indices: set[int] = set()
        need_acquire = True
        active_key = ""
        cooled_down_cnt = 0
        transient_attempt = 0
        max_transient = 3

        while True:
            keys_total = max(1, len(self.api_keys))
            if self.api_keys and need_acquire:
                picked_idx, cooled_down_cnt = tracker.acquire_key(
                    self.api_keys,
                    model=self.model,
                    base_url=self.base_url,
                    exclude_indices=tried_indices,
                    preferred_slot=self._preferred_slot,
                    skip_cooldown_if_tried=False,
                )
                if picked_idx is None:
                    tried_indices.clear()
                    picked_idx, cooled_down_cnt = tracker.acquire_key(
                        self.api_keys,
                        model=self.model,
                        base_url=self.base_url,
                        exclude_indices=tried_indices,
                        preferred_slot=self._preferred_slot,
                        skip_cooldown_if_tried=False,
                    )
                if picked_idx is not None:
                    self._key_idx = picked_idx
                    self.api_key = self.api_keys[picked_idx]
                    active_key = self.api_key
                    tried_indices.add(picked_idx)
                need_acquire = False

            try:
                if transient_attempt == 0 and len(tried_indices) <= 1:
                    if keys_total > 1:
                        skip_tag = f" · né {cooled_down_cnt} key lỗi" if cooled_down_cnt > 0 else ""
                        key_tag = f" [key {self._key_idx + 1}/{keys_total}{skip_tag}]"
                    else:
                        key_tag = ""
                    if images:
                        self._emit_status(
                            f"Đang gửi {len(images)} ảnh (~{img_kb} KB) tới {self.model}{key_tag} (timeout {int(self.timeout)}s)..."
                        )
                    else:
                        self._emit_status(
                            f"Đang gửi text-only (OCR + Story Bible) tới {self.model}{key_tag} (timeout {int(self.timeout)}s)..."
                        )
                try:
                    if self.base_url:
                        raw = self._call_openai_compatible(system_prompt, user_prompt, images, temperature)
                    else:
                        raw = self._call_gemini_rest(system_prompt, user_prompt, images, temperature)
                finally:
                    if active_key:
                        tracker.release_key(active_key, model=self.model, base_url=self.base_url)
                parsed = _extract_json_block(raw)
                if active_key:
                    tracker.record_success(active_key, model=self.model, base_url=self.base_url)
                return parsed
            except ContentBlockedError:
                raise
            except httpx.HTTPStatusError as exc:
                last_err = exc
                code = exc.response.status_code
                body_preview = exc.response.text.strip().replace("\n", " ")[:200]
                is_quota = _is_quota_error(code, body_preview)
                is_auth = _is_auth_error(code, body_preview)
                last_is_quota = is_quota
                if code == 404:
                    if self._fallback_idx > 0 and self._switch_to_fallback_model("không tồn tại (HTTP 404)"):
                        tried_indices.clear()
                        transient_attempt = 0
                        last_is_quota = False
                        need_acquire = True
                        active_key = ""
                        continue
                    hint = f" (Model '{self.model}' không tồn tại hoặc key chưa mở — bấm F3 hoặc gõ /model để chọn model hỗ trợ)"
                    raise RuntimeError(f"HTTP {code} từ {self.model}: {body_preview}{hint}") from exc
                if code in (400, 401, 403, 429) or code >= 500 or is_quota:
                    if active_key:
                        err_kind = "quota" if is_quota else ("auth" if is_auth else "transient")
                        tracker.record_error(
                            active_key, model=self.model, base_url=self.base_url, kind=err_kind
                        )
                    if len(self.api_keys) > 1 and self._rotate_key(f"HTTP {code}", exclude_indices=tried_indices):
                        active_key = self.api_key
                        tried_indices.add(self._key_idx)
                        need_acquire = False
                        continue
                    if code in (400, 401, 403) and not is_quota:
                        hint = " (API key không hợp lệ hoặc hết quyền — kiểm tra /key)" if code in (401, 403) else ""
                        raise RuntimeError(f"HTTP {code} từ {self.model}: {body_preview}{hint}") from exc
                if is_quota and self._has_fallback_model():
                    if transient_attempt < 1:
                        wait_s = 1.5
                        transient_attempt += 1
                        tried_indices.clear()
                        need_acquire = True
                        active_key = ""
                        self._emit_status(f"API trả HTTP {code} (quota), thử lại nhanh {transient_attempt}/1 sau {wait_s:.1f}s...")
                        time.sleep(wait_s)
                        continue
                    if self._switch_to_fallback_model(f"hết quota (HTTP {code})"):
                        tried_indices.clear()
                        transient_attempt = 0
                        last_is_quota = False
                        need_acquire = True
                        active_key = ""
                        continue
                if transient_attempt < max_transient:
                    wait_s = 1.5 * (2**transient_attempt)
                    transient_attempt += 1
                    tried_indices.clear()
                    need_acquire = True
                    active_key = ""
                    self._emit_status(f"API trả HTTP {code}, thử lại lần {transient_attempt}/{max_transient} sau {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                break
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_err = exc
                last_is_quota = False
                if isinstance(exc, httpx.HTTPError):
                    if active_key:
                        tracker.record_error(
                            active_key, model=self.model, base_url=self.base_url, kind="transient"
                        )
                    if len(self.api_keys) > 1 and self._rotate_key(
                        type(exc).__name__, exclude_indices=tried_indices
                    ):
                        active_key = self.api_key
                        tried_indices.add(self._key_idx)
                        need_acquire = False
                        continue
                if transient_attempt < max_transient:
                    wait_s = 1.5 * (2**transient_attempt)
                    transient_attempt += 1
                    tried_indices.clear()
                    need_acquire = True
                    active_key = ""
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
            if block_reason or images:
                raise ContentBlockedError(
                    f"Gemini API chặn nội dung ({block_reason or 'NO_CANDIDATES'}, model={self.model}). "
                    "Đây là bộ lọc cứng phía máy chủ Google đối với ảnh nhạy cảm/R18; "
                    "hãy dùng /key với endpoint OpenAI-compatible (--base-url) hoặc DeepSeek."
                )
            raise ValueError(f"Gemini returned no candidates: {data}")
        finish_reason = str(candidates[0].get("finishReason", ""))
        if finish_reason in ("PROHIBITED_CONTENT", "SAFETY", "IMAGE_SAFETY", "BLOCKLIST", "SPII"):
            raise ContentBlockedError(
                f"Gemini API chặn phản hồi ({finish_reason}, model={self.model}). "
                "Hãy dùng /key với endpoint --base-url hoặc DeepSeek nếu xử lý nội dung R18."
            )
        parts_out = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts_out if not p.get("thought") and p.get("text")]
        if not text_parts:
            text_parts = [p.get("text", "") for p in parts_out if p.get("text")]
        joined = "\n".join(text_parts)
        if not joined.strip():
            if finish_reason in ("OTHER", "RECITATION") and images:
                raise ContentBlockedError(
                    f"Gemini API trả rỗng do bộ lọc ({finish_reason}, model={self.model})."
                )
            raise ValueError(f"Gemini returned empty text (finishReason={finish_reason or 'unknown'})")
        if _looks_like_soft_refusal(joined):
            raise ContentBlockedError(f"Model từ chối mềm nội dung nhạy cảm (model={self.model}).")
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
        if images and is_groq_endpoint(self.base_url) and not is_groq_vision_model(self.model):
            raise ValueError(
                f"Model Groq '{self.model}' chỉ hỗ trợ text (không nhận ảnh trực tiếp). "
                "Hãy dùng model Vision (như meta-llama/llama-4-scout-17b-16e-instruct) hoặc bật OCR cục bộ (--no-vision)."
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
        low_model = self.model.strip().lower()
        if low_model != "deepseek-reasoner":
            payload["temperature"] = temperature
        omit_json_format = (
            low_model == "deepseek-reasoner"
            or (
                is_groq_endpoint(self.base_url)
                and (bool(images) or "qwen3" in low_model or "gpt-oss" in low_model)
            )
        )
        if json_mode and not omit_json_format:
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
            groq_excluded = (
                "whisper",
                "tts",
                "orpheus",
                "prompt-guard",
                "llama-guard",
                "safeguard",
                "embed",
                "allam",
            )
            is_groq = is_groq_endpoint(self.base_url)
            for m in raw_items:
                mid = str(m.get("id", "")).strip()
                if not mid:
                    continue
                low_mid = mid.lower()
                if is_groq and any(tok in low_mid for tok in groq_excluded):
                    continue
                owner = str(m.get("owned_by", "")).strip()
                if is_groq:
                    kind_note = "Groq Vision (hỗ trợ ảnh)" if is_groq_vision_model(mid) else "Groq siêu nhanh (OCR cục bộ)"
                    label = f"{kind_note} · {owner}" if owner else kind_note
                else:
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
        "- TUYỆT ĐỐI CẤM TỰ KIỂM DUYỆT BẰNG DẤU SAO `*` HOẶC VIẾT TẮT CHE CHỮ (CẤM TUYỆT ĐỐI kiểu `C* L*`, `l*n`, `c*c`, `đ*t`, `v*i`, `c**`, `***`, `c.ặ.c`): "
        "phải viết ĐẦY ĐỦ 100% từ Tiếng Việt tự nhiên! Nếu cần tránh từ quá thô ở cảnh nhẹ, hãy chọn từ lóng/khẩu ngữ tiếng Việt viết đầy đủ và gợi cảm (vd: `生オナホ / 肉オナホ` -> 'búp bê sống', 'món đồ chơi sống', 'cái lồn sống'; `ちんぽ` -> 'cây hàng', 'con cặc'; `まんこ` -> 'chỗ đó', 'cái lồn'), KHÔNG BAO GIỜ chèn dấu `*` vào giữa từ!\n"
        "- TUYỆT ĐỐI TRÁNH từ y khoa khô khan ('âm đạo', 'dương vật', 'giao hợp') và tiếng rên kiểu Anh ('UGH', 'GUH', 'BUH', 'NNGH', 'MMN', 'FUH'): đổi sang khẩu ngữ Việt tự nhiên theo sắc thái cảnh ('bên trong', 'chỗ đó', 'nóng hổi', 'ướt nhẹp', 'sâu quá', 'ra mất') và tiếng thở Việt ('Ư...', 'Ưm...', 'Á...', 'Aaa...', 'Hộc...', 'Khựa...').\n"
        "- CẤM dùng từ cứng, sáo rỗng ít ai nói ngoài đời như: 'hắn ta', 'ả', 'gã', 'cô nàng', 'thực sự là', 'vô cùng', 'cái đó', 'vậy sao', 'thèm thuồng lắm chứ', 'kẻ hầu', 'ấn ký' (trừ khi đúng bối cảnh kỳ ảo cổ trang: mới dùng 'ta - ngươi' hoặc danh xưng cổ trang).\n"
        "- Câu ngắn gọn, súc tích VỪA KHÍT KHUNG THOẠI (tránh dịch rườm rà dài dòng làm chữ bị bóp nhỏ hoặc tràn khung), giữ nhịp thở ngắt quãng (`...`, `!?`) và tiểu từ tự nhiên (à, ừ, nhỉ, đấy, chứ, đâu, cơ mà, mất rồi, luôn ấy, nha...).\n"
        "- KIỂM TRA CHÍNH TẢ & TÊN RIÊNG: TUYỆT ĐỐI không sai chính tả Tiếng Việt (vd: 'tao' không viết nhầm 'tạo', 'dừng lại' không viết 'đừng lại', 'cũng' không viết 'cứng'); giữ đúng chính tả chuẩn tiếng Anh/Romaji của nhân vật game/anime (vd: Blue Archive -> Iochi Mari, Utazumi Sakurako, Shimoe Koharu, tuyệt đối không bịa thêm âm tiết thừa).\n"
        "- TUYỆT ĐỐI KHÔNG để dấu ngoặc Nhật `「」『』【】` trong chuỗi `vietnamese` (hãy dùng ngoặc kép `\"\"` hoặc bỏ ngoặc vì font Việt không có glyph đó sẽ biến thành ô vuông lỗi).\n"
        "- Đúng ngữ khí nhân vật & văn phong truyện: không nhồi khẩu ngữ hài hước/tấu hài ('ối giồi ôi') vào cảnh đe dọa, uy hiếp hoặc nhân vật lạnh lùng.\n"
        "- Không giải thích dài dòng, không thêm chú thích ngoài lời thoại.\n"
        "- Tiếng thở/rên thoại của nhân vật (khi là lời nói/suy nghĩ, không phải SFX nền) dịch tự nhiên đúng chất doujinshi (vd: 'Ư...', 'Ưm...', 'Á...', 'Hộc...', 'Hả?!').\n"
        "- CHỐNG LẶP THOẠI / TRÙNG KHUNG: Nếu 2 khung [ID] bị detect chồng lên nhau hoặc tách đôi từ CÙNG 1 đoạn chữ gốc giống hệt nhau, CHỈ điền bản dịch đầy đủ vào 1 ID bao phủ tốt nhất, còn ID bị trùng lặp đặt `\"is_sfx\": true` và `\"vietnamese\": \"\"` — TUYỆT ĐỐI KHÔNG lặp lại nguyên văn cùng 1 câu thoại sang 2 khung sát nhau.\n"
        "Bắt buộc trả về đúng JSON hợp lệ."
    )


    _HUMAN_CRAFT_RULES = (
        "\nTƯ DUY DỊCH & SÁNG TÁC NHƯ CON NGƯỜI (HUMAN.md MANIFESTO):\n"
        "- 'Đúng nghĩa từ điển' chỉ là mức sàn thấp nhất — MỤC TIÊU LÀ DƯ VỊ CẢM XÚC (Emotional Resonance): căng thẳng, bất ngờ, ngập ngừng đúng nhịp, một câu kết hụt đi nửa nhịp khiến người đọc nhớ mãi.\n"
        "- PHÁ LUẬT CÓ CHỦ ĐÍCH (Break the rule on purpose): AI thường cho ra câu trung bình, tròn trĩnh, đều đều. Hãy dám cắt câu cụt, bỏ chủ ngữ, giữ khoảng lặng (`...`, `—`), đảo ngữ khi nhân vật đang sốc, nứng hoặc mất bình tĩnh.\n"
        "- CHỌN TỪ ĐẮT & CỤ THỂ (Say the more specific thing): Cấm dùng cách diễn đạt chung chung, an toàn ('rất thích', 'vô cùng', 'cảm thấy'). Chọn đúng từ sống động, gai góc hoặc ướt át mà đúng tính cách nhân vật đó sẽ buột miệng nói ra.\n"
        "- ĐỨNG VỀ PHÍA NGƯỜI ĐỌC ('What does it feel like?'): Viết ra bản thoại mà chính bạn đọc cũng phải bật cười, rùng mình, hoặc rạo rực — tuyệt đối không viết câu AI sáo rỗng 'trả bài'.\n"
    )

    _DEEPSEEK_R18_LOCALIZATION_ADDON = (
        "\nPHONG CÁCH DOUJINSHI / R18 TỰ NHIÊN & BUNG LỤA (DEEPSEEK UNCENSORED):\n"
        "- KHÔNG tự kiểm duyệt, KHÔNG làm nhạt, KHÔNG né tránh bằng từ hiền/sáo rỗng khi câu gốc gắt, dâm hoặc đang vào cảnh 18+.\n"
        "- ĐO LƯỜNG CƯỜNG ĐỘ CHỬI THỀ & TỤC TĨU ĐÚNG ĐỘ GẮT (tránh dịch yếu như 'đồ khốn', 'im mồm', 'chết tiệt' khi câu gốc cục súc/mất kiểm soát):\n"
        "  + Chửi/gắt/áp đảo: `くそ / ちくしょう` -> 'địt mẹ', 'mẹ kiếp', 'đm', 'vãi lồn'; `黙れ / うるせえ` -> 'câm mồm', 'ngậm mẹ mồm vào', 'câm đi' (KHÔNG dịch 'im mồm' yếu ớt); `てめえ / お前` (thô) -> 'mày', 'thằng chó', 'con đĩ'; `ふざけんな` -> 'đùa bố mày à', 'đéo đùa đâu'.\n"
        "- DỊCH ĐÚNG NGỮ CẢNH 18+ (KHÔNG dịch word-by-word):\n"
        "  + `だめ / やめて` khi nhân vật đang đỏ mặt/run rẩy/sướng -> dịch thành khoái cảm mất kiểm soát ('đừng mà... sâu quá...', 'đéo chịu nổi mất...', 'nữa đi... đừng dừng...♡'), KHÔNG dịch cứng kiểu 'Dừng lại!'.\n"
        "  + Chia 2 mức thanh điệu theo tính cách/tag: (1) Ngọt/e thẹn/vanilla dùng ngôn từ ướt át mềm mại ('chỗ đó nhạy cảm quá...', 'bên trong nóng hổi', 'ướt nhẹp hết rồi', 'khít chặt quá...♡'); (2) Dâm/dirty-talk/hardcore/NTR/áp đảo dùng thẳng từ gợi dục, trần trụi, gây nứng ('nứng', 'cặc', 'lồn', 'buồi', 'bú', 'liếm', 'đụ', 'chịch', 'nện', 'bắn tinh vào trong', 'nước lồn chảy ướt nhẹp', 'sướng vãi lồn', 'đéo nhịn nổi nữa').\n"
        "- NHỊP THỞ & CẢM GIÁC ĐA GIÁC QUAN (R18 PACING):\n"
        "  + Dạo đầu/ngập ngừng: câu lửng lơ, tiểu từ mềm ('à, nha, mất...'). Cao trào: câu CỰC NGẮN, đứt quãng theo từng nhịp hông/hơi thở (`Ư...— sâu... quá...!`, `Chậm... chút... đéo chịu nổi...♡`), KHÔNG viết câu dài đủ chủ-vị ngữ khi nhân vật đang mất trí.\n"
        "  + Nhấn vào nhiệt độ, độ ướt, ma sát, độ khít, sự run rẩy ('nóng ran', 'lép nhép', 'mút chặt lấy', 'nhũn hết chân', 'đầu óc trắng xóa').\n"
        "- Giữ cân bằng: đoạn mở đầu/đời thường chưa vào cảnh thì giữ khẩu ngữ tự nhiên đúng nhân vật, KHÔNG nhồi tục vô cớ; vào cảnh gắt hoặc 18+ thì bung hết cỡ đúng chất doujinshi Việt.\n"
    )

    def _localization_rules(self) -> str:
        base = self._VI_LOCALIZATION_RULES + self._HUMAN_CRAFT_RULES
        if (
            is_deepseek_model(self.model)
            or is_deepseek_endpoint(self.base_url)
            or is_groq_model(self.model)
            or is_groq_endpoint(self.base_url)
            or self.base_url
        ):
            return base + self._DEEPSEEK_R18_LOCALIZATION_ADDON
        return base

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
            '  "page_beats": {"1": "[Bối cảnh + Tư thế/Ánh mắt (nhìn thẳng POV hay bạn diễn) + Biểu cảm + Tâm lý chuyển tiếp trang kế]"}\n'
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
        raw_items = _extract_items_list(data)
        summary = str(data.get("page_summary", "")).strip() if isinstance(data, dict) else ""
        return (
            self._merge_translated_regions(regions, raw_items, skip_sfx=skip_sfx),
            summary,
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
            fallback_src = ((ocr_texts or {}).get(copy.id, "") or copy.source_text or "").strip()
            if match:
                raw_src = str(match.get("source_text", "") or "").strip()
                if raw_src and raw_src.lower() not in _PLACEHOLDER_TEXTS:
                    copy.source_text = raw_src
                elif fallback_src:
                    copy.source_text = fallback_src

                vn = _extract_vn_from_item(match, source_text=copy.source_text)
                if vn and has_asterisk_censor(vn):
                    vn = uncensor_vietnamese_asterisks(vn)
                raw_is_sfx = _parse_bool_flag(match.get("is_sfx"))
                norm_src = normalize_source_key(copy.source_text)
                is_watermark = bool(re.search(r"(?:©|https?://|\.(?:com|net|org|jp)/?)", norm_src, re.IGNORECASE))
                is_pure_floating_sfx = looks_like_floating_sfx(copy.source_text) or is_watermark
                bubble_has_spoken = bool(re.search(r"[\u3040-\u309f\uac00-\ud7af]", norm_src))

                # Protect spoken dialogue & valid translations from accidental SFX mis-tagging:
                # If the LLM produced a valid Vietnamese translation `vn` inside a speech bubble
                # (or non-SFX text), NEVER discard `vn` just because `is_sfx` was set!
                if copy.kind == "bubble":
                    if raw_is_sfx and is_pure_floating_sfx and not vn and not bubble_has_spoken:
                        copy.is_sfx = True
                    else:
                        copy.is_sfx = False
                else:
                    # For non-bubble floating text, preserve SFX detection whenever either the LLM
                    # flagged `is_sfx: true` OR local glyph analysis confirms pure SFX, so SFX is
                    # never accidentally inpainted or overwritten with translated text.
                    if is_pure_floating_sfx or raw_is_sfx:
                        copy.is_sfx = True
                    else:
                        copy.is_sfx = False

                if skip_sfx and copy.is_sfx:
                    copy.translated_text = ""
                elif vn:
                    from koharu_vn_cli.pipeline.typesetter import sanitize_render_text

                    copy.translated_text = sanitize_render_text(vn)
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
        ocr_temp = (
            0.45
            if (
                is_deepseek_model(self.model)
                or is_deepseek_endpoint(self.base_url)
                or is_groq_model(self.model)
                or is_groq_endpoint(self.base_url)
            )
            else 0.25
        )
        data = self._call_json(system_prompt, user_prompt, images=[], temperature=ocr_temp)
        raw_items = _extract_items_list(data)
        merged_sub = self._merge_translated_regions(to_translate, raw_items, ocr_texts=ocr_texts, skip_sfx=skip_sfx)
        by_sub_id = {r.id: r for r in merged_sub}
        updated = [by_sub_id.get(r.id, r) for r in working]
        summary = str(data.get("page_summary", "")).strip() if isinstance(data, dict) else ""

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
        ocr_temp = (
            0.45
            if (
                is_deepseek_model(self.model)
                or is_deepseek_endpoint(self.base_url)
                or is_groq_model(self.model)
                or is_groq_endpoint(self.base_url)
            )
            else 0.25
        )
        try:
            data = self._call_json(system_prompt, user_prompt, images=[], temperature=ocr_temp)
        except ContentBlockedError:
            data = {}

        raw_pages_list = [
            p_entry
            for p_entry in (data.get("pages", []) if isinstance(data, dict) else [])
            if isinstance(p_entry, dict)
        ]
        by_page: dict[int, dict[str, Any]] = {}
        for p_entry in raw_pages_list:
            p_raw = p_entry.get("page") or p_entry.get("page_number") or p_entry.get("page_index") or p_entry.get("id")
            parsed_p = _parse_int_id(p_raw)
            if parsed_p is not None:
                by_page[parsed_p] = p_entry

        mem = dict(translation_memory or {})
        out: dict[int, tuple[list[TextRegionItem], str]] = {}
        rolling_summaries = list(previous_summaries or [])
        for page_pos, (pno, regs, ocrs) in enumerate(scene_pages):
            entry = by_page.get(pno)
            if entry is None and len(raw_pages_list) == len(scene_pages):
                entry = raw_pages_list[page_pos]
            raw_page_items = _extract_items_list(entry) if isinstance(entry, dict) else None
            if entry and raw_page_items:
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
        with visual scene grounding, strict density adherence (`low` 1-2, `medium` 3-5, `high` 5-8),
        and spatial placement avoiding faces/focal areas.
        """
        h, w = rgb.shape[:2]
        img_b64 = _pil_to_jpeg_b64(Image.fromarray(rgb), max_side=1400, quality=90)
        eff_density = (density or "medium").strip().lower()
        if eff_density not in {"low", "medium", "high"}:
            eff_density = "medium"

        min_n, max_n = _compose_density_bounds(eff_density)
        density_guide = _compose_density_rule(eff_density)
        style_rule = _compose_style_rule(preset)
        schema_examples = _compose_items_example_json(eff_density, preset)

        bible_block = ""
        if story_bible:
            bible_txt = story_bible.to_prompt_block(page_index=page_index)
            if bible_txt:
                bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT TOÀN BỘ CG ===\n{bible_txt}\n==========================================\n"

        context_block = ""
        if previous_summaries:
            joined = "\n".join(f"- Ảnh trước {i + 1}: {s}" for i, s in enumerate(previous_summaries[-6:]))
            context_block = (
                "\nMạch truyện & lời thoại ngay trước đó (hãy nối tiếp tự nhiên, phản hồi trực tiếp diễn biến vừa xảy ra):\n"
                f"{joined}\n"
            )

        tags_block = _format_tags_block(tags)
        prompt_block = f"\nChỉ dẫn kịch bản / phong cách từ người dùng: {story_prompt}\n" if story_prompt else ""

        system_prompt = (
            "Bạn là biên kịch & đạo diễn dàn trang truyện tranh/CG chuyên nghiệp.\n"
            "Nhiệm vụ: Trước tiên hãy quan sát kỹ ĐÚNG NHỮNG GÌ CÓ TRONG TRANH (góc nhìn POV nhìn thẳng người xem hay góc thứ 3 có nhiều người, "
            "ánh mắt, biểu cảm miệng-má, bàn tay đang chạm vào đâu, trang phục, đạo cụ, bối cảnh), rồi sáng tác lời thoại Tiếng Việt theo KHẨU NGỮ ĐỜI THƯỜNG thật tự nhiên, "
            "bám sát đúng tình huống đó và đặt hộp thoại vào các khoảng trống hợp lý.\n"
            + self._localization_rules()
            + "\nNGUYÊN TẮC HIỂU NGỮ CẢNH & SÁNG TÁC THOẠI CG ĐỈNH CAO:\n"
            "- XÁC ĐỊNH RÕ GÓC NHÌN & AI ĐANG NÓI: Nếu là góc nhìn POV nhân vật nhìn thẳng người xem -> đối đáp trực tiếp với bạn đọc ('anh', 'thầy', 'cậu'...). Nếu có >= 2 nhân vật trong khung -> viết trao đổi qua lại giữa các nhân vật, đặt bóng thoại về phía người đang nói.\n"
            "- BÁM SÁT TỪNG CHI TIẾT THỊ GIÁC: Biểu cảm ngại ngùng/che mặt -> ngập ngừng thẹn thùng; cười nhếch mép/áp sát -> trêu chọc, chủ động; nhắm mắt thở dốc/đỏ bừng -> câu ngắn đứt quãng đúng nhịp cảm xúc; KHÔNG bịa chi tiết ngoài tranh.\n"
            "- Phân vai rõ ràng giữa các khung:\n"
            "  + `white_oval` / `speech_tail`: lời nói bật thành tiếng (3–12 từ/bóng, gọn sắc).\n"
            "  + `dark_translucent`: nội tâm / suy nghĩ thầm / lẩm bẩm run rẩy (chữ trắng gọn, sâu).\n"
            "  + `floating` (`none`): tiếng thở khẽ / thì thầm sát miệng / âm thanh cảm xúc ngắn (1–5 từ).\n"
            "- ĐA DẠNG CÂU TỪ & LEO THANG CẢM XÚC: mỗi trang phải đẩy tiến triển mới về tâm lý hoặc hành động, TUYỆT ĐỐI KHÔNG lặp lại câu mở đầu y hệt trang trước.\n"
            + f"\nRÀNG BUỘC SỐ LƯỢNG BẮT BUỘC: {density_guide}\n"
            + "QUY TẮC ĐẶT TỌA ĐỘ THẨM MỸ:\n"
            "- Hệ tọa độ chuẩn hóa [ymin, xmin, ymax, xmax] từ 0 đến 1000 (0=trên/trái, 1000=dưới/phải).\n"
            "- TUYỆT ĐỐI KHÔNG đặt hộp thoại che khuôn mặt, mắt, miệng, tóc mái chính hoặc điểm nhấn hành động/cơ thể.\n"
            "- Ưu tiên đặt bóng thoại ở vùng nền trống, góc trên/rìa ảnh hoặc khoảng trống rải đều quanh nhân vật (KHÔNG dồn hết vào 1 góc).\n"
            "- Kích thước `box_2d` cho bubble (`white_oval`, `dark_translucent`, `speech_tail`) PHẢI TO RÕ RÀNG để dễ đọc trên điện thoại: "
            "chiều rộng khoảng 130..250/1000 và chiều cao khoảng 130..260/1000.\n"
            "- Các hộp thoại tuyệt đối không được chồng chéo lên nhau và phải cách mép ảnh tối thiểu 20/1000.\n"
            "Bắt buộc trả về đúng JSON hợp lệ."
        )

        page_label = f" (Ảnh {page_index})" if page_index else ""
        user_prompt = (
            f"Kích thước ảnh gốc{page_label}: {w}x{h}.{bible_block}{context_block}{tags_block}{prompt_block}\n"
            f"MẬT ĐỘ THOẠI BẮT BUỘC ({eff_density.upper()} — ĐÚNG {min_n}..{max_n} MỤC): {density_guide}\n"
            f"Quy tắc loại thoại:\n{style_rule}\n\n"
            "Hãy điền `visual_scene` mô tả chính xác ảnh trước khi viết `items`, trả về JSON đúng cấu trúc:\n"
            "{\n"
            '  "visual_scene": "Mô tả nhanh góc nhìn (POV hay 2+ người), biểu cảm mắt/miệng, hành động tay & các khoảng nền trống quanh nhân vật",\n'
            '  "page_summary": "Tóm tắt 1-2 câu tình huống & xưng hô để nối tiếp cho ảnh sau",\n'
            f'  "items": [\n{schema_examples}\n  ]\n'
            "}"
        )

        images_payload: list[tuple[str, str]] = [("Bức tranh CG cần sáng tác thoại:", img_b64)]
        if prev_rgb is not None:
            images_payload.append(
                ("(Tham chiếu cảnh CG liền trước để nối tiếp ánh mắt, hành động & mạch thoại):", _pil_to_jpeg_b64(Image.fromarray(prev_rgb), max_side=720, quality=76))
            )
        if next_rgb is not None and not story_bible:
            images_payload.append(
                ("(Tham chiếu cảnh CG sau để dẫn dắt hợp lý):", _pil_to_jpeg_b64(Image.fromarray(next_rgb), max_side=720, quality=76))
            )

        data = self._call_json(
            system_prompt,
            user_prompt,
            images=images_payload,
            temperature=0.65,
        )

        items = _parse_composed_items(
            data.get("items", []), w=w, h=h, preset=preset, density=eff_density
        )
        # If user requested HIGH density (>=5) and model stopped short at 1-4 items, do 1 targeted supplement pass
        if eff_density == "high" and 0 < len(items) < min_n:
            items = self._top_up_high_density_items(
                img_b64=img_b64,
                w=w,
                h=h,
                preset=preset,
                existing_items=items,
                target_min=min_n,
                target_max=max_n,
                visual_scene=str(data.get("visual_scene", "")).strip(),
                page_summary=str(data.get("page_summary", "")).strip(),
                bible_block=bible_block,
                tags_block=tags_block,
                prompt_block=prompt_block,
            )
        return items, str(data.get("page_summary", "")).strip()

    def _top_up_high_density_items(
        self,
        *,
        img_b64: str,
        w: int,
        h: int,
        preset: ComposePreset,
        existing_items: list[TextRegionItem],
        target_min: int,
        target_max: int,
        visual_scene: str,
        page_summary: str,
        bible_block: str,
        tags_block: str,
        prompt_block: str,
    ) -> list[TextRegionItem]:
        """Supplement additional non-overlapping bubbles/whispers/thoughts when `high` density returned too few items."""
        need_more = max(1, target_min - len(existing_items))
        occupied = [
            {
                "id": it.id,
                "kind": it.kind,
                "box_2d": [
                    round(it.text_bbox[1] * 1000 / max(1, h)),
                    round(it.text_bbox[0] * 1000 / max(1, w)),
                    round(it.text_bbox[3] * 1000 / max(1, h)),
                    round(it.text_bbox[2] * 1000 / max(1, w)),
                ],
                "text": it.translated_text,
            }
            for it in existing_items
        ]
        next_id = max((it.id for it in existing_items), default=0) + 1
        sys_p = (
            "Bạn là đạo diễn dàn trang CG. Hiện đang bật chế độ MẬT ĐỘ CAO (HIGH: cần 5–8 khung thoại/ảnh) "
            "nhưng trang này mới có ít khung.\n"
            + self._localization_rules()
            + "Hãy sáng tác BỔ SUNG thêm các bóng thoại phản hồi, suy nghĩ thầm (`dark_translucent`) hoặc tiếng thở/thì thầm (`floating`) "
            "vào CÁC KHOẢNG TRỐNG CHƯA DÙNG (tuyệt đối không đè lên tọa độ đã có và không che mặt nhân vật)."
        )
        usr_p = (
            f"Ngữ cảnh ảnh: {visual_scene or page_summary}.{bible_block}{tags_block}{prompt_block}\n"
            f"Đã có {len(existing_items)} khung: {json.dumps(occupied, ensure_ascii=False)}\n"
            f"Hãy tạo THÊM từ {need_more} đến {max(need_more, target_max - len(existing_items))} khung mới (id bắt đầu từ {next_id}), "
            f"bảo đảm tổng cộng đạt {target_min}..{target_max} khung và không trùng lặp nội dung.\n"
            "Trả về JSON: {\"items\": [...]}"
        )
        try:
            extra_data = self._call_json(
                sys_p,
                usr_p,
                images=[("Ảnh CG cần bổ sung thêm thoại cho đủ mật độ HIGH:", img_b64)],
                temperature=0.7,
            )
            extra_raw = extra_data.get("items", []) if isinstance(extra_data, dict) else []
            extra_items = _parse_composed_items(
                extra_raw, w=w, h=h, preset=preset, density="medium"
            )
        except Exception:
            return existing_items

        seen = {normalize_source_key(it.translated_text).lower() for it in existing_items}
        merged = list(existing_items)
        for it in extra_items:
            norm = normalize_source_key(it.translated_text).lower()
            if norm and norm in seen:
                continue
            if norm:
                seen.add(norm)
            merged.append(it.model_copy(update={"id": len(merged) + 1}))
            if len(merged) >= target_max:
                break
        return resolve_box_overlaps(merged, w, h)

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
        - Adapts batch size by density (`high`: 6 crisp pages per sheet so every page reliably gets 5-8 items;
          `low`/`medium`: up to 18 in 1 sheet or 12-page batches).
        - If `high` density still receives <5 items for any sheet cell, automatically tops it up per page.
        """
        total = len(image_paths)
        if total == 0:
            return {}
        eff_density = (density or "medium").strip().lower()
        if eff_density not in {"low", "medium", "high"}:
            eff_density = "medium"
        min_n, _ = _compose_density_bounds(eff_density)

        batch_limit = 6 if eff_density == "high" else 12
        single_cap = 6 if eff_density == "high" else 18

        results: dict[int, tuple[list[TextRegionItem], str]] = {}
        if total > single_cap:
            running_summaries: list[str] = []
            for start in range(0, total, batch_limit):
                end = min(total, start + batch_limit)
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
                    density=eff_density,
                    story_prompt=story_prompt,
                    story_bible=story_bible,
                    previous_summaries=running_summaries,
                    tags=tags,
                )
                for p_no, val in sorted(batch_res.items()):
                    results[p_no] = val
                    if val[1]:
                        lines_snip = " / ".join(it.translated_text for it in val[0][:2] if it.translated_text)
                        running_summaries.append(f"{val[1]} ({lines_snip})" if lines_snip else val[1])
        else:
            results = self._compose_cg_sheet_batch(
                image_paths=image_paths,
                image_sizes=image_sizes,
                page_numbers=list(range(1, total + 1)),
                total_pages=total,
                preset=preset,
                density=eff_density,
                story_prompt=story_prompt,
                story_bible=story_bible,
                previous_summaries=None,
                tags=tags,
            )

        # Guarantee `high` density (5-8 items) even if sheet call skipped/under-filled individual cells
        if eff_density == "high":
            bible_txt = story_bible.to_prompt_block() if story_bible else ""
            bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT TOÀN BỘ CG ===\n{bible_txt}\n==========================================\n" if bible_txt else ""
            tags_block = _format_tags_block(tags)
            prompt_block = f"\nChỉ dẫn kịch bản / phong cách từ người dùng: {story_prompt}\n" if story_prompt else ""
            for p_no in range(1, total + 1):
                cur_items, cur_sum = results.get(p_no, ([], ""))
                if len(cur_items) >= min_n:
                    continue
                try:
                    with Image.open(image_paths[p_no - 1]) as im:
                        rgb_im = im.convert("RGB")
                        pw, ph = rgb_im.size
                        img_b64 = _pil_to_jpeg_b64(rgb_im, max_side=1400, quality=90)
                    if cur_items:
                        self._emit_status(f"Bổ sung thoại Trang {p_no}/{total} cho đủ mật độ HIGH ({len(cur_items)} -> {min_n}+)...")
                        topped = self._top_up_high_density_items(
                            img_b64=img_b64,
                            w=pw,
                            h=ph,
                            preset=preset,
                            existing_items=cur_items,
                            target_min=min_n,
                            target_max=8,
                            visual_scene=cur_sum,
                            page_summary=cur_sum,
                            bible_block=bible_block,
                            tags_block=tags_block,
                            prompt_block=prompt_block,
                        )
                        results[p_no] = (topped, cur_sum)
                    else:
                        results[p_no] = self.compose_cg_page(
                            np.array(rgb_im),
                            preset=preset,
                            density="high",
                            previous_summaries=[results[k][1] for k in sorted(results) if k < p_no and results[k][1]],
                            story_prompt=story_prompt,
                            story_bible=story_bible,
                            page_index=p_no,
                            tags=tags,
                        )
                except Exception:
                    pass
        return results

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

        eff_density = (density or "medium").strip().lower()
        if eff_density not in {"low", "medium", "high"}:
            eff_density = "medium"
        min_n, max_n = _compose_density_bounds(eff_density)
        density_guide = _compose_density_rule(eff_density)
        schema_examples = _compose_items_example_json(eff_density, preset)
        style_rule = _compose_style_rule(preset)
        bible_block = ""
        if story_bible:
            bible_txt = story_bible.to_prompt_block()
            if bible_txt:
                bible_block = f"\n=== HỒ SƠ CỐT TRUYỆN & NHÂN VẬT TOÀN BỘ CG ===\n{bible_txt}\n==========================================\n"
        prev_block = ""
        if previous_summaries:
            joined = "\n".join(f"- Cảnh trước: {s}" for s in previous_summaries[-6:])
            prev_block = f"\nDiễn biến & thoại liền trước cụm ảnh này (giữ mạch nối tiếp tự nhiên):\n{joined}\n"
        tags_block = _format_tags_block(tags)
        prompt_block = f"\nChỉ dẫn kịch bản / phong cách từ người dùng: {story_prompt}\n" if story_prompt else ""

        system_prompt = (
            "Bạn là biên kịch & đạo diễn dàn trang truyện tranh/CG chuyên nghiệp.\n"
            "Nhiệm vụ: Quan sát KỸ TỪNG Ô ẢNH được đánh số 'Trang X/Y' trên sheet tổng hợp "
            "(xác định đúng góc nhìn POV hay >=2 người, ánh mắt, nét mặt, hành động tay, bối cảnh), "
            "sáng tác lời thoại Tiếng Việt theo KHẨU NGỮ ĐỜI THƯỜNG thật tự nhiên, nối tiếp mượt mà giữa các trang, "
            "và trả về tọa độ đặt chữ riêng cho TỪNG TRANG.\n"
            + self._localization_rules()
            + "\nNGUYÊN TẮC HIỂU NGỮ CẢNH & SÁNG TÁC THOẠI CG NỐI TIẾP:\n"
            "- Phân biệt rõ POV (nói thẳng với người xem) và cảnh đối đáp nhiều người (đặt bóng về phía người nói).\n"
            "- Mỗi bóng thoại ngắn gọn (3–12 từ), tiếng thở/thì thầm `floating` siêu ngắn (1–5 từ).\n"
            "- Mỗi trang kế tiếp PHẢI có chuyển biến tâm lý/hành động mới; KHÔNG lặp lại nguyên văn câu thoại của trang trước.\n"
            + f"\nRÀNG BUỘC SỐ LƯỢNG MỖI TRANG (BẮT BUỘC): {density_guide}\n"
            + "QUY TẮC TỌA ĐỘ CỰC KỲ QUAN TRỌNG:\n"
            "- Với mỗi trang, `box_2d: [ymin, xmin, ymax, xmax]` tính từ 0 đến 1000 TRONG PHẠM VI Ô ẢNH CỦA TRANG ĐÓ (không phải tọa độ cả sheet).\n"
            "- Kích thước `box_2d` cho bubble (`white_oval`, `dark_translucent`, `speech_tail`) PHẢI TO RÕ RÀNG (rộng ~130..250/1000, cao ~130..260/1000) để chữ dễ đọc trên điện thoại.\n"
            "- TUYỆT ĐỐI KHÔNG che khuôn mặt, mắt, miệng hoặc hành động chính; các hộp không được chồng chéo và cách mép ô ảnh tối thiểu 20/1000.\n"
            "Bắt buộc trả về đúng JSON hợp lệ."
        )
        page_range_label = f"Trang {page_numbers[0]} đến {page_numbers[-1]}"
        user_prompt = (
            f"Các trang cần sáng tác trên sheet này ({len(image_paths)} ảnh: {page_range_label} / tổng {total_pages}):"
            f"{bible_block}{prev_block}{tags_block}{prompt_block}\n"
            f"MẬT ĐỘ THOẠI BẮT BUỘC MỖI ẢNH ({eff_density.upper()} — ĐÚNG {min_n}..{max_n} MỤC/TRANG): {density_guide}\n"
            f"Quy tắc loại thoại:\n{style_rule}\n\n"
            f"Hãy trả về JSON cho ĐỦ TẤT CẢ {len(image_paths)} trang ({page_range_label}) theo đúng số `page` ghi trên đầu mỗi ô ảnh:\n"
            "{\n"
            '  "pages": [\n'
            "    {\n"
            f'      "page": {page_numbers[0]},\n'
            '      "visual_scene": "Góc nhìn, nét mặt, hành động đang thấy trong trang này",\n'
            '      "page_summary": "Tóm tắt 1 câu diễn biến trang này",\n'
            f'      "items": [\n{schema_examples}\n      ]\n'
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
                items = _parse_composed_items(
                    p_obj.get("items", []), w=w, h=h, preset=preset, density=eff_density
                )
                results[p_no] = (items, str(p_obj.get("page_summary", "")).strip())
        return results


def _compose_density_bounds(density: str) -> tuple[int, int]:
    d = (density or "medium").strip().lower()
    if d == "low":
        return (1, 2)
    if d == "high":
        return (5, 8)
    return (3, 5)


def _compose_density_rule(density: str) -> str:
    d = (density or "medium").strip().lower()
    if d == "low":
        return (
            "CHẾ ĐỘ ÍT (LOW) — BẮT BUỘC CHỈ ĐÚNG 1 ĐẾN 2 MỤC/ẢNH (TUYỆT ĐỐI KHÔNG QUÁ 2 MỤC!): "
            "chỉ giữ 1 bóng thoại đắt giá nhất (+ tối đa 1 chữ thì thầm `floating` nếu thật sự cần), để khoảng trống rộng thoáng tôn bức tranh."
        )
    if d == "high":
        return (
            "CHẾ ĐỘ NHIỀU (HIGH) — BẮT BUỘC ĐẦY ĐẶN TỪ 5 ĐẾN 8 MỤC/ẢNH (TUYỆT ĐỐI KHÔNG ĐƯỢC ÍT HƠN 5 MỤC!): "
            "gồm ít nhất 2–3 `white_oval`/`speech_tail` (trao đổi qua lại nhiều nhịp), 1–2 `dark_translucent` (nội tâm thầm kín), "
            "và 2–3 `floating` (tiếng thở hổn hển / thì thầm sát tai / SFX cảm xúc đặt rải đều quanh cảnh)."
        )
    return (
        "CHẾ ĐỘ VỪA PHẢI (MEDIUM) — BẮT BUỘC TẠO ĐÚNG 3 ĐẾN 5 MỤC/ẢNH: "
        "cân bằng 2–3 bóng thoại chính + 1–2 suy nghĩ thầm (`dark_translucent`) hoặc tiếng thở/thì thầm (`floating`)."
    )


def _compose_items_example_json(density: str, preset: ComposePreset) -> str:
    d = (density or "medium").strip().lower()
    b_style = "speech_tail" if preset == ComposePreset.MANGA_TAIL else "white_oval"
    sample_1 = (
        '        {"id": 1, "kind": "bubble", "bubble_style": "'
        + ("none" if preset == ComposePreset.FLOATING_ONLY else b_style)
        + '", "box_2d": [60, 40, 240, 270], "tail_target": '
        + ("[280, 390]" if b_style == "speech_tail" else "null")
        + ', "text": "Lời thoại chính bật thành tiếng...", "text_color": [18, 18, 18], "stroke_color": null, "angle": 0}'
    )
    sample_2 = (
        '        {"id": 2, "kind": "'
        + ("floating" if preset == ComposePreset.FLOATING_ONLY else "bubble")
        + '", "bubble_style": "'
        + ("none" if preset == ComposePreset.FLOATING_ONLY else "dark_translucent")
        + '", "box_2d": [70, 700, 260, 940], "tail_target": null, "text": "Suy nghĩ thầm kín...", "text_color": [255, 255, 255], "stroke_color": [18, 18, 18], "angle": 0}'
    )
    sample_3 = (
        '        {"id": 3, "kind": "floating", "bubble_style": "none", '
        '"box_2d": [680, 60, 790, 280], "tail_target": null, "text": "Ưm... chậm thôi...", "text_color": [255, 255, 255], "stroke_color": [18, 18, 18], "angle": -4}'
    )
    sample_4 = (
        '        {"id": 4, "kind": "'
        + ("floating" if preset == ComposePreset.FLOATING_ONLY else "bubble")
        + '", "bubble_style": "'
        + ("none" if preset == ComposePreset.FLOATING_ONLY else b_style)
        + '", "box_2d": [320, 720, 500, 950], "tail_target": '
        + ("[390, 620]" if b_style == "speech_tail" else "null")
        + ', "text": "Câu đáp lại nhịp hai...", "text_color": [18, 18, 18], "stroke_color": null, "angle": 0}'
    )
    sample_5 = (
        '        {"id": 5, "kind": "floating", "bubble_style": "none", '
        '"box_2d": [740, 690, 860, 930], "tail_target": null, "text": "Nóng quá...♡", "text_color": [255, 200, 225], "stroke_color": [20, 15, 25], "angle": 5}'
    )
    if d == "low":
        return sample_1
    if d == "high":
        return ",\n".join((sample_1, sample_2, sample_3, sample_4, sample_5))
    return ",\n".join((sample_1, sample_2, sample_3))


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
    density: str = "medium",
) -> list[TextRegionItem]:
    from koharu_vn_cli.pipeline.typesetter import sanitize_render_text

    items: list[TextRegionItem] = []
    seen_texts: set[str] = set()
    for idx, raw in enumerate(raw_items or [], start=1):
        text = sanitize_render_text(str(raw.get("text", "")).strip())
        box = raw.get("box_2d")
        if not text or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        norm_txt = normalize_source_key(text).lower()
        if norm_txt and norm_txt in seen_texts:
            continue
        if norm_txt:
            seen_texts.add(norm_txt)

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
        _, max_n = _compose_density_bounds(density)
        if (density or "medium").strip().lower() == "low" and len(items) > max_n:
            bubbles = [it for it in items if it.kind == "bubble"]
            floats = [it for it in items if it.kind != "bubble"]
            kept: list[TextRegionItem] = []
            if bubbles:
                kept.append(bubbles[0])
            if len(kept) < max_n and floats:
                kept.append(floats[0])
            for it in items:
                if len(kept) >= max_n:
                    break
                if it not in kept:
                    kept.append(it)
            items = [it.model_copy(update={"id": idx}) for idx, it in enumerate(kept, start=1)]
        items = resolve_box_overlaps(items, w, h)
    return items
