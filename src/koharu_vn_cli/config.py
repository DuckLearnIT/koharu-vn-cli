"""Persistent session configuration for Koharu-VN CLI & TUI."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from koharu_vn_cli.models import ComposePreset
from koharu_vn_cli.pipeline.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    is_deepseek_endpoint,
    is_deepseek_model,
    is_groq_endpoint,
    is_groq_model,
)

CONFIG_DIR = Path.home() / ".koharu-vn"
CONFIG_FILE = CONFIG_DIR / "config.json"


def parse_api_keys(raw: str | None) -> list[str]:
    """Split comma/semicolon/newline/whitespace-separated API keys, preserving order without duplicates."""
    if not raw:
        return []
    seen: set[str] = set()
    keys: list[str] = []
    for part in re.split(r"[\s,;]+", raw.strip()):
        k = part.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def parse_model_list(raw: str | None) -> list[str]:
    """Split comma/semicolon/newline-separated model names, preserving order without duplicates."""
    if not raw:
        return []
    seen: set[str] = set()
    models: list[str] = []
    for part in re.split(r"[\s,;]+", raw.strip()):
        m = part.strip()
        if m and m not in seen:
            seen.add(m)
            models.append(m)
    return models


class AppConfig(BaseModel):
    """Saved user preferences across CLI & interactive TUI sessions."""

    api_key: str = ""
    api_keys: list[str] = Field(default_factory=list)
    deepseek_api_key: str = ""
    deepseek_api_keys: list[str] = Field(default_factory=list)
    groq_api_key: str = ""
    groq_api_keys: list[str] = Field(default_factory=list)
    base_url: Optional[str] = None
    model: str = DEFAULT_GEMINI_MODEL
    fallback_model: str = ""
    story_model: str = ""
    story_fallback_model: str = ""
    lang: str = "auto"
    compose_style: ComposePreset = ComposePreset.CG_OVAL
    density: str = "medium"
    font: Optional[str] = None
    custom_prompt: str = ""
    story_context: str = ""
    tags: str = ""
    conf_thresh: float = Field(default=0.34, ge=0.15, le=0.85)
    accurate_detector: bool = True
    save_cbz: bool = False
    output_format: str = "webp"
    sequential_names: bool = True
    story_pass: bool = True
    review_characters: bool = True
    vision_translate: bool = True
    skip_sfx: bool = True
    clean_cache: bool = True
    delete_source_zip: bool = True
    use_gpu: bool = True
    notify: bool = True
    workers: int = Field(default=0, ge=0, le=16)
    theme: str = "tokyo-night"
    last_output_dir: Optional[str] = None
    last_input_path: Optional[str] = None
    last_job_mode: Optional[str] = None

    def effective_api_keys(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> list[str]:
        """Return all configured API keys in failover order (saved list + single key + env vars)."""
        eff_model = model if model is not None else self.model
        eff_url = base_url if base_url is not None else self.base_url
        use_groq = is_groq_model(eff_model) or is_groq_endpoint(eff_url)
        use_deepseek = is_deepseek_model(eff_model) or is_deepseek_endpoint(eff_url)
        combined: list[str] = []
        seen: set[str] = set()
        saved_keys = [*self.api_keys, *parse_api_keys(self.api_key)]
        if use_groq:
            sources = [
                *self.groq_api_keys,
                *parse_api_keys(self.groq_api_key),
                *parse_api_keys(os.environ.get("GROQ_API_KEYS", "")),
                *parse_api_keys(os.environ.get("GROQ_API_KEY", "")),
            ]
        elif use_deepseek:
            sources = [
                *self.deepseek_api_keys,
                *parse_api_keys(self.deepseek_api_key),
                *parse_api_keys(os.environ.get("DEEPSEEK_API_KEYS", "")),
                *parse_api_keys(os.environ.get("DEEPSEEK_API_KEY", "")),
            ]
        else:
            gem_saved = [
                k
                for k in saved_keys
                if eff_url or (not k.startswith("sk-") and not k.startswith("gsk_"))
            ]
            sources = [
                *gem_saved,
                *parse_api_keys(os.environ.get("GEMINI_API_KEYS", "")),
                *parse_api_keys(os.environ.get("GEMINI_API_KEY", "")),
                *(parse_api_keys(os.environ.get("OPENAI_API_KEY", "")) if eff_url else []),
            ]
        for k in sources:
            if k and k not in seen:
                seen.add(k)
                combined.append(k)
        return combined

    def effective_api_key(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        keys = self.effective_api_keys(model=model, base_url=base_url)
        return keys[0] if keys else ""

    def masked_api_key(self, *, model: str | None = None, base_url: str | None = None) -> str:
        keys = self.effective_api_keys(model=model, base_url=base_url)
        if not keys:
            return "Chưa đặt"
        key = keys[0]
        masked = "*" * len(key) if len(key) <= 10 else f"{key[:4]}...{key[-4:]}"
        return f"{masked} ({len(keys)} keys)" if len(keys) > 1 else masked

    def masked_gemini_key(self) -> str:
        return self.masked_api_key(model="gemini-3.8-flash", base_url=self.base_url or "")

    def masked_deepseek_key(self) -> str:
        return self.masked_api_key(model="deepseek-chat", base_url="https://api.deepseek.com/v1")

    def masked_groq_key(self) -> str:
        return self.masked_api_key(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1")


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    """Load saved config JSON from disk, falling back cleanly to defaults."""
    if path.is_file():
        try:
            cfg = AppConfig.model_validate_json(path.read_text(encoding="utf-8"))
            if not cfg.base_url:
                all_saved = [*cfg.api_keys, *parse_api_keys(cfg.api_key)]
                sk_keys = [k for k in all_saved if k.startswith("sk-")]
                gsk_keys = [k for k in all_saved if k.startswith("gsk_")]
                non_provider = [
                    k for k in all_saved if not k.startswith("sk-") and not k.startswith("gsk_")
                ]
                if sk_keys:
                    merged_ds = list(dict.fromkeys([*cfg.deepseek_api_keys, *parse_api_keys(cfg.deepseek_api_key), *sk_keys]))
                    cfg.deepseek_api_keys = merged_ds
                    cfg.deepseek_api_key = merged_ds[0] if merged_ds else ""
                if gsk_keys:
                    merged_groq = list(dict.fromkeys([*cfg.groq_api_keys, *parse_api_keys(cfg.groq_api_key), *gsk_keys]))
                    cfg.groq_api_keys = merged_groq
                    cfg.groq_api_key = merged_groq[0] if merged_groq else ""
                if sk_keys or gsk_keys:
                    cfg.api_keys = non_provider
                    cfg.api_key = non_provider[0] if non_provider else ""
            return cfg
        except Exception:
            pass
    return AppConfig()


def save_config(cfg: AppConfig, path: Path = CONFIG_FILE) -> Path:
    """Persist user preferences to `~/.koharu-vn/config.json`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return path
