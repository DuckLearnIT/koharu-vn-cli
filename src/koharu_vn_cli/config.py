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
    review_characters: bool = True
    vision_translate: bool = True
    skip_sfx: bool = True
    clean_cache: bool = True
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
        use_deepseek = is_deepseek_model(eff_model) or is_deepseek_endpoint(eff_url)
        combined: list[str] = []
        seen: set[str] = set()
        saved_keys = [*self.api_keys, *parse_api_keys(self.api_key)]
        if use_deepseek:
            env_keys = [
                *parse_api_keys(os.environ.get("DEEPSEEK_API_KEYS", "")),
                *parse_api_keys(os.environ.get("DEEPSEEK_API_KEY", "")),
                *parse_api_keys(os.environ.get("OPENAI_API_KEY", "")),
            ]
            # Avoid sending saved Google AIza keys to DeepSeek when switching models.
            saved_filtered = [k for k in saved_keys if not k.startswith("AIza")] if env_keys or saved_keys else saved_keys
            sources = [*saved_filtered, *env_keys]
        else:
            sources = [
                *saved_keys,
                *parse_api_keys(os.environ.get("GEMINI_API_KEYS", "")),
                *parse_api_keys(os.environ.get("GEMINI_API_KEY", "")),
                *parse_api_keys(os.environ.get("OPENAI_API_KEY", "")),
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

    def masked_api_key(self) -> str:
        keys = self.effective_api_keys()
        if not keys:
            return "Chưa đặt"
        key = keys[0]
        masked = "*" * len(key) if len(key) <= 10 else f"{key[:4]}...{key[-4:]}"
        return f"{masked} ({len(keys)} keys)" if len(keys) > 1 else masked


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    """Load saved config JSON from disk, falling back cleanly to defaults."""
    if path.is_file():
        try:
            return AppConfig.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return AppConfig()


def save_config(cfg: AppConfig, path: Path = CONFIG_FILE) -> Path:
    """Persist user preferences to `~/.koharu-vn/config.json`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return path
