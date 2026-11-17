"""Automated tests for AppConfig, natural-language intent parsing, and Textual TUI interactions."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from PIL import Image

from koharu_vn_cli.config import AppConfig, load_config, save_config
from koharu_vn_cli.models import BubbleStyle, ComposePreset, PageDocument, TextRegionItem
from koharu_vn_cli.tui import (
    KoharuTUI,
    PickerModal,
    extract_path_and_remainder,
    parse_natural_intent,
    split_cmd_and_hint,
    strip_path_quotes,
)


def test_config_roundtrip_and_masking(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.json"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = AppConfig(api_key="AIzaTest1234SecretKey", compose_style=ComposePreset.MANGA_TAIL, lang="ja")
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)

    assert loaded.compose_style == ComposePreset.MANGA_TAIL
    assert loaded.lang == "ja"
    assert loaded.masked_api_key() == "AIza...tKey"


def test_path_cleaning_and_natural_intent(tmp_path: Path) -> None:
    sample_dir = tmp_path / "Manga Chap 1"
    sample_dir.mkdir()

    assert strip_path_quotes(f'& "{sample_dir}"') == str(sample_dir)

    p, rest = extract_path_and_remainder(f'dịch "{sample_dir}" xưng hô anh em')
    assert p == str(sample_dir)
    assert "anh em" in rest

    act, parsed_p, opts = parse_natural_intent(f'sáng tác thoại "{sample_dir}" kiểu đuôi nhọn ngọt ngào')
    assert act == "compose"
    assert parsed_p == str(sample_dir)
    assert opts["style"] == ComposePreset.MANGA_TAIL

    act2, _, _ = parse_natural_intent(f'dịch "{sample_dir}" phong cách tự nhiên')
    assert act2 == "translate"

    cmd_p, hint = split_cmd_and_hint(f'"{sample_dir}" văn phong kiếm hiệp')
    assert cmd_p == str(sample_dir)
    assert hint == "văn phong kiếm hiệp"


def test_tui_slash_popup_style_picker_and_rerender(tmp_path: Path, monkeypatch) -> None:
    cfg_file = tmp_path / "cfg.json"
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: load_config(cfg_file))
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda c: save_config(c, cfg_file))

    # Prepare a cached page ready for `/rerender` inside the TUI
    src_img = tmp_path / "cg.png"
    out_dir = tmp_path / "cg_out"
    cache_dir = out_dir / ".koharu_cache"
    cache_dir.mkdir(parents=True)
    Image.new("RGB", (320, 220), (140, 160, 200)).save(src_img)
    doc = PageDocument(
        image_path=str(src_img),
        cleaned_image_path=str(src_img),
        width=320,
        height=220,
        mode="compose",
        regions=[
            TextRegionItem(
                id=1,
                kind="bubble",
                bubble_style=BubbleStyle.WHITE_OVAL,
                text_bbox=(20, 20, 180, 140),
                bubble_bbox=(20, 20, 180, 140),
                translated_text="Chào mừng tới Koharu-VN CLI!",
                draw_bubble=True,
            )
        ],
    )
    (cache_dir / "cg.koharu.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    async def scenario() -> None:
        app = KoharuTUI()
        async with app.run_test(size=(120, 36)) as pilot:
            # 1. Typing `/` pops up floating slash command menu
            await pilot.press("/")
            await pilot.pause()
            popup = app.query_one("#slash-popup")
            assert popup.has_class("visible")

            # Close popup with Escape
            await pilot.press("escape")
            await pilot.pause()
            assert not popup.has_class("visible")

            # 2. Switch CG style interactively via `/style` modal picker (2nd option = manga-tail)
            app.dispatch_command("/style")
            await pilot.pause()
            assert isinstance(app.screen, PickerModal)
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.cfg.compose_style == ComposePreset.MANGA_TAIL

            # 3. Toggle sidebar via Ctrl+B
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app.has_class("sidebar-hidden")
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert not app.has_class("sidebar-hidden")

            # 3b. `/tags` sets and clears genre tags in config
            app.dispatch_command("/tags vanilla, NTR")
            await pilot.pause()
            assert app.cfg.tags == "vanilla, NTR"
            app.dispatch_command("/tags clear")
            await pilot.pause()
            assert app.cfg.tags == ""

            # 3c. `/story-model` sets and clears primary Pass-1 model
            app.dispatch_command("/story-model gemini-3.1-pro-preview")
            await pilot.pause()
            assert app.cfg.story_model == "gemini-3.1-pro-preview"
            app.dispatch_command("/story-model off")
            await pilot.pause()
            assert app.cfg.story_model == ""

            # 3d. `/key` supports multiple rotating API keys
            app.dispatch_command("/key AIzaKeyOne111, AIzaKeyTwo222")
            await pilot.pause()
            assert app.cfg.effective_api_keys()[:2] == ["AIzaKeyOne111", "AIzaKeyTwo222"]
            assert "2 keys" in app.cfg.masked_api_key()

            # 3e. `/cache` toggles auto-cleaning `.koharu_cache`
            assert app.cfg.clean_cache is True
            app.dispatch_command("/cache keep")
            await pilot.pause()
            assert app.cfg.clean_cache is False
            app.dispatch_command("/cache clean")
            await pilot.pause()
            assert app.cfg.clean_cache is True

            # 3f. `/workers` sets explicit parallel API worker count or resets to auto
            app.dispatch_command("/workers 3")
            await pilot.pause()
            assert app.cfg.workers == 3
            app.dispatch_command("/workers auto")
            await pilot.pause()
            assert app.cfg.workers == 0

            # 4. Run `/rerender` in background worker inside TUI
            app.dispatch_command(f'/rerender "{out_dir}"')
            deadline = time.monotonic() + 8.0
            while not (out_dir / "cg.png").is_file() and time.monotonic() < deadline:
                await pilot.pause(0.05)
            assert (out_dir / "cg.png").is_file()

    asyncio.run(scenario())
