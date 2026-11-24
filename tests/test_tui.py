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

    cfg = AppConfig(api_key="AIzaTest1234SecretKey", compose_style=ComposePreset.MANGA_TAIL, lang="ja", story_pass=False)
    save_config(cfg, cfg_path)
    loaded = load_config(cfg_path)

    assert loaded.compose_style == ComposePreset.MANGA_TAIL
    assert loaded.lang == "ja"
    assert loaded.story_pass is False
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

            # 3g. `/notify` toggles desktop notification setting
            assert app.cfg.notify is True
            app.dispatch_command("/notify off")
            await pilot.pause()
            assert app.cfg.notify is False
            app.dispatch_command("/notify on")
            await pilot.pause()
            assert app.cfg.notify is True

            # 3h. `/pass1` & `/story-pass` toggle Pass 1 Story Bible setting
            assert app.cfg.story_pass is True
            app.dispatch_command("/pass1 off")
            await pilot.pause()
            assert app.cfg.story_pass is False
            app.dispatch_command("/pass1 on")
            await pilot.pause()
            assert app.cfg.story_pass is True
            app.dispatch_command("/story-pass skip")
            await pilot.pause()
            assert app.cfg.story_pass is False
            app.dispatch_command("/pass1")
            await pilot.pause()
            assert app.cfg.story_pass is True

            # 4. Run `/rerender` in background worker inside TUI
            app.dispatch_command(f'/rerender "{out_dir}"')
            deadline = time.monotonic() + 8.0
            while not (out_dir / "cg.png").is_file() and time.monotonic() < deadline:
                await pilot.pause(0.05)
            assert (out_dir / "cg.png").is_file()

    asyncio.run(scenario())


def test_tui_zip_drop_intent_and_review_timeout_and_notify(tmp_path: Path, monkeypatch) -> None:
    from koharu_vn_cli.models import CharacterProfile, StoryBible
    from koharu_vn_cli.notify import _build_windows_toast_script
    from koharu_vn_cli.tui import StoryBibleReviewModal

    cfg_file = tmp_path / "cfg.json"
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: load_config(cfg_file))
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda c: save_config(c, cfg_file))

    zip_file = tmp_path / "Manga Chap 1.zip"
    zip_file.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    # 1. Dropping a .zip path directly (quoted, PowerShell &drop, or file:// URI) triggers translate directly
    for dropped in (
        f'"{zip_file}"',
        f"& '{zip_file}'",
        zip_file.as_uri(),
    ):
        act, parsed_p, opts = parse_natural_intent(dropped)
        assert act == "translate"
        assert parsed_p is not None and Path(parsed_p) == zip_file
        assert opts["prompt"] == ""

    # Dropping .zip with explicit compose keyword still picks compose
    act_c, _, _ = parse_natural_intent(f'sáng tác "{zip_file}"')
    assert act_c == "compose"

    # 2. Toast XML escapes special characters cleanly
    ps_src = _build_windows_toast_script("Koharu <Test>", "Lỗi & timeout > 60s", is_error=True)
    assert "Koharu &lt;Test&gt;" in ps_src
    assert "Lỗi &amp; timeout &gt; 60s" in ps_src

    # 3. StoryBibleReviewModal defaults to 10s idle timeout & auto-confirms when idle
    bible = StoryBible(
        source_game="Blue Archive",
        characters=[CharacterProfile(name="Hoshino", appearance="tóc hồng", pronoun_rules="tớ - cậu")],
    )
    default_modal = StoryBibleReviewModal(bible)
    assert default_modal._timeout_seconds == 10

    results: list[StoryBible | None] = []
    timed_out_flag: list[bool] = []

    async def review_timeout_scenario() -> None:
        app = KoharuTUI()
        async with app.run_test(size=(120, 36)) as pilot:
            modal = StoryBibleReviewModal(bible, timeout_seconds=1)
            app.push_screen(modal, lambda res: (results.append(res), timed_out_flag.append(modal.timed_out)))
            await pilot.pause(1.35)
            assert timed_out_flag == [True]
            assert len(results) == 1 and results[0] is not None
            assert results[0].source_game == "Blue Archive"
            assert results[0].characters[0].name == "Hoshino"

    asyncio.run(review_timeout_scenario())

    # 4. `_start_translate` sends notification on success and on error
    notifs: list[tuple[str, str, bool, bool]] = []
    monkeypatch.setattr(
        "koharu_vn_cli.tui.send_desktop_notification",
        lambda title, msg, is_error=False, enabled=True: (notifs.append((title, msg, is_error, enabled)), True)[1],
    )
    monkeypatch.setattr(
        "koharu_vn_cli.tui.run_translate_pipeline",
        lambda p, **_kw: (tmp_path / "out_ok", []),
    )
    app_ok = KoharuTUI()
    app_ok.call_from_thread = lambda fn, *a, **kw: None  # type: ignore[method-assign]
    app_ok._start_translate.__wrapped__(app_ok, zip_file)
    assert len(notifs) == 1
    assert notifs[0][2] is False and "Dịch xong" in notifs[0][0]

    notifs.clear()

    def boom(*_a, **_kw):
        raise RuntimeError("API broken")

    monkeypatch.setattr("koharu_vn_cli.tui.run_translate_pipeline", boom)
    app_ok._start_translate.__wrapped__(app_ok, zip_file)
    assert len(notifs) == 1
    assert notifs[0][2] is True and "API broken" in notifs[0][1]


def test_skip_pass1_cli_tui_and_natural_intent(tmp_path: Path, monkeypatch) -> None:
    """Verify `--skip-pass1`, `/pass1 off`, and natural-language skip Pass 1 across CLI & TUI."""
    from typer.testing import CliRunner

    from koharu_vn_cli.cli import app as cli_app

    cfg_file = tmp_path / "cfg.json"
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: load_config(cfg_file))
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda c: save_config(c, cfg_file))
    monkeypatch.setattr("koharu_vn_cli.cli.load_config", lambda: load_config(cfg_file))

    sample_dir = tmp_path / "Doujin Sample"
    sample_dir.mkdir()
    zip_file = tmp_path / "CG Pack.zip"
    zip_file.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    # 1. Natural intent extracts Vietnamese/English skip Pass 1 instructions
    act, parsed_p, opts = parse_natural_intent(f'dịch "{sample_dir}" bỏ qua pass 1 xưng hô anh em')
    assert act == "translate"
    assert parsed_p == str(sample_dir)
    assert opts["story_pass"] is False
    assert "anh em" in opts["prompt"]

    act_c, _, opts_c = parse_natural_intent(f'sáng tác "{sample_dir}" skip pass 1 ngọt ngào')
    assert act_c == "compose"
    assert opts_c["story_pass"] is False

    # 2. TUI `/translate --skip-pass1`, `cfg.story_pass=False`, and `_start_compose` propagate `story_pass=False`
    seen_translate: list[bool] = []
    seen_compose: list[bool] = []
    monkeypatch.setattr(
        "koharu_vn_cli.tui.run_translate_pipeline",
        lambda p, **kw: (seen_translate.append(bool(kw.get("story_pass"))), (tmp_path / "out_t", []))[1],
    )
    monkeypatch.setattr(
        "koharu_vn_cli.tui.run_compose_pipeline",
        lambda p, **kw: (seen_compose.append(bool(kw.get("story_pass"))), (tmp_path / "out_c", []))[1],
    )
    tui_app = KoharuTUI()
    tui_app.cfg.api_key = "test-key"
    tui_app.call_from_thread = lambda fn, *a, **kw: None  # type: ignore[method-assign]
    tui_app._start_translate.__wrapped__(tui_app, zip_file, story_pass=False)
    assert seen_translate == [False]

    seen_translate.clear()
    tui_app.cfg.story_pass = False
    tui_app._start_translate.__wrapped__(tui_app, zip_file)
    tui_app._start_compose.__wrapped__(tui_app, sample_dir)
    assert seen_translate == [False]
    assert seen_compose == [False]

    # 3. Headless CLI `--skip-pass1` & saved config `story_pass=False`
    cli_seen_t: list[bool] = []
    cli_seen_c: list[bool] = []
    monkeypatch.setattr(
        "koharu_vn_cli.cli.run_translate_pipeline",
        lambda p, **kw: (cli_seen_t.append(bool(kw.get("story_pass"))), (tmp_path / "out_cli_t", []))[1],
    )
    monkeypatch.setattr(
        "koharu_vn_cli.cli.run_compose_pipeline",
        lambda p, **kw: (cli_seen_c.append(bool(kw.get("story_pass"))), (tmp_path / "out_cli_c", []))[1],
    )
    runner = CliRunner()
    res1 = runner.invoke(cli_app, ["translate", str(zip_file), "--skip-pass1", "--api-key", "test-key"])
    assert res1.exit_code == 0, res1.output
    res2 = runner.invoke(cli_app, ["compose", str(sample_dir), "--skip-pass1", "--api-key", "test-key"])
    assert res2.exit_code == 0, res2.output
    assert cli_seen_t == [False]
    assert cli_seen_c == [False]

    # Even without --skip-pass1 flag, saved config story_pass=False applies by default, and --pass1 overrides it
    save_config(AppConfig(api_key="test-key", story_pass=False), cfg_file)
    cli_seen_t.clear()
    res3 = runner.invoke(cli_app, ["translate", str(zip_file)])
    assert res3.exit_code == 0, res3.output
    res4 = runner.invoke(cli_app, ["translate", str(zip_file), "--pass1"])
    assert res4.exit_code == 0, res4.output
    assert cli_seen_t == [False, True]


def test_tui_queue_and_chrome_handle_square_brackets_in_filenames(tmp_path: Path, monkeypatch) -> None:
    """Doujinshi filenames like '(C100) [Solar Milk Soft] ...zip' must never raise Rich MarkupError."""
    zip_a = tmp_path / "(C100) [Solar Milk Soft (Solar Milk)] Barbara Bad End (Genshin Impact).zip"
    zip_b = tmp_path / "[Circle [Sub]] Chapter [bold] Test.zip"
    zip_a.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    zip_b.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    async def scenario() -> None:
        app = KoharuTUI()
        app.cfg.api_key = "test-key"
        app.cfg.last_output_dir = str(tmp_path / "(C100) [Solar Milk Soft] out")
        app.cfg.custom_prompt = "nhân vật [Barbara] nói nhẹ nhàng"
        app.cfg.tags = "[Genshin] vanilla"

        async with app.run_test(size=(120, 36)) as pilot:
            # Pretend a job is already running so /queue enqueues immediately and refreshes sidebar chrome
            app.is_running_job = True
            app.dispatch_command(f'/queue "{zip_a}" "{zip_b}"')
            await pilot.pause(0.05)

            assert len(app.job_queue) == 2
            # Exercise queue table, status updates, timer tick, malformed markup log, and queue remove
            app.dispatch_command("/queue list")
            app._set_job_ui(True, f"Dịch truyện: {zip_a.name}", 25.0)
            app._tick_job_timer()
            app._log_ai("Malformed [/dim] [unclosed tag message", style="yellow")
            app.dispatch_command("/queue remove 1")
            await pilot.pause(0.05)
            assert len(app.job_queue) == 1

    asyncio.run(scenario())


