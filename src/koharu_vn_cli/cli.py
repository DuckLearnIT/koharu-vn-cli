"""Koharu-VN CLI: OpenCode/Claude-Code style TUI + headless manga/CG commands."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import typer
from rich.console import Console
from rich.markup import escape as escape_markup
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from koharu_vn_cli.config import load_config
from koharu_vn_cli.models import CharacterProfile, ComposePreset, StoryBible
from koharu_vn_cli.notify import send_desktop_notification
from koharu_vn_cli.pipeline.fonts import available_fonts
from koharu_vn_cli.pipeline.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    DEEPSEEK_BASE_URL,
    VisionLLMClient,
    is_deepseek_model,
)
from koharu_vn_cli.pipeline.runner import (
    ModelCache,
    PipelineHooks,
    pack_images_to_webp,
    run_compose_pipeline,
    run_rerender_pipeline,
    run_translate_pipeline,
)
from koharu_vn_cli.tui import launch_tui

app = typer.Typer(
    name="koharu-vn",
    help="AI CLI dịch truyện Nhật/Hàn/Trung & tự sáng tác thoại CG sang Tiếng Việt (chạy không cần Rust/MSVC).",
)
console = Console()


@app.callback(invoke_without_command=True)
def default_entry(
    ctx: typer.Context,
    theme: Optional[str] = typer.Option(None, "--theme", help="Theme giao diện TUI (vd: tokyo-night, catppuccin-mocha)."),
) -> None:
    """Chạy `koharu-vn` không kèm lệnh con sẽ mở giao diện AI TUI tương tác (giống Claude Code / OpenCode)."""
    if ctx.invoked_subcommand is None:
        launch_tui(theme=theme)


@app.command()
def ui(
    theme: Optional[str] = typer.Option(None, "--theme", help="Theme giao diện TUI (tokyo-night, catppuccin-mocha, dracula, nord)."),
) -> None:
    """Mở giao diện AI CLI tương tác toàn màn hình (kiểu Claude Code / OpenCode)."""
    launch_tui(theme=theme)


def _resolve_auth(api_key: str | None, base_url: str | None, model: str | None) -> tuple[str | None, list[str], str | None, str]:
    cfg = load_config()
    eff_model = model or cfg.model or DEFAULT_GEMINI_MODEL
    eff_url = base_url or cfg.base_url
    if not eff_url and is_deepseek_model(eff_model):
        eff_url = DEEPSEEK_BASE_URL
    cli_keys = [k.strip() for k in (api_key or "").replace(";", ",").split(",") if k.strip()] if api_key else []
    eff_keys = cli_keys or cfg.effective_api_keys(model=eff_model, base_url=eff_url)
    eff_key = eff_keys[0] if eff_keys else None
    return eff_key, eff_keys, eff_url, eff_model


def _cli_prompt_with_timeout(
    prompt_text: str,
    default: str = "",
    *,
    timeout_seconds: float = 10.0,
) -> tuple[str, bool]:
    """Read interactive CLI input with inactivity timeout; returns `(value, timed_out)`."""
    if timeout_seconds <= 0 or not sys.stdin or not sys.stdin.isatty():
        return str(typer.prompt(prompt_text, default=default)), False

    sys.stdout.write(f"{prompt_text} [{default}]: ")
    sys.stdout.flush()

    if sys.platform.startswith("win"):
        import msvcrt

        buf: list[str] = []
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                deadline = time.monotonic() + timeout_seconds
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    val = "".join(buf).strip()
                    return (val if val else default), False
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch in ("\x08", "\x7f"):
                    if buf:
                        buf.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                if ch in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                buf.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
            else:
                time.sleep(0.05)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return default, True

    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return default, True
    line = sys.stdin.readline().strip()
    return (line if line else default), False


def _cli_confirm_story_bible(
    bible: StoryBible,
    progress: Progress,
    *,
    timeout_seconds: float = 10.0,
) -> StoryBible:
    """Allow terminal user to inspect & fix detected game/character names (auto-confirms after 10s idle)."""
    progress.stop()
    try:
        console.print("\n[bold magenta]=== Kiểm tra Game & Nhân vật (Pass 1) ===[/bold magenta]")
        console.print(
            f"[dim](Tự động xác nhận giữ nguyên nếu bạn không phản hồi trong {int(timeout_seconds)} giây)[/dim]"
        )
        console.print(f"Game / Series: [cyan]{bible.source_game or '(chưa rõ)'}[/cyan]")
        new_game, timed_out = _cli_prompt_with_timeout(
            "Sửa tên Game / Series (Enter để giữ nguyên)",
            default=bible.source_game,
            timeout_seconds=timeout_seconds,
        )
        if timed_out:
            console.print(f"[yellow]⏱ Hết {int(timeout_seconds)}s không phản hồi -> tự động xác nhận & tiếp tục![/yellow]")
            return bible

        updated_chars: list[CharacterProfile] = []
        for idx, ch in enumerate(bible.characters, start=1):
            console.print(f"  [{idx}] [bold]{ch.name}[/bold] ({ch.appearance}) -> [green]{ch.pronoun_rules}[/green]")
            new_name, t1 = _cli_prompt_with_timeout(
                f"    Tên nhân vật #{idx}", default=ch.name, timeout_seconds=timeout_seconds
            )
            if t1:
                console.print(f"[yellow]⏱ Hết {int(timeout_seconds)}s không phản hồi -> tự động xác nhận các mục còn lại![/yellow]")
                updated_chars.extend(bible.characters[idx - 1 :])
                return bible.model_copy(update={"source_game": new_game.strip(), "characters": updated_chars})
            new_pron, t2 = _cli_prompt_with_timeout(
                f"    Xưng hô #{idx}", default=ch.pronoun_rules, timeout_seconds=timeout_seconds
            )
            if t2:
                updated_chars.append(ch.model_copy(update={"name": new_name.strip() or ch.name}))
                updated_chars.extend(bible.characters[idx:])
                console.print(f"[yellow]⏱ Hết {int(timeout_seconds)}s không phản hồi -> tự động xác nhận các mục còn lại![/yellow]")
                return bible.model_copy(update={"source_game": new_game.strip(), "characters": updated_chars})
            updated_chars.append(
                ch.model_copy(update={"name": new_name.strip() or ch.name, "pronoun_rules": new_pron.strip()})
            )
        return bible.model_copy(update={"source_game": new_game.strip(), "characters": updated_chars})
    finally:
        progress.start()


def _execute_translate_cli(
    input_path: Path,
    *,
    output: Optional[Path],
    lang: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    fallback_model: Optional[str],
    story_model: Optional[str],
    story_fallback: Optional[str],
    prompt: Optional[str],
    story: Optional[str],
    tags: Optional[str],
    story_pass: bool,
    font: Optional[str],
    conf: float,
    accurate_detector: bool,
    save_cbz: bool,
    output_format: Optional[str],
    seq_names: Optional[bool],
    review_chars: bool,
    vision: Optional[bool],
    skip_sfx: Optional[bool],
    gpu: Optional[bool],
    clean_cache: Optional[bool],
    delete_zip: Optional[bool] = None,
    notify: Optional[bool] = None,
    workers: Optional[int] = None,
    resume: bool = False,
    retranslate_untranslated: bool = False,
) -> None:
    cfg = load_config()
    eff_key, eff_keys, eff_url, eff_model = _resolve_auth(api_key, base_url, model)
    eff_format = output_format or cfg.output_format
    eff_seq = cfg.sequential_names if seq_names is None else seq_names
    eff_vision = cfg.vision_translate if vision is None else vision
    eff_skip_sfx = cfg.skip_sfx if skip_sfx is None else skip_sfx
    eff_gpu = cfg.use_gpu if gpu is None else gpu
    eff_clean_cache = cfg.clean_cache if clean_cache is None else clean_cache
    eff_delete_zip = cfg.delete_source_zip if delete_zip is None else delete_zip
    eff_notify = cfg.notify if notify is None else notify
    eff_workers = cfg.workers if workers is None else max(0, min(workers, 16))
    job_title = "Đang tái dịch trang chưa dịch..." if retranslate_untranslated else "Đang dịch truyện..."
    err_notif_title = "Koharu-VN · Lỗi tái dịch" if retranslate_untranslated else "Koharu-VN · Lỗi dịch truyện"
    done_notif_title = "Koharu-VN · Tái dịch xong!" if retranslate_untranslated else "Koharu-VN · Dịch xong!"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(job_title, total=None)
        completed_pages = 0

        def on_step(idx: int, total: int, name: str, stage: str, detail: str) -> None:
            progress.update(
                task,
                total=max(1, total),
                description=f"[{idx}/{total}] {escape_markup(name)}: {escape_markup(detail)}",
            )

        def on_page_done(idx: int, total: int, name: str, doc, out_path: Path) -> None:
            nonlocal completed_pages
            completed_pages += 1
            progress.update(task, total=max(1, total), completed=completed_pages)

        def on_story_bible(bible) -> None:
            game_tag = f" [{escape_markup(bible.source_game)}]" if getattr(bible, "source_game", "") else ""
            progress.console.print(
                f"[cyan]✓ Pass 1 Story Bible{game_tag}:[/cyan] {escape_markup(bible.title or 'Đã phân tích')} "
                f"({len(bible.characters)} nhân vật) — [dim]{escape_markup(bible.full_plot_summary[:140])}[/dim]"
            )

        try:
            out_dir, docs = run_translate_pipeline(
                input_path,
                output=output,
                lang=lang,
                api_key=eff_key,
                api_keys=eff_keys,
                base_url=eff_url,
                model=eff_model,
                fallback_model=fallback_model if fallback_model is not None else (cfg.fallback_model or None),
                story_model=story_model or cfg.story_model or None,
                story_fallback_model=story_fallback if story_fallback is not None else (cfg.story_fallback_model or None),
                prompt=prompt,
                story=story,
                tags=tags if tags is not None else (cfg.tags or None),
                story_pass=story_pass,
                font=font,
                conf=conf,
                accurate_detector=accurate_detector,
                save_cbz=save_cbz,
                output_format=eff_format,
                sequential_names=eff_seq,
                vision_translate=eff_vision,
                retranslate_untranslated=retranslate_untranslated,
                skip_sfx=eff_skip_sfx,
                use_gpu=eff_gpu,
                resume=resume,
                clean_cache=eff_clean_cache,
                delete_source_zip=eff_delete_zip,
                max_workers=eff_workers,
                hooks=PipelineHooks(
                    on_step=on_step,
                    on_page_done=on_page_done,
                    on_story_bible=on_story_bible,
                    on_confirm_story_bible=(lambda b: _cli_confirm_story_bible(b, progress)) if review_chars else None,
                ),
            )
        except (FileNotFoundError, ValueError) as exc:
            console.bell()
            send_desktop_notification(
                err_notif_title,
                str(exc),
                is_error=True,
                enabled=eff_notify,
            )
            console.print(f"[red]{escape_markup(str(exc))}[/red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.bell()
            send_desktop_notification(
                err_notif_title,
                str(exc),
                is_error=True,
                enabled=eff_notify,
            )
            raise

    console.bell()
    send_desktop_notification(
        done_notif_title,
        f"Đã hoàn tất {len(docs)} trang -> {out_dir.name}",
        enabled=eff_notify,
    )
    out_esc = escape_markup(str(out_dir))
    console.print(f"[bold green]Hoàn tất ({len(docs)} trang)![/bold green] Ảnh kết quả lưu tại: {out_esc}")
    if (out_dir / ".koharu_cache").is_dir():
        cache_esc = escape_markup(str(out_dir / ".koharu_cache"))
        console.print(
            f"[dim]Mẹo: Chỉnh thoại trong {cache_esc}/*.koharu.json rồi chạy `koharu-vn rerender {out_esc}` (0 giây API).[/dim]"
        )
    else:
        console.print("[dim]Đã tự dọn sạch `.koharu_cache` sau khi hoàn tất 100%.[/dim]")


@app.command()
def translate(
    input_path: Path = typer.Argument(..., help="Ảnh, thư mục truyện, hoặc file .zip/.cbz cần dịch."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Thư mục lưu ảnh đã dịch."),
    lang: str = typer.Option("auto", "--lang", help="Ngôn ngữ gốc: auto | ja | ko | zh | en."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="GEMINI_API_KEY", help="Gemini API key (hoặc ~/.koharu-vn/config.json)."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint OpenAI-compatible tùy chọn (vd: http://127.0.0.1:8317/v1)."),
    model: Optional[str] = typer.Option(None, "--model", help="Model PHỤ (nhanh) để dịch từng trang hoặc sáng tác thoại."),
    fallback_model: Optional[str] = typer.Option(None, "--fallback-model", help="Model fallback cho PHỤ khi hết quota."),
    story_model: Optional[str] = typer.Option(None, "--story-model", help="Model CHÍNH (mạnh) cho Pass 1 phân tích full cốt truyện, nhân vật & tình huống."),
    story_fallback: Optional[str] = typer.Option(None, "--story-fallback", help="Model fallback cho CHÍNH khi bị prohibit (thử 1 lượt rồi bỏ qua Pass 1)."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Hướng dẫn ngữ cảnh/xưng hô thêm cho người dịch."),
    story: Optional[str] = typer.Option(None, "--story", help="Cốt truyện/nhân vật/lore (nhập chữ hoặc đường dẫn file .txt/.md)."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Tag thể loại (vd: 'vanilla, romance' hoặc 'NTR, angst')."),
    story_pass: bool = typer.Option(True, "--story-pass/--no-story-pass", help="Quét trước toàn bộ chương (Pass 1) để khóa nhân vật, xưng hô & cốt truyện."),
    font: Optional[str] = typer.Option(None, "--font", help="Font chữ Việt (mali-bold, pangolin, patrick-hand, bangers, hoặc đường dẫn .ttf)."),
    conf: float = typer.Option(0.34, "--conf", help="Ngưỡng confidence phát hiện khung thoại (0.2 - 0.8)."),
    accurate_detector: bool = typer.Option(True, "--accurate-detector/--fast-detector", help="Dùng model detector FP32 (168MB, chuẩn cao) hoặc --fast-detector INT8 (43MB)."),
    save_cbz: bool = typer.Option(False, "--save-cbz", help="Đóng gói thêm file .cbz sau khi dịch xong."),
    output_format: Optional[str] = typer.Option(None, "--format", help="Định dạng ảnh xuất: webp (max quality 100) | original | png | jpg."),
    seq_names: Optional[bool] = typer.Option(None, "--seq-names/--no-seq-names", help="Đổi tên ảnh xuất theo số thứ tự đọc (001, 002...)."),
    review_chars: bool = typer.Option(False, "--review-chars", help="Dừng sau Pass 1 để kiểm tra & sửa tên game/nhân vật trước khi dịch."),
    vision: Optional[bool] = typer.Option(None, "--vision/--no-vision", help="Pass 2 dùng Vision gửi ảnh (mặc định) hoặc --no-vision để OCR cục bộ + model con dịch text-only."),
    skip_sfx: Optional[bool] = typer.Option(None, "--skip-sfx/--keep-sfx", help="Bỏ qua không dịch & không inpaint vào SFX để tránh rối tranh (mặc định: bật)."),
    gpu: Optional[bool] = typer.Option(None, "--gpu/--cpu", help="Chạy Detector + LaMa Inpaint + OCR trên GPU (mặc định: tự động GPU CUDA/DirectML nếu có)."),
    clean_cache: Optional[bool] = typer.Option(None, "--clean-cache/--keep-cache", help="Tự động xóa thư mục .koharu_cache sau khi dịch xong 100% (mặc định: bật)."),
    delete_zip: Optional[bool] = typer.Option(None, "--delete-zip/--keep-zip", help="Tự động xóa file .zip gốc sau khi xác minh 100% ảnh xuất có chữ dịch mới (mặc định: bật)."),
    notify: Optional[bool] = typer.Option(None, "--notify/--no-notify", help="Gửi thông báo Windows khi dịch xong hoặc gặp lỗi (mặc định: bật)."),
    workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Số luồng API dịch song song (0 = tự động theo số API key)."),
    resume: bool = typer.Option(False, "--resume", help="Bỏ qua các trang đã dịch sẵn trong thư mục đích."),
) -> None:
    """Dịch truyện tranh (Nhật, Hàn, Trung, Anh -> Tiếng Việt) sửa triệt để lỗi cỡ chữ, viền & đè chữ."""
    _execute_translate_cli(
        input_path,
        output=output,
        lang=lang,
        api_key=api_key,
        base_url=base_url,
        model=model,
        fallback_model=fallback_model,
        story_model=story_model,
        story_fallback=story_fallback,
        prompt=prompt,
        story=story,
        tags=tags,
        story_pass=story_pass,
        font=font,
        conf=conf,
        accurate_detector=accurate_detector,
        save_cbz=save_cbz,
        output_format=output_format,
        seq_names=seq_names,
        review_chars=review_chars,
        vision=vision,
        skip_sfx=skip_sfx,
        gpu=gpu,
        clean_cache=clean_cache,
        delete_zip=delete_zip,
        notify=notify,
        workers=workers,
        resume=resume,
        retranslate_untranslated=False,
    )


@app.command()
def retranslate(
    input_path: Path = typer.Argument(..., help="Thư mục/ảnh/file .zip đã xuất tải từ Drive về (bỏ qua ảnh đã dịch, chỉ dịch lại ảnh chưa dịch)."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Thư mục lưu bộ ảnh đã sửa (mặc định: <tên>_fixed)."),
    lang: str = typer.Option("auto", "--lang", help="Ngôn ngữ gốc: auto | ja | ko | zh | en."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="GEMINI_API_KEY", help="API key Gemini / DeepSeek."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint OpenAI-compatible tùy chọn."),
    model: Optional[str] = typer.Option(None, "--model", help="Model dịch lại (vd: gemini-3.8-flash hoặc deepseek-chat)."),
    fallback_model: Optional[str] = typer.Option(None, "--fallback-model", help="Model fallback khi hết quota."),
    story_model: Optional[str] = typer.Option(None, "--story-model", help="Model phụ/chính fallback."),
    story_fallback: Optional[str] = typer.Option(None, "--story-fallback", help="Model fallback khi bị prohibit."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Hướng dẫn ngữ cảnh/xưng hô thêm."),
    story: Optional[str] = typer.Option(None, "--story", help="Cốt truyện/nhân vật/lore (chữ hoặc file .txt/.md)."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Tag thể loại (vd: 'vanilla, romance')."),
    story_pass: bool = typer.Option(False, "--story-pass/--no-story-pass", help="Có chạy Pass 1 Story Bible trước khi tái dịch không (mặc định: tắt để quét nhanh)."),
    font: Optional[str] = typer.Option(None, "--font", help="Font chữ Việt."),
    conf: float = typer.Option(0.34, "--conf", help="Ngưỡng confidence phát hiện khung thoại."),
    accurate_detector: bool = typer.Option(True, "--accurate-detector/--fast-detector", help="Dùng detector FP32 chuẩn cao hoặc INT8."),
    save_cbz: bool = typer.Option(False, "--save-cbz", help="Đóng gói thêm file .cbz sau khi hoàn tất."),
    output_format: Optional[str] = typer.Option(None, "--format", help="Định dạng ảnh xuất: webp | original | png | jpg."),
    seq_names: Optional[bool] = typer.Option(None, "--seq-names/--no-seq-names", help="Đổi tên ảnh xuất theo số thứ tự."),
    vision: Optional[bool] = typer.Option(None, "--vision/--no-vision", help="Dùng Vision hay OCR cục bộ (--no-vision)."),
    skip_sfx: Optional[bool] = typer.Option(None, "--skip-sfx/--keep-sfx", help="Bỏ qua SFX trên tranh."),
    gpu: Optional[bool] = typer.Option(None, "--gpu/--cpu", help="Chạy ONNX trên GPU/CPU."),
    clean_cache: Optional[bool] = typer.Option(None, "--clean-cache/--keep-cache", help="Tự động xóa .koharu_cache sau khi hoàn tất."),
    delete_zip: Optional[bool] = typer.Option(None, "--delete-zip/--keep-zip", help="Tự động xóa file .zip gốc sau khi xác minh 100% ảnh xuất có chữ dịch mới."),
    notify: Optional[bool] = typer.Option(None, "--notify/--no-notify", help="Gửi thông báo Windows khi xong/lỗi."),
    workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Số luồng API song song."),
) -> None:
    """Tái dịch ảnh/zip tải từ Google Drive: tự bỏ qua trang đã dịch Tiếng Việt, chỉ dịch + inpaint + chèn chữ vào trang bị sót."""
    _execute_translate_cli(
        input_path,
        output=output,
        lang=lang,
        api_key=api_key,
        base_url=base_url,
        model=model,
        fallback_model=fallback_model,
        story_model=story_model,
        story_fallback=story_fallback,
        prompt=prompt,
        story=story,
        tags=tags,
        story_pass=story_pass,
        font=font,
        conf=conf,
        accurate_detector=accurate_detector,
        save_cbz=save_cbz,
        output_format=output_format,
        seq_names=seq_names,
        review_chars=False,
        vision=vision,
        skip_sfx=skip_sfx,
        gpu=gpu,
        clean_cache=clean_cache,
        delete_zip=delete_zip,
        notify=notify,
        workers=workers,
        resume=False,
        retranslate_untranslated=True,
    )


@app.command()
def compose(
    input_path: Path = typer.Argument(..., help="Ảnh CG không lời, thư mục bộ ảnh CG, hoặc file .zip/.cbz."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Thư mục lưu ảnh CG đã ghép thoại."),
    style: ComposePreset = typer.Option(
        ComposePreset.CG_OVAL,
        "--style",
        help="Kiểu thoại: cg-oval (oval trắng/đen mờ + chữ thường) | manga-tail (có đuôi trỏ + oval + chữ thường) | mixed | floating-only.",
    ),
    density: str = typer.Option("medium", "--density", help="Mật độ thoại mỗi ảnh: low | medium | high."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Gợi ý cốt truyện, tên nhân vật, xưng hô, thể loại."),
    story: Optional[str] = typer.Option(None, "--story", help="Cốt truyện/nhân vật/lore (nhập chữ hoặc đường dẫn file .txt/.md)."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Tag thể loại sáng tác CG (vd: 'vanilla, sweet' hoặc 'NTR, tâm lý')."),
    story_pass: bool = typer.Option(True, "--story-pass/--no-story-pass", help="Quét trước toàn bộ bộ ảnh CG (Pass 1) để dựng mạch truyện & nhân vật."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="GEMINI_API_KEY", help="Gemini API key (hoặc ~/.koharu-vn/config.json)."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint OpenAI-compatible tùy chọn."),
    model: Optional[str] = typer.Option(None, "--model", help="Model PHỤ (nhanh) để dịch từng trang hoặc sáng tác thoại."),
    fallback_model: Optional[str] = typer.Option(None, "--fallback-model", help="Model fallback cho PHỤ khi hết quota."),
    story_model: Optional[str] = typer.Option(None, "--story-model", help="Model CHÍNH (mạnh) cho Pass 1 phân tích full cốt truyện, nhân vật & tình huống."),
    story_fallback: Optional[str] = typer.Option(None, "--story-fallback", help="Model fallback cho CHÍNH khi bị prohibit (thử 1 lượt rồi bỏ qua Pass 1)."),
    font: Optional[str] = typer.Option(None, "--font", help="Ghi đè font chữ Việt cho tất cả câu thoại."),
    pages: Optional[str] = typer.Option(None, "-p", "--pages", help="Chọn phạm vi trang compose (vd: '1-20', '21-41', '1-10, 15')."),
    save_cbz: bool = typer.Option(False, "--save-cbz", help="Đóng gói thêm file .cbz."),
    output_format: Optional[str] = typer.Option(None, "--format", help="Định dạng ảnh xuất: webp (max quality 100) | original | png | jpg."),
    seq_names: Optional[bool] = typer.Option(None, "--seq-names/--no-seq-names", help="Đổi tên ảnh xuất theo số thứ tự đọc (001, 002...)."),
    review_chars: bool = typer.Option(False, "--review-chars", help="Dừng sau Pass 1 để kiểm tra & sửa tên game/nhân vật trước khi sáng tác."),
    clean_cache: Optional[bool] = typer.Option(None, "--clean-cache/--keep-cache", help="Tự động xóa thư mục .koharu_cache sau khi sáng tác xong 100% (mặc định: bật)."),
    delete_zip: Optional[bool] = typer.Option(None, "--delete-zip/--keep-zip", help="Tự động xóa file .zip gốc sau khi sáng tác 100% ảnh (không xóa khi chỉ chọn --pages một phần)."),
    notify: Optional[bool] = typer.Option(None, "--notify/--no-notify", help="Gửi thông báo Windows khi sáng tác xong hoặc gặp lỗi (mặc định: bật)."),
    resume: bool = typer.Option(False, "--resume", help="Bỏ qua ảnh đã sáng tác sẵn trong thư mục đích."),
) -> None:
    """Tự sáng tác lời thoại Tiếng Việt cho ảnh Artist CG không lời (hỗ trợ cả Bubble Text & Chữ thường)."""
    cfg = load_config()
    eff_key, eff_keys, eff_url, eff_model = _resolve_auth(api_key, base_url, model)
    eff_format = output_format or cfg.output_format
    eff_seq = cfg.sequential_names if seq_names is None else seq_names
    eff_clean_cache = cfg.clean_cache if clean_cache is None else clean_cache
    eff_delete_zip = cfg.delete_source_zip if delete_zip is None else delete_zip
    eff_notify = cfg.notify if notify is None else notify
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("AI đang sáng tác thoại CG...", total=None)
        completed_pages = 0

        def on_step(idx: int, total: int, name: str, stage: str, detail: str) -> None:
            progress.update(
                task,
                total=max(1, total),
                description=f"[{idx}/{total}] {escape_markup(name)}: {escape_markup(detail)}",
            )

        def on_page_done(idx: int, total: int, name: str, doc, out_path: Path) -> None:
            nonlocal completed_pages
            completed_pages += 1
            progress.update(task, total=max(1, total), completed=completed_pages)

        def on_story_bible(bible) -> None:
            game_tag = f" [{escape_markup(bible.source_game)}]" if getattr(bible, "source_game", "") else ""
            progress.console.print(
                f"[cyan]✓ Pass 1 CG Story Bible{game_tag}:[/cyan] {escape_markup(bible.title or 'Đã phân tích')} "
                f"({len(bible.characters)} nhân vật) — [dim]{escape_markup(bible.full_plot_summary[:140])}[/dim]"
            )

        try:
            out_dir, docs = run_compose_pipeline(
                input_path,
                output=output,
                style=style,
                density=density,
                prompt=prompt,
                story=story,
                tags=tags if tags is not None else (cfg.tags or None),
                story_pass=story_pass,
                api_key=eff_key,
                api_keys=eff_keys,
                base_url=eff_url,
                model=eff_model,
                fallback_model=fallback_model if fallback_model is not None else (cfg.fallback_model or None),
                story_model=story_model or cfg.story_model or None,
                story_fallback_model=story_fallback if story_fallback is not None else (cfg.story_fallback_model or None),
                font=font,
                save_cbz=save_cbz,
                output_format=eff_format,
                sequential_names=eff_seq,
                page_range=pages,
                resume=resume,
                clean_cache=eff_clean_cache,
                delete_source_zip=eff_delete_zip,
                hooks=PipelineHooks(
                    on_step=on_step,
                    on_page_done=on_page_done,
                    on_story_bible=on_story_bible,
                    on_confirm_story_bible=(lambda b: _cli_confirm_story_bible(b, progress)) if review_chars else None,
                ),
            )
        except (FileNotFoundError, ValueError) as exc:
            console.bell()
            send_desktop_notification(
                "Koharu-VN · Lỗi sáng tác CG",
                str(exc),
                is_error=True,
                enabled=eff_notify,
            )
            console.print(f"[red]{escape_markup(str(exc))}[/red]")
            raise typer.Exit(1)
        except Exception as exc:
            console.bell()
            send_desktop_notification(
                "Koharu-VN · Lỗi sáng tác CG",
                str(exc),
                is_error=True,
                enabled=eff_notify,
            )
            raise

    console.bell()
    send_desktop_notification(
        "Koharu-VN · Sáng tác CG xong!",
        f"Đã hoàn tất {len(docs)} ảnh CG -> {out_dir.name}",
        enabled=eff_notify,
    )
    console.print(f"[bold green]Hoàn tất sáng tác ({len(docs)} ảnh)![/bold green] Lưu tại: {escape_markup(str(out_dir))}")


@app.command()
def rerender(
    output_dir: Path = typer.Argument(..., help="Thư mục output chứa .koharu_cache/*.koharu.json."),
    font: Optional[str] = typer.Option(None, "--font", help="Đổi font cho toàn bộ các trang."),
    font_scale: float = typer.Option(1.0, "--font-scale", help="Tỉ lệ phóng/thu cỡ chữ (vd: 1.15)."),
    output_format: str = typer.Option("original", "--format", help="Định dạng ảnh xuất: original | webp | png | jpg."),
    seq_names: bool = typer.Option(False, "--seq-names/--no-seq-names", help="Đổi tên ảnh xuất theo số thứ tự đọc (001, 002...)."),
) -> None:
    """Render lại nhanh toàn bộ ảnh từ file .koharu.json mà không tốn API hay chạy lại ML."""
    try:
        docs = run_rerender_pipeline(
            output_dir,
            font=font,
            font_scale=font_scale,
            output_format=output_format,
            sequential_names=seq_names,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{escape_markup(str(exc))}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Đã render lại {len(docs)} trang tại {escape_markup(str(output_dir))}![/bold green]")


@app.command()
def rename(
    input_path: Path = typer.Argument(..., help="Thư mục ảnh hoặc .zip/.cbz có tên lộn xộn nhưng đúng thứ tự đọc."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Thư mục lưu ảnh WebP mới (mặc định <tên>_webp)."),
    in_place: bool = typer.Option(False, "--in-place", help="Đổi tên & chuyển WebP trực tiếp ngay trong thư mục gốc."),
) -> None:
    """Đổi tên ảnh theo số thứ tự đọc (001.webp, 002.webp...) và chuyển sang WebP max quality (100)."""
    try:
        out_dir, files = pack_images_to_webp(input_path, output=output, in_place=in_place)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{escape_markup(str(exc))}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[bold green]Đã chuyển {len(files)} ảnh sang WebP (quality=100) & đánh số 001..{len(files):03d}![/bold green] Lưu tại: {escape_markup(str(out_dir))}"
    )


@app.command()
def models(
    accurate: bool = typer.Option(False, "--accurate-detector", help="Tải thêm bản detector FP32 (168MB)."),
    remote: bool = typer.Option(False, "--remote", help="Gọi API lấy danh sách model Gemini mới nhất hiện khả dụng."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="GEMINI_API_KEY", help="Gemini API key tùy chọn cho --remote."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint OpenAI-compatible tùy chọn cho --remote."),
) -> None:
    """Kiểm tra & tải trước các model ONNX, font Tiếng Việt và (tùy chọn) lấy model Gemini mới nhất từ API."""
    console.print("[cyan]Đang kiểm tra / tải model ONNX vào cache (~/.cache)...[/cyan]")
    ModelCache().warmup_all(accurate=accurate)

    cfg = load_config()
    gem_keys = ([api_key] if api_key else []) or cfg.effective_api_keys(model="gemini-3.8-flash", base_url=base_url or cfg.base_url or "")
    ds_keys = cfg.effective_api_keys(model="deepseek-chat", base_url=DEEPSEEK_BASE_URL)
    eff_key = gem_keys[0] if gem_keys else None
    eff_url = base_url or cfg.base_url

    table = Table(title="Tài nguyên Koharu-VN CLI đã sẵn sàng")
    table.add_column("Thành phần", style="cyan")
    table.add_column("Chi tiết", style="green")
    table.add_row("Detector ONNX", "ogkalu/comic-text-and-bubble-detector (RT-DETRv2) + mayocream/comic-text-detector")
    table.add_row("Text Mask ONNX", "mayocream/comic-text-detector (YOLOv5 blk + UNet seg + DBNet det)")
    table.add_row("Inpainter ONNX", "mayocream/lama-manga-onnx (anime-manga-big-lama)")
    table.add_row("Font Tiếng Việt", ", ".join(available_fonts()))
    table.add_row(
        "GEMINI_API_KEY",
        f"Đã thiết lập ({cfg.masked_api_key(model='gemini-3.8-flash', base_url=eff_url or '')})"
        if eff_key
        else "Chưa đặt (chạy `koharu-vn` -> `/key` hoặc đặt env GEMINI_API_KEY)",
    )
    table.add_row(
        "DEEPSEEK_API_KEY",
        f"Đã thiết lập ({cfg.masked_deepseek_key()})"
        if ds_keys
        else "Chưa đặt (chạy `koharu-vn` -> `/key` hoặc đặt env DEEPSEEK_API_KEY)",
    )
    console.print(table)

    if remote:
        if not eff_key and not ds_keys and not eff_url:
            console.print("[yellow]Cần có GEMINI_API_KEY, DEEPSEEK_API_KEY hoặc --base-url để lấy danh sách model từ API.[/yellow]")
            return
        remote_models: list[tuple[str, str]] = []
        if eff_key or eff_url:
            try:
                client = VisionLLMClient(api_key=eff_key, model="gemini-3.8-flash", base_url=eff_url)
                remote_models.extend(client.list_available_models(limit=20))
            except Exception as exc:
                console.print(f"[yellow]Không lấy được model Gemini/Proxy: {exc}[/yellow]")
        if ds_keys:
            try:
                ds_client = VisionLLMClient(api_key=ds_keys[0], model="deepseek-chat", base_url=DEEPSEEK_BASE_URL)
                remote_models.extend(ds_client.list_available_models(limit=20))
            except Exception as exc:
                console.print(f"[yellow]Không lấy được model DeepSeek: {exc}[/yellow]")
        if remote_models:
            mt = Table(title="Model Gemini / DeepSeek mới nhất từ API")
            mt.add_column("Model ID", style="bold cyan")
            mt.add_column("Mô tả", style="white")
            for mid, desc in remote_models:
                mt.add_row(mid, desc)
            console.print(mt)


@app.command()
def queue(
    paths: list[Path] = typer.Argument(..., help="Danh sách nhiều thư mục hoặc file .zip/.cbz cần dịch liên tiếp."),
    retranslate: bool = typer.Option(False, "--retranslate", help="Dùng chế độ tái dịch (bỏ qua ảnh đã có chữ Việt, chỉ dịch ảnh lỗi/chưa dịch)."),
    lang: str = typer.Option("auto", "--lang", help="Ngôn ngữ gốc: auto | ja | ko | zh | en."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="GEMINI_API_KEY", help="API key Gemini / DeepSeek."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint OpenAI-compatible tùy chọn."),
    model: Optional[str] = typer.Option(None, "--model", help="Model PHỤ (nhanh) dịch từng trang."),
    fallback_model: Optional[str] = typer.Option(None, "--fallback-model", help="Model fallback cho PHỤ khi hết quota."),
    story_model: Optional[str] = typer.Option(None, "--story-model", help="Model CHÍNH (mạnh) cho Pass 1."),
    story_fallback: Optional[str] = typer.Option(None, "--story-fallback", help="Model fallback cho CHÍNH khi bị prohibit."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Hướng dẫn ngữ cảnh/xưng hô chung cho cả hàng chờ."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Tag thể loại chung cho hàng chờ."),
    story_pass: bool = typer.Option(True, "--story-pass/--no-story-pass", help="Quét cốt truyện Pass 1 trước khi dịch."),
    font: Optional[str] = typer.Option(None, "--font", help="Font chữ Việt."),
    conf: float = typer.Option(0.34, "--conf", help="Ngưỡng confidence phát hiện khung thoại (0.2 - 0.8)."),
    accurate_detector: bool = typer.Option(True, "--accurate-detector/--fast-detector", help="Chọn model detector FP32 hoặc INT8."),
    save_cbz: bool = typer.Option(False, "--save-cbz", help="Đóng gói thêm file .cbz sau mỗi tác phẩm."),
    output_format: Optional[str] = typer.Option(None, "--format", help="Định dạng ảnh xuất: webp | original | png | jpg."),
    seq_names: Optional[bool] = typer.Option(None, "--seq-names/--no-seq-names", help="Đổi tên ảnh xuất theo số thứ tự."),
    vision: Optional[bool] = typer.Option(None, "--vision/--no-vision", help="Dùng Vision hay OCR cục bộ + model text."),
    skip_sfx: Optional[bool] = typer.Option(None, "--skip-sfx/--keep-sfx", help="Bỏ qua không dịch & không inpaint vào SFX."),
    gpu: Optional[bool] = typer.Option(None, "--gpu/--cpu", help="Chạy ONNX/OCR trên GPU hay CPU."),
    clean_cache: Optional[bool] = typer.Option(None, "--clean-cache/--keep-cache", help="Tự động xóa .koharu_cache sau mỗi tác phẩm."),
    delete_zip: Optional[bool] = typer.Option(None, "--delete-zip/--keep-zip", help="Tự động xóa file .zip gốc sau khi xác minh 100% ảnh xuất có chữ dịch mới."),
    notify: Optional[bool] = typer.Option(None, "--notify/--no-notify", help="Gửi thông báo Windows sau mỗi tác phẩm."),
    workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Số luồng API dịch song song."),
    resume: bool = typer.Option(False, "--resume", help="Bỏ qua các trang đã dịch sẵn trong thư mục đích."),
) -> None:
    """Dịch liên tiếp nhiều tác phẩm trong hàng chờ từ đầu đến cuối."""
    valid_paths = [p for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]
    for m in missing:
        console.print(f"[yellow]Bỏ qua đường dẫn không tồn tại: {escape_markup(str(m))}[/yellow]")
    if not valid_paths:
        console.print("[red]Không có tác phẩm hợp lệ nào trong hàng chờ.[/red]")
        raise typer.Exit(code=1)

    total = len(valid_paths)
    console.print(f"[bold cyan]Hàng chờ gồm {total} tác phẩm — sẽ chạy tuần tự từng bộ.[/bold cyan]")
    for idx, p in enumerate(valid_paths, start=1):
        console.print(f"\n[bold magenta]>>> Hàng chờ [{idx}/{total}]: {escape_markup(p.name)}[/bold magenta]")
        _execute_translate_cli(
            p,
            output=None,
            lang=lang,
            api_key=api_key,
            base_url=base_url,
            model=model,
            fallback_model=fallback_model,
            story_model=story_model,
            story_fallback=story_fallback,
            prompt=prompt,
            story=None,
            tags=tags,
            story_pass=False if retranslate else story_pass,
            font=font,
            conf=conf,
            accurate_detector=accurate_detector,
            save_cbz=save_cbz,
            output_format=output_format,
            seq_names=seq_names,
            review_chars=False,
            vision=vision,
            skip_sfx=skip_sfx,
            gpu=gpu,
            clean_cache=clean_cache,
            delete_zip=delete_zip,
            notify=notify,
            workers=workers,
            resume=resume,
            retranslate_untranslated=retranslate,
        )
    console.print(f"\n[bold green]✓ Đã hoàn tất toàn bộ {total}/{total} tác phẩm trong hàng chờ![/bold green]")


_SUBCOMMANDS = {"translate", "retranslate", "queue", "compose", "rerender", "rename", "models", "ui"}


def _clean_cli_drop_arg(raw: str) -> str:
    s = raw.strip()
    if s.startswith("& "):
        s = s[2:].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        s = s[1:-1].strip()
    if s.lower().startswith("file:///"):
        s = unquote(s[8:])
    return s


def main() -> None:
    if len(sys.argv) >= 2:
        cand = _clean_cli_drop_arg(sys.argv[1])
        if (
            cand
            and not cand.startswith("-")
            and cand.lower() not in _SUBCOMMANDS
            and (Path(cand).exists() or cand.lower().endswith((".zip", ".cbz")))
        ):
            sys.argv = [sys.argv[0], "translate", cand, *sys.argv[2:]]
    app()


if __name__ == "__main__":
    main()
