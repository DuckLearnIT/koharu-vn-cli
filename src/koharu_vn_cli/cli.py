"""Koharu-VN CLI: OpenCode/Claude-Code style TUI + headless manga/CG commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from koharu_vn_cli.config import load_config
from koharu_vn_cli.models import ComposePreset
from koharu_vn_cli.pipeline.fonts import available_fonts
from koharu_vn_cli.pipeline.gemini_client import DEFAULT_GEMINI_MODEL, VisionLLMClient
from koharu_vn_cli.pipeline.runner import (
    ModelCache,
    PipelineHooks,
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
    cli_keys = [k.strip() for k in (api_key or "").replace(";", ",").split(",") if k.strip()] if api_key else []
    eff_keys = cli_keys or cfg.effective_api_keys()
    eff_key = eff_keys[0] if eff_keys else None
    eff_url = base_url or cfg.base_url
    eff_model = model or cfg.model or DEFAULT_GEMINI_MODEL
    return eff_key, eff_keys, eff_url, eff_model


@app.command()
def translate(
    input_path: Path = typer.Argument(..., help="Ảnh, thư mục truyện, hoặc file .zip/.cbz cần dịch."),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Thư mục lưu ảnh đã dịch."),
    lang: str = typer.Option("auto", "--lang", help="Ngôn ngữ gốc: auto | ja | ko | zh | en."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="GEMINI_API_KEY", help="Gemini API key (hoặc ~/.koharu-vn/config.json)."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Endpoint OpenAI-compatible tùy chọn (vd: http://127.0.0.1:8317/v1)."),
    model: Optional[str] = typer.Option(None, "--model", help="Model PHỤ (nhanh) để dịch từng trang hoặc sáng tác thoại."),
    story_model: Optional[str] = typer.Option(None, "--story-model", help="Model CHÍNH (mạnh) cho Pass 1 phân tích full cốt truyện, nhân vật & tình huống."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Hướng dẫn ngữ cảnh/xưng hô thêm cho người dịch."),
    story: Optional[str] = typer.Option(None, "--story", help="Cốt truyện/nhân vật/lore (nhập chữ hoặc đường dẫn file .txt/.md)."),
    tags: Optional[str] = typer.Option(None, "--tags", help="Tag thể loại (vd: 'vanilla, romance' hoặc 'NTR, angst')."),
    story_pass: bool = typer.Option(True, "--story-pass/--no-story-pass", help="Quét trước toàn bộ chương (Pass 1) để khóa nhân vật, xưng hô & cốt truyện."),
    font: Optional[str] = typer.Option(None, "--font", help="Font chữ Việt (mali-bold, pangolin, patrick-hand, bangers, hoặc đường dẫn .ttf)."),
    conf: float = typer.Option(0.34, "--conf", help="Ngưỡng confidence phát hiện khung thoại (0.2 - 0.8)."),
    accurate_detector: bool = typer.Option(True, "--accurate-detector/--fast-detector", help="Dùng model detector FP32 (168MB, chuẩn cao) hoặc --fast-detector INT8 (43MB)."),
    save_cbz: bool = typer.Option(False, "--save-cbz", help="Đóng gói thêm file .cbz sau khi dịch xong."),
    resume: bool = typer.Option(False, "--resume", help="Bỏ qua các trang đã dịch sẵn trong thư mục đích."),
) -> None:
    """Dịch truyện tranh (Nhật, Hàn, Trung, Anh -> Tiếng Việt) sửa triệt để lỗi cỡ chữ, viền & đè chữ."""
    eff_key, eff_keys, eff_url, eff_model = _resolve_auth(api_key, base_url, model)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Đang dịch truyện...", total=None)

        def on_step(idx: int, total: int, name: str, stage: str, detail: str) -> None:
            progress.update(task, total=max(1, total), description=f"[{idx}/{total}] {name}: {detail}")

        def on_page_done(idx: int, total: int, name: str, doc, out_path: Path) -> None:
            progress.update(task, completed=idx)

        def on_story_bible(bible) -> None:
            progress.console.print(
                f"[cyan]✓ Pass 1 Story Bible:[/cyan] {bible.title or 'Đã phân tích'} "
                f"({len(bible.characters)} nhân vật) — [dim]{bible.full_plot_summary[:140]}[/dim]"
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
                story_model=story_model or load_config().story_model or None,
                prompt=prompt,
                story=story,
                tags=tags if tags is not None else (load_config().tags or None),
                story_pass=story_pass,
                font=font,
                conf=conf,
                accurate_detector=accurate_detector,
                save_cbz=save_cbz,
                resume=resume,
                hooks=PipelineHooks(on_step=on_step, on_page_done=on_page_done, on_story_bible=on_story_bible),
            )
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold green]Hoàn tất ({len(docs)} trang)![/bold green] Ảnh kết quả lưu tại: {out_dir}")
    console.print(f"[dim]Mẹo: Chỉnh thoại trong {out_dir / '.koharu_cache'}/*.koharu.json rồi chạy `koharu-vn rerender {out_dir}` (0 giây API).[/dim]")


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
    story_model: Optional[str] = typer.Option(None, "--story-model", help="Model CHÍNH (mạnh) cho Pass 1 phân tích full cốt truyện, nhân vật & tình huống."),
    font: Optional[str] = typer.Option(None, "--font", help="Ghi đè font chữ Việt cho tất cả câu thoại."),
    save_cbz: bool = typer.Option(False, "--save-cbz", help="Đóng gói thêm file .cbz."),
    resume: bool = typer.Option(False, "--resume", help="Bỏ qua ảnh đã sáng tác sẵn trong thư mục đích."),
) -> None:
    """Tự sáng tác lời thoại Tiếng Việt cho ảnh Artist CG không lời (hỗ trợ cả Bubble Text & Chữ thường)."""
    eff_key, eff_keys, eff_url, eff_model = _resolve_auth(api_key, base_url, model)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("AI đang sáng tác thoại CG...", total=None)

        def on_step(idx: int, total: int, name: str, stage: str, detail: str) -> None:
            progress.update(task, total=max(1, total), description=f"[{idx}/{total}] {name}: {detail}")

        def on_page_done(idx: int, total: int, name: str, doc, out_path: Path) -> None:
            progress.update(task, completed=idx)

        def on_story_bible(bible) -> None:
            progress.console.print(
                f"[cyan]✓ Pass 1 CG Story Bible:[/cyan] {bible.title or 'Đã phân tích'} "
                f"({len(bible.characters)} nhân vật) — [dim]{bible.full_plot_summary[:140]}[/dim]"
            )

        try:
            out_dir, docs = run_compose_pipeline(
                input_path,
                output=output,
                style=style,
                density=density,
                prompt=prompt,
                story=story,
                tags=tags if tags is not None else (load_config().tags or None),
                story_pass=story_pass,
                api_key=eff_key,
                api_keys=eff_keys,
                base_url=eff_url,
                model=eff_model,
                story_model=story_model or load_config().story_model or None,
                font=font,
                save_cbz=save_cbz,
                resume=resume,
                hooks=PipelineHooks(on_step=on_step, on_page_done=on_page_done, on_story_bible=on_story_bible),
            )
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[bold green]Hoàn tất sáng tác ({len(docs)} ảnh)![/bold green] Lưu tại: {out_dir}")


@app.command()
def rerender(
    output_dir: Path = typer.Argument(..., help="Thư mục output chứa .koharu_cache/*.koharu.json."),
    font: Optional[str] = typer.Option(None, "--font", help="Đổi font cho toàn bộ các trang."),
    font_scale: float = typer.Option(1.0, "--font-scale", help="Tỉ lệ phóng/thu cỡ chữ (vd: 1.15)."),
) -> None:
    """Render lại nhanh toàn bộ ảnh từ file .koharu.json mà không tốn API hay chạy lại ML."""
    try:
        docs = run_rerender_pipeline(output_dir, font=font, font_scale=font_scale)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Đã render lại {len(docs)} trang tại {output_dir}![/bold green]")


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
    eff_key = api_key or cfg.effective_api_key() or None
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
        f"Đã thiết lập ({cfg.masked_api_key()})"
        if eff_key
        else "Chưa đặt (chạy `koharu-vn` -> `/key` hoặc đặt env GEMINI_API_KEY)",
    )
    console.print(table)

    if remote:
        if not eff_key and not eff_url:
            console.print("[yellow]Cần có GEMINI_API_KEY hoặc --base-url để lấy danh sách model từ API.[/yellow]")
            return
        try:
            client = VisionLLMClient(api_key=eff_key, base_url=eff_url)
            remote_models = client.list_available_models(limit=20)
            mt = Table(title="Model Gemini / Vision mới nhất từ API")
            mt.add_column("Model ID", style="bold cyan")
            mt.add_column("Mô tả", style="white")
            for mid, desc in remote_models:
                mt.add_row(mid, desc)
            console.print(mt)
        except Exception as exc:
            console.print(f"[red]Không lấy được danh sách model từ API: {exc}[/red]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
