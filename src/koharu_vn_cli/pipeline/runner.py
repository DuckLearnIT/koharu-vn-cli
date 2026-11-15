"""Shared execution pipelines for translate, compose, and rerender across CLI and TUI."""

from __future__ import annotations

import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from koharu_vn_cli.models import ComposePreset, PageDocument, StoryBible, TextRegionItem
from koharu_vn_cli.pipeline.bubble_renderer import render_bubbles, resolve_box_overlaps
from koharu_vn_cli.pipeline.detector import ComicDetector, _ensure_ctd_model, active_ort_provider_label
from koharu_vn_cli.pipeline.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    ContentBlockedError,
    VisionLLMClient,
    normalize_source_key,
    region_needs_translation,
)
from koharu_vn_cli.pipeline.inpaint import (
    HybridInpainter,
    build_erasure_mask,
    classify_region_colors,
)
from koharu_vn_cli.pipeline.ocr import MangaOcrONNX
from koharu_vn_cli.pipeline.scene_cluster import SceneGroup, cluster_story_scenes
from koharu_vn_cli.pipeline.typesetter import (
    adapt_regions_for_readability,
    render_text_regions,
    solve_layout,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class PipelineCancelledError(RuntimeError):
    """Raised when a user cancels an active TUI/CLI job cleanly."""


@dataclass
class PipelineHooks:
    """Optional progress and inspection callbacks used by Rich CLI or Textual TUI."""

    on_step: Optional[Callable[[int, int, str, str, str], None]] = None
    on_page_done: Optional[Callable[[int, int, str, PageDocument, Path], None]] = None
    on_story_bible: Optional[Callable[[StoryBible], None]] = None
    on_confirm_story_bible: Optional[Callable[[StoryBible], StoryBible | None]] = None
    is_cancelled: Optional[Callable[[], bool]] = None

    def emit_step(self, idx: int, total: int, name: str, stage: str, detail: str = "") -> None:
        if self.on_step:
            self.on_step(idx, total, name, stage, detail)

    def emit_page_done(
        self, idx: int, total: int, name: str, doc: PageDocument, out_path: Path
    ) -> None:
        if self.on_page_done:
            self.on_page_done(idx, total, name, doc, out_path)

    def emit_story_bible(self, bible: StoryBible) -> None:
        if self.on_story_bible:
            self.on_story_bible(bible)

    def confirm_story_bible(self, bible: StoryBible) -> StoryBible:
        if self.on_confirm_story_bible:
            updated = self.on_confirm_story_bible(bible)
            if updated is not None:
                return updated
        return bible

    def check_cancelled(self) -> None:
        if self.is_cancelled and self.is_cancelled():
            raise PipelineCancelledError("Người dùng đã hủy tác vụ.")


class ModelCache:
    """Keep loaded ONNX sessions warm across pages and interactive TUI jobs."""

    def __init__(self) -> None:
        self._detector: ComicDetector | None = None
        self._detector_key: tuple[bool, bool] | None = None
        self._inpainter: HybridInpainter | None = None
        self._inpainter_gpu: bool | None = None
        self._ocr: MangaOcrONNX | None = None
        self._ocr_gpu: bool | None = None

    def get_detector(self, accurate: bool = True, use_gpu: bool = True) -> ComicDetector:
        key = (accurate, use_gpu)
        if self._detector is None or self._detector_key != key:
            self._detector = ComicDetector(accurate=accurate, use_gpu=use_gpu)
            self._detector_key = key
        return self._detector

    def get_inpainter(self, use_gpu: bool = True) -> HybridInpainter:
        if self._inpainter is None or self._inpainter_gpu != use_gpu:
            self._inpainter = HybridInpainter(use_gpu=use_gpu)
            self._inpainter_gpu = use_gpu
        return self._inpainter

    def get_ocr(self, use_gpu: bool = True) -> MangaOcrONNX:
        if self._ocr is None or self._ocr_gpu != use_gpu:
            self._ocr = MangaOcrONNX(use_gpu=use_gpu)
            self._ocr_gpu = use_gpu
        return self._ocr

    def warmup_all(self, accurate: bool = True, use_gpu: bool = True) -> tuple[ComicDetector, HybridInpainter]:
        det = self.get_detector(accurate=accurate, use_gpu=use_gpu)
        _ensure_ctd_model()
        inp = self.get_inpainter(use_gpu=use_gpu)
        return det, inp


_EMBEDDED_PAGE_RE = re.compile(
    r"^file_\d+_(\d+)_(?:webp|png|jpe?g|bmp|gif)(?:_|$)", re.IGNORECASE
)


def _detect_image_ext(path: Path) -> str | None:
    """Detect image format from suffix or magic bytes (handles extensionless scraped/cached files)."""
    suf = path.suffix.lower()
    if suf in IMAGE_EXTS:
        return suf
    if suf in {".zip", ".cbz", ".json", ".txt", ".md", ".pdf"}:
        return None
    try:
        with path.open("rb") as fh:
            hdr = fh.read(16)
    except OSError:
        return None
    if hdr.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if hdr.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(hdr) >= 12 and hdr[:4] == b"RIFF" and hdr[8:12] == b"WEBP":
        return ".webp"
    if hdr.startswith(b"BM"):
        return ".bmp"
    if hdr[:6] in (b"GIF87a", b"GIF89a"):
        return ".png"
    return None


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and _detect_image_ext(path) is not None


def _resolve_output_ext(path: Path, output_format: str = "original") -> str:
    fmt = (output_format or "original").strip().lower().lstrip(".")
    if fmt == "webp":
        return ".webp"
    if fmt in {"jpg", "jpeg"}:
        return ".jpg"
    if fmt == "png":
        return ".png"
    if path.suffix.lower() in IMAGE_EXTS:
        return path.suffix.lower()
    return _detect_image_ext(path) or ".png"


def _output_image_name(
    path: Path,
    *,
    output_format: str = "original",
    seq_index: int | None = None,
    seq_pad: int = 3,
) -> str:
    ext = _resolve_output_ext(path, output_format=output_format)
    if seq_index is not None:
        return f"{seq_index:0{seq_pad}d}{ext}"
    if path.suffix.lower() in IMAGE_EXTS and output_format in ("original", "", None):
        return path.name
    stem = path.stem if path.suffix.lower() in IMAGE_EXTS else path.name
    return f"{stem}{ext}"


def natural_key(path: Path) -> tuple[int, int, list[int | str]]:
    parts: list[int | str] = [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]
    m = _EMBEDDED_PAGE_RE.match(path.name)
    if m:
        return (0, int(m.group(1)), parts)
    return (1, 0, parts)


def collect_inputs(input_path: Path, extract_dir: Path | None) -> tuple[list[Path], str]:
    """Collect naturally sorted image paths from a single image, folder, or .zip/.cbz archive."""
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() in {".zip", ".cbz"}:
            if extract_dir is None:
                raise ValueError("extract_dir is required for archive input")
            root_resolved = extract_dir.resolve()
            with zipfile.ZipFile(input_path, "r") as zf:
                for member in zf.infolist():
                    target = (root_resolved / member.filename).resolve()
                    if not target.is_relative_to(root_resolved):
                        continue
                    zf.extract(member, extract_dir)
            imgs = sorted(
                (p for p in extract_dir.rglob("*") if _is_supported_image(p)),
                key=natural_key,
            )
            return imgs, input_path.stem
        if _is_supported_image(input_path):
            return [input_path], input_path.stem
        raise ValueError(f"Định dạng file không hỗ trợ: {input_path.suffix or '(không có đuôi file)'}")

    imgs = sorted(
        (p for p in input_path.iterdir() if _is_supported_image(p)),
        key=natural_key,
    )
    if not imgs:
        imgs = sorted(
            (p for p in input_path.rglob("*") if _is_supported_image(p)),
            key=natural_key,
        )
    return imgs, input_path.name


def load_rgb(path: Path) -> np.ndarray:
    """Load image safely using Pillow (supports Unicode/Vietnamese filenames on Windows)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def save_image(pil_img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        pil_img.convert("RGB").save(out_path, format="JPEG", quality=96)
    elif ext == ".webp":
        pil_img.convert("RGB").save(out_path, format="WEBP", quality=100, method=6)
    else:
        pil_img.convert("RGB").save(out_path, format="PNG")


def render_document(base_rgb: np.ndarray, doc: PageDocument) -> Image.Image:
    canvas = Image.fromarray(base_rgb).convert("RGBA")
    canvas = render_bubbles(canvas, doc.regions)
    canvas = render_text_regions(canvas, doc.regions)
    return canvas


def package_cbz(output_dir: Path, cbz_path: Path) -> None:
    imgs = sorted(
        (p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )
    with zipfile.ZipFile(cbz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img in imgs:
            zf.write(img, arcname=img.name)


def expand_composed_boxes_if_tight(
    regions: list[TextRegionItem], img_w: int, img_h: int
) -> list[TextRegionItem]:
    """Ensure compose bubbles & floating boxes are comfortably large and readable on mobile/desktop."""
    min_dim = min(img_w, img_h)
    bubble_target_font = max(18, int(round(min_dim * 0.024)))
    floating_target_font = max(16, int(round(min_dim * 0.020)))
    max_bw = max(200, int(round(img_w * 0.36)))
    max_bh = max(160, int(round(img_h * 0.36)))

    adjusted: list[TextRegionItem] = []
    for reg in regions:
        copy = reg.model_copy()
        text_len = len((copy.translated_text or copy.source_text or "").strip())
        is_bubble = copy.kind == "bubble"
        target_font = bubble_target_font if is_bubble else floating_target_font

        if is_bubble:
            if text_len <= 12:
                min_bw = max(112, int(round(img_w * 0.095)))
                min_bh = max(86, int(round(img_h * 0.095)))
            else:
                min_bw = max(150, int(round(img_w * 0.125)))
                min_bh = max(110, int(round(img_h * 0.12)))
        else:
            min_bw, min_bh = 48, 28

        bx1, by1, bx2, by2 = copy.bubble_bbox or copy.text_bbox
        bw = max(min_bw, bx2 - bx1)
        bh = max(min_bh, by2 - by1)
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

        def _clamp_centered(cur_w: int, cur_h: int) -> tuple[int, int, int, int]:
            cw = min(max_bw, max(24, cur_w))
            ch = min(max_bh, max(20, cur_h))
            nx1 = max(8, min(img_w - 8 - cw, cx - cw // 2))
            ny1 = max(8, min(img_h - 8 - ch, cy - ch // 2))
            return (nx1, ny1, min(img_w - 8, nx1 + cw), min(img_h - 8, ny1 + ch))

        box = _clamp_centered(bw, bh)
        copy.text_bbox = box
        if is_bubble:
            copy.bubble_bbox = box

        for _ in range(5):
            layout = solve_layout(copy)
            if layout.font_size >= target_font and not layout.overflowed:
                break
            scale = 1.24 if layout.overflowed else 1.18
            nbw = int(round((copy.text_bbox[2] - copy.text_bbox[0]) * scale))
            nbh = int(round((copy.text_bbox[3] - copy.text_bbox[1]) * scale))
            if nbw >= max_bw and nbh >= max_bh:
                break
            box = _clamp_centered(nbw, nbh)
            copy.text_bbox = box
            if is_bubble:
                copy.bubble_bbox = box

        adjusted.append(copy)
    return resolve_box_overlaps(adjusted, img_w, img_h, gap=max(10, int(min_dim * 0.012)))


def resolve_story_text(story: str | None) -> str | None:
    """Resolve user `--story` argument as a .txt/.md file content if valid path, else plain text."""
    if not story:
        return None
    s = story.strip()
    cand = Path(s).expanduser()
    if cand.is_file() and cand.suffix.lower() in {".txt", ".md", ".json"}:
        return cand.read_text(encoding="utf-8", errors="ignore").strip()
    return s


def _load_or_build_story_bible(
    llm: VisionLLMClient,
    images: list[Path],
    cache_dir: Path,
    base_name: str,
    *,
    mode: str,
    user_story: str | None,
    prompt: str | None,
    tags: str | None = None,
    enable_story_pass: bool,
    resume: bool,
    scenes: list[SceneGroup] | None = None,
    hooks: PipelineHooks,
) -> StoryBible | None:
    """Shared Pass-1 global story & character analysis across translate and compose pipelines."""
    bible_path = cache_dir / "story_bible.json"
    if resume and bible_path.is_file():
        try:
            bible = StoryBible.model_validate_json(bible_path.read_text(encoding="utf-8"))
            hooks.emit_story_bible(bible)
            return bible
        except Exception:
            pass

    resolved_story = resolve_story_text(user_story)
    if not enable_story_pass:
        return StoryBible(full_plot_summary=resolved_story) if resolved_story else None

    hooks.emit_step(
        0,
        len(images),
        base_name,
        "story_pass",
        f"Pass 1 [Model chính: {llm.model}]: Tự động xem full {len(images)} ảnh -> phân tích cốt truyện, nhân vật & xưng hô...",
    )
    try:
        extra_kw = {"scenes": scenes} if scenes and len(scenes) >= 2 else {}
        bible = llm.build_story_bible(
            images,
            mode=mode,
            user_story=resolved_story,
            extra_instructions=prompt,
            tags=tags,
            **extra_kw,
        )
    except ContentBlockedError as exc:
        hooks.emit_step(
            0,
            len(images),
            base_name,
            "warn",
            f"Pass 1 bị bộ lọc nội dung chặn ({exc}) — tiếp tục dịch từng trang...",
        )
        return StoryBible(full_plot_summary=resolved_story) if resolved_story else None
    hooks.emit_story_bible(bible)
    confirmed = hooks.confirm_story_bible(bible)
    if confirmed is not bible:
        bible = confirmed
        hooks.emit_story_bible(bible)
    bible_path.write_text(bible.model_dump_json(indent=2), encoding="utf-8")
    return bible


def run_translate_pipeline(
    input_path: Path,
    *,
    output: Path | None = None,
    lang: str = "auto",
    api_key: str | None = None,
    api_keys: list[str] | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    story_model: str | None = None,
    prompt: str | None = None,
    story: str | None = None,
    tags: str | None = None,
    story_pass: bool = True,
    vision_translate: bool = True,
    use_gpu: bool = True,
    font: str | None = None,
    conf: float = 0.34,
    accurate_detector: bool = True,
    save_cbz: bool = False,
    output_format: str = "original",
    sequential_names: bool = False,
    skip_blocked: bool = True,
    resume: bool = False,
    model_cache: ModelCache | None = None,
    hooks: PipelineHooks | None = None,
) -> tuple[Path, list[PageDocument]]:
    """Run full manga translation pipeline end-to-end and return output dir & documents."""
    hooks = hooks or PipelineHooks()
    cache = model_cache or ModelCache()

    with tempfile.TemporaryDirectory() as tmp_dir:
        images, base_name = collect_inputs(input_path, Path(tmp_dir))
        if not images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong đường dẫn đã chọn.")

        out_dir = output or input_path.parent / f"{base_name}_vn"
        cache_dir = out_dir / ".koharu_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        total = len(images)
        eff_story_model = (story_model or "").strip() or model
        mode_tag = "Vision" if vision_translate else "No-Vision OCR"
        hooks.emit_step(
            0,
            total,
            base_name,
            "init",
            f"Kiểm tra API & ONNX (Chính: {eff_story_model} -> Phụ ({mode_tag}): {model})...",
        )
        llm = VisionLLMClient(
            api_key=api_key,
            api_keys=api_keys,
            model=model,
            base_url=base_url,
            on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
        )
        story_llm = (
            llm
            if eff_story_model == model
            else VisionLLMClient(
                api_key=api_key,
                api_keys=api_keys,
                model=eff_story_model,
                base_url=base_url,
                on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
            )
        )
        if isinstance(cache, ModelCache):
            detector = cache.get_detector(accurate=accurate_detector, use_gpu=use_gpu)
            inpainter = cache.get_inpainter(use_gpu=use_gpu)
            ocr_engine = cache.get_ocr(use_gpu=use_gpu) if not vision_translate else None
        else:
            detector = cache.get_detector(accurate=accurate_detector)
            inpainter = cache.get_inpainter()
            ocr_engine = cache.get_ocr() if not vision_translate else None

        det_dev = active_ort_provider_label(getattr(detector, "_rtdetr", None))
        inp_dev = active_ort_provider_label(getattr(inpainter, "_lama", None))
        hooks.emit_step(
            0,
            total,
            base_name,
            "device",
            f"Thiết bị ONNX: Detector={det_dev} · LaMa Inpaint={inp_dev}",
        )

        scenes = cluster_story_scenes(images)
        if len(scenes) >= 2:
            hooks.emit_step(
                0,
                total,
                base_name,
                "scene_cluster",
                f"Phân tuyến thông minh: gom {total} ảnh thành {len(scenes)} tuyến/oneshot riêng để không trộn ngữ cảnh.",
            )

        story_bible = _load_or_build_story_bible(
            story_llm,
            images,
            cache_dir,
            base_name,
            mode="translate",
            user_story=story,
            prompt=prompt,
            tags=tags,
            enable_story_pass=story_pass,
            resume=resume,
            scenes=scenes,
            hooks=hooks,
        )

        docs: list[PageDocument] = []
        seq_pad = max(3, len(str(total)))
        translation_memory: dict[str, str] = {}

        def _remember_translations(items: list[TextRegionItem]) -> None:
            for item in items:
                if item.source_text and item.translated_text and not region_needs_translation(item):
                    key = normalize_source_key(item.source_text)
                    if key:
                        translation_memory[key] = item.translated_text.strip()

        def _apply_memory_to_untranslated(items: list[TextRegionItem], ocr_texts: dict[int, str] | None = None) -> list[TextRegionItem]:
            fixed: list[TextRegionItem] = []
            for item in items:
                if region_needs_translation(item):
                    src_key = normalize_source_key((ocr_texts or {}).get(item.id, "") or item.source_text)
                    cached_vn = translation_memory.get(src_key, "")
                    if cached_vn:
                        item = item.model_copy(update={"translated_text": cached_vn})
                fixed.append(item)
            return fixed

        for sc in scenes:
            scene_summaries: list[str] = []
            # Tuple: (idx, img_path, out_img_path, json_path, clean_path, w, h, rgb, clean_rgb, region_items, ocr_texts)
            pending_ocr_pages: list[
                tuple[int, Path, Path, Path, Path, int, int, np.ndarray, np.ndarray, list[TextRegionItem], dict[int, str]]
            ] = []

            for p_idx in sc.page_indices:
                idx = p_idx + 1
                img_path = images[p_idx]
                hooks.check_cancelled()
                out_img_path = out_dir / _output_image_name(
                    img_path,
                    output_format=output_format,
                    seq_index=idx if sequential_names else None,
                    seq_pad=seq_pad,
                )
                json_path = cache_dir / f"{img_path.stem}.koharu.json"

                if resume and out_img_path.is_file() and json_path.is_file():
                    prev_doc = PageDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
                    if not any(region_needs_translation(r) for r in prev_doc.regions):
                        _remember_translations(prev_doc.regions)
                        if prev_doc.page_summary:
                            scene_summaries.append(prev_doc.page_summary)
                        docs.append(prev_doc)
                        hooks.emit_step(idx, total, img_path.name, "skip", "Bỏ qua trang đã dịch (--resume)")
                        hooks.emit_page_done(idx, total, img_path.name, prev_doc, out_img_path)
                        continue
                    hooks.emit_step(idx, total, img_path.name, "retry", "Phát hiện vùng chưa dịch ở cache cũ -> dịch lại trang này...")

                rgb = load_rgb(img_path)
                h, w = rgb.shape[:2]

                hooks.emit_step(idx, total, img_path.name, "detect", f"Đọc ảnh {w}×{h}, đang dò khung thoại...")
                detected, seg_mask = detector.detect(rgb, conf_thresh=conf)
                if not detected:
                    save_image(Image.fromarray(rgb), out_img_path)
                    doc = PageDocument(
                        image_path=str(img_path.resolve()),
                        cleaned_image_path=str(out_img_path.resolve()),
                        width=w,
                        height=h,
                        mode="translate",
                        source_language=lang,
                        regions=[],
                    )
                    docs.append(doc)
                    hooks.emit_step(idx, total, img_path.name, "empty", "Không có khung thoại, giữ nguyên ảnh gốc")
                    hooks.emit_page_done(idx, total, img_path.name, doc, out_img_path)
                    continue

                hooks.check_cancelled()
                t0 = time.perf_counter()
                hooks.emit_step(
                    idx,
                    total,
                    img_path.name,
                    "inpaint",
                    f"Phát hiện {len(detected)} vùng chữ — đang tạo mask & tẩy chữ gốc...",
                )
                erasure_mask = build_erasure_mask(rgb, seg_mask, detected)
                clean_rgb = inpainter.inpaint(rgb, erasure_mask)

                clean_path = cache_dir / f"{img_path.stem}.clean.png"
                save_image(Image.fromarray(clean_rgb), clean_path)

                region_items: list[TextRegionItem] = []
                for reg_idx, reg in enumerate(detected, start=1):
                    t_col, s_col, s_factor, is_dark = classify_region_colors(
                        rgb, clean_rgb, erasure_mask, reg
                    )
                    region_items.append(
                        TextRegionItem(
                            id=reg_idx,
                            kind="bubble" if reg.is_bubble else "floating",
                            text_bbox=reg.text_bbox,
                            bubble_bbox=reg.bubble_bbox,
                            bubble_contour=reg.bubble_contour,
                            text_color=t_col,
                            stroke_color=s_col,
                            stroke_width=None if s_factor > 0 else 0.0,
                            font_name=font or ("animeace3" if reg.is_bubble else "pangolin"),
                            angle=reg.angle,
                            is_dark_bubble=is_dark,
                            draw_bubble=False,
                            confidence=reg.confidence,
                        )
                    )

                if not vision_translate and ocr_engine is not None:
                    hooks.check_cancelled()
                    hooks.emit_step(
                        idx,
                        total,
                        img_path.name,
                        "ocr",
                        f"Đang OCR cục bộ {len(region_items)} vùng chữ (No-Vision)...",
                    )
                    ocr_texts = ocr_engine.recognize_regions(rgb, region_items)
                    pending_ocr_pages.append(
                        (idx, img_path, out_img_path, json_path, clean_path, w, h, rgb, clean_rgb, region_items, ocr_texts)
                    )
                    continue

                hooks.check_cancelled()
                hooks.emit_step(
                    idx,
                    total,
                    img_path.name,
                    "llm",
                    f"Đang gửi {len(region_items)} vùng chữ lên {model} để OCR & dịch ({time.perf_counter() - t0:.1f}s)...",
                )
                prev_rgb = load_rgb(images[idx - 2]) if (idx > 1 and not story_bible) else None
                next_rgb = load_rgb(images[idx]) if (idx < total and not story_bible) else None
                try:
                    translated_items, summary = llm.translate_page(
                        rgb=rgb,
                        regions=region_items,
                        source_lang=lang,
                        previous_summaries=scene_summaries,
                        extra_instructions=prompt,
                        story_bible=story_bible,
                        page_index=idx,
                        prev_rgb=prev_rgb,
                        next_rgb=next_rgb,
                        tags=tags,
                    )
                except ContentBlockedError as exc:
                    fallback_done = False
                    if hasattr(cache, "get_ocr"):
                        try:
                            if ocr_engine is None:
                                ocr_engine = (
                                    cache.get_ocr(use_gpu=use_gpu)
                                    if isinstance(cache, ModelCache)
                                    else cache.get_ocr()
                                )
                            hooks.emit_step(
                                idx,
                                total,
                                img_path.name,
                                "ocr_fallback",
                                f"⚠ Trang {idx}/{total} bị lọc ảnh -> tự động fallback OCR cục bộ + dịch text-only (0 ảnh)...",
                            )
                            ocr_texts = ocr_engine.recognize_regions(rgb, region_items)
                            ocr_kw = {
                                "regions": region_items,
                                "ocr_texts": ocr_texts,
                                "source_lang": lang,
                                "previous_summaries": scene_summaries,
                                "extra_instructions": prompt,
                                "story_bible": story_bible,
                                "page_index": idx,
                                "tags": tags,
                                "translation_memory": translation_memory,
                            }
                            try:
                                translated_items, summary = llm.translate_page_from_ocr(**ocr_kw)
                                fallback_done = True
                            except ContentBlockedError as inner_exc:
                                if story_llm is not llm:
                                    translated_items, summary = story_llm.translate_page_from_ocr(**ocr_kw)
                                    fallback_done = True
                                else:
                                    exc = inner_exc
                        except ContentBlockedError as ocr_exc:
                            exc = ocr_exc
                    if not fallback_done:
                        if not skip_blocked:
                            raise
                        save_image(Image.fromarray(rgb), out_img_path)
                        hooks.emit_step(
                            idx,
                            total,
                            img_path.name,
                            "blocked",
                            f"⚠ Trang {idx}/{total} bị bộ lọc chặn ({exc}) — tạm giữ ảnh gốc & tiếp tục trang sau!",
                        )
                        continue
                translated_items = _apply_memory_to_untranslated(translated_items)
                if any(region_needs_translation(r) for r in translated_items):
                    still_missing = [r for r in translated_items if region_needs_translation(r)]
                    if hasattr(cache, "get_ocr"):
                        try:
                            if ocr_engine is None:
                                ocr_engine = (
                                    cache.get_ocr(use_gpu=use_gpu)
                                    if isinstance(cache, ModelCache)
                                    else cache.get_ocr()
                                )
                            missing_ocr = ocr_engine.recognize_regions(rgb, still_missing)
                            for mr in still_missing:
                                if missing_ocr.get(mr.id):
                                    mr.source_text = missing_ocr[mr.id]
                            still_missing = _apply_memory_to_untranslated(still_missing, missing_ocr)
                            need_api = [r for r in still_missing if region_needs_translation(r)]
                            if need_api:
                                retried_items, retry_sum = llm.translate_page_from_ocr(
                                    regions=need_api,
                                    ocr_texts=missing_ocr,
                                    source_lang=lang,
                                    previous_summaries=scene_summaries,
                                    extra_instructions=prompt,
                                    story_bible=story_bible,
                                    page_index=idx,
                                    tags=tags,
                                    translation_memory=translation_memory,
                                )
                                by_missing_id = {r.id: r for r in retried_items}
                            else:
                                by_missing_id = {r.id: r for r in still_missing}
                                retry_sum = ""
                            translated_items = [by_missing_id.get(r.id, r) for r in translated_items]
                            if not summary and retry_sum:
                                summary = retry_sum
                        except Exception:
                            pass
                _remember_translations(translated_items)
                if summary:
                    scene_summaries.append(summary)
                translated_items = adapt_regions_for_readability(translated_items, w, h)

                hooks.check_cancelled()
                hooks.emit_step(idx, total, img_path.name, "typeset", "Đang sắp chữ Tiếng Việt 2× supersampling...")
                doc = PageDocument(
                    image_path=str(img_path.resolve()),
                    cleaned_image_path=str(clean_path.resolve()),
                    width=w,
                    height=h,
                    mode="translate",
                    source_language=lang,
                    page_summary=summary,
                    regions=translated_items,
                )
                json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

                final_img = render_document(clean_rgb, doc)
                save_image(final_img, out_img_path)
                docs.append(doc)
                hooks.emit_page_done(idx, total, img_path.name, doc, out_img_path)

            if pending_ocr_pages:
                for start_i in range(0, len(pending_ocr_pages), 10):
                    chunk = pending_ocr_pages[start_i : start_i + 10]
                    hooks.check_cancelled()
                    batch_input = [(item[0], item[9], item[10]) for item in chunk]
                    hooks.emit_step(
                        chunk[0][0],
                        total,
                        chunk[0][1].name,
                        "llm",
                        f"Tuyến {sc.scene_id}: Gửi text OCR {len(chunk)} trang tới {model} (0 ảnh)...",
                    )
                    try:
                        batch_res = llm.translate_scene_from_ocr(
                            batch_input,
                            source_lang=lang,
                            previous_summaries=scene_summaries,
                            extra_instructions=prompt,
                            story_bible=story_bible,
                            tags=tags,
                            translation_memory=translation_memory,
                            scene_id=sc.scene_id,
                        )
                    except ContentBlockedError as exc:
                        if not skip_blocked:
                            raise
                        for idx, img_path, out_img_path, *_ in chunk:
                            rgb_orig = load_rgb(img_path)
                            save_image(Image.fromarray(rgb_orig), out_img_path)
                            hooks.emit_step(
                                idx,
                                total,
                                img_path.name,
                                "blocked",
                                f"⚠ Tuyến {sc.scene_id} trang {idx} bị chặn ({exc}) — tạm giữ ảnh gốc!",
                            )
                        continue

                    for idx, img_path, out_img_path, json_path, clean_path, w, h, _rgb, clean_rgb, reg_items, _ocrs in chunk:
                        translated_items, summary = batch_res.get(idx, (reg_items, ""))
                        translated_items = _apply_memory_to_untranslated(translated_items, _ocrs)
                        _remember_translations(translated_items)
                        if summary:
                            scene_summaries.append(summary)
                        translated_items = adapt_regions_for_readability(translated_items, w, h)
                        hooks.emit_step(idx, total, img_path.name, "typeset", "Đang sắp chữ Tiếng Việt 2× supersampling...")
                        doc = PageDocument(
                            image_path=str(img_path.resolve()),
                            cleaned_image_path=str(clean_path.resolve()),
                            width=w,
                            height=h,
                            mode="translate",
                            source_language=lang,
                            page_summary=summary,
                            regions=translated_items,
                        )
                        json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
                        final_img = render_document(clean_rgb, doc)
                        save_image(final_img, out_img_path)
                        docs.append(doc)
                        hooks.emit_page_done(idx, total, img_path.name, doc, out_img_path)

        if save_cbz:
            package_cbz(out_dir, out_dir.parent / f"{out_dir.name}.cbz")
        return out_dir, docs


def run_compose_pipeline(
    input_path: Path,
    *,
    output: Path | None = None,
    style: ComposePreset = ComposePreset.CG_OVAL,
    density: str = "medium",
    prompt: str | None = None,
    story: str | None = None,
    tags: str | None = None,
    story_pass: bool = True,
    api_key: str | None = None,
    api_keys: list[str] | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    story_model: str | None = None,
    font: str | None = None,
    save_cbz: bool = False,
    output_format: str = "original",
    sequential_names: bool = False,
    skip_blocked: bool = True,
    resume: bool = False,
    hooks: PipelineHooks | None = None,
) -> tuple[Path, list[PageDocument]]:
    """Run Artist CG dialogue composition pipeline end-to-end."""
    hooks = hooks or PipelineHooks()

    with tempfile.TemporaryDirectory() as tmp_dir:
        images, base_name = collect_inputs(input_path, Path(tmp_dir))
        if not images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong đường dẫn đã chọn.")

        out_dir = output or input_path.parent / f"{base_name}_composed"
        cache_dir = out_dir / ".koharu_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        total = len(images)
        eff_story_model = (story_model or "").strip() or model
        hooks.emit_step(
            0,
            total,
            base_name,
            "init",
            f"Khởi tạo sáng tác CG ({style.value} | Chính: {eff_story_model} -> Phụ: {model})...",
        )
        llm = VisionLLMClient(
            api_key=api_key,
            api_keys=api_keys,
            model=model,
            base_url=base_url,
            on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
        )
        story_llm = (
            llm
            if eff_story_model == model
            else VisionLLMClient(
                api_key=api_key,
                api_keys=api_keys,
                model=eff_story_model,
                base_url=base_url,
                on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
            )
        )

        scenes = cluster_story_scenes(images)
        if len(scenes) >= 2:
            hooks.emit_step(
                0,
                total,
                base_name,
                "scene_cluster",
                f"Phân tuyến thông minh: gom {total} ảnh thành {len(scenes)} tuyến/oneshot riêng để không trộn ngữ cảnh.",
            )

        story_bible = _load_or_build_story_bible(
            story_llm,
            images,
            cache_dir,
            base_name,
            mode="compose",
            user_story=story,
            prompt=prompt,
            tags=tags,
            enable_story_pass=story_pass,
            resume=resume,
            scenes=scenes,
            hooks=hooks,
        )

        previous_summaries: list[str] = []
        docs: list[PageDocument] = []

        # Fast 1-sheet batch composition when there are multiple CG pages (up to 36 per sheet)
        sheet_results: dict[int, tuple[list[TextRegionItem], str]] = {}
        if total > 1:
            msg = (
                f"Đang gộp {total} ảnh thành các sheet 12 ảnh nét cao để sáng tác nối tiếp..."
                if total > 18
                else f"Đang gộp {total} ảnh thành 1 sheet tổng hợp để sáng tác nhanh trong 1 lượt..."
            )
            hooks.emit_step(
                0,
                total,
                base_name,
                "llm_sheet",
                msg,
            )
            all_sizes: list[tuple[int, int]] = []
            for p in images:
                with Image.open(p) as im:
                    all_sizes.append(im.size)
            try:
                sheet_results = llm.compose_cg_sheet(
                    image_paths=images,
                    image_sizes=all_sizes,
                    preset=style,
                    density=density,
                    story_prompt=prompt,
                    story_bible=story_bible,
                    tags=tags,
                )
            except ContentBlockedError as exc:
                hooks.emit_step(
                    0,
                    total,
                    base_name,
                    "warn",
                    f"Sheet bị bộ lọc nội dung chặn ({exc}) — chuyển sang sáng tác từng trang...",
                )
                sheet_results = {}

        seq_pad = max(3, len(str(total)))
        for idx, img_path in enumerate(images, start=1):
            hooks.check_cancelled()
            out_img_path = out_dir / _output_image_name(
                img_path,
                output_format=output_format,
                seq_index=idx if sequential_names else None,
                seq_pad=seq_pad,
            )
            json_path = cache_dir / f"{img_path.stem}.koharu.json"

            if resume and out_img_path.is_file() and json_path.is_file():
                prev_doc = PageDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
                if prev_doc.page_summary:
                    previous_summaries.append(prev_doc.page_summary)
                docs.append(prev_doc)
                hooks.emit_step(idx, total, img_path.name, "skip", "Bỏ qua ảnh đã sáng tác (--resume)")
                hooks.emit_page_done(idx, total, img_path.name, prev_doc, out_img_path)
                continue

            rgb = load_rgb(img_path)
            h, w = rgb.shape[:2]

            if idx in sheet_results and sheet_results[idx][0]:
                items, summary = sheet_results[idx]
                hooks.emit_step(idx, total, img_path.name, "render", f"Áp thoại từ sheet ({len(items)} bóng)...")
            else:
                hooks.emit_step(
                    idx,
                    total,
                    img_path.name,
                    "llm",
                    f"Phân tích bối cảnh CG ({w}×{h}) & sáng tác thoại [{style.value}]...",
                )
                try:
                    items, summary = llm.compose_cg_page(
                        rgb=rgb,
                        preset=style,
                        density=density,
                        previous_summaries=previous_summaries,
                        story_prompt=prompt,
                        story_bible=story_bible,
                        page_index=idx,
                        tags=tags,
                    )
                except ContentBlockedError as exc:
                    save_image(Image.fromarray(rgb), out_img_path)
                    hooks.emit_step(
                        idx,
                        total,
                        img_path.name,
                        "blocked",
                        f"⚠ Ảnh {idx}/{total} bị bộ lọc chặn ({exc}) — tạm giữ ảnh gốc & tiếp tục ảnh sau!",
                    )
                    continue
            if font:
                for it in items:
                    it.font_name = font
            if items:
                items = expand_composed_boxes_if_tight(items, w, h)
            if summary:
                previous_summaries.append(summary)

            hooks.check_cancelled()
            hooks.emit_step(idx, total, img_path.name, "render", "Vẽ bubble/chữ thường & xuất ảnh CG...")
            doc = PageDocument(
                image_path=str(img_path.resolve()),
                cleaned_image_path=str(img_path.resolve()),
                width=w,
                height=h,
                mode="compose",
                source_language="vi",
                page_summary=summary,
                regions=items,
            )
            json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

            final_img = render_document(rgb, doc)
            save_image(final_img, out_img_path)
            docs.append(doc)
            hooks.emit_page_done(idx, total, img_path.name, doc, out_img_path)

        if save_cbz:
            package_cbz(out_dir, out_dir.parent / f"{out_dir.name}.cbz")
        return out_dir, docs


def run_rerender_pipeline(
    output_dir: Path,
    *,
    font: str | None = None,
    font_scale: float = 1.0,
    output_format: str = "original",
    sequential_names: bool = False,
    hooks: PipelineHooks | None = None,
) -> list[PageDocument]:
    """Re-render cached `.koharu.json` pages locally without calling any API or ML detector."""
    hooks = hooks or PipelineHooks()
    cache_dir = output_dir / ".koharu_cache"
    json_files = sorted(cache_dir.glob("*.koharu.json"), key=natural_key) if cache_dir.is_dir() else []
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file .koharu.json nào trong {cache_dir}")

    total = len(json_files)
    seq_pad = max(3, len(str(total)))
    docs: list[PageDocument] = []
    for idx, jp in enumerate(json_files, start=1):
        hooks.check_cancelled()
        doc = PageDocument.model_validate_json(jp.read_text(encoding="utf-8"))
        base_path = Path(doc.cleaned_image_path or doc.image_path)
        if not base_path.is_file():
            base_path = Path(doc.image_path)
        base_rgb = load_rgb(base_path)

        if font or abs(font_scale - 1.0) > 1e-3:
            for reg in doc.regions:
                if font:
                    reg.font_name = font
                if abs(font_scale - 1.0) > 1e-3:
                    solved = solve_layout(reg)
                    reg.font_size = max(10.0, solved.font_size * font_scale)

        orig_path = Path(doc.image_path)
        out_name = _output_image_name(
            orig_path,
            output_format=output_format,
            seq_index=idx if sequential_names else None,
            seq_pad=seq_pad,
        )
        out_path = output_dir / out_name
        hooks.emit_step(idx, total, out_name, "render", "Render lại ảnh từ .koharu.json...")
        out_img = render_document(base_rgb, doc)
        save_image(out_img, out_path)
        docs.append(doc)
        hooks.emit_page_done(idx, total, out_name, doc, out_path)

    return docs


def pack_images_to_webp(
    input_path: Path,
    *,
    output: Path | None = None,
    in_place: bool = False,
    sequential_names: bool = True,
    hooks: PipelineHooks | None = None,
) -> tuple[Path, list[Path]]:
    """
    Sort messy image filenames in natural reading order, convert to WebP max quality (100),
    and rename sequentially (001.webp, 002.webp, ...).
    """
    hooks = hooks or PipelineHooks()
    with tempfile.TemporaryDirectory() as tmp_dir:
        images, base_name = collect_inputs(input_path, Path(tmp_dir))
        if not images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong đường dẫn đã chọn.")

        total = len(images)
        seq_pad = max(3, len(str(total)))
        written: list[Path] = []

        if in_place and input_path.is_dir():
            out_dir = input_path
            staging = input_path / f".koharu_pack_{os.getpid()}"
            staging.mkdir(parents=True, exist_ok=True)
            try:
                staged_files: list[tuple[Path, str]] = []
                for idx, img_path in enumerate(images, start=1):
                    hooks.check_cancelled()
                    out_name = _output_image_name(
                        img_path,
                        output_format="webp",
                        seq_index=idx if sequential_names else None,
                        seq_pad=seq_pad,
                    )
                    hooks.emit_step(idx, total, img_path.name, "pack", f"Chuyển {img_path.name} -> {out_name} (WebP 100%)")
                    staged_path = staging / out_name
                    with Image.open(img_path) as im:
                        save_image(im, staged_path)
                    staged_files.append((staged_path, out_name))

                # Remove original images only after all WebP files succeeded
                for old_img in images:
                    if old_img.is_file():
                        old_img.unlink()
                for staged_path, out_name in staged_files:
                    dest = out_dir / out_name
                    staged_path.replace(dest)
                    written.append(dest)
            finally:
                if staging.exists():
                    for leftover in staging.iterdir():
                        leftover.unlink(missing_ok=True)
                    staging.rmdir()
            return out_dir, written

        out_dir = output or input_path.parent / f"{base_name}_webp"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_path in enumerate(images, start=1):
            hooks.check_cancelled()
            out_name = _output_image_name(
                img_path,
                output_format="webp",
                seq_index=idx if sequential_names else None,
                seq_pad=seq_pad,
            )
            hooks.emit_step(idx, total, img_path.name, "pack", f"Chuyển {img_path.name} -> {out_name} (WebP 100%)")
            dest = out_dir / out_name
            with Image.open(img_path) as im:
                save_image(im, dest)
            written.append(dest)
        return out_dir, written
