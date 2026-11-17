"""Shared execution pipelines for translate, compose, and rerender across CLI and TUI."""

from __future__ import annotations

import os
import queue
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from koharu_vn_cli.models import ComposePreset, PageDocument, StoryBible, TextRegionItem
from koharu_vn_cli.pipeline.bubble_renderer import render_bubbles, resolve_box_overlaps
from koharu_vn_cli.pipeline.detector import (
    ComicDetector,
    DetectedRegion,
    _ensure_ctd_model,
    active_ort_provider_label,
)
from koharu_vn_cli.pipeline.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    ContentBlockedError,
    VisionLLMClient,
    looks_like_floating_sfx,
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
    active_regions = [
        r for r in doc.regions if not (r.is_sfx and not (r.translated_text or "").strip())
    ]
    canvas = render_bubbles(canvas, active_regions)
    canvas = render_text_regions(canvas, active_regions)
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


@dataclass
class _PreparedVisionPage:
    scene_id: int
    idx: int
    img_path: Path
    out_img_path: Path
    json_path: Path
    clean_path: Path
    w: int
    h: int
    rgb: np.ndarray
    seg_mask: np.ndarray
    detected: list[DetectedRegion]
    region_items: list[TextRegionItem]


@dataclass
class _PreparedOcrBatch:
    scene_id: int
    pages: list[
        tuple[int, Path, Path, Path, Path, int, int, bytes, list[DetectedRegion], list[TextRegionItem], dict[int, str]]
    ]


@dataclass
class _TranslatedRenderTask:
    idx: int
    img_path: Path
    out_img_path: Path
    json_path: Path
    clean_path: Path
    w: int
    h: int
    rgb: np.ndarray
    seg_mask: np.ndarray
    detected: list[DetectedRegion]
    translated_items: list[TextRegionItem]
    summary: str


def _q_put(
    q: queue.Queue[object],
    item: object,
    abort_event: threading.Event,
    hooks: PipelineHooks,
) -> bool:
    """Put item into a bounded queue without deadlocking on abort or cancellation."""
    while not abort_event.is_set():
        if hooks.is_cancelled and hooks.is_cancelled():
            abort_event.set()
            return False
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _q_get(
    q: queue.Queue[object],
    abort_event: threading.Event,
    hooks: PipelineHooks,
) -> tuple[bool, object]:
    """Get item from queue cleanly without deadlocking on abort or cancellation."""
    while not abort_event.is_set():
        if hooks.is_cancelled and hooks.is_cancelled():
            abort_event.set()
            return False, None
        try:
            return True, q.get(timeout=0.1)
        except queue.Empty:
            continue
    return False, None


def _cleanup_cache_dir(
    cache_dir: Path,
    *,
    total: int,
    completed_count: int,
    clean_cache: bool,
    hooks: PipelineHooks,
    base_name: str,
) -> None:
    """Remove `.koharu_cache` once all pages succeeded 100%, keeping it when blocked/incomplete for `--resume`."""
    if not clean_cache or not cache_dir.is_dir():
        return
    if completed_count >= total > 0:
        shutil.rmtree(cache_dir, ignore_errors=True)
        hooks.emit_step(
            total,
            total,
            base_name,
            "clean_cache",
            "Đã hoàn tất 100% -> tự động xóa .koharu_cache cho gọn thư mục.",
        )
    else:
        remaining = max(0, total - completed_count)
        hooks.emit_step(
            completed_count,
            total,
            base_name,
            "keep_cache",
            f"Giữ lại .koharu_cache vì còn {remaining} trang chưa xong (gõ /resume để chạy tiếp).",
        )


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
    story_fallback_llm: VisionLLMClient | None = None,
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
    extra_kw = {"scenes": scenes} if scenes and len(scenes) >= 2 else {}
    try:
        bible = llm.build_story_bible(
            images,
            mode=mode,
            user_story=resolved_story,
            extra_instructions=prompt,
            tags=tags,
            **extra_kw,
        )
    except ContentBlockedError as exc:
        if story_fallback_llm is not None and story_fallback_llm.model != llm.model:
            hooks.emit_step(
                0,
                len(images),
                base_name,
                "warn",
                f"Pass 1 [Chính: {llm.model}] bị PROHIBIT ({exc}) -> thử model fallback chính [{story_fallback_llm.model}] (1 lượt)...",
            )
            try:
                bible = story_fallback_llm.build_story_bible(
                    images,
                    mode=mode,
                    user_story=resolved_story,
                    extra_instructions=prompt,
                    tags=tags,
                    **extra_kw,
                )
            except Exception as fb_exc:
                hooks.emit_step(
                    0,
                    len(images),
                    base_name,
                    "warn",
                    f"Fallback chính [{story_fallback_llm.model}] cũng bị chặn/lỗi ({fb_exc}) — bỏ qua Pass 1, tiếp tục dịch từng trang...",
                )
                return StoryBible(full_plot_summary=resolved_story) if resolved_story else None
        else:
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
    fallback_model: str | None = None,
    story_model: str | None = None,
    story_fallback_model: str | None = None,
    prompt: str | None = None,
    story: str | None = None,
    tags: str | None = None,
    story_pass: bool = True,
    vision_translate: bool = True,
    skip_sfx: bool = True,
    use_gpu: bool = True,
    font: str | None = None,
    conf: float = 0.34,
    accurate_detector: bool = True,
    save_cbz: bool = False,
    output_format: str = "original",
    sequential_names: bool = False,
    skip_blocked: bool = True,
    resume: bool = False,
    clean_cache: bool = True,
    max_workers: int = 0,
    model_cache: ModelCache | None = None,
    hooks: PipelineHooks | None = None,
) -> tuple[Path, list[PageDocument]]:
    """Run full manga translation pipeline end-to-end and return output dir & documents."""
    hooks = hooks or PipelineHooks()
    cache = model_cache or ModelCache()
    hooks_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        images, base_name = collect_inputs(input_path, Path(tmp_dir))
        if not images:
            raise ValueError("Không tìm thấy ảnh hợp lệ nào trong đường dẫn đã chọn.")

        out_dir = output or input_path.parent / f"{base_name}_vn"
        cache_dir = out_dir / ".koharu_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        total = len(images)

        def _emit_step(idx: int, name: str, stage: str, detail: str) -> None:
            with hooks_lock:
                hooks.emit_step(idx, total, name, stage, detail)

        def _emit_page_done(idx: int, name: str, doc: PageDocument, out_path: Path) -> None:
            with hooks_lock:
                hooks.emit_page_done(idx, total, name, doc, out_path)

        eff_story_model = (story_model or "").strip() or model
        eff_fallback = (fallback_model or "").strip() or None
        eff_story_fallback = (story_fallback_model or "").strip() or None
        mode_tag = "Vision" if vision_translate else "No-Vision OCR"
        fb_s_tag = f" [FB: {eff_story_fallback}]" if eff_story_fallback else ""
        fb_m_tag = f" [FB quota: {eff_fallback}]" if eff_fallback else ""
        _emit_step(
            0,
            base_name,
            "init",
            f"Kiểm tra API & ONNX (Chính: {eff_story_model}{fb_s_tag} -> Phụ ({mode_tag}): {model}{fb_m_tag})...",
        )
        STORY_MODEL_TIMEOUT = 180.0
        llm = VisionLLMClient(
            api_key=api_key,
            api_keys=api_keys,
            model=model,
            fallback_model=eff_fallback,
            base_url=base_url,
            on_status=lambda msg: _emit_step(0, base_name, "api", msg),
        )
        story_llm = VisionLLMClient(
            api_key=api_key,
            api_keys=api_keys,
            model=eff_story_model,
            base_url=base_url,
            timeout=STORY_MODEL_TIMEOUT,
            on_status=lambda msg: _emit_step(0, base_name, "api", msg),
        )
        story_fallback_llm = (
            VisionLLMClient(
                api_key=api_key,
                api_keys=api_keys,
                model=eff_story_fallback,
                base_url=base_url,
                timeout=STORY_MODEL_TIMEOUT,
                on_status=lambda msg: _emit_step(0, base_name, "api", msg),
            )
            if eff_story_fallback and eff_story_fallback != eff_story_model
            else None
        )
        key_pool = list(llm.api_keys)
        if max_workers > 0:
            worker_count = max(1, min(max_workers, total, 16))
        else:
            worker_count = max(1, min(len(key_pool) or 1, total, 12))

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
        pipe_desc = (
            f"3 tầng song song ({worker_count} luồng API từ {len(key_pool) or 1} key)"
            if worker_count > 1
            else "3 tầng gối đầu (Detect + LLM + Inpaint)"
        )
        _emit_step(
            0,
            base_name,
            "device",
            f"Thiết bị ONNX: Detector={det_dev} · LaMa Inpaint={inp_dev} · Pipeline: {pipe_desc}",
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
            story_fallback_llm=story_fallback_llm,
            hooks=hooks,
        )

        docs_by_idx: dict[int, PageDocument] = {}
        seq_pad = max(3, len(str(total)))
        translation_memory: dict[str, str] = {}
        scene_summaries_by_page: dict[int, dict[int, str]] = {sc.scene_id: {} for sc in scenes}

        state_lock = threading.Lock()
        gpu_lock = threading.Lock()
        ocr_init_lock = threading.Lock()
        error_lock = threading.Lock()
        worker_done_lock = threading.Lock()
        abort_event = threading.Event()
        first_error: list[BaseException] = []

        def _rotated_keys(w_idx: int) -> list[str] | None:
            if not key_pool:
                return None
            offset = w_idx % len(key_pool)
            return key_pool[offset:] + key_pool[:offset]

        worker_clients: list[tuple[VisionLLMClient, VisionLLMClient, VisionLLMClient | None]] = []
        for w_idx in range(worker_count):
            if w_idx == 0:
                worker_clients.append((llm, story_llm, story_fallback_llm))
                continue
            r_keys = _rotated_keys(w_idx)
            r_primary = r_keys[0] if r_keys else api_key
            worker_clients.append(
                (
                    VisionLLMClient(
                        api_key=r_primary,
                        api_keys=r_keys,
                        model=model,
                        fallback_model=eff_fallback,
                        base_url=base_url,
                        on_status=lambda msg: _emit_step(0, base_name, "api", msg),
                    ),
                    VisionLLMClient(
                        api_key=r_primary,
                        api_keys=r_keys,
                        model=eff_story_model,
                        base_url=base_url,
                        timeout=STORY_MODEL_TIMEOUT,
                        on_status=lambda msg: _emit_step(0, base_name, "api", msg),
                    ),
                    (
                        VisionLLMClient(
                            api_key=r_primary,
                            api_keys=r_keys,
                            model=eff_story_fallback,
                            base_url=base_url,
                            timeout=STORY_MODEL_TIMEOUT,
                            on_status=lambda msg: _emit_step(0, base_name, "api", msg),
                        )
                        if eff_story_fallback and eff_story_fallback != eff_story_model
                        else None
                    ),
                )
            )

        def _record_error(exc: BaseException) -> None:
            with error_lock:
                if not first_error:
                    first_error.append(exc)
            abort_event.set()

        def _ensure_ocr_engine():
            nonlocal ocr_engine
            with ocr_init_lock:
                if ocr_engine is None and hasattr(cache, "get_ocr"):
                    ocr_engine = (
                        cache.get_ocr(use_gpu=use_gpu)
                        if isinstance(cache, ModelCache)
                        else cache.get_ocr()
                    )
                return ocr_engine

        def _remember_translations(items: list[TextRegionItem]) -> None:
            for item in items:
                if (
                    not item.is_sfx
                    and item.source_text
                    and item.translated_text
                    and not region_needs_translation(item)
                ):
                    key = normalize_source_key(item.source_text)
                    if key:
                        translation_memory[key] = item.translated_text.strip()

        def _apply_memory_to_untranslated(
            items: list[TextRegionItem],
            mem_snapshot: dict[str, str],
            ocr_texts: dict[int, str] | None = None,
        ) -> list[TextRegionItem]:
            fixed: list[TextRegionItem] = []
            for item in items:
                if skip_sfx and item.kind == "floating" and looks_like_floating_sfx(item.source_text):
                    item = item.model_copy(update={"is_sfx": True, "translated_text": ""})
                if skip_sfx and item.is_sfx:
                    fixed.append(item.model_copy(update={"translated_text": ""}))
                    continue
                if region_needs_translation(item):
                    src_key = normalize_source_key((ocr_texts or {}).get(item.id, "") or item.source_text)
                    cached_vn = mem_snapshot.get(src_key, "")
                    if cached_vn:
                        item = item.model_copy(update={"translated_text": cached_vn})
                fixed.append(item)
            return fixed

        def _scene_prev_summaries(scene_id: int, before_idx: int) -> list[str]:
            with state_lock:
                by_page = scene_summaries_by_page.get(scene_id, {})
                return [txt for p_num, txt in sorted(by_page.items()) if p_num < before_idx and txt]

        def _finalize_inpaint_and_colors(
            idx: int,
            img_name: str,
            rgb: np.ndarray,
            seg_mask: np.ndarray,
            detected: list[DetectedRegion],
            items: list[TextRegionItem],
        ) -> tuple[np.ndarray, list[TextRegionItem]]:
            """Build erasure mask and run LaMa inpainting only on non-SFX regions so artwork SFX stays intact."""
            by_id_det = {reg_idx: det for reg_idx, det in enumerate(detected, start=1)}
            active_pairs = [
                (by_id_det[r.id], r)
                for r in items
                if r.id in by_id_det and not (skip_sfx and r.is_sfx)
            ]
            sfx_count = sum(1 for r in items if skip_sfx and r.is_sfx)
            active_detected = [det for det, _ in active_pairs]
            if not active_detected:
                _emit_step(
                    idx,
                    img_name,
                    "inpaint",
                    f"Chỉ có {sfx_count} SFX — giữ nguyên tranh gốc không tẩy/dịch!",
                )
                return rgb, items

            _emit_step(
                idx,
                img_name,
                "inpaint",
                (
                    f"Bỏ qua {sfx_count} SFX · đang tạo mask & tẩy {len(active_detected)} vùng thoại..."
                    if sfx_count
                    else f"Phát hiện {len(active_detected)} vùng thoại — đang tạo mask & tẩy chữ gốc..."
                ),
            )
            erasure_mask = build_erasure_mask(rgb, seg_mask, active_detected)
            with gpu_lock:
                clean_rgb = inpainter.inpaint(rgb, erasure_mask)
            updated_by_id: dict[int, TextRegionItem] = {}
            for det, reg in active_pairs:
                t_col, s_col, s_factor, is_dark = classify_region_colors(
                    rgb, clean_rgb, erasure_mask, det
                )
                updated_by_id[reg.id] = reg.model_copy(
                    update={
                        "text_color": t_col,
                        "stroke_color": s_col,
                        "stroke_width": None if s_factor > 0 else 0.0,
                        "is_dark_bubble": is_dark,
                        "font_name": reg.font_name or font or ("animeace3" if det.is_bubble else "pangolin"),
                    }
                )
            merged = [updated_by_id.get(r.id, r) for r in items]
            return clean_rgb, merged

        buf_size = max(4, worker_count + 3)
        translate_queue: queue.Queue[object] = queue.Queue(maxsize=buf_size)
        finalize_queue: queue.Queue[object] = queue.Queue(maxsize=buf_size)

        def _detect_producer() -> None:
            try:
                for sc in scenes:
                    pending_ocr_pages: list[
                        tuple[
                            int,
                            Path,
                            Path,
                            Path,
                            Path,
                            int,
                            int,
                            bytes,
                            list[DetectedRegion],
                            list[TextRegionItem],
                            dict[int, str],
                        ]
                    ] = []

                    for p_idx in sc.page_indices:
                        if abort_event.is_set():
                            return
                        hooks.check_cancelled()
                        idx = p_idx + 1
                        img_path = images[p_idx]
                        out_img_path = out_dir / _output_image_name(
                            img_path,
                            output_format=output_format,
                            seq_index=idx if sequential_names else None,
                            seq_pad=seq_pad,
                        )
                        json_path = cache_dir / f"{img_path.stem}.koharu.json"

                        if resume and out_img_path.is_file() and json_path.is_file():
                            prev_doc = PageDocument.model_validate_json(
                                json_path.read_text(encoding="utf-8")
                            )
                            if not any(region_needs_translation(r) for r in prev_doc.regions):
                                with state_lock:
                                    _remember_translations(prev_doc.regions)
                                    if prev_doc.page_summary:
                                        scene_summaries_by_page[sc.scene_id][idx] = prev_doc.page_summary
                                    docs_by_idx[idx] = prev_doc
                                _emit_step(idx, img_path.name, "skip", "Bỏ qua trang đã dịch (--resume)")
                                _emit_page_done(idx, img_path.name, prev_doc, out_img_path)
                                continue
                            _emit_step(
                                idx,
                                img_path.name,
                                "retry",
                                "Phát hiện vùng chưa dịch ở cache cũ -> dịch lại trang này...",
                            )

                        rgb = load_rgb(img_path)
                        h, w = rgb.shape[:2]

                        _emit_step(idx, img_path.name, "detect", f"Đọc ảnh {w}×{h}, đang dò khung thoại...")
                        with gpu_lock:
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
                            with state_lock:
                                docs_by_idx[idx] = doc
                            _emit_step(idx, img_path.name, "empty", "Không có khung thoại, giữ nguyên ảnh gốc")
                            _emit_page_done(idx, img_path.name, doc, out_img_path)
                            continue

                        if abort_event.is_set():
                            return
                        hooks.check_cancelled()
                        clean_path = cache_dir / f"{img_path.stem}.clean.png"
                        region_items: list[TextRegionItem] = [
                            TextRegionItem(
                                id=reg_idx,
                                kind="bubble" if reg.is_bubble else "floating",
                                text_bbox=reg.text_bbox,
                                bubble_bbox=reg.bubble_bbox,
                                bubble_contour=reg.bubble_contour,
                                font_name=font or ("animeace3" if reg.is_bubble else "pangolin"),
                                angle=reg.angle,
                                is_dark_bubble=reg.is_dark_bubble,
                                draw_bubble=False,
                                confidence=reg.confidence,
                            )
                            for reg_idx, reg in enumerate(detected, start=1)
                        ]

                        if not vision_translate and ocr_engine is not None:
                            _emit_step(
                                idx,
                                img_path.name,
                                "ocr",
                                f"Đang OCR cục bộ {len(region_items)} vùng chữ (No-Vision)...",
                            )
                            with gpu_lock:
                                ocr_texts = ocr_engine.recognize_regions(rgb, region_items)
                            if skip_sfx:
                                for reg_it in region_items:
                                    raw_ocr = (ocr_texts.get(reg_it.id, "") or "").strip()
                                    if raw_ocr:
                                        reg_it.source_text = raw_ocr
                                    if reg_it.kind == "floating" and looks_like_floating_sfx(raw_ocr):
                                        reg_it.is_sfx = True
                            ok_enc, seg_buf = cv2.imencode(".png", seg_mask)
                            seg_bytes = seg_buf.tobytes() if ok_enc else b""
                            pending_ocr_pages.append(
                                (
                                    idx,
                                    img_path,
                                    out_img_path,
                                    json_path,
                                    clean_path,
                                    w,
                                    h,
                                    seg_bytes,
                                    detected,
                                    region_items,
                                    ocr_texts,
                                )
                            )
                            if len(pending_ocr_pages) >= 10:
                                batch = _PreparedOcrBatch(scene_id=sc.scene_id, pages=pending_ocr_pages)
                                pending_ocr_pages = []
                                if not _q_put(translate_queue, batch, abort_event, hooks):
                                    return
                            continue

                        task = _PreparedVisionPage(
                            scene_id=sc.scene_id,
                            idx=idx,
                            img_path=img_path,
                            out_img_path=out_img_path,
                            json_path=json_path,
                            clean_path=clean_path,
                            w=w,
                            h=h,
                            rgb=rgb,
                            seg_mask=seg_mask,
                            detected=detected,
                            region_items=region_items,
                        )
                        if not _q_put(translate_queue, task, abort_event, hooks):
                            return

                    if pending_ocr_pages:
                        batch = _PreparedOcrBatch(scene_id=sc.scene_id, pages=pending_ocr_pages)
                        if not _q_put(translate_queue, batch, abort_event, hooks):
                            return
            except BaseException as exc:
                _record_error(exc)
            finally:
                if not abort_event.is_set():
                    for _ in range(worker_count):
                        if not _q_put(translate_queue, None, abort_event, hooks):
                            break

        remaining_llm_workers = worker_count

        def _translate_vision_task(
            task: _PreparedVisionPage,
            worker_prefix: str,
            w_llm: VisionLLMClient,
            w_story_llm: VisionLLMClient,
            w_story_fallback: VisionLLMClient | None,
        ) -> None:
            t0 = time.perf_counter()
            idx = task.idx
            _emit_step(
                idx,
                task.img_path.name,
                "llm",
                f"{worker_prefix}Đang gửi {len(task.region_items)} vùng chữ lên {w_llm.model} để OCR & dịch ({time.perf_counter() - t0:.1f}s)...",
            )
            prev_rgb = load_rgb(images[idx - 2]) if (idx > 1 and not story_bible) else None
            next_rgb = load_rgb(images[idx]) if (idx < total and not story_bible) else None
            prev_summaries = _scene_prev_summaries(task.scene_id, idx)

            try:
                translated_items, summary = w_llm.translate_page(
                    rgb=task.rgb,
                    regions=task.region_items,
                    source_lang=lang,
                    previous_summaries=prev_summaries,
                    extra_instructions=prompt,
                    story_bible=story_bible,
                    page_index=idx,
                    prev_rgb=prev_rgb,
                    next_rgb=next_rgb,
                    tags=tags,
                    skip_sfx=skip_sfx,
                )
            except ContentBlockedError as exc:
                fallback_done = False
                eng = _ensure_ocr_engine()
                if eng is not None:
                    try:
                        _emit_step(
                            idx,
                            task.img_path.name,
                            "ocr_fallback",
                            f"⚠ {worker_prefix}Trang {idx}/{total} bị lọc ảnh -> tự động fallback OCR cục bộ + dịch text-only (0 ảnh)...",
                        )
                        with gpu_lock:
                            ocr_texts = eng.recognize_regions(task.rgb, task.region_items)
                        with state_lock:
                            mem_snapshot = dict(translation_memory)
                        ocr_kw = {
                            "regions": task.region_items,
                            "ocr_texts": ocr_texts,
                            "source_lang": lang,
                            "previous_summaries": _scene_prev_summaries(task.scene_id, idx),
                            "extra_instructions": prompt,
                            "story_bible": story_bible,
                            "page_index": idx,
                            "tags": tags,
                            "translation_memory": mem_snapshot,
                            "skip_sfx": skip_sfx,
                        }
                        try:
                            translated_items, summary = w_llm.translate_page_from_ocr(**ocr_kw)
                            fallback_done = True
                        except ContentBlockedError as inner_exc:
                            exc = inner_exc
                            if w_story_llm.model != w_llm.model:
                                try:
                                    translated_items, summary = w_story_llm.translate_page_from_ocr(**ocr_kw)
                                    fallback_done = True
                                except ContentBlockedError as s_exc:
                                    exc = s_exc
                            if not fallback_done and w_story_fallback is not None:
                                translated_items, summary = w_story_fallback.translate_page_from_ocr(**ocr_kw)
                                fallback_done = True
                    except ContentBlockedError as ocr_exc:
                        exc = ocr_exc
                if not fallback_done:
                    if not skip_blocked:
                        raise
                    save_image(Image.fromarray(task.rgb), task.out_img_path)
                    _emit_step(
                        idx,
                        task.img_path.name,
                        "blocked",
                        f"⚠ Trang {idx}/{total} bị bộ lọc chặn ({exc}) — tạm giữ ảnh gốc & tiếp tục trang sau!",
                    )
                    return

            with state_lock:
                mem_snapshot = dict(translation_memory)
            translated_items = _apply_memory_to_untranslated(translated_items, mem_snapshot)
            if any(region_needs_translation(r) for r in translated_items):
                still_missing = [r for r in translated_items if region_needs_translation(r)]
                eng = _ensure_ocr_engine()
                if eng is not None:
                    try:
                        with gpu_lock:
                            missing_ocr = eng.recognize_regions(task.rgb, still_missing)
                        for mr in still_missing:
                            if missing_ocr.get(mr.id):
                                mr.source_text = missing_ocr[mr.id]
                        with state_lock:
                            mem_snapshot = dict(translation_memory)
                        still_missing = _apply_memory_to_untranslated(
                            still_missing, mem_snapshot, missing_ocr
                        )
                        need_api = [r for r in still_missing if region_needs_translation(r)]
                        if need_api:
                            retried_items, retry_sum = w_llm.translate_page_from_ocr(
                                regions=need_api,
                                ocr_texts=missing_ocr,
                                source_lang=lang,
                                previous_summaries=_scene_prev_summaries(task.scene_id, idx),
                                extra_instructions=prompt,
                                story_bible=story_bible,
                                page_index=idx,
                                tags=tags,
                                translation_memory=mem_snapshot,
                                skip_sfx=skip_sfx,
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

            with state_lock:
                _remember_translations(translated_items)
                if summary:
                    scene_summaries_by_page[task.scene_id][idx] = summary

            _q_put(
                finalize_queue,
                _TranslatedRenderTask(
                    idx=idx,
                    img_path=task.img_path,
                    out_img_path=task.out_img_path,
                    json_path=task.json_path,
                    clean_path=task.clean_path,
                    w=task.w,
                    h=task.h,
                    rgb=task.rgb,
                    seg_mask=task.seg_mask,
                    detected=task.detected,
                    translated_items=translated_items,
                    summary=summary,
                ),
                abort_event,
                hooks,
            )

        def _translate_ocr_batch_task(
            batch: _PreparedOcrBatch,
            worker_prefix: str,
            w_llm: VisionLLMClient,
        ) -> None:
            chunk = batch.pages
            batch_input = [
                (item[0], [r for r in item[9] if not (skip_sfx and r.is_sfx)], item[10])
                for item in chunk
                if any(not (skip_sfx and r.is_sfx) for r in item[9])
            ]
            batch_res: dict[int, tuple[list[TextRegionItem], str]] = {}
            if batch_input:
                first_idx = chunk[0][0]
                _emit_step(
                    first_idx,
                    chunk[0][1].name,
                    "llm",
                    f"{worker_prefix}Tuyến {batch.scene_id}: Gửi text OCR {len(batch_input)} trang tới {w_llm.model} (0 ảnh)...",
                )
                with state_lock:
                    mem_snapshot = dict(translation_memory)
                try:
                    batch_res = w_llm.translate_scene_from_ocr(
                        batch_input,
                        source_lang=lang,
                        previous_summaries=_scene_prev_summaries(batch.scene_id, first_idx),
                        extra_instructions=prompt,
                        story_bible=story_bible,
                        tags=tags,
                        translation_memory=mem_snapshot,
                        skip_sfx=skip_sfx,
                        scene_id=batch.scene_id,
                    )
                except ContentBlockedError as exc:
                    if not skip_blocked:
                        raise
                    for idx, img_path, out_img_path, *_ in chunk:
                        rgb_orig = load_rgb(img_path)
                        save_image(Image.fromarray(rgb_orig), out_img_path)
                        _emit_step(
                            idx,
                            img_path.name,
                            "blocked",
                            f"⚠ Tuyến {batch.scene_id} trang {idx} bị chặn ({exc}) — tạm giữ ảnh gốc!",
                        )
                    return

            for idx, img_path, out_img_path, json_path, clean_path, w, h, seg_bytes, det_list, reg_items, _ocrs in chunk:
                sub_items, summary = batch_res.get(idx, (reg_items, ""))
                sub_map = {r.id: r for r in sub_items}
                translated_items = [sub_map.get(r.id, r) for r in reg_items]
                with state_lock:
                    mem_snapshot = dict(translation_memory)
                translated_items = _apply_memory_to_untranslated(
                    translated_items, mem_snapshot, _ocrs
                )
                with state_lock:
                    _remember_translations(translated_items)
                    if summary:
                        scene_summaries_by_page[batch.scene_id][idx] = summary

                rgb_page = load_rgb(img_path)
                seg_arr = (
                    cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if seg_bytes
                    else None
                )
                if seg_arr is None:
                    seg_arr = np.zeros((h, w), dtype=np.uint8)
                if not _q_put(
                    finalize_queue,
                    _TranslatedRenderTask(
                        idx=idx,
                        img_path=img_path,
                        out_img_path=out_img_path,
                        json_path=json_path,
                        clean_path=clean_path,
                        w=w,
                        h=h,
                        rgb=rgb_page,
                        seg_mask=seg_arr,
                        detected=det_list,
                        translated_items=translated_items,
                        summary=summary,
                    ),
                    abort_event,
                    hooks,
                ):
                    return

        def _llm_worker(worker_id: int) -> None:
            nonlocal remaining_llm_workers
            w_llm, w_story_llm, w_story_fallback = worker_clients[worker_id]
            worker_prefix = f"[W{worker_id + 1}] " if worker_count > 1 else ""
            try:
                while not abort_event.is_set():
                    ok, task = _q_get(translate_queue, abort_event, hooks)
                    if not ok or task is None:
                        break
                    hooks.check_cancelled()
                    if isinstance(task, _PreparedVisionPage):
                        _translate_vision_task(
                            task, worker_prefix, w_llm, w_story_llm, w_story_fallback
                        )
                    elif isinstance(task, _PreparedOcrBatch):
                        _translate_ocr_batch_task(task, worker_prefix, w_llm)
            except BaseException as exc:
                _record_error(exc)
            finally:
                with worker_done_lock:
                    remaining_llm_workers -= 1
                    is_last = remaining_llm_workers == 0
                if is_last and not abort_event.is_set():
                    _q_put(finalize_queue, None, abort_event, hooks)

        def _finalize_consumer() -> None:
            try:
                while not abort_event.is_set():
                    ok, rtask = _q_get(finalize_queue, abort_event, hooks)
                    if not ok or rtask is None:
                        break
                    hooks.check_cancelled()
                    assert isinstance(rtask, _TranslatedRenderTask)
                    clean_rgb, items_ready = _finalize_inpaint_and_colors(
                        rtask.idx,
                        rtask.img_path.name,
                        rtask.rgb,
                        rtask.seg_mask,
                        rtask.detected,
                        rtask.translated_items,
                    )
                    save_image(Image.fromarray(clean_rgb), rtask.clean_path)
                    items_ready = adapt_regions_for_readability(items_ready, rtask.w, rtask.h)

                    hooks.check_cancelled()
                    _emit_step(
                        rtask.idx,
                        rtask.img_path.name,
                        "typeset",
                        "Đang sắp chữ Tiếng Việt 2× supersampling...",
                    )
                    doc = PageDocument(
                        image_path=str(rtask.img_path.resolve()),
                        cleaned_image_path=str(rtask.clean_path.resolve()),
                        width=rtask.w,
                        height=rtask.h,
                        mode="translate",
                        source_language=lang,
                        page_summary=rtask.summary,
                        regions=items_ready,
                    )
                    rtask.json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")

                    final_img = render_document(clean_rgb, doc)
                    save_image(final_img, rtask.out_img_path)
                    with state_lock:
                        docs_by_idx[rtask.idx] = doc
                    _emit_page_done(rtask.idx, rtask.img_path.name, doc, rtask.out_img_path)
            except BaseException as exc:
                _record_error(exc)

        threads: list[threading.Thread] = [
            threading.Thread(target=_detect_producer, name="koharu-detect", daemon=True),
            *[
                threading.Thread(
                    target=_llm_worker, args=(w_i,), name=f"koharu-llm-{w_i + 1}", daemon=True
                )
                for w_i in range(worker_count)
            ],
            threading.Thread(target=_finalize_consumer, name="koharu-finalize", daemon=True),
        ]
        for th in threads:
            th.start()
        try:
            for th in threads:
                th.join()
        except BaseException as exc:
            _record_error(exc)
            for th in threads:
                th.join(timeout=1.0)
            raise

        if first_error:
            raise first_error[0]

        docs = [docs_by_idx[i] for i in sorted(docs_by_idx)]

        if save_cbz:
            package_cbz(out_dir, out_dir.parent / f"{out_dir.name}.cbz")
        _cleanup_cache_dir(
            cache_dir,
            total=total,
            completed_count=len(docs),
            clean_cache=clean_cache,
            hooks=hooks,
            base_name=base_name,
        )
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
    fallback_model: str | None = None,
    story_model: str | None = None,
    story_fallback_model: str | None = None,
    font: str | None = None,
    save_cbz: bool = False,
    output_format: str = "original",
    sequential_names: bool = False,
    skip_blocked: bool = True,
    resume: bool = False,
    clean_cache: bool = True,
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
        eff_fallback = (fallback_model or "").strip() or None
        eff_story_fallback = (story_fallback_model or "").strip() or None
        fb_s_tag = f" [FB: {eff_story_fallback}]" if eff_story_fallback else ""
        fb_m_tag = f" [FB quota: {eff_fallback}]" if eff_fallback else ""
        hooks.emit_step(
            0,
            total,
            base_name,
            "init",
            f"Khởi tạo sáng tác CG ({style.value} | Chính: {eff_story_model}{fb_s_tag} -> Phụ: {model}{fb_m_tag})...",
        )
        STORY_MODEL_TIMEOUT = 180.0
        llm = VisionLLMClient(
            api_key=api_key,
            api_keys=api_keys,
            model=model,
            fallback_model=eff_fallback,
            base_url=base_url,
            on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
        )
        story_llm = VisionLLMClient(
            api_key=api_key,
            api_keys=api_keys,
            model=eff_story_model,
            base_url=base_url,
            timeout=STORY_MODEL_TIMEOUT,
            on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
        )
        story_fallback_llm = (
            VisionLLMClient(
                api_key=api_key,
                api_keys=api_keys,
                model=eff_story_fallback,
                base_url=base_url,
                timeout=STORY_MODEL_TIMEOUT,
                on_status=lambda msg: hooks.emit_step(0, total, base_name, "api", msg),
            )
            if eff_story_fallback and eff_story_fallback != eff_story_model
            else None
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
            story_fallback_llm=story_fallback_llm,
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
        _cleanup_cache_dir(
            cache_dir,
            total=total,
            completed_count=len(docs),
            clean_cache=clean_cache,
            hooks=hooks,
            base_name=base_name,
        )
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
