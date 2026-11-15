"""Text and speech-bubble detection using ONNX models + Containment NMS + Bubble Flood-Fill."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

CACHE_DIR = Path.home() / ".cache" / "koharu-vn"
CTD_URL = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"


@dataclass
class DetectedRegion:
    """Intermediate detection result before OCR/translation."""

    text_bbox: tuple[int, int, int, int]
    bubble_bbox: tuple[int, int, int, int] | None
    bubble_contour: list[tuple[float, float]] | None
    is_bubble: bool
    is_dark_bubble: bool
    confidence: float
    angle: float = 0.0


def _ensure_ctd_model() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "comictextdetector.pt.onnx"
    if path.is_file() and path.stat().st_size > 50_000_000:
        return path
    with httpx.stream("GET", CTD_URL, follow_redirects=True, timeout=180) as resp:
        resp.raise_for_status()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_bytes(65536):
                fh.write(chunk)
        tmp.replace(path)
    return path


_ORT_PRELOADED = False


def _preload_ort_dlls() -> None:
    global _ORT_PRELOADED
    if _ORT_PRELOADED:
        return
    _ORT_PRELOADED = True
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except Exception:
            pass


def _ort_providers(use_gpu: bool = True) -> list[str]:
    if not use_gpu:
        return ["CPUExecutionProvider"]
    _preload_ort_dlls()
    available = ort.get_available_providers()
    preferred = [
        p
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
        if p in available
    ]
    return preferred or ["CPUExecutionProvider"]


def _create_ort_session(
    model_path: str | Path,
    *,
    use_gpu: bool = True,
    cpu_threads: int | None = None,
    low_mem: bool = False,
) -> ort.InferenceSession:
    """Create an ONNXRuntime session preferring GPU (CUDA/DirectML) with automatic CPU fallback."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if low_mem:
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
    if cpu_threads is not None:
        opts.intra_op_num_threads = max(1, cpu_threads)
    providers = _ort_providers(use_gpu=use_gpu)
    try:
        return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    except Exception:
        if providers != ["CPUExecutionProvider"]:
            cpu_opts = ort.SessionOptions()
            cpu_opts.log_severity_level = 3
            cpu_opts.enable_cpu_mem_arena = False
            cpu_opts.enable_mem_pattern = False
            if cpu_threads is not None:
                cpu_opts.intra_op_num_threads = max(1, cpu_threads)
            return ort.InferenceSession(str(model_path), sess_options=cpu_opts, providers=["CPUExecutionProvider"])
        raise


def active_ort_provider_label(session: ort.InferenceSession | None) -> str:
    """Return a concise human-readable execution device label for telemetry."""
    if session is None:
        return "CPU"
    try:
        provs = session.get_providers()
    except Exception:
        return "CPU"
    if "CUDAExecutionProvider" in provs:
        return "GPU (CUDA)"
    if "DmlExecutionProvider" in provs:
        return "GPU (DirectML)"
    return "CPU"


def box_area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / max(1, union)


def containment_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Fraction of `inner` area contained inside `outer`."""
    area_in = box_area(inner)
    if area_in <= 0:
        return 0.0
    return intersection_area(inner, outer) / area_in


def union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _clamp_box(b: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(b[0]))))
    y1 = max(0, min(h - 1, int(round(b[1]))))
    x2 = max(x1 + 1, min(w, int(round(b[2]))))
    y2 = max(y1 + 1, min(h, int(round(b[3]))))
    return (x1, y1, x2, y2)


def _containment_nms(
    items: list[tuple[tuple[int, int, int, int], float, int]],
    iou_thresh: float = 0.45,
    contain_thresh: float = 0.55,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """Suppress duplicate boxes AND smaller sub-boxes swallowed inside a larger detection."""
    ordered = sorted(items, key=lambda x: (x[1], box_area(x[0])), reverse=True)
    kept: list[tuple[tuple[int, int, int, int], float, int]] = []
    for box, score, label in ordered:
        duplicate = False
        for k_box, _, _ in kept:
            if iou(box, k_box) >= iou_thresh:
                duplicate = True
                break
            inter = intersection_area(box, k_box)
            min_area = min(box_area(box), box_area(k_box))
            if min_area > 0 and (inter / min_area) >= contain_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append((box, score, label))
    return kept


def _compute_tile_windows(w: int, h: int, max_side: int = 960, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Slice ultra-tall webtoons or ultra-wide strips into overlapping tiles."""
    if w <= max_side and h <= max_side and max(w, h) / max(1, min(w, h)) <= 1.8:
        return [(0, 0, w, h)]

    tile_size = min(max(min(w, h) * 2, 640), 1024)
    tile_w = min(w, tile_size)
    tile_h = min(h, tile_size)
    step_x = max(1, int(tile_w * (1.0 - overlap)))
    step_y = max(1, int(tile_h * (1.0 - overlap)))

    xs = [0] if w <= tile_w else list(range(0, max(1, w - tile_w), step_x)) + [w - tile_w]
    ys = [0] if h <= tile_h else list(range(0, max(1, h - tile_h), step_y)) + [h - tile_h]

    windows: list[tuple[int, int, int, int]] = [(0, 0, w, h)]
    seen: set[tuple[int, int, int, int]] = {(0, 0, w, h)}
    for y in ys:
        for x in xs:
            win = (x, y, min(w, x + tile_w), min(h, y + tile_h))
            if win not in seen:
                seen.add(win)
                windows.append(win)
    return windows


def _flood_fill_bubble_contour(
    rgb: np.ndarray,
    text_mask: np.ndarray,
    seed_box: tuple[int, int, int, int],
    known_bubble_box: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int, int, int] | None, list[tuple[float, float]] | None, bool]:
    """
    Recover the true interior polygon of a speech bubble (white OR dark translucent).

    Fixes Koharu Bug #1 & #3 where failed bubble containment forced Vietnamese text
    into the narrow original Japanese column box.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Determine search ROI
    if known_bubble_box is not None:
        bx1, by1, bx2, by2 = known_bubble_box
        roi_x1 = max(0, bx1 - 4)
        roi_y1 = max(0, by1 - 4)
        roi_x2 = min(w, bx2 + 4)
        roi_y2 = min(h, by2 + 4)
    else:
        tx1, ty1, tx2, ty2 = seed_box
        tw, th = tx2 - tx1, ty2 - ty1
        pad_x = max(24, int(tw * 0.9))
        pad_y = max(24, int(th * 0.9))
        roi_x1 = max(0, tx1 - pad_x)
        roi_y1 = max(0, ty1 - pad_y)
        roi_x2 = min(w, tx2 + pad_x)
        roi_y2 = min(h, ty2 + pad_y)

    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_mask = text_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    rh, rw = roi_gray.shape[:2]
    if rh < 12 or rw < 12:
        return known_bubble_box, None, False

    # Sample non-text pixels inside the text_bbox to identify bubble interior brightness
    lx1 = max(0, seed_box[0] - roi_x1)
    ly1 = max(0, seed_box[1] - roi_y1)
    lx2 = min(rw, seed_box[2] - roi_x1)
    ly2 = min(rh, seed_box[3] - roi_y1)

    sub_gray = roi_gray[ly1:ly2, lx1:lx2]
    sub_tmask = roi_mask[ly1:ly2, lx1:lx2]
    bg_pixels = sub_gray[sub_tmask < 64]
    if bg_pixels.size < 16:
        bg_pixels = sub_gray.ravel()

    bg_median = np.median(roi_gray[ly1:ly2, lx1:lx2][sub_tmask < 64]) if np.any(sub_tmask < 64) else np.median(roi_gray[ly1:ly2, lx1:lx2])
    bg_median = float(bg_median)
    bg_pixels_rgb = rgb[roi_y1:roi_y2, roi_x1:roi_x2][ly1:ly2, lx1:lx2]
    if np.any(sub_tmask < 64):
        bg_rgb_sample = bg_pixels_rgb[sub_tmask < 64]
    else:
        bg_rgb_sample = bg_pixels_rgb.reshape(-1, 3)
    bg_med_rgb = np.median(bg_rgb_sample, axis=0)
    bg_sat = float(np.max(bg_med_rgb) - np.min(bg_med_rgb))
    bg_std = float(np.std(bg_pixels))
    is_dark = bg_median < 115.0

    # If RT-DETR detected NO bubble around this text, only accept flood-fill if the background
    # is genuinely a neutral monochrome white/black bubble interior (not saturated sky/skin/hair!).
    if known_bubble_box is None:
        if bg_sat > 24.0:
            return None, None, is_dark
        if bg_std > 42.0 and not (bg_median > 220.0 or bg_median < 60.0):
            return None, None, is_dark

    # Temporarily inpaint text pixels in roi_gray so flood-fill flows smoothly across letters
    clean_roi = cv2.inpaint(roi_gray, (roi_mask > 80).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

    # Pick a seed point inside the text box whose value is closest to bg_median
    seed_crop = np.abs(clean_roi[ly1:ly2, lx1:lx2].astype(np.float32) - bg_median)
    sy_rel, sx_rel = np.unravel_index(np.argmin(seed_crop), seed_crop.shape)
    seed_pt = (int(lx1 + sx_rel), int(ly1 + sy_rel))

    ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
    tol = 20 if is_dark else 24
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    _, _, ff_mask, _ = cv2.floodFill(clean_roi.copy(), ff_mask, seed_pt, 255, tol, tol, flags)
    filled = ff_mask[1:-1, 1:-1]

    # Clean holes and extract largest contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    area = int(np.count_nonzero(filled))
    text_area = max(1, (lx2 - lx1) * (ly2 - ly1))
    roi_area = rw * rh

    # Guard against flood-fill leaking out into the full page background
    leaked = area > roi_area * 0.88 and known_bubble_box is None
    if leaked or area < text_area * 0.65:
        if known_bubble_box is not None:
            return known_bubble_box, None, is_dark
        return None, None, is_dark

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return known_bubble_box, None, is_dark

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    c_area = cv2.contourArea(largest)
    circularity = (4.0 * np.pi * c_area) / max(1e-5, peri * peri)
    if known_bubble_box is None and circularity < 0.25:
        return None, None, is_dark
    approx = cv2.approxPolyDP(largest, 0.012 * peri, True)
    pts = [(float(p[0][0] + roi_x1), float(p[0][1] + roi_y1)) for p in approx]

    x, y, cw, ch = cv2.boundingRect(largest)
    filled_box = (roi_x1 + x, roi_y1 + y, roi_x1 + x + cw, roi_y1 + y + ch)

    if known_bubble_box is not None:
        # Never let a partial translucent flood-fill shrink RT-DETR's bubble box or chop off text
        kb = known_bubble_box
        if (
            containment_ratio(seed_box, filled_box) >= 0.96
            and box_area(filled_box) >= box_area(kb) * 0.72
            and containment_ratio(filled_box, (kb[0] - 8, kb[1] - 8, kb[2] + 8, kb[3] + 8)) >= 0.45
        ):
            return filled_box, pts if len(pts) >= 4 else None, is_dark
        return known_bubble_box, None, is_dark

    # Accept flood-filled contour only if it surrounds the text cleanly and isn't overly jagged
    if containment_ratio(seed_box, filled_box) >= 0.70 and area <= text_area * 9.0:
        return filled_box, pts if len(pts) >= 4 else None, is_dark

    return None, None, is_dark


def _expand_box_with_seg_mask(
    box: tuple[int, int, int, int],
    seg_mask: np.ndarray,
    bounds: tuple[int, int, int, int] | None = None,
    max_expand: int = 28,
) -> tuple[int, int, int, int]:
    """
    Expand an RT-DETR text box to enclose connected glyph pixels from CTD `seg_mask`
    that overlap the box (fixes clipped top/bottom characters on vertical text/SFX).
    """
    h, w = seg_mask.shape[:2]
    x1, y1, x2, y2 = box
    rx1 = max(0, x1 - max_expand)
    ry1 = max(0, y1 - max_expand)
    rx2 = min(w, x2 + max_expand)
    ry2 = min(h, y2 + max_expand)
    if bounds is not None:
        rx1 = max(rx1, bounds[0] + 2)
        ry1 = max(ry1, bounds[1] + 2)
        rx2 = min(rx2, bounds[2] - 2)
        ry2 = min(ry2, bounds[3] - 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return box

    roi = (seg_mask[ry1:ry2, rx1:rx2] > 42).astype(np.uint8) * 255
    if np.count_nonzero(roi) == 0:
        return box

    # Local coordinates of the original box inside ROI
    ox1, oy1 = max(0, x1 - rx1), max(0, y1 - ry1)
    ox2, oy2 = min(rx2 - rx1, x2 - rx1), min(ry2 - ry1, y2 - ry1)

    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
    nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    for lbl in range(1, num_lbl):
        cx, cy, cw, ch, area = stats[lbl]
        if area < 6:
            continue
        # Check if component overlaps the core box (or sits within 4px of it)
        if cx + cw >= ox1 - 4 and cx <= ox2 + 4 and cy + ch >= oy1 - 4 and cy <= oy2 + 4:
            # Verify component actually belongs to the glyphs inside the box
            comp_in_orig = labels[max(0, oy1 - 3) : min(ry2 - ry1, oy2 + 3), max(0, ox1 - 3) : min(rx2 - rx1, ox2 + 3)]
            if np.any(comp_in_orig == lbl):
                nx1 = min(nx1, rx1 + int(cx))
                ny1 = min(ny1, ry1 + int(cy))
                nx2 = max(nx2, rx1 + int(cx) + int(cw))
                ny2 = max(ny2, ry1 + int(cy) + int(ch))
    return (int(max(0, nx1)), int(max(0, ny1)), int(min(w, nx2)), int(min(h, ny2)))


def _enhance_contrast(rgb: np.ndarray) -> np.ndarray:
    """Apply gentle LAB-luminance CLAHE to boost faint translucent bubbles & colored SFX."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _tile_clip_score(
    score: float,
    box: tuple[int, int, int, int],
    tile_rect: tuple[int, int, int, int],
    full_w: int,
    full_h: int,
) -> float:
    """Downweight boxes clipped along an interior tile boundary so unclipped boxes win NMS."""
    tx1, ty1, tx2, ty2 = tile_rect
    if (tx1, ty1, tx2, ty2) == (0, 0, full_w, full_h):
        return score
    x1, y1, x2, y2 = box
    clipped = (
        (tx1 > 0 and x1 <= tx1 + 3)
        or (ty1 > 0 and y1 <= ty1 + 3)
        or (tx2 < full_w and x2 >= tx2 - 3)
        or (ty2 < full_h and y2 >= ty2 - 3)
    )
    return score * 0.78 if clipped else score


def _promote_complete_boxes(
    items: list[tuple[tuple[int, int, int, int], float, int]],
    text_mask: np.ndarray,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """
    When a lower-confidence complete box `L` encloses a higher-confidence partial fragment `S`
    AND the uncovered extra region `L \\ S` actually contains text pixels in `text_mask`,
    promote `L` so `_containment_nms` does not discard top/bottom characters of vertical lines.
    """
    if len(items) < 2:
        return items

    h, w = text_mask.shape[:2]
    fg = text_mask > 65
    promoted: list[tuple[tuple[int, int, int, int], float, int]] = []
    for box_l, sc_l, lbl_l in items:
        best_sc = sc_l
        area_l = box_area(box_l)
        if sc_l >= 0.24 and area_l > 0:
            for box_s, sc_s, _ in items:
                if sc_s <= best_sc:
                    continue
                area_s = box_area(box_s)
                if area_s <= 0 or area_l <= area_s * 1.15 or area_l > area_s * 4.5:
                    continue
                if containment_ratio(box_s, box_l) < 0.80:
                    continue
                lx1, ly1, lx2, ly2 = max(0, box_l[0]), max(0, box_l[1]), min(w, box_l[2]), min(h, box_l[3])
                sx1, sy1, sx2, sy2 = max(lx1, box_s[0]), max(ly1, box_s[1]), min(lx2, box_s[2]), min(ly2, box_s[3])
                sub_fg = fg[ly1:ly2, lx1:lx2].copy()
                sub_fg[sy1 - ly1 : sy2 - ly1, sx1 - lx1 : sx2 - lx1] = False
                extra_area = max(1, (lx2 - lx1) * (ly2 - ly1) - (sx2 - sx1) * (sy2 - sy1))
                extra_px = int(np.count_nonzero(sub_fg))
                if extra_px >= 18 and (extra_px / extra_area) >= 0.03:
                    best_sc = max(best_sc, sc_s + 0.01)
        promoted.append((box_l, best_sc, lbl_l))
    return promoted


def _discover_unboxed_text_regions(
    rgb: np.ndarray,
    text_mask: np.ndarray,
    existing_boxes: list[tuple[int, int, int, int]],
) -> list[DetectedRegion]:
    """Recover standalone glyph clusters lit up in UNet+DBNet `text_mask` that box detectors missed."""
    h, w = text_mask.shape[:2]
    uncovered = (text_mask > 105).astype(np.uint8) * 255
    for ex1, ey1, ex2, ey2 in existing_boxes:
        pad = 5
        uncovered[max(0, ey1 - pad) : min(h, ey2 + pad), max(0, ex1 - pad) : min(w, ex2 + pad)] = 0

    if np.count_nonzero(uncovered) < 120:
        return []

    closed = cv2.morphologyEx(uncovered, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    recovered: list[DetectedRegion] = []
    page_area = h * w
    for lbl in range(1, num_lbl):
        cx, cy, cw, ch, area = (int(v) for v in stats[lbl])
        if cw < 12 or ch < 12 or area < 140 or area > page_area * 0.25:
            continue
        aspect = cw / max(1, ch)
        if aspect < 0.06 or aspect > 15.0:
            continue
        box = (max(0, cx - 2), max(0, cy - 2), min(w, cx + cw + 2), min(h, cy + ch + 2))
        crop_m = text_mask[box[1] : box[3], box[0] : box[2]]
        if crop_m.size == 0 or int(crop_m.max()) < 155:
            continue
        refined_box, contour, is_dark = _flood_fill_bubble_contour(rgb, text_mask, box, None)
        recovered.append(
            DetectedRegion(
                text_bbox=box,
                bubble_bbox=refined_box,
                bubble_contour=contour,
                is_bubble=refined_box is not None,
                is_dark_bubble=is_dark,
                confidence=0.45,
            )
        )
    return recovered


def _to_prob_01(arr: np.ndarray) -> np.ndarray:
    """Convert raw logits or probability array to clipped `[0.0, 1.0]` probabilities."""
    if arr.size and (arr.min() < -0.05 or arr.max() > 1.05):
        return 1.0 / (1.0 + np.exp(-np.clip(arr, -20.0, 20.0)))
    return np.clip(arr, 0.0, 1.0)


def _decode_ctd_blk(
    blk: np.ndarray,
    scale: float,
    crop_w: int,
    crop_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    conf_thresh: float = 0.36,
) -> list[tuple[tuple[int, int, int, int], float, int]]:
    """Decode ComicTextDetector YOLOv5 `blk` head `[1, N, 7]` into deduplicated text boxes."""
    arr = np.asarray(blk, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 6 or len(arr) == 0 or scale <= 0:
        return []

    obj = arr[:, 4]
    cls_conf = np.max(arr[:, 5:], axis=1)
    scores = obj * cls_conf
    keep = (scores >= conf_thresh) & (arr[:, 2] >= 4.0) & (arr[:, 3] >= 4.0)
    if not np.any(keep):
        return []

    hits: list[tuple[tuple[int, int, int, int], float, int]] = []
    inv_scale = 1.0 / scale
    for row, sc in zip(arr[keep], scores[keep]):
        cx, cy, rw, rh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        x1 = (cx - rw * 0.5) * inv_scale
        y1 = (cy - rh * 0.5) * inv_scale
        x2 = (cx + rw * 0.5) * inv_scale
        y2 = (cy + rh * 0.5) * inv_scale
        ix1 = max(0, min(crop_w - 1, int(round(x1))))
        iy1 = max(0, min(crop_h - 1, int(round(y1))))
        ix2 = max(ix1 + 1, min(crop_w, int(round(x2))))
        iy2 = max(iy1 + 1, min(crop_h, int(round(y2))))
        if (ix2 - ix1) * (iy2 - iy1) < 36:
            continue
        box = (ix1 + offset_x, iy1 + offset_y, ix2 + offset_x, iy2 + offset_y)
        # Use label=2 (`text_free`) so standalone hits don't fake speech bubbles while
        # hits inside an RT-DETR bubble still automatically group into `bubble_groups`.
        hits.append((box, float(sc), 2))
    return _containment_nms(hits, iou_thresh=0.45, contain_thresh=0.65)


class ComicDetector:
    """Combined RT-DETRv2 bubble/text detector + full 3-head ComicTextDetector (`blk` + UNet `seg` + DBNet `det`)."""

    def __init__(self, accurate: bool = True, use_gpu: bool = True) -> None:
        det_filename = "detector.onnx" if accurate else "detector_int8.onnx"
        det_path = hf_hub_download("ogkalu/comic-text-and-bubble-detector", det_filename)
        self._rtdetr = _create_ort_session(det_path, use_gpu=use_gpu)

        ctd_path = _ensure_ctd_model()
        self._ctd = _create_ort_session(ctd_path, use_gpu=use_gpu)

    def _run_ctd_crop(
        self,
        rgb_crop: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        blk_conf_thresh: float = 0.36,
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], float, int]]]:
        h, w = rgb_crop.shape[:2]
        scale = 1024.0 / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(rgb_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((1024, 1024, 3), dtype=np.float32)
        canvas[:nh, :nw] = resized.astype(np.float32) / 255.0
        inp = np.transpose(canvas, (2, 0, 1))[None]

        blk, seg, det = self._ctd.run(None, {"images": inp})
        seg_prob = _to_prob_01(seg[0, 0, :nh, :nw])
        det_prob = _to_prob_01(det[0, 0, :nh, :nw])
        fused_prob = np.maximum(seg_prob, det_prob)
        seg_full = cv2.resize(fused_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(seg_full * 255.0, 0, 255).astype(np.uint8)
        blk_hits = _decode_ctd_blk(
            blk, scale, w, h, offset_x=offset_x, offset_y=offset_y, conf_thresh=blk_conf_thresh
        )
        return mask, blk_hits

    def detect_ctd(
        self,
        rgb: np.ndarray,
        blk_conf_thresh: float = 0.28,
        enhanced_rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], float, int]]]:
        """Return fused uint8 (H, W) UNet+DBNet text mask and mask-verified CTD `blk` text boxes."""
        h, w = rgb.shape[:2]
        full_mask, raw_blks = self._run_ctd_crop(rgb, 0, 0, blk_conf_thresh=blk_conf_thresh)
        if max(h, w) > 1280 or max(w, h) / max(1, min(w, h)) > 1.8:
            for wx1, wy1, wx2, wy2 in _compute_tile_windows(w, h, max_side=1024, overlap=0.22):
                if (wx1, wy1, wx2, wy2) == (0, 0, w, h):
                    continue
                tile_mask, tile_blks = self._run_ctd_crop(
                    rgb[wy1:wy2, wx1:wx2], wx1, wy1, blk_conf_thresh=blk_conf_thresh
                )
                full_mask[wy1:wy2, wx1:wx2] = np.maximum(full_mask[wy1:wy2, wx1:wx2], tile_mask)
                raw_blks.extend(
                    (b, _tile_clip_score(s, b, (wx1, wy1, wx2, wy2), w, h), l) for b, s, l in tile_blks
                )

        # Contrast-boosted pass captures translucent/low-contrast kanji & colored SFX
        if enhanced_rgb is not None:
            enh_mask, enh_blks = self._run_ctd_crop(
                enhanced_rgb, 0, 0, blk_conf_thresh=max(0.22, blk_conf_thresh - 0.04)
            )
            full_mask = np.maximum(full_mask, enh_mask)
            raw_blks.extend((b, s * 0.94, l) for b, s, l in enh_blks)

        # Keep CTD `blk` hits backed by CTD's own fused UNet+DBNet mask to suppress texture/hair noise
        verified_blks: list[tuple[tuple[int, int, int, int], float, int]] = []
        for b_box, b_score, b_lbl in _containment_nms(raw_blks, iou_thresh=0.45, contain_thresh=0.65):
            x1, y1, x2, y2 = b_box
            seg_crop = full_mask[y1:y2, x1:x2]
            if seg_crop.size == 0:
                continue
            seg_ratio = float(np.count_nonzero(seg_crop > 65)) / float(seg_crop.size)
            if seg_ratio >= 0.02:
                verified_blks.append((b_box, b_score, b_lbl))
        return full_mask, verified_blks

    def detect_text_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Return a uint8 (H, W) pixel probability mask [0..255] using multi-tile CTD (UNet + DBNet)."""
        return self.detect_ctd(rgb)[0]

    def _run_rtdetr_tile(
        self,
        rgb_tile: np.ndarray,
        offset_x: int,
        offset_y: int,
        conf_thresh: float,
        full_w: int | None = None,
        full_h: int | None = None,
    ) -> tuple[list[tuple[tuple[int, int, int, int], float, int]], list[tuple[tuple[int, int, int, int], float, int]]]:
        th, tw = rgb_tile.shape[:2]
        fw = full_w if full_w is not None else offset_x + tw
        fh = full_h if full_h is not None else offset_y + th
        tile_rect = (offset_x, offset_y, offset_x + tw, offset_y + th)

        resized = cv2.resize(rgb_tile, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        inp = np.transpose(resized, (2, 0, 1))[None]
        labels, boxes, scores = self._rtdetr.run(
            None,
            {"images": inp, "orig_target_sizes": np.array([[tw, th]], dtype=np.int64)},
        )
        bubbles: list[tuple[tuple[int, int, int, int], float, int]] = []
        texts: list[tuple[tuple[int, int, int, int], float, int]] = []
        for lbl, box, sc in zip(labels[0], boxes[0], scores[0]):
            score = float(sc)
            label = int(lbl)
            min_score = 0.18 if label in (1, 2) else conf_thresh
            if score < min_score:
                continue
            x1, y1, x2, y2 = float(box[0]) + offset_x, float(box[1]) + offset_y, float(box[2]) + offset_x, float(box[3]) + offset_y
            clamped = _clamp_box((x1, y1, x2, y2), offset_x + tw, offset_y + th)
            if box_area(clamped) < 36:
                continue
            ranked_score = _tile_clip_score(score, clamped, tile_rect, fw, fh)
            if label == 0:
                bubbles.append((clamped, ranked_score, label))
            elif label in (1, 2):
                texts.append((clamped, ranked_score, label))
        return bubbles, texts

    def detect(
        self,
        rgb: np.ndarray,
        conf_thresh: float = 0.34,
    ) -> tuple[list[DetectedRegion], np.ndarray]:
        """Detect deduplicated text/bubble regions and pixel-level text mask."""
        h, w = rgb.shape[:2]
        enhanced = _enhance_contrast(rgb)
        text_mask, ctd_texts = self.detect_ctd(rgb, blk_conf_thresh=0.26, enhanced_rgb=enhanced)

        raw_bubbles: list[tuple[tuple[int, int, int, int], float, int]] = []
        raw_texts: list[tuple[tuple[int, int, int, int], float, int]] = list(ctd_texts)

        for wx1, wy1, wx2, wy2 in _compute_tile_windows(w, h):
            b_hits, t_hits = self._run_rtdetr_tile(
                rgb[wy1:wy2, wx1:wx2], wx1, wy1, conf_thresh, full_w=w, full_h=h
            )
            raw_bubbles.extend(b_hits)
            raw_texts.extend(t_hits)

        # Secondary contrast pass captures faint/translucent bubbles & colored SFX
        cb_hits, ct_hits = self._run_rtdetr_tile(enhanced, 0, 0, conf_thresh, full_w=w, full_h=h)
        raw_bubbles.extend((b, s * 0.94, l) for b, s, l in cb_hits)
        raw_texts.extend((b, s * 0.94, l) for b, s, l in ct_hits)

        bubbles = _containment_nms(raw_bubbles, iou_thresh=0.45, contain_thresh=0.70)
        raw_texts = _promote_complete_boxes(raw_texts, text_mask)
        texts = _containment_nms(raw_texts, iou_thresh=0.40, contain_thresh=0.52)

        # Verify low-confidence text candidates (0.18..conf_thresh) against fused UNet+DBNet text_mask
        verified_texts: list[tuple[tuple[int, int, int, int], float, int]] = []
        for t_box, t_score, t_lbl in texts:
            if t_score < conf_thresh:
                x1, y1, x2, y2 = t_box
                seg_crop = text_mask[y1:y2, x1:x2]
                seg_ratio = float(np.count_nonzero(seg_crop > 65)) / max(1, seg_crop.size)
                if seg_ratio < 0.025:
                    continue
            verified_texts.append((t_box, t_score, t_lbl))
        texts = verified_texts

        # Group text boxes belonging to the same speech bubble so multi-column
        # Japanese/Chinese text in one bubble becomes ONE dialogue unit (fixes overlapping text!).
        bubble_groups: dict[int, list[tuple[tuple[int, int, int, int], float, int]]] = {}
        standalone_texts: list[tuple[tuple[int, int, int, int], float, int]] = []

        for t_box, t_score, t_lbl in texts:
            best_b_idx: int | None = None
            best_score = 0.0
            for idx, (b_box, _, _) in enumerate(bubbles):
                c = containment_ratio(t_box, b_box)
                if c >= 0.55 and c > best_score:
                    best_score = c
                    best_b_idx = idx
            if best_b_idx is not None:
                bubble_groups.setdefault(best_b_idx, []).append((t_box, t_score, t_lbl))
            else:
                standalone_texts.append((t_box, t_score, t_lbl))

        regions: list[DetectedRegion] = []

        # 1. Bubble-contained text groups
        for b_idx, members in bubble_groups.items():
            b_box, b_score, _ = bubbles[b_idx]
            merged_tbox = union_box([m[0] for m in members])
            merged_tbox = _expand_box_with_seg_mask(merged_tbox, text_mask, bounds=b_box, max_expand=24)
            avg_conf = max(b_score, max(m[1] for m in members))
            refined_box, contour, is_dark = _flood_fill_bubble_contour(rgb, text_mask, merged_tbox, b_box)
            regions.append(
                DetectedRegion(
                    text_bbox=merged_tbox,
                    bubble_bbox=refined_box or b_box,
                    bubble_contour=contour,
                    is_bubble=True,
                    is_dark_bubble=is_dark,
                    confidence=avg_conf,
                )
            )

        # 2. Standalone texts (may still sit inside an undetected white/dark translucent bubble!)
        for t_box, t_score, t_lbl in standalone_texts:
            expanded_tbox = _expand_box_with_seg_mask(t_box, text_mask, bounds=None, max_expand=28)
            refined_box, contour, is_dark = _flood_fill_bubble_contour(rgb, text_mask, expanded_tbox, None)
            is_bubble = refined_box is not None or t_lbl == 1
            regions.append(
                DetectedRegion(
                    text_bbox=expanded_tbox,
                    bubble_bbox=refined_box,
                    bubble_contour=contour,
                    is_bubble=is_bubble,
                    is_dark_bubble=is_dark,
                    confidence=t_score,
                )
            )

        # 3. Fallback: recover strong UNet+DBNet glyph clusters missed by box detectors
        regions.extend(_discover_unboxed_text_regions(rgb, text_mask, [r.text_bbox for r in regions]))

        # Sort in Manga reading order: top-to-bottom bands, right-to-left within bands
        regions.sort(key=lambda r: (r.text_bbox[1] // max(40, h // 8), -((r.text_bbox[0] + r.text_bbox[2]) // 2)))
        return regions, text_mask
