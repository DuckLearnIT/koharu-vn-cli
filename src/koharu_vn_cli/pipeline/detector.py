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
    line_bboxes: list[tuple[int, int, int, int]] | None = None


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
    low_mem: bool = True,
) -> ort.InferenceSession:
    """Create an ONNXRuntime session preferring GPU (CUDA/DirectML) with automatic CPU fallback."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if low_mem:
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
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
            cpu_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
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
        roi_x1 = max(0, bx1 - 6)
        roi_y1 = max(0, by1 - 6)
        roi_x2 = min(w, bx2 + 6)
        roi_y2 = min(h, by2 + 6)
    else:
        tx1, ty1, tx2, ty2 = seed_box
        tw, th = tx2 - tx1, ty2 - ty1
        pad_x = max(28, int(tw * 0.95))
        pad_y = max(28, int(th * 0.95))
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

    # Temporarily erase text strokes in roi_gray so flood-fill flows smoothly across dense dialogue columns
    text_erase = cv2.dilate(
        (roi_mask > 38).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    if (bg_median > 195.0 or bg_median < 55.0) and bg_std < 38.0:
        clean_roi = roi_gray.copy()
        clean_roi[text_erase > 0] = int(round(bg_median))
    else:
        clean_roi = cv2.inpaint(roi_gray, text_erase, 4, cv2.INPAINT_TELEA)

    # Pick up to 3 candidate seed points (left/center/right inside seed_box) closest to bg_median
    # so a dense vertical column splitting a bubble in half cannot trap flood-fill on one side.
    seed_diff = np.abs(clean_roi[ly1:ly2, lx1:lx2].astype(np.float32) - bg_median)
    sh, sw = seed_diff.shape
    tol = 22 if is_dark else 26
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE

    seed_pts: list[tuple[int, int]] = []
    if sw >= 18:
        bands = [(0, sw // 3), (sw // 3, (2 * sw) // 3), ((2 * sw) // 3, sw)]
    else:
        bands = [(0, sw)]
    for b_left, b_right in bands:
        if b_right <= b_left:
            continue
        band_slice = seed_diff[:, b_left:b_right]
        sy_rel, sx_rel = np.unravel_index(np.argmin(band_slice), band_slice.shape)
        if float(band_slice[sy_rel, sx_rel]) <= tol * 0.9 or not seed_pts:
            seed_pts.append((int(lx1 + b_left + sx_rel), int(ly1 + sy_rel)))

    filled = np.zeros((rh, rw), dtype=np.uint8)
    roi_area = rw * rh
    for sp in seed_pts:
        ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        _, _, ff_mask, _ = cv2.floodFill(clean_roi.copy(), ff_mask, sp, 255, tol, tol, flags)
        cand = ff_mask[1:-1, 1:-1]
        cand_area = int(np.count_nonzero(cand))
        if known_bubble_box is None and cand_area > roi_area * 0.88:
            continue
        filled = cv2.bitwise_or(filled, cand)

    # Clean holes and extract largest contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    area = int(np.count_nonzero(filled))
    text_area = max(1, (lx2 - lx1) * (ly2 - ly1))

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
    max_expand: int = 34,
) -> tuple[int, int, int, int]:
    """
    Expand an RT-DETR text box to enclose connected glyph pixels from CTD `seg_mask`
    that overlap the box (fixes clipped top/bottom characters, punctuation & furigana).
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

    raw_roi = (seg_mask[ry1:ry2, rx1:rx2] > 28).astype(np.uint8) * 255
    if np.count_nonzero(raw_roi) == 0:
        return box

    # Bridge tiny gaps between main vertical strokes, ellipsis dots, heart marks & furigana
    v_bridge = cv2.morphologyEx(raw_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
    roi = cv2.bitwise_or(
        v_bridge,
        cv2.morphologyEx(raw_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))),
    )

    # Local coordinates of the original box inside ROI
    ox1, oy1 = max(0, x1 - rx1), max(0, y1 - ry1)
    ox2, oy2 = min(rx2 - rx1, x2 - rx1), min(ry2 - ry1, y2 - ry1)

    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(roi)
    nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    for lbl in range(1, num_lbl):
        cx, cy, cw, ch, area = stats[lbl]
        if area < 6:
            continue
        # Check if component overlaps the core box (or sits within 5px of it)
        if cx + cw >= ox1 - 5 and cx <= ox2 + 5 and cy + ch >= oy1 - 5 and cy <= oy2 + 5:
            # Verify component actually belongs to the glyphs inside the box
            comp_in_orig = labels[max(0, oy1 - 4) : min(ry2 - ry1, oy2 + 4), max(0, ox1 - 4) : min(rx2 - rx1, ox2 + 4)]
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
        bw_l = box_l[2] - box_l[0]
        if sc_l >= 0.24 and area_l > 0:
            for box_s, sc_s, _ in items:
                if sc_s <= best_sc:
                    continue
                area_s = box_area(box_s)
                bw_s = max(1, box_s[2] - box_s[0])
                if area_s <= 0 or area_l <= area_s * 1.15 or area_l > area_s * 4.5:
                    continue
                # Never merge two horizontally separate dialogue/floating columns into one wide mega-box
                if bw_l > max(bw_s * 1.95, 180):
                    continue
                if containment_ratio(box_s, box_l) < 0.80:
                    continue
                lx1, ly1, lx2, ly2 = max(0, box_l[0]), max(0, box_l[1]), min(w, box_l[2]), min(h, box_l[3])
                sx1, sy1, sx2, sy2 = max(lx1, box_s[0]), max(ly1, box_s[1]), min(lx2, box_s[2]), min(ly2, box_s[3])
                sub_fg = fg[ly1:ly2, lx1:lx2].copy()
                sub_fg[sy1 - ly1 : sy2 - ly1, sx1 - lx1 : sx2 - lx1] = False
                extra_area = max(1, (lx2 - lx1) * (ly2 - ly1) - (sx2 - sx1) * (sy2 - sy1))
                extra_px = int(np.count_nonzero(sub_fg))
                if extra_px >= 18 and (extra_px / extra_area) >= 0.035:
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
    uncovered = (text_mask > 70).astype(np.uint8) * 255
    for ex1, ey1, ex2, ey2 in existing_boxes:
        pad = 6
        uncovered[max(0, ey1 - pad) : min(h, ey2 + pad), max(0, ex1 - pad) : min(w, ex2 + pad)] = 0

    if np.count_nonzero(uncovered) < 85:
        return []

    closed = cv2.morphologyEx(uncovered, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 13)))
    num_lbl, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    recovered: list[DetectedRegion] = []
    page_area = h * w
    for lbl in range(1, num_lbl):
        cx, cy, cw, ch, area = (int(v) for v in stats[lbl])
        if cw < 12 or ch < 12 or area < 90 or area > page_area * 0.28:
            continue
        aspect = cw / max(1, ch)
        if aspect < 0.05 or aspect > 16.0:
            continue
        box = (max(0, cx - 2), max(0, cy - 2), min(w, cx + cw + 2), min(h, cy + ch + 2))
        crop_m = text_mask[box[1] : box[3], box[0] : box[2]]
        if crop_m.size == 0 or int(crop_m.max()) < 118:
            continue
        fg_ratio = float(np.count_nonzero(crop_m > 58)) / max(1, crop_m.size)
        if fg_ratio < 0.025:
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


def _merge_overlapping_bubble_regions(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """
    Merge duplicate detections that share the same physical speech bubble
    (e.g., an edge column missed by initial grouping then flood-filled into the same balloon).
    """
    if len(regions) < 2:
        return regions

    merged: list[DetectedRegion] = []
    for reg in sorted(
        regions,
        key=lambda r: (
            1 if r.is_bubble else 0,
            box_area(r.bubble_bbox or r.text_bbox),
            r.confidence,
        ),
        reverse=True,
    ):
        r_bbox = reg.bubble_bbox
        matched_idx: int | None = None
        for idx, kept in enumerate(merged):
            if not kept.is_bubble or kept.bubble_bbox is None:
                continue
            k_bbox = kept.bubble_bbox
            if r_bbox is not None and iou(r_bbox, k_bbox) >= 0.70:
                matched_idx = idx
                break
            if containment_ratio(reg.text_bbox, k_bbox) >= 0.65:
                matched_idx = idx
                break
            if r_bbox is not None and containment_ratio(kept.text_bbox, r_bbox) >= 0.65:
                matched_idx = idx
                break

        if matched_idx is None:
            merged.append(reg)
            continue

        kept = merged[matched_idx]
        new_tbox = union_box([kept.text_bbox, reg.text_bbox])
        if kept.bubble_bbox is not None and r_bbox is not None:
            if box_area(r_bbox) > box_area(kept.bubble_bbox):
                new_bbox = r_bbox
                new_contour = reg.bubble_contour or kept.bubble_contour
            else:
                new_bbox = kept.bubble_bbox
                new_contour = kept.bubble_contour or reg.bubble_contour
        else:
            new_bbox = kept.bubble_bbox or r_bbox
            new_contour = kept.bubble_contour or reg.bubble_contour

        new_lines: list[tuple[int, int, int, int]] = []
        for src in (kept, reg):
            if src.line_bboxes:
                new_lines.extend(src.line_bboxes)
            else:
                new_lines.append(src.text_bbox)

        merged[matched_idx] = DetectedRegion(
            text_bbox=new_tbox,
            bubble_bbox=new_bbox,
            bubble_contour=new_contour,
            is_bubble=True,
            is_dark_bubble=kept.is_dark_bubble,
            confidence=max(kept.confidence, reg.confidence),
            angle=kept.angle,
            line_bboxes=new_lines,
        )
    return merged


def _group_text_lines_into_blocks(
    items: list[tuple[tuple[int, int, int, int], float, int]],
) -> list[tuple[tuple[int, int, int, int], float, int, list[tuple[int, int, int, int]]]]:
    """
    Group adjacent vertical Japanese columns (or stacked horizontal lines) belonging to
    the same floating dialogue paragraph into a single translation block while retaining
    exact per-line bounding boxes for surgical inpainting.
    """
    if not items:
        return []
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        ax1, ay1, ax2, ay2 = items[i][0]
        aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
        a_vert = ah >= aw * 1.35
        a_horiz = aw >= ah * 1.35
        for j in range(i + 1, n):
            bx1, by1, bx2, by2 = items[j][0]
            bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
            b_vert = bh >= bw * 1.35
            b_horiz = bw >= bh * 1.35

            gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
            gap_y = max(0, max(ay1, by1) - min(ay2, by2))
            ov_x = max(0, min(ax2, bx2) - max(ax1, bx1))
            ov_y = max(0, min(ay2, by2) - max(ay1, by1))

            # Vertical manga/CG dialogue columns sitting side-by-side in the same paragraph
            if a_vert and b_vert:
                ref_w = (aw + bw) * 0.5
                min_h = max(1, min(ah, bh))
                top_diff = abs(ay1 - by1)
                mid_diff = abs((ay1 + ay2) * 0.5 - (by1 + by2) * 0.5)
                if (
                    gap_x <= max(26.0, ref_w * 0.72)
                    and (ov_y / min_h) >= 0.35
                    and (top_diff <= max(72.0, ref_w * 1.55) or mid_diff <= max(85.0, min_h * 0.30))
                ):
                    union(i, j)
                    continue

            # Horizontal stacked dialogue lines in the same paragraph
            if a_horiz and b_horiz:
                ref_h = (ah + bh) * 0.5
                min_w = max(1, min(aw, bw))
                if gap_y <= max(22.0, ref_h * 0.75) and (ov_x / min_w) >= 0.35:
                    union(i, j)
                    continue

    groups: dict[int, list[tuple[tuple[int, int, int, int], float, int]]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)

    blocks: list[tuple[tuple[int, int, int, int], float, int, list[tuple[int, int, int, int]]]] = []
    for members in groups.values():
        line_boxes = [m[0] for m in members]
        ubox = union_box(line_boxes)
        best_sc = max(m[1] for m in members)
        # Preserve bubble-text hint if any member had label=1
        lbl = 1 if any(m[2] == 1 for m in members) else 2
        blocks.append((ubox, best_sc, lbl, line_boxes))
    return blocks


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


YSG_REPO_ID = "YSGforMTL/YSGYoloDetector"
YSG_YOLO11S_FILE = "YOLOv11合/S/model.onnx"
YSG_YOLO11N_FILE = "YOLOv11合/N/110best.onnx"
YSG_RFDETR_LARGE_FILE = "RFDETR_1.0/RFDETR_3000_55_Large_ONNX/inference_model.sim.onnx"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ComicDetector:
    """
    Combined YSG YOLOv11-S 1024x1024 (accurate=True) or YSG YOLOv11-N (accurate=False)
    + ComicTextDetector UNet/DBNet pixel text mask.
    Lightweight (~36MB), 1024x1024 native resolution, safe on 8GB RAM laptops & 4GB GPUs.
    """

    def __init__(self, accurate: bool = True, use_gpu: bool = True) -> None:
        self._accurate = accurate
        self._ysg_yolo11s: ort.InferenceSession | None = None
        self._ysg_rfdetr: ort.InferenceSession | None = None
        self._rtdetr: ort.InferenceSession | None = None

        yolo_file = YSG_YOLO11S_FILE if accurate else YSG_YOLO11N_FILE
        y11_path = hf_hub_download(YSG_REPO_ID, yolo_file)
        self._ysg_yolo11s = _create_ort_session(y11_path, use_gpu=use_gpu, low_mem=True)
        self._rtdetr = self._ysg_yolo11s

        ctd_path = _ensure_ctd_model()
        self._ctd = _create_ort_session(ctd_path, use_gpu=use_gpu, low_mem=True)

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

        # Use standard 114/255 neutral gray padding instead of black so page borders never form false edges
        canvas = np.full((1024, 1024, 3), 114.0 / 255.0, dtype=np.float32)
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
        blk_conf_thresh: float = 0.34,
        enhanced_rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], float, int]]]:
        """Return uint8 (H, W) CTD text mask and mask-verified CTD `blk` text boxes."""
        h, w = rgb.shape[:2]
        full_mask, raw_blks = self._run_ctd_crop(rgb, 0, 0, blk_conf_thresh=blk_conf_thresh)
        # Only tile CTD on ultra-tall webtoons / extreme aspect ratios to avoid unnecessary RAM spikes
        if max(w, h) / max(1, min(w, h)) > 2.2 and max(h, w) > 1600:
            for wx1, wy1, wx2, wy2 in _compute_tile_windows(w, h, max_side=1280, overlap=0.20)[:6]:
                if (wx1, wy1, wx2, wy2) == (0, 0, w, h):
                    continue
                tile_mask, tile_blks = self._run_ctd_crop(
                    rgb[wy1:wy2, wx1:wx2], wx1, wy1, blk_conf_thresh=blk_conf_thresh
                )
                full_mask[wy1:wy2, wx1:wx2] = np.maximum(full_mask[wy1:wy2, wx1:wx2], tile_mask)
                raw_blks.extend(
                    (b, _tile_clip_score(s, b, (wx1, wy1, wx2, wy2), w, h), l) for b, s, l in tile_blks
                )

        if enhanced_rgb is not None:
            enh_mask, enh_blks = self._run_ctd_crop(
                enhanced_rgb, 0, 0, blk_conf_thresh=max(0.28, blk_conf_thresh - 0.04)
            )
            full_mask = np.maximum(full_mask, enh_mask)
            raw_blks.extend((b, s * 0.94, l) for b, s, l in enh_blks)

        verified_blks: list[tuple[tuple[int, int, int, int], float, int]] = []
        for b_box, b_score, b_lbl in _containment_nms(raw_blks, iou_thresh=0.45, contain_thresh=0.65):
            x1, y1, x2, y2 = b_box
            seg_crop = full_mask[y1:y2, x1:x2]
            if seg_crop.size == 0:
                continue
            seg_ratio = float(np.count_nonzero(seg_crop > 65)) / float(seg_crop.size)
            if seg_ratio >= 0.025:
                verified_blks.append((b_box, b_score, b_lbl))
        return full_mask, verified_blks

    def detect_text_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Return a uint8 (H, W) pixel probability mask [0..255] using CTD."""
        return self.detect_ctd(rgb)[0]

    def _run_ysg_yolo11s(
        self,
        rgb_tile: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        conf_thresh: float = 0.34,
        full_w: int | None = None,
        full_h: int | None = None,
    ) -> tuple[list[tuple[tuple[int, int, int, int], float, int]], list[tuple[tuple[int, int, int, int], float, int]]]:
        """Run YSG YOLOv11-S (1024x1024) returning (bubbles, text_lines)."""
        if self._ysg_yolo11s is None:
            return [], []
        th, tw = rgb_tile.shape[:2]
        fw = full_w if full_w is not None else offset_x + tw
        fh = full_h if full_h is not None else offset_y + th
        tile_rect = (offset_x, offset_y, offset_x + tw, offset_y + th)

        ishape = self._ysg_yolo11s.get_inputs()[0].shape
        ih = int(ishape[2]) if isinstance(ishape[2], int) else 1024
        iw = int(ishape[3]) if isinstance(ishape[3], int) else 1024
        scale = min(iw / max(1.0, float(tw)), ih / max(1.0, float(th)))
        nw, nh = max(1, int(round(tw * scale))), max(1, int(round(th * scale)))
        pad_x, pad_y = (iw - nw) // 2, (ih - nh) // 2

        canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = cv2.resize(
            rgb_tile, (nw, nh), interpolation=cv2.INTER_LINEAR
        )
        inp = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        arr = self._ysg_yolo11s.run(None, {self._ysg_yolo11s.get_inputs()[0].name: inp})[0][0]
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T  # [N, 4+C]

        cls_scores = arr[:, 4:]
        scores = np.max(cls_scores, axis=1)
        cids = np.argmax(cls_scores, axis=1)
        keep = np.where(scores >= max(0.30, conf_thresh - 0.02))[0]

        bubbles: list[tuple[tuple[int, int, int, int], float, int]] = []
        texts: list[tuple[tuple[int, int, int, int], float, int]] = []
        inv_scale = 1.0 / scale
        for idx in keep:
            cx, cy, bw, bh = (float(v) for v in arr[idx, :4])
            sc = float(scores[idx])
            cid = int(cids[idx])
            x1 = (cx - bw * 0.5 - pad_x) * inv_scale + offset_x
            y1 = (cy - bh * 0.5 - pad_y) * inv_scale + offset_y
            x2 = (cx + bw * 0.5 - pad_x) * inv_scale + offset_x
            y2 = (cy + bh * 0.5 - pad_y) * inv_scale + offset_y
            clamped = _clamp_box((x1, y1, x2, y2), offset_x + tw, offset_y + th)
            if box_area(clamped) < 48:
                continue
            ranked = _tile_clip_score(sc, clamped, tile_rect, fw, fh)
            if cid == 5:
                bubbles.append((clamped, ranked, 0))
            else:
                texts.append((clamped, ranked, 2))
        return bubbles, texts

    def _run_ysg_rfdetr(
        self,
        rgb_tile: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        conf_thresh: float = 0.36,
        full_w: int | None = None,
        full_h: int | None = None,
    ) -> tuple[list[tuple[tuple[int, int, int, int], float, int]], list[tuple[tuple[int, int, int, int], float, int]]]:
        """Run YSG RF-DETR Large (704x704) returning (bubbles, text_lines)."""
        if self._ysg_rfdetr is None:
            return [], []
        th, tw = rgb_tile.shape[:2]
        fw = full_w if full_w is not None else offset_x + tw
        fh = full_h if full_h is not None else offset_y + th
        tile_rect = (offset_x, offset_y, offset_x + tw, offset_y + th)

        ishape = self._ysg_rfdetr.get_inputs()[0].shape
        ih = int(ishape[2]) if isinstance(ishape[2], int) else 704
        iw = int(ishape[3]) if isinstance(ishape[3], int) else 704
        rz = cv2.resize(rgb_tile, (iw, ih), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        rz = (rz - _IMAGENET_MEAN) / _IMAGENET_STD
        inp = np.transpose(rz, (2, 0, 1))[None]
        dets, logits = self._ysg_rfdetr.run(None, {self._ysg_rfdetr.get_inputs()[0].name: inp})
        probs = _to_prob_01(logits[0])

        bubbles: list[tuple[tuple[int, int, int, int], float, int]] = []
        texts: list[tuple[tuple[int, int, int, int], float, int]] = []
        for i in range(len(probs)):
            cid = int(np.argmax(probs[i]))
            sc = float(probs[i, cid])
            min_sc = max(0.36, conf_thresh) if cid == 5 else max(0.34, conf_thresh)
            if sc < min_sc:
                continue
            cx, cy, bw, bh = (float(v) for v in dets[0, i])
            x1 = (cx - bw * 0.5) * tw + offset_x
            y1 = (cy - bh * 0.5) * th + offset_y
            x2 = (cx + bw * 0.5) * tw + offset_x
            y2 = (cy + bh * 0.5) * th + offset_y
            clamped = _clamp_box((x1, y1, x2, y2), offset_x + tw, offset_y + th)
            if box_area(clamped) < 48:
                continue
            ranked = _tile_clip_score(sc, clamped, tile_rect, fw, fh)
            if cid == 5:
                bubbles.append((clamped, ranked, 0))
            else:
                texts.append((clamped, ranked, 2))
        return bubbles, texts

    def detect(
        self,
        rgb: np.ndarray,
        conf_thresh: float = 0.34,
    ) -> tuple[list[DetectedRegion], np.ndarray]:
        """Detect deduplicated text/bubble regions and pixel-level text mask."""
        h, w = rgb.shape[:2]
        text_mask, ctd_texts = self.detect_ctd(rgb, blk_conf_thresh=0.38, enhanced_rgb=None)

        raw_bubbles: list[tuple[tuple[int, int, int, int], float, int]] = []
        raw_lines: list[tuple[tuple[int, int, int, int], float, int]] = []

        # Primary pass: YSG RF-DETR Large (accurate=True) or YSG YOLOv11-S 1024x1024 (accurate=False)
        # Only tile on ultra-tall webtoons (aspect > 2.2) to keep RAM usage compact and fast.
        is_webtoon = (max(w, h) / max(1, min(w, h)) > 2.2) and (max(w, h) > 1600)
        windows = _compute_tile_windows(w, h, max_side=1280, overlap=0.20) if is_webtoon else [(0, 0, w, h)]
        for wx1, wy1, wx2, wy2 in windows:
            tile = rgb[wy1:wy2, wx1:wx2]
            if self._ysg_rfdetr is not None:
                brf, trf = self._run_ysg_rfdetr(tile, wx1, wy1, conf_thresh=max(0.35, conf_thresh), full_w=w, full_h=h)
                raw_bubbles.extend(brf)
                raw_lines.extend(trf)
            if self._ysg_yolo11s is not None:
                b11, t11 = self._run_ysg_yolo11s(tile, wx1, wy1, conf_thresh=conf_thresh, full_w=w, full_h=h)
                raw_bubbles.extend(b11)
                raw_lines.extend(t11)

        # Deduplicate bubble containers and per-column/per-line boxes
        bubbles = _containment_nms(raw_bubbles, iou_thresh=0.45, contain_thresh=0.70)
        lines = _containment_nms(raw_lines, iou_thresh=0.45, contain_thresh=0.68)

        # Keep CTD fallback boxes ONLY where YSG detected nothing overlapping at all
        for c_box, c_sc, c_lbl in ctd_texts:
            if c_sc < 0.45:
                continue
            if any(iou(c_box, lb) >= 0.20 or containment_ratio(c_box, lb) >= 0.45 or containment_ratio(lb, c_box) >= 0.45 for lb, _, _ in lines):
                continue
            lines.append((c_box, c_sc * 0.90, c_lbl))

        # Drop tiny noise boxes with zero support in CTD text_mask unless YSG confidence is very high
        verified_lines: list[tuple[tuple[int, int, int, int], float, int]] = []
        for t_box, t_score, t_lbl in lines:
            x1, y1, x2, y2 = t_box
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            seg_crop = text_mask[y1:y2, x1:x2]
            seg_ratio = float(np.count_nonzero(seg_crop > 55)) / max(1, seg_crop.size)
            if t_score < 0.55 and seg_ratio < 0.015:
                continue
            verified_lines.append((t_box, t_score, t_lbl))
        lines = verified_lines

        # Group text lines belonging to the same detected speech bubble
        bubble_groups: dict[int, list[tuple[tuple[int, int, int, int], float, int]]] = {}
        standalone_lines: list[tuple[tuple[int, int, int, int], float, int]] = []

        for t_box, t_score, t_lbl in lines:
            best_b_idx: int | None = None
            best_score = 0.0
            tcx = (t_box[0] + t_box[2]) * 0.5
            tcy = (t_box[1] + t_box[3]) * 0.5
            for idx, (b_box, _, _) in enumerate(bubbles):
                c = containment_ratio(t_box, b_box)
                center_inside = (
                    (b_box[0] - 6) <= tcx <= (b_box[2] + 6)
                    and (b_box[1] - 6) <= tcy <= (b_box[3] + 6)
                )
                if (c >= 0.55 or (c >= 0.40 and center_inside)) and c > best_score:
                    best_score = c
                    best_b_idx = idx
            if best_b_idx is not None:
                bubble_groups.setdefault(best_b_idx, []).append((t_box, t_score, t_lbl))
            else:
                standalone_lines.append((t_box, t_score, t_lbl))

        regions: list[DetectedRegion] = []

        # 1. Bubble-contained text groups
        for b_idx, members in bubble_groups.items():
            b_box, b_score, _ = bubbles[b_idx]
            member_lines = [m[0] for m in members]
            merged_tbox = union_box(member_lines)
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
                    line_bboxes=member_lines,
                )
            )

        # 2. Standalone text lines -> group adjacent vertical columns / horizontal lines into dialogue blocks
        for block_box, block_score, block_lbl, block_lines in _group_text_lines_into_blocks(standalone_lines):
            expanded_tbox = _expand_box_with_seg_mask(block_box, text_mask, bounds=None, max_expand=20)
            refined_box, contour, is_dark = _flood_fill_bubble_contour(rgb, text_mask, expanded_tbox, None)
            is_bubble = refined_box is not None or block_lbl == 1
            regions.append(
                DetectedRegion(
                    text_bbox=expanded_tbox,
                    bubble_bbox=refined_box,
                    bubble_contour=contour,
                    is_bubble=is_bubble,
                    is_dark_bubble=is_dark,
                    confidence=block_score,
                    line_bboxes=block_lines,
                )
            )

        # 3. Fallback: recover strong UNet+DBNet glyph clusters completely missed by box detectors
        regions.extend(_discover_unboxed_text_regions(rgb, text_mask, [r.text_bbox for r in regions]))

        # 4. Merge any duplicate regions belonging to the exact same speech bubble
        regions = _merge_overlapping_bubble_regions(regions)

        # Sort in Manga reading order: top-to-bottom bands, right-to-left within bands
        regions.sort(key=lambda r: (r.text_bbox[1] // max(40, h // 8), -((r.text_bbox[0] + r.text_bbox[2]) // 2)))
        return regions, text_mask
