"""Text and speech-bubble detection using ONNX models + Containment NMS + Bubble Flood-Fill."""

from __future__ import annotations

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


def _ort_providers() -> list[str]:
    available = ort.get_available_providers()
    preferred = [
        p
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
        if p in available
    ]
    return preferred or ["CPUExecutionProvider"]


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
        # Intersect with known bubble box + small margin so contour stays inside detected bubble
        kb = known_bubble_box
        if containment_ratio(filled_box, (kb[0] - 8, kb[1] - 8, kb[2] + 8, kb[3] + 8)) >= 0.45:
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
                nx1 = min(nx1, rx1 + cx)
                ny1 = min(ny1, ry1 + cy)
                nx2 = max(nx2, rx1 + cx + cw)
                ny2 = max(ny2, ry1 + cy + ch)
    return (max(0, nx1), max(0, ny1), min(w, nx2), min(h, ny2))


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

    def __init__(self, accurate: bool = True) -> None:
        providers = _ort_providers()
        det_filename = "detector.onnx" if accurate else "detector_int8.onnx"
        det_path = hf_hub_download("ogkalu/comic-text-and-bubble-detector", det_filename)
        self._rtdetr = ort.InferenceSession(det_path, providers=providers)

        ctd_path = _ensure_ctd_model()
        self._ctd = ort.InferenceSession(str(ctd_path), providers=providers)

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
        self, rgb: np.ndarray, blk_conf_thresh: float = 0.36
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
                raw_blks.extend(tile_blks)

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
        self, rgb_tile: np.ndarray, offset_x: int, offset_y: int, conf_thresh: float
    ) -> tuple[list[tuple[tuple[int, int, int, int], float, int]], list[tuple[tuple[int, int, int, int], float, int]]]:
        th, tw = rgb_tile.shape[:2]
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
            min_score = 0.18 if label == 2 else conf_thresh
            if score < min_score:
                continue
            x1, y1, x2, y2 = float(box[0]) + offset_x, float(box[1]) + offset_y, float(box[2]) + offset_x, float(box[3]) + offset_y
            clamped = _clamp_box((x1, y1, x2, y2), offset_x + tw, offset_y + th)
            if box_area(clamped) < 36:
                continue
            if label == 0:
                bubbles.append((clamped, score, label))
            elif label in (1, 2):
                texts.append((clamped, score, label))
        return bubbles, texts

    def detect(
        self,
        rgb: np.ndarray,
        conf_thresh: float = 0.34,
    ) -> tuple[list[DetectedRegion], np.ndarray]:
        """Detect deduplicated text/bubble regions and pixel-level text mask."""
        h, w = rgb.shape[:2]
        text_mask, ctd_texts = self.detect_ctd(rgb, blk_conf_thresh=max(0.32, conf_thresh))

        raw_bubbles: list[tuple[tuple[int, int, int, int], float, int]] = []
        raw_texts: list[tuple[tuple[int, int, int, int], float, int]] = list(ctd_texts)

        for wx1, wy1, wx2, wy2 in _compute_tile_windows(w, h):
            b_hits, t_hits = self._run_rtdetr_tile(rgb[wy1:wy2, wx1:wx2], wx1, wy1, conf_thresh)
            raw_bubbles.extend(b_hits)
            raw_texts.extend(t_hits)

        bubbles = _containment_nms(raw_bubbles, iou_thresh=0.45, contain_thresh=0.70)
        texts = _containment_nms(raw_texts, iou_thresh=0.40, contain_thresh=0.52)

        # Verify low-confidence floating text candidates (0.18..conf_thresh) against text_mask
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
            merged_tbox = _expand_box_with_seg_mask(merged_tbox, text_mask, bounds=b_box, max_expand=16)
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
            expanded_tbox = _expand_box_with_seg_mask(t_box, text_mask, bounds=None, max_expand=26)
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

        # Sort in Manga reading order: top-to-bottom bands, right-to-left within bands
        regions.sort(key=lambda r: (r.text_bbox[1] // max(40, h // 8), -((r.text_bbox[0] + r.text_bbox[2]) // 2)))
        return regions, text_mask
