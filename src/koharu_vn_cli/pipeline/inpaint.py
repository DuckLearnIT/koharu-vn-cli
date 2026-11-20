"""Hybrid inpainting (fast solid bubble clean + LaMa ONNX) and background-aware color classification."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from koharu_vn_cli.pipeline.detector import DetectedRegion, _create_ort_session


def _luma(rgb: tuple[int, int, int] | np.ndarray) -> float:
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def _safe_bubble_interior_roi(
    reg: DetectedRegion,
    img_w: int,
    img_h: int,
    erode_px: int = 4,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """
    Build a strict interior mask for a speech bubble inset by `erode_px` from the border
    so aggressive ink erasure never nibbles into black balloon outlines.
    """
    if not reg.is_bubble:
        return None
    bbox = reg.bubble_bbox
    if bbox is None:
        return None
    bx1 = max(0, int(bbox[0]))
    by1 = max(0, int(bbox[1]))
    bx2 = min(img_w, int(bbox[2]))
    by2 = min(img_h, int(bbox[3]))
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 14 or bh < 14:
        return None

    interior = np.zeros((bh, bw), dtype=np.uint8)
    if reg.bubble_contour and len(reg.bubble_contour) >= 4:
        pts = np.array(
            [[int(round(px - bx1)), int(round(py - by1))] for px, py in reg.bubble_contour],
            dtype=np.int32,
        )
        cv2.fillPoly(interior, [pts], 255)
    else:
        # Fallback: superellipse interior matching oval balloons without touching rectangular corners
        yy, xx = np.ogrid[:bh, :bw]
        cx, cy = (bw - 1) * 0.5, (bh - 1) * 0.5
        rx = max(2.0, bw * 0.5 - 2.0)
        ry = max(2.0, bh * 0.5 - 2.0)
        p = 2.6
        inside = (np.abs((xx - cx) / rx) ** p + np.abs((yy - cy) / ry) ** p) <= 1.0
        interior[inside] = 255

    eff_erode = min(erode_px, max(1, min(bw, bh) // 10))
    k = eff_erode * 2 + 1
    interior = cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if np.count_nonzero(interior) < 24:
        return None
    return (bx1, by1, bx2, by2), interior


def _estimate_flat_bg(
    pixels_rgb: np.ndarray,
) -> tuple[bool, np.ndarray, float]:
    """
    Robustly determine whether background pixels form a flat solid color (white/black/tint)
    using percentile IQR instead of raw std so a few stray anti-aliased ink pixels never
    falsely force a clean white speech bubble into LaMa.
    Returns (is_flat, clean_fill_rgb, bg_luma).
    """
    if len(pixels_rgb) < 16:
        return False, np.array([255, 255, 255], dtype=np.uint8), 255.0

    arr = pixels_rgb.astype(np.float32)
    p15 = np.percentile(arr, 15, axis=0)
    p50 = np.percentile(arr, 50, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p85 = np.percentile(arr, 85, axis=0)
    p25 = np.percentile(arr, 25, axis=0)

    iqr = float(np.mean(p85 - p15))
    mad = float(np.mean(np.median(np.abs(arr - p50), axis=0)))
    luma_p75 = _luma(p75)
    luma_p50 = _luma(p50)
    luma_p25 = _luma(p25)

    # White / off-white speech bubble: tolerate up to ~18% stray ink pixels in ring
    if luma_p75 >= 224.0 and iqr <= 22.0 and mad <= 8.0:
        fill = np.clip(np.round(p85 if luma_p75 >= 240.0 else p75), 0, 255).astype(np.uint8)
        return True, fill, luma_p75

    # Solid black bubble
    if luma_p25 <= 32.0 and iqr <= 22.0 and mad <= 8.0:
        fill = np.clip(np.round(p15 if luma_p25 <= 16.0 else p25), 0, 255).astype(np.uint8)
        return True, fill, luma_p25

    # Uniform pastel / tinted bubble
    if iqr <= 12.0 and mad <= 5.0:
        fill = np.clip(np.round(p50), 0, 255).astype(np.uint8)
        return True, fill, luma_p50

    return False, np.clip(np.round(p50), 0, 255).astype(np.uint8), luma_p50


def build_erasure_mask(
    rgb: np.ndarray,
    seg_mask: np.ndarray,
    regions: list[DetectedRegion],
) -> np.ndarray:
    """
    Build a precise pixel mask covering original text glyphs (including thin vertical strokes,
    furigana & punctuation tails) inside detected boxes without erasing bubble borders or artwork.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for reg in regions:
        x1, y1, x2, y2 = reg.text_bbox
        # Expand slightly more on floating outlined glyphs to catch outer borders & furigana
        pad_box = 5 if reg.is_bubble else 8
        px1 = max(0, x1 - pad_box)
        py1 = max(0, y1 - pad_box)
        px2 = min(w, x2 + pad_box)
        py2 = min(h, y2 + pad_box)

        bubble_info = _safe_bubble_interior_roi(reg, w, h, erode_px=4) if reg.is_bubble else None
        # For speech bubbles with a known safe interior, expand scan ROI to cover the full
        # interior so thin vertical ink tails & furigana outside a tight text_bbox are completely erased.
        if bubble_info is not None:
            (bx1, by1, bx2, by2), safe_int = bubble_info
            px1 = min(px1, bx1)
            py1 = min(py1, by1)
            px2 = max(px2, bx2)
            py2 = max(py2, by2)
        else:
            safe_int = None

        if px2 <= px1 or py2 <= py1:
            continue

        roi_h, roi_w = py2 - py1, px2 - px1
        roi_seg = seg_mask[py1:py2, px1:px2]
        roi_gray = gray[py1:py2, px1:px2]
        roi_rgb = rgb[py1:py2, px1:px2]

        # Build ROI-aligned safe interior mask and text_bbox mask
        safe_roi: np.ndarray | None = None
        if bubble_info is not None and safe_int is not None:
            (bx1, by1, bx2, by2), _ = bubble_info
            safe_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            safe_roi[by1 - py1 : by2 - py1, bx1 - px1 : bx2 - px1] = safe_int

        tbox_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        tx1_l = max(0, x1 - pad_box - px1)
        ty1_l = max(0, y1 - pad_box - py1)
        tx2_l = min(roi_w, x2 + pad_box - px1)
        ty2_l = min(roi_h, y2 + pad_box - py1)
        if tx2_l > tx1_l and ty2_l > ty1_l:
            tbox_roi[ty1_l:ty2_l, tx1_l:tx2_l] = 255

        # Restrict seg seed to text_bbox union safe_interior so outer artwork never leaks in
        seed_allowed = cv2.bitwise_or(tbox_roi, safe_roi) if safe_roi is not None else tbox_roi
        local_mask = ((roi_seg > 36) & (seed_allowed > 0)).astype(np.uint8) * 255
        soft_mask = cv2.dilate(
            ((roi_seg > 16) & (seed_allowed > 0)).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        local_mask = cv2.bitwise_or(local_mask, cv2.bitwise_and(soft_mask, seed_allowed))

        # Capture contrast ink & outlined glyph borders inside bubbles OR floating text
        fg_ratio = np.count_nonzero(local_mask) / max(1, np.count_nonzero(seed_allowed))
        if reg.is_bubble or fg_ratio < 0.08:
            dilated_seed = cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            bg_mask = (seed_allowed > 0) & (dilated_seed == 0)
            if safe_roi is not None and np.count_nonzero((safe_roi > 0) & (dilated_seed == 0)) >= 24:
                bg_mask = (safe_roi > 0) & (dilated_seed == 0)
            interior_bg = roi_gray[bg_mask]
            interior_bg_rgb = roi_rgb[bg_mask]
            if interior_bg.size >= 24:
                bg_sample = interior_bg
                is_flat_bubble, _, _ = _estimate_flat_bg(interior_bg_rgb)
            else:
                bg_sample = np.concatenate([
                    roi_gray[0, :],
                    roi_gray[-1, :],
                    roi_gray[:, 0],
                    roi_gray[:, -1],
                ])
                is_flat_bubble = False

            # For bright flat bubbles, anchor background to upper quartile so ink never drags bg_med grey
            if is_flat_bubble and float(np.percentile(bg_sample, 75)) >= 210.0:
                bg_med = float(np.percentile(bg_sample, 80))
            elif is_flat_bubble and float(np.percentile(bg_sample, 25)) <= 45.0:
                bg_med = float(np.percentile(bg_sample, 20))
            else:
                bg_med = float(np.median(bg_sample)) if bg_sample.size > 0 else float(np.median(roi_gray))

            diff = np.abs(roi_gray.astype(np.float32) - bg_med)
            otsu_val, _ = cv2.threshold(
                np.clip(diff, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if reg.is_bubble and is_flat_bubble:
                thresh = max(11.0, min(26.0, float(otsu_val) * 0.48))
            elif reg.is_bubble:
                thresh = max(14.0, min(32.0, float(otsu_val) * 0.56))
            else:
                thresh = max(24.0, float(otsu_val))

            # On flat bubbles, clean ANY dark/contrast ink across the entire safe interior
            # (erases vertical stroke ghosts, furigana & punctuation outside tight text_bbox).
            ink_scope = safe_roi if (safe_roi is not None and is_flat_bubble) else tbox_roi
            extra = ((diff > thresh) & (ink_scope > 0)).astype(np.uint8) * 255
            local_mask = cv2.bitwise_or(local_mask, extra)
        else:
            # Floating outlined glyph border absorption: check narrow ring around detected glyphs
            halo_ring = cv2.subtract(
                cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
                local_mask,
            )
            clean_bg = roi_gray[cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) == 0]
            if clean_bg.size >= 24:
                bg_med = float(np.median(clean_bg))
                bg_mad = float(np.median(np.abs(clean_bg.astype(np.float32) - bg_med)))
                outline_thresh = max(20.0, bg_mad * 3.2)
                halo_outliers = (
                    (halo_ring > 0) & (np.abs(roi_gray.astype(np.float32) - bg_med) > outline_thresh)
                ).astype(np.uint8) * 255
                local_mask = cv2.bitwise_or(local_mask, halo_outliers)

        # Close internal gaps between adjacent strokes BEFORE dilating so LaMa never sees trapped ink dots
        close_k = 5 if reg.is_bubble else 7
        local_mask = cv2.morphologyEx(
            local_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        )

        # Slightly fuller dilation so LaMa never sees residual dark stroke halos or furigana
        k_size = 11 if reg.is_bubble else 13
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        local_mask = cv2.dilate(local_mask, kernel, iterations=1)

        # Keep bubble outlines crisp by clipping dilated mask to safe_interior where available,
        # while still honoring text_bbox if a bubble_bbox underestimated the text extent.
        if reg.bubble_bbox is not None and reg.is_bubble:
            bx1, by1, bx2, by2 = reg.bubble_bbox
            if safe_roi is not None and (x1 >= bx1 - 2 and y1 >= by1 - 2 and x2 <= bx2 + 2 and y2 <= by2 + 2):
                # Allow 1px beyond safe_roi only when strictly needed, never touching the black outline
                safe_clip = cv2.dilate(safe_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                local_mask = cv2.bitwise_and(local_mask, safe_clip)
            else:
                safe_x1 = max(px1, bx1 + 2) if bx1 - 2 <= x1 <= bx1 + 3 else px1
                safe_y1 = max(py1, by1 + 2) if by1 - 2 <= y1 <= by1 + 3 else py1
                safe_x2 = min(px2, bx2 - 2) if bx2 - 3 <= x2 <= bx2 + 2 else px2
                safe_y2 = min(py2, by2 - 2) if by2 - 3 <= y2 <= by2 + 2 else py2
                clip = np.zeros_like(local_mask)
                if safe_x2 > safe_x1 and safe_y2 > safe_y1:
                    clip[safe_y1 - py1 : safe_y2 - py1, safe_x1 - px1 : safe_x2 - px1] = 255
                    local_mask = cv2.bitwise_and(local_mask, clip)

        final_mask[py1:py2, px1:px2] = cv2.bitwise_or(final_mask[py1:py2, px1:px2], local_mask)

    return final_mask


def classify_region_colors(
    orig_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    text_mask: np.ndarray,
    reg: DetectedRegion,
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None, float, bool]:
    """
    Determine crisp fill color and stroke color directly from the CLEANED background.

    Fixes Koharu Bug #2 where black translucent bubbles were mistaken for thick black outlines.
    Returns (text_color, stroke_color, stroke_width_factor, is_dark_bubble).
    """
    x1, y1, x2, y2 = reg.text_bbox
    bg_crop = clean_rgb[y1:y2, x1:x2]
    if bg_crop.size == 0:
        return (20, 20, 20), None, 0.0, False

    bg_median = np.median(bg_crop.reshape(-1, 3), axis=0)
    bg_std = float(np.mean(np.std(bg_crop.reshape(-1, 3), axis=0)))
    bg_luma = _luma(bg_median)
    is_dark = bg_luma < 125.0

    if reg.is_bubble:
        if is_dark:
            # Dark bubble (solid or translucent): crisp white text!
            # Only add a subtle thin outline if background behind translucent bubble is textured.
            if bg_std >= 28.0:
                return (255, 255, 255), (12, 12, 12), 0.06, True
            return (255, 255, 255), None, 0.0, True
        # White/bright bubble: clean dark ink, no outline
        if bg_std >= 38.0:
            return (18, 18, 18), (255, 255, 255), 0.06, False
        return (18, 18, 18), None, 0.0, False

    # Floating / SFX text over artwork
    orig_crop = orig_rgb[y1:y2, x1:x2]
    mask_crop = text_mask[y1:y2, x1:x2] > 110
    ink_pixels = orig_crop[mask_crop]
    if len(ink_pixels) >= 12:
        ink_med = np.median(ink_pixels, axis=0)
        ink_luma = _luma(ink_med)
        # Preserve vivid tint or pick pure white/black for crisp readability
        saturation = float(np.max(ink_med) - np.min(ink_med))
        if saturation > 55.0:
            fill = (int(ink_med[0]), int(ink_med[1]), int(ink_med[2]))
            stroke = (255, 255, 255) if ink_luma < 150.0 else (15, 15, 15)
            return fill, stroke, 0.09, is_dark
        if ink_luma >= 145.0:
            return (255, 255, 255), (16, 16, 16), 0.09, is_dark
        return (18, 18, 18), (255, 255, 255), 0.09, is_dark

    if is_dark:
        return (255, 255, 255), (16, 16, 16), 0.09, True
    return (18, 18, 18), (255, 255, 255), 0.09, False


class HybridInpainter:
    """Erases detected text using fast solid-fill for uniform bubbles and LaMa-Manga ONNX for complex areas."""

    def __init__(self, use_gpu: bool = True) -> None:
        self._lama_path = hf_hub_download("mayocream/lama-manga-onnx", "lama-manga.onnx")
        self._use_gpu = use_gpu
        self._lama = _create_ort_session(self._lama_path, use_gpu=use_gpu)

    def _run_lama_patch(self, rgb_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
        ph, pw = rgb_patch.shape[:2]
        if ph <= 512 and pw <= 512:
            if ph == 512 and pw == 512:
                rgb_512 = rgb_patch
                msk_512 = mask_patch
            else:
                rgb_512 = cv2.copyMakeBorder(rgb_patch, 0, 512 - ph, 0, 512 - pw, cv2.BORDER_REPLICATE)
                msk_512 = cv2.copyMakeBorder(mask_patch, 0, 512 - ph, 0, 512 - pw, cv2.BORDER_CONSTANT, value=0)
            img_512 = rgb_512.astype(np.float32) / 255.0
            msk_bin = (msk_512 > 127).astype(np.float32)
        else:
            img_512 = cv2.resize(rgb_patch, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            msk_resized = cv2.resize(mask_patch, (512, 512), interpolation=cv2.INTER_AREA)
            msk_bin = (msk_resized > 24).astype(np.float32)

        inp_img = np.transpose(img_512, (2, 0, 1))[None]
        inp_msk = msk_bin[None, None]
        try:
            out = self._lama.run(None, {"image": inp_img, "mask": inp_msk})[0][0]
        except Exception:
            # DirectML fails on LaMa's FFC FFT MatMul_5 node; safely downgrade to CPU without losing progress.
            if self._lama.get_providers() != ["CPUExecutionProvider"]:
                self._lama = _create_ort_session(self._lama_path, use_gpu=False)
                out = self._lama.run(None, {"image": inp_img, "mask": inp_msk})[0][0]
            else:
                raise
        # mayocream/lama-manga-onnx outputs float RGB in [0.0, 1.0]
        pred_512 = np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0.0, 255.0).astype(np.uint8)
        if ph <= 512 and pw <= 512:
            return pred_512[:ph, :pw]
        return cv2.resize(pred_512, (pw, ph), interpolation=cv2.INTER_LINEAR)

    def inpaint(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        regions: list[DetectedRegion] | None = None,
    ) -> np.ndarray:
        """Inpaint all masked text components cleanly while preserving non-masked regions."""
        result = rgb.copy()
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return result

        h, w = rgb.shape[:2]

        # Pass 0: Direct spotless fill inside flat-colored speech bubble interiors.
        # Eliminates vertical grey LaMa ghosts on narrow white/solid speech balloons.
        if regions:
            k_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            for reg in regions:
                bubble_info = _safe_bubble_interior_roi(reg, w, h, erode_px=3) if reg.is_bubble else None
                if bubble_info is None:
                    continue
                (bx1, by1, bx2, by2), safe_int = bubble_info
                sub_bin = binary[by1:by2, bx1:bx2]
                target_msk = cv2.bitwise_and(sub_bin, safe_int)
                if np.count_nonzero(target_msk) == 0:
                    continue
                bg_zone = (safe_int > 0) & (cv2.dilate(target_msk, k_sep) == 0)
                bg_pixels = result[by1:by2, bx1:bx2][bg_zone]
                is_flat, fill_color, _ = _estimate_flat_bg(bg_pixels)
                if is_flat:
                    patch = result[by1:by2, bx1:bx2]
                    alpha = cv2.GaussianBlur(target_msk.astype(np.float32) / 255.0, (3, 3), 0.7)[..., None]
                    blended = patch.astype(np.float32) * (1.0 - alpha) + fill_color.astype(np.float32) * alpha
                    result[by1:by2, bx1:bx2] = np.clip(blended, 0, 255).astype(np.uint8)
                    # Clear cleaned interior pixels from remaining LaMa mask
                    binary[by1:by2, bx1:bx2][target_msk > 0] = 0

        if np.count_nonzero(binary) == 0:
            return result

        # Group nearby mask components so LaMa processes each bubble/text block in context
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        grouped = cv2.dilate(binary, connect_kernel, iterations=1)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(grouped)

        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close_lama = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        for lbl in range(1, num_labels):
            gx, gy, gw, gh, _ = stats[lbl]
            if gw < 2 or gh < 2:
                continue

            sub_mask = binary[gy : gy + gh, gx : gx + gw]
            if np.count_nonzero(sub_mask) == 0:
                continue

            # Robust check if the surrounding ring inside this region is uniform solid color
            ring = cv2.subtract(cv2.dilate(sub_mask, k_outer), cv2.dilate(sub_mask, k_inner))
            ring_pixels = result[gy : gy + gh, gx : gx + gw][ring > 0]
            if len(ring_pixels) >= 16:
                is_flat, fill_color, _ = _estimate_flat_bg(ring_pixels)
                if is_flat:
                    patch = result[gy : gy + gh, gx : gx + gw]
                    alpha = cv2.GaussianBlur(sub_mask.astype(np.float32) / 255.0, (3, 3), 0.7)[..., None]
                    blended = patch.astype(np.float32) * (1.0 - alpha) + fill_color.astype(np.float32) * alpha
                    result[gy : gy + gh, gx : gx + gw] = np.clip(blended, 0, 255).astype(np.uint8)
                    continue

            # Complex / translucent background: run LaMa at exact 512x512 when block <= 440px
            cx, cy = gx + gw // 2, gy + gh // 2
            max_span = max(gw, gh)
            side = 512 if max_span <= 440 else min(max(512, int(max_span * 1.45)), max(h, w))
            side = min(side, max(h, w))
            half = side // 2
            px1 = max(0, cx - half)
            py1 = max(0, cy - half)
            px2 = min(w, px1 + side)
            py2 = min(h, py1 + side)
            # Re-adjust start if clipped on right/bottom
            px1 = max(0, px2 - side)
            py1 = max(0, py2 - side)

            patch_rgb = result[py1:py2, px1:px2]
            patch_msk = cv2.morphologyEx(binary[py1:py2, px1:px2], cv2.MORPH_CLOSE, k_close_lama)
            pred = self._run_lama_patch(patch_rgb, patch_msk)

            # Post-LaMa ghost suppressor on bright near-white interiors: if ring is bright flat white
            # but LaMa hallucinated dark grey residue inside the core mask, clean it to match ring.
            if len(ring_pixels) >= 16:
                ring_p80 = np.percentile(ring_pixels.astype(np.float32), 80, axis=0)
                ring_iqr = float(
                    np.mean(
                        np.percentile(ring_pixels.astype(np.float32), 85, axis=0)
                        - np.percentile(ring_pixels.astype(np.float32), 15, axis=0)
                    )
                )
                if _luma(ring_p80) >= 232.0 and ring_iqr <= 28.0:
                    core = cv2.erode(patch_msk, k_inner) > 0
                    pred_luma = (
                        0.299 * pred[:, :, 0].astype(np.float32)
                        + 0.587 * pred[:, :, 1].astype(np.float32)
                        + 0.114 * pred[:, :, 2].astype(np.float32)
                    )
                    ghost = core & (pred_luma < (_luma(ring_p80) - 22.0))
                    if np.any(ghost):
                        pred[ghost] = np.clip(np.round(ring_p80), 0, 255).astype(np.uint8)

            alpha = cv2.GaussianBlur(patch_msk.astype(np.float32) / 255.0, (5, 5), 1.0)[..., None]
            blended = patch_rgb.astype(np.float32) * (1.0 - alpha) + pred.astype(np.float32) * alpha
            result[py1:py2, px1:px2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
