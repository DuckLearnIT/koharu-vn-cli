"""Hybrid inpainting (fast solid bubble clean + LaMa ONNX) and background-aware color classification."""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from koharu_vn_cli.pipeline.detector import DetectedRegion, _ort_providers


def _luma(rgb: tuple[int, int, int] | np.ndarray) -> float:
    return float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])


def build_erasure_mask(
    rgb: np.ndarray,
    seg_mask: np.ndarray,
    regions: list[DetectedRegion],
) -> np.ndarray:
    """
    Build a precise pixel mask covering original text glyphs inside detected boxes
    without erasing bubble borders or surrounding artwork.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for reg in regions:
        x1, y1, x2, y2 = reg.text_bbox
        # Expand slightly more on floating outlined glyphs to catch outer borders & furigana
        pad_box = 4 if reg.is_bubble else 7
        px1 = max(0, x1 - pad_box)
        py1 = max(0, y1 - pad_box)
        px2 = min(w, x2 + pad_box)
        py2 = min(h, y2 + pad_box)
        if px2 <= px1 or py2 <= py1:
            continue

        roi_seg = seg_mask[py1:py2, px1:px2]
        local_mask = (roi_seg > 42).astype(np.uint8) * 255
        soft_mask = cv2.dilate((roi_seg > 18).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        local_mask = cv2.bitwise_or(local_mask, soft_mask)

        # Always capture contrast ink & outlined glyph borders inside bubbles OR floating text!
        # Outlined floating SFX (bright fill + dark stroke) often leaves dark stroke pixels just
        # outside CTD's core mask; absorbing adjacent high-contrast halo prevents LaMa ghosts.
        roi_gray = gray[py1:py2, px1:px2]
        fg_ratio = np.count_nonzero(local_mask) / max(1, local_mask.size)
        if reg.is_bubble or fg_ratio < 0.08:
            # Prefer unmasked interior pixels (avoids outer hair/skin/border corrupting translucent bubbles)
            dilated_seed = cv2.dilate(local_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            interior_bg = roi_gray[dilated_seed == 0]
            if interior_bg.size >= 24:
                bg_sample = interior_bg
            else:
                bg_sample = np.concatenate([
                    roi_gray[0, :],
                    roi_gray[-1, :],
                    roi_gray[:, 0],
                    roi_gray[:, -1],
                ])
            bg_med = float(np.median(bg_sample)) if bg_sample.size > 0 else float(np.median(roi_gray))
            diff = np.abs(roi_gray.astype(np.float32) - bg_med)
            otsu_val, _ = cv2.threshold(
                np.clip(diff, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            thresh = max(14.0, min(34.0, float(otsu_val) * 0.58)) if reg.is_bubble else max(26.0, float(otsu_val))
            extra = (diff > thresh).astype(np.uint8) * 255
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
                outline_thresh = max(22.0, bg_mad * 3.5)
                halo_outliers = (
                    (halo_ring > 0) & (np.abs(roi_gray.astype(np.float32) - bg_med) > outline_thresh)
                ).astype(np.uint8) * 255
                local_mask = cv2.bitwise_or(local_mask, halo_outliers)

        # Slightly fuller dilation so LaMa never sees residual dark stroke halos or furigana
        k_size = 9 if reg.is_bubble else 11
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        local_mask = cv2.dilate(local_mask, kernel, iterations=1)

        # Only trim 2px at an edge if text_bbox actually touches that bubble border —
        # NEVER chop pixels when text_bbox extends beyond an underestimated bubble_bbox!
        if reg.bubble_bbox is not None and reg.is_bubble:
            bx1, by1, bx2, by2 = reg.bubble_bbox
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

    def __init__(self) -> None:
        lama_path = hf_hub_download("mayocream/lama-manga-onnx", "lama-manga.onnx")
        self._lama = ort.InferenceSession(lama_path, providers=_ort_providers())

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
        out = self._lama.run(None, {"image": inp_img, "mask": inp_msk})[0][0]
        # mayocream/lama-manga-onnx outputs float RGB in [0.0, 1.0]
        pred_512 = np.clip(np.transpose(out, (1, 2, 0)) * 255.0, 0.0, 255.0).astype(np.uint8)
        if ph <= 512 and pw <= 512:
            return pred_512[:ph, :pw]
        return cv2.resize(pred_512, (pw, ph), interpolation=cv2.INTER_LINEAR)

    def inpaint(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Inpaint all masked text components cleanly while preserving non-masked regions."""
        result = rgb.copy()
        binary = (mask > 127).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return result

        h, w = rgb.shape[:2]
        # Group nearby mask components so LaMa processes each bubble/text block in context
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        grouped = cv2.dilate(binary, connect_kernel, iterations=1)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(grouped)

        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for lbl in range(1, num_labels):
            gx, gy, gw, gh, _ = stats[lbl]
            if gw < 2 or gh < 2:
                continue

            sub_mask = binary[gy : gy + gh, gx : gx + gw]
            if np.count_nonzero(sub_mask) == 0:
                continue

            # Check if the surrounding ring inside this region is uniform solid color
            ring = cv2.subtract(cv2.dilate(sub_mask, k_outer), cv2.dilate(sub_mask, k_inner))
            ring_pixels = result[gy : gy + gh, gx : gx + gw][ring > 0]
            if len(ring_pixels) >= 16:
                ring_std = float(np.mean(np.std(ring_pixels, axis=0)))
                ring_med = np.median(ring_pixels, axis=0)
                ring_luma = _luma(ring_med)
                is_flat_white = ring_luma > 228.0 and ring_std < 14.0
                is_flat_black = ring_luma < 28.0 and ring_std < 14.0
                is_flat_tint = ring_std < 7.5
                if is_flat_white or is_flat_black or is_flat_tint:
                    fill_color = np.clip(np.round(ring_med), 0, 255).astype(np.uint8)
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
            patch_msk = binary[py1:py2, px1:px2]
            pred = self._run_lama_patch(patch_rgb, patch_msk)

            alpha = cv2.GaussianBlur(patch_msk.astype(np.float32) / 255.0, (5, 5), 1.0)[..., None]
            blended = patch_rgb.astype(np.float32) * (1.0 - alpha) + pred.astype(np.float32) * alpha
            result[py1:py2, px1:px2] = np.clip(blended, 0, 255).astype(np.uint8)

        return result
