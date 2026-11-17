"""Vietnamese-first Area-Maximizing Typesetter with 2x Supersampling and Bubble-Contour Fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from koharu_vn_cli.models import TextRegionItem
from koharu_vn_cli.pipeline.fonts import resolve_font_path

SYMBOL_FALLBACK_FONTS = [
    r"C:\Windows\Fonts\seguisym.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


@dataclass
class LinePlacement:
    text: str
    center_x: float
    top_y: float
    width: float


@dataclass
class SolvedLayout:
    font_path: str
    font_size: int
    line_height: float
    lines: list[LinePlacement]
    overflowed: bool = False


@lru_cache(maxsize=256)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=max(6, size))


@lru_cache(maxsize=4096)
def _has_glyph(font_path: str, size: int, ch: str) -> bool:
    if ch.isspace():
        return True
    font = _load_font(font_path, size)
    notdef = bytes(font.getmask("\ufffe"))
    glyph = bytes(font.getmask(ch))
    return len(glyph) > 0 and glyph != notdef


def _pick_font_for_char(primary_path: str, size: int, ch: str) -> str:
    if _has_glyph(primary_path, size, ch):
        return primary_path
    for fb in SYMBOL_FALLBACK_FONTS:
        if Path(fb).is_file() and _has_glyph(fb, size, ch):
            return fb
    return primary_path


def _split_font_runs(text: str, primary_path: str, size: int) -> list[tuple[str, str]]:
    """Split a line into contiguous runs sharing the same resolved font path (supports ♡, ♪, ★)."""
    if not text:
        return []
    runs: list[tuple[str, str]] = []
    cur_path = _pick_font_for_char(primary_path, size, text[0])
    cur_buf = [text[0]]
    for ch in text[1:]:
        p = _pick_font_for_char(primary_path, size, ch)
        if p == cur_path:
            cur_buf.append(ch)
        else:
            runs.append((cur_path, "".join(cur_buf)))
            cur_path = p
            cur_buf = [ch]
    runs.append((cur_path, "".join(cur_buf)))
    return runs


def _measure_text_width(font: ImageFont.FreeTypeFont, text: str) -> float:
    if not text:
        return 0.0
    return float(max(0.0, font.getlength(text)))


def _measure_runs_width(primary_path: str, size: int, text: str) -> float:
    total = 0.0
    for fpath, chunk in _split_font_runs(text, primary_path, size):
        total += _measure_text_width(_load_font(fpath, size), chunk)
    return total


def _polygon_chord_at_y(
    contour: Sequence[tuple[float, float]], y: float
) -> tuple[float, float] | None:
    """Intersect horizontal scanline at `y` with a closed polygon contour."""
    n = len(contour)
    if n < 3:
        return None
    xs: list[float] = []
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2:
        return None
    xs.sort()
    # Pick widest interior pair
    best: tuple[float, float] | None = None
    for i in range(0, len(xs) - 1, 2):
        span = xs[i + 1] - xs[i]
        if best is None or span > (best[1] - best[0]):
            best = (xs[i], xs[i + 1])
    return best


def _ellipse_chord_at_y(
    box: tuple[int, int, int, int], y: float
) -> tuple[float, float] | None:
    """Superellipse (p=2.6) chord width matching real manga/CG oval speech balloons."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    rx = max(1.0, (x2 - x1) * 0.5)
    ry = max(1.0, (y2 - y1) * 0.5)
    norm_y = abs(y - cy) / ry
    if norm_y >= 0.985:
        return None
    p = 2.6
    half_w = rx * (max(0.0, 1.0 - norm_y ** p) ** (1.0 / p))
    return (cx - half_w, cx + half_w)


def _line_spans_for_block(
    box: tuple[int, int, int, int],
    contour: Sequence[tuple[float, float]] | None,
    line_count: int,
    font_size: int,
    line_height: float,
    pad_x: float,
    pad_y: float,
    is_bubble: bool,
    is_rect: bool = False,
) -> list[tuple[float, float, float]] | None:
    """
    Compute (center_x, usable_width, top_y) for each of `line_count` vertically centered lines.
    Returns None if the block exceeds available vertical bubble height.
    """
    x1, y1, x2, y2 = box
    box_w = max(8.0, float(x2 - x1))
    box_h = max(8.0, float(y2 - y1))
    avail_h = box_h - 2.0 * pad_y
    if avail_h <= font_size * 0.85:
        return None

    # Include small extra headroom for Vietnamese stacked accents above caps (Ấ, Ể, Ự)
    ink_top_extra = font_size * 0.14
    block_h = ink_top_extra + font_size + (line_count - 1) * line_height
    if block_h > avail_h + 1.5:
        return None

    cy = (y1 + y2) * 0.5
    y_start = max(y1 + pad_y, cy - block_h * 0.5 + ink_top_extra * 0.4)

    box_cx = (x1 + x2) * 0.5
    spans: list[tuple[float, float, float]] = []
    for i in range(line_count):
        top_y = y_start + i * line_height
        mid_y = top_y + font_size * 0.5
        bot_y = top_y + font_size * 0.96

        if not is_bubble or is_rect:
            spans.append((box_cx, max(12.0, box_w - 2.0 * pad_x), top_y))
            continue

        # Sample across top, mid, and bottom of line ink band so ascenders/descenders fit inside curve.
        # Guard against noisy segmentation contours that jitter center_x left/right between lines.
        left_max = float(x1)
        right_min = float(x2)
        for sample_y in (top_y + font_size * 0.12, mid_y, bot_y):
            chord = None
            if contour and len(contour) >= 4:
                poly_chord = _polygon_chord_at_y(contour, sample_y)
                if poly_chord is not None:
                    poly_cx = (poly_chord[0] + poly_chord[1]) * 0.5
                    if abs(poly_cx - box_cx) <= box_w * 0.10:
                        chord = poly_chord
            if chord is None:
                chord = _ellipse_chord_at_y(box, sample_y)
            if chord is None:
                return None
            left_max = max(left_max, chord[0])
            right_min = min(right_min, chord[1])

        usable_w = (right_min - left_max) - 2.0 * pad_x
        if usable_w < font_size * 0.9:
            return None
        spans.append(((left_max + right_min) * 0.5, usable_w, top_y))

    return spans


def _wrap_paragraph_dp(
    words: list[str],
    widths: list[float],
    font_path: str,
    font_size: int,
    space_w: float,
) -> tuple[list[str], float] | None:
    """
    Exact DP word wrapping fitting `words` into exactly `len(widths)` lines,
    minimizing raggedness and avoiding awkward single-word orphans.
    """
    n = len(words)
    k = len(widths)
    if k == 0 or n < k:
        return None

    word_ws = [_measure_runs_width(font_path, font_size, w) for w in words]

    def line_width(i: int, j: int) -> float:
        return sum(word_ws[i:j]) + space_w * max(0, j - i - 1)

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(k + 1)]
    parent = [[-1] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0

    for line_idx in range(k):
        avail = widths[line_idx]
        remaining_lines = k - line_idx - 1
        for start in range(line_idx, n - remaining_lines):
            base = dp[line_idx][start]
            if not math.isfinite(base):
                continue
            for end in range(start + 1, n - remaining_lines + 1):
                lw = line_width(start, end)
                if lw > avail + 0.5:
                    break
                slack = max(0.0, avail - lw) / max(1.0, avail)
                cost = base + (slack * slack) * 100.0
                # Penalize single short orphan word on last line if multi-line
                if line_idx == k - 1 and k > 1 and (end - start) == 1 and len(words[start]) <= 3:
                    cost += 18.0
                if cost < dp[line_idx + 1][end]:
                    dp[line_idx + 1][end] = cost
                    parent[line_idx + 1][end] = start

    if not math.isfinite(dp[k][n]):
        return None

    breaks = [0] * k
    cur = n
    for line_idx in range(k, 0, -1):
        prev = parent[line_idx][cur]
        if prev < 0:
            return None
        breaks[line_idx - 1] = cur
        cur = prev

    lines: list[str] = []
    start = 0
    for end in breaks:
        lines.append(" ".join(words[start:end]))
        start = end
    return lines, dp[k][n] / k


def _greedy_fallback_wrap(
    text: str,
    max_w: float,
    font_path: str,
    font_size: int,
) -> list[str]:
    """Fallback word/char wrapper when a tiny box cannot satisfy strict contour constraints."""
    lines: list[str] = []
    for para in text.splitlines() or [text]:
        words = para.split()
        if not words:
            continue
        cur = ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if _measure_runs_width(font_path, font_size, cand) <= max_w or not cur:
                if _measure_runs_width(font_path, font_size, cand) <= max_w:
                    cur = cand
                else:
                    # Break extra-long token by chars
                    if cur:
                        lines.append(cur)
                        cur = ""
                    chunk = ""
                    for ch in w:
                        if _measure_runs_width(font_path, font_size, chunk + ch) <= max_w or not chunk:
                            chunk += ch
                        else:
                            lines.append(chunk)
                            chunk = ch
                    cur = chunk
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    return lines or [text]


def solve_layout(
    region: TextRegionItem,
    canvas_size: tuple[int, int] | None = None,
) -> SolvedLayout:
    """
    Find the LARGEST crisp font size and balanced line layout that fills the bubble/frame
    without overflowing or squeezing Vietnamese diacritics.
    """
    font_path = str(resolve_font_path(region.font_name, kind=region.kind))
    if region.is_sfx and not (region.translated_text or "").strip():
        return SolvedLayout(font_path=font_path, font_size=14, line_height=17.0, lines=[])
    text = (region.translated_text or region.source_text or "").strip()

    box = region.bubble_bbox if (region.kind == "bubble" and region.bubble_bbox) else region.text_bbox
    x1, y1, x2, y2 = box
    box_w = max(12.0, float(x2 - x1))
    box_h = max(12.0, float(y2 - y1))

    is_bubble = region.kind == "bubble"
    is_rect = is_bubble and region.is_dark_bubble and not region.draw_bubble
    contour = region.bubble_contour if (is_bubble and not region.draw_bubble and not is_rect) else None

    # Vertical Japanese floating text columns are tall & narrow: widen virtual box so
    # horizontal Vietnamese text stays readable instead of wrapping 1 word per line at 9px.
    if not is_bubble and box_w < 130.0 and box_h > box_w * 1.45:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        target_w = max(box_w, min(320.0, max(160.0, math.sqrt(max(1, len(text)) * 190.0))))
        if canvas_size is not None:
            cw, ch = canvas_size
            target_w = min(target_w, max(80.0, cw - 12.0))
            nx1 = max(6.0, min(cw - target_w - 6.0, cx - target_w * 0.5))
            nx2 = min(float(cw - 6), nx1 + target_w)
            ny1 = max(6.0, min(float(ch - 24), y1))
            ny2 = min(float(ch - 6), max(ny1 + box_h, cy + box_h * 0.5))
        else:
            nx1 = cx - target_w * 0.5
            nx2 = cx + target_w * 0.5
            ny1, ny2 = float(y1), float(y2)
        box = (int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2)))
        x1, y1, x2, y2 = box
        box_w = max(12.0, float(x2 - x1))
        box_h = max(12.0, float(y2 - y1))

    # Generous yet compact margins (fixes Koharu's 1.5*font_size horizontal pinch bug!)
    pad_x = max(3.0, min(10.0, box_w * 0.065)) if is_bubble else max(2.0, box_w * 0.03)
    pad_y = max(5.0, min(14.0, box_h * 0.085)) if is_bubble else max(2.0, box_h * 0.03)

    # Explicit paragraphs or flattened words
    paragraphs = [p.split() for p in text.splitlines() if p.strip()]
    words = [w for p in paragraphs for w in p]
    if not words:
        return SolvedLayout(font_path=font_path, font_size=14, line_height=17.0, lines=[])

    # Cap max font size proportionally so 1-word exclamations look tasteful while dialogue fills frame
    if region.font_size is not None:
        max_fs = min_fs = max(8, int(round(region.font_size)))
    else:
        if is_bubble:
            max_fs = int(max(14, min(42, box_w * 0.42, box_h * 0.36)))
            if len(words) <= 2:
                max_fs = min(max_fs, int(max(14, min(box_w * 0.34, box_h * 0.24))))
            min_fs = 10 if min(box_w, box_h) < 72 else 12
        else:
            max_fs = int(max(16, min(44, max(box_w, box_h * 0.45) * 0.38)))
            min_fs = 13

    line_ratio = 1.24 if is_bubble else 1.18

    for fs in range(max_fs, min_fs - 1, -1):
        font = _load_font(font_path, fs)
        space_w = _measure_text_width(font, " ") or fs * 0.28
        lh = fs * line_ratio
        avail_h = box_h - 2.0 * pad_y
        max_lines = min(len(words), max(1, int(math.floor((avail_h - fs * 0.14) / lh)) + 1))

        # Test all feasible line counts at this font size and pick best balanced oval distribution
        best_score = float("inf")
        best_result: SolvedLayout | None = None

        for k in range(1, max_lines + 1):
            spans = _line_spans_for_block(box, contour, k, fs, lh, pad_x, pad_y, is_bubble, is_rect=is_rect)
            if spans is None:
                continue
            widths = [s[1] for s in spans]
            # Handle explicit newlines when line count permits, otherwise join smoothly
            wrapped = _wrap_paragraph_dp(words, widths, font_path, fs, space_w)
            if wrapped is None:
                continue
            lines_text, dp_cost = wrapped

            # Prefer natural vertical fill (55% - 85% of bubble height) & balanced line widths
            block_h = fs + (k - 1) * lh
            fill_ratio = block_h / max(1.0, avail_h)
            fill_penalty = 0.0 if 0.45 <= fill_ratio <= 0.88 else abs(fill_ratio - 0.65) * 40.0
            score = dp_cost + fill_penalty

            if score < best_score:
                best_score = score
                placements = [
                    LinePlacement(
                        text=lt,
                        center_x=spans[idx][0],
                        top_y=spans[idx][2],
                        width=_measure_runs_width(font_path, fs, lt),
                    )
                    for idx, lt in enumerate(lines_text)
                ]
                best_result = SolvedLayout(
                    font_path=font_path,
                    font_size=fs,
                    line_height=lh,
                    lines=placements,
                    overflowed=False,
                )

        if best_result is not None:
            return best_result

    # Ultimate fallback at min_fs with greedy wrap so text never disappears
    fs = min_fs
    lh = fs * line_ratio
    lines_text = _greedy_fallback_wrap(text, max(16.0, box_w - 2.0 * pad_x), font_path, fs)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    block_h = fs + (len(lines_text) - 1) * lh
    y0 = max(y1 + 2.0, cy - block_h * 0.5)
    placements = [
        LinePlacement(
            text=lt,
            center_x=cx,
            top_y=y0 + idx * lh,
            width=_measure_runs_width(font_path, fs, lt),
        )
        for idx, lt in enumerate(lines_text)
    ]
    return SolvedLayout(
        font_path=font_path,
        font_size=fs,
        line_height=lh,
        lines=placements,
        overflowed=True,
    )


def adapt_regions_for_readability(
    regions: Sequence[TextRegionItem],
    img_w: int,
    img_h: int,
) -> list[TextRegionItem]:
    """
    Prevent translated Vietnamese text from being crushed inside narrow vertical Japanese boxes.
    Widens tall-narrow floating regions and tight bubble boxes while staying inside image bounds.
    """
    min_readable = max(13, min(16, int(round(min(img_w, img_h) * 0.015))))
    adapted: list[TextRegionItem] = []

    for reg in regions:
        copy = reg.model_copy()
        if copy.is_sfx and not (copy.translated_text or "").strip():
            adapted.append(copy)
            continue
        text = (copy.translated_text or copy.source_text or "").strip()
        if not text:
            adapted.append(copy)
            continue

        if copy.kind == "floating":
            x1, y1, x2, y2 = copy.text_bbox
            bw, bh = x2 - x1, y2 - y1
            if bw < max(130, int(img_w * 0.14)) and bh > bw * 1.35:
                cx = (x1 + x2) * 0.5
                target_w = max(bw, min(int(img_w * 0.34), max(165, int(math.sqrt(len(text) * 190)))))
                nx1 = max(6, min(img_w - target_w - 6, int(round(cx - target_w * 0.5))))
                nx2 = min(img_w - 6, nx1 + target_w)
                copy.text_bbox = (nx1, y1, nx2, max(y2, min(img_h - 6, y1 + max(bh, 48))))
        else:
            layout = solve_layout(copy, canvas_size=(img_w, img_h))
            if layout.font_size < min_readable or layout.overflowed:
                bx1, by1, bx2, by2 = copy.bubble_bbox or copy.text_bbox
                bw, bh = bx2 - bx1, by2 - by1
                target_w = max(bw, min(int(img_w * 0.24), max(125, int(bw * 1.45))))
                target_h = max(bh, min(int(img_h * 0.22), int(bh * 1.15)))
                cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                nx1 = max(6, min(img_w - target_w - 6, int(round(cx - target_w * 0.5))))
                ny1 = max(6, min(img_h - target_h - 6, int(round(cy - target_h * 0.5))))
                new_box = (nx1, ny1, min(img_w - 6, nx1 + target_w), min(img_h - 6, ny1 + target_h))
                copy.text_bbox = new_box
                if copy.bubble_bbox is not None:
                    copy.bubble_bbox = new_box
                    # Clear old narrow contour so typesetter uses the widened bounding ellipse
                    copy.bubble_contour = None
        adapted.append(copy)

    return adapted


def render_text_regions(canvas_rgba: Image.Image, regions: Sequence[TextRegionItem]) -> Image.Image:
    """
    Render all Vietnamese text regions onto `canvas_rgba` using 2x supersampling,
    separate stroke+fill passes (keeping Vietnamese diacritics open & razor sharp).
    """
    if not regions:
        return canvas_rgba

    w, h = canvas_rgba.size
    out = canvas_rgba.copy()
    scale = 2

    for reg in regions:
        if reg.is_sfx and not (reg.translated_text or "").strip():
            continue
        layout = solve_layout(reg, canvas_size=(w, h))
        if not layout.lines:
            continue

        fill_rgb = reg.text_color or ((255, 255, 255) if reg.is_dark_bubble else (18, 18, 18))
        fill_rgba = (fill_rgb[0], fill_rgb[1], fill_rgb[2], 255)

        stroke_px = float(reg.stroke_width or 0.0)
        if stroke_px <= 0.0 and reg.stroke_color is not None:
            stroke_px = min(2.8, max(1.3, round(layout.font_size * 0.085, 1)))
        stroke_rgba = (
            (reg.stroke_color[0], reg.stroke_color[1], reg.stroke_color[2], 255)
            if (reg.stroke_color is not None and stroke_px > 0.0)
            else None
        )

        # Compute tight bounding patch around the lines + stroke padding
        size_hi = layout.font_size * scale
        pad = int(math.ceil(stroke_px * 2 + layout.font_size * 0.65))
        min_x = min(lp.center_x - lp.width * 0.5 for lp in layout.lines) - pad
        max_x = max(lp.center_x + lp.width * 0.5 for lp in layout.lines) + pad
        min_y = min(lp.top_y for lp in layout.lines) - pad
        max_y = max(lp.top_y + layout.font_size * 1.35 for lp in layout.lines) + pad

        px1 = max(0, int(math.floor(min_x)))
        py1 = max(0, int(math.floor(min_y)))
        px2 = min(w, int(math.ceil(max_x)))
        py2 = min(h, int(math.ceil(max_y)))
        pw, ph = px2 - px1, py2 - py1
        if pw < 4 or ph < 4:
            continue

        patch_hi = Image.new("RGBA", (pw * scale, ph * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch_hi)
        stroke_hi = max(1, int(round(stroke_px * scale))) if stroke_rgba else 0
        primary_hi = _load_font(layout.font_path, size_hi)
        base_ascent, _ = primary_hi.getmetrics()

        for lp in layout.lines:
            runs = _split_font_runs(lp.text, layout.font_path, size_hi)
            run_widths = [
                _measure_text_width(_load_font(fp, size_hi), chunk) for fp, chunk in runs
            ]
            total_w_hi = sum(run_widths)
            start_x = (lp.center_x - px1) * scale - total_w_hi * 0.5
            baseline_y = (lp.top_y - py1) * scale + base_ascent

            # Pass 1: stroke for all runs first so stroke never covers adjacent glyph fill
            if stroke_rgba is not None and stroke_hi > 0:
                cur_x = start_x
                for (fp, chunk), rw in zip(runs, run_widths):
                    f_hi = _load_font(fp, size_hi)
                    draw.text(
                        (cur_x, baseline_y),
                        chunk,
                        font=f_hi,
                        fill=stroke_rgba,
                        stroke_width=stroke_hi,
                        stroke_fill=stroke_rgba,
                        anchor="ls",
                    )
                    cur_x += rw

            # Pass 2: crisp fill on top
            cur_x = start_x
            for (fp, chunk), rw in zip(runs, run_widths):
                f_hi = _load_font(fp, size_hi)
                draw.text(
                    (cur_x, baseline_y),
                    chunk,
                    font=f_hi,
                    fill=fill_rgba,
                    stroke_width=0,
                    anchor="ls",
                )
                cur_x += rw

        patch_1x = patch_hi.resize((pw, ph), Image.Resampling.LANCZOS)
        if abs(reg.angle) > 0.5:
            patch_1x = patch_1x.rotate(-reg.angle, resample=Image.Resampling.BICUBIC, expand=True)
            # Re-center rotated patch around original box center
            ocx, ocy = (px1 + px2) // 2, (py1 + py2) // 2
            rx1 = ocx - patch_1x.width // 2
            ry1 = ocy - patch_1x.height // 2
            out.alpha_composite(patch_1x, dest=(max(0, rx1), max(0, ry1)))
        else:
            out.alpha_composite(patch_1x, dest=(px1, py1))

    return out
