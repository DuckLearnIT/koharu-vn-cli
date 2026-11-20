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
    scale_x: float = 1.0


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


_CJK_PUNCT_MAP = str.maketrans(
    {
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "【": "(",
        "】": ")",
        "〈": '"',
        "〉": '"',
        "《": '"',
        "》": '"',
        "〜": "~",
        "～": "~",
        "・": "·",
        "　": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def sanitize_render_text(text: str, primary_path: str | None = None, size: int = 16) -> str:
    """Replace CJK brackets/unrenderable symbols with crisp equivalents so tofu boxes never appear."""
    cleaned = (text or "").translate(_CJK_PUNCT_MAP)
    if primary_path is None:
        return cleaned
    out: list[str] = []
    for ch in cleaned:
        if ch.isspace() or _has_glyph(primary_path, size, ch):
            out.append(ch)
            continue
        if any(Path(fb).is_file() and _has_glyph(fb, size, ch) for fb in SYMBOL_FALLBACK_FONTS):
            out.append(ch)
    return "".join(out)


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


def _is_rectangular_contour(
    box: tuple[int, int, int, int],
    contour: Sequence[tuple[float, float]] | None,
) -> bool:
    """Detect rectangular/narration boxes (including white narration frames) via polygon fill ratio."""
    if not contour or len(contour) < 4:
        return False
    bx1, by1, bx2, by2 = box
    box_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    # Shoelace polygon area
    n = len(contour)
    area2 = 0.0
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    poly_area = abs(area2) * 0.5
    return (poly_area / box_area) >= 0.87


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


_CLAUSE_BREAK_CHARS = (".", "!", "?", "…", ",", ";", ":", "—", "~", "♡", "♪", '"', ")")


def _wrap_paragraph_dp(
    words: list[str],
    widths: list[float],
    font_path: str,
    font_size: int,
    space_w: float,
    scale_x: float = 1.0,
) -> tuple[list[str], float] | None:
    """
    Exact DP word wrapping fitting `words` into exactly `len(widths)` lines,
    minimizing raggedness, rewarding natural Vietnamese clause breaks, and avoiding
    single-word staircase lines or awkward short orphans.
    """
    n = len(words)
    k = len(widths)
    if k == 0 or n < k:
        return None

    eff_sx = max(0.75, min(1.0, float(scale_x)))
    word_ws = [_measure_runs_width(font_path, font_size, w) * eff_sx for w in words]
    eff_space = space_w * eff_sx

    def line_width(i: int, j: int) -> float:
        return sum(word_ws[i:j]) + eff_space * max(0, j - i - 1)

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

                word_cnt = end - start
                last_word = words[end - 1]

                # Reward natural pauses after punctuation on interior line breaks
                if line_idx < k - 1 and last_word.endswith(_CLAUSE_BREAK_CHARS):
                    cost = max(0.0, cost - 5.5)

                # Penalize single-word lines when the sentence has enough words to group naturally
                if k > 1 and word_cnt == 1 and n >= 4:
                    if line_idx == k - 1:
                        # Strong penalty for tiny orphan word dangling on the last line
                        cost += 28.0 if len(last_word) <= 4 else 12.0
                    elif line_idx == 0:
                        cost += 14.0 if len(last_word) <= 3 else 6.0
                    else:
                        cost += 10.0

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
    text = sanitize_render_text(
        (region.translated_text or region.source_text or "").strip(),
        font_path,
        18,
    )

    box = region.bubble_bbox if (region.kind == "bubble" and region.bubble_bbox) else region.text_bbox
    x1, y1, x2, y2 = box
    box_w = max(12.0, float(x2 - x1))
    box_h = max(12.0, float(y2 - y1))

    is_bubble = region.kind == "bubble"
    is_rect = (
        is_bubble
        and not region.draw_bubble
        and (
            region.is_dark_bubble
            or _is_rectangular_contour(box, region.bubble_contour)
        )
    )
    contour = region.bubble_contour if (is_bubble and not region.draw_bubble and not is_rect) else None

    # Keep tall narrow standalone floating boxes inside safe canvas margins if solve_layout is called directly
    if not is_bubble and box_w < 120.0 and box_h > box_w * 1.45:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        target_w = max(box_w, min(280.0, max(160.0, math.sqrt(max(1, len(text)) * 170.0))))
        if canvas_size is not None:
            cw, ch = canvas_size
            edge_pad = max(10.0, cw * 0.012)
            target_w = min(target_w, max(80.0, cw - 2.0 * edge_pad))
            nx1 = max(edge_pad, min(cw - target_w - edge_pad, cx - target_w * 0.5))
            nx2 = min(float(cw - edge_pad), nx1 + target_w)
            ny1 = max(edge_pad, min(float(ch - 24), y1))
            ny2 = min(float(ch - edge_pad), max(ny1 + box_h, cy + box_h * 0.5))
        else:
            nx1 = cx - target_w * 0.5
            nx2 = cx + target_w * 0.5
            ny1, ny2 = float(y1), float(y2)
        box = (int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2)))
        x1, y1, x2, y2 = box
        box_w = max(12.0, float(x2 - x1))
        box_h = max(12.0, float(y2 - y1))
    elif canvas_size is not None and not is_bubble:
        cw, ch = canvas_size
        edge_pad = max(10.0, cw * 0.012)
        nx1 = max(edge_pad, min(float(cw - edge_pad - 24), float(x1)))
        nx2 = min(float(cw - edge_pad), max(nx1 + 24.0, float(x2)))
        ny1 = max(edge_pad, min(float(ch - edge_pad - 18), float(y1)))
        ny2 = min(float(ch - edge_pad), max(ny1 + 18.0, float(y2)))
        box = (int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2)))
        x1, y1, x2, y2 = box
        box_w = max(12.0, float(x2 - x1))
        box_h = max(12.0, float(y2 - y1))

    # Explicit paragraphs or flattened words
    paragraphs = [p.split() for p in text.splitlines() if p.strip()]
    words = [w for p in paragraphs for w in p]
    if not words:
        return SolvedLayout(font_path=font_path, font_size=14, line_height=17.0, lines=[])

    word_count = len(words)
    char_count = len(text)
    is_dense = word_count >= 12 or char_count >= 56
    is_very_dense = word_count >= 22 or char_count >= 105
    is_narrow_tall = is_bubble and (box_h > box_w * 1.45)

    # Scale font caps proportionally with high-res page dimensions (e.g. 1200x1800 CG pages)
    page_scale = 1.0
    if canvas_size is not None:
        page_scale = max(1.0, min(1.65, min(canvas_size[0], canvas_size[1]) / 850.0))

    # Cap max font size proportionally so dialogue & floating lines look balanced
    if region.font_size is not None:
        max_fs = min_fs = max(8, int(round(region.font_size)))
        hard_min_fs = min_fs
    else:
        if is_bubble:
            max_cap = int(round(38 * page_scale))
            max_fs = int(max(14, min(max_cap, box_w * 0.40, box_h * 0.34)))
            if word_count <= 2:
                max_fs = min(max_fs, int(max(14, min(box_w * 0.32, box_h * 0.24))))
            elif is_very_dense:
                max_fs = min(max_fs, int(max(14, min(int(round(26 * page_scale)), math.sqrt((box_w * box_h) / max(1, char_count)) * 1.35))))
            elif is_dense:
                max_fs = min(max_fs, int(max(15, min(int(round(30 * page_scale)), math.sqrt((box_w * box_h) / max(1, char_count)) * 1.45))))
            min_fs = 10 if min(box_w, box_h) < 76 else 11
            hard_min_fs = 9 if (is_dense or min(box_w, box_h) < 72) else 10
        else:
            float_cap = int(round(32 * page_scale))
            max_fs = int(max(15, min(float_cap, box_w * 0.24, max(box_w * 0.16, box_h * 0.28))))
            if word_count <= 2:
                max_fs = min(max_fs, int(max(15, min(box_w * 0.28, box_h * 0.24))))
            min_fs = 10
            hard_min_fs = 9 if is_dense else 10

    def _solve_tier(
        t_pad_x: float,
        t_pad_y: float,
        t_line_ratio: float,
        t_scale_x: float,
        fs_hi: int,
        fs_lo: int,
    ) -> tuple[SolvedLayout, float] | None:
        """
        Solve balanced layout within a single padding/line-height/scale_x tier.
        Evaluates a small window below the first fitting font size so narrow bubbles
        prefer clean 2+ word lines over awkward 1-word-per-line staircases.
        """
        best_overall: tuple[SolvedLayout, float] | None = None
        first_fit_fs: int | None = None
        first_fit_staircase = False

        for fs in range(fs_hi, fs_lo - 1, -1):
            if first_fit_fs is not None:
                if first_fit_staircase:
                    if fs < max(fs_lo, int(round(first_fit_fs * 0.58))):
                        break
                else:
                    if (first_fit_fs - fs) > 3 or fs < int(round(first_fit_fs * 0.80)):
                        break

            font = _load_font(font_path, fs)
            space_w = _measure_text_width(font, " ") or fs * 0.28
            lh = fs * t_line_ratio
            avail_h = box_h - 2.0 * t_pad_y
            max_lines = min(word_count, max(1, int(math.floor((avail_h - fs * 0.14) / lh)) + 1))

            best_k_score = float("inf")
            best_k_layout: SolvedLayout | None = None
            best_k_single_ratio = 0.0

            for k in range(1, max_lines + 1):
                spans = _line_spans_for_block(
                    box, contour, k, fs, lh, t_pad_x, t_pad_y, is_bubble, is_rect=is_rect
                )
                if spans is None:
                    continue
                widths = [s[1] for s in spans]
                wrapped = _wrap_paragraph_dp(
                    words, widths, font_path, fs, space_w, scale_x=t_scale_x
                )
                if wrapped is None:
                    continue
                lines_text, dp_cost = wrapped

                # Prefer natural vertical fill (48% - 88% of bubble height) & avoid 1-word staircases
                block_h = fs + (k - 1) * lh
                fill_ratio = block_h / max(1.0, avail_h)
                fill_penalty = 0.0 if 0.45 <= fill_ratio <= 0.88 else abs(fill_ratio - 0.68) * 34.0

                single_word_lines = sum(1 for lt in lines_text if len(lt.split()) == 1)
                single_ratio = single_word_lines / max(1, k)
                staircase_penalty = 0.0
                if word_count >= 4 and k >= 3 and box_w >= 62.0 and single_ratio > 0.34:
                    staircase_penalty = (single_ratio - 0.25) * 145.0

                k_score = dp_cost + fill_penalty + staircase_penalty
                if k_score < best_k_score:
                    best_k_score = k_score
                    best_k_single_ratio = single_ratio
                    placements = [
                        LinePlacement(
                            text=lt,
                            center_x=spans[idx][0],
                            top_y=spans[idx][2],
                            width=_measure_runs_width(font_path, fs, lt) * t_scale_x,
                        )
                        for idx, lt in enumerate(lines_text)
                    ]
                    best_k_layout = SolvedLayout(
                        font_path=font_path,
                        font_size=fs,
                        line_height=lh,
                        lines=placements,
                        overflowed=False,
                        scale_x=t_scale_x,
                    )

            if best_k_layout is None:
                continue

            if first_fit_fs is None:
                first_fit_fs = fs
                first_fit_staircase = (
                    word_count >= 4
                    and len(best_k_layout.lines) >= 4
                    and best_k_single_ratio > 0.45
                )

            # Trade-off: larger font is strongly preferred, UNLESS dropping font size removes a
            # severe 1-word-per-line staircase inside a narrow oval bubble.
            size_bonus = (fs - fs_lo) * (4.5 if first_fit_staircase else 7.5)
            combined = best_k_score - size_bonus
            if best_overall is None or combined < best_overall[1]:
                best_overall = (best_k_layout, combined)

            # Fast exit as soon as we reach a clean multi-word layout (no severe staircase)
            if word_count < 4 or best_k_single_ratio <= 0.34 or len(best_k_layout.lines) <= 2:
                break

        return best_overall

    # Base comfortable margins (Tier 1)
    base_pad_x = max(3.0, min(10.0, box_w * 0.062)) if is_bubble else max(4.0, box_w * 0.04)
    base_pad_y = max(5.0, min(13.0, box_h * 0.080)) if is_bubble else max(3.0, box_h * 0.035)
    base_line_ratio = (1.17 if is_dense else 1.22) if is_bubble else 1.16

    tier1 = _solve_tier(base_pad_x, base_pad_y, base_line_ratio, 1.0, max_fs, min_fs)

    # Comfort floor: if dense or narrow bubble got squeezed below comfortable size, try compact Tier 2
    comfort_fs = 15 if (is_dense and min(box_w, box_h) >= 100) else (14 if is_bubble else 13)
    need_tier2 = (
        region.font_size is None
        and (
            tier1 is None
            or tier1[0].font_size < comfort_fs
            or (is_narrow_tall and word_count >= 5 and len(tier1[0].lines) >= 5)
        )
    )

    best_pick = tier1
    if need_tier2:
        t2_pad_x = max(2.5, min(6.5, box_w * 0.040)) if is_bubble else max(3.0, box_w * 0.03)
        t2_pad_y = max(3.5, min(8.5, box_h * 0.052)) if is_bubble else max(2.5, box_h * 0.025)
        t2_lr = 1.13 if is_bubble else 1.12
        t2_sx = 0.92 if (is_dense or is_narrow_tall) else 0.95
        tier2 = _solve_tier(t2_pad_x, t2_pad_y, t2_lr, t2_sx, max_fs, min_fs)
        if tier2 is not None:
            if best_pick is None:
                best_pick = tier2
            else:
                # Prefer Tier 2 when it boosts readable font size WITHOUT worsening quality,
                # or cuts awkward extra lines with clearly better typography score.
                fs_gain = tier2[0].font_size - best_pick[0].font_size
                line_drop = len(best_pick[0].lines) - len(tier2[0].lines)
                if (fs_gain >= 1 and tier2[1] <= best_pick[1] + 8.0) or (
                    line_drop >= 1 and tier2[1] < best_pick[1] - 4.0
                ):
                    best_pick = tier2

    # Tier 3 rescue for ultra-dense bubbles that still overflow or drop below 11px
    if region.font_size is None and (best_pick is None or (is_very_dense and best_pick[0].font_size <= 11)):
        t3_pad_x = 2.0 if is_bubble else 2.5
        t3_pad_y = 3.0 if is_bubble else 2.0
        tier3 = _solve_tier(t3_pad_x, t3_pad_y, 1.09, 0.88, max_fs, hard_min_fs)
        if tier3 is not None and (best_pick is None or tier3[0].font_size > best_pick[0].font_size):
            best_pick = tier3

    if best_pick is not None:
        return best_pick[0]

    # Ultimate fallback at hard_min_fs with slight condensation & greedy wrap so text never disappears
    fs = hard_min_fs
    lh = fs * 1.10
    fallback_sx = 0.88 if is_dense else 0.94
    eff_max_w = max(16.0, (box_w - 4.0) / fallback_sx)
    lines_text = _greedy_fallback_wrap(text, eff_max_w, font_path, fs)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    block_h = fs + (len(lines_text) - 1) * lh
    y0 = max(y1 + 2.0, cy - block_h * 0.5)
    placements = [
        LinePlacement(
            text=lt,
            center_x=cx,
            top_y=y0 + idx * lh,
            width=_measure_runs_width(font_path, fs, lt) * fallback_sx,
        )
        for idx, lt in enumerate(lines_text)
    ]
    return SolvedLayout(
        font_path=font_path,
        font_size=fs,
        line_height=lh,
        lines=placements,
        overflowed=True,
        scale_x=fallback_sx,
    )


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 8) -> bool:
    return not (
        a[2] + gap <= b[0]
        or b[2] + gap <= a[0]
        or a[3] + gap <= b[1]
        or b[3] + gap <= a[1]
    )


def adapt_regions_for_readability(
    regions: Sequence[TextRegionItem],
    img_w: int,
    img_h: int,
) -> list[TextRegionItem]:
    """
    Prevent translated Vietnamese text from being crushed OR overlapping adjacent regions:
    1. Widens tall-narrow floating regions only into conflict-free horizontal space.
    2. Separates any remaining floating/floating or floating/bubble overlaps cleanly inside safe margins.
    3. Harmonizes outlier floating font sizes alongside speech bubbles on the same page.
    """
    margin_x = max(10, int(round(img_w * 0.015)))
    margin_y = max(10, int(round(img_h * 0.015)))
    gap = 8
    min_readable = max(13, min(16, int(round(min(img_w, img_h) * 0.015))))

    def clamp_box(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        bw = min(max(24, b[2] - b[0]), max(24, img_w - 2 * margin_x))
        bh = min(max(20, b[3] - b[1]), max(20, img_h - 2 * margin_y))
        x1 = max(margin_x, min(img_w - margin_x - bw, b[0]))
        y1 = max(margin_y, min(img_h - margin_y - bh, b[1]))
        return (x1, y1, x1 + bw, y1 + bh)

    adapted = [reg.model_copy() for reg in regions]
    # Sanitize text before sizing/rendering so CJK brackets never skew measurements or render tofu
    for copy in adapted:
        if copy.translated_text:
            copy.translated_text = sanitize_render_text(copy.translated_text)

    active_boxes = [
        (idx, (copy.bubble_bbox or copy.text_bbox))
        for idx, copy in enumerate(adapted)
        if not (copy.is_sfx and not (copy.translated_text or "").strip())
        and (copy.translated_text or copy.source_text or "").strip()
    ]

    # Step 1: Neighbor-aware floating widening & tight-bubble expansion
    for idx, copy in enumerate(adapted):
        if copy.is_sfx and not (copy.translated_text or "").strip():
            continue
        text = (copy.translated_text or copy.source_text or "").strip()
        if not text:
            continue

        if copy.kind == "floating":
            x1, y1, x2, y2 = clamp_box(copy.text_bbox)
            bw, bh = x2 - x1, y2 - y1
            cx_orig = (x1 + x2) * 0.5
            cy_orig = (y1 + y2) * 0.5

            if bw < max(195, int(img_w * 0.22)) and bh > bw * 1.45:
                # Tall vertical Japanese floating column(s): widen around ORIGINAL center first,
                # then compact vertical height to actual rendered Vietnamese text band so distant
                # upper/lower boxes don't falsely collide and push text sideways onto character faces!
                desired_w = max(
                    bw,
                    min(int(img_w * 0.30), max(210, int(bw * 1.35), int(math.sqrt(len(text) * 260)))),
                )
                # Estimate vertical span around cy_orig to check true horizontal neighbors only
                est_h = min(bh, max(96, int(desired_w * 0.95)))
                ey1 = int(round(cy_orig - est_h * 0.5))
                ey2 = int(round(cy_orig + est_h * 0.5))

                free_left = margin_x
                free_right = img_w - margin_x
                for o_idx, ob in active_boxes:
                    if o_idx == idx:
                        continue
                    if ob[3] + gap <= ey1 or ey2 + gap <= ob[1]:
                        continue
                    ocx = (ob[0] + ob[2]) * 0.5
                    if ob[2] <= x1 + 6 or ocx < cx_orig:
                        free_left = max(free_left, ob[2] + gap)
                    if ob[0] >= x2 - 6 or ocx >= cx_orig:
                        free_right = min(free_right, ob[0] - gap)

                avail_span = free_right - free_left
                if avail_span >= max(bw, 120):
                    eff_w = min(desired_w, avail_span)
                    nx1 = max(free_left, min(free_right - eff_w, int(round(cx_orig - eff_w * 0.5))))
                    nx2 = min(free_right, nx1 + eff_w)
                else:
                    eff_w = min(desired_w, max(bw, int(img_w * 0.26)))
                    nx1 = max(margin_x, min(img_w - margin_x - eff_w, int(round(cx_orig - eff_w * 0.5))))
                    nx2 = min(img_w - margin_x, nx1 + eff_w)

                trial = copy.model_copy(update={"text_bbox": clamp_box((nx1, y1, nx2, y2))})
                trial_lay = solve_layout(trial, canvas_size=(img_w, img_h))
                if trial_lay.lines:
                    used_h = int(
                        math.ceil(
                            trial_lay.font_size + (len(trial_lay.lines) - 1) * trial_lay.line_height + 28
                        )
                    )
                    compact_h = min(bh, max(used_h, 64))
                    ny1 = int(round(cy_orig - compact_h * 0.5))
                    ny2 = ny1 + compact_h
                else:
                    ny1, ny2 = y1, max(y2, min(img_h - margin_y, y1 + max(bh, 48)))
                copy.text_bbox = clamp_box((nx1, ny1, nx2, ny2))
            else:
                copy.text_bbox = clamp_box((x1, y1, x2, y2))
        else:
            layout = solve_layout(copy, canvas_size=(img_w, img_h))
            if layout.font_size < min_readable or layout.overflowed:
                bx1, by1, bx2, by2 = copy.bubble_bbox or copy.text_bbox
                bw, bh = bx2 - bx1, by2 - by1
                # Compute neighbor-free horizontal span so tight bubbles never collide with neighbors
                free_left = margin_x
                free_right = img_w - margin_x
                for o_idx, ob in active_boxes:
                    if o_idx == idx:
                        continue
                    if ob[3] + gap <= by1 or by2 + gap <= ob[1]:
                        continue
                    ocx = (ob[0] + ob[2]) * 0.5
                    cx_cur = (bx1 + bx2) * 0.5
                    if ob[2] <= bx1 + 6 or ocx < cx_cur:
                        free_left = max(free_left, ob[2] + gap)
                    if ob[0] >= bx2 - 6 or ocx >= cx_cur:
                        free_right = min(free_right, ob[0] - gap)

                desired_w = max(bw, min(int(img_w * 0.28), max(125, int(bw * 1.38))))
                avail_w = max(bw, free_right - free_left)
                target_w = min(desired_w, avail_w)
                target_h = max(bh, min(int(img_h * 0.24), int(bh * 1.15)))
                cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                nx1 = max(free_left, min(max(free_left, free_right - target_w), int(round(cx - target_w * 0.5))))
                ny1 = max(margin_y, min(img_h - target_h - margin_y, int(round(cy - target_h * 0.5))))
                new_box = clamp_box((nx1, ny1, nx1 + target_w, ny1 + target_h))
                grew_significantly = (new_box[2] - new_box[0]) > bw * 1.08 or (new_box[3] - new_box[1]) > bh * 1.08
                copy.text_bbox = new_box
                if copy.bubble_bbox is not None and grew_significantly:
                    copy.bubble_bbox = new_box
                    copy.bubble_contour = None

    # Step 2: Iterative overlap resolution moving ONLY floating boxes (anchors keep real speech balloons fixed)
    floating_indices = [
        i
        for i, c in enumerate(adapted)
        if c.kind == "floating"
        and not (c.is_sfx and not (c.translated_text or "").strip())
        and (c.translated_text or c.source_text or "").strip()
    ]
    all_render_indices = [
        i
        for i, c in enumerate(adapted)
        if not (c.is_sfx and not (c.translated_text or "").strip())
        and (c.translated_text or c.source_text or "").strip()
    ]

    for _ in range(20):
        moved = False
        for i in floating_indices:
            a = adapted[i].text_bbox
            for j in all_render_indices:
                if i == j:
                    continue
                b = adapted[j].bubble_bbox or adapted[j].text_bbox
                if not _rect_overlap(a, b, gap=gap):
                    continue
                moved = True
                ox = min(a[2] + gap, b[2] + gap) - max(a[0], b[0])
                oy = min(a[3] + gap, b[3] + gap) - max(a[1], b[1])
                acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
                bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
                if oy <= ox * 1.15:
                    dir_y = -1 if acy <= bcy else 1
                    shift = int(math.ceil(oy * (0.6 if adapted[j].kind == "floating" else 1.0)))
                    a = clamp_box((a[0], a[1] + dir_y * shift, a[2], a[3] + dir_y * shift))
                    adapted[i].text_bbox = a
                else:
                    dir_x = -1 if acx <= bcx else 1
                    shift = int(math.ceil(ox * (0.6 if adapted[j].kind == "floating" else 1.0)))
                    a = clamp_box((a[0] + dir_x * shift, a[1], a[2] + dir_x * shift, a[3]))
                    adapted[i].text_bbox = a
        if not moved:
            break

    # Step 3: Page-level font harmonization across dialogue bubbles AND floating text.
    # Prevents a 2-word bubble from jumping to 34px right next to a 25-word dense bubble at 13px.
    bubble_sizes = [
        solve_layout(c, canvas_size=(img_w, img_h)).font_size
        for c in adapted
        if c.kind == "bubble" and (c.translated_text or c.source_text or "").strip()
    ]
    page_scale = max(1.0, min(1.65, min(img_w, img_h) / 850.0))
    ref_fs = (
        sorted(bubble_sizes)[len(bubble_sizes) // 2]
        if bubble_sizes
        else max(16, min(int(round(26 * page_scale)), int(round(min(img_w, img_h) * 0.024))))
    )
    max_float_fs = max(22, min(int(round(32 * page_scale)), int(round(ref_fs * 1.4))))
    max_normal_bubble_fs = max(22, min(int(round(34 * page_scale)), int(round(ref_fs * 1.45))))
    max_shout_bubble_fs = max(26, min(int(round(38 * page_scale)), int(round(ref_fs * 1.65))))

    for copy in adapted:
        if copy.font_size is not None:
            continue
        if copy.is_sfx and not (copy.translated_text or "").strip():
            continue
        txt = (copy.translated_text or copy.source_text or "").strip()
        if not txt:
            continue
        lay = solve_layout(copy, canvas_size=(img_w, img_h))
        if copy.kind == "floating":
            if lay.font_size > max_float_fs:
                copy.font_size = float(max_float_fs)
        elif len(bubble_sizes) >= 3:
            is_shout = "!" in txt or (len(txt) <= 12 and txt.isupper())
            cap_fs = max_shout_bubble_fs if is_shout else max_normal_bubble_fs
            if lay.font_size > cap_fs:
                copy.font_size = float(cap_fs)

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

        # Clamp line centers inside page safe margin so glyphs & strokes never clip at left/right border
        edge_safe = max(6.0, w * 0.008) + stroke_px + 2.0
        clamped_lines: list[LinePlacement] = []
        for lp in layout.lines:
            half_w = lp.width * 0.5
            safe_cx = min(max(lp.center_x, edge_safe + half_w), max(edge_safe + half_w, w - edge_safe - half_w))
            clamped_lines.append(LinePlacement(text=lp.text, center_x=safe_cx, top_y=lp.top_y, width=lp.width))

        # Compute tight bounding patch around the lines + stroke padding
        size_hi = layout.font_size * scale
        pad = int(math.ceil(stroke_px * 2 + layout.font_size * 0.65))
        min_x = min(lp.center_x - lp.width * 0.5 for lp in clamped_lines) - pad
        max_x = max(lp.center_x + lp.width * 0.5 for lp in clamped_lines) + pad
        min_y = min(lp.top_y for lp in clamped_lines) - pad
        max_y = max(lp.top_y + layout.font_size * 1.35 for lp in clamped_lines) + pad

        px1 = max(0, int(math.floor(min_x)))
        py1 = max(0, int(math.floor(min_y)))
        px2 = min(w, int(math.ceil(max_x)))
        py2 = min(h, int(math.ceil(max_y)))
        pw, ph = px2 - px1, py2 - py1
        if pw < 4 or ph < 4:
            continue

        eff_sx = max(0.75, min(1.0, float(layout.scale_x)))
        patch_w_hi = max(8, int(round(pw * scale / eff_sx)))
        patch_h_hi = ph * scale
        patch_hi = Image.new("RGBA", (patch_w_hi, patch_h_hi), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch_hi)
        stroke_hi = max(1, int(round(stroke_px * scale))) if stroke_rgba else 0
        primary_hi = _load_font(layout.font_path, size_hi)
        base_ascent, _ = primary_hi.getmetrics()
        x_scale_hi = scale / eff_sx

        for lp in clamped_lines:
            runs = _split_font_runs(lp.text, layout.font_path, size_hi)
            run_widths = [
                _measure_text_width(_load_font(fp, size_hi), chunk) for fp, chunk in runs
            ]
            total_w_hi = sum(run_widths)
            start_x = (lp.center_x - px1) * x_scale_hi - total_w_hi * 0.5
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
