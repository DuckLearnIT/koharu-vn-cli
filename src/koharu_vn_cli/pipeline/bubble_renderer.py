"""Vector-quality 2x supersampled speech bubble rendering & anti-overlap placement for CG compose."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from koharu_vn_cli.models import BubbleStyle, TextRegionItem


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 8) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def resolve_box_overlaps(
    items: list[TextRegionItem],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.015,
    gap: int = 10,
) -> list[TextRegionItem]:
    """
    Clamp proposed boxes inside safe page margins and iteratively separate overlapping boxes.
    """
    margin_x = max(6, int(img_w * margin_ratio))
    margin_y = max(6, int(img_h * margin_ratio))

    def clamp_rect(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        bw = min(max(36, b[2] - b[0]), img_w - 2 * margin_x)
        bh = min(max(24, b[3] - b[1]), img_h - 2 * margin_y)
        x1 = max(margin_x, min(img_w - margin_x - bw, b[0]))
        y1 = max(margin_y, min(img_h - margin_y - bh, b[1]))
        return (x1, y1, x1 + bw, y1 + bh)

    boxes = [clamp_rect(it.bubble_bbox or it.text_bbox) for it in items]

    for _ in range(25):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a = boxes[i]
                b = boxes[j]
                if not _boxes_overlap(a, b, pad=gap):
                    continue
                moved = True
                ox = min(a[2] + gap, b[2] + gap) - max(a[0], b[0])
                oy = min(a[3] + gap, b[3] + gap) - max(a[1], b[1])
                acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
                bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5

                # Push along the axis requiring smaller displacement
                if ox <= oy:
                    shift = int(math.ceil(ox * 0.55))
                    dir_x = 1 if bcx >= acx else -1
                    b = clamp_rect((b[0] + dir_x * shift, b[1], b[2] + dir_x * shift, b[3]))
                    a = clamp_rect((a[0] - dir_x * (shift // 2), a[1], a[2] - dir_x * (shift // 2), a[3]))
                else:
                    shift = int(math.ceil(oy * 0.55))
                    dir_y = 1 if bcy >= acy else -1
                    b = clamp_rect((b[0], b[1] + dir_y * shift, b[2], b[3] + dir_y * shift))
                    a = clamp_rect((a[0], a[1] - dir_y * (shift // 2), a[2], a[3] - dir_y * (shift // 2)))
                boxes[i] = a
                boxes[j] = b
        if not moved:
            break

    updated: list[TextRegionItem] = []
    for item, box in zip(items, boxes):
        item_copy = item.model_copy()
        if item_copy.kind == "bubble":
            item_copy.bubble_bbox = box
            item_copy.text_bbox = box
        else:
            item_copy.text_bbox = box
            item_copy.bubble_bbox = None
        updated.append(item_copy)
    return updated


def _compute_tail_polygon(
    box: tuple[int, int, int, int],
    target: tuple[int, int],
    scale: int = 2,
) -> list[tuple[float, float]] | None:
    """Build a clean triangular speech-bubble tail pointing from ellipse center toward target."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5 * scale
    cy = (y1 + y2) * 0.5 * scale
    rx = max(10.0, (x2 - x1) * 0.5 * scale)
    ry = max(10.0, (y2 - y1) * 0.5 * scale)

    tx = target[0] * scale
    ty = target[1] * scale
    dx = tx - cx
    dy = ty - cy
    dist = math.hypot(dx, dy)
    if dist < 1.0:
        return None

    # Normalized ray intersection with ellipse boundary
    ux, uy = dx / dist, dy / dist
    ell_r = 1.0 / math.sqrt((ux / rx) ** 2 + (uy / ry) ** 2)
    edge_dist = ell_r
    if dist <= edge_dist * 1.05:
        return None

    # Stop tip ~72% from edge toward target (or cap max tail length) so it never pokes faces
    tail_len = min(max(16.0 * scale, (dist - edge_dist) * 0.72), max(rx, ry) * 1.15)
    if tail_len < 8.0 * scale:
        return None

    tip = (cx + ux * (edge_dist + tail_len), cy + uy * (edge_dist + tail_len))

    # Base points inside the ellipse so tail merges seamlessly
    px, py = -uy, ux
    base_half = max(6.0 * scale, min(14.0 * scale, min(rx, ry) * 0.24))
    anchor_r = edge_dist * 0.65
    base1 = (cx + ux * anchor_r + px * base_half, cy + uy * anchor_r + py * base_half)
    base2 = (cx + ux * anchor_r - px * base_half, cy + uy * anchor_r - py * base_half)
    return [base1, tip, base2]


def render_bubbles(canvas_rgba: Image.Image, regions: Sequence[TextRegionItem]) -> Image.Image:
    """
    Draw synthetic speech bubbles (white_oval, dark_translucent, speech_tail) for `compose`
    regions using 2x supersampling for crisp anti-aliased edges.
    """
    draw_list = [
        r for r in regions
        if r.draw_bubble and r.kind == "bubble" and r.bubble_style != BubbleStyle.NONE
    ]
    if not draw_list:
        return canvas_rgba

    w, h = canvas_rgba.size
    scale = 2
    overlay_hi = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))

    for reg in draw_list:
        box = reg.bubble_bbox or reg.text_bbox
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            continue

        if reg.bubble_style == BubbleStyle.DARK_TRANSLUCENT:
            # Feathered dark oval (matches translucent charcoal ovals in CG edits)
            # Ensure >= 3.2 * blur_radius outside the ellipse so blur fades to 0 alpha smoothly
            bw_hi = (x2 - x1) * scale
            bh_hi = (y2 - y1) * scale
            blur_radius = max(4.0, min(bw_hi, bh_hi) * 0.055)
            pad = int(math.ceil(blur_radius * 3.2))

            ox1 = max(0, x1 * scale - pad)
            oy1 = max(0, y1 * scale - pad)
            ox2 = min(w * scale, x2 * scale + pad)
            oy2 = min(h * scale, y2 * scale + pad)
            pw, ph = ox2 - ox1, oy2 - oy1
            if pw < 8 or ph < 8:
                continue

            mask_hi = Image.new("L", (pw, ph), 0)
            md = ImageDraw.Draw(mask_hi)
            # Draw ellipse slightly inset from target box, keeping >= 3.2*sigma margin to patch border
            ex1 = (x1 * scale - ox1) + max(2, int(blur_radius * 0.45))
            ey1 = (y1 * scale - oy1) + max(2, int(blur_radius * 0.45))
            ex2 = (x2 * scale - ox1) - max(2, int(blur_radius * 0.45))
            ey2 = (y2 * scale - oy1) - max(2, int(blur_radius * 0.45))
            if ex2 > ex1 + 4 and ey2 > ey1 + 4:
                md.ellipse((ex1, ey1, ex2, ey2), fill=220)
            mask_hi = mask_hi.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            dark_patch = Image.new("RGBA", (pw, ph), (12, 12, 16, 255))
            dark_patch.putalpha(mask_hi)
            overlay_hi.alpha_composite(dark_patch, dest=(ox1, oy1))
            continue

        # WHITE_OVAL or SPEECH_TAIL
        sx1, sy1, sx2, sy2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        stroke_px = max(2, int(round(1.6 * scale)))

        tail_pts = None
        if reg.bubble_style == BubbleStyle.SPEECH_TAIL and reg.tail_target is not None:
            tail_pts = _compute_tail_polygon(box, reg.tail_target, scale=scale)

        layer_hi = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer_hi)

        # Pass 1: Dark outer outline (expanded by stroke_px) so tail & oval share one seamless border
        outline_col = (25, 25, 28, 245)
        ld.ellipse(
            (sx1 - stroke_px, sy1 - stroke_px, sx2 + stroke_px, sy2 + stroke_px),
            fill=outline_col,
        )
        if tail_pts is not None:
            # Expand tail slightly along normals using thick line + polygon fill
            ld.polygon(tail_pts, fill=outline_col)
            ld.line(tail_pts + [tail_pts[0]], fill=outline_col, width=stroke_px * 2, joint="curve")

        # Pass 2: Crisp white interior on top (zero internal seam line!)
        fill_col = (255, 255, 255, 248)
        ld.ellipse((sx1, sy1, sx2, sy2), fill=fill_col)
        if tail_pts is not None:
            ld.polygon(tail_pts, fill=fill_col)

        overlay_hi = Image.alpha_composite(overlay_hi, layer_hi)

    overlay_1x = overlay_hi.resize((w, h), Image.Resampling.LANCZOS)
    return Image.alpha_composite(canvas_rgba, overlay_1x)
