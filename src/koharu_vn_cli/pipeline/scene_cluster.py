"""Fast local visual clustering to group consecutive similar images into storylines/scenes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_TRAILING_INT_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class SceneGroup:
    """Consecutive pages belonging to one visual storyline / CG set / mini-oneshot."""

    scene_id: int
    page_indices: list[int]
    rep_index: int

    @property
    def start_page(self) -> int:
        return self.page_indices[0] + 1

    @property
    def end_page(self) -> int:
        return self.page_indices[-1] + 1


def _trailing_int(path: Path) -> int | None:
    m = _TRAILING_INT_RE.search(path.stem)
    return int(m.group(1)) if m else None


def _image_signature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute compact L2-normalized structural vector + RGB histogram vector."""
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32)

    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).reshape(-1)
    gray = gray - float(gray.mean())
    g_norm = float(np.linalg.norm(gray))
    if g_norm > 1e-6:
        gray = gray / g_norm
    else:
        # Uniform frame: fall back to unit constant vector so two uniform tiles don't score 0 structure
        gray = np.full_like(gray, 1.0 / np.sqrt(gray.size))

    hists: list[np.ndarray] = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=16, range=(0.0, 256.0))
        hists.append(h.astype(np.float32))
    hist = np.concatenate(hists)
    h_norm = float(np.linalg.norm(hist))
    if h_norm > 1e-6:
        hist = hist / h_norm
    return gray, hist


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def cluster_story_scenes(
    image_paths: list[Path],
    *,
    similarity_threshold: float = 0.62,
    borderline_gap_threshold: float = 0.74,
) -> list[SceneGroup]:
    """Group consecutive visually similar pages into distinct `SceneGroup` storylines."""
    n = len(image_paths)
    if n == 0:
        return []
    if n == 1:
        return [SceneGroup(scene_id=1, page_indices=[0], rep_index=0)]

    sigs = [_image_signature(p) for p in image_paths]
    nums = [_trailing_int(p) for p in image_paths]

    groups: list[list[int]] = [[0]]
    anchor_g, anchor_h = sigs[0]

    for i in range(1, n):
        g_cur, h_cur = sigs[i]
        g_prev, h_prev = sigs[i - 1]

        sim_prev = 0.68 * max(0.0, _cosine(g_cur, g_prev)) + 0.32 * max(0.0, _cosine(h_cur, h_prev))
        sim_anchor = 0.68 * max(0.0, _cosine(g_cur, anchor_g)) + 0.32 * max(0.0, _cosine(h_cur, anchor_h))
        eff_sim = max(sim_prev, sim_anchor * 0.96)

        num_prev, num_cur = nums[i - 1], nums[i]
        large_num_gap = num_prev is not None and num_cur is not None and (num_cur - num_prev) >= 5

        should_split = eff_sim < similarity_threshold or (large_num_gap and eff_sim < borderline_gap_threshold)
        if should_split:
            groups.append([i])
            anchor_g, anchor_h = g_cur, h_cur
        else:
            groups[-1].append(i)

    scenes: list[SceneGroup] = []
    for s_idx, idxs in enumerate(groups, start=1):
        rep = idxs[len(idxs) // 3]
        scenes.append(SceneGroup(scene_id=s_idx, page_indices=idxs, rep_index=rep))
    return scenes


def page_to_scene_map(scenes: list[SceneGroup]) -> dict[int, SceneGroup]:
    """Map 0-based `page_index` -> `SceneGroup`."""
    mapping: dict[int, SceneGroup] = {}
    for sc in scenes:
        for idx in sc.page_indices:
            mapping[idx] = sc
    return mapping
