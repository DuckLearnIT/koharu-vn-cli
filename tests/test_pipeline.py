"""Automated tests for Vietnamese typesetting, deduplication, color classification, and CG bubble rendering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from koharu_vn_cli.cli import app
from koharu_vn_cli.models import BubbleStyle, ComposePreset, PageDocument, TextRegionItem
from koharu_vn_cli.pipeline.bubble_renderer import render_bubbles, resolve_box_overlaps
from koharu_vn_cli.pipeline.detector import DetectedRegion, _containment_nms
from koharu_vn_cli.pipeline.gemini_client import VisionLLMClient, _extract_json_block
from koharu_vn_cli.pipeline.inpaint import classify_region_colors
from koharu_vn_cli.pipeline.typesetter import render_text_regions, solve_layout


def test_containment_nms_eliminates_nested_subbox() -> None:
    boxes = [
        ((50, 80, 130, 180), 0.90, 1),
        ((70, 90, 128, 175), 0.72, 1),  # nested sub-box causing duplicate overlap
        ((100, 20, 130, 48), 0.85, 2),
        ((100, 20, 130, 48), 0.60, 1),  # cross-class duplicate
    ]
    kept = _containment_nms(boxes)
    assert len(kept) == 2
    assert kept[0][0] == (50, 80, 130, 180)
    assert kept[1][0] == (100, 20, 130, 48)


def test_vietnamese_typesetter_fills_tall_narrow_bubble() -> None:
    """Ensure tall bubbles (like User Image 2) use a large readable font instead of tiny 9px center text."""
    reg = TextRegionItem(
        id=1,
        kind="bubble",
        text_bbox=(23, 75, 70, 109),
        bubble_bbox=(14, 4, 82, 172),
        translated_text="Xin hãy thuần hóa con mèo bướng bỉnh này theo sở thích của thầy đi... ư...",
        font_name="mali-bold",
    )
    layout = solve_layout(reg)
    assert not layout.overflowed
    assert layout.font_size >= 11
    assert len(layout.lines) >= 5
    # Lines should span a healthy portion of the 168px bubble height
    span_y = (layout.lines[-1].top_y + layout.font_size) - layout.lines[0].top_y
    assert span_y >= 65.0


def test_color_classifier_avoids_thick_black_stroke_on_dark_bubble() -> None:
    """Ensure dark translucent bubbles render crisp white text with NO heavy black cloud border."""
    orig = np.full((120, 120, 3), 24, dtype=np.uint8)
    clean = np.full((120, 120, 3), 24, dtype=np.uint8)
    mask = np.zeros((120, 120), dtype=np.uint8)
    reg = DetectedRegion(
        text_bbox=(20, 20, 100, 100),
        bubble_bbox=(10, 10, 110, 110),
        bubble_contour=None,
        is_bubble=True,
        is_dark_bubble=True,
        confidence=0.95,
    )
    fill, stroke, s_factor, is_dark = classify_region_colors(orig, clean, mask, reg)
    assert is_dark is True
    assert fill == (255, 255, 255)
    assert stroke is None
    assert s_factor == 0.0


def test_compose_overlap_resolver_separates_colliding_bubbles() -> None:
    items = [
        TextRegionItem(
            id=1,
            kind="bubble",
            bubble_style=BubbleStyle.WHITE_OVAL,
            text_bbox=(100, 100, 220, 260),
            bubble_bbox=(100, 100, 220, 260),
            translated_text="Em không ngờ anh lại chờ ở đây...",
            draw_bubble=True,
        ),
        TextRegionItem(
            id=2,
            kind="bubble",
            bubble_style=BubbleStyle.DARK_TRANSLUCENT,
            text_bbox=(150, 130, 270, 280),
            bubble_bbox=(150, 130, 270, 280),
            translated_text="Thật là... cứng đầu quá đi mất!",
            is_dark_bubble=True,
            draw_bubble=True,
        ),
    ]
    resolved = resolve_box_overlaps(items, 800, 600)
    a = resolved[0].bubble_bbox
    b = resolved[1].bubble_bbox
    assert a is not None and b is not None
    inter_x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    assert inter_x * inter_y == 0


def test_render_compose_presets_both_bubble_and_floating() -> None:
    canvas = Image.new("RGBA", (600, 420), (140, 165, 210, 255))
    items = [
        TextRegionItem(
            id=1,
            kind="bubble",
            bubble_style=BubbleStyle.WHITE_OVAL,
            text_bbox=(40, 30, 145, 240),
            bubble_bbox=(40, 30, 145, 240),
            translated_text="Ơ kìa... sao tự nhiên lại nắm tay em thế?",
            text_color=(18, 18, 18),
            draw_bubble=True,
        ),
        TextRegionItem(
            id=2,
            kind="bubble",
            bubble_style=BubbleStyle.DARK_TRANSLUCENT,
            text_bbox=(430, 40, 535, 250),
            bubble_bbox=(430, 40, 535, 250),
            translated_text="Biết ngay là em cũng đang mong chờ mà...",
            text_color=(255, 255, 255),
            is_dark_bubble=True,
            draw_bubble=True,
        ),
        TextRegionItem(
            id=3,
            kind="bubble",
            bubble_style=BubbleStyle.SPEECH_TAIL,
            text_bbox=(200, 25, 360, 125),
            bubble_bbox=(200, 25, 360, 125),
            tail_target=(290, 220),
            translated_text="Đừng nhìn chằm chằm như vậy!",
            text_color=(18, 18, 18),
            draw_bubble=True,
        ),
        TextRegionItem(
            id=4,
            kind="floating",
            bubble_style=BubbleStyle.NONE,
            text_bbox=(220, 310, 340, 370),
            translated_text="THÌNH THỊCH♡",
            text_color=(255, 195, 220),
            stroke_color=(40, 15, 30),
            angle=-8.0,
            font_name="pangolin",
            draw_bubble=False,
        ),
    ]
    out = render_bubbles(canvas, items)
    out = render_text_regions(out, items)
    assert out.size == (600, 420)
    # Verify pixels actually changed inside both bubble and floating areas
    arr = np.array(out)
    assert np.any(arr[50:120, 50:120, :3] > 240)  # white oval
    assert np.any(arr[80:200, 450:510, 0] < 60)  # dark translucent oval


def test_extract_json_block_handles_markdown_fences() -> None:
    raw = '```json\n{"page_summary": "ok", "items": [{"id": 1}]}\n```'
    parsed = _extract_json_block(raw)
    assert parsed["page_summary"] == "ok"
    assert parsed["items"][0]["id"] == 1


def test_compose_cg_json_parsing_respects_preset(monkeypatch) -> None:
    client = VisionLLMClient(api_key="dummy-test-key")
    mock_payload = {
        "page_summary": "Hai nhân vật đang trò chuyện thân mật.",
        "items": [
            {
                "id": 1,
                "kind": "bubble",
                "bubble_style": "speech_tail",
                "box_2d": [50, 50, 450, 220],
                "tail_target": [300, 350],
                "text": "Đừng lại gần quá như thế chứ...",
            },
            {
                "id": 2,
                "kind": "floating",
                "bubble_style": "none",
                "box_2d": [700, 400, 820, 580],
                "text": "CHỤT♡",
                "text_color": [255, 255, 255],
                "stroke_color": [250, 250, 250],  # deliberately low contrast -> must auto-fix!
                "angle": -6,
            },
        ],
    }
    monkeypatch.setattr(client, "_call_json", lambda *a, **kw: mock_payload)
    dummy_rgb = np.zeros((400, 600, 3), dtype=np.uint8)

    # Under CG_OVAL preset, speech_tail automatically normalizes to white_oval
    items_oval, summary = client.compose_cg_page(dummy_rgb, preset=ComposePreset.CG_OVAL)
    assert summary == "Hai nhân vật đang trò chuyện thân mật."
    assert items_oval[0].bubble_style == BubbleStyle.WHITE_OVAL
    assert items_oval[1].kind == "floating"
    assert items_oval[1].stroke_color == (18, 18, 18)  # fixed low contrast!

    # Under MANGA_TAIL preset, speech_tail and tail_target are preserved
    items_tail, _ = client.compose_cg_page(dummy_rgb, preset=ComposePreset.MANGA_TAIL)
    assert items_tail[0].bubble_style == BubbleStyle.SPEECH_TAIL
    assert items_tail[0].tail_target is not None


def test_dark_translucent_bubble_fades_smoothly_without_box_clip() -> None:
    canvas = Image.new("RGBA", (400, 400), (200, 200, 200, 255))
    reg = TextRegionItem(
        id=1,
        kind="bubble",
        bubble_style=BubbleStyle.DARK_TRANSLUCENT,
        text_bbox=(120, 60, 280, 340),
        bubble_bbox=(120, 60, 280, 340),
        translated_text="Nói nhỏ thôi...",
        draw_bubble=True,
    )
    out = np.array(render_bubbles(canvas, [reg]))
    # Center is dark translucent, while 45px outside top/bottom fades completely back to background
    assert out[200, 200, 0] < 90
    assert abs(int(out[15, 200, 0]) - 200) <= 1


def test_cli_rerender_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "page_01.png"
    out_dir = tmp_path / "page_01_out"
    cache_dir = out_dir / ".koharu_cache"
    cache_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), (180, 190, 210)).save(src)

    doc = PageDocument(
        image_path=str(src),
        cleaned_image_path=str(src),
        width=320,
        height=240,
        mode="compose",
        regions=[
            TextRegionItem(
                id=1,
                kind="bubble",
                bubble_style=BubbleStyle.SPEECH_TAIL,
                text_bbox=(30, 20, 210, 90),
                bubble_bbox=(30, 20, 210, 90),
                tail_target=(150, 170),
                translated_text="Ơ kìa!",
                draw_bubble=True,
            )
        ],
    )
    (cache_dir / "page_01.koharu.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")

    result = CliRunner().invoke(app, ["rerender", str(out_dir), "--font", "pangolin", "--font-scale", "1.1"])
    assert result.exit_code == 0
    assert (out_dir / "page_01.png").is_file()


def test_list_available_models_filters_and_ranks_newest(monkeypatch) -> None:
    import httpx

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "models": [
                    {
                        "name": "models/gemini-2.0-flash",
                        "displayName": "Gemini 2.0 Flash",
                        "description": "Older fast model",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "displayName": "Embedding",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/gemini-2.5-flash",
                        "displayName": "Gemini 2.5 Flash",
                        "description": "Balanced model",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-3.0-pro",
                        "displayName": "Gemini 3.0 Pro",
                        "description": "Newest flagship",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/imagen-3.0",
                        "displayName": "Imagen",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            }

    class FakeClient:
        def __init__(self, *a, **kw) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a) -> None:
            return None

        def get(self, url, headers=None):
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    models = VisionLLMClient(api_key="dummy").list_available_models()
    ids = [m[0] for m in models]
    assert ids == ["gemini-3.0-pro", "gemini-2.5-flash", "gemini-2.0-flash"]


def test_story_bible_pass_and_context_injection(tmp_path: Path, monkeypatch) -> None:
    from koharu_vn_cli.pipeline.runner import resolve_story_text

    lore_file = tmp_path / "story.txt"
    lore_file.write_text("Sư phụ (tóc trắng) và đệ tử (tóc đen) xưng ta - con.", encoding="utf-8")
    assert "Sư phụ" in (resolve_story_text(str(lore_file)) or "")

    pages = []
    for i in range(3):
        p = tmp_path / f"p{i + 1}.png"
        Image.new("RGB", (200, 260), (100 + i * 30, 120, 140)).save(p)
        pages.append(p)

    captured_calls: list[dict] = []

    def fake_call_json(sys_p: str, usr_p: str, images: list, temperature: float = 0.3) -> dict:
        captured_calls.append({"user": usr_p, "img_count": len(images)})
        if "Story Bible" in usr_p:
            return {
                "title": "Chương 1: Cuộc gặp lại",
                "genre_and_tone": "tình cảm, nhẹ nhàng",
                "pre_event_backstory": "Hai thầy trò từng xa cách 3 năm sau biến cố ở đền.",
                "during_event_plot": "Cả hai bất ngờ chạm mặt trước cổng đền và bắt đầu trò chuyện.",
                "full_plot_summary": "Hai nhân vật gặp lại ở sân đền rồi hóa giải hiểu lầm ở trang cuối.",
                "characters": [
                    {
                        "name": "Sư phụ",
                        "appearance": "tóc trắng",
                        "role_and_tone": "điềm đạm",
                        "pronoun_rules": "ta - con",
                    }
                ],
                "page_beats": {"1": "Mở đầu lạnh nhạt", "2": "Nhắc chuyện cũ", "3": "Hòa giải"},
            }
        return {
            "detected_language": "ja",
            "page_summary": "Trang 2 tiếp tục nhắc chuyện cũ.",
            "items": [{"id": 1, "source_text": "久しぶり", "vietnamese": "Lâu rồi không gặp, con.", "is_sfx": False}],
        }

    client = VisionLLMClient(api_key="dummy")
    monkeypatch.setattr(client, "_call_json", fake_call_json)

    bible = client.build_story_bible(pages, mode="translate", user_story=resolve_story_text(str(lore_file)))
    assert bible.title == "Chương 1: Cuộc gặp lại"
    assert "3 năm" in bible.pre_event_backstory
    assert "chạm mặt" in bible.during_event_plot
    assert bible.characters[0].pronoun_rules == "ta - con"
    assert "3" in bible.page_beats

    reg = TextRegionItem(id=1, kind="bubble", text_bbox=(10, 10, 120, 80))
    dummy_rgb = np.zeros((260, 200, 3), dtype=np.uint8)
    updated, summary = client.translate_page(
        dummy_rgb,
        [reg],
        story_bible=bible,
        page_index=2,
        prev_rgb=dummy_rgb,
        next_rgb=dummy_rgb,
    )
    assert updated[0].translated_text == "Lâu rồi không gặp, con."
    assert "ta - con" in captured_calls[-1]["user"]
    assert "Cốt truyện TRƯỚC sự kiện ảnh" in captured_calls[-1]["user"]
    assert "Diễn biến TRONG sự kiện ảnh" in captured_calls[-1]["user"]
    assert captured_calls[-1]["img_count"] == 4  # annotated + collage + prev_rgb + next_rgb


def test_call_json_fails_fast_on_404_and_emits_status(monkeypatch):
    import httpx
    import pytest

    statuses: list[str] = []
    attempts = 0

    def fake_rest(self, system_prompt, user_prompt, images, temperature=0.3):
        nonlocal attempts
        attempts += 1
        req = httpx.Request("POST", "https://example.invalid/v1beta/models/x:generateContent")
        resp = httpx.Response(404, request=req, text='{"error":{"message":"models/x is not found"}}')
        raise httpx.HTTPStatusError("404 Not Found", request=req, response=resp)

    client = VisionLLMClient(api_key="dummy", model="gemini-missing", on_status=statuses.append)
    monkeypatch.setattr(VisionLLMClient, "_call_gemini_rest", fake_rest)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        client._call_json("sys", "usr", [("image/jpeg", "AAAA")])

    assert attempts == 1  # must fail fast without 4x retry sleep
    assert any("Đang gửi 1 ảnh" in s for s in statuses)


def test_call_json_retries_on_429_and_succeeds(monkeypatch):
    import httpx

    statuses: list[str] = []
    attempts = 0

    def fake_rest(self, system_prompt, user_prompt, images, temperature=0.3):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            req = httpx.Request("POST", "https://example.invalid/v1beta/models/x:generateContent")
            resp = httpx.Response(429, request=req, text='{"error":{"message":"rate limit"}}')
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)
        return '{"ok": true}'

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.time.sleep", lambda _s: None)
    client = VisionLLMClient(api_key="dummy", model="gemini-test", on_status=statuses.append)
    monkeypatch.setattr(VisionLLMClient, "_call_gemini_rest", fake_rest)

    res = client._call_json("sys", "usr", [("image/jpeg", "AAAA")])
    assert res == {"ok": True}
    assert attempts == 2
    assert any("HTTP 429" in s for s in statuses)


def test_tags_injection_and_natural_spoken_vietnamese_rules(monkeypatch):
    captured: list[dict[str, str]] = []

    def fake_call_json(system_prompt, user_prompt, images, temperature=0.3):
        captured.append({"system": system_prompt, "user": user_prompt})
        if "Story Bible" in user_prompt:
            return {"title": "Test", "genre_and_tone": "NTR", "characters": [], "page_beats": {}}
        return {"page_summary": "ok", "items": []}

    client = VisionLLMClient(api_key="dummy")
    monkeypatch.setattr(client, "_call_json", fake_call_json)

    dummy_rgb = np.zeros((240, 200, 3), dtype=np.uint8)
    client.compose_cg_page(dummy_rgb, tags="vanilla, sweet")
    assert "[THỂ LOẠI / TAGS]: vanilla, sweet" in captured[-1]["user"]
    assert "KHẨU NGỮ ĐỜI THƯỜNG" in captured[-1]["system"]
    assert "HUMAN.md" in captured[-1]["system"]

    reg = TextRegionItem(id=1, kind="bubble", text_bbox=(10, 10, 120, 80))
    client.translate_page(dummy_rgb, [reg], tags="NTR, angst")
    assert "[THỂ LOẠI / TAGS]: NTR, angst" in captured[-1]["user"]
    assert "KHẨU NGỮ TIẾNG VIỆT ĐỜI THƯỜNG" in captured[-1]["system"]
    assert "HUMAN.md" in captured[-1]["system"]


def test_single_full_sheet_packing_and_compose_cg_sheet(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.pipeline.gemini_client import _build_story_contact_sheets

    img_paths: list[Path] = []
    sizes: list[tuple[int, int]] = []
    for i in range(17):
        p = tmp_path / f"cg_{i + 1:02d}.png"
        Image.new("RGB", (400, 500), (60 + i * 5, 80, 120)).save(p)
        img_paths.append(p)
        sizes.append((400, 500))

    sheets = _build_story_contact_sheets(img_paths, max_pages=36)
    assert len(sheets) == 1  # 17 images packed into 1 single full sheet

    client = VisionLLMClient(api_key="dummy")

    def fake_call_json(system_prompt, user_prompt, images, temperature=0.65):
        assert len(images) == 1
        return {
            "pages": [
                {
                    "page": 1,
                    "page_summary": "Cảnh mở đầu",
                    "items": [
                        {
                            "id": 1,
                            "kind": "bubble",
                            "bubble_style": "white_oval",
                            "box_2d": [50, 50, 300, 450],
                            "text": "Đợi chút đã nào!",
                        }
                    ],
                },
                {
                    "page": 17,
                    "page_summary": "Cảnh kết",
                    "items": [
                        {
                            "id": 1,
                            "kind": "floating",
                            "bubble_style": "none",
                            "box_2d": [600, 500, 800, 900],
                            "text": "Ư...",
                        }
                    ],
                },
            ]
        }

    monkeypatch.setattr(client, "_call_json", fake_call_json)
    res = client.compose_cg_sheet(img_paths, sizes, preset=ComposePreset.CG_OVAL, tags="vanilla")
    assert 1 in res and 17 in res
    assert res[1][0][0].translated_text == "Đợi chút đã nào!"
    assert res[1][0][0].text_bbox == (20, 25, 180, 150)
    assert res[17][0][0].kind == "floating"


def test_prohibited_content_fails_fast_without_retries(monkeypatch):
    import pytest

    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError

    attempts = 0

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}

    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            return FakeResp()

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.httpx.Client", FakeClient)
    client = VisionLLMClient(api_key="dummy", model="gemini-4-hydrogen")

    with pytest.raises(ContentBlockedError, match="PROHIBITED_CONTENT"):
        client._call_json("sys", "usr", [("image/jpeg", "AAAA")])

    assert attempts == 1


def test_dual_model_story_master_and_worker_compose(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.models import StoryBible
    from koharu_vn_cli.pipeline.runner import run_compose_pipeline

    in_dir = tmp_path / "cg_set"
    in_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (320, 240), (90 + i * 20, 110, 140)).save(in_dir / f"p{i + 1}.png")

    calls: list[tuple[str, str]] = []

    def fake_build_bible(self, image_paths, mode="translate", user_story=None, extra_instructions=None, tags=None, scenes=None):
        calls.append(("pass1_story", self.model))
        return StoryBible(title="Dual Test", genre_and_tone="vanilla", full_plot_summary="Cốt truyện chuẩn")

    def fake_compose_sheet(self, image_paths, image_sizes, preset=ComposePreset.CG_OVAL, density="medium", story_prompt=None, story_bible=None, tags=None):
        calls.append(("pass2_compose", self.model))
        assert story_bible is not None and story_bible.title == "Dual Test"
        return {
            1: ([TextRegionItem(id=1, kind="bubble", bubble_style=BubbleStyle.WHITE_OVAL, text_bbox=(10, 10, 160, 80), bubble_bbox=(10, 10, 160, 80), translated_text="Ổn rồi!", draw_bubble=True)], "p1"),
            2: ([TextRegionItem(id=1, kind="bubble", bubble_style=BubbleStyle.WHITE_OVAL, text_bbox=(10, 10, 160, 80), bubble_bbox=(10, 10, 160, 80), translated_text="Đi thôi!", draw_bubble=True)], "p2"),
        }

    monkeypatch.setattr(VisionLLMClient, "build_story_bible", fake_build_bible)
    monkeypatch.setattr(VisionLLMClient, "compose_cg_sheet", fake_compose_sheet)

    out_dir, docs = run_compose_pipeline(
        in_dir,
        output=tmp_path / "out",
        api_key="dummy",
        model="gemini-fast-worker",
        story_model="gemini-strong-master",
    )
    assert len(docs) == 2
    assert (out_dir / "p1.png").is_file()
    assert calls == [
        ("pass1_story", "gemini-strong-master"),
        ("pass2_compose", "gemini-fast-worker"),
    ]


def test_narrow_floating_box_widens_and_dark_rect_centers_lines():
    from koharu_vn_cli.pipeline.typesetter import adapt_regions_for_readability, solve_layout

    # 1. Tall narrow vertical Japanese floating text column (36x300)
    floating_reg = TextRegionItem(
        id=1,
        kind="floating",
        text_bbox=(200, 100, 236, 400),
        translated_text="Áaaa... không... ngập hết cả vào rồi...",
    )
    adapted = adapt_regions_for_readability([floating_reg], img_w=900, img_h=1200)
    assert adapted[0].text_bbox[2] - adapted[0].text_bbox[0] >= 160
    layout_float = solve_layout(adapted[0], canvas_size=(900, 1200))
    assert layout_float.font_size >= 14

    # 2. Dark translucent rectangular box with noisy asymmetric contour
    dark_rect = TextRegionItem(
        id=2,
        kind="bubble",
        text_bbox=(100, 100, 280, 340),
        bubble_bbox=(100, 100, 280, 340),
        bubble_contour=[(100, 100), (220, 150), (280, 220), (275, 340), (140, 340), (100, 200)],
        translated_text="Hôm nay sẽ là lần cuối, để lộ ra thì sợ chết khiếp ấy chứ...",
        is_dark_bubble=True,
        draw_bubble=False,
    )
    layout_rect = solve_layout(dark_rect, canvas_size=(900, 1200))
    box_cx = 190.0
    assert len(layout_rect.lines) >= 2
    assert all(abs(lp.center_x - box_cx) < 1.0 for lp in layout_rect.lines)


def test_multi_api_key_rotation_on_quota_or_auth_error(monkeypatch):
    import httpx

    used_keys: list[str] = []
    statuses: list[str] = []

    def fake_rest(self, system_prompt, user_prompt, images, temperature=0.3):
        used_keys.append(self.api_key)
        if self.api_key == "key_1":
            req = httpx.Request("POST", "https://example.invalid")
            raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req, text="quota"))
        if self.api_key == "key_2":
            req = httpx.Request("POST", "https://example.invalid")
            raise httpx.HTTPStatusError("403", request=req, response=httpx.Response(403, request=req, text="disabled"))
        return '{"rotated": true}'

    monkeypatch.setattr(VisionLLMClient, "_call_gemini_rest", fake_rest)
    client = VisionLLMClient(api_keys=["key_1", "key_2", "key_3"], on_status=statuses.append)
    res = client._call_json("sys", "usr", [("image/jpeg", "AAAA")])

    assert res == {"rotated": True}
    assert used_keys == ["key_1", "key_2", "key_3"]
    assert any("Đổi sang API key #2/3" in s for s in statuses)
    assert any("Đổi sang API key #3/3" in s for s in statuses)


def test_collect_inputs_handles_extensionless_webp_and_sorts_by_page_number(tmp_path: Path):
    import io

    from koharu_vn_cli.pipeline.runner import _output_image_name, collect_inputs

    buf = io.BytesIO()
    Image.new("RGB", (80, 100), (120, 140, 160)).save(buf, format="WEBP")
    webp_bytes = buf.getvalue()

    names = [
        "file_1745069399225_17_webp_8c4664",
        "file_1745069383855_3_webp_4c8aee",
        "file_1745069397662_15_webp_2f877b",
    ]
    for name in names:
        (tmp_path / name).write_bytes(webp_bytes)

    imgs, _ = collect_inputs(tmp_path, extract_dir=None)
    assert [p.name for p in imgs] == [
        "file_1745069383855_3_webp_4c8aee",
        "file_1745069397662_15_webp_2f877b",
        "file_1745069399225_17_webp_8c4664",
    ]
    assert _output_image_name(imgs[0]) == "file_1745069383855_3_webp_4c8aee.webp"


def test_seg_box_expansion_and_outlined_floating_erasure_mask():
    from koharu_vn_cli.pipeline.detector import _expand_box_with_seg_mask
    from koharu_vn_cli.pipeline.inpaint import build_erasure_mask

    # 1. Clipping recovery: connected glyph component extends 12px above RT-DETR box
    seg = np.zeros((200, 200), dtype=np.uint8)
    seg[40:120, 80:110] = 220
    clipped_box = (78, 55, 112, 122)
    expanded = _expand_box_with_seg_mask(clipped_box, seg)
    assert expanded[1] <= 40  # top extended to include full glyph

    # 2. Outlined floating text on purple-brown cloth: dark 3px outline around bright stroke
    rgb = np.full((180, 180, 3), (85, 62, 72), dtype=np.uint8)
    # Dark outer outline (16px wide) vs bright inner stroke (6px wide covered by CTD)
    rgb[50:120, 72:88] = (18, 12, 16)
    rgb[54:116, 76:84] = (240, 235, 240)
    seg_mask = np.zeros((180, 180), dtype=np.uint8)
    seg_mask[54:116, 76:84] = 210

    reg = DetectedRegion(
        text_bbox=(74, 52, 86, 118),
        bubble_bbox=None,
        bubble_contour=None,
        is_bubble=False,
        is_dark_bubble=True,
        confidence=0.85,
    )
    mask = build_erasure_mask(rgb, seg_mask, [reg])
    # Both bright interior AND dark outer stroke border pixels must be masked!
    assert mask[80, 80] == 255
    assert mask[80, 73] == 255


def test_ctd_blk_decode_and_unet_dbnet_fusion():
    from koharu_vn_cli.pipeline.detector import ComicDetector, _decode_ctd_blk

    # 1. Verify YOLOv5 blk decoding in 1024-scaled coords with NMS + offset + clamping
    blk = np.zeros((1, 4, 7), dtype=np.float32)
    blk[0, 0] = [512.0, 256.0, 128.0, 256.0, 0.92, 0.05, 0.96]  # high-conf box
    blk[0, 1] = [514.0, 258.0, 124.0, 252.0, 0.70, 0.05, 0.95]  # overlapping duplicate -> suppressed
    blk[0, 2] = [200.0, 200.0, 40.0, 40.0, 0.15, 0.10, 0.80]    # low-conf -> filtered
    blk[0, 3] = [800.0, 800.0, 2.0, 2.0, 0.99, 0.01, 0.99]      # tiny 2px -> filtered

    decoded = _decode_ctd_blk(blk, scale=2.0, crop_w=512, crop_h=512, offset_x=10, offset_y=20, conf_thresh=0.35)
    assert len(decoded) == 1
    box, score, lbl = decoded[0]
    assert box == (234, 84, 298, 212)  # (512+-64)/2 + 10, (256+-128)/2 + 20
    assert score > 0.85
    assert lbl == 2

    # 2. Verify ComicDetector._run_ctd_crop fuses UNet `seg` + DBNet `det[0, 0]` and verifies blk
    class _FakeCTD:
        def run(self, _out, _inp):
            fake_blk = np.zeros((1, 2, 7), dtype=np.float32)
            fake_blk[0, 0] = [512.0, 512.0, 200.0, 300.0, 0.90, 0.02, 0.98]  # backed by DBNet mask
            fake_blk[0, 1] = [100.0, 100.0, 80.0, 80.0, 0.88, 0.02, 0.98]   # empty background -> pruned
            fake_seg = np.zeros((1, 1, 1024, 1024), dtype=np.float32)
            fake_seg[0, 0, 400:460, 460:520] = 0.85
            fake_det = np.zeros((1, 2, 1024, 1024), dtype=np.float32)
            fake_det[0, 0, 520:600, 480:560] = 0.92  # line missed by UNet `seg`, caught by DBNet `det`
            return [fake_blk, fake_seg, fake_det]

    det = ComicDetector.__new__(ComicDetector)
    det._ctd = _FakeCTD()
    mask, blks = det.detect_ctd(np.full((512, 512, 3), 240, dtype=np.uint8))
    # DBNet region must be reflected in fused mask
    assert mask[280, 260] > 200
    assert mask[215, 245] > 200
    # Only the mask-backed blk survives, texture false positive on empty corner is pruned
    assert len(blks) == 1
    assert blks[0][0] == (206, 181, 306, 331)


def test_lama_manga_patch_scaling_and_padding():
    from koharu_vn_cli.pipeline.inpaint import HybridInpainter

    recorded_shapes: list[tuple[tuple[int, ...], float, float]] = []

    class _FakeLamaManga:
        def run(self, _out, feeds):
            inp_img = feeds["image"]
            inp_msk = feeds["mask"]
            recorded_shapes.append((inp_img.shape, float(inp_img.min()), float(inp_img.max())))
            assert inp_img.shape == (1, 3, 512, 512)
            assert inp_msk.shape == (1, 1, 512, 512)
            # mayocream/lama-manga-onnx returns [0.0, 1.0] RGB tensor
            out = inp_img.copy()
            # Simulate clean background fill (R=0.80, G=0.60, B=0.40) inside masked zone
            m = inp_msk[0, 0] > 0.5
            out[0, 0, m] = 0.80
            out[0, 1, m] = 0.60
            out[0, 2, m] = 0.40
            return [out]

    inpainter = HybridInpainter.__new__(HybridInpainter)
    inpainter._lama = _FakeLamaManga()

    # Exact 512x512 patch -> output correctly scaled *255 to uint8 RGB
    rgb_512 = np.full((512, 512, 3), (100, 100, 100), dtype=np.uint8)
    msk_512 = np.zeros((512, 512), dtype=np.uint8)
    msk_512[100:150, 100:150] = 255
    pred_512 = inpainter._run_lama_patch(rgb_512, msk_512)
    assert pred_512.shape == (512, 512, 3)
    assert pred_512.dtype == np.uint8
    assert tuple(int(v) for v in pred_512[125, 125]) == (204, 153, 102)

    # Smaller <512 patch -> padded to 512x512 (no aspect distortion) and cropped back
    rgb_small = np.full((240, 320, 3), (50, 60, 70), dtype=np.uint8)
    msk_small = np.zeros((240, 320), dtype=np.uint8)
    msk_small[40:80, 60:120] = 255
    pred_small = inpainter._run_lama_patch(rgb_small, msk_small)
    assert pred_small.shape == (240, 320, 3)
    assert tuple(int(v) for v in pred_small[60, 90]) == (204, 153, 102)
    assert tuple(int(v) for v in pred_small[10, 10]) == (50, 60, 70)

    # Larger >512 patch -> resized cleanly back to original dimensions
    rgb_big = np.full((700, 640, 3), (40, 80, 120), dtype=np.uint8)
    msk_big = np.zeros((700, 640), dtype=np.uint8)
    msk_big[200:400, 200:400] = 255
    pred_big = inpainter._run_lama_patch(rgb_big, msk_big)
    assert pred_big.shape == (700, 640, 3)
    assert len(recorded_shapes) == 3


def test_promote_complete_boxes_and_tile_clip_penalty():
    from koharu_vn_cli.pipeline.detector import (
        _containment_nms,
        _promote_complete_boxes,
        _tile_clip_score,
    )

    # Full vertical span (conf 0.42) encompasses partial lower fragment (conf 0.78),
    # and the top extension (y=30..65) contains real glyph pixels in text_mask.
    text_mask = np.zeros((150, 120), dtype=np.uint8)
    text_mask[35:62, 40:65] = 200
    text_mask[72:105, 40:65] = 210
    full_box = (36, 30, 70, 110)
    frag_box = (38, 68, 68, 108)
    promoted = _promote_complete_boxes([(frag_box, 0.78, 2), (full_box, 0.42, 2)], text_mask)
    kept = _containment_nms(promoted, iou_thresh=0.40, contain_thresh=0.52)
    assert len(kept) == 1
    assert kept[0][0] == full_box

    # Clipped tile-edge box vs unclipped full-page box -> unclipped box wins!
    unclipped = (137, 6, 226, 244)
    clipped = (146, 7, 226, 243)
    s_unclipped = _tile_clip_score(0.89, unclipped, (0, 0, 786, 245), 786, 245)
    s_clipped = _tile_clip_score(0.92, clipped, (146, 0, 786, 245), 786, 245)
    assert s_unclipped > s_clipped


def test_translucent_bubble_never_chops_top_glyphs_in_erasure_mask():
    from koharu_vn_cli.pipeline.detector import _flood_fill_bubble_contour
    from koharu_vn_cli.pipeline.inpaint import build_erasure_mask

    # Translucent gradient bubble (top darkened over hair, bottom bright over skin)
    rgb = np.full((200, 160, 3), 210, dtype=np.uint8)
    rgb[:75, :] = 95  # darker top region breaks naive fixed-range floodFill
    rgb[20:55, 65:95] = 20  # dark kanji glyphs at the top!
    text_mask = np.zeros((200, 160), dtype=np.uint8)
    text_mask[22:52, 68:92] = 220

    known_box = (45, 5, 115, 190)
    seed_text = (58, 15, 102, 180)
    refined, _, _ = _flood_fill_bubble_contour(rgb, text_mask, seed_text, known_box)
    assert refined is not None
    # Must preserve full top extent instead of chopping below y=75!
    assert refined[1] <= 15

    # Even if an underestimated bubble_bbox starts below text_bbox, build_erasure_mask must NOT clip top glyphs
    reg = DetectedRegion(
        text_bbox=seed_text,
        bubble_bbox=(45, 80, 115, 190),
        bubble_contour=None,
        is_bubble=True,
        is_dark_bubble=False,
        confidence=0.90,
    )
    mask = build_erasure_mask(rgb, text_mask, [reg])
    assert int(np.count_nonzero(mask[20:55, 65:95])) > 500


def test_compose_oval_enlarged_to_readable_size_on_cg():
    from koharu_vn_cli.models import BubbleStyle, TextRegionItem
    from koharu_vn_cli.pipeline.runner import expand_composed_boxes_if_tight
    from koharu_vn_cli.pipeline.typesetter import solve_layout

    tiny_oval = TextRegionItem(
        id=1,
        kind="bubble",
        bubble_style=BubbleStyle.WHITE_OVAL,
        text_bbox=(240, 560, 305, 625),  # 65x65px tiny box on 1280x885 CG
        bubble_bbox=(240, 560, 305, 625),
        translated_text="THA CHO EM ĐI MÀ...",
        draw_bubble=True,
    )
    enlarged = expand_composed_boxes_if_tight([tiny_oval], 1280, 885)
    box = enlarged[0].bubble_bbox
    assert box is not None
    assert (box[2] - box[0]) >= 150
    assert (box[3] - box[1]) >= 105
    layout = solve_layout(enlarged[0])
    assert layout.font_size >= 18
    assert not layout.overflowed


def test_pack_images_to_webp_sequential_in_place_and_copy(tmp_path: Path):
    from koharu_vn_cli.pipeline.runner import pack_images_to_webp

    in_dir = tmp_path / "messy_cg"
    in_dir.mkdir()
    Image.new("RGB", (64, 64), (200, 10, 10)).save(in_dir / "file_1745069383855_10_webp", format="WEBP")
    Image.new("RGB", (64, 64), (10, 200, 10)).save(in_dir / "file_1745069383855_2_webp", format="WEBP")
    Image.new("RGB", (64, 64), (10, 10, 200)).save(in_dir / "file_1745069383855_3_webp", format="WEBP")

    out_dir, files = pack_images_to_webp(in_dir)
    assert [f.name for f in files] == ["001.webp", "002.webp", "003.webp"]
    # Verify reading order preserved: 2 (green) -> 3 (blue) -> 10 (red)
    with Image.open(out_dir / "001.webp") as im1:
        assert im1.format == "WEBP"
        assert im1.getpixel((32, 32))[1] > 150
    with Image.open(out_dir / "002.webp") as im2:
        assert im2.getpixel((32, 32))[2] > 150
    with Image.open(out_dir / "003.webp") as im3:
        assert im3.getpixel((32, 32))[0] > 150

    # Test safe in-place conversion
    _, in_place_files = pack_images_to_webp(in_dir, in_place=True)
    assert sorted(p.name for p in in_dir.iterdir()) == ["001.webp", "002.webp", "003.webp"]
    assert len(in_place_files) == 3


def test_story_bible_confirm_hook_allows_character_edits(tmp_path: Path):
    from koharu_vn_cli.models import CharacterProfile, StoryBible
    from koharu_vn_cli.pipeline.runner import PipelineHooks, _load_or_build_story_bible

    class FakeLLM:
        model = "gemini-3.1-pro-preview"

        def build_story_bible(self, *args, **kwargs) -> StoryBible:
            return StoryBible(
                title="Cảnh CG",
                source_game="Honkai: Star Rail",
                characters=[CharacterProfile(name="Cô gái tóc xanh", pronoun_rules="anh - em")],
            )

    cache_dir = tmp_path / ".koharu_cache"
    cache_dir.mkdir()

    def confirm(b: StoryBible) -> StoryBible:
        return b.model_copy(
            update={
                "characters": [CharacterProfile(name="Robin", pronoun_rules="anh - em")]
            }
        )

    bible = _load_or_build_story_bible(
        FakeLLM(),  # type: ignore[arg-type]
        [],
        cache_dir,
        "cg_test",
        mode="compose",
        user_story=None,
        prompt=None,
        enable_story_pass=True,
        resume=False,
        hooks=PipelineHooks(on_confirm_story_bible=confirm),
    )
    assert bible is not None
    assert bible.source_game == "Honkai: Star Rail"
    assert bible.characters[0].name == "Robin"
    assert "Honkai: Star Rail" in bible.to_prompt_block()
    saved = StoryBible.model_validate_json((cache_dir / "story_bible.json").read_text(encoding="utf-8"))
    assert saved.characters[0].name == "Robin"


def test_extract_json_block_handles_think_tags_trailing_commas_and_truncation():
    from koharu_vn_cli.pipeline.gemini_client import _extract_json_block

    raw_think = (
        "<think>Checking {character: Robin} and braces [1,2]</think>\n"
        'Here is the JSON:\n{"title": "Test", "characters": [{"name": "Robin",},],}\n'
        "Note: hope this {helps}!"
    )
    parsed = _extract_json_block(raw_think)
    assert parsed["title"] == "Test"
    assert parsed["characters"][0]["name"] == "Robin"

    truncated = '{"title": "Trunc", "characters": [{"name": "Firefly", "pronoun_rules": "anh - em"}'
    repaired = _extract_json_block(truncated)
    assert repaired["title"] == "Trunc"
    assert repaired["characters"][0]["name"] == "Firefly"

    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError
    import pytest

    with pytest.raises(ContentBlockedError) as exc_info:
        _extract_json_block("Tôi không thể đáp ứng yêu cầu này do quy tắc an toàn.")
    assert "/tags clear" in str(exc_info.value)


def test_long_chapter_keyframes_multisheets_and_batched_compose(tmp_path: Path):
    from koharu_vn_cli.pipeline.gemini_client import VisionLLMClient, _build_story_contact_sheets

    img_paths: list[Path] = []
    sizes: list[tuple[int, int]] = []
    for i in range(25):
        p = tmp_path / f"p_{i + 1:03d}.jpg"
        Image.new("RGB", (600, 800), (80 + (i * 5) % 150, 120, 180)).save(p)
        img_paths.append(p)
        sizes.append((600, 800))

    # Pass 1 with 25 pages -> 3 high-res keyframes + ceil(25/12)=3 crisp sheets = 6 images total
    sheets = _build_story_contact_sheets(
        img_paths,
        max_pages=60,
        pages_per_sheet=12,
        include_keyframes=True,
    )
    assert len(sheets) == 6
    assert "Ảnh cận/rõ nét" in sheets[0][0]
    assert "Trang 1/25" in sheets[0][0]
    assert "Sheet tổng hợp 12 trang" in sheets[3][0]
    assert "Sheet tổng hợp 1 trang" in sheets[5][0]

    # Compose > 18 pages chunks into batches of 12 with running summaries
    client = VisionLLMClient(api_key="dummy")
    seen_prompts: list[str] = []

    def fake_call_json(system_prompt, user_prompt, images, temperature=0.65):
        seen_prompts.append(user_prompt)
        assert len(images) == 1
        pages_out = []
        # Detect batch range from sheet label
        label = images[0][0]
        import re as _re
        m = _re.search(r"Trang (\d+) đến Trang (\d+)", label)
        assert m is not None
        s_p, e_p = int(m.group(1)), int(m.group(2))
        for p_no in range(s_p, e_p + 1):
            pages_out.append(
                {
                    "page": p_no,
                    "page_summary": f"Tóm tắt trang {p_no}",
                    "items": [
                        {
                            "id": 1,
                            "kind": "bubble",
                            "bubble_style": "white_oval",
                            "box_2d": [50, 50, 260, 320],
                            "text": f"Thoại {p_no}",
                        }
                    ],
                }
            )
        return {"pages": pages_out}

    client._call_json = fake_call_json  # type: ignore[method-assign]
    res = client.compose_cg_sheet(img_paths, sizes)
    assert len(seen_prompts) == 3  # 12 + 12 + 1
    assert "Tóm tắt trang 12" in seen_prompts[1]
    assert set(res.keys()) == set(range(1, 26))


def test_translate_skips_blocked_page_and_resumes_cleanly(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    for i in range(1, 4):
        Image.new("RGB", (240, 320), (240, 240, 240)).save(in_dir / f"p{i}.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            return [DetectedRegion((40, 40, 180, 120), (30, 30, 190, 130), None, True, False, 0.9)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            return rgb.copy()

    class FakeCache:
        def get_detector(self, accurate=True):
            return FakeDet()

        def get_inpainter(self):
            return FakeInp()

    fail_p2 = {"flag": True}
    calls: list[int] = []

    def fake_translate_page(self, rgb, regions, **kwargs):
        p_idx = kwargs["page_index"]
        calls.append(p_idx)
        if p_idx == 2 and fail_p2["flag"]:
            raise ContentBlockedError("PROHIBITED_CONTENT")
        out = [r.model_copy(update={"source_text": "JP", "translated_text": f"Trang {p_idx}"}) for r in regions]
        return out, f"Tóm tắt {p_idx}"

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.VisionLLMClient.translate_page", fake_translate_page)

    out_dir, docs1 = run_translate_pipeline(
        in_dir,
        api_key="dummy",
        story_pass=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    # Page 1 and 3 succeeded despite page 2 block; page 2 has fallback image but no .koharu.json
    assert len(docs1) == 2
    assert (out_dir / "p1.png").is_file() and (out_dir / "p2.png").is_file() and (out_dir / "p3.png").is_file()
    assert (out_dir / ".koharu_cache" / "p1.koharu.json").is_file()
    assert not (out_dir / ".koharu_cache" / "p2.koharu.json").is_file()
    assert (out_dir / ".koharu_cache" / "p3.koharu.json").is_file()

    # Resume after switching provider -> skips 1 & 3, retries only page 2!
    fail_p2["flag"] = False
    calls.clear()
    _, docs2 = run_translate_pipeline(
        in_dir,
        output=out_dir,
        api_key="dummy",
        story_pass=False,
        resume=True,
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert calls == [2]
    assert len(docs2) == 3
    assert (out_dir / ".koharu_cache" / "p2.koharu.json").is_file()


def test_cluster_story_scenes_groups_similar_runs_and_isolates_story_bible(tmp_path: Path):
    from koharu_vn_cli.models import SceneArc, StoryBible
    from koharu_vn_cli.pipeline.scene_cluster import cluster_story_scenes

    # Oneshot 1 (160..162): bright left half, dark right half
    paths: list[Path] = []
    for num in (160, 161, 162):
        arr = np.zeros((96, 96, 3), dtype=np.uint8)
        arr[:, :48] = (230, 210, 200)
        arr[:, 48:] = (40, 30, 50)
        arr[20 + (num - 160) * 2 : 30 + (num - 160) * 2, 10:30] = (120, 80, 90)  # tiny expression change
        p = tmp_path / f"{num}.jpg"
        Image.fromarray(arr).save(p)
        paths.append(p)

    # Oneshot 2 (250..252): completely different pattern (top dark navy, bottom warm gold)
    for num in (250, 251, 252):
        arr = np.zeros((96, 96, 3), dtype=np.uint8)
        arr[:48, :] = (20, 35, 95)
        arr[48:, :] = (240, 200, 110)
        p = tmp_path / f"{num}.jpg"
        Image.fromarray(arr).save(p)
        paths.append(p)

    scenes = cluster_story_scenes(paths)
    assert len(scenes) == 2
    assert scenes[0].page_indices == [0, 1, 2]
    assert scenes[0].start_page == 1 and scenes[0].end_page == 3
    assert scenes[1].page_indices == [3, 4, 5]
    assert scenes[1].start_page == 4 and scenes[1].end_page == 6

    bible = StoryBible(
        title="Ayaya Anthology",
        scene_arcs=[
            SceneArc(scene_id=1, start_page=1, end_page=3, title="Cảnh 1", plot="Ngại ngùng phòng ngủ", pronoun_notes="anh - em"),
            SceneArc(scene_id=2, start_page=4, end_page=6, title="Cảnh 2", plot="Trêu chọc ngoài hành lang", pronoun_notes="cậu - tớ"),
        ],
    )
    blk_p2 = bible.to_prompt_block(page_index=2)
    assert "Tuyến 1" in blk_p2 and "Ngại ngùng phòng ngủ" in blk_p2
    assert "Trêu chọc ngoài hành lang" not in blk_p2

    blk_p5 = bible.to_prompt_block(page_index=5)
    assert "Tuyến 2" in blk_p5 and "Trêu chọc ngoài hành lang" in blk_p5
    assert "Ngại ngùng phòng ngủ" not in blk_p5


def test_no_vision_translate_uses_local_ocr_and_batches_by_scene(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "oneshots"
    in_dir.mkdir()
    for i, name in enumerate(("160.png", "161.png", "250.png")):
        arr = np.zeros((160, 160, 3), dtype=np.uint8)
        if i < 2:
            arr[:, :80] = (235, 225, 220)
            arr[:, 80:] = (45, 40, 55)
        else:
            arr[:80, :] = (25, 45, 110)
            arr[80:, :] = (245, 210, 120)
        Image.fromarray(arr).save(in_dir / name)

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            return [DetectedRegion((30, 30, 120, 90), (20, 20, 130, 100), None, True, False, 0.9)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            return rgb.copy()

    class FakeOcr:
        def recognize_regions(self, rgb, regions):
            return {r.id: "助けて…" for r in regions}

    class FakeCache:
        def get_detector(self, accurate=True):
            return FakeDet()

        def get_inpainter(self):
            return FakeInp()

        def get_ocr(self):
            return FakeOcr()

    recorded_calls: list[tuple[str, int, str]] = []

    def fake_call_json(self, system_prompt, user_prompt, images, temperature=0.3):
        img_count = len(images)
        recorded_calls.append((self.model, img_count, user_prompt))
        if img_count > 0:
            # Pass 1 Story Bible only!
            return {
                "title": "Collection",
                "source_game": "Genshin Impact",
                "full_plot_summary": "Hai tuyến riêng.",
                "characters": [],
                "scene_arcs": [
                    {"scene_id": 1, "start_page": 1, "end_page": 2, "title": "Oneshot 1", "plot": "Cảnh A", "pronoun_notes": "anh - em"},
                    {"scene_id": 2, "start_page": 3, "end_page": 3, "title": "Oneshot 2", "plot": "Cảnh B", "pronoun_notes": "cậu - tớ"},
                ],
                "page_beats": {},
            }
        # Pass 2 text-only (0 images!)
        if "[Trang 1]" in user_prompt and "[Trang 2]" in user_prompt:
            return {
                "pages": [
                    {"page": 1, "page_summary": "Tuyến 1 trang 1", "items": [{"id": 1, "source_text": "助けて…", "vietnamese": "Cứu em với…"}]},
                    {"page": 2, "page_summary": "Tuyến 1 trang 2", "items": [{"id": 1, "source_text": "助けて…", "vietnamese": "Đừng mà anh…"}]},
                ]
            }
        return {
            "detected_language": "ja",
            "page_summary": "Tuyến 2 trang 3",
            "items": [{"id": 1, "source_text": "助けて…", "vietnamese": "Tha cho tớ…"}],
        }

    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_call_json)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out",
        api_key="dummy",
        model="fast-text-worker",
        story_model="strong-vision-master",
        vision_translate=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )

    assert len(docs) == 3
    assert docs[0].regions[0].source_text == "助けて…"
    assert docs[0].regions[0].translated_text == "Cứu em với…"
    assert docs[1].regions[0].translated_text == "Đừng mà anh…"
    assert docs[2].regions[0].translated_text == "Tha cho tớ…"
    assert (out_dir / "160.png").is_file() and (out_dir / "250.png").is_file()

    # Verify Pass 1 used story_model WITH images, while Pass 2 sent ZERO images in 2 scene calls!
    assert len(recorded_calls) == 3
    assert recorded_calls[0][0] == "strong-vision-master" and recorded_calls[0][1] > 0
    assert recorded_calls[1][0] == "fast-text-worker" and recorded_calls[1][1] == 0
    assert recorded_calls[2][0] == "fast-text-worker" and recorded_calls[2][1] == 0
    assert "Tuyến 1 trang 1" not in recorded_calls[2][2]  # scene summaries isolated between scenes!


def test_tui_vision_slash_toggle(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.config import AppConfig
    from koharu_vn_cli.tui import KoharuTUI

    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda cfg: None)
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: AppConfig(vision_translate=True))
    app_tui = KoharuTUI()
    logs: list[str] = []
    app_tui._log_user = lambda _msg: None  # type: ignore[method-assign]
    app_tui._log_ai = lambda msg, style="": logs.append(msg)  # type: ignore[method-assign]

    app_tui._handle_submission("/vision off")
    assert app_tui.cfg.vision_translate is False
    assert any("TẮT Vision" in m for m in logs)

    app_tui._handle_submission("/vision on")
    assert app_tui.cfg.vision_translate is True
    assert any("BẬT Vision" in m for m in logs)


def test_ort_session_gpu_selection_and_runtime_cpu_fallback(monkeypatch, tmp_path: Path):
    import onnxruntime as ort
    from koharu_vn_cli.config import AppConfig
    from koharu_vn_cli.pipeline import detector, inpaint
    from koharu_vn_cli.tui import KoharuTUI

    monkeypatch.setattr(detector, "_ORT_PRELOADED", True)
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert detector._ort_providers(use_gpu=True) == [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert detector._ort_providers(use_gpu=False) == ["CPUExecutionProvider"]

    # If GPU provider raises during session creation, fall back cleanly to CPUExecutionProvider
    seen_providers: list[list[str]] = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            seen_providers.append(list(providers or []))
            if providers and providers[0] != "CPUExecutionProvider":
                raise RuntimeError("mock GPU init failure")
            self._p = list(providers or ["CPUExecutionProvider"])

        def get_providers(self):
            return self._p

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    sess = detector._create_ort_session(tmp_path / "dummy.onnx", use_gpu=True)
    assert seen_providers == [
        ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert detector.active_ort_provider_label(sess) == "CPU"

    # HybridInpainter runtime fallback when GPU session raises inside run()
    monkeypatch.setattr(inpaint, "hf_hub_download", lambda repo, fn: str(tmp_path / "lama.onnx"))

    class RuntimeFailGpuSession:
        def __init__(self, is_gpu: bool) -> None:
            self._is_gpu = is_gpu

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"] if self._is_gpu else ["CPUExecutionProvider"]

        def run(self, _out, _feed):
            if self._is_gpu:
                raise RuntimeError("DirectML MatMul_5 failed")
            return [np.zeros((1, 3, 512, 512), dtype=np.float32)]

    create_calls: list[bool] = []
    monkeypatch.setattr(
        inpaint,
        "_create_ort_session",
        lambda path, use_gpu=True: (create_calls.append(use_gpu), RuntimeFailGpuSession(use_gpu))[1],
    )
    inp = inpaint.HybridInpainter(use_gpu=True)
    patch_out = inp._run_lama_patch(np.zeros((64, 64, 3), dtype=np.uint8), np.full((64, 64), 255, dtype=np.uint8))
    assert patch_out.shape == (64, 64, 3)
    assert create_calls == [True, False]

    # /gpu slash command toggle
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda cfg: None)
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: AppConfig(use_gpu=True))
    app_tui = KoharuTUI()
    logs: list[str] = []
    app_tui._log_user = lambda _msg: None  # type: ignore[method-assign]
    app_tui._log_ai = lambda msg, style="": logs.append(msg)  # type: ignore[method-assign]
    app_tui._handle_submission("/gpu off")
    assert app_tui.cfg.use_gpu is False
    app_tui._handle_submission("/gpu on")
    assert app_tui.cfg.use_gpu is True


def test_compose_cli_and_tui_do_not_pass_translate_only_kwargs(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.config import AppConfig
    from koharu_vn_cli.tui import KoharuTUI

    img_dir = tmp_path / "cg"
    img_dir.mkdir()
    Image.new("RGB", (200, 200), (220, 220, 220)).save(img_dir / "01.png")

    seen_calls: list[dict[str, object]] = []

    def fake_compose_pipeline(input_path, **kwargs):
        seen_calls.append(kwargs)
        return tmp_path / "out", []

    monkeypatch.setattr("koharu_vn_cli.cli.run_compose_pipeline", fake_compose_pipeline)
    monkeypatch.setattr("koharu_vn_cli.tui.run_compose_pipeline", fake_compose_pipeline)
    monkeypatch.setattr("koharu_vn_cli.cli.load_config", lambda: AppConfig(api_key="dummy"))
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: AppConfig(api_key="dummy"))
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda cfg: None)

    runner = CliRunner()
    res = runner.invoke(app, ["compose", str(img_dir)])
    assert res.exit_code == 0, res.output

    app_tui = KoharuTUI()
    app_tui.call_from_thread = lambda fn, *a, **kw: None  # type: ignore[method-assign]
    app_tui._start_compose.__wrapped__(app_tui, img_dir)

    assert len(seen_calls) == 2
    for call_kw in seen_calls:
        assert "vision_translate" not in call_kw
        assert "use_gpu" not in call_kw


def test_vision_blocked_image_falls_back_to_local_ocr_translation(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    Image.new("RGB", (240, 320), (240, 240, 240)).save(in_dir / "p1.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            return [DetectedRegion((30, 30, 180, 120), (20, 20, 190, 130), None, True, False, 0.9)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            return rgb.copy()

    class FakeOcr:
        def recognize_regions(self, rgb, regions):
            return {r.id: "やめて…" for r in regions}

    class FakeCache:
        def get_detector(self, accurate=True):
            return FakeDet()

        def get_inpainter(self):
            return FakeInp()

        def get_ocr(self):
            return FakeOcr()

    def blocked_vision(self, *args, **kwargs):
        raise ContentBlockedError("PROHIBITED_CONTENT")

    ocr_text_calls: list[tuple[dict[int, str], int]] = []

    def fake_ocr_translate(self, regions, ocr_texts, **kwargs):
        ocr_text_calls.append((ocr_texts, kwargs["page_index"]))
        out = [r.model_copy(update={"source_text": ocr_texts[r.id], "translated_text": "Dừng lại đi…"}) for r in regions]
        return out, "Đã fallback OCR thành công"

    monkeypatch.setattr(VisionLLMClient, "translate_page", blocked_vision)
    monkeypatch.setattr(VisionLLMClient, "translate_page_from_ocr", fake_ocr_translate)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out",
        api_key="dummy",
        story_pass=False,
        vision_translate=True,
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(ocr_text_calls) == 1
    assert ocr_text_calls[0][0] == {1: "やめて…"}
    assert len(docs) == 1
    assert docs[0].regions[0].translated_text == "Dừng lại đi…"
    assert (out_dir / "p1.png").is_file()
    assert (out_dir / ".koharu_cache" / "p1.koharu.json").is_file()


def test_ocr_translate_retries_untranslated_and_uses_translation_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from koharu_vn_cli.pipeline.gemini_client import (
        VisionLLMClient,
        normalize_source_key,
        region_needs_translation,
    )

    client = VisionLLMClient(api_key="dummy", model="gemini-3.5-flash-lite")
    responses = [
        # First pass: alternate key + string ID on reg 1, kana echo on reg 2
        {
            "regions": [
                {"id": "ID 1", "translation": "Sướng quá đi mất!"},
                {"id": 2, "vietnamese": "あぁっ…！"},
            ]
        },
        # Automatic retry asks only for reg 2 and gets Vietnamese
        {"items": [{"id": 2, "vietnamese": "Á…!"}]},
        # Third call: simulate LLM echoing Japanese on retryable pass, rescued by translation_memory
        {"items": [{"id": 1, "vietnamese": "気持ちいい！"}]},
        {"items": [{"id": 1, "vietnamese": "気持ちいい！"}]},
    ]
    prompts: list[str] = []

    def fake_call_json(self, system_prompt, user_prompt, **kwargs):
        prompts.append(user_prompt)
        return responses.pop(0)

    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_call_json)

    regions = [
        TextRegionItem(
            id=1,
            kind="bubble",
            text_bbox=(0, 0, 50, 50),
            source_text="気持ちいい！",
            translated_text="",
        ),
        TextRegionItem(
            id=2,
            kind="floating",
            text_bbox=(60, 0, 100, 40),
            source_text="あぁっ…！",
            translated_text="",
        ),
    ]
    ocr_texts = {1: "気持ちいい！", 2: "あぁっ…！"}
    mem: dict[str, str] = {}
    out, _ = client.translate_page_from_ocr(regions, ocr_texts, translation_memory=mem)

    assert len(prompts) == 2
    assert out[0].translated_text == "Sướng quá đi mất!"
    assert out[1].translated_text == "Á…!"
    assert all(not region_needs_translation(r) for r in out)

    for r in out:
        mem[normalize_source_key(r.source_text)] = r.translated_text

    # Even if LLM keeps echoing Japanese, translation_memory fills the existing Vietnamese translation
    cached_regions = [
        TextRegionItem(
            id=1,
            kind="bubble",
            text_bbox=(0, 0, 50, 50),
            source_text="気持ちいい！",
            translated_text="",
        )
    ]
    out_cached, _ = client.translate_page_from_ocr(
        cached_regions, {1: "気持ちいい！"}, translation_memory=mem
    )
    assert out_cached[0].translated_text == "Sướng quá đi mất!"


def test_pipeline_reuses_memory_and_resumes_untranslated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError, VisionLLMClient
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    Image.new("RGB", (240, 320), (240, 240, 240)).save(in_dir / "p1.png")
    Image.new("RGB", (240, 320), (240, 240, 240)).save(in_dir / "p2.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            return [DetectedRegion((30, 30, 180, 120), (20, 20, 190, 130), None, True, False, 0.9)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            return rgb.copy()

    class FakeOcr:
        def recognize_regions(self, rgb, regions):
            return {r.id: "やめて…" for r in regions}

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDet()

        def get_inpainter(self, use_gpu=True):
            return FakeInp()

        def get_ocr(self, use_gpu=True):
            return FakeOcr()

    def blocked_vision(self, *args, **kwargs):
        raise ContentBlockedError("PROHIBITED_CONTENT")

    call_idx = 0

    def fake_ocr_translate(self, regions, ocr_texts, **kwargs):
        nonlocal call_idx
        call_idx += 1
        out = []
        for r in regions:
            src = ocr_texts.get(r.id, r.source_text)
            # First call succeeds; subsequent calls simulate small LLM leaving Japanese untranslated
            vn = "Dừng lại đi…" if call_idx == 1 else "やめて…"
            out.append(r.model_copy(update={"source_text": src, "translated_text": vn}))
        return out, ""

    monkeypatch.setattr(VisionLLMClient, "translate_page", blocked_vision)
    monkeypatch.setattr(VisionLLMClient, "translate_page_from_ocr", fake_ocr_translate)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out",
        api_key="dummy",
        story_pass=False,
        vision_translate=True,
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    # p1 translates via OCR LLM, p2 gets untranslated echo from LLM but is rescued by p1's translation_memory
    assert [d.regions[0].translated_text for d in docs] == ["Dừng lại đi…", "Dừng lại đi…"]

    # Corrupt p1 cached JSON with untranslated kana to verify --resume re-translates it from memory/OCR
    cache_p1 = out_dir / ".koharu_cache" / "p1.koharu.json"
    bad_doc = docs[0].model_copy(deep=True)
    bad_doc.regions[0].translated_text = "やめて…"
    cache_p1.write_text(bad_doc.model_dump_json(), encoding="utf-8")

    call_idx = 0
    _, resumed_docs = run_translate_pipeline(
        in_dir,
        output=out_dir,
        api_key="dummy",
        story_pass=False,
        vision_translate=True,
        resume=True,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert [d.regions[0].translated_text for d in resumed_docs] == ["Dừng lại đi…", "Dừng lại đi…"]


def test_story_model_prohibit_tries_story_fallback_once_then_skips_pass1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from koharu_vn_cli.models import StoryBible
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError, VisionLLMClient
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    Image.new("RGB", (200, 200), (240, 240, 240)).save(in_dir / "p1.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            return [DetectedRegion((20, 20, 120, 80), (15, 15, 125, 85), None, True, False, 0.9)], np.zeros(rgb.shape[:2], dtype=np.uint8)

    class FakeInp:
        def inpaint(self, rgb, mask):
            return rgb.copy()

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDet()

        def get_inpainter(self, use_gpu=True):
            return FakeInp()

    story_calls: list[str] = []

    def fake_story(self, images, **kwargs):
        story_calls.append(self.model)
        if self.model == "story-pro":
            raise ContentBlockedError("PROHIBITED_CONTENT")
        if self.model == "story-safe-ok":
            return StoryBible(title="Safe Story", full_plot_summary="Dịch ngọt")
        raise ContentBlockedError("PROHIBITED_CONTENT_AGAIN")

    seen_story_titles: list[str] = []

    def fake_translate(self, rgb, regions, **kwargs):
        sb = kwargs.get("story_bible")
        seen_story_titles.append(sb.title if sb else "")
        return [r.model_copy(update={"source_text": "愛", "translated_text": "Yêu"}) for r in regions], ""

    monkeypatch.setattr(VisionLLMClient, "build_story_bible", fake_story)
    monkeypatch.setattr(VisionLLMClient, "translate_page", fake_translate)

    # Case A: primary story model hits prohibit -> story_fallback succeeds in 1 turn
    _, docs_a = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out_a",
        api_key="dummy",
        model="fast-worker",
        story_model="story-pro",
        story_fallback_model="story-safe-ok",
        story_pass=True,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert story_calls == ["story-pro", "story-safe-ok"]
    assert seen_story_titles[-1] == "Safe Story"
    assert docs_a[0].regions[0].translated_text == "Yêu"

    # Case B: both primary and fallback hit prohibit -> 1 turn on fallback, then skip Pass 1 ("kệ") and continue Pass 2
    story_calls.clear()
    seen_story_titles.clear()
    _, docs_b = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out_b",
        api_key="dummy",
        model="fast-worker",
        story_model="story-pro",
        story_fallback_model="story-safe-blocked",
        story_pass=True,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert story_calls == ["story-pro", "story-safe-blocked"]
    assert seen_story_titles[-1] == ""
    assert docs_b[0].regions[0].translated_text == "Yêu"


def test_worker_model_quota_exhausted_switches_to_fallback_and_tui_cmds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    import pytest
    from koharu_vn_cli.config import AppConfig
    from koharu_vn_cli.pipeline.gemini_client import QuotaExhaustedError, VisionLLMClient
    from koharu_vn_cli.tui import KoharuTUI

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.time.sleep", lambda _s: None)

    models_called: list[str] = []

    def fake_rest(self, system_prompt, user_prompt, images, temperature, json_mode=True):
        models_called.append(self.model)
        if self.model == "fast-primary":
            req = httpx.Request("POST", "https://generativelanguage.googleapis.com")
            resp = httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}}, request=req)
            raise httpx.HTTPStatusError("quota exceeded", request=req, response=resp)
        return '{"ok": true}'

    monkeypatch.setattr(VisionLLMClient, "_call_gemini_rest", fake_rest)

    # Without fallback -> raises QuotaExhaustedError after retries
    no_fb_client = VisionLLMClient(api_key="k1", model="fast-primary")
    with pytest.raises(QuotaExhaustedError):
        no_fb_client._call_json("sys", "usr", [])

    # With fallback_model -> switches permanently on quota exhaustion and succeeds
    models_called.clear()
    fb_client = VisionLLMClient(api_key="k1", model="fast-primary", fallback_model="fast-backup")
    out = fb_client._call_json("sys", "usr", [])
    assert out == {"ok": True}
    assert fb_client.model == "fast-backup"
    assert models_called[-1] == "fast-backup"

    # Next call goes straight to fallback model without hitting exhausted primary again
    models_called.clear()
    out2 = fb_client._call_json("sys", "usr", [])
    assert out2 == {"ok": True}
    assert models_called == ["fast-backup"]

    # TUI slash commands `/fallback` and `/story-fallback`
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda cfg: None)
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: AppConfig())
    app_tui = KoharuTUI()
    app_tui._log_user = lambda _msg: None  # type: ignore[method-assign]
    app_tui._log_ai = lambda _msg, style="": None  # type: ignore[method-assign]
    app_tui._handle_submission("/fallback gemini-2.5-flash")
    app_tui._handle_submission("/story-fallback gemini-2.5-pro")
    assert app_tui.cfg.fallback_model == "gemini-2.5-flash"
    assert app_tui.cfg.story_fallback_model == "gemini-2.5-pro"
    app_tui._handle_submission("/fallback off")
    assert app_tui.cfg.fallback_model == ""


def test_skip_sfx_prevents_translation_and_inpainting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from koharu_vn_cli.config import AppConfig
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import (
        VisionLLMClient,
        looks_like_floating_sfx,
        region_needs_translation,
    )
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline
    from koharu_vn_cli.tui import KoharuTUI

    # Floating spoken moans must NOT be mistaken for SFX; pure SFX glyphs should
    assert looks_like_floating_sfx("あっ…ぐっ！！") is False
    assert looks_like_floating_sfx("ドキッ") is True
    assert looks_like_floating_sfx("ズチュッ") is True

    in_dir = tmp_path / "sfx_manga"
    in_dir.mkdir()
    img = np.full((320, 300, 3), 245, dtype=np.uint8)
    # Paint distinctive magenta SFX artwork inside (180, 180, 260, 250)
    img[195:235, 195:245] = (230, 30, 150)
    Image.fromarray(img).save(in_dir / "p1.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            mask[40:80, 40:110] = 255  # bubble text glyphs
            mask[195:235, 195:245] = 255  # SFX glyphs
            regs = [
                DetectedRegion((35, 35, 115, 95), (25, 25, 125, 110), None, True, False, 0.95),
                DetectedRegion((180, 180, 260, 250), None, None, False, False, 0.92),
            ]
            return regs, mask

    seen_masks: list[np.ndarray] = []

    class FakeInp:
        def inpaint(self, rgb, mask):
            seen_masks.append(mask.copy())
            out = rgb.copy()
            out[mask > 0] = (255, 255, 255)
            return out

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDet()

        def get_inpainter(self, use_gpu=True):
            return FakeInp()

        def get_ocr(self, use_gpu=True):
            class FakeOcr:
                def recognize_regions(self, rgb, regions):
                    return {1: "気持ちいい", 2: "ドキッ"}

            return FakeOcr()

    def fake_vision_call(self, system_prompt, user_prompt, **kwargs):
        return {
            "detected_language": "ja",
            "page_summary": "Cảnh thử nghiệm",
            "items": [
                {"id": 1, "source_text": "気持ちいい", "vietnamese": "Sướng quá!", "is_sfx": False},
                {"id": 2, "source_text": "ドキッ", "vietnamese": "THỊCH", "is_sfx": True},
            ],
        }

    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_vision_call)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out_vision",
        api_key="dummy",
        story_pass=False,
        vision_translate=True,
        skip_sfx=True,
        output_format="png",
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(seen_masks) == 1
    # Region 1 (bubble) WAS masked for inpainting; Region 2 (SFX) had ZERO inpaint pixels
    assert np.count_nonzero(seen_masks[0][35:95, 35:115]) > 0
    assert np.count_nonzero(seen_masks[0][180:250, 180:260]) == 0
    assert docs[0].regions[0].translated_text == "Sướng quá!"
    assert docs[0].regions[1].is_sfx is True
    assert docs[0].regions[1].translated_text == ""
    assert not region_needs_translation(docs[0].regions[1])

    # Rendered output keeps the exact original SFX artwork pixels untouched!
    rendered = np.array(Image.open(out_dir / "p1.png").convert("RGB"))
    assert np.array_equal(rendered[195:235, 195:245], img[195:235, 195:245])

    # No-Vision OCR mode also skips SFX & preserves SFX artwork
    seen_masks.clear()

    def fake_ocr_call(self, system_prompt, user_prompt, **kwargs):
        return {
            "detected_language": "ja",
            "page_summary": "Cảnh OCR",
            "items": [{"id": 1, "source_text": "気持ちいい", "vietnamese": "Sướng quá!", "is_sfx": False}],
        }

    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_ocr_call)
    out_novis, docs_novis = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out_novision",
        api_key="dummy",
        story_pass=False,
        vision_translate=False,
        skip_sfx=True,
        output_format="png",
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(seen_masks) == 1
    assert np.count_nonzero(seen_masks[0][180:250, 180:260]) == 0
    assert docs_novis[0].regions[1].is_sfx is True
    assert docs_novis[0].regions[1].translated_text == ""
    rendered_nv = np.array(Image.open(out_novis / "p1.png").convert("RGB"))
    assert np.array_equal(rendered_nv[195:235, 195:245], img[195:235, 195:245])

    # TUI `/sfx` command
    monkeypatch.setattr("koharu_vn_cli.tui.save_config", lambda cfg: None)
    monkeypatch.setattr("koharu_vn_cli.tui.load_config", lambda: AppConfig(skip_sfx=True))
    app_tui = KoharuTUI()
    app_tui._log_user = lambda _msg: None  # type: ignore[method-assign]
    app_tui._log_ai = lambda _msg, style="": None  # type: ignore[method-assign]
    app_tui._handle_submission("/sfx translate")
    assert app_tui.cfg.skip_sfx is False
    app_tui._handle_submission("/sfx skip")
    assert app_tui.cfg.skip_sfx is True


def test_story_timeout_180_and_ocr_fallback_inpaints_and_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from koharu_vn_cli.models import StoryBible
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError, VisionLLMClient
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    orig = np.full((240, 240, 3), 255, dtype=np.uint8)
    # Paint dark Japanese glyph strokes inside the speech bubble (40:110, 40:140)
    orig[55:95, 55:125] = (15, 15, 15)
    Image.fromarray(orig).save(in_dir / "p1.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            mask[55:95, 55:125] = 255
            return [DetectedRegion((40, 40, 140, 110), (30, 30, 150, 120), None, True, False, 0.95)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            out = rgb.copy()
            out[mask > 127] = (255, 255, 255)
            return out

    class FakeOcr:
        def recognize_regions(self, rgb, regions):
            return {r.id: "やめて…" for r in regions}

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDet()

        def get_inpainter(self, use_gpu=True):
            return FakeInp()

        def get_ocr(self, use_gpu=True):
            return FakeOcr()

    seen_timeouts: dict[str, float] = {}

    def fake_story(self, images, **kwargs):
        seen_timeouts[self.model] = self.timeout
        return StoryBible(title="Test", full_plot_summary="OK")

    def blocked_vision(self, *args, **kwargs):
        raise ContentBlockedError("PROHIBITED_CONTENT")

    def fake_ocr_call(self, system_prompt, user_prompt, **kwargs):
        if self.model != "story-fb-ok":
            raise ContentBlockedError("TEXT_BLOCKED")
        # Even if OCR LLM mis-flags a spoken hiragana bubble as is_sfx=True, safeguard keeps it
        return {
            "page_summary": "Cảnh fallback",
            "items": [{"id": 1, "source_text": "やめて…", "vietnamese": "Dừng lại đi…", "is_sfx": True}],
        }

    monkeypatch.setattr(VisionLLMClient, "build_story_bible", fake_story)
    monkeypatch.setattr(VisionLLMClient, "translate_page", blocked_vision)
    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_ocr_call)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out",
        api_key="dummy",
        model="fast-worker",
        story_model="main-story",
        story_fallback_model="story-fb-ok",
        story_pass=True,
        vision_translate=True,
        output_format="png",
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )

    assert seen_timeouts["main-story"] == 180.0
    assert docs[0].regions[0].translated_text == "Dừng lại đi…"
    assert docs[0].regions[0].is_sfx is False

    clean_arr = np.array(Image.open(docs[0].cleaned_image_path).convert("RGB"))
    out_arr = np.array(Image.open(out_dir / "p1.png").convert("RGB"))
    # Verify actual pixel inpainting erased original black strokes AND typesetter rendered Vietnamese pixels
    assert not np.array_equal(clean_arr[55:95, 55:125], orig[55:95, 55:125])
    assert not np.array_equal(out_arr[40:110, 40:140], clean_arr[40:110, 40:140])
    assert int(np.min(out_arr[40:110, 40:140])) < 100


def test_clean_cache_removes_cache_only_when_all_pages_complete(tmp_path: Path, monkeypatch):
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError
    from koharu_vn_cli.pipeline.runner import run_compose_pipeline, run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    for i in (1, 2):
        Image.new("RGB", (240, 320), (240, 240, 240)).save(in_dir / f"p{i}.png")

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            return [DetectedRegion((40, 40, 180, 120), (30, 30, 190, 130), None, True, False, 0.9)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            return rgb.copy()

    class FakeCache:
        def get_detector(self, accurate=True):
            return FakeDet()

        def get_inpainter(self):
            return FakeInp()

    block_p2 = {"flag": True}

    def fake_translate_page(self, rgb, regions, **kwargs):
        p_idx = kwargs["page_index"]
        if p_idx == 2 and block_p2["flag"]:
            raise ContentBlockedError("PROHIBITED_CONTENT")
        out = [r.model_copy(update={"source_text": "JP", "translated_text": f"Trang {p_idx}"}) for r in regions]
        return out, f"Tóm tắt {p_idx}"

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.VisionLLMClient.translate_page", fake_translate_page)

    # 1. Incomplete run (page 2 blocked) must KEEP .koharu_cache even when clean_cache=True
    out_dir, docs_partial = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out_translate",
        api_key="dummy",
        story_pass=False,
        clean_cache=True,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(docs_partial) == 1
    assert (out_dir / ".koharu_cache" / "p1.koharu.json").is_file()

    # 2. Finishing all pages with clean_cache=False keeps .koharu_cache
    block_p2["flag"] = False
    _, docs_kept = run_translate_pipeline(
        in_dir,
        output=out_dir,
        api_key="dummy",
        story_pass=False,
        resume=True,
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(docs_kept) == 2
    assert (out_dir / ".koharu_cache" / "p2.koharu.json").is_file()

    # 3. Completing 100% with default clean_cache=True deletes .koharu_cache cleanly
    _, docs_done = run_translate_pipeline(
        in_dir,
        output=out_dir,
        api_key="dummy",
        story_pass=False,
        resume=True,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(docs_done) == 2
    assert (out_dir / "p1.png").is_file()
    assert (out_dir / "p2.png").is_file()
    assert not (out_dir / ".koharu_cache").exists()

    # 4. Compose pipeline also cleans .koharu_cache on 100% completion when clean_cache=True
    def fake_compose_cg_sheet(self, image_paths, image_sizes, **_kwargs):
        return {
            i: (
                [
                    TextRegionItem(
                        id=1,
                        kind="bubble",
                        bubble_style=BubbleStyle.WHITE_OVAL,
                        text_bbox=(20, 20, 160, 90),
                        bubble_bbox=(20, 20, 160, 90),
                        translated_text=f"Thoại {i}",
                        draw_bubble=True,
                    )
                ],
                f"Cảnh {i}",
            )
            for i in range(1, len(image_paths) + 1)
        }

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.VisionLLMClient.compose_cg_sheet", fake_compose_cg_sheet)
    cg_out_kept, cg_docs_kept = run_compose_pipeline(
        in_dir,
        output=tmp_path / "out_cg_kept",
        api_key="dummy",
        story_pass=False,
        clean_cache=False,
    )
    assert len(cg_docs_kept) == 2
    assert (cg_out_kept / ".koharu_cache").is_dir()

    cg_out_clean, cg_docs_clean = run_compose_pipeline(
        in_dir,
        output=tmp_path / "out_cg_clean",
        api_key="dummy",
        story_pass=False,
    )
    assert len(cg_docs_clean) == 2
    assert (cg_out_clean / "p1.png").is_file()
    assert not (cg_out_clean / ".koharu_cache").exists()


def test_parallel_3stage_sliding_workers_and_pipelined_detect_inpaint(tmp_path: Path, monkeypatch):
    import threading
    import time
    import pytest
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    for i in range(1, 5):
        Image.new("RGB", (240, 320), (240, 240, 240)).save(in_dir / f"p{i}.png")

    gpu_mu = threading.Lock()
    active_gpu = 0
    gpu_overlap = False
    detect_count = 0

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            nonlocal active_gpu, gpu_overlap, detect_count
            with gpu_mu:
                active_gpu += 1
                if active_gpu > 1:
                    gpu_overlap = True
            try:
                detect_count += 1
                time.sleep(0.01)
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                return [DetectedRegion((40, 40, 180, 120), (30, 30, 190, 130), None, True, False, 0.9)], mask
            finally:
                with gpu_mu:
                    active_gpu -= 1

    class FakeInp:
        def inpaint(self, rgb, mask):
            nonlocal active_gpu, gpu_overlap
            with gpu_mu:
                active_gpu += 1
                if active_gpu > 1:
                    gpu_overlap = True
            try:
                time.sleep(0.01)
                return rgb.copy()
            finally:
                with gpu_mu:
                    active_gpu -= 1

    class FakeCache:
        def get_detector(self, accurate=True):
            return FakeDet()

        def get_inpainter(self):
            return FakeInp()

    llm_mu = threading.Lock()
    active_llm = 0
    max_active_llm = 0
    used_keys: list[str] = []
    started_order: list[int] = []
    detect_counts_at_llm_start: list[int] = []

    def fake_translate_page(self, rgb, regions, **kwargs):
        nonlocal active_llm, max_active_llm
        p_idx = kwargs["page_index"]
        with llm_mu:
            active_llm += 1
            max_active_llm = max(max_active_llm, active_llm)
            used_keys.append(self.api_key)
            started_order.append(p_idx)
            detect_counts_at_llm_start.append(detect_count)
        try:
            # Make page 1 slower than page 2 so worker 2 steals page 3 while worker 1 still finishes page 1
            time.sleep(0.14 if p_idx == 1 else 0.05)
            out = [r.model_copy(update={"source_text": "JP", "translated_text": f"Trang {p_idx}"}) for r in regions]
            return out, f"Tóm tắt {p_idx}"
        finally:
            with llm_mu:
                active_llm -= 1

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.VisionLLMClient.translate_page", fake_translate_page)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out_parallel",
        api_keys=["key-A", "key-B"],
        story_pass=False,
        max_workers=0,  # auto -> 2 workers matching 2 API keys
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )

    assert not gpu_overlap
    assert max(detect_counts_at_llm_start) == 4
    assert max_active_llm == 2
    assert set(used_keys) == {"key-A", "key-B"}
    assert started_order[0] in (1, 2) and set(started_order[:2]) == {1, 2}
    # All 4 docs returned sorted strictly 1..4 even when page 2 completed before page 1
    assert [d.regions[0].translated_text for d in docs] == ["Trang 1", "Trang 2", "Trang 3", "Trang 4"]
    assert all((out_dir / f"p{i}.png").is_file() for i in range(1, 5))

    # Fatal error when skip_blocked=False must unwind all 3 stages cleanly without deadlock
    def fail_on_p3(self, rgb, regions, **kwargs):
        if kwargs["page_index"] == 3:
            raise ContentBlockedError("FATAL_BLOCK")
        time.sleep(0.02)
        out = [r.model_copy(update={"source_text": "JP", "translated_text": "OK"}) for r in regions]
        return out, ""

    monkeypatch.setattr("koharu_vn_cli.pipeline.gemini_client.VisionLLMClient.translate_page", fail_on_p3)
    with pytest.raises(ContentBlockedError, match="FATAL_BLOCK"):
        run_translate_pipeline(
            in_dir,
            output=tmp_path / "out_err",
            api_keys=["key-A", "key-B"],
            story_pass=False,
            skip_blocked=False,
            model_cache=FakeCache(),  # type: ignore[arg-type]
        )


def test_zip_drop_cli_nested_sorting_macos_skip_and_notifications(tmp_path: Path, monkeypatch):
    import io
    import sys
    import zipfile
    import pytest
    from koharu_vn_cli import cli
    from koharu_vn_cli.pipeline.runner import collect_inputs

    # 1. Nested multi-chapter .zip with __MACOSX metadata & duplicate stems (001.png in chap1 and chap2)
    zip_path = tmp_path / "Series Pack.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("__MACOSX/chap1/._001.png", b"\x00\x05\x16\x07AppleDouble")
        buf1 = io.BytesIO()
        Image.new("RGB", (80, 100), (200, 10, 10)).save(buf1, format="PNG")
        buf2 = io.BytesIO()
        Image.new("RGB", (80, 100), (10, 200, 10)).save(buf2, format="PNG")
        buf10 = io.BytesIO()
        Image.new("RGB", (80, 100), (10, 10, 200)).save(buf10, format="PNG")
        zf.writestr("chap10/001.png", buf10.getvalue())
        zf.writestr("chap2/001.png", buf2.getvalue())
        zf.writestr("chap1/001.png", buf1.getvalue())

    ext_dir = tmp_path / "ext"
    imgs, name = collect_inputs(zip_path, ext_dir)
    assert name == "Series Pack"
    assert [p.parent.name for p in imgs] == ["chap1", "chap2", "chap10"]

    # Corrupt zip raises user-friendly ValueError
    bad_zip = tmp_path / "broken.zip"
    bad_zip.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="File nén không hợp lệ"):
        collect_inputs(bad_zip, tmp_path / "ext_bad")

    # 2. Dropping .zip straight onto `koharu-vn <file.zip>` routes `main()` to `translate` & fires notifications
    routed_args: list[Path] = []
    cli_notifs: list[tuple[str, str, bool, bool]] = []

    def fake_translate(input_path: Path, **_kw):
        routed_args.append(input_path)
        out = tmp_path / "Series Pack_vn"
        out.mkdir(exist_ok=True)
        return out, []

    monkeypatch.setattr(cli, "run_translate_pipeline", fake_translate)
    monkeypatch.setattr(
        cli,
        "send_desktop_notification",
        lambda title, msg, is_error=False, enabled=True: (cli_notifs.append((title, msg, is_error, enabled)), True)[1],
    )
    monkeypatch.setattr(cli, "load_config", lambda: __import__("koharu_vn_cli.config", fromlist=["AppConfig"]).AppConfig(api_key="dummy"))
    monkeypatch.setattr(sys, "argv", ["koharu-vn", f'"{zip_path}"'])
    with pytest.raises(SystemExit) as exit_exc:
        cli.main()
    assert exit_exc.value.code in (0, None)
    assert routed_args == [zip_path]
    assert len(cli_notifs) == 1 and cli_notifs[0][2] is False

    # Error path in CLI also sends error notification
    cli_notifs.clear()
    monkeypatch.setattr(cli, "run_translate_pipeline", lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("Zip lỗi")))
    res_err = CliRunner().invoke(cli.app, ["translate", str(zip_path), "--api-key", "dummy"])
    assert res_err.exit_code == 1
    assert len(cli_notifs) == 1 and cli_notifs[0][2] is True and "Zip lỗi" in cli_notifs[0][1]


def test_write_reader_metadata_sidecar(tmp_path: Path) -> None:
    """Verify koharu.meta.json export from StoryBible/tags and skipping empty metadata."""
    from koharu_vn_cli.models import CharacterProfile, StoryBible
    from koharu_vn_cli.pipeline.runner import _write_reader_metadata

    empty_dir = tmp_path / "empty_comic"
    empty_dir.mkdir()
    _write_reader_metadata(empty_dir, None, None)
    assert not (empty_dir / "koharu.meta.json").exists()

    _write_reader_metadata(empty_dir, StoryBible(title="No Meta"), "")
    assert not (empty_dir / "koharu.meta.json").exists()

    rich_dir = tmp_path / "rich_comic"
    rich_dir.mkdir()
    bible = StoryBible(
        title="Fontaine Tea",
        source_game="Genshin Impact",
        full_plot_summary="Buổi trà chiều ngọt ngào tại Fontaine.",
        characters=[
            CharacterProfile(name="Furina", pronoun_rules="em - anh"),
            CharacterProfile(name="Neuvillette", pronoun_rules="tôi - cô"),
        ],
    )
    _write_reader_metadata(rich_dir, bible, "romance, slice of life, Furina")
    meta_path = rich_dir / "koharu.meta.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    assert data["parody"] == "Genshin Impact"
    assert "Furina" in data["characters"]
    assert "Neuvillette" in data["characters"]
    assert "romance" in data["tags"]
    assert "slice of life" in data["tags"]
    assert data["summary"] == "Buổi trà chiều ngọt ngào tại Fontaine."


def test_deepseek_api_routing_payload_and_ocr_auto_mode(monkeypatch, tmp_path: Path) -> None:
    """Verify DeepSeek auto-routing, key isolation, string message payload, and compose guard."""
    import httpx
    import pytest

    from koharu_vn_cli.config import AppConfig
    from koharu_vn_cli.pipeline import gemini_client as gc
    from koharu_vn_cli.pipeline.runner import run_compose_pipeline

    monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-gemini")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-deepseek-1")

    cfg = AppConfig(
        api_key="AIza-saved-gemini",
        deepseek_api_key="sk-saved-deepseek",
        base_url="https://my-proxy.example/v1",
        model="deepseek-chat",
    )
    assert cfg.effective_api_keys() == ["sk-saved-deepseek", "sk-fake-deepseek-1"]
    assert cfg.effective_api_keys(model="gemini-3.8-flash") == ["AIza-saved-gemini", "AIza-fake-gemini"]
    assert cfg.masked_gemini_key().startswith("AIza")
    assert cfg.masked_deepseek_key().startswith("sk-s")
    assert cfg.api_key == "AIza-saved-gemini"
    assert cfg.base_url == "https://my-proxy.example/v1"

    client = VisionLLMClient(model="deepseek-chat")
    assert client.base_url == gc.DEEPSEEK_BASE_URL
    assert client.api_keys == ["sk-fake-deepseek-1"]
    assert gc.is_text_only_model("deepseek-chat", client.base_url) is True

    captured_payloads: list[dict] = []

    class FakeResp:
        status_code = 200
        text = '{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'

        def json(self) -> dict:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, json=None):
            captured_payloads.append({"url": url, "headers": headers, "json": json})
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeHttpClient)

    out = client._call_openai_compatible("sys", "dịch dòng này", [], temperature=0.3)
    assert out == '{"ok": true}'
    assert captured_payloads[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert isinstance(captured_payloads[0]["json"]["messages"][1]["content"], str)
    assert "temperature" in captured_payloads[0]["json"]

    r1 = VisionLLMClient(model="deepseek-reasoner")
    r1._call_openai_compatible("sys", "dịch dòng này", [], temperature=0.3)
    assert "temperature" not in captured_payloads[1]["json"]

    with pytest.raises(ValueError, match="DeepSeek API"):
        client._call_openai_compatible("sys", "xem ảnh", [("p1", "AAA")], temperature=0.3)

    img_dir = tmp_path / "cg"
    img_dir.mkdir()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(img_dir / "01.png")
    with pytest.raises(ValueError, match="Vision"):
        run_compose_pipeline(img_dir, model="deepseek-chat", api_key="sk-fake-deepseek-1")


def test_vision_prohibit_ocr_fallback_chain_and_untranslated_page_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify Vision PROHIBIT + empty Vision items both recover via OCR fallback, inpaint, and render text."""
    from koharu_vn_cli.models import StoryBible
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import ContentBlockedError, VisionLLMClient
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline

    in_dir = tmp_path / "manga"
    in_dir.mkdir()
    for name in ("p1.png", "p2.png"):
        orig = np.full((240, 240, 3), 255, dtype=np.uint8)
        orig[55:95, 55:125] = (15, 15, 15)  # dark ink inside speech bubble
        Image.fromarray(orig).save(in_dir / name)

    inpaint_calls: list[int] = []

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            mask[55:95, 55:125] = 255
            return [DetectedRegion((40, 40, 160, 120), (30, 30, 170, 130), None, True, False, 0.95)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            inpaint_calls.append(int(np.count_nonzero(mask)))
            out = rgb.copy()
            out[mask > 127] = (255, 255, 255)
            return out

    class FakeOcr:
        def recognize_regions(self, rgb, regions):
            return {r.id: "イっちゃう…" for r in regions}

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDet()

        def get_inpainter(self, use_gpu=True):
            return FakeInp()

        def get_ocr(self, use_gpu=True):
            return FakeOcr()

    def fake_story(self, images, **kwargs):
        return StoryBible(title="R18 Test", full_plot_summary="Cốt truyện R18")

    def fake_translate_page(self, rgb, regions, **kwargs):
        if kwargs.get("page_index") == 1:
            raise ContentBlockedError("PROHIBITED_CONTENT")
        # Page 2 simulates fast Vision returning empty items (source_text="", translated_text="")
        return [r.model_copy() for r in regions], ""

    def fake_call_json(self, system_prompt, user_prompt, images, temperature=0.3):
        # Simulate Gemini blocking OCR prompt when StoryBible is attached, succeeding on clean retry or fallback
        if "HỒ SƠ CỐT TRUYỆN" in user_prompt and self.model == "gemini-3.8-flash":
            raise ContentBlockedError("PROHIBITED_CONTENT_TEXT")
        # Also simulate LLM echoing placeholder source_text="..." and is_sfx=True alongside valid Vietnamese text
        return {
            "page_summary": "Đã dịch OCR",
            "items": [
                {
                    "id": 1,
                    "source_text": "...",
                    "vietnamese": "Sướng quá không chịu nổi!",
                    "is_sfx": True,
                }
            ],
        }

    monkeypatch.setattr(VisionLLMClient, "build_story_bible", fake_story)
    monkeypatch.setattr(VisionLLMClient, "translate_page", fake_translate_page)
    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_call_json)

    out_dir, docs = run_translate_pipeline(
        in_dir,
        output=tmp_path / "out",
        api_key="dummy",
        model="gemini-3.8-flash",
        story_pass=True,
        vision_translate=True,
        output_format="png",
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )

    assert len(docs) == 2
    assert len(inpaint_calls) == 2 and all(c > 0 for c in inpaint_calls)
    for idx, fname in enumerate(("p1.png", "p2.png")):
        assert docs[idx].regions[0].is_sfx is False
        assert docs[idx].regions[0].translated_text == "Sướng quá không chịu nổi!"
        clean_arr = np.array(Image.open(out_dir / ".koharu_cache" / f"p{idx + 1}.clean.png").convert("RGB"))
        out_arr = np.array(Image.open(out_dir / fname).convert("RGB"))
        # Clean image was inpainted white, and final output image has dark rendered Vietnamese glyphs
        assert int(clean_arr[55:95, 55:125].mean()) == 255
        assert not np.array_equal(out_arr, clean_arr)


def test_retranslate_drive_skips_vietnamese_and_fixes_untranslated(tmp_path: Path, monkeypatch) -> None:
    """Verify /retranslate skips already-translated Vietnamese pages/bubbles and fixes untranslated ones."""
    from koharu_vn_cli import cli
    from koharu_vn_cli.pipeline.detector import DetectedRegion
    from koharu_vn_cli.pipeline.gemini_client import VisionLLMClient
    from koharu_vn_cli.pipeline.runner import classify_retranslate_regions, run_retranslate_pipeline
    from koharu_vn_cli.tui import parse_natural_intent

    # Unit-check classifier directly: Vietnamese Latin skipped, pure floating SFX skipped, CJK caught
    regs_cls = [
        TextRegionItem(id=1, kind="bubble", text_bbox=(10, 10, 80, 50)),
        TextRegionItem(id=2, kind="bubble", text_bbox=(10, 60, 80, 100)),
        TextRegionItem(id=3, kind="floating", text_bbox=(10, 110, 80, 150)),
        TextRegionItem(id=4, kind="bubble", text_bbox=(10, 160, 80, 200)),
    ]
    cls_res = classify_retranslate_regions(
        regs_cls,
        {
            1: "suNe.cuA",       # simulated MangaOCR output on Vietnamese bubble
            2: "いや......",      # short dark-bg Vietnamese interjection misread as 2 kana on VI page
            3: "ドキッ",          # pure floating SFX
            4: "本当に出すの?",    # genuinely missed Japanese bubble (cjk >= 3, latin == 0)
        },
        skip_sfx=True,
    )
    assert cls_res == {4}

    # Natural Vietnamese intent detection
    act, p_str, opts = parse_natural_intent('tái dịch "C:/tmp/drive_dl.zip" xưng hô anh - em')
    assert act == "retranslate" and p_str == "C:/tmp/drive_dl.zip" and "anh - em" in opts["prompt"]

    drive_dir = tmp_path / "from_drive"
    drive_dir.mkdir()
    # Page 1: already translated Vietnamese (with distinctive marker corner pixel)
    img1 = np.full((240, 240, 3), 245, dtype=np.uint8)
    img1[0:10, 0:10] = (12, 34, 56)
    Image.fromarray(img1).save(drive_dir / "001.png")

    # Page 2: untranslated Japanese page blocked previously
    img2 = np.full((240, 240, 3), 255, dtype=np.uint8)
    img2[50:90, 50:140] = (15, 15, 15)
    Image.fromarray(img2).save(drive_dir / "002.png")

    # Page 3: partial page -> bubble 1 already Vietnamese (top), bubble 2 still Japanese (bottom)
    img3 = np.full((240, 240, 3), 255, dtype=np.uint8)
    img3[20:55, 40:140] = (30, 120, 200)   # already-rendered bubble 1 marker
    img3[145:185, 50:140] = (15, 15, 15)   # untranslated Japanese ink in bubble 2
    Image.fromarray(img3).save(drive_dir / "003.png")

    inpainted_masks: list[tuple[int, int]] = []
    llm_calls: list[tuple[str, int]] = []

    class FakeDet:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            # Detect by marker pixel in img3[25, 45]
            if tuple(rgb[25, 45]) == (30, 120, 200):
                mask[20:55, 40:140] = 255
                mask[145:185, 50:140] = 255
                return [
                    DetectedRegion((40, 20, 140, 55), (30, 15, 150, 60), None, True, False, 0.95),
                    DetectedRegion((50, 145, 140, 185), (40, 135, 150, 195), None, True, False, 0.95),
                ], mask
            mask[50:90, 50:140] = 255
            return [DetectedRegion((40, 40, 150, 110), (30, 30, 160, 120), None, True, False, 0.95)], mask

    class FakeInp:
        def inpaint(self, rgb, mask):
            inpainted_masks.append((int(np.count_nonzero(mask[:100, :])), int(np.count_nonzero(mask[100:, :]))))
            out = rgb.copy()
            out[mask > 127] = (255, 255, 255)
            return out

    class FakeOcr:
        def recognize_regions(self, rgb, regions):
            if tuple(rgb[5, 5]) == (12, 34, 56):
                return {r.id: "Suongqua...dungma!" for r in regions}  # Page 1 already Vietnamese
            if tuple(rgb[25, 45]) == (30, 120, 200):
                return {1: "Anhyeuemma", 2: "本当に出すの?"}          # Page 3 partial
            return {1: "やめて…"}                                       # Page 2 untranslated

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDet()

        def get_inpainter(self, use_gpu=True):
            return FakeInp()

        def get_ocr(self, use_gpu=True):
            return FakeOcr()

    def fake_ocr_call(self, system_prompt, user_prompt, images, temperature=0.3):
        llm_calls.append((self.model, len(images)))
        if "[Trang 2]" in user_prompt and "[Trang 3]" in user_prompt:
            return {
                "pages": [
                    {"page": 2, "page_summary": "p2", "items": [{"id": 1, "source_text": "やめて…", "vietnamese": "Đừng mà…"}]},
                    {"page": 3, "page_summary": "p3", "items": [{"id": 2, "source_text": "本当に出すの?", "vietnamese": "Bắn vào trong thật hả?"}]},
                ]
            }
        return {
            "page_summary": "single",
            "items": [
                {
                    "id": 1 if "やめて" in user_prompt else 2,
                    "source_text": "やめて…" if "やめて" in user_prompt else "本当に出すの?",
                    "vietnamese": "Đừng mà…" if "やめて" in user_prompt else "Bắn vào trong thật hả?",
                }
            ],
        }

    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_ocr_call)

    out_dir, docs = run_retranslate_pipeline(
        drive_dir,
        model="deepseek-chat",
        api_key="sk-fake-deepseek",
        output_format="png",
        clean_cache=False,
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )

    assert out_dir == drive_dir.parent / "from_drive_fixed"
    assert len(docs) == 3
    # Page 1 skipped: exact marker pixels untouched, zero LLM vision calls
    out1 = np.array(Image.open(out_dir / "001.png").convert("RGB"))
    assert np.array_equal(out1, img1)
    assert docs[0].regions == []
    # Only Page 2 & Page 3 reached the LLM (text-only 0 images) and inpainter (2 calls, not 3!)
    assert len(inpainted_masks) == 2
    assert all(img_cnt == 0 for _, img_cnt in llm_calls)
    # Page 3 only inpainted bottom bubble #2 (0 top mask pixels), leaving bubble #1 (30, 120, 200) untouched!
    assert inpainted_masks[1][0] == 0 and inpainted_masks[1][1] > 0
    out3 = np.array(Image.open(out_dir / "003.png").convert("RGB"))
    assert tuple(out3[25, 45]) == (30, 120, 200)
    assert docs[1].regions[0].translated_text == "Đừng mà…"
    assert docs[2].regions[0].id == 2 and docs[2].regions[0].translated_text == "Bắn vào trong thật hả?"

    # CLI `retranslate` command parses cleanly with help
    res_help = CliRunner().invoke(cli.app, ["retranslate", "--help"])
    assert res_help.exit_code == 0 and "Google Drive" in res_help.stdout


def test_sfx_never_inpainted_or_translated_and_tui_cli_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Floating SFX must never be translated or inpainted when skip_sfx=True, and TUI/CLI queue runs jobs sequentially."""
    from typer.testing import CliRunner

    from koharu_vn_cli import cli
    from koharu_vn_cli.pipeline.gemini_client import (
        TextRegionItem,
        VisionLLMClient,
        looks_like_floating_sfx,
    )
    from koharu_vn_cli.pipeline.runner import run_translate_pipeline
    from koharu_vn_cli.tui import KoharuTUI

    # 1) Katakana + Hiragana SFX recognition without matching actual dialogue
    assert looks_like_floating_sfx("ぐちゅぐちゅッ♡") is True
    assert looks_like_floating_sfx("ドクッドクッ!!") is True
    assert looks_like_floating_sfx("ビクンビクンッ") is True
    assert looks_like_floating_sfx("先輩…だめですっ") is False
    assert looks_like_floating_sfx("これ以上はやめて") is False

    # 2) _merge_translated_regions preserves is_sfx=True & clears vietnamese when model returned both
    base_regions = [
        TextRegionItem(id=1, kind="bubble", text_bbox=(20, 20, 120, 80), source_text="やめろ"),
        TextRegionItem(id=2, kind="floating", text_bbox=(30, 120, 160, 180), source_text="どきどきッ"),
        TextRegionItem(id=3, kind="floating", text_bbox=(30, 190, 160, 240), source_text="ぐちゅぐちゅッ♡"),
    ]
    raw_items = [
        {"id": 1, "is_sfx": False, "source": "やめろ", "vietnamese": "Dừng lại"},
        {"id": 2, "is_sfx": True, "source": "どきどきッ", "vietnamese": "Thình thịch"},
        {"id": 3, "is_sfx": False, "source": "ぐちゅぐちゅッ♡", "vietnamese": "Nhóp nhép"},
    ]
    merged_skip = VisionLLMClient._merge_translated_regions(base_regions, raw_items, skip_sfx=True)
    assert merged_skip[0].is_sfx is False and merged_skip[0].translated_text == "Dừng lại"
    assert merged_skip[1].is_sfx is True and merged_skip[1].translated_text == ""
    assert merged_skip[2].is_sfx is True and merged_skip[2].translated_text == ""

    # 3) End-to-end mini pipeline: floating SFX region leaves canvas pixels untouched and mask excludes it
    manga_dir = tmp_path / "sfx_check"
    manga_dir.mkdir()
    img_arr = np.full((240, 200, 3), 255, dtype=np.uint8)
    # Paint distinctive SFX ink pixels inside floating box (2) so any accidental inpaint is immediately detected
    img_arr[140:160, 60:110] = (200, 30, 40)
    Image.fromarray(img_arr).save(manga_dir / "001.png")

    class FakeDetector:
        def detect(self, rgb, conf_thresh: float = 0.34):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            mask[25:75, 25:135] = 255
            mask[135:175, 45:145] = 255
            return [
                DetectedRegion((20, 20, 140, 80), (15, 15, 145, 85), None, True, False, 0.95),
                DetectedRegion((40, 130, 150, 180), None, None, False, False, 0.90),
            ], mask

    inpaint_masks: list[tuple[int, int]] = []

    class FakeInpainter:
        def inpaint(self, rgb, mask):
            inpaint_masks.append((int((mask[20:80, 20:140] > 0).sum()), int((mask[130:180, 40:150] > 0).sum())))
            out = rgb.copy()
            out[mask > 127] = (255, 255, 255)
            return out

    class FakeCache:
        def get_detector(self, accurate=True, use_gpu=True):
            return FakeDetector()

        def get_inpainter(self, use_gpu=True):
            return FakeInpainter()

        def get_ocr(self, use_gpu=True):
            return None

    def fake_translate_call(self, system_prompt, user_prompt, images=None, temperature=0.25):
        # Deliberately return non-empty vietnamese alongside is_sfx=True to verify hard SFX guard
        return {
            "page_summary": "ok",
            "items": [
                {"id": 1, "is_sfx": False, "source": "やめろ", "vietnamese": "Dừng lại!"},
                {"id": 2, "is_sfx": True, "source": "ドクッ", "vietnamese": "Thình thịch!"},
            ],
        }

    monkeypatch.setattr(VisionLLMClient, "_call_json", fake_translate_call)
    out_dir, docs = run_translate_pipeline(
        manga_dir,
        output=tmp_path / "sfx_out",
        api_key="sk-fake",
        model="gemini-3.8-flash",
        story_pass=False,
        skip_sfx=True,
        vision_translate=True,
        clean_cache=False,
        output_format="png",
        model_cache=FakeCache(),  # type: ignore[arg-type]
    )
    assert len(inpaint_masks) == 1
    assert inpaint_masks[0][0] > 0  # Bubble was masked/inpainted
    assert inpaint_masks[0][1] == 0  # Floating SFX received zero mask pixels
    out_pixels = np.array(Image.open(out_dir / "001.png").convert("RGB"))
    assert tuple(out_pixels[150, 80]) == (200, 30, 40)
    sfx_reg = next(r for r in docs[0].regions if r.id == 2)
    assert sfx_reg.is_sfx is True and sfx_reg.translated_text == ""

    # 4) TUI queue: auto-enqueue while running, multi-path /queue, remove, cancel-pause, auto-advance, clear
    app = KoharuTUI()
    monkeypatch.setattr(app, "_refresh_chrome", lambda: None)
    monkeypatch.setattr(app, "_log_ai", lambda *a, **kw: None)
    monkeypatch.setattr(app, "_ensure_llm_ready", lambda needs_llm=True: True)
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        app,
        "_start_translate",
        lambda input_path, prompt="", tags="", **kw: dispatched.append(("translate", str(input_path))),
    )

    p1 = tmp_path / "vol1"
    p2 = tmp_path / "vol2"
    p3 = tmp_path / "vol3"
    for p in (p1, p2, p3):
        p.mkdir()

    app.is_running_job = True
    app._launch_translate(p1, "giữ tên riêng")
    app._handle_queue_command(f'"{p2}" "{p3}"')
    assert [j.input_path for j in app.job_queue] == [p1, p2, p3]

    app._handle_queue_command("remove 2")
    assert [j.input_path for j in app.job_queue] == [p1, p3]

    # Esc cancellation leaves remaining queue paused instead of launching next item
    app.is_running_job = False
    app._on_worker_finished(cancelled=True)
    assert dispatched == [] and [j.input_path for j in app.job_queue] == [p1, p3]

    # Normal finish pops & dispatches the head item automatically
    app._on_worker_finished(cancelled=False)
    assert dispatched == [("translate", str(p1))]
    assert [j.input_path for j in app.job_queue] == [p3]

    app._handle_queue_command("clear")
    assert app.job_queue == []

    # 5) CLI `queue` runs all provided paths sequentially
    cli_calls: list[Path] = []
    monkeypatch.setattr(
        cli,
        "_execute_translate_cli",
        lambda input_path, **kw: cli_calls.append(Path(input_path)),
    )
    res_q = CliRunner().invoke(cli.app, ["queue", str(p1), str(p2), "--no-notify"])
    assert res_q.exit_code == 0
    assert cli_calls == [p1, p2]







