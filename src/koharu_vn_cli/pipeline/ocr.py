"""Offline CPU ONNX OCR powered by mayocream/manga-ocr-onnx (zero PyTorch required)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from koharu_vn_cli.models import TextRegionItem
from koharu_vn_cli.pipeline.detector import _create_ort_session

MANGA_OCR_REPO = "mayocream/manga-ocr-onnx"
_CLS_ID = 2
_SEP_ID = 3
_FIRST_NORMAL_ID = 5


def _ensure_manga_ocr_files() -> tuple[Path, Path, Path]:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    enc = Path(hf_hub_download(MANGA_OCR_REPO, "encoder_model.onnx"))
    dec = Path(hf_hub_download(MANGA_OCR_REPO, "decoder_model.onnx"))
    vocab = Path(hf_hub_download(MANGA_OCR_REPO, "vocab.txt"))
    return enc, dec, vocab


def _clean_ocr_text(text: str) -> str:
    text = text.replace(" ", "").replace("\u3000", "")
    text = re.sub(r"・{3,}", "…", text)
    return text.strip()


class MangaOcrONNX:
    """Fast local CPU manga/CG text recognizer using ONNX encoder + decoder."""

    def __init__(
        self,
        encoder_path: str | Path | None = None,
        decoder_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        *,
        max_length: int = 128,
        use_gpu: bool = True,
    ) -> None:
        if encoder_path and decoder_path and vocab_path:
            enc, dec, voc = Path(encoder_path), Path(decoder_path), Path(vocab_path)
        else:
            enc, dec, voc = _ensure_manga_ocr_files()

        threads = max(1, min(4, os.cpu_count() or 2))
        self._enc_path = enc
        self._dec_path = dec
        self._cpu_threads = threads
        self.encoder = _create_ort_session(enc, use_gpu=use_gpu, cpu_threads=threads, low_mem=True)
        self.decoder = _create_ort_session(dec, use_gpu=use_gpu, cpu_threads=threads, low_mem=True)
        self.vocab = voc.read_text(encoding="utf-8").splitlines()
        self.max_length = max(16, max_length)

    def _downgrade_to_cpu(self) -> bool:
        if (
            self.encoder.get_providers() == ["CPUExecutionProvider"]
            and self.decoder.get_providers() == ["CPUExecutionProvider"]
        ):
            return False
        self.encoder = _create_ort_session(
            self._enc_path, use_gpu=False, cpu_threads=self._cpu_threads, low_mem=True
        )
        self.decoder = _create_ort_session(
            self._dec_path, use_gpu=False, cpu_threads=self._cpu_threads, low_mem=True
        )
        return True

    @staticmethod
    def _preprocess(img: Image.Image) -> np.ndarray:
        gray_rgb = img.convert("L").convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(gray_rgb, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return arr.transpose(2, 0, 1)[None, ...]

    def _decode_ids(self, token_ids: list[int]) -> str:
        pieces: list[str] = []
        for tid in token_ids:
            if tid < _FIRST_NORMAL_ID or tid >= len(self.vocab):
                continue
            tok = self.vocab[tid]
            if tok.startswith("##"):
                tok = tok[2:]
            pieces.append(tok)
        return _clean_ocr_text("".join(pieces))

    def recognize_crop(self, img: Image.Image | np.ndarray) -> str:
        """Run ViT-BERT autoregressive decoding on a single cropped text box."""
        pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        pixel_values = self._preprocess(pil)
        try:
            enc_hidden = self.encoder.run(None, {"pixel_values": pixel_values})[0]
            ids: list[int] = [_CLS_ID]
            for _ in range(self.max_length):
                inp = np.asarray([ids], dtype=np.int64)
                logits = self.decoder.run(
                    None,
                    {"input_ids": inp, "encoder_hidden_states": enc_hidden},
                )[0]
                next_id = int(np.argmax(logits[0, -1]))
                ids.append(next_id)
                if next_id == _SEP_ID:
                    break
            return self._decode_ids(ids)
        except Exception:
            if not self._downgrade_to_cpu():
                raise
            enc_hidden = self.encoder.run(None, {"pixel_values": pixel_values})[0]
            ids = [_CLS_ID]
            for _ in range(self.max_length):
                inp = np.asarray([ids], dtype=np.int64)
                logits = self.decoder.run(
                    None,
                    {"input_ids": inp, "encoder_hidden_states": enc_hidden},
                )[0]
                next_id = int(np.argmax(logits[0, -1]))
                ids.append(next_id)
                if next_id == _SEP_ID:
                    break
            return self._decode_ids(ids)

    def recognize_regions(self, rgb: np.ndarray, regions: list[TextRegionItem]) -> dict[int, str]:
        """Crop each detected text region from `rgb` and return `{region_id: source_text}`."""
        if not regions:
            return {}
        page_pil = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        results: dict[int, str] = {}

        for reg in regions:
            x1, y1, x2, y2 = reg.text_bbox
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = reg.bubble_bbox
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad_x = max(4, int(bw * 0.08))
            pad_y = max(4, int(bh * 0.08))
            crop = page_pil.crop(
                (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(w, x2 + pad_x),
                    min(h, y2 + pad_y),
                )
            )
            results[reg.id] = self.recognize_crop(crop)
        return results
