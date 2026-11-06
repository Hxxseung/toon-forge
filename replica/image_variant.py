"""이미지만 변형 — 캐릭터 / 배경 / 화풍 (강도 weak·medium·strong).

- 프롬프트 = 강도별 템플릿(configs/default.yaml) + 작품별 재설계 표(configs/works/<work_id>.yaml)
- 편집 후 원래 크기로 되돌리고, 텍스트 블록 영역은 원본 픽셀을 다시 덮어 글자를 그대로 유지
  (→ '이미지만' 변형에서 텍스트가 섞여 바뀌는 것을 막음)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .ocr import Block
from .text_render import stroke_mask


def build_prompt(axis: str, strength: str, templates: dict[str, Any], table: dict[str, Any]) -> tuple[str, list[str]]:
    """반환: (프롬프트, 참조 이미지 경로들)"""
    t = templates[axis][strength]
    design, refs = "", []
    if axis == "character":
        chars = table.get("character_redesign", []) or []
        design = "; ".join(f"{c.get('from', 'the character')} → {c['to']}" for c in chars) or \
                 "give every character a different hairstyle, hair color and outfit"
        refs = [c["ref_image"] for c in chars if c.get("ref_image")]
    elif axis == "background":
        design = table.get("background_redesign") or "a different but plausible location with the same mood"
    elif axis == "style":
        design = table.get("style_target") or "a different digital webtoon art style"
    prompt = t.format(design=design).strip() + " " + templates["keep_suffix"].strip()
    return prompt, refs


def protect_text(edited: np.ndarray, original: np.ndarray, blocks: list[Block], dilate: int = 2) -> np.ndarray:
    """원본의 글자 '획' 픽셀만 다시 붙임 (사각형째 붙이면 바뀐 말풍선·배경 위에 네모 자국이 남음)."""
    if not blocks:
        return edited
    m = np.zeros(original.shape[:2], np.uint8)
    for b in blocks:
        for bb in ([l.bbox for l in b.lines] or [b.bbox]):
            m |= stroke_mask(original, bb, pad=4, dilate=dilate)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255, (3, 3), 0)[..., None]
    return (original * m + edited * (1 - m)).astype(np.uint8)


def pixel_change_ratio(a: np.ndarray, b: np.ndarray, thr: int = 30) -> float:
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2) > thr).mean())


def run_image_edit(editor, orig: Image.Image, axis: str, strength: str, seed: int,
                   templates: dict[str, Any], table: dict[str, Any], blocks: list[Block],
                   protect: bool = True, ref_root: str | Path = ".") -> tuple[np.ndarray, dict[str, Any]]:
    prompt, ref_paths = build_prompt(axis, strength, templates, table)
    refs = [Image.open(Path(ref_root) / p) for p in ref_paths] if editor.supports_refs else []
    out = editor.edit(orig, prompt, seed, refs=refs or None)
    if out.size != orig.size:
        out = out.resize(orig.size, Image.LANCZOS)
    o, e = np.asarray(orig.convert("RGB")), np.asarray(out.convert("RGB"))
    if protect:
        e = protect_text(e, o, blocks)
    info = {**editor.info(), "prompt": prompt, "ref_images": ref_paths if refs else [],
            "protect_text": protect, "n_protected_blocks": len(blocks) if protect else 0}
    return e, info
