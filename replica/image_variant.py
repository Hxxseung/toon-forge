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


def is_grayscale(img: Image.Image, sat_thr: float = 18, frac: float = 0.97) -> bool:
    """흑백(그레이스케일) 컷 판정: 채도가 낮은 픽셀이 대부분이면 흑백."""
    hsv = np.asarray(img.convert("RGB").resize((256, max(1, int(256 * img.height / img.width)))).convert("HSV"))
    return float((hsv[..., 1] < sat_thr).mean()) >= frac


def _pick(rng: np.random.Generator, pool: list[str], n: int) -> list[str]:
    return [pool[i] for i in rng.permutation(len(pool))[:n]]


def auto_design(work_id: str, strength: str, gray: bool, pools: dict[str, Any]) -> str:
    """치환표가 없을 때 작품 단위로 고정된 구체적 변경 규칙을 만듦.
    같은 작품이면 모든 컷에서 같은 규칙 → 같은 인물이 항상 같은 모습으로 바뀜 (컷마다 랜덤 X)."""
    from .config import stable_seed
    rng = np.random.default_rng(stable_seed("design", work_id))
    if gray:  # 흑백: 색 이름 대신 톤·형태
        tone = pools["gray_hair_tone"][rng.integers(len(pools["gray_hair_tone"]))]
        style = pools["hairstyle_swap"][rng.integers(len(pools["hairstyle_swap"]))]
        cloth = _pick(rng, pools["gray_outfit"], 1)[0]
        parts = {"weak": [tone], "medium": [tone, cloth], "strong": [tone, style, cloth]}[strength]
    else:
        src = pools["hair_from"]
        dst = _pick(rng, pools["hair_to"], len(src))
        for i, a in enumerate(src):  # "blue → blue" 처럼 같은 색으로 가는 쌍 제거
            if dst[i] in a:
                alt = [c for c in pools["hair_to"] if c not in a and c not in dst]
                dst[i] = alt[0] if alt else dst[(i + 1) % len(dst)]
        hair = ", ".join(f"{a} hair becomes {b} hair" for a, b in zip(src, dst))
        cloth = _pick(rng, pools["outfit"], 1)[0]
        style = pools["hairstyle_swap"][rng.integers(len(pools["hairstyle_swap"]))]
        parts = {"weak": [hair], "medium": [hair, cloth], "strong": [hair, style, cloth]}[strength]
    return "; ".join(parts)


def build_prompt(axis: str, strength: str, templates: dict[str, Any], table: dict[str, Any],
                 work_id: str = "", gray: bool = False) -> tuple[str, list[str]]:
    """반환: (프롬프트, 참조 이미지 경로들)"""
    t = templates[axis][strength]
    design, refs = "", []
    if axis == "character":
        chars = table.get("character_redesign", []) or []
        if chars:
            design = "; ".join(f"{c.get('from', 'the person')} → {c['to']}" for c in chars)
            refs = [c["ref_image"] for c in chars if c.get("ref_image")]
        else:
            design = auto_design(work_id, strength, gray, templates["auto_pools"])
    elif axis == "background":
        design = table.get("background_redesign") or "a different but plausible location with the same mood"
    elif axis == "style":
        design = table.get("style_target") or "a different digital webtoon art style"
    keep = templates["keep_suffix"].strip()
    if gray:
        keep += " " + templates["keep_gray"].strip()
    return t.format(design=design).strip() + " " + keep, refs


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
                   protect: bool = True, ref_root: str | Path = ".", work_id: str = "",
                   person_boxes: list | None = None, mask_pad: float = 0.15) -> tuple[np.ndarray, dict[str, Any]]:
    """person_boxes가 주어지면(캐릭터 축) 사람 영역 밖은 원본으로 되돌림 → 배경에 사람·물체가 새로 생기는 것 방지."""
    gray = is_grayscale(orig)
    prompt, ref_paths = build_prompt(axis, strength, templates, table, work_id, gray)
    refs = [Image.open(Path(ref_root) / p) for p in ref_paths] if editor.supports_refs else []
    out = editor.edit(orig, prompt, seed, refs=refs or None)
    if out.size != orig.size:
        out = out.resize(orig.size, Image.LANCZOS)
    o, e = np.asarray(orig.convert("RGB")), np.asarray(out.convert("RGB"))
    masked = axis == "character" and bool(person_boxes)
    if masked:
        from .people import people_mask
        m = people_mask(o.shape[:2], person_boxes, mask_pad)[..., None]
        e = (o * (1 - m) + e * m).astype(np.uint8)
    if protect:
        e = protect_text(e, o, blocks)
    info = {**editor.info(), "prompt": prompt, "grayscale_input": gray, "ref_images": ref_paths if refs else [],
            "person_mask": {"applied": masked, "n_boxes": len(person_boxes or []), "pad_ratio": mask_pad},
            "protect_text": protect, "n_protected_blocks": len(blocks) if protect else 0}
    return e, info