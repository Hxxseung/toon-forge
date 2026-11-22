"""이미지 변형 품질 검사(QC) 지표.

파일럿(2026-11-11)에서 본 실패 유형 → 지표:
- 인물이 늘어남 (마마신 c030: 1명 → 2명, c126: 1명 → 3~4명)  → person_count_orig / person_count_out
- 사람 영역 안에서 구도가 바뀜 (c126 얼굴 확대)                → structure_ncc (낮을수록 많이 바뀜)
- 마스크 경계에 이음새 (c126, c069)                            → seam_score (높을수록 경계가 튐)
- 말풍선 글자가 사라짐 (CAT c041 "나옹~")                       → text_retention (낮을수록 글자 손실)
기준값은 파일럿 결과로 정함 → 처음엔 측정만 (`cli qc`).
"""
from __future__ import annotations

from difflib import SequenceMatcher

import cv2
import numpy as np

from .people import people_mask


def merge_boxes(boxes: list[list], iou_thr: float = 0.3, contain_thr: float = 0.6) -> list[list[float]]:
    """같은 사람의 person/face 박스, 타일 경계에서 겹친 박스를 한 덩어리로 → 인원수 근사."""
    bs = [list(map(float, b[:4])) for b in boxes]
    parent = list(range(len(bs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            a, b = bs[i], bs[j]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            if inter <= 0:
                continue
            union = area(a) + area(b) - inter
            if inter / union >= iou_thr or inter / max(1e-6, min(area(a), area(b))) >= contain_thr:
                parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for i, b in enumerate(bs):
        groups.setdefault(find(i), []).append(b)
    return [[min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g)]
            for g in groups.values()]


def structure_ncc(orig: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float | None:
    """사람 영역 안의 '큰 형태' 유사도: 흐린 흑백 영상의 정규화 상관 (색이 바뀌어도 형태가 같으면 높음)."""
    m = mask > 0.5
    if m.sum() < 100:
        return None
    s = max(2.0, min(orig.shape[:2]) / 120)
    g = [cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), s) for x in (orig, out)]
    a, b = g[0][m], g[1][m]
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-6 else None


def seam_score(orig: np.ndarray, out: np.ndarray, mask: np.ndarray, band: int = 6) -> float | None:
    """마스크 경계 띠에서 '원본에 없던 경계선'의 세기 (0~1). 이음새가 보이면 커짐."""
    edge = cv2.morphologyEx((mask > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ring = cv2.dilate(edge, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0
    if ring.sum() < 50:
        return None

    def grad(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    extra = np.clip(grad(out) - grad(orig), 0, None)[ring]
    return float(extra.mean() / 255.0)


def _compact(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def text_retention(ocr_engine, out: np.ndarray, blocks, pad: int = 8) -> float | None:
    """원본 OCR 블록(대사·효과음) 자리를 결과 이미지에서 다시 읽어 글자가 남았는지 (0~1, 블록 평균)."""
    import tempfile
    from pathlib import Path

    from PIL import Image
    H, W = out.shape[:2]
    scores = []
    for b in blocks:
        ref = _compact(b.text)
        if b.category not in ("dialogue", "sfx") or len(ref) < 1:
            continue
        x0, y0, x1, y1 = b.bbox
        crop = out[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
        if crop.size == 0:
            continue
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.png"
            Image.fromarray(crop).save(p)
            got = _compact("".join(l.text for l in ocr_engine.lines(p)))
        scores.append(SequenceMatcher(None, ref, got).ratio())
    return float(np.mean(scores)) if scores else None


def measure(orig: np.ndarray, out: np.ndarray, orig_boxes: list[list], out_boxes: list[list] | None,
            mask_pad: float, ocr_engine=None, blocks=None) -> dict:
    mask = people_mask(orig.shape[:2], orig_boxes, mask_pad) if orig_boxes else np.ones(orig.shape[:2], np.float32)
    q = {
        "person_count_orig": len(merge_boxes(orig_boxes)) if orig_boxes else 0,
        "person_count_out": len(merge_boxes(out_boxes)) if out_boxes is not None else None,
        "structure_ncc": structure_ncc(orig, out, mask),
        "seam_score": seam_score(orig, out, mask) if orig_boxes else None,
    }
    if ocr_engine is not None and blocks:
        q["text_retention"] = text_retention(ocr_engine, out, blocks)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}
