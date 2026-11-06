"""컷 OCR → 텍스트 블록(dialogue / sfx / credit).

엔진: PaddleOCR (PP-OCRv5, korean). paddlepaddle 3.3.x CPU에서 oneDNN 문제가 있었으므로 3.2.2 권장.
결과 JSON은 사람이 직접 고칠 수 있게 저장하고, 변형 단계는 이 JSON만 읽는다
(→ OCR 오류를 고치면 변형 결과도 그대로 따라감).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CREDIT_PAT = re.compile(r"(글\s*[:/|·]|그림\s*[:/|·]|원작|각색|작가|어시|채색|©|copyright|studio|스튜디오|제작)", re.I)
HANGUL = re.compile(r"[가-힣]")


@dataclass
class Line:
    text: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    score: float


@dataclass
class Block:
    id: int
    category: str                     # dialogue / sfx / credit
    text: str
    bbox: tuple[int, int, int, int]
    lines: list[Line] = field(default_factory=list)


class PaddleKoreanOCR:
    def __init__(self, min_score: float = 0.5, device: str | None = None):
        from paddleocr import PaddleOCR  # 지연 import
        kw = dict(lang="korean", use_doc_orientation_classify=False,
                  use_doc_unwarping=False, use_textline_orientation=False)
        if device:
            kw["device"] = device
        self.ocr = PaddleOCR(**kw)
        self.min_score = min_score

    def lines(self, image_path: str | Path) -> list[Line]:
        res = self.ocr.predict(str(image_path))
        out: list[Line] = []
        for r in res:
            d = r.json["res"] if hasattr(r, "json") and "res" in r.json else r
            texts, scores = d["rec_texts"], d["rec_scores"]
            boxes = d.get("rec_boxes")
            if boxes is None:  # polygon만 있으면 외접 사각형
                boxes = [[min(p[0] for p in q), min(p[1] for p in q),
                          max(p[0] for p in q), max(p[1] for p in q)] for q in d["rec_polys"]]
            for t, s, b in zip(texts, scores, boxes):
                if s >= self.min_score and t.strip():
                    out.append(Line(t.strip(), tuple(int(v) for v in b), float(s)))
        return out


def group_lines(lines: list[Line], y_gap_ratio: float = 0.9, x_overlap: float = 0.2) -> list[list[Line]]:
    """세로로 가깝고 가로로 겹치는 줄을 한 블록(말풍선 하나)으로 묶음."""
    lines = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    groups: list[list[Line]] = []
    for ln in lines:
        h = ln.bbox[3] - ln.bbox[1]
        placed = False
        for g in groups:
            last = g[-1]
            gx0, gx1 = min(l.bbox[0] for l in g), max(l.bbox[2] for l in g)
            ov = min(gx1, ln.bbox[2]) - max(gx0, ln.bbox[0])
            w = min(gx1 - gx0, ln.bbox[2] - ln.bbox[0])
            if 0 <= ln.bbox[1] - last.bbox[3] <= h * y_gap_ratio and w > 0 and ov / w >= x_overlap:
                g.append(ln)
                placed = True
                break
        if not placed:
            groups.append([ln])
    return groups


def classify(group: list[Line], median_h: float, is_edge_cut: bool) -> str:
    text = " ".join(l.text for l in group)
    if CREDIT_PAT.search(text) or (is_edge_cut and len(group) >= 2 and not text.endswith(("?", "!", "."))):
        return "credit"
    compact = re.sub(r"[\s!?~.…]", "", text)
    h = max(l.bbox[3] - l.bbox[1] for l in group)
    # 효과음: 짧고(≤4자) 글자가 크거나, 같은 글자 반복(쾅쾅, 두근두근)
    if len(group) == 1 and HANGUL.search(compact) and len(compact) <= 4 and (
            h > 1.6 * median_h or re.fullmatch(r"(.{1,2})\1+", compact) or text.endswith("!")):
        return "sfx"
    return "dialogue"


def build_blocks(lines: list[Line], is_edge_cut: bool = False) -> list[Block]:
    if not lines:
        return []
    hs = sorted(l.bbox[3] - l.bbox[1] for l in lines)
    med = hs[len(hs) // 2]
    blocks = []
    for i, g in enumerate(group_lines(lines)):
        bb = (min(l.bbox[0] for l in g), min(l.bbox[1] for l in g),
              max(l.bbox[2] for l in g), max(l.bbox[3] for l in g))
        blocks.append(Block(i, classify(g, med, is_edge_cut), "\n".join(l.text for l in g), bb, g))
    return blocks


def save_blocks(path: str | Path, cut_id: str, blocks: list[Block], engine: str = "paddleocr-ppocrv5-ko"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cut_id": cut_id, "engine": engine, "reviewed": False,
                   "blocks": [asdict(b) for b in blocks]}, f, ensure_ascii=False, indent=1)


def load_blocks(path: str | Path) -> list[Block]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [Block(b["id"], b["category"], b["text"], tuple(b["bbox"]),
                  [Line(l["text"], tuple(l["bbox"]), l["score"]) for l in b.get("lines", [])])
            for b in d["blocks"]]
