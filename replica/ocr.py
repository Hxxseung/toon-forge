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


MEANINGFUL = re.compile(r"[가-힣A-Za-z0-9]")


def drop_noise(lines: list[Line]) -> list[Line]:
    """'#', '…', '~' 처럼 글자·숫자가 하나도 없는 조각은 버림."""
    return [l for l in lines if MEANINGFUL.search(l.text)]


def merge_words(lines: list[Line], x_gap_ratio: float = 1.2, y_overlap: float = 0.5) -> list[Line]:
    """같은 줄에 나란히 있는 단어 박스를 한 줄로 합침.
    (PP-OCR이 손글씨·띄어쓰기 넓은 대사를 단어마다 따로 검출하는 경우)"""
    # 왼쪽→오른쪽 순서로 처리 (위→아래로 하면 몇 px 높은 오른쪽 단어가 먼저 와서 줄이 끊김)
    items = sorted(lines, key=lambda l: (l.bbox[0], l.bbox[1]))
    rows: list[list[Line]] = []
    for ln in items:
        h = ln.bbox[3] - ln.bbox[1]
        best = None
        for row in rows:
            last = max(row, key=lambda l: l.bbox[2])
            lh = last.bbox[3] - last.bbox[1]
            ov = min(last.bbox[3], ln.bbox[3]) - max(last.bbox[1], ln.bbox[1])
            gap = ln.bbox[0] - last.bbox[2]
            if ov >= y_overlap * min(h, lh) and -0.3 * h <= gap <= x_gap_ratio * max(h, lh):
                best = row
                break
        if best is None:
            rows.append([ln])
        else:
            best.append(ln)
    out = []
    for row in rows:
        row.sort(key=lambda l: l.bbox[0])
        bb = (min(l.bbox[0] for l in row), min(l.bbox[1] for l in row),
              max(l.bbox[2] for l in row), max(l.bbox[3] for l in row))
        out.append(Line(" ".join(l.text for l in row), bb, min(l.score for l in row)))
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
            if -0.3 * h <= ln.bbox[1] - last.bbox[3] <= h * y_gap_ratio and w > 0 and ov / w >= x_overlap:
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
    # 효과음: 한 줄·짧고(≤4자) 글자가 대사보다 확실히 크거나, 같은 글자 반복(쾅쾅, 두근두근)
    # ('그래!'처럼 느낌표로 끝나는 짧은 대사는 효과음 아님)
    if len(group) == 1 and HANGUL.search(compact) and len(compact) <= 4 and (
            h > 1.6 * median_h or re.fullmatch(r"(.{1,2})\1+", compact)):
        return "sfx"
    return "dialogue"


def build_blocks(lines: list[Line], is_edge_cut: bool = False) -> list[Block]:
    """OCR 원시 줄 → 잡음 제거 → 같은 줄 단어 합치기 → 말풍선 단위로 묶기 → 분류."""
    lines = merge_words(drop_noise(lines))
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


def save_blocks(path: str | Path, cut_id: str, blocks: list[Block], engine: str = "paddleocr-ppocrv5-ko",
                raw_lines: list[Line] | None = None):
    """raw_lines: OCR이 낸 원시 줄(단어) — 저장해 두면 묶는 규칙을 바꿔도 OCR을 다시 안 돌려도 됨."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = {"cut_id": cut_id, "engine": engine, "reviewed": False, "blocks": [asdict(b) for b in blocks]}
    if raw_lines is not None:
        d["raw_lines"] = [asdict(l) for l in raw_lines]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def load_raw_lines(path: str | Path) -> tuple[dict, list[Line]]:
    """저장된 원시 줄. 예전 JSON(raw_lines 없음)은 블록 안 lines를 원시 줄로 사용."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    src = d.get("raw_lines") or [l for b in d["blocks"] for l in b.get("lines", [])]
    return d, [Line(l["text"], tuple(l["bbox"]), l["score"]) for l in src]


def load_blocks(path: str | Path) -> list[Block]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [Block(b["id"], b["category"], b["text"], tuple(b["bbox"]),
                  [Line(l["text"], tuple(l["bbox"]), l["score"]) for l in b.get("lines", [])])
            for b in d["blocks"]]