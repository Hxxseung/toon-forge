"""컷 안의 사람(캐릭터) 검출 — 캐릭터 변형 대상 판정 + 편집 영역 마스크.

OWLv2(개방형 검출, 그림체에도 비교적 강함)에 "a person", "a human face"로 질의.
세로로 긴 웹툰 컷은 정사각형으로 패딩되면 너무 작아지므로 정사각 타일로 잘라 검출 후 좌표를 합침.
결과: outputs/people/<cut_id>.json  {has_person, boxes:[[x0,y0,x1,y1,score,label]], ...}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def tile_offsets(length: int, tile: int, overlap: float = 0.25) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    offs = list(range(0, length - tile + 1, step))
    if offs[-1] != length - tile:
        offs.append(length - tile)
    return offs


def tiles(img: Image.Image, overlap: float = 0.25):
    """긴 쪽을 따라 정사각 타일 (짧은 변 길이). (crop, (dx, dy))"""
    w, h = img.size
    t = min(w, h)
    if h >= w:
        return [(img.crop((0, y, w, y + t)), (0, y)) for y in tile_offsets(h, t, overlap)]
    return [(img.crop((x, 0, x + t, h)), (x, 0)) for x in tile_offsets(w, t, overlap)]


class PersonDetector:
    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble",
                 queries=("a person", "a human face"), threshold: float = 0.25, device: str | None = None,
                 min_box_ratio: float = 0.002):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.model_id, self.queries, self.threshold = model_id, list(queries), threshold
        self.min_box_ratio = min_box_ratio  # 컷 면적 대비 이보다 작은 박스는 버림 (잡음)

    def _detect_tile(self, im: Image.Image):
        import torch
        inputs = self.proc(text=[self.queries], images=im, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        s = max(im.size)  # OWLv2는 정사각형으로 패딩 → 좌표 기준은 패딩된 정사각형
        res = self.proc.image_processor.post_process_object_detection(
            out, threshold=self.threshold, target_sizes=torch.tensor([[s, s]], device=self.device))[0]
        return [(*[float(v) for v in b.tolist()], float(sc), self.queries[int(lb)])
                for b, sc, lb in zip(res["boxes"], res["scores"], res["labels"])]

    def detect(self, img: Image.Image) -> list[list]:
        img = img.convert("RGB")
        W, H = img.size
        boxes = []
        for crop, (dx, dy) in tiles(img):
            for x0, y0, x1, y1, sc, lb in self._detect_tile(crop):
                x0, x1 = max(0, x0 + dx), min(W, x1 + dx)
                y0, y1 = max(0, y0 + dy), min(H, y1 + dy)
                if (x1 - x0) * (y1 - y0) >= self.min_box_ratio * W * H:
                    boxes.append([round(x0), round(y0), round(x1), round(y1), round(sc, 3), lb])
        return boxes

    def info(self) -> dict:
        return {"model": self.model_id, "queries": self.queries, "threshold": self.threshold}


def save_people(path: str | Path, cut_id: str, boxes: list[list], info: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"cut_id": cut_id, "has_person": bool(boxes), "n_boxes": len(boxes), **info, "boxes": boxes},
                  f, ensure_ascii=False, indent=1)


def load_people(path: str | Path) -> dict | None:
    p = Path(path)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def people_mask(shape_hw: tuple[int, int], boxes: list[list], pad_ratio: float = 0.15, feather: int = 15) -> np.ndarray:
    """사람 박스(가로·세로로 pad_ratio만큼 넓힘)의 합집합 → 0~1 마스크 (경계는 부드럽게)."""
    H, W = shape_hw
    m = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1, *_ in boxes:
        pw, ph = (x1 - x0) * pad_ratio, (y1 - y0) * pad_ratio
        m[int(max(0, y0 - ph)):int(min(H, y1 + ph)), int(max(0, x0 - pw)):int(min(W, x1 + pw))] = 1.0
    if feather:
        k = 2 * feather + 1
        m = np.clip(cv2.GaussianBlur(m, (k, k), 0) * 2 - 0.5, 0, 1)  # 박스 안쪽은 1 유지, 바깥으로만 번짐
    return m
