"""텍스트 블록 지우기(획 단위 inpaint) + 다시 쓰기(원래 줄 높이·정렬·박스 폭 맞춤)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ocr import Block


def _bg_and_text_color(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """박스 테두리 픽셀 중앙값 = 배경색, 배경과 먼 픽셀 = 글자 획."""
    border = np.concatenate([region[0], region[-1], region[:, 0], region[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.abs(region.astype(np.float32) - bg).sum(axis=2)
    return bg, dist, region[dist > 60] if (dist > 60).any() else np.array([[0, 0, 0]])


def stroke_mask(img: np.ndarray, bbox, pad: int = 4, thr: float = 60, dilate: int = 2) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(W, bbox[2] + pad), min(H, bbox[3] + pad)
    mask = np.zeros((H, W), np.uint8)
    region = img[y0:y1, x0:x1]
    if region.size == 0:
        return mask
    _, dist, _ = _bg_and_text_color(region)
    m = (dist > thr).astype(np.uint8) * 255
    if dilate:
        m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    mask[y0:y1, x0:x1] = m
    return mask


def erase_block(img: np.ndarray, block: Block, pad: int = 4) -> np.ndarray:
    m = np.zeros(img.shape[:2], np.uint8)
    for ln in block.lines or []:
        m |= stroke_mask(img, ln.bbox, pad)
    if not block.lines:
        m = stroke_mask(img, block.bbox, pad)
    return cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA)


def _fit_font(font_path: str, lines: list[str], box_w: int, target_h: int, min_size: int = 8):
    size = max(min_size, int(target_h * 1.15))
    while size > min_size:
        f = ImageFont.truetype(font_path, size)
        if max(f.getlength(l) for l in lines) <= box_w:
            return f
        size -= 1
    return ImageFont.truetype(font_path, min_size)


def wrap_like(text: str, n_lines: int) -> list[str]:
    """원래 줄 수에 맞춰 단어 경계로 나눔 (원문 줄바꿈이 있으면 유지)."""
    if "\n" in text:
        return text.split("\n")
    words = text.split()
    if n_lines <= 1 or len(words) <= 1:
        return [text]
    total = len(text)
    out, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and len(cand) > total / n_lines * 1.1 and len(out) < n_lines - 1:
            out.append(cur)
            cur = w
        else:
            cur = cand
    return out + [cur]


def _fill_and_outline(strokes: np.ndarray):
    """획 픽셀에 밝은 색과 어두운 색이 둘 다 많으면 = 채움 + 외곽선 글자(효과음 등)."""
    lum = strokes.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    dark, bright = strokes[lum < 80], strokes[lum > 150]
    n = max(1, len(strokes))
    if len(dark) / n > 0.15 and len(bright) / n > 0.15:
        return tuple(int(c) for c in np.median(bright, axis=0)), tuple(int(c) for c in np.median(dark, axis=0))
    return tuple(int(c) for c in np.median(strokes, axis=0)), None


def draw_block(img: np.ndarray, block: Block, new_text: str, font_path: str,
               orig_img: np.ndarray | None = None, align: str = "center") -> np.ndarray:
    ref = orig_img if orig_img is not None else img
    x0, y0, x1, y1 = block.bbox
    region = ref[y0:y1, x0:x1]
    bg, _, strokes = _bg_and_text_color(region)
    color, outline = _fill_and_outline(strokes)
    contrast = np.abs(np.array(color, float) - bg).sum()

    orig_lines = [l.text for l in block.lines] or block.text.split("\n")
    line_h = int(np.median([l.bbox[3] - l.bbox[1] for l in block.lines])) if block.lines else (y1 - y0) // max(1, len(orig_lines))
    new_lines = wrap_like(new_text, len(orig_lines))
    if block.category == "sfx":  # 효과음은 글자 수가 늘면 박스를 좌우로 넓혀 원래 글자 크기 유지
        old_n = max(1, len(block.text.replace("\n", "")))
        grow = max(0.0, (len(new_text) / old_n - 1) * (x1 - x0)) / 2
        x0, x1 = int(max(0, x0 - grow)), int(min(img.shape[1], x1 + grow))
    font = _fit_font(font_path, new_lines, x1 - x0, line_h)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    asc, desc = font.getmetrics()
    step = max(line_h, asc + desc) * 1.15 if len(new_lines) > 1 else 0
    total_h = (len(new_lines) - 1) * step + asc + desc
    y = y0 + ((y1 - y0) - total_h) / 2
    for ln in new_lines:
        w = font.getlength(ln)
        x = x0 + ((x1 - x0) - w) / 2 if align == "center" else x0
        kw = {}
        if outline is not None:  # 원본이 외곽선 글자
            kw = dict(stroke_width=max(2, font.size // 16), stroke_fill=outline)
        elif contrast < 120:  # 배경 위 글자처럼 대비가 낮으면 외곽선
            kw = dict(stroke_width=max(1, font.size // 12), stroke_fill=tuple(int(255 - c) for c in color))
        d.text((x, y), ln, font=font, fill=color, **kw)
        y += step
    return np.asarray(pil)
