"""원작 회차 → 컷 분할.

입력: <data_root>/<work_id>/<episode>/*.jpg|png  (세로 스크롤 조각, 파일명 순서 = 위→아래)
1) 조각을 같은 너비로 맞춰 세로로 이어 붙임
2) 가로 한 줄 전체가 거의 단색(흰/검정 등)인 행 = 여백(gutter)
3) min_gutter 픽셀 이상 이어진 여백에서 자르고, 너무 작은 조각은 버림
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Cut:
    image: Image.Image
    y0: int
    y1: int
    slices: list[str]


def list_slices(ep_dir: Path) -> list[Path]:
    return sorted(p for p in ep_dir.iterdir() if p.suffix.lower() in IMG_EXT)


def stitch(paths: list[Path]) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w = min(im.width for im in imgs)
    rows, spans, y = [], [], 0
    for p, im in zip(paths, imgs):
        if im.width != w:
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        a = np.asarray(im)
        rows.append(a)
        spans.append((p.name, y, y + a.shape[0]))
        y += a.shape[0]
    return np.concatenate(rows, axis=0), spans


def gutter_rows(arr: np.ndarray, std_thr: float = 6.0, margin_ratio: float = 0.0) -> np.ndarray:
    """행 단위 여백 판정. 행의 픽셀 표준편차가 작으면(=단색) 여백."""
    g = arr.astype(np.float32).mean(axis=2)
    if margin_ratio > 0:  # 좌우 가장자리 무시 옵션
        m = int(g.shape[1] * margin_ratio)
        g = g[:, m:g.shape[1] - m]
    return g.std(axis=1) < std_thr


def gutter_rows_partial(arr: np.ndarray, frac: float = 0.9, white: int = 235, black: int = 20) -> np.ndarray:
    """행의 frac 이상이 흰색(>=white) 또는 검은색(<=black)이면 여백.
    말풍선·효과선이 여백을 일부 가로질러도 여백으로 인식 (긴 컷 재분할에만 사용)."""
    g = arr.astype(np.int16).max(axis=2), arr.astype(np.int16).min(axis=2)
    is_white = (g[1] >= white).mean(axis=1)
    is_black = (g[0] <= black).mean(axis=1)
    return (is_white >= frac) | (is_black >= frac)


def _segments(content: np.ndarray, min_gutter: int, offset: int = 0) -> list[tuple[int, int]]:
    """content[행]=True 인 구간들. min_gutter 미만의 짧은 여백은 컷 내부로 취급."""
    H, i, segs = len(content), 0, []
    while i < H:
        if not content[i]:
            i += 1
            continue
        j = i
        while j < H:
            if content[j]:
                j += 1
                continue
            k = j
            while k < H and not content[k]:
                k += 1
            if k - j >= min_gutter or k == H:
                break
            j = k
        segs.append((i + offset, j + offset))
        i = j
    return segs


def split_cuts(arr: np.ndarray, spans, min_gutter: int = 40, min_cut_h: int = 80,
               std_thr: float = 6.0, pad: int = 0, max_cut_h: int = 0,
               resplit_gutter: int = 15, resplit_std: float = 12.0, merge_below: int = 0,
               resplit_frac: float = 0.9) -> list[Cut]:
    """
    1) 기본 기준으로 분할
    2) max_cut_h 보다 긴 컷만 완화된 기준(resplit_*)으로 한 번 더 분할 — 여백이 좁거나 노이즈 낀 경우
    3) min_cut_h 미만 = 잡음으로 버림, merge_below 미만 = 위 컷에 붙임(첫 컷이면 아래 컷에)
    """
    H = arr.shape[0]
    segs = _segments(~gutter_rows(arr, std_thr), min_gutter)

    if max_cut_h:
        out = []
        for y0, y1 in segs:
            if y1 - y0 > max_cut_h:
                part = arr[y0:y1]
                gap = gutter_rows(part, resplit_std)
                if resplit_frac:
                    gap |= gutter_rows_partial(part, resplit_frac)
                sub = _segments(~gap, resplit_gutter, y0)
                out += sub if len(sub) > 1 else [(y0, y1)]
            else:
                out.append((y0, y1))
        segs = out

    segs = [s for s in segs if s[1] - s[0] >= min_cut_h]
    if merge_below and len(segs) > 1:
        merged: list[list[int]] = []
        for y0, y1 in segs:
            if merged and y1 - y0 < merge_below:
                merged[-1][1] = y1                      # 위 컷에 붙임 (사이 여백 포함)
            elif merged and merged[-1][1] - merged[-1][0] < merge_below:
                merged[-1][1] = y1                      # 첫 조각이 짧았으면 아래 컷과 합침
            else:
                merged.append([y0, y1])
        segs = [tuple(s) for s in merged]

    cuts = []
    for y0, y1 in segs:
        a, b = max(0, y0 - pad), min(H, y1 + pad)
        names = [n for n, s, e in spans if s < b and e > a]
        cuts.append(Cut(Image.fromarray(arr[a:b]), a, b, names))
    return cuts