"""원작 | 변형(태그별) 을 나란히 붙인 비교 이미지.

python tools/compare_sheet.py                                  # image 변형 전부, 태그별로 열
python tools/compare_sheet.py --tags "" lightning --limit 5    # 기본(태그 없음) vs lightning
python tools/compare_sheet.py --type both --axis character --strength medium
→ check_<시각>.png  (행 = 컷, 열 = 원작 + 태그별 변형). 아래에 프롬프트도 출력.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replica.cli import load_original  # noqa: E402
from replica.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--type", default="image")
    ap.add_argument("--axis")
    ap.add_argument("--strength")
    ap.add_argument("--tags", nargs="*", help="비교할 태그들 (태그 없음은 \"\")")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(cfg["out_root"])
    origs = {r["cut_id"]: r for r in map(json.loads, open(out / "manifests/originals.jsonl"))}
    vs = [v for v in map(json.loads, open(out / "manifests/variants.jsonl"))
          if v["variant_type"] == a.type and v["status"] == "ok" and v["path"]
          and (not a.axis or v["axis"] == a.axis) and (not a.strength or v["strength"] == a.strength)]

    # 컷 → {태그: 변형}
    by_cut: dict[str, dict[str, dict]] = {}
    for v in vs:
        tag = (re.search(r"-s\d+-?(.*)$", v["variant_id"]) or [None, ""])[1]
        if a.tags is not None and tag not in a.tags:
            continue
        by_cut.setdefault(v["original_cut_id"], {})[tag] = v
    tags = a.tags if a.tags is not None else sorted({t for d in by_cut.values() for t in d})
    cuts = list(by_cut)[: a.limit]
    if not cuts:
        sys.exit("비교할 변형이 없습니다.")

    H, gap, head = a.height, 12, 28
    rows, prompts = [], {}
    for cid in cuts:
        o = load_original(cfg, origs[cid])
        s = H / o.height
        ims = [o.resize((max(1, int(o.width * s)), H))]
        for t in tags:
            v = by_cut[cid].get(t)
            if v:
                ims.append(Image.open(out / v["path"]).convert("RGB").resize(ims[0].size))
                prompts[t] = (v.get("image_edit") or {}).get("prompt", "")
            else:
                ims.append(Image.new("RGB", ims[0].size, (200, 200, 200)))
        row = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), H + head), "white")
        d, x = ImageDraw.Draw(row), 0
        for i, (im, label) in enumerate(zip(ims, ["original"] + [t or "(base)" for t in tags])):
            row.paste(im, (x, head))
            d.text((x + 4, 6), f"{cid} | {label}" if i == 0 else label, fill="black")
            x += im.width + gap
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height + gap for r in rows)), (128, 128, 128))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + gap
    name = f"check_{time.strftime('%H%M%S')}.png"
    sheet.save(name)
    print("저장:", name, f"({len(cuts)}컷 × 열 {['original'] + [t or '(base)' for t in tags]})")
    for t, p in prompts.items():
        print(f"\n[{t or '(base)'}] {p}")


if __name__ == "__main__":
    main()