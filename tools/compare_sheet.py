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

from PIL import Image, ImageDraw, ImageFont

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
    ap.add_argument("--offset", type=int, default=0, help="앞에서 N개 건너뛰기 (긴 목록을 나눠 볼 때)")
    ap.add_argument("--include-failed", action="store_true", help="QC 불합격(failed_qc) 결과도 표시")
    ap.add_argument("--filter", choices=["retried", "flagged", "failed", "review"],
                    help="retried=재시도 후 합격, flagged=low_structure 표시, failed=불합격, review=retried+flagged")
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(cfg["out_root"])
    origs = {r["cut_id"]: r for r in map(json.loads, open(out / "manifests/originals.jsonl"))}
    vs = [v for v in map(json.loads, open(out / "manifests/variants.jsonl"))
          if v["variant_type"] == a.type and (v["status"] == "ok" or (a.include_failed and v["status"] == "failed_qc"))
          and v["path"]
          and (not a.axis or v["axis"] == a.axis) and (not a.strength or v["strength"] == a.strength)]

    if a.filter:  # QC 결과로 골라 보기 (사람 검토용)
        def keep(v):
            q = v.get("qc") or {}
            retried = v["status"] == "ok" and bool(q.get("attempts"))
            flagged = v["status"] == "ok" and "low_structure" in (q.get("flags") or [])
            return {"retried": retried, "flagged": flagged, "failed": v["status"] == "failed_qc",
                    "review": retried or flagged}[a.filter]
        if a.filter == "failed":
            a.include_failed = True
        vs = [v for v in map(json.loads, open(out / "manifests/variants.jsonl"))
              if v["variant_type"] == a.type and v["path"] and keep(v)
              and (not a.axis or v["axis"] == a.axis) and (not a.strength or v["strength"] == a.strength)]

    # 컷 → {태그: 변형}
    by_cut: dict[str, dict[str, dict]] = {}
    for v in vs:
        tag = (re.search(r"-s\d+-?(.*)$", v["variant_id"]) or [None, ""])[1]
        if a.tags is not None and tag not in a.tags:
            continue
        key = v["original_cut_id"] if not a.filter else f'{v["original_cut_id"]}|{v["strength"]}'  # 검토: 변형마다 한 줄
        by_cut.setdefault(key, {})[tag] = v
    tags = a.tags if a.tags is not None else sorted({t for d in by_cut.values() for t in d})
    cuts = list(by_cut)[a.offset: a.offset + a.limit]
    if not cuts:
        sys.exit("비교할 변형이 없습니다.")

    H, gap, head = a.height, 12, 44

    def font(sz):
        try:
            return ImageFont.load_default(size=sz)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()
    f_lab, f_num = font(20), font(64)
    rows, prompts, index = [], {}, {}
    for n, key in enumerate(cuts, a.offset + 1):  # 번호는 전체 기준 (나눠 봐도 겹치지 않음)
        cid = key.split("|")[0]
        o = load_original(cfg, origs[cid])
        s = H / o.height
        ims = [o.resize((max(1, int(o.width * s)), H))]
        for t in tags:
            v = by_cut[key].get(t)
            if v:
                ims.append(Image.open(out / v["path"]).convert("RGB").resize(ims[0].size))
                prompts[t] = (v.get("image_edit") or {}).get("prompt", "")
            else:
                ims.append(Image.new("RGB", ims[0].size, (200, 200, 200)))
        row = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), H + head), "white")
        d, x = ImageDraw.Draw(row), 0
        labels = ["original"]
        for t in tags:
            v = by_cut[key].get(t)
            q = (v or {}).get("qc") or {}
            labels.append(f'{t or "(base)"} {v["strength"]} {v["status"]} seam={q.get("seam_score")} ncc={q.get("structure_ncc")}'
                          f'{" retry" + str(len(q["attempts"]) - 1) if q.get("attempts") else ""}' if v else (t or "(base)"))
        for i, (im, label) in enumerate(zip(ims, labels)):
            row.paste(im, (x, head))
            d.text((x + 4, 8), f"{cid}" if i == 0 else label, fill="black", font=f_lab)
            x += im.width + gap
        # 큰 번호 (검토할 때 이 번호로 reject)
        d.rectangle([0, head, 110, head + 80], fill="yellow")
        d.text((8, head + 6), f"#{n}", fill="black", font=f_num)
        index[n] = [by_cut[key][t]["variant_id"] for t in tags if by_cut[key].get(t)]
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height + gap for r in rows)), (128, 128, 128))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + gap
    name = f"check_{time.strftime('%H%M%S')}.png"
    sheet.save(name)
    import json as _json
    with open(name.replace(".png", ".json"), "w", encoding="utf-8") as fj:  # 번호 → variant_id
        _json.dump(index, fj, ensure_ascii=False, indent=1)
    print("저장:", name, "+", name.replace(".png", ".json"), f"({len(cuts)}컷 × 열 {['original'] + [t or '(base)' for t in tags]})")
    for t, p in prompts.items():
        print(f"\n[{t or '(base)'}] {p}")


if __name__ == "__main__":
    main()
