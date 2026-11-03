"""'[연재] 작품명 N화 - 작가' 폴더들 → data/originals/<work_id>/ep###/ 로 심볼릭 링크 (원본은 그대로).

작품 ID는 data/works.csv 에 고정 저장 (다시 실행하거나 작품이 늘어도 기존 ID 유지).

python tools/prepare_data.py --src data                    # 전체 목록만 출력 (링크 안 만듦)
python tools/prepare_data.py --src data --pick 1 5 9 12 20 # 목록 번호로 골라서 링크
python tools/prepare_data.py --src data --pick all         # 전부 링크
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

# [출처] (선택) 작품명 [웹툰](선택) N화|N권 - 작가 ( - 복사본)(선택)
PAT = re.compile(r"^(?:\[(?P<src>[^\]]+)\]\s*)?(?P<title>.+?)(?:\s*\[(?P<tag>[^\]]+)\])?\s+"
                 r"(?P<ep>\d+)(?P<unit>화|권)\s*-\s*(?P<author>.+?)(?P<copy>\s*-\s*복사본)?$")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def norm(s: str) -> str:
    """Mac에서 온 파일명은 한글이 자모로 분해(NFD)돼 있음 → NFC로 합치고 특수 공백 정리."""
    return re.sub(r"[ 　]", " ", unicodedata.normalize("NFC", s)).strip()


def scan(src: Path):
    eps, bad = [], []
    for d in sorted((p for p in src.iterdir() if p.is_dir()), key=lambda p: norm(p.name)):
        name = norm(d.name)
        m = PAT.match(name)
        if not m:
            if name != "originals":
                bad.append(name)
            continue
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        g = m.groupdict()
        eps.append({"src": g["src"] or g["tag"] or "", "title": g["title"].strip(), "author": g["author"].strip(),
                    "ep": int(g["ep"]), "unit": g["unit"], "copy": bool(g["copy"]), "dir": d, "n_images": n})
    return eps, bad


def load_works(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r["title"], r["author"]): r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--dst", default=None, help="기본: <src>/originals")
    ap.add_argument("--pick", nargs="*", help="목록 번호들 또는 all")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    dst = Path(a.dst).resolve() if a.dst else src / "originals"

    eps, bad = scan(src)
    works_csv = src / "works.csv"
    works = load_works(works_csv)
    for e in eps:  # 새 작품에만 다음 번호 부여
        key = (e["title"], e["author"])
        if key not in works:
            works[key] = {"work_id": f"w{len(works) + 1:03d}", "title": e["title"],
                          "author": e["author"], "source": e["src"]}
        e["work_id"] = works[key]["work_id"]
    with open(works_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["work_id", "title", "author", "source"])
        w.writeheader()
        w.writerows(sorted(works.values(), key=lambda r: r["work_id"]))

    for i, e in enumerate(eps, 1):
        flags = (" ⚠복사본" if e["copy"] else "") + (" ⚠권 단위" if e["unit"] == "권" else "") + \
                (" ⚠이미지 없음" if e["n_images"] == 0 else "")
        src_tag = f"[{e['src']}] " if e["src"] else ""
        print(f"{i:3d}  {e['work_id']}/ep{e['ep']:03d}  {src_tag}{e['title']} {e['ep']}{e['unit']} - {e['author']}"
              f"  ({e['n_images']}장){flags}")
    if bad:
        print("\n[형식이 달라 건너뜀]", *bad, sep="\n  ")

    if not a.pick:
        print(f"\n링크를 만들려면 --pick 번호들 (또는 --pick all). 작품 ID 표: {works_csv}")
        return
    if a.pick == ["all"]:
        chosen = eps
    else:
        bad_k = [k for k in a.pick if not k.isdigit() or not 1 <= int(k) <= len(eps)]
        if bad_k:
            raise SystemExit(f"잘못된 번호: {bad_k} (1~{len(eps)} 사이로 골라 주세요)")
        chosen = [eps[int(k) - 1] for k in a.pick]
    for e in chosen:
        link = dst / e["work_id"] / f"ep{e['ep']:03d}"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            print(f"이미 있음: {link}")
            continue
        link.symlink_to(e["dir"], target_is_directory=True)
        print(f"링크: {link.relative_to(src)} → {e['dir'].name}")


if __name__ == "__main__":
    main()