"""GPU·OCR 없이 전체 흐름 확인: 합성 회차 → cut → (가짜 OCR JSON) → text / image(dummy) / both → export.

python tests/test_synthetic.py <font.ttf>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replica import cli  # noqa: E402
from replica import manifest as M  # noqa: E402
from replica.korean import substitute_entities  # noqa: E402
from replica.ocr import Block, Line, save_blocks  # noqa: E402

FONT = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
TMP = ROOT / "tests" / "_tmp"


def make_episode(d: Path):
    """흰 여백(60px)으로 나뉜 컷 3개짜리 세로 이미지를 2조각으로 저장. 말풍선 텍스트 위치를 반환."""
    W, font = 600, ImageFont.truetype(FONT, 26)
    big = ImageFont.truetype(FONT, 56)
    H = 60 + 3 * (400 + 60)
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    texts = [["카일이 왔어!", "천명산으로 가자"], ["세라야, 흑막검을 줘"], ["쾅"]]
    boxes = []
    for i in range(3):
        y = 60 + i * 460
        dr.rectangle([20, y, W - 20, y + 400], fill=(90 + 40 * i, 140, 200 - 30 * i), outline="black", width=4)
        dr.ellipse([60 + 20 * i, y + 40, 60 + 20 * i + 120, y + 360], fill=(240, 200, 170))  # 인물 대충
        cut_boxes = []
        if i < 2:
            dr.ellipse([250, y + 30, 560, y + 190], fill="white", outline="black", width=3)
            ty = y + 70
            for t in texts[i]:
                w = dr.textlength(t, font=font)
                x = 405 - w / 2
                bb = dr.textbbox((x, ty), t, font=font)
                dr.text((x, ty), t, font=font, fill="black")
                cut_boxes.append((t, tuple(int(v) for v in bb)))
                ty += 38
        else:
            bb = dr.textbbox((350, y + 250), "쾅", font=big)
            dr.text((350, y + 250), "쾅", font=big, fill=(255, 230, 0), stroke_width=3, stroke_fill="black")
            cut_boxes.append(("쾅", tuple(int(v) for v in bb)))
        boxes.append(cut_boxes)
    d.mkdir(parents=True, exist_ok=True)
    img.crop((0, 0, W, 700)).save(d / "001.jpg", quality=95)
    img.crop((0, 700, W, H)).save(d / "002.jpg", quality=95)
    return boxes


def main():
    # 조사 보정 단위 테스트
    m = {"카일": "레온", "세라": "아리아", "천명산": "백운산", "흑막검": "월광검"}
    for s, want in [("카일이 왔어!", "레온이 왔어!"), ("세라야, 흑막검을 줘", "아리아야, 월광검을 줘"),
                    ("천명산으로 가자", "백운산으로 가자"), ("카일을 봐", "레온을 봐"), ("세라는 어디?", "아리아는 어디?"),
                    ("카닐이 왔다", "레온이 왔다")]:
        got, rules = substitute_entities(s, m)
        print(f"  {s} → {got}  {rules}")
        assert got == want, (got, want)
    # '카일이'의 '이'는 받침 있는 레온 → '이' 유지 / '세라야'는 아리아(받침 없음) → '야'

    shutil.rmtree(TMP, ignore_errors=True)
    boxes = make_episode(TMP / "data/originals/demo/ep001")
    (TMP / "configs/works").mkdir(parents=True)
    shutil.copy(ROOT / "configs/works/_example.yaml", TMP / "configs/works/demo.yaml")
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    cfg["text"]["fonts"] = {k: FONT for k in ("dialogue", "sfx", "credit")}
    cfg["image"]["backend"] = "dummy"
    (TMP / "configs/test.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    c = ["-c", str(TMP / "configs/test.yaml")]

    cli.main(["cut", *c])
    origs = list(M.originals_manifest(TMP / "outputs").read())
    assert len(origs) == 3, len(origs)

    # 가짜 OCR (실제로는 `cli ocr`가 PaddleOCR로 생성). 컷 좌표로 변환.
    for r, cb in zip(origs, boxes):
        y0 = r["y_range"][0]
        lines = [Line(t, (b[0], b[1] - y0, b[2], b[3] - y0), 0.99) for t, b in cb]
        if len(lines) == 1 and lines[0].text == "쾅":
            blocks = [Block(0, "sfx", "쾅", lines[0].bbox, lines)]
        else:
            bb = (min(l.bbox[0] for l in lines), min(l.bbox[1] for l in lines),
                  max(l.bbox[2] for l in lines), max(l.bbox[3] for l in lines))
            blocks = [Block(0, "dialogue", "\n".join(l.text for l in lines), bb, lines)]
        save_blocks(TMP / "outputs" / r["ocr_path"], r["cut_id"], blocks)

    cli.main(["text", *c])
    cli.main(["image", *c, "--axis", "character", "--strength", "weak", "strong"])
    cli.main(["both", *c, "--axis", "character", "background", "--strength", "medium"])
    cli.main(["image", *c, "--axis", "character", "--strength", "weak"])  # 재실행 → 건너뛰어야 함
    cli.main(["export", *c])

    vs = list(M.variants_manifest(TMP / "outputs").read())
    by = {}
    for v in vs:
        by.setdefault((v["variant_type"], v["status"]), 0)
        by[(v["variant_type"], v["status"])] += 1
    print("variants:", by)
    assert by[("text", "ok")] == 3 and by[("image", "ok")] == 3 * 2 + 3 * 2 and by[("both", "ok")] == 3 * 2
    assert len({v["variant_id"] for v in vs}) == len(vs)
    for v in vs:
        if v["path"]:
            assert (TMP / "outputs" / v["path"]).exists()
        if v["variant_type"] == "both":
            assert v["parent_variant_id"] in {x["variant_id"] for x in vs}
    t0 = next(v for v in vs if v["variant_type"] == "text" and v["cut_index"] == 0)
    print(json.dumps(t0["text_edit"]["changes"], ensure_ascii=False, indent=1))
    print("OK")


if __name__ == "__main__":
    main()
