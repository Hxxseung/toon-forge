"""캐릭터 재구성 방식 비교 실험 (매니페스트는 건드리지 않음).

목표: 포즈·윤곽은 유지, 캐릭터는 새로 그림 (그림체는 달라져도 됨).
같은 컷을 세 방식으로 만들어 한 장으로 비교:
  A whole : 컷 전체를 한 번에 편집 (지금까지의 방식)
  B crop  : 사람마다 잘라서 따로 편집 → 제자리에 붙임 (작은 얼굴도 크게 그려짐, 다른 사람·배경에 영향 없음)
  C sketch: B + 포즈 선 스케치(얼굴 안쪽 선은 지움)를 두 번째 입력으로 → 윤곽을 따라 새 캐릭터를 그림
  D outline: C에서 스케치를 '바깥 윤곽선(실루엣)만'으로 → 얼굴·머리·옷 선을 주지 않아 새로 그리게 함
  E fill   : 원본에서 사람 자리를 회색 실루엣으로 지우고(희미한 큰 선만 남김) 그 자리에 새 캐릭터를 그리게 함
D·E는 SAM(facebook/sam-vit-base)으로 사람 영역을 정밀하게 따서 씀 (실패하면 사각형으로 대체).
  F / G  : 편집 모델 대신 SDXL 애니 모델 + ControlNet(깊이) + 인페인팅.
           사람 영역(SAM)만 완전히 새로 그리되, 원본에서 뽑은 깊이 지도로 자세·윤곽을 따라가게 함.
           F = 느슨하게(0.5), G = 자세를 더 강하게(0.85). Qwen 모델을 쓰지 않음 (3080 10GB에서 바로 동작).

python tools/try_methods.py                                   # 기본 컷 8개
python tools/try_methods.py --cuts w005__ep001__c126 w047__ep003__c047 --methods B C
→ try_<시각>.png (행 = 컷, 열 = 원작 | A | B | C), 개별 이미지는 outputs/try/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replica import cli  # noqa: E402
from replica import manifest as M  # noqa: E402
from replica.config import load_config, stable_seed  # noqa: E402
from replica.image_editors import build_editor  # noqa: E402
from replica.image_variant import auto_design, is_grayscale, protect_text  # noqa: E402
from replica.people import cluster_boxes, load_people, people_mask  # noqa: E402

DEFAULT_CUTS = ["w001__ep003__c030", "w001__ep003__c041", "w005__ep001__c126", "w005__ep001__c061",
                "w047__ep003__c047", "w047__ep003__c068", "w057__ep003__c075", "w057__ep003__c047"]

KEEP = ("Keep exactly the same pose, body outline, size and position in the frame for every person, and the same "
        "number of people. Keep the background unchanged. Draw every face completely, with eyes, nose and mouth. "
        "Do not add people, speech bubbles or text.")
P_WHOLE = "Redraw every person in this image as a completely new character: a different face, {design}. " + KEEP
P_CROP = P_WHOLE
P_SKETCH = ("Image 2 is a line sketch showing the pose and body outline of the people in image 1. Redraw every person "
            "in image 1 as a completely new character that follows the pose and outline of the sketch: a different "
            "face, {design}. " + KEEP)
GRAY = " This is a black-and-white manga panel: keep it black-and-white, do not add color."


def sketch(crop: np.ndarray, faces: list[list[float]]) -> Image.Image:
    """포즈 선 스케치: 흐리게 한 뒤 윤곽선만, 얼굴 안쪽 선은 지움 (새 얼굴을 그리게)."""
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (0, 0), max(1.0, min(crop.shape[:2]) / 250))
    e = cv2.Canny(g, 60, 150)
    for x0, y0, x1, y1 in faces:
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.42, (y1 - y0) * 0.42
        cv2.ellipse(e, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 0, -1)
    e = cv2.dilate(e, np.ones((2, 2), np.uint8))
    return Image.fromarray(255 - e).convert("RGB")


P_OUTLINE = ("Image 2 shows only the outer silhouette outline of the people in image 1. Redraw every person in image 1 "
             "as a completely new character whose body fits exactly inside that outline in the same pose, with a newly "
             "designed face, hairstyle and clothes: {design}. " + KEEP)
P_FILL = ("The flat gray shapes mark where people are; the faint lines inside hint at their pose. Draw a completely new "
          "character filling each gray shape exactly, in that pose, with a newly designed face, hairstyle and clothes: "
          "{design}. Draw every face completely, with eyes, nose and mouth. Keep everything outside the gray shapes "
          "unchanged. One character per gray shape; do not add people, speech bubbles or text.")


class Segmenter:
    """SAM으로 박스 안의 사람 영역을 정밀하게 분리 (CPU). 불러오기 실패하면 사각형 마스크로 대체."""

    def __init__(self, model_id="facebook/sam-vit-base", enabled=True):
        self.model = None
        if not enabled:
            return
        try:
            from transformers import SamModel, SamProcessor
            self.proc = SamProcessor.from_pretrained(model_id)
            self.model = SamModel.from_pretrained(model_id).eval()
        except Exception as e:  # noqa: BLE001
            print(f"[try] SAM을 불러오지 못해 사각형 마스크로 대체합니다 ({type(e).__name__}: {e})")

    def mask(self, img: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        H, W = img.shape[:2]
        m = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            ok = False
            if self.model is not None:
                try:
                    import torch
                    inp = self.proc(Image.fromarray(img), input_boxes=[[[float(x0), float(y0), float(x1), float(y1)]]],
                                    return_tensors="pt")
                    with torch.no_grad():
                        out = self.model(**inp)
                    ms = self.proc.image_processor.post_process_masks(
                        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0][0]
                    best = int(out.iou_scores[0, 0].argmax())
                    mm = ms[best].numpy().astype(bool)
                    if 0.05 < mm[int(y0):int(y1), int(x0):int(x1)].mean() < 0.98:  # 너무 작거나 박스 전체면 실패로 봄
                        m |= mm
                        ok = True
                except Exception as e:  # noqa: BLE001
                    print(f"[try] SAM 실패 → 사각형 ({type(e).__name__}: {e})")
            if not ok:
                m[int(y0):int(y1), int(x0):int(x1)] = True
        return m


def soft(mask: np.ndarray, grow: int, feather: int = 9) -> np.ndarray:
    m = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(np.float32)
    k = 2 * feather + 1
    return np.clip(np.maximum(m, cv2.GaussianBlur(cv2.dilate(m, np.ones((k, k), np.uint8)), (k, k), 0)), 0, 1)


def outline_image(mask: np.ndarray) -> Image.Image:
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
    return Image.fromarray(np.where(edge[..., None] > 0, 0, 255).astype(np.uint8).repeat(3, axis=2))


def fill_image(crop: np.ndarray, mask: np.ndarray) -> Image.Image:
    """사람 자리를 회색으로 지움 + 아주 흐린 큰 선(팔다리 방향 힌트)과 윤곽선만 남김 → 원래 얼굴·옷 정보는 사라짐."""
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (0, 0), max(2.0, min(crop.shape[:2]) / 90))
    hint = cv2.dilate(cv2.Canny(g, 40, 110), np.ones((2, 2), np.uint8)) > 0
    out = crop.copy()
    out[mask] = 190
    out[mask & hint] = 150
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = 60
    return Image.fromarray(out)


def edit_seg(editor, seg, orig: Image.Image, boxes: list[list], prompt: str, seed: int, pad: float, mode: str):
    """D(outline) / E(fill): 사람 박스 묶음마다 잘라서 SAM 마스크 기반으로 편집·합성."""
    o = np.asarray(orig).copy()
    out = o.copy()
    H, W = o.shape[:2]
    persons = [b[:4] for b in boxes if "face" not in str(b[5] if len(b) > 5 else "")] or [b[:4] for b in boxes]
    for i, (x0, y0, x1, y1) in enumerate(cluster_boxes(boxes)):
        pw, ph = (x1 - x0) * (pad + 0.15), (y1 - y0) * (pad + 0.15)
        cx0, cy0, cx1, cy1 = int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))
        if cx1 - cx0 < 32 or cy1 - cy0 < 32:
            continue
        crop = o[cy0:cy1, cx0:cx1]
        inside = [[p[0] - cx0, p[1] - cy0, p[2] - cx0, p[3] - cy0] for p in persons
                  if p[0] >= x0 - 5 and p[2] <= x1 + 5 and p[1] >= y0 - 5 and p[3] <= y1 + 5] \
            or [[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]]
        mask = seg.mask(crop, inside)
        grow = max(4, int(min(crop.shape[:2]) * 0.03))  # 새 머리 모양이 조금 커져도 들어가게
        if mode == "D":
            e = editor.edit(Image.fromarray(crop), prompt, seed + i, refs=[outline_image(mask)])
        else:
            big = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
            e = editor.edit(fill_image(crop, big), prompt, seed + i)
        e = np.asarray(e.convert("RGB").resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS))
        m = soft(mask, grow * 2)[..., None]
        out[cy0:cy1, cx0:cx1] = (out[cy0:cy1, cx0:cx1] * (1 - m) + e * m).astype(np.uint8)
    return out


HAIR_COLORS = ["silver", "pink", "blonde", "red", "blue", "green", "purple", "orange", "white", "black"]
HAIR_STYLES = ["short hair", "long hair", "ponytail", "wavy hair", "bob cut", "braid", "hair bun", "spiked hair"]
OUTFITS = ["hoodie", "school uniform", "leather jacket", "kimono", "armor", "suit", "sweater", "cloak", "t-shirt and jeans"]
NEG = ("multiple views, character sheet, reference sheet, collage, comic panels, text, speech bubble, watermark, "
       "signature, lowres, bad anatomy, bad hands, extra limbs, blurry")


def snap(w: int, h: int, area: int = 1024 * 1024, mult: int = 64) -> tuple[int, int]:
    sc = (area / (w * h)) ** 0.5
    return max(mult, round(w * sc / mult) * mult), max(mult, round(h * sc / mult) * mult)


class SdxlControlInpaint:
    """SDXL(애니 체크포인트) + ControlNet depth + 인페인팅. 마스크 안은 새로 그리고, 깊이 지도로 자세를 유지."""

    def __init__(self, base: str, controlnet: str, depth_model: str):
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
        from transformers import pipeline as hf_pipeline
        cn = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(base, controlnet=cn, torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.depth = hf_pipeline("depth-estimation", model=depth_model, device="cpu")

    def __call__(self, crop: Image.Image, mask: np.ndarray, prompt: str, seed: int, scale: float, steps: int = 28):
        import torch
        w, h = snap(*crop.size)
        img = crop.resize((w, h), Image.LANCZOS)
        mk = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        depth = self.depth(img)["depth"].convert("RGB").resize((w, h))
        return self.pipe(prompt=prompt, negative_prompt=NEG, image=img, mask_image=mk, control_image=depth,
                         width=w, height=h, strength=0.99, num_inference_steps=steps, guidance_scale=6.0,
                         controlnet_conditioning_scale=scale,
                         generator=torch.Generator("cpu").manual_seed(seed)).images[0]


def edit_sdxl(gen, seg, orig: Image.Image, boxes: list[list], seed: int, pad: float, scale: float, gray: bool, work_id: str):
    o = np.asarray(orig).copy()
    out = o.copy()
    H, W = o.shape[:2]
    persons = [b[:4] for b in boxes if "face" not in str(b[5] if len(b) > 5 else "")] or [b[:4] for b in boxes]
    prompts = []
    for i, (x0, y0, x1, y1) in enumerate(cluster_boxes(boxes)):
        pw, ph = (x1 - x0) * (pad + 0.15), (y1 - y0) * (pad + 0.15)
        cx0, cy0, cx1, cy1 = int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))
        if cx1 - cx0 < 32 or cy1 - cy0 < 32:
            continue
        crop = o[cy0:cy1, cx0:cx1]
        inside = [[p[0] - cx0, p[1] - cy0, p[2] - cx0, p[3] - cy0] for p in persons
                  if p[0] >= x0 - 5 and p[2] <= x1 + 5 and p[1] >= y0 - 5 and p[3] <= y1 + 5] \
            or [[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]]
        mask = seg.mask(crop, inside)
        grow = max(4, int(min(crop.shape[:2]) * 0.03))
        big = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
        rng = np.random.default_rng(stable_seed("sdxl-design", work_id, i))   # 작품·인물 순서별 고정 디자인
        design = f"{rng.choice(HAIR_COLORS)} hair, {rng.choice(HAIR_STYLES)}, {rng.choice(OUTFITS)}"
        count = "solo, " if len(inside) == 1 else ""
        prompt = (f"{count}{design}, " + ("monochrome, greyscale, screentone, manga, " if gray else "")
                  + "masterpiece, high score, great score, absurdres")
        prompts.append(prompt)
        e = gen(Image.fromarray(crop), big, prompt, seed + i, scale)
        e = np.asarray(e.convert("RGB").resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS))
        m = soft(mask, grow)[..., None]
        out[cy0:cy1, cx0:cx1] = (out[cy0:cy1, cx0:cx1] * (1 - m) + e * m).astype(np.uint8)
    return out, prompts


def edit_crops(editor, orig: Image.Image, boxes: list[list], prompt: str, seed: int, pad: float, use_sketch: bool):
    o = np.asarray(orig).copy()
    out = o.copy()
    H, W = o.shape[:2]
    faces = [b for b in boxes if "face" in str(b[5] if len(b) > 5 else "")]
    for i, (x0, y0, x1, y1) in enumerate(cluster_boxes(boxes)):
        pw, ph = (x1 - x0) * (pad + 0.15), (y1 - y0) * (pad + 0.15)   # 마스크보다 조금 넓게 잘라 주변 맥락 포함
        cx0, cy0, cx1, cy1 = int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))
        if cx1 - cx0 < 32 or cy1 - cy0 < 32:
            continue
        crop = o[cy0:cy1, cx0:cx1]
        refs = None
        if use_sketch:
            fs = [[f[0] - cx0, f[1] - cy0, f[2] - cx0, f[3] - cy0] for f in faces
                  if f[0] >= x0 - 5 and f[2] <= x1 + 5 and f[1] >= y0 - 5 and f[3] <= y1 + 5]
            refs = [sketch(crop, fs)]
        e = editor.edit(Image.fromarray(crop), prompt, seed + i, refs=refs).convert("RGB")
        e = np.asarray(e.resize((cx1 - cx0, cy1 - cy0), Image.LANCZOS))
        m = people_mask((H, W), [[x0, y0, x1, y1]], pad)[cy0:cy1, cx0:cx1, None]
        out[cy0:cy1, cx0:cx1] = (out[cy0:cy1, cx0:cx1] * (1 - m) + e * m).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--cuts", nargs="*", default=DEFAULT_CUTS)
    ap.add_argument("--methods", nargs="*", default=["F", "G"], choices=["A", "B", "C", "D", "E", "F", "G"])
    ap.add_argument("--sdxl-base", default="cagliostrolab/animagine-xl-4.0", help="F·G용 SDXL 체크포인트")
    ap.add_argument("--sdxl-controlnet", default="diffusers/controlnet-depth-sdxl-1.0")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--no-sam", action="store_true", help="SAM 없이 사각형 마스크로 (비교·디버그용)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--height", type=int, default=520)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_root = Path(cfg["out_root"])
    origs = M.originals_manifest(out_root).by_id()
    editor = build_editor(cfg["image"]["backends"][a.backend or cfg["image"]["backend"]]) \
        if set(a.methods) & set("ABCDE") else None
    gen = SdxlControlInpaint(a.sdxl_base, a.sdxl_controlnet, a.depth_model) if set(a.methods) & set("FG") else None
    pad = cfg["image"].get("person_mask_pad", 0.2)
    pools = cfg["image"]["templates"]["auto_pools"]
    (out_root / "try").mkdir(parents=True, exist_ok=True)
    seg = Segmenter(enabled=not a.no_sam) if set(a.methods) & set("DEFG") else None

    rows = []
    for cid in a.cuts:
        if cid not in origs:
            print("없는 컷:", cid)
            continue
        r = origs[cid]
        ppl = load_people(out_root / f"people/{cid}.json")
        if not ppl or not ppl["boxes"]:
            print("사람 없음:", cid)
            continue
        orig = cli.load_original(cfg, r)
        o = np.asarray(orig)
        gray = is_grayscale(orig)
        design = auto_design(r["work_id"], "strong", gray, pools)
        seed = stable_seed(cid, "try")
        blocks = cli._blocks(cfg, r) or []
        res = {}
        for mth in a.methods:
            t0 = time.time()
            if mth in ("F", "G"):
                e, ps = edit_sdxl(gen, seg, orig, ppl["boxes"], seed, pad, {"F": 0.5, "G": 0.85}[mth], gray, r["work_id"])
                e = protect_text(e, o, blocks)
                Image.fromarray(e).save(out_root / "try" / f"{cid}__{mth}.png")
                res[mth] = e
                print(f"[try] {cid} {mth} {time.time() - t0:.0f}s | {ps}", flush=True)
                continue
            tpl = {"A": P_WHOLE, "B": P_CROP, "C": P_SKETCH, "D": P_OUTLINE, "E": P_FILL}[mth]
            prompt = tpl.format(design=design) + (GRAY if gray else "")
            if mth == "A":
                e = np.asarray(editor.edit(orig, prompt, seed).convert("RGB").resize(orig.size, Image.LANCZOS))
                m = people_mask(o.shape[:2], ppl["boxes"], pad)[..., None]
                e = (o * (1 - m) + e * m).astype(np.uint8)
            elif mth in ("D", "E"):
                e = edit_seg(editor, seg, orig, ppl["boxes"], prompt, seed, pad, mth)
            else:
                e = edit_crops(editor, orig, ppl["boxes"], prompt, seed, pad, use_sketch=(mth == "C"))
            e = protect_text(e, o, blocks)
            Image.fromarray(e).save(out_root / "try" / f"{cid}__{mth}.png")
            res[mth] = e
            print(f"[try] {cid} {mth} {time.time() - t0:.0f}s", flush=True)
        s = a.height / orig.height
        ims = [orig.resize((max(1, int(orig.width * s)), a.height))] + \
              [Image.fromarray(res[m]).resize((max(1, int(orig.width * s)), a.height)) for m in a.methods]
        row = Image.new("RGB", (sum(i.width for i in ims) + 12 * (len(ims) - 1), a.height + 40), "white")
        d, x = ImageDraw.Draw(row), 0
        try:
            font = ImageFont.load_default(size=22)
        except TypeError:
            font = ImageFont.load_default()
        names = {"A": "A whole", "B": "B crop", "C": "C crop+sketch", "D": "D outline only", "E": "E gray fill",
                 "F": "F sdxl depth 0.5", "G": "G sdxl depth 0.85"}
        for im, lab in zip(ims, [cid] + [names[m] for m in a.methods]):
            row.paste(im, (x, 40))
            d.text((x + 4, 8), lab, fill="black", font=font)
            x += im.width + 12
        rows.append(row)
    if not rows:
        sys.exit("만든 결과가 없습니다.")
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 12 for r in rows)), (128, 128, 128))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 12
    name = f"try_{time.strftime('%H%M%S')}.png"
    sheet.save(name)
    print("저장:", name)


if __name__ == "__main__":
    main()