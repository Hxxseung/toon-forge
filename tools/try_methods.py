"""캐릭터 변형 실험: F / G (SDXL + ControlNet depth + SAM 인페인팅). 매니페스트는 건드리지 않음.

목표: 포즈·윤곽은 유지, 캐릭터(얼굴·머리·옷)는 새로 그림 (그림체는 달라져도 됨).
  사람 영역(SAM)만 SDXL(Animagine XL 4.0)로 새로 그리되, 원본에서 뽑은 깊이 지도(ControlNet depth)로 자세를 따라가게 함.
  F = 자세를 느슨하게(F_SCALE), G = 자세를 강하게(G_SCALE). 3080 10GB에서 동작.
A~E(Qwen 편집 방식 비교)는 데이터셋에 쓰지 않아 뺐음 → git 태그 exp-qwen-AtoE (커밋 2c5bff6)에 남아 있음.

python tools/try_methods.py                                     # 기본 컷 8개, F·G
python tools/try_methods.py --methods G --cuts w047__ep003__c047
F_SCALE=0.6 G_SCALE=0.75 python tools/try_methods.py            # 값만 바꿔 바로 실행 (코드 수정 없이)
→ try_<시각>.png (행 = 컷, 열 = 원작 | F | G), 개별 이미지는 outputs/try/<컷>__F.png
"""
from __future__ import annotations

import argparse
import os
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
from replica.image_variant import is_grayscale, protect_text  # noqa: E402
from replica.people import cluster_boxes, load_people  # noqa: E402

# ───────────── 설정: 여기만 고치면 바로 적용 (같은 이름의 환경변수가 있으면 그 값이 우선) ─────────────
F_SCALE = float(os.environ.get("F_SCALE", 0.5))       # F: ControlNet depth 세기. 낮을수록 자세가 자유로움
G_SCALE = float(os.environ.get("G_SCALE", 0.85))      # G: ControlNet depth 세기. 높을수록 원본 자세에 가까움
STRENGTH = float(os.environ.get("FG_STRENGTH", 0.99))  # 인페인팅 강도 (1에 가까울수록 사람 영역을 완전히 새로 그림)
STEPS = int(os.environ.get("FG_STEPS", 28))            # 생성 스텝 수
GUIDANCE = float(os.environ.get("FG_GUIDANCE", 6.0))   # 프롬프트를 따르는 정도
# ─────────────────────────────────────────────────────────────────────────────────────────────────────

DEFAULT_CUTS = ["w001__ep003__c030", "w001__ep003__c041", "w005__ep001__c126", "w005__ep001__c061",
                "w047__ep003__c047", "w047__ep003__c068", "w057__ep003__c075", "w057__ep003__c047"]


def scales() -> dict[str, float]:
    return {"F": F_SCALE, "G": G_SCALE}


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

    def __call__(self, crop: Image.Image, mask: np.ndarray, prompt: str, seed: int, scale: float, steps: int | None = None):
        steps = steps or STEPS
        import torch
        w, h = snap(*crop.size)
        img = crop.resize((w, h), Image.LANCZOS)
        mk = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        depth = self.depth(img)["depth"].convert("RGB").resize((w, h))
        return self.pipe(prompt=prompt, negative_prompt=NEG, image=img, mask_image=mk, control_image=depth,
                         width=w, height=h, strength=STRENGTH, num_inference_steps=steps, guidance_scale=GUIDANCE,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--cuts", nargs="*", default=DEFAULT_CUTS)
    ap.add_argument("--methods", nargs="*", default=["F", "G"], choices=["F", "G"])
    ap.add_argument("--sdxl-base", default="cagliostrolab/animagine-xl-4.0", help="F·G용 SDXL 체크포인트")
    ap.add_argument("--sdxl-controlnet", default="diffusers/controlnet-depth-sdxl-1.0")
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--no-sam", action="store_true", help="SAM 없이 사각형 마스크로 (비교·디버그용)")
    ap.add_argument("--height", type=int, default=520)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out_root = Path(cfg["out_root"])
    origs = M.originals_manifest(out_root).by_id()
    gen = SdxlControlInpaint(a.sdxl_base, a.sdxl_controlnet, a.depth_model)
    pad = cfg["image"].get("person_mask_pad", 0.2)
    (out_root / "try").mkdir(parents=True, exist_ok=True)
    seg = Segmenter(enabled=not a.no_sam)
    print(f"[try] F={F_SCALE} G={G_SCALE} strength={STRENGTH} steps={STEPS} guidance={GUIDANCE}", flush=True)

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
        seed = stable_seed(cid, "try")
        blocks = cli._blocks(cfg, r) or []
        res = {}
        for mth in a.methods:
            t0 = time.time()
            e, ps = edit_sdxl(gen, seg, orig, ppl["boxes"], seed, pad, scales()[mth], gray, r["work_id"])
            e = protect_text(e, o, blocks)
            Image.fromarray(e).save(out_root / "try" / f"{cid}__{mth}.png")
            res[mth] = e
            print(f"[try] {cid} {mth} {time.time() - t0:.0f}s | {ps}", flush=True)
        s = a.height / orig.height
        ims = [orig.resize((max(1, int(orig.width * s)), a.height))] + \
              [Image.fromarray(res[m]).resize((max(1, int(orig.width * s)), a.height)) for m in a.methods]
        row = Image.new("RGB", (sum(i.width for i in ims) + 12 * (len(ims) - 1), a.height + 40), "white")
        d, x = ImageDraw.Draw(row), 0
        try:
            font = ImageFont.load_default(size=22)
        except TypeError:
            font = ImageFont.load_default()
        names = {m: f"{m} sdxl depth {v}" for m, v in scales().items()}
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