"""생성형 이미지 편집 백엔드.

모두 같은 인터페이스: editor.edit(image, prompt, seed, refs=None) -> PIL.Image
- qwen       : Qwen/Qwen-Image-Edit (Apache-2.0)
- qwen_plus  : Qwen/Qwen-Image-Edit-2509 (여러 장 입력 → 캐릭터 참조 이미지 사용 가능)
- flux_kontext : black-forest-labs/FLUX.1-Kontext-dev (비상업 라이선스! 공개 데이터셋엔 주의)
- dummy      : 모델 없이 색만 바꾸는 테스트용
모델 버전은 바뀌므로 model_id는 config에서 덮어쓸 수 있게 함.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def snap_size(w: int, h: int, target_area: int = 1024 * 1024, mult: int = 16) -> tuple[int, int]:
    """비율 유지, 면적 ≈ target_area, 16의 배수. 세로로 긴 컷도 비율 그대로."""
    s = (target_area / (w * h)) ** 0.5
    return max(mult, round(w * s / mult) * mult), max(mult, round(h * s / mult) * mult)


class BaseEditor:
    name = "base"
    model_id = ""
    supports_refs = False
    params: dict[str, Any] = {}

    def edit(self, image: Image.Image, prompt: str, seed: int, refs: list[Image.Image] | None = None) -> Image.Image:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, **self.params}


class DummyEditor(BaseEditor):
    name = "dummy"
    model_id = "none"

    def edit(self, image, prompt, seed, refs=None):
        rng = np.random.default_rng(seed)
        a = np.asarray(image.convert("RGB")).astype(np.int16)
        a = np.clip(a[..., rng.permutation(3)] + rng.integers(-20, 20, 3), 0, 255).astype(np.uint8)
        return Image.fromarray(a)


class _DiffusersEditor(BaseEditor):
    pipe_cls = ""

    def __init__(self, model_id: str, steps: int, guidance: float, device: str = "cuda",
                 cpu_offload: bool = False, target_area: int = 1024 * 1024, **extra):
        import torch
        import diffusers
        cls = getattr(diffusers, self.pipe_cls)
        self.pipe = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.model_id, self.target_area, self.extra = model_id, target_area, extra
        self.params = {"steps": steps, "guidance": guidance, "target_area": target_area, **extra}

    def _gen(self, seed):
        import torch
        return torch.Generator(device="cpu").manual_seed(seed)


class QwenEditor(_DiffusersEditor):
    name, pipe_cls = "qwen", "QwenImageEditPipeline"

    def edit(self, image, prompt, seed, refs=None):
        w, h = snap_size(*image.size, self.target_area)
        out = self.pipe(image=image.convert("RGB").resize((w, h), Image.LANCZOS), prompt=prompt,
                        negative_prompt=self.extra.get("negative_prompt", " "),
                        true_cfg_scale=self.params["guidance"], num_inference_steps=self.params["steps"],
                        generator=self._gen(seed)).images[0]
        return out


class QwenPlusEditor(_DiffusersEditor):
    name, pipe_cls, supports_refs = "qwen_plus", "QwenImageEditPlusPipeline", True

    def edit(self, image, prompt, seed, refs=None):
        w, h = snap_size(*image.size, self.target_area)
        imgs = [image.convert("RGB").resize((w, h), Image.LANCZOS)] + [r.convert("RGB") for r in (refs or [])]
        return self.pipe(image=imgs if len(imgs) > 1 else imgs[0], prompt=prompt,
                         negative_prompt=self.extra.get("negative_prompt", " "),
                         true_cfg_scale=self.params["guidance"], num_inference_steps=self.params["steps"],
                         generator=self._gen(seed)).images[0]


class FluxKontextEditor(_DiffusersEditor):
    name, pipe_cls = "flux_kontext", "FluxKontextPipeline"

    def edit(self, image, prompt, seed, refs=None):
        w, h = snap_size(*image.size, self.target_area)
        return self.pipe(image=image.convert("RGB").resize((w, h), Image.LANCZOS), prompt=prompt,
                         guidance_scale=self.params["guidance"], num_inference_steps=self.params["steps"],
                         width=w, height=h, generator=self._gen(seed)).images[0]


class NunchakuQwenEditor(BaseEditor):
    """Qwen-Image-Edit-2509 + Nunchaku 4-bit(INT4) 트랜스포머 — RTX 30/40 (VRAM 10~12GB)용.

    VRAM < offload_threshold_gb 이면 트랜스포머 블록 단위 오프로드 + 나머지 순차 오프로드
    (Nunchaku 공식 예제 방식). 시스템 RAM은 64GB 이상 권장.
    """
    name, supports_refs = "qwen_nunchaku", True

    def __init__(self, base_model_id: str, transformer_repo: str, transformer_file: str, steps: int,
                 guidance: float, device: str = "cuda", target_area: int = 1024 * 1024,
                 offload_threshold_gb: float = 18, num_blocks_on_gpu: int = 1, use_pin_memory: bool = False,
                 **extra):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from nunchaku import NunchakuQwenImageTransformer2DModel
        torch.cuda.set_device(torch.device(device))
        tr = NunchakuQwenImageTransformer2DModel.from_pretrained(f"{transformer_repo}/{transformer_file}")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(base_model_id, transformer=tr, torch_dtype=torch.bfloat16)
        vram = torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30
        if vram > offload_threshold_gb:
            self.pipe.enable_model_cpu_offload(device=device)
            mode = "model_offload"
        else:
            # pin memory: CPU↔GPU 전송이 빨라지지만 RAM을 고정으로 더 씀 (프로세스 2개면 RAM 여유 필요)
            tr.set_offload(True, use_pin_memory=use_pin_memory, num_blocks_on_gpu=num_blocks_on_gpu)
            self.pipe._exclude_from_cpu_offload.append("transformer")
            self.pipe.enable_sequential_cpu_offload(device=device)
            mode = f"block_offload(num_blocks_on_gpu={num_blocks_on_gpu}, pin={use_pin_memory})"
        self.pipe.set_progress_bar_config(disable=True)
        self.model_id = f"{base_model_id} + {transformer_repo}/{transformer_file}"
        self.target_area, self.extra = target_area, extra
        self.params = {"steps": steps, "guidance": guidance, "target_area": target_area,
                       "quant": "nunchaku-int4", "offload": mode, **extra}

    def edit(self, image, prompt, seed, refs=None):
        import torch
        w, h = snap_size(*image.size, self.target_area)
        imgs = [image.convert("RGB").resize((w, h), Image.LANCZOS)] + [r.convert("RGB") for r in (refs or [])]
        return self.pipe(image=imgs if len(imgs) > 1 else imgs[0], prompt=prompt,
                         negative_prompt=self.extra.get("negative_prompt", " "),
                         true_cfg_scale=self.params["guidance"], num_inference_steps=self.params["steps"],
                         generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]


EDITORS = {"dummy": DummyEditor, "qwen": QwenEditor, "qwen_plus": QwenPlusEditor,
           "flux_kontext": FluxKontextEditor, "qwen_nunchaku": NunchakuQwenEditor}


def build_editor(backend_cfg: dict[str, Any], device: str | None = None) -> BaseEditor:
    cfg = dict(backend_cfg)
    name = cfg.pop("name")
    if name == "dummy":
        return DummyEditor()
    if device:
        cfg["device"] = device
    return EDITORS[name](**cfg)