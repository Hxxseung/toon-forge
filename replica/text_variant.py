"""텍스트만 변형 — 대사(이름·지명·아이템 치환, 선택적 의역), 효과음, 크레딧.

작품 단위 표(configs/works/<work_id>.yaml)를 고정해 쓰므로, 한 작품 안의 모든 컷에서
같은 이름은 항상 같은 새 이름으로 바뀐다 (실제 2차창작과 같은 일관성).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .korean import substitute_entities
from .manifest import TextChange
from .ocr import Block
from .text_render import draw_block, erase_block


def flat_entities(table: dict[str, Any]) -> dict[str, str]:
    ents = table.get("entities", {}) or {}
    out: dict[str, str] = {}
    for group in ents.values():          # person / place / item / org ...
        out.update(group or {})
    return out


class Paraphraser:
    """선택: LLM으로 대사 의역 (서사·의미 유지). 이름은 자리표시자로 보호."""

    PROMPT = ("다음 웹툰 대사를 의미와 말투(존댓말/반말), 감정은 그대로 두고 다른 표현으로 바꿔 써라. "
              "길이는 비슷하게, ⟦숫자⟧ 표시는 절대 바꾸지 말 것. 바꾼 대사 한 줄만 출력.\n대사: {text}")

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        # device_map="auto": 7B bf16(~15GB)가 3080 한 장에 안 들어가면 두 GPU에 나눠 올림
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.model_id = model_id

    def __call__(self, text: str, protect: list[str]) -> str:
        import torch
        orig, keep = text, {}
        for i, w in enumerate(sorted(set(protect), key=len, reverse=True)):
            if w in text:
                keep[f"⟦{i}⟧"] = w
                text = text.replace(w, f"⟦{i}⟧")
        msgs = [{"role": "user", "content": self.PROMPT.format(text=text)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=128, do_sample=False)  # greedy = 재현 가능
        res = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        for k, w in keep.items():
            res = res.replace(k, w)
        ok = res and "⟦" not in res and all(w in res for w in keep.values())
        return res if ok else orig  # 자리표시자가 깨지면 원문 유지


def plan_changes(blocks: list[Block], table: dict[str, Any], fuzzy: bool = True,
                 paraphraser: Paraphraser | None = None, categories=("dialogue", "sfx", "credit")) -> list[TextChange]:
    ents = flat_entities(table)
    sfx_map = table.get("sfx", {}) or {}
    credit = table.get("credit", {}) or {}
    changes: list[TextChange] = []
    for b in blocks:
        if b.category not in categories:
            continue
        new, rules = b.text, []
        if b.category == "dialogue":
            new, rules = substitute_entities(b.text, ents, fuzzy=fuzzy)
            if paraphraser is not None:
                p = paraphraser(new, protect=list(ents.values()))
                if p and p != new:
                    new, rules = p, rules + [f"llm_paraphrase:{paraphraser.model_id}"]
        elif b.category == "sfx":
            key = b.text.replace("\n", "")
            for k in sorted(sfx_map, key=len, reverse=True):
                if k in key:
                    new = key.replace(k, sfx_map[k])
                    rules.append(f"sfx_table:{k}→{sfx_map[k]}")
                    break
        elif b.category == "credit":
            for k, v in (credit.get("replace") or {}).items():
                if k in new:
                    new = new.replace(k, v)
                    rules.append(f"credit_table:{k}→{v}")
            if not rules and credit.get("default_text"):
                new, rules = credit["default_text"], ["credit_default"]
        if new != b.text and rules:
            changes.append(TextChange(b.id, b.category, b.text, new, tuple(b.bbox), rules))
    return changes


def apply_text_variant(img: np.ndarray, blocks: list[Block], changes: list[TextChange],
                       font_paths: dict[str, str], orig_img: np.ndarray | None = None) -> np.ndarray:
    """바뀌는 블록만 지우고 다시 씀. orig_img: 글자색 추정에 쓸 원본 (both 변형에서 유용)."""
    by_id = {b.id: b for b in blocks}
    out = img.copy()
    for c in changes:
        b = by_id[c.block_id]
        out = erase_block(out, b)
    for c in changes:
        b = by_id[c.block_id]
        font = font_paths.get(c.category, font_paths["dialogue"])
        out = draw_block(out, b, c.after, font, orig_img=orig_img if orig_img is not None else img)
    return out


def changes_to_dict(changes: list[TextChange]) -> list[dict]:
    return [asdict(c) for c in changes]
