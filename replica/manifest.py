"""원작/변형 매니페스트.

- originals.jsonl : 원작 컷 1개 = 1줄
- variants.jsonl  : 변형 컷 1개 = 1줄 (원작 컷 ID로 정답 짝 연결)
JSONL은 추가 전용(append-only)이며, 같은 ID가 이미 있으면 다시 만들지 않는다(재개 가능).
CSV는 `export_csv`로 평탄화해서 뽑는다 (분석·공유용).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

VARIANT_TYPES = ("image", "text", "both")


def sha256_array(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_commit(repo_dir: str | Path = ".") -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else "nogit"
    except Exception:
        return "nogit"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_cut_id(work_id: str, episode: str, cut_index: int) -> str:
    return f"{work_id}__{episode}__c{cut_index:03d}"


def make_variant_id(cut_id: str, variant_type: str, axis: str, strength: str, seed: int) -> str:
    return f"{cut_id}__{variant_type}-{axis}-{strength}-s{seed}"


@dataclass
class OriginalRecord:
    cut_id: str
    work_id: str
    episode: str
    cut_index: int
    source_dir: str                # 원본 회차 폴더 (data_root 기준). 원본 컷 이미지는 저장하지 않음
    width: int
    height: int
    sha256: str                    # 잘라낸 컷 픽셀(RGB uint8 배열)의 해시 — 원본이 바뀌었는지 확인용
    source_slices: list[str]       # 이 컷이 걸쳐 있는 원본 조각 파일들
    y_range: tuple[int, int]       # 회차를 이어 붙인 이미지에서의 세로 위치 → 이 범위를 잘라 원본 컷 복원
    n_cuts_in_episode: int = -1
    ocr_path: str | None = None    # OCR 결과 JSON (텍스트 블록)
    created_at: str = field(default_factory=now_iso)


@dataclass
class TextChange:
    block_id: int
    category: str                  # dialogue / sfx / credit
    before: str
    after: str
    bbox: tuple[int, int, int, int]
    rules: list[str]               # 적용된 규칙 (entity:카일→레온, josa, sfx_table, credit_table, llm_paraphrase ...)


@dataclass
class VariantRecord:
    variant_id: str
    original_cut_id: str           # 정답 짝 — 원작 컷
    work_id: str
    episode: str
    cut_index: int
    variant_type: str              # image / text / both
    axis: str                      # character / background / style / combo / text 등
    strength: str                  # weak / medium / strong
    seed: int
    path: str
    width: int
    height: int
    sha256: str
    image_edit: dict[str, Any] | None = None   # backend, model_id, prompt, steps, guidance, protect_text ...
    text_edit: dict[str, Any] | None = None    # table_version, changes[list[TextChange]], n_changed ...
    parent_variant_id: str | None = None       # both = 이미지 변형 위에 텍스트 변형을 얹은 경우 그 이미지 변형 ID
    qc: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"                         # ok / skipped_no_text / failed
    code_version: str = ""
    created_at: str = field(default_factory=now_iso)


class JsonlManifest:
    def __init__(self, path: str | Path, key: str):
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids = {r[key] for r in self.read()} if self.path.exists() else set()

    def read(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def __contains__(self, rid: str) -> bool:
        return rid in self._ids

    def append(self, rec) -> None:
        d = asdict(rec) if hasattr(rec, "__dataclass_fields__") else dict(rec)
        if d[self.key] in self._ids:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self._ids.add(d[self.key])

    def by_id(self) -> dict[str, dict]:
        return {r[self.key]: r for r in self.read()}


def originals_manifest(out_root: str | Path) -> JsonlManifest:
    return JsonlManifest(Path(out_root) / "manifests" / "originals.jsonl", "cut_id")


def variants_manifest(out_root: str | Path) -> JsonlManifest:
    return JsonlManifest(Path(out_root) / "manifests" / "variants.jsonl", "variant_id")


VARIANT_CSV_COLS = [
    "variant_id", "original_cut_id", "work_id", "episode", "cut_index", "variant_type",
    "axis", "strength", "seed", "path", "original_source_dir", "original_y_start", "original_y_end",
    "width", "height", "status",
    "parent_variant_id", "backend", "model_id", "prompt", "n_text_changed", "text_changes",
    "pixel_change_ratio", "sha256", "code_version", "created_at",
]


def export_csv(out_root: str | Path) -> tuple[Path, Path]:
    """originals.csv / variants.csv (정답 짝 = original_cut_id) 생성."""
    out_root = Path(out_root)
    origs = originals_manifest(out_root).by_id()
    o_csv = out_root / "manifests" / "originals.csv"
    o_cols = ["cut_id", "work_id", "episode", "cut_index", "source_dir", "y_start", "y_end", "width", "height",
              "n_cuts_in_episode", "ocr_path", "sha256"]
    with open(o_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=o_cols, extrasaction="ignore")
        w.writeheader()
        for r in origs.values():
            w.writerow({**r, "y_start": r["y_range"][0], "y_end": r["y_range"][1]})

    v_csv = out_root / "manifests" / "variants.csv"
    with open(v_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=VARIANT_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in variants_manifest(out_root).read():
            ie, te = r.get("image_edit") or {}, r.get("text_edit") or {}
            changes = te.get("changes", [])
            o = origs.get(r["original_cut_id"], {})
            yr = o.get("y_range", ["", ""])
            w.writerow({
                **r,
                "original_source_dir": o.get("source_dir", ""),
                "original_y_start": yr[0], "original_y_end": yr[1],
                "backend": ie.get("backend", ""), "model_id": ie.get("model_id", ""),
                "prompt": ie.get("prompt", ""),
                "n_text_changed": len(changes),
                "text_changes": " | ".join(f"{c['before']}→{c['after']}" for c in changes),
                "pixel_change_ratio": r.get("qc", {}).get("pixel_change_ratio", ""),
            })
    return o_csv, v_csv