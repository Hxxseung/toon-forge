from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    base = Path(path).resolve().parent.parent  # configs/ 의 상위 = 레포 루트
    for k in ("data_root", "out_root", "work_tables"):
        p = Path(cfg[k])
        cfg[k] = str(p if p.is_absolute() else (base / p).resolve())
    return cfg


def load_work_table(table_dir: str | Path, work_id: str) -> dict[str, Any]:
    """작품별 변형 표 (configs/works/<work_id>.yaml). 없으면 빈 표."""
    p = Path(table_dir) / f"{work_id}.yaml"
    return load_yaml(p) if p.exists() else {}


def stable_seed(*parts: Any, base: int = 0) -> int:
    """(컷 ID, 축, 강도) 같은 조합에서 항상 같은 시드가 나오게."""
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return (int(h[:8], 16) + base) % (2**31 - 1)
