"""실행 진입점.

python -m replica.cli cut    -c configs/default.yaml           # 원작 → 컷 + originals.jsonl
python -m replica.cli ocr    -c configs/default.yaml           # 컷 OCR → outputs/ocr/<cut_id>.json
python -m replica.cli text   -c configs/default.yaml --text-level full
python -m replica.cli image  -c configs/default.yaml --axis character --strength weak medium
python -m replica.cli both   -c configs/default.yaml --axis character --strength medium --text-level full
python -m replica.cli export -c configs/default.yaml           # originals.csv / variants.csv
공통: --works A B (작품 제한), --limit N (컷 수 제한), --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import manifest as M
from .config import load_config, load_work_table, stable_seed
from .cuts import list_slices, split_cuts, stitch
from .ocr import load_blocks

TEXT_LEVELS = {
    "names": dict(categories=("dialogue",), paraphrase=False),           # 대사 속 이름·지명·아이템만
    "full": dict(categories=("dialogue", "sfx", "credit"), paraphrase=False),  # + 효과음 + 크레딧
    "paraphrase": dict(categories=("dialogue", "sfx", "credit"), paraphrase=True),  # + 대사 의역(LLM)
}


# ───────────────────────── helpers
def _iter_originals(cfg, args):
    recs = list(M.originals_manifest(cfg["out_root"]).read())
    if args.works:
        recs = [r for r in recs if r["work_id"] in args.works]
    if args.limit:
        recs = recs[: args.limit]
    if args.shard:  # "i/n": 컷을 번갈아 나눔 (GPU별 프로세스 병렬 실행용)
        i, n = map(int, args.shard.split("/"))
        recs = recs[i::n]
    return recs


def _claim(cfg, args, key: str) -> bool:
    """--queue 모드: 같은 run 이름을 쓰는 프로세스들끼리 작업을 하나씩 선점.
    파일을 O_EXCL로 만드는 데 성공한 프로세스만 그 작업을 함 → 먼저 끝난 GPU가 남은 일을 가져감."""
    if not getattr(args, "queue", None):
        return True
    d = Path(cfg["out_root"]) / ".claims" / args.queue
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / key, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


class Progress:
    """이 프로세스가 처리한 개수·장당 시간·남은 시간(대략) 출력."""

    def __init__(self, tag: str, todo: int, workers: int = 1):
        self.tag, self.todo, self.workers = tag, todo, max(1, workers)
        self.n, self.t0 = 0, time.time()
        dev = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.who = f"gpu{dev}" if dev not in (None, "") else "main"
        print(f"[{self.tag} {self.who}] 남은 작업 {todo}개 (전체 기준, 작업자 {self.workers}명)", flush=True)

    def step(self, label: str):
        self.n += 1
        el = time.time() - self.t0
        avg = el / self.n
        left = max(0, self.todo / self.workers - self.n) * avg
        print(f"[{self.tag} {self.who}] {self.n}개 완료 | {avg:.1f}s/개 | 남은 시간 ≈ {left/3600:.1f}h | {label}",
              flush=True)


def _abs(cfg, rel: str) -> Path:
    return Path(cfg["out_root"]) / rel


def _save(cfg, arr: np.ndarray, vtype: str, rec: dict, vid: str) -> tuple[str, str]:
    rel = Path("variants") / vtype / rec["work_id"] / rec["episode"] / f"{vid}.png"
    p = _abs(cfg, str(rel))
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(p)
    return str(rel), M.sha256_file(p)


_EP_CACHE: dict[str, np.ndarray] = {}


def load_original(cfg, rec) -> Image.Image:
    """원본 컷 복원: 원본 회차 조각을 이어 붙여 y_range만 잘라냄 (디스크에 저장하지 않음).
    회차 하나만 메모리에 캐시 → 매니페스트 순서(회차별)로 돌면 회차당 한 번만 읽음."""
    key = rec["source_dir"]
    if key not in _EP_CACHE:
        _EP_CACHE.clear()
        arr, _ = stitch(list_slices(Path(cfg["data_root"]) / key))
        _EP_CACHE[key] = arr
    y0, y1 = rec["y_range"]
    cut = _EP_CACHE[key][y0:y1]
    if M.sha256_array(cut) != rec["sha256"]:
        raise RuntimeError(f"{rec['cut_id']}: 원본 이미지가 분할 때와 다릅니다 ({key}). `cut`을 다시 실행하세요.")
    return Image.fromarray(cut)


def _blocks(cfg, rec):
    p = _abs(cfg, rec["ocr_path"]) if rec.get("ocr_path") else None
    return load_blocks(p) if p and p.exists() else None


def _fonts(cfg):
    return cfg["text"]["fonts"]


# ───────────────────────── commands
def cmd_cut(cfg, args):
    src = Path(cfg["data_root"]) / cfg["originals_dir"]
    if not src.is_dir():
        raise SystemExit(f"{src} 가 없습니다. 먼저 `python tools/prepare_data.py --src data --pick …` 로 회차를 연결하세요.")
    man = M.originals_manifest(cfg["out_root"])
    cc = cfg["cut"]
    works = sorted(d for d in src.iterdir() if d.is_dir() and (not args.works or d.name in args.works))
    n = 0
    for wd in works:
        for ed in sorted(d for d in wd.iterdir() if d.is_dir()):
            slices = list_slices(ed)
            if not slices:
                continue
            arr, spans = stitch(slices)
            cuts = split_cuts(arr, spans, cc["min_gutter"], cc["min_cut_h"], cc["std_thr"], cc["pad"],
                              cc.get("max_cut_h", 0), cc.get("resplit_gutter", 15), cc.get("resplit_std", 12.0),
                              cc.get("merge_below", 0), cc.get("resplit_frac", 0.9))
            src_rel = str(Path(cfg["originals_dir"]) / wd.name / ed.name)
            for i, c in enumerate(cuts):
                cid = M.make_cut_id(wd.name, ed.name, i)
                if cid in man and not args.overwrite:
                    continue
                man.append(M.OriginalRecord(  # 원본 컷 이미지는 저장하지 않고 위치·해시만 기록
                    cid, wd.name, ed.name, i, src_rel, c.image.width, c.image.height,
                    M.sha256_array(np.asarray(c.image)), c.slices, (c.y0, c.y1), len(cuts),
                    ocr_path=f"ocr/{cid}.json"))
                n += 1
            print(f"[cut] {wd.name}/{ed.name}: {len(cuts)} cuts")
    print(f"[cut] new records: {n}")


def cmd_ocr(cfg, args):
    from .ocr import PaddleKoreanOCR, build_blocks, save_blocks
    eng = PaddleKoreanOCR(cfg["ocr"]["min_score"], cfg["ocr"].get("device"))
    for r in _iter_originals(cfg, args):
        out = _abs(cfg, r["ocr_path"])
        if (out.exists() and not args.overwrite) or not _claim(cfg, args, "ocr__" + r["cut_id"]):
            continue
        with tempfile.TemporaryDirectory() as td:  # OCR 입력용 임시 파일 — 끝나면 바로 삭제
            tmp = Path(td) / "cut.png"
            load_original(cfg, r).save(tmp)
            lines = eng.lines(tmp)
        edge = r["cut_index"] in (0, r["n_cuts_in_episode"] - 1)
        save_blocks(out, r["cut_id"], build_blocks(lines, is_edge_cut=edge))
        print(f"[ocr] {r['cut_id']}: {len(lines)} lines")


def _make_paraphraser(cfg, level):
    if not TEXT_LEVELS[level]["paraphrase"]:
        return None
    from .text_variant import Paraphraser
    return Paraphraser(cfg["text"]["paraphrase_model"])


def _text_variant(cfg, rec, base_img: np.ndarray, orig_img: np.ndarray, level: str, para):
    from .text_variant import apply_text_variant, changes_to_dict, plan_changes
    blocks = _blocks(cfg, rec)
    if blocks is None:
        return None, None
    table = load_work_table(cfg["work_tables"], rec["work_id"])
    lv = TEXT_LEVELS[level]
    changes = plan_changes(blocks, table, cfg["text"]["fuzzy_names"], para, lv["categories"])
    out = apply_text_variant(base_img, blocks, changes, _fonts(cfg), orig_img=orig_img) if changes else None
    info = {"level": level, "table_version": table.get("version", "none"), "categories": list(lv["categories"]),
            "n_changed": len(changes), "changes": changes_to_dict(changes)}
    return out, info


def cmd_text(cfg, args):
    man = M.variants_manifest(cfg["out_root"])
    para = _make_paraphraser(cfg, args.text_level)
    ver = M.git_commit(Path(__file__).parent)
    for r in _iter_originals(cfg, args):
        vid = M.make_variant_id(r["cut_id"], "text", "text", args.text_level, 0)
        if (vid in man and not args.overwrite) or not _claim(cfg, args, vid):
            continue
        orig = np.asarray(load_original(cfg, r))
        out, info = _text_variant(cfg, r, orig, orig, args.text_level, para)
        base = dict(variant_id=vid, original_cut_id=r["cut_id"], work_id=r["work_id"], episode=r["episode"],
                    cut_index=r["cut_index"], variant_type="text", axis="text", strength=args.text_level,
                    seed=0, width=r["width"], height=r["height"], text_edit=info, code_version=ver)
        if out is None:  # OCR 없음 / 바꿀 텍스트 없음 → 기록만 (재실행 시 건너뜀)
            man.append(M.VariantRecord(**base, path="", sha256="", status="skipped_no_text" if info else "skipped_no_ocr"))
            continue
        rel, sha = _save(cfg, out, "text", r, vid)
        from .image_variant import pixel_change_ratio
        man.append(M.VariantRecord(**base, path=rel, sha256=sha, qc={"pixel_change_ratio": pixel_change_ratio(orig, out)}))
        print(f"[text] {vid}: {info['n_changed']} blocks")


def _image_one(cfg, editor, r, axis, strength, man, ver, overwrite):
    from .image_variant import pixel_change_ratio, run_image_edit
    seed = stable_seed(r["cut_id"], axis, strength, base=cfg["image"]["seed_base"])
    vid = M.make_variant_id(r["cut_id"], "image", axis, strength, seed)
    if vid in man and not overwrite:
        return vid
    orig = load_original(cfg, r)
    blocks = _blocks(cfg, r) or []
    table = load_work_table(cfg["work_tables"], r["work_id"])
    out, info = run_image_edit(editor, orig, axis, strength, seed, cfg["image"]["templates"], table, blocks,
                               protect=cfg["image"]["protect_text"], ref_root=cfg["work_tables"])
    rel, sha = _save(cfg, out, "image", r, vid)
    man.append(M.VariantRecord(
        variant_id=vid, original_cut_id=r["cut_id"], work_id=r["work_id"], episode=r["episode"],
        cut_index=r["cut_index"], variant_type="image", axis=axis, strength=strength, seed=seed,
        path=rel, width=r["width"], height=r["height"], sha256=sha, image_edit=info, code_version=ver,
        qc={"pixel_change_ratio": pixel_change_ratio(np.asarray(orig), out)}))
    return vid


def _image_vid(cfg, r, ax, st):
    seed = stable_seed(r["cut_id"], ax, st, base=cfg["image"]["seed_base"])
    return M.make_variant_id(r["cut_id"], "image", ax, st, seed)


def cmd_image(cfg, args):
    from .image_editors import build_editor
    man = M.variants_manifest(cfg["out_root"])
    ver = M.git_commit(Path(__file__).parent)
    jobs = [(r, ax, st) for r in _iter_originals(cfg, args) for ax in args.axis for st in args.strength]
    todo = [j for j in jobs if args.overwrite or _image_vid(cfg, *j) not in man]
    if not todo:
        print("[image] 할 작업 없음")
        return
    prog = Progress("image", len(todo), args.workers)
    editor = build_editor(cfg["image"]["backends"][args.backend or cfg["image"]["backend"]], args.device)
    for r, ax, st in todo:
        vid = _image_vid(cfg, r, ax, st)
        if not _claim(cfg, args, vid):
            continue
        _image_one(cfg, editor, r, ax, st, man, ver, args.overwrite)
        prog.step(vid)


def cmd_both(cfg, args):
    """이미지 변형(없으면 먼저 생성) 위에 텍스트 변형을 얹음 → parent_variant_id로 연결."""
    from .image_editors import build_editor
    from .image_variant import pixel_change_ratio
    editor = None
    man = M.variants_manifest(cfg["out_root"])
    para = _make_paraphraser(cfg, args.text_level)
    ver = M.git_commit(Path(__file__).parent)
    jobs = [(r, ax, st) for r in _iter_originals(cfg, args) for ax in args.axis for st in args.strength]
    prog = Progress("both", len(jobs), args.workers)
    for r, ax, st in jobs:
        seed = stable_seed(r["cut_id"], ax, st, base=cfg["image"]["seed_base"])
        vid = M.make_variant_id(r["cut_id"], "both", f"{ax}+text", f"{st}+{args.text_level}", seed)
        if (vid in man and not args.overwrite) or _blocks(cfg, r) is None or not _claim(cfg, args, vid):
            continue
        pid = M.make_variant_id(r["cut_id"], "image", ax, st, seed)
        if pid not in man:
            man.refresh()  # 다른 GPU가 방금 만들었을 수 있음
        if pid not in man:
            if _claim(cfg, args, pid):  # 부모 이미지 변형을 여기서 생성 (모델은 처음 필요할 때만 로드)
                editor = editor or build_editor(cfg["image"]["backends"][args.backend or cfg["image"]["backend"]],
                                                args.device)
                _image_one(cfg, editor, r, ax, st, man, ver, False)
            else:  # 다른 GPU가 만드는 중 → 끝날 때까지 기다림 (최대 30분)
                t0 = time.time()
                while pid not in man and time.time() - t0 < 1800:
                    time.sleep(5)
                    man.refresh()
                if pid not in man:
                    print(f"[both] {pid} 대기 시간 초과 → 건너뜀", flush=True)
                    continue
        parent = man.by_id()[pid]
        img = np.asarray(Image.open(_abs(cfg, parent["path"])).convert("RGB"))
        orig = np.asarray(load_original(cfg, r))
        out, tinfo = _text_variant(cfg, r, img, orig, args.text_level, para)
        if out is None:  # 바꿀 텍스트 없음 → both를 만들지 않음 (image 변형과 동일하므로)
            continue
        rel, sha = _save(cfg, out, "both", r, vid)
        man.append(M.VariantRecord(
            variant_id=vid, original_cut_id=r["cut_id"], work_id=r["work_id"], episode=r["episode"],
            cut_index=r["cut_index"], variant_type="both", axis=ax, strength=st, seed=seed, path=rel,
            width=r["width"], height=r["height"], sha256=sha, image_edit=parent["image_edit"],
            text_edit=tinfo, parent_variant_id=pid, code_version=ver,
            qc={"pixel_change_ratio": pixel_change_ratio(orig, out)}))
        prog.step(vid)


def cmd_export(cfg, args):
    o, v = M.export_csv(cfg["out_root"])
    print(f"[export] {o}\n[export] {v}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="replica")
    ap.add_argument("command", choices=["cut", "ocr", "text", "image", "both", "export"])
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--works", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--axis", nargs="*", default=None)
    ap.add_argument("--strength", nargs="*", default=None)
    ap.add_argument("--text-level", default=None, choices=list(TEXT_LEVELS))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--shard", default=None, help="i/n — 컷을 n등분해 i번째만 처리 (예: GPU0은 0/2, GPU1은 1/2)")
    ap.add_argument("--device", default=None, help="편집 모델을 올릴 장치 (예: cuda:0). CUDA_VISIBLE_DEVICES를 쓰면 생략")
    ap.add_argument("--queue", default=None, help="작업 큐 이름. 같은 이름의 프로세스들이 작업을 나눠 가짐 (먼저 끝난 쪽이 더 가져감)")
    ap.add_argument("--workers", type=int, default=1, help="함께 도는 프로세스 수 (남은 시간 계산용)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    args.axis = args.axis or cfg["image"]["axes"]
    args.strength = args.strength or cfg["image"]["strengths"]
    args.text_level = args.text_level or cfg["text"]["default_level"]
    {"cut": cmd_cut, "ocr": cmd_ocr, "text": cmd_text, "image": cmd_image,
     "both": cmd_both, "export": cmd_export}[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())