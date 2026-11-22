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
    if getattr(args, "only_person", False):  # 사람 검출 결과에서 사람이 있는 컷만
        from .people import load_people
        recs = [r for r in recs if (load_people(_abs(cfg, f"people/{r['cut_id']}.json")) or {}).get("has_person")]
    if getattr(args, "per_work", None):  # 작품마다 N컷을 회차 전체에 고르게 (시드 고정 → 항상 같은 컷)
        by_w: dict[str, list] = {}
        for r in recs:
            by_w.setdefault(r["work_id"], []).append(r)
        recs = []
        for w, rs in by_w.items():
            n = min(args.per_work, len(rs))
            idx = sorted({round(i * (len(rs) - 1) / max(1, n - 1)) for i in range(n)}) if n > 1 else [len(rs) // 2]
            recs += [rs[i] for i in idx]
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


_FONT_CACHE: dict | None = None


def _fonts(cfg):
    """카테고리별 폰트. 파일이 없으면 있는 폰트로 대체하고 한 번만 알림
    (예: Ubuntu fonts-nanum 에는 NanumGothicExtraBold 가 없음 → fonts-nanum-extra)."""
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE
    fonts = dict(cfg["text"]["fonts"])
    ok = [p for p in fonts.values() if Path(p).exists()]
    if not ok:
        raise SystemExit(f"글자 폰트 파일이 하나도 없습니다: {sorted(set(fonts.values()))}\n"
                         f"  sudo apt install fonts-nanum fonts-nanum-extra  또는 configs/default.yaml 의 text.fonts 경로를 수정하세요.")
    fallback = fonts.get("dialogue") if Path(fonts.get("dialogue", "")).exists() else ok[0]
    for k, p in fonts.items():
        if not Path(p).exists():
            print(f"[text] 폰트 없음: {k} = {p} → {fallback} 로 대체 (sudo apt install fonts-nanum-extra 로 설치 가능)")
            fonts[k] = fallback
    _FONT_CACHE = fonts
    return fonts


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
        save_blocks(out, r["cut_id"], build_blocks(lines, is_edge_cut=edge), raw_lines=lines)
        print(f"[ocr] {r['cut_id']}: {len(lines)} lines")


def cmd_regroup(cfg, args):
    """OCR을 다시 돌리지 않고, 저장된 원시 줄로 블록만 다시 묶음 (묶는 규칙을 바꿨을 때).
    사람이 검수한 파일(reviewed: true)은 건드리지 않음."""
    from .ocr import build_blocks, load_raw_lines, save_blocks
    n = skipped = 0
    for r in _iter_originals(cfg, args):
        p = _abs(cfg, r["ocr_path"])
        if not p.exists():
            continue
        d, raw = load_raw_lines(p)
        if d.get("reviewed"):
            skipped += 1
            continue
        edge = r["cut_index"] in (0, r["n_cuts_in_episode"] - 1)
        before = len(d["blocks"])
        blocks = build_blocks(raw, is_edge_cut=edge)
        save_blocks(p, r["cut_id"], blocks, d.get("engine", "paddleocr-ppocrv5-ko"), raw_lines=raw)
        n += 1
        if args.verbose:
            print(f"[regroup] {r['cut_id']}: {before} → {len(blocks)} blocks")
    print(f"[regroup] {n}개 다시 묶음, 검수 완료라 건너뜀 {skipped}개")


def cmd_people(cfg, args):
    """컷마다 사람 검출 → outputs/people/<cut_id>.json (캐릭터 변형 대상 판정·편집 마스크용)."""
    from .people import PersonDetector, save_people
    pc = cfg.get("people", {})
    todo = [r for r in _iter_originals(cfg, args)
            if args.overwrite or not _abs(cfg, f"people/{r['cut_id']}.json").exists()]
    if not todo:
        print("[people] 할 작업 없음")
        return
    det = PersonDetector(pc.get("model_id", "google/owlv2-base-patch16-ensemble"),
                         pc.get("queries", ["a person", "a human face"]), pc.get("threshold", 0.25),
                         device=args.device)
    prog = Progress("people", len(todo), args.workers)
    for r in todo:
        if not _claim(cfg, args, "people__" + r["cut_id"]):
            continue
        boxes = det.detect(load_original(cfg, r))
        save_people(_abs(cfg, f"people/{r['cut_id']}.json"), r["cut_id"], boxes, det.info())
        prog.step(f"{r['cut_id']} 사람 {'있음' if boxes else '없음'} ({len(boxes)})")


def cmd_qc(cfg, args):
    """이미지 변형 품질 측정 (거르지 않고 측정만) → outputs/qc/<variant_id>.json + manifests/qc_<tag>.csv.
    기준값을 정하기 위한 단계: 파일럿 결과에서 잘된 컷/망가진 컷의 값을 비교."""
    import csv
    import json
    from .people import PersonDetector, load_people
    from .qc import measure
    pc = cfg.get("people", {})
    rows = [v for v in M.variants_manifest(cfg["out_root"]).read()
            if v["variant_type"] == "image" and v["status"] == "ok" and v["path"]
            and (not args.tag or v["variant_id"].endswith("-" + args.tag))
            and (not args.axis_filter or v["axis"] in args.axis_filter)]
    if not rows:
        print("[qc] 대상 변형 없음")
        return
    origs = M.originals_manifest(cfg["out_root"]).by_id()
    det = PersonDetector(pc.get("model_id", "google/owlv2-base-patch16-ensemble"),
                         pc.get("queries", ["a person", "a human face"]), pc.get("threshold", 0.25),
                         device=args.device)
    ocr = None
    if not args.no_text:
        from .ocr import PaddleKoreanOCR
        ocr = PaddleKoreanOCR(cfg["ocr"]["min_score"], cfg["ocr"].get("device"))
    prog = Progress("qc", len(rows), args.workers)
    out_rows = []
    for v in rows:
        r = origs[v["original_cut_id"]]
        qp = _abs(cfg, f"qc/{v['variant_id']}.json")
        if qp.exists() and not args.overwrite:
            out_rows.append(json.load(open(qp, encoding="utf-8")))
            continue
        if not _claim(cfg, args, "qc__" + v["variant_id"]):
            continue
        orig = np.asarray(load_original(cfg, r))
        out = np.asarray(Image.open(_abs(cfg, v["path"])).convert("RGB"))
        ppl = load_people(_abs(cfg, f"people/{r['cut_id']}.json")) or {"boxes": []}
        q = measure(orig, out, ppl["boxes"], det.detect(Image.fromarray(out)),
                    cfg["image"].get("person_mask_pad", 0.15), ocr, _blocks(cfg, r))
        rec = {"variant_id": v["variant_id"], "original_cut_id": r["cut_id"], "axis": v["axis"],
               "strength": v["strength"], **q}
        qp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(qp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        out_rows.append(rec)
        prog.step(f"{v['variant_id']} {q}")
    tag = args.tag or "all"
    cp = _abs(cfg, f"manifests/qc_{tag}.csv")
    cols = ["variant_id", "original_cut_id", "axis", "strength", "person_count_orig", "person_count_out",
            "structure_ncc", "seam_score", "text_retention"]
    with open(cp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda x: x["variant_id"]))
    print(f"[qc] {len(out_rows)}개 → {cp}")


def cmd_reject(cfg, args):
    """검토에서 뺄 변형 기록: python -m replica.cli reject <variant_id 또는 cut_id…> --reason "2명으로 늘어남"
    cut_id만 주면 그 컷의 이미지 변형 전부(--strength로 좁힘)."""
    ids, vs = [], list(M.variants_manifest(cfg["out_root"]).read())
    if args.sheet:  # 비교 이미지 번호로 제외: --sheet check_123456.json --rows 3 7 12
        import json
        index = json.load(open(args.sheet, encoding="utf-8"))
        for n in args.rows or []:
            if str(n) not in index:
                print(f"[reject] 번호 없음: #{n}")
            ids += index.get(str(n), [])
    for x in args.ids:
        hit = [v["variant_id"] for v in vs if v["variant_id"] == x or (
            v["original_cut_id"] == x and v["variant_type"] == "image"
            and (not args.strength_given or v["strength"] in args.strength))]
        if not hit:
            print(f"[reject] 없음: {x}")
        ids += hit
    n = M.add_rejects(cfg["out_root"], ids, args.reason or "")
    print(f"[reject] {n}개 추가 → outputs/manifests/rejects.csv (export·both에서 제외)")


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
    info = {"level": level, "table_version": table.get("version", "none"), "categories": list(lv["categories"]),
            "n_changed": len(changes), "changes": changes_to_dict(changes)}
    try:  # 한 컷의 글자 렌더링 실패가 전체 실행을 멈추지 않게
        out = apply_text_variant(base_img, blocks, changes, _fonts(cfg), orig_img=orig_img) if changes else None
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
        print(f"[text] {rec['cut_id']} 렌더링 실패 → 건너뜀 ({info['error']})", flush=True)
        return None, info
    return out, info


def cmd_text(cfg, args):
    man = M.variants_manifest(cfg["out_root"])
    para = _make_paraphraser(cfg, args.text_level)
    ver = M.git_commit(Path(__file__).parent)
    no_table, made = set(), 0
    for r in _iter_originals(cfg, args):
        tver = _table_version(cfg, r["work_id"])
        if tver is None:  # 치환표가 없으면 기록하지 않음 (나중에 표를 만들면 그때 생성)
            no_table.add(r["work_id"])
            continue
        vid = M.make_variant_id(r["cut_id"], "text", "text", f"{args.text_level}@{tver}", 0)
        if (vid in man and not args.overwrite) or not _claim(cfg, args, vid):
            continue
        orig = np.asarray(load_original(cfg, r))
        out, info = _text_variant(cfg, r, orig, orig, args.text_level, para)
        base = dict(variant_id=vid, original_cut_id=r["cut_id"], work_id=r["work_id"], episode=r["episode"],
                    cut_index=r["cut_index"], variant_type="text", axis="text", strength=f"{args.text_level}@{tver}",
                    seed=0, width=r["width"], height=r["height"], text_edit=info, code_version=ver)
        if out is None:  # OCR 없음 / 바꿀 텍스트 없음 / 렌더링 실패 → 기록 (실패는 failed_render로 남겨 나중에 확인)
            st = "skipped_no_ocr" if not info else ("failed_render" if info.get("error") else "skipped_no_text")
            man.append(M.VariantRecord(**base, path="", sha256="", status=st))
            continue
        rel, sha = _save(cfg, out, "text", r, vid)
        from .image_variant import pixel_change_ratio
        man.append(M.VariantRecord(**base, path=rel, sha256=sha, qc={"pixel_change_ratio": pixel_change_ratio(orig, out)}))
        made += 1
        print(f"[text] {vid}: {info['n_changed']} blocks")
    if no_table:
        print(f"[text] 치환표 없음 → 건너뜀: {sorted(no_table)}  (python tools/make_tables.py 로 초안 생성)")
    print(f"[text] 새로 만든 텍스트 변형 {made}개")


def _table_version(cfg, work_id: str) -> str | None:
    """작품 치환표 버전 (텍스트 변형 ID에 들어감 → 표를 고치고 버전을 올리면 새로 생성). 표가 없으면 None."""
    p = Path(cfg["work_tables"]) / f"{work_id}.yaml"
    if not p.exists():
        return None
    return str(load_work_table(cfg["work_tables"], work_id).get("version", "none"))


def _image_one(cfg, editor, r, axis, strength, man, ver, overwrite, tag=""):
    from .image_variant import pixel_change_ratio, run_image_edit
    seed = stable_seed(r["cut_id"], axis, strength, base=cfg["image"]["seed_base"])
    vid = M.make_variant_id(r["cut_id"], "image", axis, strength, seed, tag)
    if vid in man and not overwrite:
        return vid
    boxes = None
    if axis == "character" and cfg["image"].get("require_person", True):
        from .people import load_people
        ppl = load_people(_abs(cfg, f"people/{r['cut_id']}.json"))
        if ppl is not None and not ppl["has_person"]:  # 사람 없는 컷 → 캐릭터 변형 안 함 (모델이 사람을 새로 그려 넣음)
            man.append(M.VariantRecord(
                variant_id=vid, original_cut_id=r["cut_id"], work_id=r["work_id"], episode=r["episode"],
                cut_index=r["cut_index"], variant_type="image", axis=axis, strength=strength, seed=seed,
                path="", width=r["width"], height=r["height"], sha256="", status="skipped_no_person",
                code_version=ver))
            return vid
        boxes = ppl["boxes"] if ppl and cfg["image"].get("person_mask", True) else None
    orig = load_original(cfg, r)
    o = np.asarray(orig)
    blocks = _blocks(cfg, r) or []
    table = load_work_table(cfg["work_tables"], r["work_id"])
    pad = cfg["image"].get("person_mask_pad", 0.15)
    qcc = cfg["image"].get("qc", {}) or {}
    tries = 1 + (int(qcc.get("max_retries", 0)) if qcc.get("enabled") else 0)
    attempts, best = [], None
    for k in range(tries):  # QC 불합격이면 시드를 바꿔 다시 생성, 가장 나은(이음새가 작은) 결과를 남김
        s_k = seed if k == 0 else stable_seed(r["cut_id"], axis, strength, "retry", k, base=cfg["image"]["seed_base"])
        out, info = run_image_edit(editor, orig, axis, strength, s_k, cfg["image"]["templates"], table, blocks,
                                   protect=cfg["image"]["protect_text"], ref_root=cfg["work_tables"],
                                   work_id=r["work_id"], person_boxes=boxes, mask_pad=pad)
        q = _qc_quick(o, out, boxes, pad, qcc)
        attempts.append({"seed": s_k, **q})
        if best is None or (q.get("structure_ncc") or -1) > (best[2].get("structure_ncc") or -1):  # 포즈가 가장 잘 유지된 시도
            best = (out, info, q, s_k)
        if not q["fail"]:
            break
    out, info, q, used_seed = best
    rel, sha = _save(cfg, out, "image", r, vid)
    status = "failed_qc" if q["fail"] else "ok"  # failed_qc: 이미지는 남기되(확인용) both·평가에서는 제외
    man.append(M.VariantRecord(
        variant_id=vid, original_cut_id=r["cut_id"], work_id=r["work_id"], episode=r["episode"],
        cut_index=r["cut_index"], variant_type="image", axis=axis, strength=strength, seed=used_seed,
        path=rel, width=r["width"], height=r["height"], sha256=sha, image_edit=info, code_version=ver,
        status=status,
        qc={"pixel_change_ratio": pixel_change_ratio(o, out), **{k: v for k, v in q.items() if k != "fail"},
            "attempts": attempts if len(attempts) > 1 else []}))
    return vid


def _qc_quick(o: np.ndarray, out: np.ndarray, boxes, pad: float, qcc: dict) -> dict:
    """생성 중 바로 계산하는 가벼운 QC (모델 없이). 기준값은 configs의 image.qc."""
    from .people import people_mask
    from .qc import seam_score, structure_ncc
    if not boxes:
        return {"fail": False, "flags": []}
    m = people_mask(o.shape[:2], boxes, pad)
    seam, ncc = seam_score(o, out, m), structure_ncc(o, out, m)
    flags = []
    if seam is not None and seam > qcc.get("seam_fail_gt", 0.28):
        flags.append("seam")
    # 목표가 '같은 포즈, 다른 캐릭터'이므로 사람 영역의 큰 형태가 무너지면(여러 명으로 바뀜 등) 불합격
    if ncc is not None and ncc < qcc.get("structure_fail_lt", 0.4):
        flags.append("low_structure")
    fail = qcc.get("enabled", False) and bool(flags)
    return {"fail": bool(fail), "seam_score": None if seam is None else round(seam, 4),
            "structure_ncc": None if ncc is None else round(ncc, 4), "flags": flags}


def _image_vid(cfg, r, ax, st, tag=""):
    seed = stable_seed(r["cut_id"], ax, st, base=cfg["image"]["seed_base"])
    return M.make_variant_id(r["cut_id"], "image", ax, st, seed, tag)


def cmd_image(cfg, args):
    from .image_editors import build_editor
    man = M.variants_manifest(cfg["out_root"])
    ver = M.git_commit(Path(__file__).parent)
    jobs = [(r, ax, st) for r in _iter_originals(cfg, args) for ax in args.axis for st in args.strength]
    todo = [j for j in jobs if args.overwrite or _image_vid(cfg, *j, args.tag) not in man]
    if cfg["image"].get("require_person", True) and any(ax == "character" for _, ax, _ in todo):
        missing = sorted({r["cut_id"] for r, ax, _ in todo
                          if ax == "character" and not _abs(cfg, f"people/{r['cut_id']}.json").exists()})
        if missing:
            raise SystemExit(f"사람 검출 결과가 없는 컷 {len(missing)}개 (예: {missing[0]}). "
                             f"먼저 `python -m replica.cli people` 을 실행하세요.")
    if not todo:
        print("[image] 할 작업 없음")
        return
    prog = Progress("image", len(todo), args.workers)
    editor = build_editor(cfg["image"]["backends"][args.backend or cfg["image"]["backend"]], args.device)
    for r, ax, st in todo:
        vid = _image_vid(cfg, r, ax, st, args.tag)
        if not _claim(cfg, args, vid):
            continue
        _image_one(cfg, editor, r, ax, st, man, ver, args.overwrite, args.tag)
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
    rejects = M.load_rejects(cfg["out_root"])
    prog = Progress("both", len(jobs), args.workers)
    for r, ax, st in jobs:
        seed = stable_seed(r["cut_id"], ax, st, base=cfg["image"]["seed_base"])
        tver = _table_version(cfg, r["work_id"])
        if tver is None:
            continue
        vid = M.make_variant_id(r["cut_id"], "both", f"{ax}+text", f"{st}+{args.text_level}@{tver}", seed, args.tag)
        if (vid in man and not args.overwrite) or _blocks(cfg, r) is None or not _claim(cfg, args, vid):
            continue
        pid = M.make_variant_id(r["cut_id"], "image", ax, st, seed, args.tag)
        if pid not in man:
            man.refresh()  # 다른 GPU가 방금 만들었을 수 있음
        if pid not in man:
            if _claim(cfg, args, pid):  # 부모 이미지 변형을 여기서 생성 (모델은 처음 필요할 때만 로드)
                editor = editor or build_editor(cfg["image"]["backends"][args.backend or cfg["image"]["backend"]],
                                                args.device)
                _image_one(cfg, editor, r, ax, st, man, ver, False, args.tag)
            else:  # 다른 GPU가 만드는 중 → 끝날 때까지 기다림 (최대 30분)
                t0 = time.time()
                while pid not in man and time.time() - t0 < 1800:
                    time.sleep(5)
                    man.refresh()
                if pid not in man:
                    print(f"[both] {pid} 대기 시간 초과 → 건너뜀", flush=True)
                    continue
        parent = man.by_id()[pid]
        if parent["status"] != "ok" or pid in rejects:  # 사람 없음·QC 불합격·사람이 뺀 이미지 변형 위에는 안 만듦
            continue
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
    ap.add_argument("command", choices=["cut", "ocr", "regroup", "people", "text", "image", "both", "qc", "reject", "export"])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--works", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-work", type=int, help="작품마다 N컷을 회차 전체에서 고르게 뽑음 (파일럿 샘플용)")
    ap.add_argument("--only-person", action="store_true", help="사람이 검출된 컷만 (people 결과 필요)")
    ap.add_argument("--no-text", action="store_true", help="qc: 글자 보존 검사(OCR) 생략 → 빠름")
    ap.add_argument("--axis-filter", nargs="*", help="qc: 이 축의 변형만")
    ap.add_argument("ids", nargs="*", help="reject: 뺄 variant_id 또는 cut_id")
    ap.add_argument("--reason", default="", help="reject: 이유")
    ap.add_argument("--sheet", help="reject: compare_sheet가 만든 check_*.json")
    ap.add_argument("--rows", nargs="*", type=int, help="reject: 그 비교 이미지의 번호들")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--axis", nargs="*", default=None)
    ap.add_argument("--strength", nargs="*", default=None)
    ap.add_argument("--text-level", default=None, choices=list(TEXT_LEVELS))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--tag", default="", help="변형 ID 꼬리표 — 다른 설정으로 같은 컷을 비교할 때 (예: --tag lightning)")
    ap.add_argument("--shard", default=None, help="i/n — 컷을 n등분해 i번째만 처리 (예: GPU0은 0/2, GPU1은 1/2)")
    ap.add_argument("--device", default=None, help="편집 모델을 올릴 장치 (예: cuda:0). CUDA_VISIBLE_DEVICES를 쓰면 생략")
    ap.add_argument("--queue", default=None, help="작업 큐 이름. 같은 이름의 프로세스들이 작업을 나눠 가짐 (먼저 끝난 쪽이 더 가져감)")
    ap.add_argument("--workers", type=int, default=1, help="함께 도는 프로세스 수 (남은 시간 계산용)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    args.strength_given = args.strength is not None
    args.axis = args.axis or cfg["image"]["axes"]
    args.strength = args.strength or cfg["image"]["strengths"]
    args.text_level = args.text_level or cfg["text"]["default_level"]
    {"cut": cmd_cut, "ocr": cmd_ocr, "regroup": cmd_regroup, "people": cmd_people, "qc": cmd_qc, "reject": cmd_reject, "text": cmd_text, "image": cmd_image,
     "both": cmd_both, "export": cmd_export}[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
