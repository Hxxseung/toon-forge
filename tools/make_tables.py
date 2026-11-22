"""OCR 결과 → 작품별 치환표 초안 (configs/works/<work_id>.yaml).

후보 추출 (kiwipiepy 형태소 분석):
  - 고유명사(NNP)                      예: 카일
  - 한 어절 안에 붙은 명사 묶음(조어)   예: 천명산(천+명+산), 흑막검(흑막+검), 마마신(마마+신)
  - 호칭 앞 단어 (OO씨/님/형/누나/선배)
새 이름은 작품마다 고정 시드로 제안 → 사람이 열어서 틀린 후보를 지우고 version을 올리면 확정.
효과음: OCR에서 sfx로 분류된 블록 → 늘이기(쾅→콰앙) 또는 비슷한 의성어로.
크레딧: data/works.csv의 작가명 → 가상 필명.

python tools/make_tables.py                 # 모든 작품 초안 (이미 있는 파일은 건너뜀)
python tools/make_tables.py --works w001 --overwrite --min-freq 1
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replica import manifest as M  # noqa: E402
from replica.config import load_config, stable_seed  # noqa: E402
from replica.korean import is_hangul  # noqa: E402

PLACE_SUF = ("왕국", "제국", "대륙", "마을", "학교", "학원", "동굴", "산", "성", "국", "궁", "강", "숲", "시", "도시", "촌", "곡")
ITEM_SUF = ("반지", "목걸이", "검", "창", "활", "부", "석", "주", "패", "갑", "서")
HONORIFIC = ("씨", "님", "형", "형님", "누나", "언니", "오빠", "선배", "군", "양", "아저씨", "할배", "선생")
# 흔한 명사 묶음은 고유명사가 아님 (조어 후보에서 제외)
STOP = {"도련님", "선생님", "사람", "우리", "여기", "거기", "저기", "지금", "오늘", "내일", "어제", "정말", "진짜", "그냥",
        "하나", "나중", "이번", "다음", "처음", "마지막", "아이", "엄마", "아빠", "친구", "동생", "형님", "누나", "언니",
        "오빠", "할배", "할머니", "할아버지", "아저씨", "아줌마", "시간", "생각", "얘기", "이야기", "자기", "자신"}

PERSON_POOL = ["레온", "아리아", "하준", "서윤", "루카", "노아", "에린", "시온", "다온", "유나", "카이", "리안",
               "세린", "도윤", "하린", "로이", "제이", "미르", "소하", "이안", "벨라", "테오", "라온", "주노"]
SURNAMES = "김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허남심노하곽성차주우구민유나진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부황보"
GIVEN = ["지호", "서진", "민준", "예린", "하늘", "수아", "건우", "채원", "태윤", "나연", "시우", "가은", "도현", "유진"]
STEM_POOL = ["백운", "청람", "은월", "홍염", "흑림", "설화", "벽하", "금란", "현무", "자운", "월영", "적송", "남해", "북천"]
PEN_POOL = ["가람", "누리", "별하", "소담", "한결", "다솜", "이든", "새벽", "바림", "온새"]
SFX_SYN = {"쾅": "콰앙", "쿵": "쿠웅", "펑": "퍼엉", "휙": "휘익", "탁": "타악", "슥": "스윽", "두근": "콩닥", "꿀꺽": "벌컥",
           "흠칫": "움찔", "벌떡": "번쩍", "짝": "착", "툭": "톡", "퍽": "빡", "꽝": "콰광", "휴": "후우", "헉": "허억",
           "덜컥": "철컥", "끼익": "삐걱", "부스럭": "바스락", "스윽": "쓰윽", "번쩍": "반짝", "철썩": "찰싹", "덜덜": "부들"}

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"


def elongate(syl: str) -> str:
    """받침 있는 음절을 늘임: 쾅 → 콰+앙 = 콰앙, 받침 없으면 같은 모음으로: 휴 → 휴우"""
    if not is_hangul(syl):
        return syl + syl
    i = ord(syl) - 0xAC00
    cho, jung, jg = i // 588, (i % 588) // 28, i % 28
    head = chr(0xAC00 + cho * 588 + jung * 28)
    tail = chr(0xAC00 + _CHO.index("ㅇ") * 588 + jung * 28 + jg) if jg else chr(0xAC00 + _CHO.index("ㅇ") * 588 + jung * 28)
    return head + tail


def sfx_variant(s: str) -> str:
    core = re.sub(r"[^가-힣]", "", s)
    if not core:
        return s
    m = re.fullmatch(r"(.{1,2})\1+", core)  # 반복형 먼저: 쿵쿵 → 쿠웅쿠웅, 두근두근 → 콩닥콩닥
    if m:
        unit = m.group(1)
        new_unit = SFX_SYN.get(unit) or "".join(elongate(c) for c in unit)
        return s.replace(core, new_unit * (len(core) // len(unit)), 1)
    for k in sorted(SFX_SYN, key=len, reverse=True):
        if k in core:
            return s.replace(k, SFX_SYN[k], 1)
    return s.replace(core[-1], elongate(core[-1]), 1)


def classify(word: str, origin: str = "person") -> str:
    if word.endswith(PLACE_SUF) and len(word) >= 2:
        return "place"
    if word.endswith(ITEM_SUF) and len(word) >= 2:
        return "item"
    return "person" if origin == "person" else "term"  # 조어(마마신 등)는 사람 이름이 아니라 용어


def propose(word: str, kind: str, rng: np.random.Generator, used: set) -> str:
    def take(pool):
        for i in rng.permutation(len(pool)):
            if pool[i] not in used:
                return pool[i]
        return pool[0] + "2"
    if kind == "person":
        if len(word) == 3 and word[0] in SURNAMES:  # 한국식 성+이름 → 성+이름
            new = SURNAMES[int(rng.integers(30))] + take(GIVEN)
        else:
            new = take(PERSON_POOL)
    elif kind == "term":  # 용어: 마지막 글자(신·족·단 등)는 두고 앞부분만 → 마마신 → 백운신
        new = take(STEM_POOL)[: max(1, len(word) - 1)] + word[-1]
    else:
        suf = next((s for s in (PLACE_SUF if kind == "place" else ITEM_SUF) if word.endswith(s)), "")
        new = take(STEM_POOL) + suf if suf else take(STEM_POOL)
    used.add(new)
    return new


def extract(texts: list[str], kiwi, min_freq: int):
    """반환: {후보: (빈도, 예문, 종류)} — 종류: person(고유명사·호칭 앞) / coined(조어 명사 묶음)"""
    cnt, ex, kind = Counter(), {}, {}
    for t in texts:
        for eojeol in re.split(r"\s+", t):
            if not eojeol:
                continue
            toks = kiwi.tokenize(eojeol)
            run = []  # 붙어 있는 명사 묶음
            for tk in toks + [None]:
                if tk is not None and tk.tag in ("NNP", "NNG", "NNB", "NR", "SL"):
                    run.append(tk)
                    continue
                if run:
                    word = "".join(x.form for x in run)
                    has_nnp = any(x.tag == "NNP" for x in run)
                    compound = len(run) >= 2
                    nxt = tk.form if tk is not None else ""
                    honor = nxt in HONORIFIC or any(word.endswith(h) and len(word) > len(h) + 1 for h in ("씨",))
                    if (has_nnp or compound or honor) and 2 <= len(word) <= 6 and word not in STOP \
                            and re.fullmatch(r"[가-힣A-Za-z]+", word):
                        if has_nnp and word.endswith(("야", "아")) and len(word) >= 3:  # 세라야 → 세라 (호격)
                            word = word[:-1]
                        cnt[word] += 1
                        ex.setdefault(word, t.replace("\n", " ")[:40])
                        if has_nnp or honor:
                            kind[word] = "person"
                        else:
                            kind.setdefault(word, "coined")
                    run = []
    return {w: (c, ex[w], kind[w]) for w, c in cnt.most_common() if c >= min_freq}


def yq(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/default.yaml")
    ap.add_argument("--works", nargs="*")
    ap.add_argument("--min-freq", type=int, default=2, help="이보다 적게 나온 후보는 주석으로만 (기본 2)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    cfg = load_config(a.config)
    out = Path(cfg["out_root"])
    works_csv = Path(cfg["data_root"]) / "works.csv"
    authors = {r["work_id"]: r for r in csv.DictReader(open(works_csv, encoding="utf-8-sig"))} if works_csv.exists() else {}

    by_work: dict[str, dict[str, list]] = defaultdict(lambda: {"dialogue": [], "sfx": [], "credit": []})
    for r in M.originals_manifest(out).read():
        if a.works and r["work_id"] not in a.works:
            continue
        p = out / r["ocr_path"]
        if not p.exists():
            continue
        for b in json.load(open(p, encoding="utf-8"))["blocks"]:
            by_work[r["work_id"]][b["category"]].append(b["text"])

    tdir = Path(cfg["work_tables"])
    tdir.mkdir(parents=True, exist_ok=True)
    for w, d in sorted(by_work.items()):
        path = tdir / f"{w}.yaml"
        if path.exists() and not a.overwrite:
            print(f"[{w}] 이미 있음 → 건너뜀 ({path})")
            continue
        rng = np.random.default_rng(stable_seed("table", w))
        cands = extract(d["dialogue"], kiwi, 1)
        used: set = set(cands)
        groups: dict[str, list] = defaultdict(list)
        low = []
        for word, (c, ex, origin) in cands.items():
            if c < a.min_freq:
                low.append((word, c, ex, origin))
                continue
            kind = classify(word, origin)
            groups[kind].append((word, propose(word, kind, rng, used), c, ex))
        sfx = Counter(re.sub(r"\s+", "", s) for s in d["sfx"])
        info = authors.get(w, {})
        lines = [f"# {w} 치환표 초안 — tools/make_tables.py 자동 생성. 검토 후 version을 v1으로 바꾸세요.",
                 f"# 작품: {info.get('title', '?')} / 작가: {info.get('author', '?')}",
                 "# 틀린 후보(일반 명사, OCR 오독)는 줄을 지우고, 새 이름이 어색하면 바꾸세요.",
                 "# 표를 고친 뒤 version을 올리면 text·both 변형이 새 ID로 다시 만들어집니다.",
                 'version: "draft"', "", "entities:"]
        for kind in ("person", "place", "item", "term"):
            lines.append(f"  {kind}:" + ("" if groups[kind] else " {}"))
            for word, new, c, ex in groups[kind]:
                lines.append(f"    {yq(word)}: {yq(new)}    # {c}회 | {ex}")
        if low:
            lines.append(f"# 한 번만 나온 후보 (필요하면 위로 옮기기):")
            for word, c, ex, origin in low[:40]:
                k = classify(word, origin)
                lines.append(f"#    {yq(word)}: {yq(propose(word, k, rng, used))}    # [{k}] {ex}")
        lines += ["", "sfx:" + ("" if sfx else " {}")]
        for s, c in sfx.most_common():
            new = sfx_variant(s)
            if new != s:
                lines.append(f"  {yq(s)}: {yq(new)}    # {c}회")
        pen = PEN_POOL[stable_seed("pen", w) % len(PEN_POOL)]
        lines += ["", "credit:", "  replace:"]
        if info.get("author"):
            lines.append(f"    {yq(info['author'])}: {yq(pen)}")
        else:
            lines[-1] += " {}"
        lines += ["  default_text: null", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        n = sum(len(v) for v in groups.values())
        print(f"[{w}] 후보 {n}개(+한 번만 {len(low)}개), 효과음 {len(sfx)}개 → {path}")


if __name__ == "__main__":
    main()
