"""한국어 처리: 받침 판정, 조사 보정, 자모 단위 비교(OCR 오독 이름 매칭)."""
from __future__ import annotations

import re

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]


def is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def jong(ch: str) -> int:
    """종성 인덱스 (0 = 받침 없음). 한글이 아니면 숫자/영문 끝을 대략 판정."""
    if is_hangul(ch):
        return (ord(ch) - 0xAC00) % 28
    if ch.isdigit():
        return 0 if ch in "2459" else 8 if ch in "178" else 1
    return 0


def decompose(s: str) -> str:
    out = []
    for ch in s:
        if is_hangul(ch):
            i = ord(ch) - 0xAC00
            out += [_CHO[i // 588], _JUNG[(i % 588) // 28], _JONG[i % 28]]
        else:
            out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def jamo_close(a: str, b: str, max_dist: int = 1) -> bool:
    return len(a) == len(b) and a != b and edit_distance(decompose(a), decompose(b)) <= max_dist


# (받침 있을 때, 받침 없을 때) — 긴 것부터 매칭
JOSA_PAIRS = [
    ("이에요", "예요"), ("이라고", "라고"), ("이라는", "라는"), ("이랑", "랑"), ("이나", "나"),
    ("이며", "며"), ("이여", "여"), ("이야", "야"), ("으로", "로"), ("이", "가"), ("은", "는"),
    ("을", "를"), ("과", "와"), ("아", "야"),
]
_JOSA_ALL = sorted({p for pair in JOSA_PAIRS for p in pair}, key=len, reverse=True)


def fix_josa(word: str, josa: str, orig_word: str) -> tuple[str, bool]:
    """word 뒤에 붙을 조사를 받침에 맞게 바꿈. (조사, 애매했는지)"""
    if not josa or not word:
        return josa, False
    j = jong(word[-1])
    has, is_rieul = j != 0, j == 8
    ambiguous = False
    for a, b in JOSA_PAIRS:
        if josa not in (a, b):
            continue
        if josa == "야" and jong(orig_word[-1]) == 0:
            ambiguous = True  # 호격 '야'(→아) / 서술격 '야'(→이야) 구분 불가 → 호격으로
        if (a, b) == ("으로", "로"):
            return ("으로" if has and not is_rieul else "로"), ambiguous
        return (a if has else b), ambiguous
    return josa, False


def substitute_entities(text: str, mapping: dict[str, str], fuzzy: bool = True) -> tuple[str, list[str]]:
    """이름 치환 + 조사 보정. 반환: (새 문장, 적용 규칙 목록)"""
    rules: list[str] = []
    if not mapping:
        return text, rules
    names = sorted(mapping, key=len, reverse=True)
    josa_re = "|".join(map(re.escape, _JOSA_ALL))
    pat = re.compile(rf"({'|'.join(map(re.escape, names))})({josa_re})?")

    def rep(m: re.Match) -> str:
        old, josa = m.group(1), m.group(2) or ""
        new = mapping[old]
        j2, amb = fix_josa(new, josa, old)
        rules.append(f"entity:{old}→{new}")
        if j2 != josa:
            rules.append(f"josa:{josa}→{j2}" + ("(ambiguous)" if amb else ""))
        return new + j2

    text = pat.sub(rep, text)

    if fuzzy:  # OCR이 자모 하나 틀리게 읽은 이름도 치환 (예: 카일 → 카닐)
        def fuzzy_rep(m: re.Match) -> str:
            tok = m.group(0)
            for old in names:
                n = len(old)
                if len(tok) >= n and jamo_close(tok[:n], old) and tok[:n] not in mapping.values():
                    rest = tok[n:]
                    josa = rest if rest in _JOSA_ALL else ""
                    j2, _ = fix_josa(mapping[old], josa, old) if josa else ("", False)
                    rules.append(f"entity_fuzzy:{tok[:n]}≈{old}→{mapping[old]}")
                    return mapping[old] + (j2 if josa else rest)
            return tok
        text = re.sub(r"[가-힣]+", fuzzy_rep, text)
    return text, rules
