"""Test-like corruptions of train rows, shared by train.py (augmentation) and stress_eval.py.
Training-time only: not part of submit/."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_LAT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# jamo indices (initial / medial / final) of common Korean transliteration confusions
_L_SWAP = {0: 15, 15: 0, 3: 16, 16: 3, 7: 17, 17: 7, 12: 14, 14: 12, 9: 10, 10: 9}  # ㄱㅋ ㄷㅌ ㅂㅍ ㅈㅊ ㅅㅆ
_V_SWAP = {1: 5, 5: 1, 4: 8, 8: 4, 18: 13, 13: 18, 6: 12, 12: 6, 10: 11, 11: 10}  # ㅐㅔ ㅓㅗ ㅡㅜ ㅕㅛ ㅙㅚ
_T_SWAP = {4: 21, 21: 4, 0: 4, 8: 4, 1: 0, 16: 0, 19: 0}  # ㄴㅇ, add ㄴ, ㄹ->ㄴ, drop ㄱ/ㅁ/ㅅ
_HANGUL = re.compile(r"[가-힣]")


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and bool(_HANGUL.search(value))


def _syl(L, V, T):
    return chr(0xAC00 + L * 588 + V * 28 + T)


def hangul_jamo_typo(name: str, rng) -> str:
    pos = [i for i, ch in enumerate(name) if 0xAC00 <= ord(ch) <= 0xD7A3 and ch not in "호제"]
    if not pos:
        return name
    i = pos[rng.integers(len(pos))]
    code = ord(name[i]) - 0xAC00
    L, V, T = code // 588, (code % 588) // 28, code % 28
    for kind in rng.permutation(3):
        if kind == 0 and L in _L_SWAP:
            L = _L_SWAP[L]; break
        if kind == 1 and V in _V_SWAP:
            V = _V_SWAP[V]; break
        if kind == 2 and T in _T_SWAP:
            T = _T_SWAP[T]; break
    return name[:i] + _syl(L, V, T) + name[i + 1:]


def hangul_spacing(name: str, rng) -> str:
    if " " in name:
        return name.replace(" ", "")
    h = [i for i, ch in enumerate(name) if 0xAC00 <= ord(ch) <= 0xD7A3]
    if len(h) < 3:
        return name
    i = h[rng.integers(1, len(h) - 1)]
    return name[:i] + " " + name[i:]


def hangul_suffix(name: str, rng) -> str:
    """Toggle the '호' suffix and the '제N' prefix convention."""
    s = name.strip()
    if s.endswith("호"):
        return s[:-1]
    m = re.search(r"\s*(\d+)\s*$", s)
    if m:
        return "제" + m.group(1) + s[:m.start()].strip() + "호"
    return s + "호"


def latin_typo(name: str, rng, n: int = 1) -> str:
    s = name
    for _ in range(n):
        pos = [i for i, ch in enumerate(s) if ch.isalpha()]
        if len(pos) < 3:
            return s
        i = pos[rng.integers(len(pos))]
        op = rng.integers(4)
        if op == 0:
            s = s[:i] + s[i + 1:]                                      # delete
        elif op == 1:
            s = s[:i] + s[i] + s[i:]                                   # double (VICCTORY)
        elif op == 2:
            s = s[:i] + _LAT[rng.integers(26)] + s[i + 1:]             # substitute
        elif i + 1 < len(s) and s[i + 1].isalpha():
            s = s[:i] + s[i + 1] + s[i] + s[i + 2:]                    # transpose
    return s


def latin_spacing(name: str, rng) -> str:
    if " " in name:
        return name.replace(" ", "", 1)
    if len(name) < 5:
        return name
    i = int(rng.integers(2, len(name) - 2))
    return name[:i] + " " + name[i:]


def no_gt(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan)


def gt_noise(df: pd.DataFrame, rng, lo: float = 0.03, hi: float = 0.10) -> pd.DataFrame:
    f = 1 + rng.choice([-1, 1], len(df)) * rng.uniform(lo, hi, len(df))
    return df.assign(gross_tonnage=(df["gross_tonnage"] * f).round(),
                     international_gross_tonnage=(df["international_gross_tonnage"] * f).round())


def map_name(df: pd.DataFrame, fn, rng) -> pd.DataFrame:
    return df.assign(vessel_name=[fn(n, rng) if isinstance(n, str) else n for n in df["vessel_name"]])


def name_typo(name: str, rng) -> str:
    """One random test-like name corruption (Hangul: jamo/spacing/호; latin: typo/spacing)."""
    if is_hangul(name):
        fn = (hangul_jamo_typo, hangul_jamo_typo, hangul_spacing, hangul_suffix)[rng.integers(4)]
        return fn(name, rng)
    return latin_typo(name, rng, 1) if rng.random() < 0.75 else latin_spacing(name, rng)
