"""Explain the H4 model on a few records: how a record is read, which catalog vessels are
retrieved, and how the ranker scores them. For showing the method on one's own screen.

usage:
  python demo_model.py                                   # built-in examples
  python demo_model.py --name "테르모필래" --gt 80219      # any record (tonnage optional)
  python demo_model.py --name "모닝캐서린" --top 8
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "open"
EXAMPLES = [  # (record name, gross tonnage or None) — spelled the way port records write them
    ("테르모필레", 80219),
    ("모닝캐서린", 60876),      # written without the space
    ("에스티아이 슬론", 64827),
    ("장호 윈호", None),        # tonnage missing
    ("STI GUARD", 63965),
]


def load_model():
    work = Path(tempfile.mkdtemp(prefix="h4_demo_"))
    zipfile.ZipFile(HERE / "submit_H4.zip").extractall(work)
    sys.path.insert(0, str(work)); sys.path.insert(0, str(work / "model"))
    import script  # H4's own entry point: same loading code as on the server
    import vessel_lib
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    t = time.time()
    m = script.load_matcher(catalog)
    print(f"[모델] H4: LightGBM 랭커 {len(m.booster.models)}개 평균, 특징 {len(vessel_lib.FEATURES)}개, "
          f"카탈로그 {len(catalog):,}척 색인 ({time.time() - t:.0f}초)\n")
    return m, vessel_lib, catalog


def explain(m, V, catalog, name, gt, top):
    row = {"vessel_name": name, "gross_tonnage": gt if gt is not None else np.nan,
           "international_gross_tonnage": gt if gt is not None else np.nan, "flag_name": np.nan,
           "vessel_type": np.nan, "call_sign": ""}
    q, cands, X = next(iter(m.candidates([row])))
    score = m.booster.predict(X)
    order = np.argsort(-score)[:top]
    F = V.FEATURES
    col = {f: F.index(f) for f in ("p_ratio", "ko_ratio", "gap_min", "tset")}
    print("=" * 96)
    print(f"기록: {name!r}   톤수: {gt if gt is not None else '없음'}")
    if q.hangul:
        print(f"  한글 → 영문 추정: {q.clean!r}   (사전 적용 비율 {q.dict_cov:.0%})")
        print(f"  발음 키: {q.pcomp!r}   한글 비교 키: {q.ko_k!r}")
    else:
        print(f"  정리된 이름: {q.clean!r}   발음 키: {q.pcomp!r}")
    print(f"  후보 {len(cands)}척 검색 → 랭커가 점수를 매겨 1위를 고름\n")
    ko = dict(zip(catalog["vessel_name"].astype(str), catalog["vessel_name"].astype(str)))
    names = catalog["vessel_name"].astype(str).to_numpy()
    gts = pd.to_numeric(catalog["gross_tonnage"], errors="coerce").to_numpy()
    print(f"  {'순위':<4}{'카탈로그 선박명':<26}{'톤수':>9}{'랭커 점수':>10}{'발음 유사도':>11}{'한글표기 유사도':>15}{'톤수 차이':>10}")
    for r, j in enumerate(order, 1):
        i = cands[j]
        kr = X[j, col["ko_ratio"]]
        print(f"  {r:<4}{names[i][:24]:<26}{gts[i]:>9.0f}{score[j]:>10.2f}{X[j, col['p_ratio']]:>11.0f}"
              f"{(f'{kr:.0f}' if kr >= 0 else '-'):>15}{X[j, col['gap_min']]:>10.1%}")
    print(f"\n  → 예측: {names[cands[order[0]]]}  (한글표기 = 카탈로그를 LLM으로 미리 한글로 바꾼 표기와의 유사도)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--gt", type=float); ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    m, V, catalog = load_model()
    for name, gt in ([(a.name, a.gt)] if a.name else EXAMPLES):
        explain(m, V, catalog, name, gt, a.top)


if __name__ == "__main__":
    main()
