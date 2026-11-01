"""Explain the greedy LLM model (submit_llm_fixed_greedy.zip) on a few records, on one's own screen.

The LLM itself needs a GPU, so its English readings come from train_readings_greedy.csv, which
the same 4-bit Qwen2.5-7B produced on a Kaggle GPU (greedy decoding, as on the server).
A name missing from that file is judged without the LLM (the base model), as the server would
if a reading failed. Example vessels are left out of the learned statistics, so each is judged
like an unseen vessel.

usage:
  python demo_llm_greedy.py                                # built-in examples
  python demo_llm_greedy.py --name "탄호이져" --top 8
  python demo_llm_greedy.py --name "모닝캐서린" --no-gt     # hide the tonnage
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "open"
ZIP = HERE / "submit_llm_fixed_greedy.zip"
READINGS = HERE / "teammate_llm" / "train_readings_greedy.csv"
EXAMPLES = [("테르모필레", True), ("모닝캐서린", True), ("에스티아이 슬론", True), ("장호 윈호", False), ("탄호이져", True)]


def load():
    work = Path(tempfile.mkdtemp(prefix="llm_demo_"))
    with zipfile.ZipFile(ZIP) as z:  # everything except the 5.8GB LLM weights
        for n in z.namelist():
            if not n.startswith("model/qwen7b_4bit/"):
                z.extract(n, work)
    sys.path.insert(0, str(work / "model"))
    from vm.core import FEATS, Catalog, Stats, featurize
    from vm.ko import load_dict
    t = time.time()
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    cat = Catalog(catalog, load_dict(work / "model" / "catalog_ko.csv"))
    full = [lgb.Booster(model_str=p.read_text(encoding="utf-8")) for p in sorted((work / "model").glob("lgb_[0-9]*.txt"))]
    base = [lgb.Booster(model_str=p.read_text(encoding="utf-8")) for p in sorted((work / "model").glob("lgb_base_*.txt"))]
    rd = pd.read_csv(READINGS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    readings = dict(zip(rd["name"], rd["en_guess"]))
    print(f"[모델] LLM greedy 버전: 랭커 {len(full)}개 + 예비 랭커 {len(base)}개, 특징 {len(FEATS)}개, "
          f"카탈로그 {len(catalog):,}척 ({time.time() - t:.0f}초)")
    print(f"[LLM] 서버에서는 Qwen2.5-7B(4비트)가 실행 중에 한글 기록을 영문으로 복원해요. "
          f"여기서는 같은 모델이 GPU에서 미리 만든 {len(readings):,}개 결과를 불러와요.\n")
    return FEATS, Stats, featurize, cat, catalog, full, base, readings


def explain(env, train, name, use_gt, top):
    FEATS, Stats, featurize, cat, catalog, full, base, readings = env
    rec = train[train["vessel_name"].astype(str).str.replace(" ", "") == name.replace(" ", "")]  # spacing-insensitive
    src = rec.iloc[0].to_dict() if len(rec) else {"vessel_name": name, "flag_name": np.nan, "vessel_type": np.nan}
    truth = src.get("target_vessel_id")
    seen = train[train["target_vessel_id"] != truth] if truth else train  # judge it like an unseen vessel
    st = Stats(seen, cat)
    row = {"vessel_name": name, "call_sign": "", "flag_name": src.get("flag_name"), "vessel_type": src.get("vessel_type"),
           "gross_tonnage": src.get("gross_tonnage") if use_gt else np.nan,
           "international_gross_tonnage": src.get("international_gross_tonnage") if use_gt else np.nan}
    reading = readings.get(name)
    if reading:
        row["en_guess"] = reading
    idx, F = featurize(row, cat, st)
    n_base = FEATS.index("en_exact")
    models, X = (full, F) if reading else (base, F[:, :n_base])
    score = np.mean([m.predict(X) for m in models], axis=0)
    order = np.argsort(-score)[:top]
    c = {f: FEATS.index(f) for f in ("en_ratio", "ko_ratio", "skel_ratio", "gap_min")}
    names = catalog["vessel_name"].astype(str).to_numpy()
    gts = pd.to_numeric(catalog["gross_tonnage"], errors="coerce").to_numpy()
    gt = row["gross_tonnage"]
    print("=" * 100)
    print(f"기록: {name!r}   톤수: {('없음' if pd.isna(gt) else f'{gt:,.0f}')}")
    print(f"  LLM 영문 복원: {reading!r}" if reading else "  LLM 영문 복원: 없음 → LLM 없이 예비 모델로 판단")
    print(f"  후보 {len(idx)}척 검색 → 랭커가 점수를 매겨 1위를 고름\n")
    print(f"  {'순위':<4}{'카탈로그 선박명':<26}{'톤수':>9}{'점수':>7}{'영문복원 유사도':>15}{'한글표기 유사도':>15}{'발음 유사도':>11}{'톤수 차이':>10}")
    for r, j in enumerate(order, 1):
        i = idx[j]
        def pct(v): return "-" if v < 0 else f"{v * 100:.0f}"
        mark = "  ← 정답" if truth and cat.ids[i] == truth else ""
        print(f"  {r:<4}{names[i][:24]:<26}{gts[i]:>9.0f}{score[j]:>7.3f}{pct(F[j, c['en_ratio']]):>15}"
              f"{pct(F[j, c['ko_ratio']]):>15}{pct(F[j, c['skel_ratio']]):>11}{min(F[j, c['gap_min']], 1):>10.1%}{mark}")
    ok = truth and cat.ids[idx[order[0]]] == truth
    print(f"\n  → 예측: {names[idx[order[0]]]}" + ("  (정답)" if ok else ("  (오답)" if truth else "")) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--no-gt", action="store_true"); ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    env = load()
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    train = train[train["target_vessel_id"].isin(env[3].id2i)]
    for name, use_gt in ([(a.name, not a.no_gt)] if a.name else EXAMPLES):
        explain(env, train, name, use_gt, a.top)


if __name__ == "__main__":
    main()
