"""Turn synth.csv (Kaggle LLM output for train vessels) into train-like records: the Korean name replaces
the name of one of that vessel's own train rows, everything else (tonnage, flag, type, call sign) is kept.
Allowed per DACON (talk #418501): LLM augmentation built from train data.

usage (check): python load_synth.py synth.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "open"  # open_data/open


def clean_ok(ko: str) -> bool:
    """Keep Hangul/digit spellings only (the 7B model sometimes leaves Latin letters or emoji: 숑LTE, 트라이😉프)
    with at least 2 Hangul syllables."""
    return bool(re.fullmatch(r"[가-힣0-9 \-.]+", ko.strip())) and len(re.findall("[가-힣]", ko)) >= 2


def load_records(synth_path, seed: int = 0, min_cos: float | None = None) -> pd.DataFrame:
    """min_cos: drop records whose LaBSE similarity to the true English name (score_synth.py) is below it;
    vessel-type words the model wrote instead of a name (화학물류선, 상품 운반선) are dropped too."""
    rng = np.random.default_rng(seed)
    syn = pd.read_csv(synth_path, encoding="utf-8-sig").dropna(subset=["ko"])
    syn = syn[syn["ko"].map(clean_ok)]
    if min_cos is not None:
        q = pd.read_csv(Path(synth_path).with_name(Path(synth_path).stem + "_quality.csv"), encoding="utf-8-sig")
        syn = syn.merge(q[["vessel_id", "ko", "cos"]].drop_duplicates(["vessel_id", "ko"]), on=["vessel_id", "ko"], how="left")
        syn = syn[(syn["cos"] >= min_cos) & ~syn["ko"].str.contains("운반선|화학물|물류선|화물선|유조선")]
    # a mixed/short/meaning spelling identical to the sound one adds nothing
    sound = syn[syn["style"] == "sound"].set_index("vessel_id")["ko"].str.replace(" ", "")
    same = (syn["style"] != "sound") & (syn["ko"].str.replace(" ", "") == syn["vessel_id"].map(sound))
    syn = syn[~same].drop_duplicates(["vessel_id", "ko"])
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    rows_of = tr.groupby("target_vessel_id").indices
    syn = syn[syn["vessel_id"].isin(rows_of)]
    pick = [rows_of[v][rng.integers(len(rows_of[v]))] for v in syn["vessel_id"]]
    out = tr.iloc[pick].copy()
    out["source_row"] = pick  # the train row it was made from (keeps folds vessel-disjoint)
    out["sample_id"] = [f"SYN_{k:06d}" for k in range(len(out))]
    out["vessel_name"] = syn["ko"].to_numpy()
    out["style"] = syn["style"].to_numpy()
    return out.reset_index(drop=True)


if __name__ == "__main__":
    d = load_records(sys.argv[1])
    print(d.head(20).to_string()); print(d["style"].value_counts()); print("tonnage missing", d.gross_tonnage.isna().mean())
