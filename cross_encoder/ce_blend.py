"""Choose how much the cross-encoder counts: final = ranker score + w * ce, over the ranker's top-K.

Both scores are out-of-fold (dump_ce.py, ce_train.py), so this is a vessel-disjoint estimate of
Hangul top-1 accuracy per source (train's own rows, LLM records) and condition (as_is, no_gt).

usage: python ce/ce_blend.py [--pairs analysis/ce_pairs.csv] [--oof ce/out/ce_oof.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def top1(d: pd.DataFrame, w: float, k: int) -> pd.Series:
    x = d[(d["rank"] >= 1) & (d["rank"] <= k)]
    x = x.assign(f=x["score"] + w * x["ce"])
    best = x.loc[x.groupby("rid")["f"].idxmax()]
    return best.groupby(["src", "cond"])["label"].mean()


def main(pairs: Path, oof: Path):
    d = pd.read_csv(pairs, encoding="utf-8-sig")
    ce = pd.read_csv(oof, encoding="utf-8-sig")
    d["ce"] = ce["ce"].to_numpy()  # same row order as ce_pairs.csv
    assert (ce["rid"].to_numpy() == d["rid"].to_numpy()).all()
    rows = []
    for k in (5, 10, 20):
        for w in (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 100):
            r = top1(d, w, k)
            rows.append({"k": k, "w": w, **{f"{s}/{c}": v for (s, c), v in r.items()}})
    res = pd.DataFrame(rows)
    res["score"] = 0.5 * res["synth/as_is"] + 0.5 * res["train/as_is"]  # pick on records with tonnage
    print(res.round(4).to_string(index=False))
    best = res.sort_values("score", ascending=False).iloc[0]
    print(f"\nbest: k={int(best.k)} w={best.w}")
    x = d[(d["rank"] >= 1) & (d["rank"] <= best.k)]
    for w in (0, best.w):
        f = x.assign(f=x["score"] + w * x["ce"])
        b = f.loc[f.groupby("rid")["f"].idxmax()]
        print(f"w={w}:", b[b["cond"] == "as_is"].groupby("style")["label"].mean().round(4).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "analysis" / "ce_pairs.csv")
    ap.add_argument("--oof", type=Path, default=ROOT / "ce" / "out" / "ce_oof.csv")
    a = ap.parse_args()
    main(a.pairs, a.oof)
