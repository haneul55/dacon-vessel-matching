"""How much the LLM's multiple-choice answer (llm_mc.py) and the cross-encoder (ce_train.py) should count
on top of the ranker, over the ranker's top-K for Hangul records.

final = ranker score + w_llm * LLM log-prob + w_ce * cross-encoder score, applied only when the ranker's
top-2 margin is below `gate` (a confident ranker is left alone). All three scores are out-of-fold
or untrained, so the estimate is vessel-disjoint.

usage: python llm/llm_blend.py [--mc llm/out/llm_mc.csv] [--ce ce/out/ce_oof.csv]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def evaluate(d: pd.DataFrame, w_llm: float, w_ce: float, gate: float) -> pd.Series:
    f = d["score"] + w_llm * d["logp"] + w_ce * d["ce"]
    f = np.where(d["margin"] < gate, f, d["score"])
    x = d.assign(f=f)
    best = x.loc[x.groupby("rid")["f"].idxmax()]
    return best.groupby(["src", "cond"])["label"].mean()


def main(pairs: Path, mc: Path, ce: Path | None):
    d = pd.read_csv(pairs, encoding="utf-8-sig")
    d["ce"] = pd.read_csv(ce, encoding="utf-8-sig")["ce"].to_numpy() if ce and ce.exists() else 0.0
    s1 = d[d["rank"] == 1].set_index("rid")["score"]
    s2 = d[d["rank"] == 2].set_index("rid")["score"]
    d["margin"] = d["rid"].map((s1 - s2).fillna(9e9))  # ranker's confidence: top-1 minus top-2 score
    m = pd.read_csv(mc, encoding="utf-8-sig")
    d = d.merge(m[["rid", "cand_id", "logp"]], on=["rid", "cand_id"], how="inner")  # top-K rows the LLM saw
    w_ces = (0, 0.5, 1, 2) if (ce and ce.exists()) else (0,)
    rows = []
    for w_llm, w_ce, gate in itertools.product((0, 0.25, 0.5, 1, 2, 4), w_ces, (0.5, 1, 2, 4, 1e9)):
        r = evaluate(d, w_llm, w_ce, gate)
        rows.append({"w_llm": w_llm, "w_ce": w_ce, "gate": gate, **{f"{a}/{b}": v for (a, b), v in r.items()}})
    res = pd.DataFrame(rows)
    res["pick"] = 0.5 * res["synth/as_is"] + 0.5 * res["train/as_is"]
    base = res[(res.w_llm == 0) & (res.w_ce == 0)].iloc[0]
    print("ranker alone:", base.drop(["w_llm", "w_ce", "gate", "pick"]).round(4).to_dict())
    print(res.sort_values("pick", ascending=False).head(15).round(4).to_string(index=False))
    b = res.sort_values("pick", ascending=False).iloc[0]
    print(f"\nbest: w_llm={b.w_llm} w_ce={b.w_ce} gate={b.gate}")
    for w_llm, w_ce in ((0, 0), (b.w_llm, b.w_ce)):
        f = np.where(d["margin"] < b.gate, d["score"] + w_llm * d["logp"] + w_ce * d["ce"], d["score"])
        x = d.assign(f=f)
        best = x.loc[x.groupby("rid")["f"].idxmax()]
        print(f"w_llm={w_llm} w_ce={w_ce} by style (as_is):",
              best[best["cond"] == "as_is"].groupby("style")["label"].mean().round(4).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=ROOT / "analysis" / "ce_pairs.csv")
    ap.add_argument("--mc", type=Path, default=ROOT / "llm" / "out" / "llm_mc.csv")
    ap.add_argument("--ce", type=Path, default=ROOT / "ce" / "out" / "ce_oof.csv")
    a = ap.parse_args()
    main(a.pairs, a.mc, a.ce)
