"""CPU: combine E's ranker scores with LLM scores for Hangul rows and pick the setting.

final = ranker_score + w * (llm - max llm in the row), applied only when E's top1-top2 margin < tau,
over E's top-K. Grid over (K, w, tau, norm); norm = "sum" (lp) or "tok" (lp / ntok).
Reports per-condition Hangul accuracy (E alone vs best), plus an honest leave-one-fold-out
estimate (setting chosen on 4 folds, scored on the 5th). Writes the chosen setting to
submit_llm/model/llm_config.json when --write is given.

usage: python combine_llm.py llm/llm_scores_q7b.csv [--write]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KS, WS, TAUS = [3, 5, 10], [0.0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 1e6], [np.inf, 4, 2, 1, 0.5, 0.25]
NORMS = ["sum", "tok"]
CONDS = ["clean", "no_gt", "gt_noise_3-10%", "hangul_jamo_typo", "typo + no_gt", "name_only"]


def load(topk: Path, scores: Path) -> pd.DataFrame:
    df = pd.read_csv(topk, encoding="utf-8-sig")
    sc = pd.read_csv(scores, encoding="utf-8-sig")
    df = df.merge(sc, on=["sample_id", "cand_id"], how="left")
    df["tok"] = df["lp"] / df["ntok"]
    df["sum"] = df["lp"]
    df = df.sort_values(["sample_id", "rank"]).reset_index(drop=True)
    # rows arrive sorted by E's score, so the top1-top2 margin is rank 1 minus rank 2
    s1 = df[df["rank"] == 1].set_index("sample_id")["ranker_score"]
    s2 = df[df["rank"] == 2].set_index("sample_id")["ranker_score"].reindex(s1.index).fillna(-np.inf)
    df["margin"] = df["sample_id"].map(s1 - s2)
    return df


def evaluate(df: pd.DataFrame, k: int, w: float, tau: float, norm: str) -> pd.Series:
    """Per-row correctness (index = sample_id) of the combined pick."""
    d = df[df["rank"] <= k]
    rel = d[norm] - d.groupby("sample_id", sort=False)[norm].transform("max")
    use = (d["margin"] < tau) & d[norm].notna()
    score = d["ranker_score"].to_numpy() + np.where(use, w * rel.fillna(0).to_numpy(), 0.0)
    best = pd.Series(score, index=d.index).groupby(d["sample_id"], sort=False).idxmax()
    return pd.Series(d.loc[best.to_numpy(), "label"].astype(bool).to_numpy(), index=best.index)


def table(df: pd.DataFrame, correct: pd.Series) -> pd.Series:
    cond = df.drop_duplicates("sample_id").set_index("sample_id")["condition"]
    return correct.groupby(cond.loc[correct.index]).mean().reindex(CONDS)


def objective(t: pd.Series, base: pd.Series) -> float:
    """Mean accuracy over conditions; never accept a drop on clean data."""
    return -1.0 if t["clean"] < base["clean"] - 1e-9 else float(t.mean())


def main(scores: Path, write: bool):
    df = load(ROOT / "analysis" / "topk_han.csv", scores)
    print(f"LLM scores cover {df['lp'].notna().mean():.1%} of the top-10 pairs")
    grid = list(itertools.product(KS, WS, TAUS, NORMS))
    folds = sorted(df["fold"].unique())

    def search(part: pd.DataFrame):
        base = table(part, evaluate(part, 10, 0.0, np.inf, "sum"))
        best = max(grid, key=lambda p: objective(table(part, evaluate(part, *p)), base))
        return best, base

    best, base = search(df)
    t_best = table(df, evaluate(df, *best))
    print(f"\nbest setting K={best[0]} w={best[1]} tau={best[2]} norm={best[3]}  (chosen on all folds)")
    print(pd.DataFrame({"E_alone": base, "E+LLM": t_best}).round(4).to_string())

    # honest estimate: choose on 4 folds, score the held-out fold
    held = []
    for f in folds:
        p, _ = search(df[df["fold"] != f])
        part = df[df["fold"] == f]
        held.append(evaluate(part, *p))
        print(f"  fold {f}: chose K={p[0]} w={p[1]} tau={p[2]} norm={p[3]}")
    t_cv = table(df, pd.concat(held))
    print("\nleave-one-fold-out estimate")
    print(pd.DataFrame({"E_alone": base, "E+LLM (CV)": t_cv}).round(4).to_string())
    n_rows = df.drop_duplicates("sample_id").groupby("condition").size().reindex(CONDS)
    print("\nrows changed vs E (best setting):",
          ((t_best - base) * n_rows).round().astype(int).to_dict())

    if write:
        out = ROOT / "submit_llm" / "model" / "llm_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg = dict(k=best[0], w=best[1], tau=None if np.isinf(best[2]) else best[2], norm=best[3],
                   scores_file=scores.name)
        out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"\nwrote {out}: {cfg}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    main(a.scores, a.write)
