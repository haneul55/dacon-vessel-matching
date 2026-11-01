"""Out-of-fold top-K candidates of the ranker for every Hangul record we have, for the cross-encoder.

Records: train's own Hangul rows (clean, no_gt) and the LLM-written records of train vessels
(llm_synth/synth.csv; as_is, no_gt). In fold k a record is ranked by the fold-k ranker, which never
saw that vessel (vessel-disjoint, like stress_eval / synth_eval). The true vessel is appended with
rank -1 when the ranker's top-K misses it, so the cross-encoder always sees a positive in training;
evaluation uses rank >= 1 rows only.

Output analysis/ce_pairs.csv:
  rid, src (train|synth), style, cond, fold, query, cand, cand_id, score (ranker), rank, label

usage: MODEL_DIR=<model> HANJA=1 EMB_FT=<labse_ft/out> SYNTH_AUG=<synth.csv> N_SEEDS=1 python dump_ce.py [--k 20]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")
from stress_eval import ROOT, prepare_folds  # noqa: E402

sys.path.insert(0, str(ROOT / "llm_synth"))
from load_synth import load_records  # noqa: E402


def main(data: Path, k_top: int, out: Path):
    t0 = time.time()
    tr, cat, idx, hang, folds, matchers, models = prepare_folds(data)
    from train import use_fold
    names = cat["vessel_name"].astype(str).to_numpy()
    pos = {v: i for i, v in enumerate(idx.ids)}
    syn = load_records(ROOT / "llm_synth" / "synth.csv")
    fold_of = np.empty(len(tr), dtype=int)
    for k, (_, b) in enumerate(folds):
        fold_of[b] = k
    syn["fold"] = fold_of[syn["source_row"].to_numpy()]
    th = tr[hang].assign(fold=fold_of[hang], style="train")
    rows, rid = [], 0
    for k in range(len(folds)):
        use_fold(k)
        for src, part in (("train", th[th["fold"] == k]), ("synth", syn[syn["fold"] == k])):
            for cond in ("as_is", "no_gt"):
                df = part if cond == "as_is" else part.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan)
                recs = df.to_dict("records")
                for s in range(0, len(recs), 256):
                    for rec, (_, cands, X) in zip(recs[s:s + 256], matchers[k].candidates(recs[s:s + 256])):
                        sc = models[k].predict(X)
                        order = np.argsort(-sc, kind="stable")[:k_top]
                        top = cands[order]
                        t = pos[rec["target_vessel_id"]]
                        base = (rid, src, rec["style"], cond, k, rec["vessel_name"])
                        rows += [base + (names[c], idx.ids[c], float(sc[o]), r + 1, int(c == t))
                                 for r, (c, o) in enumerate(zip(top, order))]
                        if t not in set(top):
                            hit = np.flatnonzero(cands == t)
                            rows.append(base + (names[t], idx.ids[t], float(sc[hit[0]]) if len(hit) else np.nan, -1, 1))
                        rid += 1
            print(f"fold {k} {src} done ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows, columns=["rid", "src", "style", "cond", "fold", "query", "cand", "cand_id", "score",
                                    "rank", "label"])
    d.to_csv(out, index=False, encoding="utf-8-sig")
    top1 = d[d["rank"] == 1].groupby(["src", "cond"])["label"].mean()
    inK = d[d["rank"] >= 1].groupby(["src", "cond", "rid"])["label"].max().groupby(["src", "cond"]).mean()
    print(pd.DataFrame({"top1": top1, f"in_top{k_top}": inK}).round(4))
    print(f"saved {out} ({len(d)} rows, {d['rid'].nunique()} records, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "ce_pairs.csv")
    a = ap.parse_args()
    main(a.data, a.k, a.out)
