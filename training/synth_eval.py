"""Score a model on the LLM-written Korean records (llm_synth/synth.csv), per style.

Vessel-disjoint like stress_eval: in fold k, a synthetic record is scored by the fold-k ranker only when
the train row it was made from is held out in fold k. Records keep their train row's tonnage, so
"as_is" is the realistic condition; "no_gt" removes tonnage.

usage: MODEL_DIR=<model> HANJA=1 [EMB=1] python synth_eval.py --tag name
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


def main(data: Path, tag: str):
    t0 = time.time()
    tr, cat, idx, hang, folds, matchers, models = prepare_folds(data)
    syn = load_records(ROOT / "llm_synth" / "synth.csv")
    fold_of = np.empty(len(tr), dtype=int)
    for k, (_, b) in enumerate(folds):
        fold_of[b] = k
    syn["fold"] = fold_of[syn["source_row"].to_numpy()]
    names = cat.set_index("vessel_id")["vessel_name"]
    res = []
    from train import use_fold
    for k in range(len(folds)):
        use_fold(k)
        part = syn[syn["fold"] == k]
        for cond in ("as_is", "no_gt"):
            df = part if cond == "as_is" else part.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan)
            recs = df.to_dict("records")
            for s in range(0, len(recs), 256):
                for rec, (_, cands, X) in zip(recs[s:s + 256], matchers[k].candidates(recs[s:s + 256])):
                    pred = idx.ids[cands[int(np.argmax(models[k].predict(X)))]]
                    res.append((rec["style"], cond, pred == rec["target_vessel_id"],
                                bool((idx.ids[cands] == rec["target_vessel_id"]).any()),
                                rec["vessel_name"], names[rec["target_vessel_id"]], names[pred]))
        print(f"fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(res, columns=["style", "cond", "correct", "recall", "record", "true_name", "pred_name"])
    out = r.groupby(["cond", "style"]).agg(n=("correct", "size"), acc=("correct", "mean"), recall=("recall", "mean"))
    tot = r.groupby("cond").agg(n=("correct", "size"), acc=("correct", "mean"), recall=("recall", "mean"))
    tot.index = pd.MultiIndex.from_tuples([(c, "ALL") for c in tot.index])
    out = pd.concat([out, tot]).sort_index()
    print(out.round(4).to_string())
    (ROOT / "analysis").mkdir(exist_ok=True)
    out.to_csv(ROOT / "analysis" / f"synth_eval_{tag}.csv", encoding="utf-8-sig")
    r[~r.correct].to_csv(ROOT / "analysis" / f"synth_errors_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    ap.add_argument("--tag", default="latest")
    a = ap.parse_args()
    main(a.data, a.tag)
