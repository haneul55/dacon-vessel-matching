"""Export E's top-K candidates for every held-out Hangul row (vessel-disjoint folds), under
clean and test-like conditions, in the format the LLM scoring script reads.

Output analysis/topk_han.csv columns:
  sample_id (condition|original id), condition, fold, query (Korean name shown to the model),
  cand (catalog English name), cand_id, ranker_score, rank (1 = E's pick), label (1 = true vessel)

usage: python export_topk.py [--data ../open] [--k 10]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stress_eval import CONDITIONS, ROOT, prepare_folds

EXPORT_CONDITIONS = ["clean", "no_gt", "gt_noise_3-10%", "hangul_jamo_typo", "typo + no_gt", "name_only"]


def main(data: Path, k_top: int, out: Path):
    t0 = time.time()
    tr, cat, idx, hang, folds, matchers, models = prepare_folds(data)
    names = cat["vessel_name"].to_numpy()
    rows_out, summary = [], []
    for k, (_, b) in enumerate(folds):
        base = tr.iloc[[i for i in b if hang[i]]].reset_index(drop=True)
        for ci, cname in enumerate(EXPORT_CONDITIONS):
            ci_global = list(CONDITIONS).index(cname)  # same random draws as stress_eval.py
            df = CONDITIONS[cname](base, np.random.default_rng(1000 * k + ci_global))
            recs = df.to_dict("records")
            for s in range(0, len(recs), 256):
                for rec, (_, cands, X) in zip(recs[s:s + 256], matchers[k].candidates(recs[s:s + 256])):
                    sc = models[k].predict(X)
                    order = np.argsort(-sc, kind="stable")[:k_top]
                    ids = idx.ids[cands[order]]
                    label = (ids == rec["target_vessel_id"]).astype(int)
                    sid = f"{cname}|{rec['sample_id']}"
                    for r, (j, cid, lab) in enumerate(zip(order, ids, label), start=1):
                        rows_out.append(dict(sample_id=sid, condition=cname, fold=k, query=rec["vessel_name"],
                                             cand=names[idx.id_to_index[cid]], cand_id=cid,
                                             ranker_score=float(sc[j]), rank=r, label=int(lab)))
                    summary.append((cname, int(label[0]), int(label.any())))
        print(f"fold {k} exported ({time.time() - t0:.0f}s)", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(out / "topk_han.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summary, columns=["condition", "top1", f"in_top{k_top}"]).groupby("condition").mean()
    print(s.reindex(EXPORT_CONDITIONS).round(4).to_string())
    print(f"{df['sample_id'].nunique()} groups, {len(df)} pairs -> {out / 'topk_han.csv'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis")
    a = ap.parse_args()
    main(a.data, a.k, a.out)
