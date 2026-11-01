"""Hangul rows, tonnage removed: rank of the true vessel under each name key (vessel-disjoint folds).

usage: python recall_at_k.py [--data ../open]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "inference" / "model"))
sys.path.insert(0, str(ROOT))
from rapidfuzz import fuzz  # noqa: E402
from vessel_lib import CatalogIndex, Query, _cdist, is_hangul, skeleton  # noqa: E402
from train import N_FOLDS, fit_aux  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

KS = [40, 100, 200, 300, 500, 1000]


def ranks(sim: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    s = sim.astype(np.int16)
    t = s[np.arange(len(s)), true_idx][:, None]
    return (s > t).sum(1) + 1  # optimistic tie handling: ties counted as found


def main(data: Path):
    tr = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(data / "vessel_catalog.csv", encoding="utf-8-sig")
    idx = CatalogIndex(cat)
    idx.skel_c = [skeleton(k) for k in idx.pcomp]
    hang = tr["vessel_name"].map(is_hangul).to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr["target_vessel_id"]))
    out = {k: [] for k in ["pcomp", "vkey", "acomp", "skel", "merged_max", "union_keys"]}
    worst = []
    for a, b in folds:
        _, token_dict, _, _ = fit_aux(tr.iloc[a], cat)
        rows = [i for i in b if hang[i]]
        recs = tr.iloc[rows].assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan).to_dict("records")
        qs = [Query(r, token_dict) for r in recs]
        t = np.array([idx.id_to_index[r["target_vessel_id"]] for r in recs])
        sims = {
            "pcomp": _cdist([q.pcomp for q in qs], idx.pcomp, fuzz.ratio),
            "vkey": _cdist([q.vkey for q in qs], idx.vkey, fuzz.ratio),
            "acomp": _cdist([q.acomp if q.dict_cov > 0 else "" for q in qs], idx.acomp, fuzz.ratio),
            "skel": _cdist([q.skel for q in qs], idx.skel_c, fuzz.ratio),
        }
        merged = np.maximum.reduce([sims["pcomp"], sims["vkey"], sims["acomp"]])
        r = {k: ranks(v, t) for k, v in sims.items()}
        r["merged_max"] = ranks(merged, t)
        r["union_keys"] = np.minimum.reduce([r["pcomp"], r["vkey"], r["acomp"], r["skel"]])
        for k in out:
            out[k].append(r[k])
        for q, rec, rk in zip(qs, recs, r["union_keys"]):
            worst.append((rk, rec["vessel_name"], q.pcomp, cat["vessel_name"].iat[idx.id_to_index[rec["target_vessel_id"]]],
                          idx.pcomp[idx.id_to_index[rec["target_vessel_id"]]]))
    print(f"recall@K for {sum(len(x) for x in out['pcomp'])} Hangul rows, tonnage removed")
    print(f"{'key':12s}" + "".join(f"{'@' + str(k):>8s}" for k in KS))
    for k, v in out.items():
        v = np.concatenate(v)
        print(f"{k:12s}" + "".join(f"{(v <= K).mean():8.4f}" for K in KS))
    worst.sort(key=lambda x: -x[0])
    print("\nhardest (union rank, query, query key, true name, true key):")
    for w in worst[:30]:
        print(w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    main(ap.parse_args().data)
