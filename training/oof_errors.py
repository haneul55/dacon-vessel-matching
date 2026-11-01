"""Vessel-disjoint 5-fold OOF evaluation (same protocol as train.py steps 1-2) that also
dumps every wrong row. Writes nothing into submit/.

usage: python oof_errors.py [--data ../open] [--out analysis]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "inference" / "model"))
sys.path.insert(0, str(ROOT))
from vessel_lib import FEATURES, CatalogIndex, Matcher, is_hangul  # noqa: E402
from train import N_FOLDS, build, fit_aux, fit_ranker, load_index, score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402


def main(data: Path, out: Path):
    t0 = time.time()
    tr = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(data / "vessel_catalog.csv", encoding="utf-8-sig")
    idx = load_index(cat)
    hang = tr["vessel_name"].map(is_hangul).to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr["target_vessel_id"]))

    # same GT-less augmentation as train.py
    rng = np.random.default_rng(42)
    aug_src = np.flatnonzero(hang | (rng.random(len(tr)) < 0.25))
    aug = tr.iloc[aug_src].assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan)
    pos_in_aug = {i: j for j, i in enumerate(aug_src)}
    Xs, ys, cs = [None] * len(tr), [None] * len(tr), [None] * len(tr)
    Xa, ya = [None] * len(aug), [None] * len(aug)
    for k, (a, b) in enumerate(folds):
        _, token_dict, flag_p, type_p = fit_aux(tr.iloc[a], cat)
        m = Matcher(idx, {}, token_dict, flag_p, type_p)
        recs = tr.iloc[b].to_dict("records")
        for j, (i, (_, cands, X)) in enumerate(zip(b, (c for s in range(0, len(recs), 256)
                                                        for c in m.candidates(recs[s:s + 256])))):
            Xs[i], cs[i] = X, cands
            ys[i] = (idx.ids[cands] == tr["target_vessel_id"].iat[i]).astype(np.int8)
        ab = [pos_in_aug[i] for i in b if i in pos_in_aug]
        fx, fy, _ = build(aug.iloc[ab], m)
        for j, i in enumerate(ab):
            Xa[i], ya[i] = fx[j], fy[j]
        print(f"fold {k} features done ({time.time() - t0:.0f}s)", flush=True)

    hang_a = hang[aug_src]

    def train_idx(rows_a):
        sa = set(rows_a)
        ja = [j for j, i in enumerate(aug_src) if i in sa]
        return ([Xs[i] for i in rows_a] + [Xa[j] for j in ja], [ys[i] for i in rows_a] + [ya[j] for j in ja],
                np.concatenate([hang[rows_a], hang_a[ja]]))

    pred = np.empty(len(tr), dtype=object)
    true_rank = np.full(len(tr), -1)
    top_score = np.zeros(len(tr)); true_score = np.full(len(tr), np.nan)
    for a, b in folds:
        model = fit_ranker(*train_idx(a))
        for i in b:
            s = model.predict(Xs[i])
            pred[i] = idx.ids[cs[i][int(np.argmax(s))]]
            top_score[i] = s.max()
            pos = np.flatnonzero(ys[i])
            if len(pos):
                true_score[i] = s[pos[0]]
                true_rank[i] = int((s > s[pos[0]]).sum()) + 1
    correct = pred == tr["target_vessel_id"].to_numpy()
    recall = true_rank > 0
    print("retrieval recall", score(hang, recall))
    print("unseen-vessel accuracy", score(hang, correct))

    c = cat.set_index("vessel_id")
    err = tr[~correct].copy()
    err["is_hangul"] = hang[~correct]
    err["kind"] = np.where(recall[~correct], "ranker", "retrieval_miss")
    err["true_rank"] = true_rank[~correct]
    err["n_cands"] = [len(cs[i]) for i in np.flatnonzero(~correct)]
    err["margin"] = top_score[~correct] - true_score[~correct]
    for side, ids in (("true", err["target_vessel_id"]), ("pred", pred[~correct])):
        sub = c.loc[list(ids)]
        err[f"{side}_name"] = sub["vessel_name"].to_numpy()
        err[f"{side}_gt"] = sub["gross_tonnage"].to_numpy()
        err[f"{side}_flag"] = sub["flag_name"].to_numpy()
        err[f"{side}_type"] = sub["vessel_type"].to_numpy()
    err["pred_vessel_id"] = pred[~correct]
    out.mkdir(parents=True, exist_ok=True)
    err.sort_values(["is_hangul", "kind"], ascending=[False, True]).to_csv(
        out / "oof_errors.csv", index=False, encoding="utf-8-sig")
    print(err.groupby(["is_hangul", "kind"]).size().to_string())
    print(f"saved {len(err)} wrong rows -> {out / 'oof_errors.csv'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis")
    a = ap.parse_args()
    main(a.data, a.out)
