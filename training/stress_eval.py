"""Stress validation: vessel-disjoint 5-fold (same protocol and augmentation as train.py), then
re-score the held-out vessels under test-like corruptions to see where accuracy collapses.
Writes only to analysis/.

Note: train.py's augmentation uses the same corruption families (other random draws), so
conditions it covers are optimistic; latin_typo_2 and "typo + no_gt" are never trained on.

usage: python stress_eval.py [--data ../open] [--latin 400] [--tag name]
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
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, __import__("os").environ.get("MODEL_DIR", str(ROOT.parent / "inference" / "model")))
sys.path.insert(0, str(ROOT))
from vessel_lib import CatalogIndex, Matcher, is_hangul  # noqa: E402
from train import N_FOLDS, build, fit_aux, fit_ranker, load_index, make_aug, use_fold  # noqa: E402
from augment import (gt_noise, hangul_jamo_typo, hangul_spacing, hangul_suffix,  # noqa: E402
                     latin_spacing, latin_typo, map_name, no_gt)
from sklearn.model_selection import GroupKFold  # noqa: E402

CONDITIONS = {
    "clean": lambda d, r: d,
    "no_gt": lambda d, r: no_gt(d),
    "gt_noise_3-10%": lambda d, r: gt_noise(d, r),
    "no_flag": lambda d, r: d.assign(flag_name=np.nan),
    "hangul_jamo_typo": lambda d, r: map_name(d, lambda n, g: hangul_jamo_typo(n, g) if is_hangul(n) else n, r),
    "hangul_spacing": lambda d, r: map_name(d, lambda n, g: hangul_spacing(n, g) if is_hangul(n) else n, r),
    "hangul_ho_je": lambda d, r: map_name(d, lambda n, g: hangul_suffix(n, g) if is_hangul(n) else n, r),
    "latin_typo_1": lambda d, r: map_name(d, lambda n, g: n if is_hangul(n) else latin_typo(n, g, 1), r),
    "latin_typo_2": lambda d, r: map_name(d, lambda n, g: n if is_hangul(n) else latin_typo(n, g, 2), r),
    "latin_spacing": lambda d, r: map_name(d, lambda n, g: n if is_hangul(n) else latin_spacing(n, g), r),
    "typo + no_gt": lambda d, r: no_gt(map_name(d, lambda n, g: hangul_jamo_typo(n, g) if is_hangul(n)
                                                 else latin_typo(n, g, 1), r)),
    # hard transliteration benchmark: tonnage, flag and type all removed, so the pick rests on the
    # name alone (train Hangul is ~100% with tonnage, which hides transliteration mistakes)
    "name_only": lambda d, r: no_gt(d).assign(flag_name=np.nan, vessel_type=np.nan),
    # harder, closer to the post-reset leaderboard (old 1.0 teams fell to ~0.79)
    "gt_noise_10-25%": lambda d, r: gt_noise(d, r, 0.10, 0.25),
    "hangul_typo_x2": lambda d, r: map_name(d, lambda n, g: hangul_jamo_typo(hangul_jamo_typo(n, g), g)
                                             if is_hangul(n) else n, r),
    "hard_mix": lambda d, r: hard_mix(d, r),
}


def hard_mix(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Each row gets 2-3 corruptions at once: name typo, spacing, tonnage off by 3-25% or missing,
    flag missing."""
    out = df.copy()
    for i in range(len(out)):
        ops = rng.choice(["typo", "space", "gt", "flag"], size=int(rng.integers(2, 4)), replace=False)
        name = out["vessel_name"].iat[i]
        if isinstance(name, str):
            if "typo" in ops:
                name = hangul_jamo_typo(name, rng) if is_hangul(name) else latin_typo(name, rng, 1)
            if "space" in ops:
                name = hangul_spacing(name, rng) if is_hangul(name) else latin_spacing(name, rng)
            out.iat[i, out.columns.get_loc("vessel_name")] = name
        if "gt" in ops:
            if rng.random() < 0.4:
                f = np.nan
            else:
                f = 1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.25)
            for c in ("gross_tonnage", "international_gross_tonnage"):
                v = out[c].iat[i] * f
                out.iat[i, out.columns.get_loc(c)] = round(v) if np.isfinite(v) else np.nan
        if "flag" in ops:
            out.iat[i, out.columns.get_loc("flag_name")] = np.nan
    return out


def prepare_folds(data: Path):
    """Vessel-disjoint folds with, per fold, a Matcher fit on the other folds and a ranker trained
    exactly like train.py (OOF features + augmentation). Shared with export_topk.py."""
    t0 = time.time()
    tr = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(data / "vessel_catalog.csv", encoding="utf-8-sig")
    idx = load_index(cat)
    hang = tr["vessel_name"].map(is_hangul).to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr["target_vessel_id"]))

    # training features exactly as train.py (OOF + augmentation)
    aug, aug_src = make_aug(tr, hang, cat=cat)
    pos_in_aug: dict[int, list[int]] = {}
    for j, i in enumerate(aug_src):
        pos_in_aug.setdefault(i, []).append(j)
    Xs, ys = [None] * len(tr), [None] * len(tr)
    Xa, ya = [None] * len(aug), [None] * len(aug)
    matchers = []
    for k, (a, b) in enumerate(folds):
        use_fold(k)
        _, token_dict, flag_p, type_p = fit_aux(tr.iloc[a], cat)
        m = Matcher(idx, {}, token_dict, flag_p, type_p)
        matchers.append(m)
        fx, fy, _ = build(tr.iloc[b], m)
        for j, i in enumerate(b):
            Xs[i], ys[i] = fx[j], fy[j]
        ab = [j for i in b for j in pos_in_aug.get(i, [])]
        fx, fy, _ = build(aug.iloc[ab], m)
        for j, i in enumerate(ab):
            Xa[i], ya[i] = fx[j], fy[j]
        print(f"fold {k} train features ({time.time() - t0:.0f}s)", flush=True)
    hang_a = aug["vessel_name"].map(is_hangul).to_numpy()  # synthetic hanja copies are Hangul

    def train_idx(rows_a):
        sa = set(rows_a)
        ja = [j for j, i in enumerate(aug_src) if i in sa]
        return ([Xs[i] for i in rows_a] + [Xa[j] for j in ja], [ys[i] for i in rows_a] + [ya[j] for j in ja],
                np.concatenate([hang[rows_a], hang_a[ja]]))

    models = [fit_ranker(*train_idx(a)) for a, _ in folds]
    print(f"fold rankers trained ({time.time() - t0:.0f}s)", flush=True)
    return tr, cat, idx, hang, folds, matchers, models


def main(data: Path, n_latin: int, out: Path, tag: str, only: list[str] | None = None):
    t0 = time.time()
    tr, cat, idx, hang, folds, matchers, models = prepare_folds(data)

    # evaluation subset: every held-out Hangul row + a fixed sample of Latin rows
    pick = np.random.default_rng(7)
    latin_rows = np.flatnonzero(~hang)
    eval_rows = np.sort(np.concatenate([np.flatnonzero(hang),
                                        pick.choice(latin_rows, min(n_latin * N_FOLDS, len(latin_rows)), False)]))
    results, errors = [], []
    for k, (a, b) in enumerate(folds):
        use_fold(k)
        model = models[k]
        rows = np.intersect1d(b, eval_rows)
        base = tr.iloc[rows].reset_index(drop=True)
        base_h = base["vessel_name"].map(is_hangul).to_numpy()
        for ci, (cname, fn) in enumerate(CONDITIONS.items()):
            if only and cname not in only:
                continue
            df = fn(base, np.random.default_rng(1000 * k + ci))
            recs = df.to_dict("records")
            m = matchers[k]
            for s in range(0, len(recs), 256):
                for r, (rec, (q, cands, X)) in enumerate(zip(recs[s:s + 256], m.candidates(recs[s:s + 256])),
                                                         start=s):
                    target = rec["target_vessel_id"]
                    pred = idx.ids[cands[int(np.argmax(model.predict(X)))]]
                    hit = bool((idx.ids[cands] == target).any())
                    results.append((cname, base_h[r], hit, pred == target, len(cands)))
                    if pred != target:
                        errors.append(dict(condition=cname, orig_name=base["vessel_name"].iat[r],
                                           shown_name=rec["vessel_name"], gross_tonnage=rec["gross_tonnage"],
                                           in_candidates=hit, true_id=target, pred_id=pred,
                                           true_name=cat["vessel_name"].iat[idx.id_to_index[target]],
                                           pred_name=cat["vessel_name"].iat[idx.id_to_index[pred]]))
        print(f"fold {k} stress done ({time.time() - t0:.0f}s)", flush=True)

    r = pd.DataFrame(results, columns=["condition", "hangul", "recall", "correct", "n_cands"])
    g = r.groupby(["condition", "hangul"]).agg(n=("correct", "size"), recall=("recall", "mean"),
                                                acc=("correct", "mean")).unstack("hangul")
    table = pd.DataFrame({
        "hangul_acc": g[("acc", True)], "latin_acc": g[("acc", False)],
        "weighted": 0.7 * g[("acc", True)] + 0.3 * g[("acc", False)],
        "hangul_recall": g[("recall", True)], "latin_recall": g[("recall", False)],
        "n_hangul": g[("n", True)], "n_latin": g[("n", False)],
    }).reindex([c for c in CONDITIONS if not only or c in only])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"stress_summary_{tag}.csv", encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(out / f"stress_errors_{tag}.csv", index=False, encoding="utf-8-sig")
    print(table.round(4).to_string())
    print(f"avg candidates {r['n_cands'].mean():.1f}")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    ap.add_argument("--latin", type=int, default=400, help="Latin rows evaluated per fold")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--only", nargs="*", help="evaluate only these conditions")
    a = ap.parse_args()
    main(a.data, a.latin, a.out, a.tag, a.only)
