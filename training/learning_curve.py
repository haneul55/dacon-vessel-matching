"""Learning curves of model J's ranker for the meeting report: per boosting round (LightGBM's "epoch"),
training and validation NDCG@1 (= top-1 accuracy here: one correct vessel per record) and the
LambdaRank loss proxy 1 - NDCG@1, on the same 5 vessel-disjoint folds and features as train.py.
Also counts the parameters of the ranker and of LaBSE. Writes analysis/learning_curve_*.csv/png.

usage: MODEL_DIR=submit_J/model HANJA=1 EMB=1 SYNTH_AUG=llm_synth/synth.csv python learning_curve.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import train
from train import N_FOLDS, PARAMS, build, fit_aux, load_index, make_aug, weights
from vessel_lib import FEATURES, Matcher, is_hangul
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"


def stack(Xs, ys, hang):
    keep = [i for i, y in enumerate(ys) if y.any()]
    X = np.vstack([Xs[i] for i in keep]); y = np.concatenate([ys[i] for i in keep])
    g = np.asarray([len(ys[i]) for i in keep])
    return X, y, g, weights(hang[keep], g), len(ys) - len(keep)


def main(data: Path):
    t0 = time.time()
    tr = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(data / "vessel_catalog.csv", encoding="utf-8-sig")
    idx = load_index(cat)
    hang = tr["vessel_name"].map(is_hangul).to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr["target_vessel_id"]))
    aug, aug_src = make_aug(tr, hang, cat=cat)
    hang_a = aug["vessel_name"].map(is_hangul).to_numpy()
    info = dict(train_rows=len(tr), train_hangul=int(hang.sum()), vessels=int(tr["target_vessel_id"].nunique()),
                catalog=len(cat), aug_rows=len(aug), features=len(FEATURES))
    Xs, ys = [None] * len(tr), [None] * len(tr)
    Xa, ya = [None] * len(aug), [None] * len(aug)
    pos = {}
    for j, i in enumerate(aug_src):
        pos.setdefault(i, []).append(j)
    for k, (a, b) in enumerate(folds):
        _, td, fp, tp = fit_aux(tr.iloc[a], cat)
        m = Matcher(idx, {}, td, fp, tp)
        fx, fy, _ = build(tr.iloc[b], m)
        for j, i in enumerate(b):
            Xs[i], ys[i] = fx[j], fy[j]
        ab = [j for i in b for j in pos.get(i, [])]
        fx, fy, _ = build(aug.iloc[ab], m)
        for j, i in enumerate(ab):
            Xa[i], ya[i] = fx[j], fy[j]
        print(f"fold {k} features ({time.time() - t0:.0f}s)", flush=True)
    info["avg_candidates"] = float(np.mean([len(y) for y in ys]))

    curves, sizes = [], []
    for k, (a, b) in enumerate(folds):
        sa = set(a)
        ja = [j for j, i in enumerate(aug_src) if i in sa]
        Xt, yt, gt, wt, _ = stack([Xs[i] for i in a] + [Xa[j] for j in ja], [ys[i] for i in a] + [ya[j] for j in ja],
                                  np.concatenate([hang[a], hang_a[ja]]))
        Xv, yv, gv, wv, miss = stack([Xs[i] for i in b], [ys[i] for i in b], hang[b])
        hb = np.array([i for i in b if ys[i].any()])
        hv = hang[hb]
        Xh = np.vstack([Xs[i] for i in hb[hv]]); yh = np.concatenate([ys[i] for i in hb[hv]])
        gh = np.asarray([len(ys[i]) for i in hb[hv]])
        sizes.append(dict(fold=k, train_records=len(gt), train_pairs=len(yt), val_records=len(gv), val_pairs=len(yv),
                          val_hangul=int(hv.sum()), val_without_positive=miss))
        ev = {}
        lgb.LGBMRanker(**(PARAMS | dict(random_state=42))).fit(
            Xt, yt, group=gt, sample_weight=wt,
            eval_set=[(Xt, yt), (Xv, yv), (Xh, yh)], eval_group=[gt, gv, gh], eval_names=["train", "valid", "valid_hangul"],
            eval_at=[1], callbacks=[lgb.record_evaluation(ev)])
        for name in ev:
            for it, v in enumerate(ev[name]["ndcg@1"], 1):
                curves.append((k, name, it, v))
        print(f"fold {k} curve ({time.time() - t0:.0f}s): valid ndcg@1 {ev['valid']['ndcg@1'][-1]:.4f}", flush=True)

    c = pd.DataFrame(curves, columns=["fold", "set", "round", "ndcg1"])
    c["loss"] = 1 - c["ndcg1"]
    c.to_csv(OUT / "learning_curve.csv", index=False)
    pd.DataFrame(sizes).to_csv(OUT / "learning_curve_sizes.csv", index=False)

    # parameters: final model J (3 rankers) and LaBSE
    import glob
    n_leaf = n_split = n_tree = 0
    for f in sorted(glob.glob(str(ROOT / "submit_J" / "model" / "ranker_*.txt"))):
        d = lgb.Booster(model_str=Path(f).read_text(encoding="utf-8")).dump_model()
        for t in d["tree_info"]:
            n_tree += 1
            n_leaf += t["num_leaves"]; n_split += t["num_leaves"] - 1
    info.update(ranker_trees=n_tree, ranker_leaves=n_leaf, ranker_splits=n_split,
                ranker_params=n_leaf + 2 * n_split)  # leaf values + (feature, threshold) per split
    from safetensors import safe_open
    st = ROOT / "submit_J" / "model" / "emb" / "model" / "model.safetensors"
    with safe_open(str(st), framework="np") as fh:
        shapes = {k: fh.get_slice(k).get_shape() for k in fh.keys()}
    info["labse_params"] = int(sum(int(np.prod(s)) for s in shapes.values()))
    info["labse_embedding_params"] = int(sum(int(np.prod(s)) for k, s in shapes.items() if "word_embeddings" in k))
    (OUT / "learning_curve_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    m = c.groupby(["set", "round"])["loss"].mean().unstack(0)
    for s, lab in [("train", "train"), ("valid", "validation (all)"), ("valid_hangul", "validation (Hangul)")]:
        ax[0].plot(m.index, m[s], label=lab)
    ax[0].set_xlabel("boosting round (tree)"); ax[0].set_ylabel("1 - NDCG@1 (top-1 error)")
    ax[0].set_title("Model J ranker: loss per round (mean of 5 folds)"); ax[0].legend(); ax[0].set_yscale("log")
    for k in range(N_FOLDS):
        d = c[(c.fold == k) & (c.set == "valid")]
        ax[1].plot(d["round"], d["ndcg1"], alpha=0.8, label=f"fold {k}")
    ax[1].set_xlabel("boosting round (tree)"); ax[1].set_ylabel("validation top-1 accuracy (NDCG@1)")
    ax[1].set_title("Validation accuracy per fold"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=130)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(ROOT.parent / "open")
