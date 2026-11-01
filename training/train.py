"""Train the reranker, report CV (0.7*Hangul + 0.3*non-Hangul), and build submit.zip.

usage: python train.py --data <dir with train.csv, vessel_catalog.csv>
"""

from __future__ import annotations

import argparse
import re
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, __import__("os").environ.get("MODEL_DIR", str(ROOT.parent / "inference" / "model")))  # MODEL_DIR: compare an older model code
from vessel_lib import (FEATURES, MONOTONE, BoosterAverage, CatalogIndex, CondProb, Matcher,  # noqa: E402
                        is_hangul, learn_token_dict, load_catalog_ko, name_key)
import os  # noqa: E402
import vessel_lib  # noqa: E402
sys.path.insert(0, str(ROOT))
from augment import gt_noise, map_name, name_typo, no_gt  # noqa: E402

N_FOLDS = 5
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))  # rankers differing only in seed, scores averaged
vessel_lib.ACRONYMS = os.environ.get("ACRONYMS", "1") != "0"  # 케이엠티씨 -> KMTC
vessel_lib.SPELL_CAPS = os.environ.get("SPELL_CAPS", "1") != "0"  # STI Sloane -> es ti ai (Hangul keys)
vessel_lib.PERKEY_ALWAYS = os.environ.get("PERKEY_ALWAYS", "0") == "1"  # G line (LB: hurt D -> E1)
vessel_lib.HANJA = os.environ.get("HANJA", "0") == "1"  # 금성 = Jin Cheng, 환 = Maru (rule question pending)
import hanja  # noqa: E402
if os.environ.get("EMB_FT"):  # fine-tuned LaBSE: per-fold vectors from labse_ft (out-of-fold features)
    from emb_feat import FoldVectors
    vessel_lib.EMB = FoldVectors(Path(os.environ["EMB_FT"]), Path(os.environ["EMB_FT"]) / "names_to_embed.csv")
elif os.environ.get("EMB") == "1":  # LaBSE features (model code must have emb_feat.py)
    import atexit
    from emb_feat import Embedder
    vessel_lib.EMB = Embedder(Path(vessel_lib.__file__).parent / "emb", ROOT / "analysis" / "emb_cache.npz")
    atexit.register(vessel_lib.EMB.save)


def use_fold(k) -> None:
    """Fine-tuned LaBSE has one encoder per fold: fold k's rows use the encoder that never saw them."""
    if hasattr(vessel_lib.EMB, "set_fold"):
        vessel_lib.EMB.set_fold(k)
# Besides GT-less copies, also add GT-noise and name-typo copies. Off since E1: with it (version E)
# the leaderboard lost one Hangul row (0.98789 -> 0.98567); test tonnage looks reliable, and the
# noisy copies taught the ranker to trust it less.
NOISE_AUG = False
PARAMS = dict(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
              min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              lambdarank_truncation_level=10, verbose=-1, random_state=42, deterministic=True,
              n_jobs=6,
              monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES],
              monotone_constraints_method="advanced")


KO_PATH = Path(os.environ.get("KO_FILE", ROOT.parent / "inference" / "model" / "catalog_ko.csv"))
KO2_PATH = Path(os.environ.get("KO2_FILE", ROOT.parent / "inference" / "model" / "catalog_ko2.csv"))  # llm/catalog_ko_q7b*.csv


def load_index(cat: pd.DataFrame) -> CatalogIndex:
    """Catalog index with the LLM Korean spellings, unless USE_KO=0 / USE_KO2=0 (ablations)."""
    ko = load_catalog_ko(KO_PATH) if os.environ.get("USE_KO", "1") != "0" else {}
    ko2 = load_catalog_ko(KO2_PATH) if os.environ.get("USE_KO2", "1") != "0" else {}
    print(f"catalog_ko: {len(ko)} spellings, catalog_ko2: {len(ko2)}", flush=True)
    return CatalogIndex(cat, ko, ko2)


def fit_aux(tr: pd.DataFrame, cat: pd.DataFrame):
    c = cat.set_index("vessel_id").loc[tr["target_vessel_id"]]
    flag_p = CondProb.fit(tr["flag_name"], c["flag_name"].to_numpy())
    type_p = CondProb.fit(tr["vessel_type"], c["vessel_type"].to_numpy())
    hang = tr["vessel_name"].map(is_hangul).to_numpy()
    token_dict = learn_token_dict(list(zip(tr["vessel_name"][hang], c["vessel_name"].to_numpy()[hang])))
    callsign = dict(zip(tr["call_sign"], tr["target_vessel_id"]))
    return callsign, token_dict, flag_p, type_p


def build(rows: pd.DataFrame, m: Matcher):
    Xs, ys, groups = [], [], []
    recs = rows.to_dict("records")
    for row, (_, cands, X) in zip(recs, (c for s in range(0, len(recs), 256)
                                         for c in m.candidates(recs[s:s + 256]))):
        Xs.append(X)
        ys.append((m.idx.ids[cands] == row["target_vessel_id"]).astype(np.int8))
        groups.append(len(cands))
    return Xs, ys, np.asarray(groups)


def weights(hang: np.ndarray, groups: np.ndarray) -> np.ndarray:
    w = np.where(hang, 0.7 / max(hang.sum(), 1), 0.3 / max((~hang).sum(), 1)) * len(hang)
    return np.repeat(w, groups)


def fit_ranker(Xs, ys, hang):
    keep = [i for i, y in enumerate(ys) if y.any()]  # groups without a positive teach nothing
    if os.environ.get("HANGUL_ONLY") == "1":  # K5: these trees score Hangul rows only (eng_* do the rest)
        keep = [i for i in keep if hang[i]]
    X = np.vstack([Xs[i] for i in keep]); y = np.concatenate([ys[i] for i in keep])
    g = np.asarray([len(ys[i]) for i in keep])
    w = weights(hang[keep], g)
    return BoosterAverage([lgb.LGBMRanker(**(PARAMS | dict(random_state=42 + s))).fit(X, y, group=g, sample_weight=w)
                           for s in range(N_SEEDS)])


def mixed_words(tr: pd.DataFrame, cat: pd.DataFrame, names: pd.Series, ko: dict) -> list:
    """hanja.name_words of each row's vessel, flagged Japanese like CatalogIndex does (Japan flag or Maru)."""
    flg = tr["target_vessel_id"].map(cat.set_index("vessel_id")["flag_name"]).astype(str)
    jp = (flg.eq("Japan") | names.str.contains(r"(?i)\bmaru\b")).to_numpy()
    cn = flg.isin(vessel_lib.CN_FLAGS).to_numpy()
    mixed_words.jp = jp
    return [hanja.name_words(n, ko.get(n, ""), j, j or c) for n, j, c in zip(names, jp, cn)]


def meaning_words_of(tr: pd.DataFrame, cat: pd.DataFrame, names: pd.Series, ko: dict) -> list:
    """hanja.meaning_words of each row's vessel, with the Chinese flag when the model code takes it."""
    import inspect
    if "chinese" not in inspect.signature(hanja.meaning_words).parameters:
        return [hanja.meaning_words(n, ko.get(n, "")) for n in names]
    flg = tr["target_vessel_id"].map(cat.set_index("vessel_id")["flag_name"]).astype(str)
    cn = flg.isin({"Peoples' Republic of China", "Hong Kong", "Taiwan"}).to_numpy()
    return [hanja.meaning_words(n, ko.get(n, ""), c) for n, c in zip(names, cn)]


def make_aug(tr: pd.DataFrame, hang: np.ndarray, noise: bool = NOISE_AUG, cat: pd.DataFrame | None = None):
    """Corrupted copies of train rows -> (aug rows, source row of each copy).
    Each copy goes to the same fold as its source row, so validation stays vessel-disjoint.
      1) GT removed: every Hangul row + 25% of latin rows (decide on the name alone)
      2) GT scaled by ±3-10%: same mix (do not over-trust the ±5.5% tonnage match)
      3) one name corruption (jamo typo / spacing / 호, latin typo / spacing): same mix"""
    rng = np.random.default_rng(42)
    src = [np.flatnonzero(hang | (rng.random(len(tr)) < 0.25))]
    parts = [no_gt(tr.iloc[src[0]])]
    if noise:
        src.append(np.flatnonzero(hang | (rng.random(len(tr)) < 0.25)))
        parts.append(gt_noise(tr.iloc[src[-1]], rng))
        src.append(np.flatnonzero(hang | (rng.random(len(tr)) < 0.25)))
        parts.append(map_name(tr.iloc[src[-1]], name_typo, rng))
    if vessel_lib.HANJA and cat is not None:
        # train has no record written by Sino-Korean reading: synthesize some so the ranker learns
        # to use the hj_* features (half of the rows whose vessel has a pinyin name, + all Maru vessels)
        names = tr["target_vessel_id"].map(cat.set_index("vessel_id")["vessel_name"]).astype(str)
        py = names.map(lambda n: bool(hanja.catalog_keys(n))).to_numpy() & (rng.random(len(tr)) < 0.5)
        s = np.flatnonzero(py)
        src.append(s)
        parts.append(tr.iloc[s].assign(vessel_name=[hanja.record_reading(n, rng) for n in names.iloc[s]]))
        ko = load_catalog_ko(KO_PATH)
        mr = np.flatnonzero(names.str.contains(r"(?i)\bmaru\b").to_numpy() & names.map(lambda n: bool(ko.get(n))).to_numpy())
        src.append(mr)
        parts.append(tr.iloc[mr].assign(vessel_name=[re.sub(r"\s*마루", "환", ko[n]) for n in names.iloc[mr]]))
        # half of the rows whose name reads word by word: some words by hanja, the rest by sound
        # (해풍 글로리), and Japanese names by kanji reading (신성환 = Shinsei Maru)
        if hasattr(hanja, "name_words"):  # absent in older model code (MODEL_DIR comparisons)
            nws = mixed_words(tr, cat, names, ko)
            jp = mixed_words.jp
            mx = np.flatnonzero(np.array([w is not None and (j or any(ph for _, ph in w[0])) for w, j in zip(nws, jp)])
                                & (rng.random(len(tr)) < 0.5))
            src.append(mx)
            parts.append(tr.iloc[mx].assign(vessel_name=[hanja.record_mixed(nws[i], rng) for i in mx]))
        if hasattr(hanja, "meaning_words"):  # half of the rows whose name has a translatable word
            mws = meaning_words_of(tr, cat, names, ko)
            mn = np.flatnonzero(np.array([bool(hanja.meaning_keys(w)) for w in mws]) & (rng.random(len(tr)) < 0.5))
            src.append(mn)
            parts.append(tr.iloc[mn].assign(vessel_name=[hanja.record_meaning(mws[i], rng) for i in mn]))
    rng2 = np.random.default_rng(4242)  # own stream: the copies above stay as they were
    if os.environ.get("SYNTH_AUG"):  # LLM-written Korean names of train vessels (allowed: talk #418501)
        sys.path.insert(0, str(ROOT / "llm_synth"))
        from load_synth import load_records
        mc = os.environ.get("SYNTH_MIN_COS")  # drop invented records (LaBSE similarity to the true name)
        for sp in os.environ["SYNTH_AUG"].split(";"):  # several files: "synth.csv;synth2.csv"
            syn = load_records(sp, min_cos=float(mc) if mc else None)
            src.append(syn["source_row"].to_numpy())  # same fold as the train row it was made from
            parts.append(syn[tr.columns])
            if os.environ.get("SYNTH_NOGT") == "1":  # the same records once more without tonnage (J4):
                src.append(syn["source_row"].to_numpy())  # meaning/short spellings must win on the name alone
                parts.append(no_gt(syn[tr.columns]))
    if os.environ.get("TYPO_AUG") == "1":  # name typos with tonnage intact (Hangul rows + 25% latin)
        src.append(np.flatnonzero(hang | (rng2.random(len(tr)) < 0.25)))
        parts.append(map_name(tr.iloc[src[-1]], name_typo, rng2))
    if os.environ.get("HEAVY_GT") == "1":  # tonnage off by 10-25% (beyond the ±5.5% window), 25% of rows
        src.append(np.flatnonzero(rng2.random(len(tr)) < 0.25))
        parts.append(gt_noise(tr.iloc[src[-1]], rng2, 0.10, 0.25))
    return pd.concat(parts, ignore_index=True), np.concatenate(src)


def score(hang, correct):
    h, n = correct[hang].mean(), correct[~hang].mean()
    return dict(hangul=round(h, 4), non_hangul=round(n, 4), weighted=round(0.7 * h + 0.3 * n, 4))


def main(data: Path):
    t0 = time.time()
    tr = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    cat = pd.read_csv(data / "vessel_catalog.csv", encoding="utf-8-sig")
    idx = load_index(cat)
    hang = tr["vessel_name"].map(is_hangul).to_numpy()

    # 1) out-of-fold features: folds grouped by vessel => every row is an *unseen* vessel
    #    (the leaderboard test is vessel-disjoint from train).
    #    Augmentation (make_aug): GT-less, GT-noise and name-typo copies, so the ranker
    #    also learns to decide on the name alone instead of leaning on tonnage.
    folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr["target_vessel_id"]))
    aug, aug_src = make_aug(tr, hang, cat=cat)
    Xs, ys = [None] * len(tr), [None] * len(tr)
    Xa, ya = [None] * len(aug), [None] * len(aug)
    pos_in_aug: dict[int, list[int]] = {}
    for j, i in enumerate(aug_src):
        pos_in_aug.setdefault(i, []).append(j)  # first entry = GT-less copy
    for k, (a, b) in enumerate(folds):
        use_fold(k)
        _, token_dict, flag_p, type_p = fit_aux(tr.iloc[a], cat)
        m = Matcher(idx, {}, token_dict, flag_p, type_p)
        fx, fy, _ = build(tr.iloc[b], m)
        for j, i in enumerate(b):
            Xs[i], ys[i] = fx[j], fy[j]
        ab = [j for i in b for j in pos_in_aug.get(i, [])]
        fx, fy, _ = build(aug.iloc[ab], m)
        for j, i in enumerate(ab):
            Xa[i], ya[i] = fx[j], fy[j]
        print(f"fold {k} features done ({time.time() - t0:.0f}s)")
    recall = np.array([y.any() for y in ys])
    print("retrieval recall", score(hang, recall), "avg cands", np.mean([len(y) for y in ys]).round(1))
    hang_a = aug["vessel_name"].map(is_hangul).to_numpy()  # synthetic hanja copies are Hangul

    def train_idx(rows_a):
        """Original rows `rows_a` plus their GT-less copies."""
        sa = set(rows_a)
        ja = [j for j, i in enumerate(aug_src) if i in sa]
        return ([Xs[i] for i in rows_a] + [Xa[j] for j in ja], [ys[i] for i in rows_a] + [ya[j] for j in ja],
                np.concatenate([hang[rows_a], hang_a[ja]]))

    # 2) OOF reranker accuracy on unseen vessels (as given, and Hangul with GT removed)
    correct = np.zeros(len(tr), bool)
    nogt = {}
    for a, b in folds:
        model = fit_ranker(*train_idx(a))
        for i in b:
            correct[i] = ys[i][int(np.argmax(model.predict(Xs[i])))] == 1
            if hang[i]:
                j = pos_in_aug[i][0]
                nogt[i] = ya[j][int(np.argmax(model.predict(Xa[j])))] == 1
    print("unseen-vessel accuracy", score(hang, correct))
    print(f"unseen Hangul with GT removed: {np.mean(list(nogt.values())):.4f}")

    # 3) full pipeline estimate: call_sign lookup from other folds (random row split)
    rnd = np.random.default_rng(0).permutation(len(tr)) % N_FOLDS
    full = correct.copy()
    for k in range(N_FOLDS):
        known = set(tr["call_sign"][rnd != k])
        hit = (rnd == k) & tr["call_sign"].isin(known).to_numpy()
        full[hit] = True
    print("full pipeline (random split) accuracy", score(hang, full))

    # 4) final artifacts on all train
    callsign, token_dict, flag_p, type_p = fit_aux(tr, cat)
    model = fit_ranker(*train_idx(np.arange(len(tr))))
    sub = Path(os.environ.get("SUBMIT_DIR", ROOT.parent / "inference"))  # SUBMIT_DIR: build a variant beside submit/
    out = sub / "model"
    # LightGBM's C writer cannot open non-ASCII Windows paths; write the text model from Python.
    # newline="" keeps "\n": the header stores per-tree byte sizes, so "\r\n" corrupts the model.
    for old in out.glob("ranker*.txt"):
        old.unlink()
    for s, m in enumerate(model.models):
        (out / f"ranker_{s}.txt").write_text(m.booster_.model_to_string(), encoding="utf-8", newline="")
    keys = tr["vessel_name"].map(name_key)
    (out / "artifacts.json").write_text(json.dumps(dict(
        features=FEATURES, use_ko=bool(idx.has_ko), use_ko2=bool(idx.has_ko2), acronyms=vessel_lib.ACRONYMS, spell_caps=vessel_lib.SPELL_CAPS, perkey_always=vessel_lib.PERKEY_ALWAYS, hanja=vessel_lib.HANJA, emb=getattr(vessel_lib, "EMB", None) is not None, callsign=callsign, token_dict=token_dict,
        flag_p=flag_p.table, type_p=type_p.table,
        name_map=dict(zip(keys, tr["target_vessel_id"])),        # 1:1 in train
        callsign_name=dict(zip(tr["call_sign"], keys))), ensure_ascii=False), encoding="utf-8")
    imp = pd.Series(model.models[0].booster_.feature_importance("gain"), index=FEATURES)
    print(imp.sort_values(ascending=False).head(15).round(0).to_string())
    shutil.rmtree(out / "__pycache__", ignore_errors=True)
    shutil.make_archive(str(sub), "zip", sub)
    print(f"saved {sub.name}.zip ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    main(ap.parse_args().data)
