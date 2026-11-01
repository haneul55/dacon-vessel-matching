"""Synthetic validation for records written by Sino-Korean reading (Jin Cheng -> 금성) or 환 (Maru).

train has no such record, so held-out rows (vessel-disjoint folds) whose vessel has a pinyin name
are re-written by hanja.record_reading (with a different random stream than training), and Maru
vessels get their Korean spelling with 마루 -> 환. Accuracy with and without tonnage.
Caveat: the same reading table builds and matches these records, so this checks the wiring, not
how well the table covers the real records.

usage: HANJA=1 python hanja_eval.py --tag on      (HANJA=0 ... --tag off for the baseline)
       MODEL_DIR=<old model> HANJA=1 python hanja_eval.py --gen submit/model/hanja.py --tag old_on_new
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")
from stress_eval import ROOT, prepare_folds  # noqa: E402
import hanja  # noqa: E402  (on sys.path via stress_eval)
from vessel_lib import load_catalog_ko  # noqa: E402


def load_gen(path: Path | None):
    """Record generator from another hanja.py (e.g. the wider table), so an older model can be scored
    on records its own table would not produce."""
    if path is None:
        return hanja
    import importlib.util
    spec = importlib.util.spec_from_file_location("hanja_gen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(data: Path, per_fold: int, tag: str, gen_path: Path | None = None):
    t0 = time.time()
    gen = load_gen(gen_path)
    tr, cat, idx, hang, folds, matchers, models = prepare_folds(data)
    names = tr["target_vessel_id"].map(pd.Series(cat["vessel_name"].astype(str).to_numpy(), index=cat["vessel_id"]))
    ko = load_catalog_ko(ROOT.parent / "inference" / "model" / "catalog_ko.csv")
    is_py = names.map(lambda n: bool(gen.catalog_keys(n))).to_numpy()
    is_maru = (names.str.contains(r"(?i)\bmaru\b") & names.map(lambda n: bool(ko.get(n)))).to_numpy()
    # same rule as CatalogIndex, written out so an older model (MODEL_DIR) can be scored too
    flg = tr["target_vessel_id"].map(cat.set_index("vessel_id")["flag_name"]).astype(str)
    jp = (flg.eq("Japan") | names.str.contains(r"(?i)maru")).to_numpy()
    cn = flg.isin({"Peoples' Republic of China", "Hong Kong", "Taiwan"}).to_numpy()
    nws = [gen.name_words(n, ko.get(n, ""), j, j or c) if hasattr(gen, "name_words") else None
           for n, j, c in zip(names, jp, cn)]
    is_mx = np.array([w is not None and (j or any(ph for _, ph in w[0])) for w, j in zip(nws, jp)])
    import inspect
    mw_cn = hasattr(gen, "meaning_words") and "chinese" in inspect.signature(gen.meaning_words).parameters
    mws = [(gen.meaning_words(n, ko.get(n, ""), c) if mw_cn else gen.meaning_words(n, ko.get(n, "")))
           if hasattr(gen, "meaning_words") else None for n, c in zip(names, cn)]
    is_mn = np.array([bool(gen.meaning_keys(w)) if w else False for w in mws])
    res = []
    for k, (_, b) in enumerate(folds):
        rng = np.random.default_rng(777 + k)
        p_py = rng.permutation(np.intersect1d(b, np.flatnonzero(is_py)))[:per_fold]
        p_mr = np.intersect1d(b, np.flatnonzero(is_maru))
        p_mx = rng.permutation(np.intersect1d(b, np.flatnonzero(is_mx)))[:per_fold]
        p_mn = rng.permutation(np.intersect1d(b, np.flatnonzero(is_mn)))[:per_fold]
        pick = np.concatenate([p_py, p_mr, p_mx, p_mn])
        base = tr.iloc[pick].copy()
        kind = np.array(["hanja"] * len(p_py) + ["maru"] * len(p_mr)
                        + ["japan" if jp[i] else "mixed" for i in p_mx] + ["meaning"] * len(p_mn))
        mrng = np.random.default_rng(555 + k)  # own stream: hanja/maru records stay as in earlier runs
        base["vessel_name"] = [re.sub(r"\s*마루", "환", ko[n]) if kd == "maru"
                               else gen.record_reading(n, rng) if kd == "hanja"
                               else gen.record_meaning(mws[i], mrng) if kd == "meaning"
                               else gen.record_mixed(nws[i], mrng)
                               for n, kd, i in zip(names.iloc[pick], kind, pick)]
        for cond in ("with_gt", "no_gt"):
            df = base if cond == "with_gt" else base.assign(gross_tonnage=np.nan, international_gross_tonnage=np.nan)
            recs = df.to_dict("records")
            for s in range(0, len(recs), 256):
                for r, (rec, (_, cands, X)) in enumerate(zip(recs[s:s + 256], matchers[k].candidates(recs[s:s + 256])), s):
                    pred = idx.ids[cands[int(np.argmax(models[k].predict(X)))]]
                    res.append((kind[r], cond, pred == rec["target_vessel_id"], bool((idx.ids[cands] == rec["target_vessel_id"]).any()),
                                rec["vessel_name"], names.iloc[pick[r]]))
        print(f"fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    r = pd.DataFrame(res, columns=["kind", "cond", "correct", "recall", "record", "catalog"])
    out = r.groupby(["kind", "cond"]).agg(n=("correct", "size"), acc=("correct", "mean"), recall=("recall", "mean"))
    print(out.round(4).to_string())
    (ROOT / "analysis").mkdir(exist_ok=True)
    out.to_csv(ROOT / "analysis" / f"hanja_eval_{tag}.csv", encoding="utf-8-sig")
    r[~r.correct].head(40).to_csv(ROOT / "analysis" / f"hanja_errors_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "open")
    ap.add_argument("--per-fold", type=int, default=150)
    ap.add_argument("--tag", default="on")
    ap.add_argument("--gen", type=Path, help="hanja.py whose table writes the synthetic records")
    a = ap.parse_args()
    main(a.data, a.per_fold, a.tag, a.gen)
