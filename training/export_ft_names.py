"""Every Hangul text whose LaBSE vector training and validation will ask for, cleaned like
emb_feat.clean_ko -> <out>/names_to_embed.csv (column `text`). The Kaggle fine-tuning step embeds this
list once per fold encoder, so training never needs a GPU locally.

Covers: train rows, every augmented copy make_aug builds (same seeds and SYNTH_AUG files as training),
the synthetic validation records (synth_eval), and the stress_eval conditions (same seeds).

usage: MODEL_DIR=submit_K/model HANJA=1 SYNTH_AUG="llm_synth/synth.csv;llm_synth/synth2.csv" \
       python export_ft_names.py <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stress_eval import CONDITIONS, ROOT
from train import N_FOLDS, make_aug
from vessel_lib import is_hangul
from emb_feat import clean_ko
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(ROOT / "llm_synth"))
from load_synth import load_records  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
data = ROOT.parent / "open"
tr = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
cat = pd.read_csv(data / "vessel_catalog.csv", encoding="utf-8-sig")
hang = tr["vessel_name"].map(is_hangul).to_numpy()
texts = list(tr["vessel_name"])
aug, _ = make_aug(tr, hang, cat=cat)
texts += list(aug["vessel_name"])
texts += list(load_records(ROOT / "llm_synth" / "synth.csv")["vessel_name"])  # synth_eval's records

# stress_eval: same evaluation rows and seeds
folds = list(GroupKFold(N_FOLDS).split(tr, groups=tr["target_vessel_id"]))
pick = np.random.default_rng(7)
latin_rows = np.flatnonzero(~hang)
eval_rows = np.sort(np.concatenate([np.flatnonzero(hang), pick.choice(latin_rows, min(400 * N_FOLDS, len(latin_rows)), False)]))
for k, (_, b) in enumerate(folds):
    base = tr.iloc[np.intersect1d(b, eval_rows)].reset_index(drop=True)
    for ci, (cname, fn) in enumerate(CONDITIONS.items()):
        texts += list(fn(base, np.random.default_rng(1000 * k + ci))["vessel_name"])

keys = sorted({clean_ko(t) for t in texts if isinstance(t, str) and is_hangul(t)})
pd.DataFrame({"text": keys}).to_csv(out / "names_to_embed.csv", index=False, encoding="utf-8-sig")
print(len(keys), "texts ->", out / "names_to_embed.csv")
