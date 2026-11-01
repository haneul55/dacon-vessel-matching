"""Quality score of each LLM-written record: LaBSE cosine between the Korean record and the vessel's
English catalog name (same encoder and cleaning as the model). Very low scores are mostly invented
words (화학물류선 for Bow Triumph) that teach the ranker to ignore the name. -> <synth>_quality.csv

usage: python llm_synth/score_synth.py llm_synth/synth.csv submit_J/model
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

synth, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(model_dir))
from emb_feat import Embedder  # noqa: E402

root = Path(__file__).resolve().parents[1]
emb = Embedder(model_dir / "emb", root / "analysis" / "emb_cache.npz")
cat = pd.read_csv(root.parent / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
row = {v: i for i, v in enumerate(cat["vessel_id"])}
d = pd.read_csv(synth, encoding="utf-8-sig").dropna(subset=["ko"]).drop_duplicates(["vessel_id", "ko"])
d["cos"] = [float(emb.C[row[v]] @ emb.vec(k)) for v, k in zip(d["vessel_id"], d["ko"])]
emb.save()
d[["vessel_id", "ko", "style", "cos"]].to_csv(synth.with_name(synth.stem + "_quality.csv"), index=False, encoding="utf-8-sig")
print(d.groupby("style")["cos"].describe().round(3))
