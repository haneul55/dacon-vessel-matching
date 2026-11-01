"""Offline: LaBSE vectors of each catalog vessel's Korean spelling (catalog_ko.csv), in vessel_catalog.csv
row order -> <model>/emb/cat_ko_emb.npy (float16, zero vector where there is no spelling). Same public
pretrained model and cleaning as emb_feat.Embedder; computed once from official data only.

usage: python emb_catalog_ko.py submit_I/model
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

model_dir = Path(sys.argv[1])
sys.path.insert(0, str(model_dir))
from emb_feat import clean_ko  # noqa: E402
from vessel_lib import load_catalog_ko  # noqa: E402

cat = pd.read_csv(Path(__file__).resolve().parent.parent / "open" / "vessel_catalog.csv", encoding="utf-8-sig")
ko = load_catalog_ko(model_dir / "catalog_ko.csv")
texts = [clean_ko(ko.get(str(n), "")) for n in cat["vessel_name"].astype(str)]
tok = AutoTokenizer.from_pretrained(str(model_dir / "emb" / "model"))
mod = AutoModel.from_pretrained(str(model_dir / "emb" / "model"), torch_dtype=torch.float32).eval()
torch.set_num_threads(4)
out = np.zeros((len(texts), 768), dtype=np.float16)
todo = [i for i, t in enumerate(texts) if t]
for s in range(0, len(todo), 128):
    ids = todo[s:s + 128]
    enc = tok([texts[i] for i in ids], padding=True, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        v = torch.nn.functional.normalize(mod(**enc).pooler_output.float(), dim=-1).numpy()
    out[ids] = v.astype(np.float16)
    if s % 12800 == 0:
        print(s, len(todo), flush=True)
np.save(model_dir / "emb" / "cat_ko_emb.npy", out)
print("saved", out.shape, "with spelling", len(todo))
