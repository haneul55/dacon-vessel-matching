"""Meaning similarity with LaBSE (a public pretrained multilingual sentence encoder), after the teammate's
probe_embfeat model (LB 0.92341 on top of H4_hanja 0.90944).

model/emb/model = LaBSE weights, model/emb/cat_emb.npy = unit embeddings of vessel_catalog.csv names (row
order). A Hangul record is embedded on its own (no batching, so its vector never depends on other rows);
vessel_lib uses it for retrieval (top 10 by similarity within ±10% tonnage) and two features."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np


def clean_ko(s) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"제\s*(\d+)", r"\1 ", s)
    s = re.sub(r"(?<=\d)\s*호|호\s*$", " ", s)
    return " ".join(s.split())


class Embedder:
    """vec(name) -> unit vector of the cleaned name; C = catalog vectors. cache_path keeps vectors between
    training runs (same vectors as computing them again: one name per forward pass)."""

    def __init__(self, emb_dir: Path, cache_path: Path | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(str(emb_dir / "model"))
        self.model = AutoModel.from_pretrained(str(emb_dir / "model"), torch_dtype=dtype).to(self.dev).eval()
        self.C = np.load(emb_dir / "cat_emb.npy").astype(np.float32)
        self.cache_path = cache_path
        self.cache: dict[str, np.ndarray] = {}
        if cache_path is not None and Path(cache_path).is_file():
            d = np.load(cache_path, allow_pickle=False)
            self.cache = dict(zip(d["keys"].tolist(), d["vecs"]))

    def vec(self, name) -> np.ndarray:
        key = clean_ko(name)
        v = self.cache.get(key)
        if v is None:
            torch = self.torch
            enc = self.tok([key], truncation=True, max_length=32, return_tensors="pt").to(self.dev)
            with torch.no_grad():
                out = self.model(**enc).pooler_output.float()
            v = torch.nn.functional.normalize(out, dim=-1)[0].cpu().numpy().astype(np.float32)
            self.cache[key] = v
        return v

    def save(self) -> None:
        if self.cache_path is not None and self.cache:
            import os
            keys = list(self.cache)
            tmp = Path(f"{self.cache_path}.{os.getpid()}.npz")  # parallel runs: replace, never half-write
            np.savez(tmp, keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))
            os.replace(tmp, self.cache_path)


class FoldVectors:
    """Training-time stand-in for Embedder with a fine-tuned encoder per fold (labse_ft/ft_labse.py).
    The fold-k encoder never saw fold k's vessels, so fold k's features stay out-of-fold: set_fold(k)
    before building fold k's rows. Vectors are precomputed on the GPU for every text train.py needs
    (export_ft_names.py); a text outside that list is an error, never a silent vector from another fold."""

    def __init__(self, ft_dir: Path, names_csv: Path):
        import pandas as pd
        self.dir = Path(ft_dir)
        self.keys = pd.read_csv(names_csv, encoding="utf-8-sig")["text"].astype(str).tolist()
        self.fold = None
        self.C = None
        self.cache: dict[str, np.ndarray] = {}

    def set_fold(self, k) -> None:
        if k == self.fold:
            return
        self.fold = k
        self.C = np.load(self.dir / f"fold{k}_cat.npy").astype(np.float32)
        vecs = np.load(self.dir / f"fold{k}_names.npy").astype(np.float32)
        self.cache = dict(zip(self.keys, vecs))

    def vec(self, name) -> np.ndarray:
        key = clean_ko(name)
        if self.fold is None:
            raise RuntimeError("FoldVectors.set_fold(k) was not called")
        if key not in self.cache:
            raise KeyError(f"no fine-tuned vector for {key!r}: rerun export_ft_names.py and the Kaggle step")
        return self.cache[key]

    def save(self) -> None:
        pass
