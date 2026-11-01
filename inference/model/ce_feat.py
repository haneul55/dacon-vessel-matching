"""Cross-encoder for Hangul records (ce/ce_train.py): reads a Korean record and an English catalog name
together and scores whether they are the same vessel. vessel_lib adds CE_W * score to the ranker's
top CE_TOP candidates of a Hangul row (one row's pairs per forward pass, so rows stay independent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_LEN = 48


class CrossScorer:
    def __init__(self, ce_dir: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(str(ce_dir))
        self.enc = AutoModel.from_pretrained(str(ce_dir), torch_dtype=dtype).to(self.dev).eval()
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(Path(ce_dir) / "head.pt", map_location="cpu"))
        self.head = self.head.to(self.dev, dtype).eval()

    def score(self, query, names) -> np.ndarray:
        torch = self.torch
        enc = self.tok([str(query)] * len(names), [str(n) for n in names], padding=True, truncation=True,
                       max_length=MAX_LEN, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            return self.head(self.enc(**enc).last_hidden_state[:, 0]).squeeze(-1).float().cpu().numpy()
