"""Cross-encoder for Hangul records (Kaggle GPU): reads (Korean record, English catalog name) together
and scores how likely they are the same vessel. It re-scores the ranker's top-K for Hangul rows only.

Training data is ce_pairs.csv from dump_ce.py: train's own Hangul rows + LLM-written records of train
vessels (DACON talk #418501: augmentation from train data only), each with the ranker's out-of-fold
top-20 candidates as hard negatives. Folds are the same GroupKFold(5) by vessel as train.py, so the
fold-k cross-encoder never sees fold k's vessels and its scores for fold k are honest (out-of-fold).

Loss: listwise softmax per record (the true vessel against up to 15 of the ranker's own top picks).
Word embeddings (385M) are frozen like ft_labse.py; the 12 transformer layers + a new linear head train.

Outputs in <out_dir>:
  ce_oof.csv     rid, cand_id, ce   (fold-k model's score for fold-k records; every row of ce_pairs.csv)
  final/model/   encoder trained on all folds + head.pt (shipped in the submission)
  log.json       per fold: loss per epoch, top-1 among the ranker's top-20 before/after (ce alone)

usage: python ce_train.py <ce_pairs.csv> <out_dir> <base_model> [--epochs 2] [--group 16] [--quick 400]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 48


class CrossEncoder(torch.nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        head = Path(path) / "head.pt"
        if head.is_file():
            self.head.load_state_dict(torch.load(head, map_location="cpu"))

    def forward(self, **enc) -> torch.Tensor:
        return self.head(self.enc(**enc).last_hidden_state[:, 0]).squeeze(-1)


def groups_of(d: pd.DataFrame, g: int, rng) -> list[np.ndarray]:
    """Per record: the true vessel's row + up to g-1 of the ranker's wrong picks (row indices into d)."""
    out = []
    for _, part in d.groupby("rid", sort=False):
        pos = part.index[part["label"].to_numpy() == 1]
        neg = part.index[(part["label"].to_numpy() == 0) & (part["rank"].to_numpy() >= 1)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        neg = neg[:g - 1] if len(neg) <= g - 1 else np.sort(rng.choice(neg, g - 1, replace=False))
        out.append(np.concatenate([pos[:1], neg]))
    return out


def train_ce(base: str, d: pd.DataFrame, tok, dev: str, epochs: int, g: int, batch_groups: int = 8,
             lr: float = 3e-5, seed: int = 0) -> tuple[CrossEncoder, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CrossEncoder(base).to(dev)
    model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n_groups = d["rid"].nunique()
    steps = epochs * math.ceil(n_groups / batch_groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / (0.1 * steps)) * max(0.0, 1 - t / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    q, c = d["query"].astype(str).to_numpy(), d["cand"].astype(str).to_numpy()
    losses = []
    for ep in range(epochs):
        model.train()
        gs = groups_of(d, g, rng)  # fresh negatives each epoch
        order = rng.permutation(len(gs))
        tot, n = 0.0, 0
        for s in range(0, len(order), batch_groups):
            bg = [gs[i] for i in order[s:s + batch_groups]]
            ix = np.concatenate(bg)
            enc = tok(list(q[ix]), list(c[ix]), padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(**enc).float()
            loss, o = 0.0, 0
            for grp in bg:  # true vessel is first in each group
                loss = loss + torch.nn.functional.cross_entropy(logits[o:o + len(grp)][None], torch.zeros(1, dtype=torch.long, device=dev))
                o += len(grp)
            loss = loss / len(bg)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(bg); n += len(bg)
        losses.append(tot / max(n, 1))
        print(f"  epoch {ep + 1}/{epochs}: loss {losses[-1]:.4f}", flush=True)
    return model, losses


@torch.no_grad()
def score(model: CrossEncoder, tok, d: pd.DataFrame, dev: str, batch: int = 512) -> np.ndarray:
    model.eval()
    q, c = d["query"].astype(str).tolist(), d["cand"].astype(str).tolist()
    out = []
    for s in range(0, len(q), batch):
        enc = tok(q[s:s + batch], c[s:s + batch], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(**enc).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def top1(d: pd.DataFrame, col: str) -> float:
    """Share of records whose best row among the ranker's top-K (rank >= 1) is the true vessel."""
    x = d[d["rank"] >= 1]
    best = x.loc[x.groupby("rid")[col].idxmax()]
    return round(float(best["label"].mean()), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("out"); ap.add_argument("base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--quick", type=int, default=0, help="CPU smoke test: this many records, 1 fold, 1 epoch")
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.pairs, encoding="utf-8-sig")
    if a.quick:
        d = d[d["rid"].isin(d["rid"].drop_duplicates().sample(a.quick, random_state=0))].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    print(f"{dev}: {len(d)} rows, {d['rid'].nunique()} records ({time.time() - t0:.0f}s)", flush=True)
    log = {"folds": []}
    d["ce"] = np.nan
    folds = sorted(d["fold"].unique())[: 1 if a.quick else None]
    for k in folds:
        tr_d, va_d = d[d["fold"] != k].reset_index(drop=True), d[d["fold"] == k]
        print(f"fold {k}: train {tr_d['rid'].nunique()} records, held-out {va_d['rid'].nunique()} ({time.time() - t0:.0f}s)", flush=True)
        model, losses = train_ce(a.base, tr_d, tok, dev, 1 if a.quick else a.epochs, a.group, seed=k)
        d.loc[va_d.index, "ce"] = score(model, tok, va_d, dev)
        va = d.loc[va_d.index]
        res = {"fold": int(k), "loss": losses, "top1_ranker": top1(va, "score"), "top1_ce": top1(va, "ce")}
        print(f"  {res}", flush=True)
        log["folds"].append(res)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    d[["rid", "cand_id", "ce"]].to_csv(out / "ce_oof.csv", index=False, encoding="utf-8-sig")
    if not a.quick:
        print(f"final model on all records ({time.time() - t0:.0f}s)", flush=True)
        model, losses = train_ce(a.base, d, tok, dev, a.epochs, a.group, seed=99)
        log["final_loss"] = losses
        fin = out / "final" / "model"
        model.enc.half().save_pretrained(fin, safe_serialization=True)
        tok.save_pretrained(fin)
        torch.save(model.head.state_dict(), fin / "head.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
