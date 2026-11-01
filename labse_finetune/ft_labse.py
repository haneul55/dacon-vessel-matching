"""Fine-tune LaBSE (module C) on (Korean record -> English catalog name) pairs. Runs on a Kaggle GPU.

Pairs come only from train data (DACON talk #418501): train's own Hangul records + the LLM-written
records of train vessels (synth*.csv). Folds are the same GroupKFold(5) by vessel as train.py, so the
fold-k encoder never sees fold k's vessels and the ranker's out-of-fold features stay honest.

Loss: in-batch contrastive (InfoNCE). A record must score its own vessel's name above every other name
in the batch, plus one hard negative per record: a different vessel with the same gross tonnage (a sister
ship) when there is one. Word embeddings (385M parameters) are frozen; the 12 transformer layers + pooler
(86M) are trained.

Outputs in <out_dir>:
  fold{k}_names.npy   vectors of names_to_embed.csv texts by the fold-k encoder  (n, 768) float16
  fold{k}_cat.npy     vectors of the catalog names by the fold-k encoder        (78595, 768) float16
  final/model/        the encoder trained on all folds (shipped in the submission)
  final_cat.npy       catalog vectors by the final encoder                       (78595, 768) float16
  log.json            per fold: loss per epoch, top-1 retrieval accuracy before/after on held-out vessels

usage: python ft_labse.py <train.csv> <vessel_catalog.csv> <names_to_embed.csv> <out_dir> <base_model>
                          <synth.csv> [<synth2.csv> ...] [--epochs 3] [--quick 300]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer

MAX_LEN, SCALE = 32, 20.0


def clean_ko(s) -> str:  # same cleaning as emb_feat.clean_ko (inference)
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"제\s*(\d+)", r"\1 ", s)
    s = re.sub(r"(?<=\d)\s*호|호\s*$", " ", s)
    return " ".join(s.split())


def synth_ok(ko: str) -> bool:  # same filter as load_synth.clean_ok: Hangul/digits only, >= 2 syllables
    return bool(re.fullmatch(r"[가-힣0-9 \-.]+", ko.strip())) and len(re.findall("[가-힣]", ko)) >= 2


def load_pairs(train: pd.DataFrame, synth_paths: list[str]) -> pd.DataFrame:
    """(text, vessel_id, kind) pairs: train Hangul records + LLM records of train vessels."""
    hang = train["vessel_name"].astype(str).str.contains("[가-힣]")
    rows = [pd.DataFrame({"text": train.loc[hang, "vessel_name"].map(clean_ko),
                          "vessel_id": train.loc[hang, "target_vessel_id"], "kind": "train"})]
    ids = set(train["target_vessel_id"])
    for p in synth_paths:
        s = pd.read_csv(p, encoding="utf-8-sig").dropna(subset=["ko"])
        s = s[s["vessel_id"].isin(ids) & s["ko"].map(synth_ok)]
        rows.append(pd.DataFrame({"text": s["ko"].map(clean_ko), "vessel_id": s["vessel_id"], "kind": "synth"}))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["text", "vessel_id"]).reset_index(drop=True)


class Encoder:
    def __init__(self, path: str, dev: str):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float32).to(dev)
        self.dev = dev

    def __call__(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.dev)
        return torch.nn.functional.normalize(self.model(**enc).pooler_output, dim=-1)

    @torch.no_grad()
    def embed(self, texts: list[str], batch: int = 512, one_by_one: bool = False) -> np.ndarray:
        """one_by_one: no padding, exactly how inference embeds a record (Embedder.vec)."""
        self.model.eval()
        out = []
        with torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
            step = 1 if one_by_one else batch
            for s in range(0, len(texts), step):
                out.append(self(texts[s:s + step]).float().cpu().numpy())
        return np.concatenate(out).astype(np.float16)


class Sisters:
    """Lookups for hard negatives, built once: vessel -> tonnage, tonnage -> names."""

    def __init__(self, cat: pd.DataFrame):
        self.gt = cat.set_index("vessel_id")["gross_tonnage"].to_dict()
        self.by_gt = cat.groupby("gross_tonnage")["vessel_name"].apply(lambda s: s.astype(str).tolist()).to_dict()
        self.name = cat.set_index("vessel_id")["vessel_name"].astype(str).to_dict()
        self.all = cat["vessel_name"].astype(str).tolist()


def hard_negative(vids: list[str], sis: Sisters, rng) -> list[str]:
    """Per pair, the name of another vessel with the same gross tonnage (sister ship), else a random one."""
    gt, by_gt, name, cat_names = sis.gt, sis.by_gt, sis.name, sis.all
    out = []
    for v in vids:
        pool = [n for n in by_gt.get(gt.get(v), []) if n != name[v]]
        out.append(pool[rng.integers(len(pool))] if pool else cat_names[rng.integers(len(cat_names))])
    return out


def train_encoder(base: str, pairs: pd.DataFrame, cat: pd.DataFrame, dev: str, epochs: int, batch: int = 64,
                  lr: float = 2e-5, seed: int = 0) -> tuple[Encoder, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(base, dev)
    enc.model.embeddings.word_embeddings.weight.requires_grad_(False)  # 385M frozen
    params = [p for p in enc.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(pairs) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / (0.1 * steps)) * max(0.0, 1 - t / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    sis = Sisters(cat)
    name = sis.name
    losses = []
    for ep in range(epochs):
        enc.model.train()
        order = rng.permutation(len(pairs))
        tot, n = 0.0, 0
        for s in range(0, len(order), batch):
            b = pairs.iloc[order[s:s + batch]]
            b = b.drop_duplicates("vessel_id")  # one record per vessel per batch: no false negatives
            vids = b["vessel_id"].tolist()
            names = [name[v] for v in vids] + hard_negative(vids, sis, rng)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                q = enc(b["text"].tolist())
                d = enc(names)
                logits = SCALE * q.float() @ d.float().T
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(vids), device=dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(vids); n += len(vids)
        losses.append(tot / max(n, 1))
        print(f"  epoch {ep + 1}/{epochs}: loss {losses[-1]:.4f}", flush=True)
    return enc, losses


def top1(enc: Encoder, pairs: pd.DataFrame, cat_vecs: np.ndarray, cat_ids: np.ndarray) -> dict:
    """Retrieval by embedding alone over the whole catalog (a diagnostic; the ranker uses more)."""
    if not len(pairs):
        return {}
    q = enc.embed(pairs["text"].tolist()).astype(np.float32)
    C = torch.from_numpy(cat_vecs.astype(np.float32))
    hit = []
    for s in range(0, len(q), 1024):
        pred = (torch.from_numpy(q[s:s + 1024]) @ C.T).argmax(1).numpy()
        hit.append(cat_ids[pred] == pairs["vessel_id"].to_numpy()[s:s + 1024])
    hit = np.concatenate(hit)
    return {k: round(float(hit[(pairs["kind"] == k).to_numpy()].mean()), 4) for k in pairs["kind"].unique()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("catalog"); ap.add_argument("names"); ap.add_argument("out")
    ap.add_argument("base"); ap.add_argument("synth", nargs="+")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--quick", type=int, default=0, help="CPU smoke test: this many pairs, 2 folds, catalog subset")
    a = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(a.train, encoding="utf-8-sig")
    cat = pd.read_csv(a.catalog, encoding="utf-8-sig")
    names = pd.read_csv(a.names, encoding="utf-8-sig")["text"].astype(str).tolist()
    pairs = load_pairs(tr, a.synth)
    print(f"{dev}: {len(pairs)} pairs ({pairs['kind'].value_counts().to_dict()}), {len(names)} names to embed", flush=True)

    # same folds as train.py: GroupKFold(5) over train rows grouped by vessel -> each vessel's fold
    folds = list(GroupKFold(5).split(tr, groups=tr["target_vessel_id"]))
    fold_of = {}
    for k, (_, b) in enumerate(folds):
        for v in tr["target_vessel_id"].iloc[b]:
            fold_of[v] = k
    pairs["fold"] = pairs["vessel_id"].map(fold_of)
    cat_texts = cat["vessel_name"].astype(str).tolist()
    cat_ids = cat["vessel_id"].to_numpy()
    n_folds = 5
    if a.quick:  # tiny CPU run: does every step work?
        pairs = pairs.sample(n=min(a.quick, len(pairs)), random_state=0)
        keep = np.isin(cat_ids, pairs["vessel_id"].unique()) | (np.arange(len(cat)) < 2000)
        cat, cat_texts, cat_ids = cat[keep].reset_index(drop=True), list(np.array(cat_texts)[keep]), cat_ids[keep]
        names = names[:50]
        n_folds = 1

    log = {"pairs": pairs["kind"].value_counts().to_dict(), "folds": []}
    base_enc = Encoder(a.base, dev)
    base_cat = base_enc.embed(cat_texts)
    del base_enc
    for k in range(n_folds):
        tr_p, va_p = pairs[pairs["fold"] != k], pairs[pairs["fold"] == k]
        print(f"fold {k}: train {len(tr_p)} pairs, held-out {len(va_p)} ({time.time() - t0:.0f}s)", flush=True)
        enc, losses = train_encoder(a.base, tr_p, cat, dev, a.epochs, seed=k)
        cat_vecs = enc.embed(cat_texts)
        before = top1(Encoder(a.base, dev), va_p, base_cat, cat_ids) if k == 0 else None
        after = top1(enc, va_p, cat_vecs, cat_ids)
        print(f"  held-out top-1 by embedding: {after}" + (f" (base LaBSE {before})" if before else ""), flush=True)
        np.save(out / f"fold{k}_cat.npy", cat_vecs)
        np.save(out / f"fold{k}_names.npy", enc.embed(names, one_by_one=True))
        log["folds"].append({"fold": k, "train_pairs": len(tr_p), "heldout_pairs": len(va_p), "loss": losses,
                             "heldout_top1": after, "heldout_top1_base": before})
        del enc
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"final encoder on all {len(pairs)} pairs ({time.time() - t0:.0f}s)", flush=True)
    enc, losses = train_encoder(a.base, pairs, cat, dev, a.epochs, seed=99)
    log["final_loss"] = losses
    enc.model.half().save_pretrained(out / "final" / "model", safe_serialization=True)
    enc.tok.save_pretrained(out / "final" / "model")
    enc.model.float()
    np.save(out / "final_cat.npy", enc.embed(cat_texts))
    (out / "log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
