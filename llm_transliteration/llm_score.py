"""Run on a GPU machine: score every (Hangul query, candidate) pair of topk_han.csv with an LLM,
optionally saving the weights for the offline submission.

usage:
  python llm_score.py --model Qwen/Qwen2.5-7B-Instruct --quant4 --tag q7b
  python llm_score.py --model Qwen/Qwen2.5-3B-Instruct --tag q3b --save weights_q3b
  python llm_score.py --model Qwen/Qwen2.5-1.5B-Instruct --tag q1p5b --save weights_q1p5b
  (--k 5 scores only E's top-5 per row; --limit 300 for a quick timing run)

output: llm_scores_<tag>.csv  (sample_id, cand_id, lp, ntok)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from llm_scorer import LLMScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace id or local folder")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--input", default="topk_han.csv")
    ap.add_argument("--quant4", action="store_true", help="4-bit (needs bitsandbytes)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows (timing)")
    ap.add_argument("--save", default="", help="folder to save tokenizer+weights for the submission")
    a = ap.parse_args()

    import torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    t0 = time.time()
    scorer = LLMScorer(a.model, quant4=a.quant4)
    print(f"model loaded {time.time() - t0:.0f}s, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f}GB",
          flush=True)
    print("sanity:", scorer.score("에버 얼라이언스", ["Ever Alliance", "Ever Reliance"]), flush=True)
    if a.save:
        scorer.save(a.save)
        size = sum(p.stat().st_size for p in Path(a.save).rglob("*") if p.is_file()) / 2**30
        print(f"saved weights -> {a.save} ({size:.2f}GB)", flush=True)

    df = pd.read_csv(a.input, encoding="utf-8-sig")
    df = df[df["rank"] <= a.k]
    groups = list(df.groupby("sample_id", sort=False))
    if a.limit:
        groups = groups[:a.limit]
    print(f"rows {len(groups)} | pairs {sum(len(g) for _, g in groups)}", flush=True)
    out, t0 = [], time.time()
    for n, (sid, g) in enumerate(groups):
        lp, ntok = scorer.score(g["query"].iloc[0], g["cand"].tolist())
        out += [dict(sample_id=sid, cand_id=c, lp=v, ntok=ntok) for c, v in zip(g["cand_id"], lp)]
        if n % 200 == 0:
            el = time.time() - t0
            print(f"{n}/{len(groups)}  {el:.0f}s  ({el / max(n, 1):.3f}s/row)", flush=True)
    pd.DataFrame(out).to_csv(f"llm_scores_{a.tag}.csv", index=False, encoding="utf-8-sig")
    print(f"done: {(time.time() - t0) / len(groups):.3f}s/row -> llm_scores_{a.tag}.csv", flush=True)


if __name__ == "__main__":
    main()
