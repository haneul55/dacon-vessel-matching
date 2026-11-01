"""LLM multiple choice (Kaggle GPU): which of the ranker's top-K English names is this Korean record?

For each record in ce_pairs.csv (dump_ce.py: the ranker's out-of-fold top-20 per Hangul record), the
top-K candidates are shown as options A, B, C... in the ranker's order, and the LLM's next-token
log-probabilities of the option letters become a score per candidate. No training: the LLM never sees
our labels, so these scores are honest on every record. The same 4-bit Qwen2.5-7B setup ran on the
DACON server before (submit_llm_fixed_greedy, 9 min), so what is measured here can ship.

Output llm_mc.csv: rid, cand_id, rank, logp   (one row per record x option)

usage: python llm_mc.py <ce_pairs.csv> <out_dir> [--model Qwen/Qwen2.5-7B-Instruct] [--k 8] [--quick 40] [--no-4bit]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "ABCDEFGHIJ"
SYSTEM = "당신은 한국 항만 입출항 기록의 선박명을 국제 선박 명부의 영문 선박명과 대조하는 전문가입니다."
QUESTION = """한국 항만 기록의 선박명: "{query}"
이 기록은 아래 선박 중 어느 것입니까? 한글 표기는 영문을 소리 나는 대로 적은 것일 수도 있고, 중국어·일본어 이름의 한자를 한국 한자음으로 읽은 것(예: Hai Feng → 해풍, Maru → 환), 영어 단어를 우리말 뜻으로 옮긴 것(예: Ocean Star → 해양성), 띄어쓰기나 일부 단어를 뺀 줄임일 수도 있습니다.
{options}
보기의 알파벳 하나로만 답하세요."""


def prompt(tok, query: str, cands: list[str]) -> str:
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION.format(query=query, options=options)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load(model_name: str, four_bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    kw = {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw = dict(device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if four_bit else torch.float32, **kw).eval()
    return tok, model


@torch.no_grad()
def letter_logp(tok, model, prompts: list[str], n_opts: list[int], batch: int = 16) -> list[np.ndarray]:
    """Log-probabilities over the first n option letters of the next token, per prompt."""
    ids = [tok.encode(L, add_special_tokens=False) for L in LETTERS]
    assert all(len(i) == 1 for i in ids), "option letters must be single tokens"
    ids = torch.tensor([i[0] for i in ids])
    out = []
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s:s + batch], return_tensors="pt", padding=True).to(model.device)
        last = model(**enc).logits[:, -1, :].float().cpu()
        for row, n in zip(last, n_opts[s:s + batch]):
            out.append(torch.log_softmax(row[ids[:n]], dim=-1).numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.pairs, encoding="utf-8-sig")
    d = d[(d["rank"] >= 1) & (d["rank"] <= a.k)].sort_values(["rid", "rank"])
    if a.quick:
        d = d[d["rid"].isin(d["rid"].drop_duplicates().sample(a.quick, random_state=0))]
    groups = [g for _, g in d.groupby("rid", sort=False)]
    tok, model = load(a.model, not a.no_4bit)
    print(f"{a.model}: {len(groups)} records, top-{a.k} ({time.time() - t0:.0f}s)", flush=True)
    rows, hit = [], []
    step = 2000
    for s in range(0, len(groups), step):
        part = groups[s:s + step]
        ps = [prompt(tok, str(g["query"].iloc[0]), g["cand"].astype(str).tolist()) for g in part]
        lps = letter_logp(tok, model, ps, [len(g) for g in part], a.batch)
        for g, lp in zip(part, lps):
            rows += list(zip(g["rid"], g["cand_id"], g["rank"], lp))
            hit.append(int(g["label"].to_numpy()[int(np.argmax(lp))] == 1))
        print(f"{s + len(part)}/{len(groups)}: LLM-alone top-1 {np.mean(hit):.4f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows, columns=["rid", "cand_id", "rank", "logp"]).to_csv(out / "llm_mc.csv", index=False, encoding="utf-8-sig")
    r1 = d[d["rank"] == 1]["label"].mean()
    print(f"ranker top-1 {r1:.4f} | LLM-alone top-1 {np.mean(hit):.4f} | done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
