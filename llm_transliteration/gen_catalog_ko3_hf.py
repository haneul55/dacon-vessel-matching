"""gen_catalog_ko3.py on plain transformers (vLLM no longer installs on Kaggle's Python 3.13).

Same prompt and output as gen_catalog_ko3.py (vessel_name, hanja, meaning), greedy, 4-bit Qwen2.5-7B.
One process per GPU (SHARD=k/n -> catalog_ko3_<k>.csv). Names most likely to need a hanja or meaning
form go first (pinyin-looking, a word in hanja.MEANINGS, a Chinese/Japanese flag), so a run cut short
by the time budget still covers the useful part. Resumable; saves every batch group.

usage: python gen_catalog_ko3_hf.py <vessel_catalog.csv> <out_dir> [model] [max_hours] [batch]
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gen_catalog_ko3 import GUIDE, SYSTEM, parse

_CJK_FLAGS = ("China", "Hong Kong", "Taiwan", "Japan", "Macao")


def priority(cat: pd.DataFrame) -> pd.Series:
    """0 = likely to have a hanja or meaning form, 1 = the rest."""
    try:
        import hanja  # the model's own tables, if hanja.py sits next to this script
        hj = cat["vessel_name"].map(lambda s: bool(hanja.catalog_keys(s)))
        mn = cat["vessel_name"].map(lambda s: any(hanja.word_meanings(t) for t in re.findall(r"[A-Za-z]+", s)))
    except ImportError:
        hj = mn = pd.Series(False, index=cat.index)
    cjk = cat["flag_name"].astype(str).str.contains("|".join(_CJK_FLAGS))
    return (~(hj | mn | cjk)).astype(int)


def main(cat_path, out_dir, model="Qwen/Qwen2.5-7B-Instruct", max_hours=2.0, batch=48):
    t0 = time.time()
    shard, n_shards = map(int, os.environ.get("SHARD", "0/1").split("/"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / (f"catalog_ko3_{shard}.csv" if n_shards > 1 else "catalog_ko3.csv")
    cat = pd.read_csv(cat_path, encoding="utf-8-sig").drop_duplicates("vessel_name")
    cat["vessel_name"] = cat["vessel_name"].astype(str)
    cat = cat.assign(p=priority(cat)).sort_values(["p", "vessel_name"], kind="stable").iloc[shard::n_shards]
    done = set(pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)["vessel_name"]) if path.exists() else set()
    todo = cat[~cat["vessel_name"].isin(done)]
    print(f"shard {shard}/{n_shards}: {len(todo)} names to do ({len(done)} done; priority 0: {(todo.p == 0).sum()})", flush=True)
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if os.environ.get("NO4BIT"):  # CPU smoke test with a small model
        lm = AutoModelForCausalLM.from_pretrained(model).eval()
    else:
        lm = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
    rows = []
    for s in range(0, len(todo), batch):
        if time.time() - t0 > float(max_hours) * 3600:
            print("time budget reached, stopping (output so far is saved)", flush=True)
            break
        part = todo.iloc[s:s + batch]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": GUIDE.format(name=r.vessel_name, flag=r.flag_name, vtype=r.vessel_type)}],
            tokenize=False, add_generation_prompt=True) for r in part.itertuples()]
        enc = tok(prompts, return_tensors="pt", padding=True).to(lm.device)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        for r, t in zip(part.itertuples(), tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            p = parse(t)
            rows.append({"vessel_name": r.vessel_name, "hanja": p.get("hanja", ""), "meaning": p.get("meaning", "")})
        if (s // batch) % 10 == 9 or s + batch >= len(todo):
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
            rows = []
            n = s + len(part)
            print(f"{n}/{len(todo)} ({(time.time() - t0) / n:.2f}s/name, {(time.time() - t0) / 3600:.2f}h)", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    print("done", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) > 2: kw["model"] = a[2]
    if len(a) > 3: kw["max_hours"] = float(a[3])
    if len(a) > 4: kw["batch"] = int(a[4])
    main(a[0], a[1], **kw)
