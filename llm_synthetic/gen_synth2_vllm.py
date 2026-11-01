"""Round 2 on vLLM (Kaggle GPU): the same prompt and output as gen_synth2.py, many times faster, so
every train vessel gets n_samples sampled answers in about an hour instead of 6-8.
Only vessels in train.csv are used (DACON talk #418501: LLM augmentation allowed only from train data).
Output synth2.csv (vessel_id, vessel_name, sample, style, ko); resumable; stops itself before max_hours.

usage: python gen_synth2_vllm.py <train.csv> <vessel_catalog.csv> <out_dir> [model] [n_samples] [max_hours] [tp]
"""
import sys
import time
from pathlib import Path

import pandas as pd

from gen_synth2 import GUIDE, SYSTEM, parse


def main(train_path, cat_path, out_dir, model="Qwen/Qwen2.5-14B-Instruct-AWQ", n_samples=4, max_hours=10.5,
         tp=2, chunk=1000, seed=11):
    from vllm import LLM, SamplingParams
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "synth2.csv"
    cat = pd.read_csv(cat_path, encoding="utf-8-sig")
    train_ids = set(pd.read_csv(train_path, encoding="utf-8-sig")["target_vessel_id"])
    cat = cat[cat["vessel_id"].isin(train_ids)].sample(frac=1, random_state=seed)
    done = set(pd.read_csv(path, encoding="utf-8-sig")["vessel_id"]) if path.exists() else set()
    todo = cat[~cat["vessel_id"].isin(done)]
    print(f"{model}: {len(todo)} vessels to go ({len(done)} done), {n_samples} samples each", flush=True)
    llm = LLM(model=model, tensor_parallel_size=int(tp), dtype="half", max_model_len=1024, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=int(n_samples), temperature=0.8, top_p=0.95, max_tokens=120, seed=seed)
    for s in range(0, len(todo), chunk):
        if time.time() - t0 > float(max_hours) * 3600:
            print("time budget reached, stopping (output so far is saved)", flush=True)
            break
        part = todo.iloc[s:s + chunk]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": GUIDE.format(name=r.vessel_name, flag=r.flag_name, vtype=r.vessel_type)}],
            tokenize=False, add_generation_prompt=True) for r in part.itertuples()]
        res = llm.generate(prompts, sp, use_tqdm=False)
        rows = [(r.vessel_id, r.vessel_name, j, st, ko)
                for r, o in zip(part.itertuples(), res) for j, c in enumerate(o.outputs) for st, ko in parse(c.text).items()]
        pd.DataFrame(rows, columns=["vessel_id", "vessel_name", "sample", "style", "ko"]).to_csv(
            path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
        print(f"{s + len(part)}/{len(todo)} vessels ({time.time() - t0:.0f}s)", flush=True)
    print("done", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) > 3: kw["model"] = a[3]
    if len(a) > 4: kw["n_samples"] = int(a[4])
    if len(a) > 5: kw["max_hours"] = float(a[5])
    if len(a) > 6: kw["tp"] = int(a[6])
    main(a[0], a[1], a[2], **kw)
