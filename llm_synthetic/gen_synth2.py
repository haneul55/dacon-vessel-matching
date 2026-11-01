"""Round 2 (Kaggle GPU): more and cleaner Korean renderings of train vessels' names.
Changes from gen_synth.py: a larger model (Qwen2.5-14B-Instruct, 4-bit; falls back to 7B), stricter rules
against invented words and Latin letters, and n_samples sampled answers per vessel (more variety).
Only vessels in train.csv are used (DACON talk #418501: LLM augmentation allowed only from train data).
Output synth2.csv (synth2_<k>.csv with SHARD=k/n) (vessel_id, vessel_name, sample, style, ko); resumable; stops itself before max_hours.

usage: python gen_synth2.py <train.csv> <vessel_catalog.csv> <out_dir> [model] [n_samples] [max_hours] [n_names]
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

STYLES = ["sound", "hanja", "meaning", "mixed", "short"]
SYSTEM = ("당신은 한국 항만 입출항 기록 담당자입니다. 외국 선박의 영문 이름을 한국 기록에 적듯이 한글로 옮깁니다. "
          "규칙: 한글과 숫자만 쓰고 영어 알파벳은 절대 쓰지 않습니다. 원래 이름에 없는 단어를 지어내지 않습니다.")
GUIDE = """선박 영문 이름: "{name}" (국적: {flag}, 선종: {vtype})
이 선박이 한국 항만 기록에서 한글로 적힐 수 있는 표기를 방식별로 하나씩 만드세요. 해당되지 않으면 null.
- sound: 발음대로 한글 표기. 약어는 알파벳을 한 글자씩 읽음. 예) Ocean Star → 오션 스타, KMTC Busan → 케이엠티씨 부산
- hanja: 중국어 병음이나 일본어 이름일 때만, 한자의 한국 한자음. 예) Jin Cheng → 금성, Hai Feng 8 → 해풍 8, Fuji Maru → 부사환
- meaning: 이름 속 실제 영어 단어를 우리말 뜻으로 번역(사람·회사 이름은 소리 그대로). 예) Ocean Star → 해양성, Golden Glory → 황금 영광, Pacific Pioneer → 태평양 개척자
- mixed: 일부 단어만 뜻이나 한자음, 나머지는 소리. 예) Hai Feng Glory → 해풍 글로리, Ocean Pioneer → 해양 파이오니어
- short: 기록자가 띄어쓰기·NO.·일부 단어를 빼고 적은 표기. 예) Pacific Pioneer No. 3 → 퍼시픽파이오니어3, SC Golden Fortune → 골든포춘
숫자는 그대로 두세요. JSON 한 줄로만 답하세요: {{"sound": ..., "hanja": ..., "meaning": ..., "mixed": ..., "short": ...}}"""


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {s: d[s].strip() for s in STYLES if isinstance(d.get(s), str) and re.search("[가-힣]", d[s])}


def load(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)).eval()
    return tok, model


def main(train_path, cat_path, out_dir, model_name="Qwen/Qwen2.5-14B-Instruct", n_samples=2, max_hours=10.5,
         n_names=100000, seed=11, batch=16):
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shard, n_shards = map(int, os.environ.get("SHARD", "0/1").split("/"))  # SHARD=k/n: one process per GPU
    path = out / ("synth2.csv" if n_shards == 1 else f"synth2_{shard}.csv")
    cat = pd.read_csv(cat_path, encoding="utf-8-sig")
    train_ids = set(pd.read_csv(train_path, encoding="utf-8-sig")["target_vessel_id"])
    cat = cat[cat["vessel_id"].isin(train_ids)]
    pick = cat.sample(n=min(n_names, len(cat)), random_state=seed)
    done = set(pd.read_csv(path, encoding="utf-8-sig")["vessel_id"]) if path.exists() else set()
    todo = pick[~pick["vessel_id"].isin(done)].iloc[shard::n_shards]
    print(f"{model_name}: {len(todo)} names to go ({len(done)} done)", flush=True)
    tok, model = load(model_name)
    for s in range(0, len(todo), batch):
        if time.time() - t0 > max_hours * 3600:
            print("time budget reached, stopping (output so far is saved)")
            break
        part = todo.iloc[s:s + batch]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": GUIDE.format(name=r.vessel_name, flag=r.flag_name, vtype=r.vessel_type)}],
            tokenize=False, add_generation_prompt=True) for r in part.itertuples()]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=n_samples > 1, temperature=0.8, top_p=0.95,
                                 num_return_sequences=n_samples, pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows = []
        for j, t in enumerate(texts):
            r = part.iloc[j // n_samples]
            rows += [(r.vessel_id, r.vessel_name, j % n_samples, st, ko) for st, ko in parse(t).items()]
        pd.DataFrame(rows, columns=["vessel_id", "vessel_name", "sample", "style", "ko"]).to_csv(
            path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
        if (s // batch) % 20 == 0:
            n = s + len(part)
            print(f"{n}/{len(todo)} ({(time.time() - t0) / n:.2f}s/name, {(time.time() - t0) / 3600:.1f}h)", flush=True)
    print("done", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) > 3: kw["model_name"] = a[3]
    if len(a) > 4: kw["n_samples"] = int(a[4])
    if len(a) > 5: kw["max_hours"] = float(a[5])
    if len(a) > 6: kw["n_names"] = int(a[6])
    main(a[0], a[1], a[2], **kw)
