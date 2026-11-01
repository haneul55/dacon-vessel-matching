"""Kaggle GPU: write Korean renderings of train vessels' names in several styles with a public pretrained
LLM (Qwen2.5-7B-Instruct, 4-bit). Only vessels that appear in train.csv are used: DACON allows LLM
augmentation "train데이터를 이용한 방법에 한해" (talk #418501, 2026-10-23). Their English names come from
vessel_catalog.csv. Output synth.csv (vessel_id, vessel_name, style, ko); resumable.

usage: python gen_synth.py <train.csv> <vessel_catalog.csv> <out_dir> [n_names] [seed]
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-7B-Instruct"
STYLES = ["sound", "hanja", "meaning", "mixed", "short"]
SYSTEM = "당신은 한국 항만 입출항 기록 담당자입니다. 외국 선박의 영문 이름을 한국 기록에 적듯이 한글로 옮깁니다."
GUIDE = """선박 영문 이름: "{name}" (국적: {flag}, 선종: {vtype})
이 선박이 한국 항만 기록에서 한글로 적힐 수 있는 표기를 방식별로 하나씩 만드세요. 해당되지 않는 방식은 null.
- sound: 영어(또는 원어) 발음대로 한글 표기. 예) Ocean Star → 오션 스타
- hanja: 중국어 병음이나 일본어 이름이면 한자의 한국 한자음. 예) Jin Cheng → 금성, Fuji Maru → 부사환
- meaning: 영어 단어를 우리말 뜻으로 번역. 예) Ocean Star → 해양성, Golden Glory → 황금영광
- mixed: 일부 단어는 뜻이나 한자음, 나머지는 소리. 예) Hai Feng Glory → 해풍 글로리
- short: 기록자가 줄여 쓰거나 일부를 빼고 적은 표기. 예) Pacific Pioneer No. 3 → 퍼시픽파이오니어3
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


def main(train_path: str, cat_path: str, out_dir: str, n_names: int = 100000, seed: int = 7, batch: int = 24):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "synth.csv"
    cat = pd.read_csv(cat_path, encoding="utf-8-sig")
    train_ids = set(pd.read_csv(train_path, encoding="utf-8-sig")["target_vessel_id"])
    cat = cat[cat["vessel_id"].isin(train_ids)]  # train vessels only
    pick = cat.sample(n=min(n_names, len(cat)), random_state=seed)
    done = set(pd.read_csv(path, encoding="utf-8-sig")["vessel_id"]) if path.exists() else set()
    todo = pick[~pick["vessel_id"].isin(done)]
    print(f"{len(todo)} names to go ({len(done)} done)", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)).eval()
    t0 = time.time()
    for s in range(0, len(todo), batch):
        part = todo.iloc[s:s + batch]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": GUIDE.format(name=r.vessel_name, flag=r.flag_name, vtype=r.vessel_type)}],
            tokenize=False, add_generation_prompt=True) for r in part.itertuples()]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows = [(r.vessel_id, r.vessel_name, st, ko) for r, t in zip(part.itertuples(), texts) for st, ko in parse(t).items()]
        pd.DataFrame(rows, columns=["vessel_id", "vessel_name", "style", "ko"]).to_csv(
            path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
        if (s // batch) % 20 == 0:
            n = s + len(part)
            print(f"{n}/{len(todo)} ({(time.time() - t0) / n:.2f}s/name)", flush=True)
    print("done", path)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(int(x) for x in a[3:5]))
