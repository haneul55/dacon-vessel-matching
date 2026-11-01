"""Kaggle GPU (vLLM): for every catalog vessel name, how a Korean port record would write it by
hanja reading (Chinese/Japanese names: Hai Feng 8 -> 해풍 8, Fuji Maru -> 부사환) and by meaning
(real English words translated: Ocean Star -> 해양성). Greedy decoding, one answer per name.

Same status as catalog_ko.csv (an LLM-written Korean spelling per catalog name, a preprocessing table
of the given catalog); whether such tables are allowed is the pending DACON question (talk #418410).

Output catalog_ko3.csv: vessel_name, hanja, meaning   (empty when the style does not apply)
Resumable: names already in the output are skipped; saves every chunk; stops before max_hours.

usage: python gen_catalog_ko3.py <vessel_catalog.csv> <out_dir> [model] [max_hours] [tp]
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

SYSTEM = ("당신은 한국 항만 입출항 기록 담당자입니다. 외국 선박의 영문 이름을 한국 기록에 적는 방식으로 한글로 옮깁니다. "
          "규칙: 한글과 숫자만 쓰고 영어 알파벳은 절대 쓰지 않습니다. 원래 이름에 없는 단어를 지어내지 않습니다.")
GUIDE = """선박 영문 이름: "{name}" (국적: {flag}, 선종: {vtype})
두 가지 한글 표기를 만드세요. 해당되지 않으면 null.
- hanja: 이름이 중국어 병음이나 일본어일 때만. 가장 그럴듯한 한자를 떠올려 한국 한자음으로 읽음. 숫자는 그대로.
  예) Jin Cheng → 금성, Hai Feng 8 → 해풍 8, Xin Hua Yuan → 신화원, Fuji Maru → 부사환, COSCO Shipping Jin Xiu → 코스코 쉬핑 금수
- meaning: 이름 속 실제 영어 단어를 우리말 뜻으로 번역. 사람·회사·지명·약어는 소리 그대로 한글로. 숫자는 그대로.
  예) Ocean Star → 해양성, Golden Glory → 황금 영광, Pacific Pioneer 3 → 태평양 개척자 3, Glovis Champion → 글로비스 챔피언
JSON 한 줄로만 답하세요: {{"hanja": ..., "meaning": ...}}"""


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    ok = lambda s: isinstance(s, str) and re.fullmatch(r"[가-힣0-9 \-.]+", s.strip()) and re.search("[가-힣]", s)
    return {k: d[k].strip() for k in ("hanja", "meaning") if ok(d.get(k))}


def main(cat_path, out_dir, model="Qwen/Qwen2.5-14B-Instruct-AWQ", max_hours=10.5, tp=2, chunk=4000):
    from vllm import LLM, SamplingParams
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "catalog_ko3.csv"
    cat = pd.read_csv(cat_path, encoding="utf-8-sig").drop_duplicates("vessel_name")
    done = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False) if path.exists() else \
        pd.DataFrame(columns=["vessel_name", "hanja", "meaning"])
    todo = cat[~cat["vessel_name"].isin(set(done["vessel_name"]))]
    print(f"{len(todo)} names to do, {len(done)} done", flush=True)
    llm = LLM(model=model, tensor_parallel_size=int(tp), dtype="half", max_model_len=1024,
              gpu_memory_utilization=0.90, enforce_eager=False)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=80)
    rows = done.to_dict("records")
    for s in range(0, len(todo), chunk):
        if time.time() - t0 > float(max_hours) * 3600:
            print("time limit: stopping (rerun to resume)", flush=True)
            break
        part = todo.iloc[s:s + chunk]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": GUIDE.format(name=r.vessel_name, flag=r.flag_name, vtype=r.vessel_type)}],
            tokenize=False, add_generation_prompt=True) for r in part.itertuples()]
        res = llm.generate(prompts, sp, use_tqdm=False)
        for r, o in zip(part.itertuples(), res):
            p = parse(o.outputs[0].text)
            rows.append({"vessel_name": r.vessel_name, "hanja": p.get("hanja", ""), "meaning": p.get("meaning", "")})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{len(rows)}/{len(cat)} names ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    print("filled:", {k: round(float((d[k] != "").mean()), 3) for k in ("hanja", "meaning")})


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if len(a) > 2: kw["model"] = a[2]
    if len(a) > 3: kw["max_hours"] = float(a[3])
    if len(a) > 4: kw["tp"] = int(a[4])
    main(a[0], a[1], **kw)
