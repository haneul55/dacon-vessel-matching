"""Generate Korean spellings of catalog vessel names with a public pretrained LLM (offline).

Shared by catalog_ko_generate.ipynb (GPU) and a local CPU dry run. Input is only the official
vessel_catalog.csv; few-shot examples come from the official train.csv. Greedy decoding with a
fixed prompt, so the output file can be regenerated exactly for code verification.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

# (catalog English name, Korean record spelling) pairs from official train.csv
FEWSHOT = [
    ("KMTC Seoul", "케이엠티씨 서울"), ("JBU Sapphire", "제이비유 사파이어"), ("HC Wand", "에이치씨 완드"),
    ("Jangho Win", "장호 윈"), ("Keoyoung Master", "거영마스타"), ("Woohyun Sky", "우현 스카이"),
    ("Dongjin Fortune", "동진포춘"), ("Ning Hai Tuo 6001", "닝하이투오 6001"), ("You Shen 3", "유센 3"),
    ("Berge Asahidake", "버지 아사히다케"), ("Sawasdee Sunrise", "싸왓디 선라이즈"),
    ("Sea Admiral", "씨 어드마이럴"), ("Star Pioneer", "스타 파이오니아"), ("Front Discovery", "프런트 디스커버리"),
    ("Reflection", "리플렉션"), ("303 Hyodong Chemi", "303효동케미"), ("Sun Edelweiss", "썬 에델바이스"),
    ("Pacific Beijing", "퍼시픽 베이징"),
]
SYSTEM = (
    "너는 외국 선박명을 한국 항만 입출항 신고서에 적는 한글 표기로 옮기는 전문가다.\n"
    "규칙: 영어는 실제 발음대로 외래어 표기법에 맞게 적는다. 영문 약어는 알파벳 이름으로 읽는다(KMTC→케이엠티씨). "
    "한국·중국·일본·태국 등 고유명사는 원래 발음대로 적는다(Jangho→장호, Ning Hai→닝하이, Asahidake→아사히다케). "
    "숫자는 그대로 둔다. 설명 없이 한글 표기 한 줄만 출력한다.\n예시:\n"
    + "\n".join(f"{a} => {b}" for a, b in FEWSHOT)
)
_KO_JE = re.compile(r"제\s*(\d+)")


def ko_key(s: object) -> str:
    """Same key as vessel_lib.ko_key: spacing-free, vessel suffix 호 dropped."""
    if not isinstance(s, str):
        return ""
    s = _KO_JE.sub(r" \1 ", unicodedata.normalize("NFC", s))
    s = re.sub(r"(?<=\d)\s*호|호\s*$", " ", s)
    return re.sub(r"[^0-9가-힣]+", "", s)


def clean_output(text: str) -> str:
    """First line of the model output, Hangul / digits / spaces only."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.split("=>")[-1]
    line = re.sub(r"[^0-9가-힣 ]+", " ", unicodedata.normalize("NFC", line))
    return re.sub(r"\s+", " ", line).strip()


class Generator:
    def __init__(self, model_id: str, quant4: bool = False, max_new_tokens: int = 32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.model_id, self.quant4, self.max_new_tokens = torch, model_id, quant4, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto")
        if quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()

    def prompt(self, name: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{name} =>"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, names: list[str]) -> list[str]:
        enc = self.tok([self.prompt(n) for n in names], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      num_beams=1, pad_token_id=self.tok.pad_token_id)
        texts = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [clean_output(t) for t in texts]


def run(names: list[str], gen, batch: int, out_csv: Path, log_every: int = 20) -> pd.DataFrame:
    """Translate `names` in length-sorted batches, appending to out_csv; resumes if it exists."""
    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        done = dict(zip(prev["vessel_name"], prev["ko"]))
        print(f"resume: {len(done)} names already done")
    todo = sorted({n for n in names if n not in done}, key=len)
    t0 = time.time()
    first = not out_csv.exists()
    for b, s in enumerate(range(0, len(todo), batch)):
        chunk = todo[s:s + batch]
        ko = gen(chunk)
        pd.DataFrame({"vessel_name": chunk, "ko": ko}).to_csv(
            out_csv, mode="w" if first else "a", header=first, index=False, encoding="utf-8-sig")
        first = False
        if b % log_every == 0:
            n = s + len(chunk)
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {el / 60:.1f}min  ETA {el / n * (len(todo) - n) / 60:.1f}min  "
                  f"e.g. {chunk[0]!r} -> {ko[0]!r}", flush=True)
    res = pd.read_csv(out_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return res.drop_duplicates("vessel_name", keep="last")


def quality(pairs: pd.DataFrame, ko_map: dict) -> dict:
    """pairs: columns en (catalog name), kr (train Hangul record). Few-shot names are excluded."""
    from rapidfuzz import fuzz
    shots = {a for a, _ in FEWSHOT}
    p = pairs[~pairs["en"].isin(shots)].copy()
    p["gen"] = p["en"].map(ko_map).fillna("")
    p["exact"] = [ko_key(a) == ko_key(b) for a, b in zip(p["kr"], p["gen"])]
    p["sim"] = [fuzz.ratio(unicodedata.normalize("NFD", ko_key(a)), unicodedata.normalize("NFD", ko_key(b)))
                for a, b in zip(p["kr"], p["gen"])]
    return dict(n=len(p), exact=round(float(p["exact"].mean()), 4), jamo_sim=round(float(p["sim"].mean()), 2),
                worst=p.sort_values("sim")[["en", "kr", "gen"]].head(15).values.tolist())


def train_pairs(train: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    h = train[train["vessel_name"].str.contains("[가-힣]", na=False)].drop_duplicates("target_vessel_id")
    en = catalog.set_index("vessel_id")["vessel_name"]
    return pd.DataFrame({"en": h["target_vessel_id"].map(en).to_numpy(), "kr": h["vessel_name"].to_numpy()})


def write_meta(path: Path, gen, batch: int, n: int) -> None:
    path.write_text(json.dumps(dict(
        model_id=gen.model_id, quant4=gen.quant4, max_new_tokens=gen.max_new_tokens, batch=batch,
        decoding="greedy (do_sample=False, num_beams=1)", fewshot=FEWSHOT, system_prompt=SYSTEM,
        n_names=n, source="official vessel_catalog.csv vessel_name only",
    ), ensure_ascii=False, indent=2), encoding="utf-8")
