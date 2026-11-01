"""LLM transliteration likelihood: log P(korean_record_name | english_catalog_name).

Adapted from the teammate's runpod_llm_score.ipynb. Used to re-rank the top-K candidates of
Hangul-name rows. Each row is scored in its own batch (only its own candidates), so a row's
result never depends on other test rows, their order, or batch composition.
"""

from __future__ import annotations

import unicodedata

FEWSHOT = [("Glovis Supreme", "글로비스 슈프림"), ("KMTC Seoul", "케이엠티씨 서울"), ("Grand Ace 9", "그랜드 에이스9"),
           ("Heung-A Bangkok", "흥아 방콕"), ("Morning Crown", "모닝 크라운"), ("STI Lombard", "에스티아이 롬바드"),
           ("Equinox Eagle", "에퀴녹스 이글"), ("Frida Knutsen", "프리다 크누센"), ("Hodaka Galaxy", "호다카 갤럭시"),
           ("Kashima Maru No. 10", "카시마 마루 10")]
SYSTEM = ("너는 선박명을 한국 항만 입출항 신고서에 적는 한글 표기로 바꾸는 전문가다. "
          "영어는 실제 발음대로 외래어 표기하고, 영문 약어는 알파벳 이름으로 읽고(KMTC→케이엠티씨), "
          "숫자는 그대로 둔다. 한글 표기 한 줄만 출력한다.\n예시:\n"
          + "\n".join(f"{a} => {b}" for a, b in FEWSHOT))


class LLMScorer:
    def __init__(self, path: str, quant4: bool = False, device: str = "cuda"):
        """path: HuggingFace id (needs internet) or a local folder saved with save()."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        kw = dict(device_map=device)
        if quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw)
        self.model.eval()

    def save(self, out_dir: str) -> None:
        """Write tokenizer + weights (safetensors) for offline loading on the evaluation server."""
        self.tok.save_pretrained(out_dir)
        self.model.save_pretrained(out_dir, safe_serialization=True)

    def _prompt(self, eng: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{eng} =>"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score(self, korean: str, candidates: list[str]) -> tuple[list[float], int]:
        """Sum of token log-probs of `korean` after each candidate's prompt; also #target tokens."""
        torch = self.torch
        target = unicodedata.normalize("NFC", korean).strip()
        tgt_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        seqs, starts = [], []
        for c in candidates:
            p_ids = self.tok(self._prompt(c), add_special_tokens=False)["input_ids"]
            starts.append(len(p_ids))
            seqs.append(p_ids + tgt_ids)
        L = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):  # right padding: positions of real tokens are fixed per sequence
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        ids, att = ids.to(self.model.device), att.to(self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        n = len(tgt_ids)
        for i, st in enumerate(starts):
            pos = torch.arange(st - 1, st - 1 + n, device=logp.device)
            tok = ids[i, st:st + n]
            out.append(float(logp[i, pos, tok].sum()))
        return out, n
