"""Assemble submit_llm/ from the current E model (submit/model) + LLM pieces and zip it.

  submit_llm/
    script.py, requirements.txt
    model/ ranker.txt, artifacts.json, vessel_lib.py   <- copied from submit/model (E)
           llm_scorer.py                                <- copied from llm/
           llm_config.json                              <- written by combine_llm.py --write
           llm/                                         <- LLM weights folder (from llm_score.py --save)

usage: python build_llm_zip.py [--weights path/to/saved_weights] [--name submit_F_llm.zip]
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "submit_llm"
LIMIT_ZIP, LIMIT_UNZIPPED = 10 * 2**30, 32 * 2**30


def main(weights: str, name: str):
    model = PKG / "model"
    model.mkdir(parents=True, exist_ok=True)
    for f in ("ranker.txt", "artifacts.json", "vessel_lib.py"):
        shutil.copy2(ROOT.parent / "inference" / "model" / f, model / f)
    shutil.copy2(ROOT / "llm" / "llm_scorer.py", model / "llm_scorer.py")
    cfg_path = model / "llm_config.json"
    if not cfg_path.is_file():
        raise SystemExit("model/llm_config.json missing: run combine_llm.py <scores> --write first")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if weights:
        dst = model / cfg.get("llm_dir", "llm")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(weights, dst)
    llm_dir = model / cfg.get("llm_dir", "llm")
    if not (llm_dir / "config.json").is_file():
        print(f"WARNING: no LLM weights in {llm_dir}: the zip will behave exactly like E")
    else:
        qc = json.loads((llm_dir / "config.json").read_text(encoding="utf-8")).get("quantization_config")
        req = (PKG / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if qc and not any(line.strip().startswith("bitsandbytes") for line in req):
            raise SystemExit("weights are quantized (bitsandbytes) but requirements.txt does not install it")
    for p in model.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)

    out = ROOT / name
    total = 0
    with zipfile.ZipFile(out, "w", allowZip64=True) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                big = p.suffix in (".safetensors", ".bin", ".gguf")
                z.write(p, p.relative_to(PKG).as_posix(),
                        compress_type=zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED)
                total += p.stat().st_size
    size = out.stat().st_size
    print(f"{out.name}: {size / 2**20:.1f}MB zipped, {total / 2**20:.1f}MB unzipped")
    if size > LIMIT_ZIP or total > LIMIT_UNZIPPED:
        raise SystemExit("over the DACON size limit (10GB zip / 32GB unzipped)")
    with zipfile.ZipFile(out) as z:
        top = {n.split("/")[0] for n in z.namelist()}
    assert {"script.py", "requirements.txt", "model"} <= top and len(top) == 3, top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--name", default="submit_F_llm.zip")
    a = ap.parse_args()
    main(a.weights, a.name)
