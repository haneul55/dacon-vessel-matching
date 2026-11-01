"""Run a submit zip exactly like the DACON server, on rows sampled from train.csv, and score it.

usage: python local_eval.py submit_B_guarded.zip [--n 3000]

The zip's model was trained on train.csv, so this score is optimistic (in-sample).
Use it to check that the zip runs and produces a valid submission; the leaderboard is the
real test. Honest out-of-sample numbers come from train.py (unseen-vessel accuracy).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def find_data(arg: str | None) -> Path:
    """Folder containing train.csv and vessel_catalog.csv (extracts ../open.zip if needed)."""
    for cand in ([Path(arg)] if arg else []) + [HERE.parent / "open", HERE / "open"]:
        if (cand / "train.csv").is_file():
            return cand
    archive = HERE.parent / "open.zip"
    if not archive.is_file():
        sys.exit("train.csv not found: pass --data <folder with train.csv and vessel_catalog.csv>")
    out = HERE.parent / "open"
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    print(f"extracted {archive.name} -> {out}")
    return out


def score(truth: pd.DataFrame, pred: pd.DataFrame) -> dict:
    m = truth.merge(pred, on="sample_id", suffixes=("", "_pred"))
    hang = m["vessel_name"].fillna("").str.contains("[가-힣]")
    ok = m["target_vessel_id"] == m["target_vessel_id_pred"]
    h, n = float(ok[hang].mean()), float(ok[~hang].mean())
    return {"rows": len(m), "hangul": round(h, 5), "non_hangul": round(n, 5),
            "weighted": round(0.7 * h + 0.3 * n, 5), "wrong": int((~ok).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", type=Path)
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=3000, help="rows sampled from train.csv (0 = all)")
    args = ap.parse_args()
    data = find_data(args.data)

    train = pd.read_csv(data / "train.csv", encoding="utf-8-sig")
    sample = train.sample(args.n, random_state=0) if 0 < args.n < len(train) else train
    work = Path(tempfile.mkdtemp(prefix="vm_eval_"))
    try:
        with zipfile.ZipFile(args.zip) as z:  # same layout the server builds
            z.extractall(work)
        (work / "data").mkdir()
        sample.drop(columns="target_vessel_id").to_csv(work / "data" / "test.csv", index=False,
                                                       encoding="utf-8-sig")
        shutil.copy(data / "vessel_catalog.csv", work / "data" / "vessel_catalog.csv")

        t0 = time.time()
        subprocess.run([sys.executable, "script.py"], cwd=work, check=True)
        pred = pd.read_csv(work / "output" / "submission.csv")
        assert list(pred.columns) == ["sample_id", "target_vessel_id"], pred.columns
        print(f"\n[{args.zip.name}] runtime {time.time() - t0:.0f}s")
        print(score(sample, pred))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
