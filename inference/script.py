"""DACON inference entry point: data/test.csv + data/vessel_catalog.csv -> output/submission.csv"""

from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
import vessel_lib  # noqa: E402
from vessel_lib import FEATURES, BoosterAverage, CatalogIndex, CondProb, Matcher, load_catalog_ko  # noqa: E402

CALLSIGN_MODE = "name_first"  # "off" | "guarded" | "hard" | "name_first"
CS_POOL = True     # rows sharing a call_sign are one vessel (1:1 in train): decide them together
CS_EXCLUDE = True  # a call_sign unseen in train cannot be a vessel train already ties to another call_sign
CS_HARD = True    # a call_sign seen in train -> that train vessel
CS_ONE2ONE = False # different unseen call_signs -> different vessels
POOL_K, POOL_T = 20, 1.0


class RowTypeSwitch:
    """Same 68 features. Hangul rows (feature 0 = is_hangul): ranker_* (teammate's newest model; the
    cross-encoder still re-scores Hangul rows). Other rows: eng_* (labse_syn rankers; the embedding
    features are -1 for non-Hangul rows in both models). Decided per row from that row's own features."""

    def __init__(self, hangul, other):
        self.hangul, self.other = hangul, other

    def predict(self, X):
        if self.other is not None and len(X) and X[0, 0] != 1.0:
            return self.other.predict(X)
        return self.hangul.predict(X)


def load_matcher(catalog: pd.DataFrame, model_dir: Path = HERE / "model") -> Matcher:
    art = json.loads((model_dir / "artifacts.json").read_text(encoding="utf-8"))
    vessel_lib.ACRONYMS = art.get("acronyms", True)  # must match training
    vessel_lib.SPELL_CAPS = art.get("spell_caps", False)
    vessel_lib.PERKEY_ALWAYS = art.get("perkey_always", False)
    vessel_lib.HANJA = art.get("hanja", False)
    if art["features"] != FEATURES:
        raise ValueError("Model and feature code versions do not match")
    # model_str: LightGBM's file reader cannot open non-ASCII Windows paths
    files = sorted(model_dir.glob("ranker*.txt"))
    if not files:
        raise FileNotFoundError("no ranker*.txt in model/")
    eng = sorted(model_dir.glob("eng_*.txt"))
    booster = RowTypeSwitch(BoosterAverage(lgb.Booster(model_str=p.read_text(encoding="utf-8")) for p in files),
                            BoosterAverage(lgb.Booster(model_str=p.read_text(encoding="utf-8")) for p in eng) if eng else None)
    ko = load_catalog_ko(model_dir / "catalog_ko.csv") if art.get("use_ko") else {}
    ko2 = load_catalog_ko(model_dir / "catalog_ko2.csv") if art.get("use_ko2") else {}
    if (art.get("use_ko") and not ko) or (art.get("use_ko2") and not ko2):
        raise FileNotFoundError("model was trained with catalog_ko*.csv but a file is missing")
    if art.get("emb"):  # LaBSE meaning features (model/emb)
        from emb_feat import Embedder
        vessel_lib.EMB = Embedder(model_dir / "emb")
    if (model_dir / "ce").is_dir():  # cross-encoder for Hangul rows (model/ce)
        from ce_feat import CrossScorer
        vessel_lib.CE = CrossScorer(model_dir / "ce")
    return Matcher(CatalogIndex(catalog, ko, ko2), art["callsign"], art["token_dict"],
                   CondProb(art["flag_p"]), CondProb(art["type_p"]), booster,
                   art.get("name_map"), art.get("callsign_name"))


def callsign_postprocess(rows, preds, details, callsign_map):
    """Train has one call_sign per vessel and one vessel per call_sign (2.zip on the leaderboard: the
    exclusion below helped, so test call_signs follow the same rule).
    CS_HARD: a row whose call_sign is in train gets that train vessel.
    CS_EXCLUDE: a row whose call_sign is not in train cannot be a vessel train ties to another call_sign.
    CS_POOL: rows sharing an unseen call_sign are decided together (summed per-row softmax).
    CS_ONE2ONE: two different unseen call_signs cannot get the same vessel; the more confident group
    keeps it and the other takes its best vessel not yet taken.
    Each rule is checked on English rows first and switched off only if they mostly disagree with it
    (test English is ~96% right, so the check only catches call_signs that do not line up with train)."""
    known = set(callsign_map.values())
    free = [(i, r, d) for i, (r, d) in enumerate(zip(rows, details)) if isinstance(r.get("call_sign"), str)]
    out = list(preds)

    seen = [(i, r, d) for i, r, d in free if r["call_sign"] in callsign_map]
    seen_eng = [(i, r, d) for i, r, d in seen if not d["hangul"]]
    clash = sum(preds[i] != callsign_map[r["call_sign"]] for i, r, d in seen_eng)
    hard = CS_HARD and clash <= 2 + 0.30 * len(seen_eng)
    print(f"call_sign check: {len(seen_eng)} English rows with a train call_sign, {clash} predicted elsewhere"
          f" -> hard lookup {'on' if hard else 'off'}")
    if hard:
        for i, r, d in seen:
            out[i] = callsign_map[r["call_sign"]]

    unseen = [(i, r, d) for i, r, d in free if r["call_sign"] not in callsign_map and not d["overridden"]]
    eng = [(i, r, d) for i, r, d in unseen if not d["hangul"]]
    conflicts = sum(d["top_ids"][0] in known for _, _, d in eng)
    exclude = CS_EXCLUDE and conflicts <= 2 + 0.30 * len(eng)
    print(f"call_sign check: {len(eng)} English rows with unseen call_sign, {conflicts} on train vessels"
          f" -> exclusion {'on' if exclude else 'off'}")
    probs, groups = {}, {}
    for i, r, d in unseen:
        ids, sc = d["top_ids"], np.asarray(d["top_scores"], dtype=float)
        if exclude:
            keep = [j for j, v in enumerate(ids) if v not in known]
            if keep:
                ids, sc = [ids[j] for j in keep], sc[keep]
        p = np.exp((sc - sc.max()) / POOL_T)
        probs[i] = dict(zip(ids, p / p.sum()))
        groups.setdefault(r["call_sign"], []).append(i)
    gscore = {}
    for cs, members in groups.items():
        tot: dict[str, float] = {}
        for i in members:
            for v, q in probs[i].items():
                tot[v] = tot.get(v, 0.0) + (q if CS_POOL else 0.0)
        for i in members:
            score = tot if CS_POOL else probs[i]
            out[i] = max(probs[i], key=lambda v: score[v])
        gscore[cs] = {v: q / len(members) for v, q in tot.items()} if CS_POOL else             {v: max(probs[i].get(v, 0.0) for i in members) for i0 in members[:1] for v in probs[i0]}

    if CS_ONE2ONE:
        claim: dict[str, list[str]] = {}
        for cs, members in groups.items():
            claim.setdefault(out[members[0]], []).append(cs)
        dup = {v: c for v, c in claim.items() if len(c) > 1}
        eng_dup = sum(1 for c in dup.values() if sum(not details[i]["hangul"] for cs in c for i in groups[cs]) > 1)
        on = eng_dup <= 2 + 0.30 * len(eng)
        print(f"call_sign check: {len(dup)} vessels claimed by several unseen call_signs ({eng_dup} between"
              f" English rows) -> one-to-one {'on' if on else 'off'}")
        if on and dup:
            taken = {v for v, c in claim.items() if len(c) == 1}
            order = sorted(((gscore[cs].get(v, 0.0), cs) for v, c in dup.items() for cs in c), reverse=True)
            moved = 0
            for conf, cs in order:
                members = groups[cs]
                for v in sorted(gscore[cs], key=gscore[cs].get, reverse=True):
                    if v not in taken and all(v in probs[i] for i in members[:1]):
                        taken.add(v)
                        moved += v != out[members[0]]
                        for i in members:
                            out[i] = v
                        break
            print(f"one-to-one: {moved} call_sign groups moved to their next free vessel")
    return out


def find_input(name: str) -> Path:
    """The guide mentions both data/ and open/; accept either, relative to cwd or this file."""
    for base in (Path.cwd(), HERE):
        for sub in ("data", "open", "data/open"):
            if (base / sub / name).is_file():
                return base / sub / name
    for base in (Path.cwd(), HERE):
        for sub in ("data", "open"):
            if (base / sub).is_dir():
                hits = sorted((base / sub).rglob(name))
                if hits:
                    return hits[0]
    raise FileNotFoundError(f"{name} not found under data/ or open/")


def run(test_path=None, catalog_path=None, output_path="output/submission.csv") -> Path:
    test = pd.read_csv(test_path or find_input("test.csv"), encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path or find_input("vessel_catalog.csv"), encoding="utf-8-sig")
    matcher = load_matcher(catalog)
    rows = test.to_dict("records")
    details: list[dict] = []
    preds = matcher.predict(rows, callsign_mode=CALLSIGN_MODE, details=details, top_k=POOL_K)
    preds = callsign_postprocess(rows, preds, details, matcher.callsign_map)
    if len(preds) != len(test) or not set(preds) <= set(matcher.idx.id_to_index):
        raise ValueError("Invalid catalog predictions")
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds})
    output_path = Path(output_path)
    targets = [output_path]
    if not output_path.is_absolute() and Path.cwd().resolve() != HERE:  # cwd-relative (baseline) + next to script
        targets.append(HERE / output_path)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {len(result)} predictions to {path}")
    return output_path


if __name__ == "__main__":
    run()
