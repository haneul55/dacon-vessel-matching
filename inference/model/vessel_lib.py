"""Row-independent vessel matching: normalization, retrieval, reranking features.

Only the official catalog and artifacts learned from train.csv are used. Each query
row is processed on its own; nothing depends on other evaluation rows.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

import hanja

# ---------------------------------------------------------------- normalization
_PREFIX = re.compile(r"^(M/?V|M/?T|MV|MT|SS|M\.V\.|M\.T\.)\s+")
_SUFFIX = re.compile(r"\s+(CO\.?|LTD\.?|INC\.?|CORP\.?|CO\.?,?\s*LTD\.?)$")
_NO = re.compile(r"\bNO\.?\s*(\d+)")
_HANGUL = re.compile(r"[가-힣]")


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and bool(_HANGUL.search(value))


def clean_text(value: object) -> str:
    """NFKC, upper case, strip M/V-style prefixes and company suffixes, keep alnum + space."""
    if not isinstance(value, str):
        return ""
    s = unicodedata.normalize("NFKC", value).upper().strip()
    s = s.replace("Ⅱ", " II").replace("Ⅲ", " III")
    s = _PREFIX.sub("", s)
    s = _SUFFIX.sub("", s)
    s = _NO.sub(r" \1", s)  # "NO. 7" -> "7"
    s = re.sub(r"[^0-9A-Z가-힣]+", " ", s)
    toks = s.split()
    if len(toks) > 1 and toks[-1] in _ROMAN:  # "CY INTEROCEAN II" -> "CY INTEROCEAN 2"
        toks[-1] = _ROMAN[toks[-1]]
    return " ".join(toks)


_ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7", "VIII": "8",
          "IX": "9", "X": "10"}
_KO_NUM = {"원": "1", "투": "2", "쓰리": "3", "포": "4", "파이브": "5", "식스": "6", "세븐": "7",
           "에잇": "8", "나인": "9", "텐": "10"}


def compact(s: str) -> str:
    return s.replace(" ", "")


def alpha(s: str) -> str:
    """Name key without spaces and digits (numbers are compared separately)."""
    return re.sub(r"[\s0-9]", "", s)


def sound_key(pkey: str) -> str:
    """Spacing-free phonetic key. Repeated letters are merged *after* spaces are removed, so
    '모닝 캐서린' and '모닝캐서린' give the same key (merging per word first left 'monink kisorin'
    vs 'moninkisorin', and the unspaced record lost to Morning Celine)."""
    return re.sub(r"([a-z])\1+", r"\1", alpha(pkey))


def digits_of(s: str) -> str:
    return " ".join(re.findall(r"\d+", s))


def clean_hangul(value: object) -> str:
    """Korean ship-name conventions: '제7하나호' -> '7 하나'.
    Only the vessel suffix 호 is dropped (end of name, or after a number). A 호 before a space is
    part of the name: '장호 윈호' -> '장호 윈' (Jangho Win), '대호 써니' (Daeho Sunny)."""
    s = clean_text(value)
    s = re.sub(r"제\s*(\d+)\s*", r" \1 ", s)
    s = re.sub(r"(\d+)\s*호\b", r"\1", s)
    s = re.sub(r"호$", "", s.strip())
    toks = s.split()
    if len(toks) > 1:  # "씨와이 인터오션 투" -> "... 2"
        toks = [toks[0]] + [_KO_NUM.get(t, t) for t in toks[1:]]
    return " ".join(toks)


# ------------------------------------------------------- Hangul -> Latin phonetics
_L = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_V = ["a", "e", "ya", "ye", "o", "e", "yo", "ye", "o", "wa", "we", "we", "yo", "u", "wo", "we", "wi",
      "yu", "", "i", "i"]  # loanword-oriented: ㅓ->o, ㅡ->(epenthetic, dropped)
_T = ["", "k", "k", "k", "n", "n", "n", "t", "l", "l", "m", "l", "l", "l", "l", "l", "m", "p", "p",
      "t", "t", "ng", "t", "t", "k", "t", "p", "t"]


def romanize(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_L[code // 588] + _V[(code % 588) // 28] + _T[code % 28])
        else:
            out.append(ch.lower())
    return "".join(out)


_EN_RULES = [
    (r"eo", "o"), (r"(?<=\w)eu", ""), (r"ae", "e"),  # Korean romanization: Seongho, Heung-A, Daeho
    (r"you", "yo"),  # Young / Keoyoung -> 영 (before ou -> au, which made it 야웅)
    (r"ce(?=a)", "sh"), (r"(?<=\w)[ts]i(?=on\b)", "sh"),  # Ocean -> 오션, -tion/-sion -> 션
    (r"ph", "f"), (r"th", "s"), (r"sh", "s"), (r"ck", "k"), (r"tch", "j"), (r"ch", "j"),
    (r"ts(?=u)", "j"), (r"g(?=[eiy])", "j"), (r"qu", "kw"), (r"\bx", "s"), (r"x", "ks"), (r"iu", "yu"),
    # silent gh (night, high), but not after n: Korean ng+h as in Jangho, Seongho, Donghae
    (r"c(?=[eiy])", "s"), (r"c", "k"), (r"(?<!n)gh", ""), (r"wh", "w"), (r"ee", "i"), (r"oo", "u"),
    (r"a([^aeiou\s])e\b", r"ei\1"),  # magic e: Ace -> 에이스, Jade -> 제이드, Lake -> 레이크
    (r"(?<=\w\w[^aeiou\s])e\b", ""),  # silent final e: Rose -> 로즈
    (r"ou", "au"), (r"y\b", "i"), (r"er\b", "o"), (r"ar\b", "a"), (r"or\b", "o"), (r"r\b", "o"),
    (r"(?<=[aeiou])r(?=[^aeiouy\s])", ""),  # non-rhotic: Mars -> 마스, Morning -> 모닝
    (r"(?<=[a-z])h(?![aeiouy])", ""),  # silent h: Chih -> 치
]
_EN_RULES = [(re.compile(a), b) for a, b in _EN_RULES]
_CLASS = str.maketrans({"b": "p", "f": "p", "v": "p", "d": "t", "g": "k", "c": "k", "q": "k",
                        "z": "j", "l": "r", "e": "i", "u": "o", "w": "o", "y": "i"})


_LETTER = dict(zip("abcdefghijklmnopqrstuvwxyz", [
    "ei", "bi", "si", "di", "i", "ep", "ji", "eiji", "ai", "jei", "kei", "el", "em", "en", "o", "pi",
    "kyu", "al", "es", "ti", "yu", "bui", "dabryu", "eks", "wai", "jet"]))


def _spell_acronyms(s: str) -> str:
    """"cnc dream" -> "si en si dream" (Korean reads acronyms letter by letter: 시앤시)."""
    # vowel-less tokens only: CNC, SM, KMTC, TY are spelled out; FU, YI, DE (pinyin) and SKY are words
    return " ".join(" ".join(_LETTER[ch] for ch in t) if t.isalpha() and len(t) <= 5
                    and not re.search(r"[aeiou]", t) and (len(t) <= 2 or "y" not in t) else t
                    for t in s.split())


SPELL_CAPS = True  # set from artifacts.json ("spell_caps") so training and inference agree
PERKEY_ALWAYS = False  # artifacts "perkey_always": per-key retrieval also with tonnage (version G)
HANJA = False  # artifacts "hanja": Sino-Korean readings of pinyin names (금성 = Jin Cheng), 환 = Maru


def spell_caps(raw: object, clean: str) -> str:
    """For Hangul comparison only: catalog tokens written in capitals (STI Sloane, CMA CGM, SITC)
    are read letter by letter, as Korean records do (에스티아이 슬론). Roman numerals and
    word-like tokens (ONE, ZIM) are kept; all-caps names give no casing signal and are skipped."""
    raw = unicodedata.normalize("NFKC", raw) if isinstance(raw, str) else ""
    if not SPELL_CAPS or not re.search(r"[a-z]", raw):
        return clean
    caps = {t for t in re.findall(r"\b[A-Z]{2,4}\b", raw)
            if t not in _ROMAN and not re.fullmatch(r"[^AEIOU]?[AEIOU][^AEIOU]E?", t)}
    if not caps:
        return clean
    return " ".join(" ".join(_LETTER[c.lower()] for c in t) if t in caps else t for t in clean.split())


def phonetic(latin: str, english: bool) -> str:
    """Map a latin string (English or romanized Korean) to a shared coarse sound key."""
    s = latin.lower()
    if english:
        s = _spell_acronyms(s)
        for pat, rep in _EN_RULES:
            s = pat.sub(rep, s)
    s = s.replace("ch", "j").replace("sy", "s")
    s = s.translate(_CLASS)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"([a-z])\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def skeleton(key: str) -> str:
    return re.sub(r"[aeiouh ]", "", key)


def vowel_key(key: str) -> str:
    """Consonants + vowel positions only: one English vowel maps to many Korean ones (Grand/그랜드)."""
    return re.sub(r"a+", "a", re.sub(r"[aeiou]", "a", key))


# ----------------------------------------------------- learned token dictionary
def learn_token_dict(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Hangul token -> English token from aligned (hangul_name, catalog_name) training pairs."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for ko, en in pairs:
        kt, et = clean_hangul(ko).split(), clean_text(en).split()
        if len(kt) == len(et):
            for a, b in zip(kt, et):
                if _HANGUL.search(a):
                    votes[a][b] += 1
        elif len(kt) == 1 and _HANGUL.search(kt[0]):
            votes[kt[0]][compact(" ".join(et))] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


# Korean readings of Latin letter names (from the teammate's model): 케이엠티씨 -> KMTC
_LETTER_KO = {
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E", "에프": "F", "지": "G",
    "에이치": "H", "에치": "H", "아이": "I", "제이": "J", "케이": "K", "엘": "L", "엠": "M", "엔": "N",
    "오": "O", "피": "P", "큐": "Q", "알": "R", "에스": "S", "티": "T", "유": "U", "브이": "V",
    "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z", "지드": "Z", "앤": "N", "엔드": "N",
}
_LETTER_KO_SORTED = sorted(_LETTER_KO, key=len, reverse=True)
ACRONYMS = True  # set from artifacts.json ("acronyms") so training and inference always agree


def hangul_acronym(tok: str) -> str | None:
    """'케이에스' -> 'KS', '제이비유' -> 'JBU'; None unless the whole token is 2-5 letter names
    (a lone 씨 / 이 stays a word: 씨 울프 = Sea Wolf)."""
    out, i = [], 0
    while i < len(tok):
        for k in _LETTER_KO_SORTED:
            if tok.startswith(k, i):
                out.append(_LETTER_KO[k])
                i += len(k)
                break
        else:
            return None
    return "".join(out) if 2 <= len(out) <= 5 else None


def translate_hangul(name: str, token_dict: dict[str, str], max_len: int = 8) -> tuple[str, float]:
    """Greedy longest-match segmentation with the token dict; unknown chunks are romanized.
    A token read entirely as letter names becomes that acronym (케이엠티씨 -> KMTC).
    Returns (latin string, fraction of hangul chars covered by the dictionary)."""
    out, covered, total = [], 0, 0
    for tok in clean_hangul(name).split():
        if not _HANGUL.search(tok):
            out.append(tok)
            continue
        acr = hangul_acronym(tok) if ACRONYMS and tok not in token_dict else None
        if acr:
            out.append(acr)
            covered += len(tok); total += len(tok)
            continue
        i, buf = 0, ""
        while i < len(tok):
            hit = None
            for j in range(min(len(tok), i + max_len), i, -1):
                if tok[i:j] in token_dict:
                    hit = j
                    break
            if hit:
                if buf:
                    out.append(romanize(buf).upper()); buf = ""
                out.append(token_dict[tok[i:hit]])
                covered += hit - i; total += hit - i; i = hit
            else:
                buf += tok[i]; total += 1; i += 1
        if buf:
            out.append(romanize(buf).upper())
    return " ".join(out), covered / max(total, 1)


# ------------------------------------------- LLM Korean spellings of catalog names
# model/catalog_ko.csv (catalog vessel_name -> Korean spelling) is generated once, offline, from the
# official vessel_catalog.csv with a public pretrained LLM (teammate's file). Hangul records are then
# also compared Hangul-to-Hangul, at the jamo level, instead of only through romanization.
_KO_JE = re.compile(r"제\s*(\d+)")


def ko_key(s: object) -> str:
    """Spacing-free Hangul key: '제2흥아 파이오니어호' -> '2흥아파이오니어', '장호 윈호' -> '장호윈'."""
    if not isinstance(s, str):
        return ""
    s = _KO_JE.sub(r" \1 ", unicodedata.normalize("NFC", s))
    s = re.sub(r"(?<=\d)\s*호|호\s*$", " ", s)  # suffix only (end / after a number); 장호 윈 keeps 장호
    return re.sub(r"[^0-9가-힣]+", "", s)


def ko_tokens(s: object) -> str:
    if not isinstance(s, str):
        return ""
    s = _KO_JE.sub(r" \1 ", unicodedata.normalize("NFC", s))
    s = re.sub(r"(?<=\d)\s*호|호\s*$", " ", s)  # suffix only (end / after a number); 장호 윈 keeps 장호
    s = re.sub(r"[^0-9가-힣]+", " ", s)
    return " ".join(re.sub(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", " ", s).split())


def jamo(s: str) -> str:
    """Conjoining jamo, so '쿠리어' vs '커리어' differ by one symbol."""
    return unicodedata.normalize("NFD", s)


def load_catalog_ko(path) -> dict[str, str]:
    """{catalog vessel_name: Korean spelling}; empty if the file is absent."""
    try:
        d = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except FileNotFoundError:
        return {}
    return {a: b for a, b in zip(d["vessel_name"], d["ko"]) if b.strip()}


# ------------------------------------------------------------------ catalog index
def num(value) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) and x > 0 else np.nan


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame, ko: dict[str, str] | None = None,
                 ko2: dict[str, str] | None = None):
        """ko: teammate's LLM Korean spellings; ko2: a second, independently generated set
        (Qwen2.5-7B, llm/catalog_ko_generate.ipynb). Each is right where the other is wrong:
        exact match with train records 31.9% / 26.1% alone, 37.4% either."""
        cat = catalog.reset_index(drop=True)
        ko, ko2 = ko or {}, ko2 or {}
        names = cat["vessel_name"].astype(str).tolist()
        self.ko_k = [ko_key(ko.get(n, "")) for n in names]
        self.ko_j = [jamo(k) for k in self.ko_k]
        self.ko_tok = [ko_tokens(ko.get(n, "")) for n in names]
        self.has_ko = any(self.ko_k)
        self.ko2_k = [ko_key(ko2.get(n, "")) for n in names]
        self.ko2_j = [jamo(k) for k in self.ko2_k]
        self.ko2_tok = [ko_tokens(ko2.get(n, "")) for n in names]
        self.has_ko2 = any(self.ko2_k)
        # Sino-Korean readings of pinyin names: Jin Cheng -> 금성, 진성 ... (empty unless HANJA)
        self.hj_keys = [hanja.catalog_keys(n) if HANJA else [] for n in names]
        self.hj_jamo = [[jamo(k) for k in ks] for ks in self.hj_keys]
        self.hj_index = defaultdict(list)
        for i, ks in enumerate(self.hj_keys):
            for k in ks:
                self.hj_index[k].append(i)
        self.hj_flat_j = [j for js in self.hj_jamo for j in js]
        self.hj_flat_i = np.asarray([i for i, js in enumerate(self.hj_jamo) for _ in js], dtype=int)
        self.maru = np.array([bool(re.search(r'\bmaru\b', n.lower())) for n in names])
        # syllable alignment: every reading combination, indexed by (count, position, reading) for
        # the first two syllables so a record with one wrong syllable is still found
        self.hj_syl = [hanja.catalog_syllables(n) if HANJA else None for n in names]
        self.hj_pos = defaultdict(list)
        for i, sy in enumerate(self.hj_syl):
            if sy is not None:
                for pos in (0, 1):
                    for r in sy[0][pos]:
                        self.hj_pos[(len(sy[0]), pos, r)].append(i)
        # English words written by meaning (Ocean Star -> 해양성), the rest by sound
        self.mn_keys = [hanja.meaning_keys(hanja.meaning_words(n, ko.get(n, ""))) if HANJA else []
                        for n in names]
        self.mn_jamo = [[jamo(k) for k in ks] for ks in self.mn_keys]
        self.mn_index = defaultdict(list)
        for i, ks in enumerate(self.mn_keys):
            for k in ks:
                self.mn_index[k].append(i)
        self.mn_flat_j = [j for js in self.mn_jamo for j in js]
        self.mn_flat_i = np.asarray([i for i, js in enumerate(self.mn_jamo) for _ in js], dtype=int)
        self.ids = cat["vessel_id"].astype(str).to_numpy()
        self.names = cat["vessel_name"].astype(str).to_numpy()  # raw names, for the cross-encoder
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.clean = [clean_text(v) for v in cat["vessel_name"]]
        self.comp = [compact(s) for s in self.clean]
        self.digits = [digits_of(s) for s in self.clean]
        self.acomp = [alpha(s) for s in self.clean]
        self.tokens = [frozenset(t for t in s.split() if not t.isdigit()) for s in self.clean]
        self.pkey = [phonetic(s, english=True) for s in self.clean]
        self.pcomp = [sound_key(k) for k in self.pkey]
        self.skel = [skeleton(k) for k in self.pcomp]
        self.vkey = [vowel_key(k) for k in self.pcomp]
        # keys used against Hangul records: capital acronyms spelled out (STI -> es ti ai)
        spelled = [spell_caps(r, c) for r, c in zip(cat["vessel_name"], self.clean)]
        self.pkey_h = [phonetic(s, english=True) if s != c else k for s, c, k in zip(spelled, self.clean, self.pkey)]
        self.pcomp_h = [sound_key(k) for k in self.pkey_h]
        self.skel_h = [skeleton(k) for k in self.pcomp_h]
        self.vkey_h = [vowel_key(k) for k in self.pcomp_h]
        self.gt = pd.to_numeric(cat["gross_tonnage"], errors="coerce").to_numpy(float)
        df = Counter(t for ts in self.tokens for t in ts)
        self.idf = {t: float(np.log(len(self.tokens) / n)) for t, n in df.items()}
        self.tok_post = defaultdict(list)
        for i, ts in enumerate(self.tokens):
            for t in ts:
                self.tok_post[t].append(i)
        self.gt_same = defaultdict(list)
        for i, g in enumerate(self.gt):
            if np.isfinite(g):
                self.gt_same[float(g)].append(i)
        self.flag = cat["flag_name"].fillna("").astype(str).to_numpy()
        self.vtype = cat["vessel_type"].fillna("").astype(str).to_numpy()
        self.exact = defaultdict(list)
        for i, c in enumerate(self.comp):
            self.exact[c].append(i)
        order = np.argsort(np.nan_to_num(self.gt, nan=-1.0), kind="mergesort")
        self.gt_order, self.gt_sorted = order, np.nan_to_num(self.gt[order], nan=-1.0)

    def gt_window(self, g: float, tol: float = 0.055) -> np.ndarray:
        if not np.isfinite(g):
            return np.empty(0, dtype=int)
        lo = np.searchsorted(self.gt_sorted, g * (1 - tol), "left")
        hi = np.searchsorted(self.gt_sorted, g * (1 + tol), "right")
        return self.gt_order[lo:hi]


# ----------------------------------------------------------------- query parsing
class Query:
    __slots__ = ("hangul", "clean", "comp", "acomp", "tokens", "digits", "pkey", "pcomp", "skel",
                 "vkey", "g", "ig", "flag", "vtype", "dict_cov", "raw_comp", "ko_k", "ko_j", "ko_tok", "maru_j")

    def __init__(self, row: dict, token_dict: dict[str, str]):
        name = row.get("vessel_name")
        self.hangul = is_hangul(name)
        self.ko_k = ko_key(name) if self.hangul else ""
        self.ko_j = jamo(self.ko_k)
        self.ko_tok = ko_tokens(name) if self.hangul else ""
        self.maru_j = jamo(hanja.maru_key(self.ko_k)) if HANJA and self.hangul else ""  # 가시마환 -> 가시마마루
        if self.hangul:
            latin, self.dict_cov = translate_hangul(name, token_dict)
            self.clean = clean_text(latin)
            # pure phonetics; dictionary-translated spelling is compared separately via acomp
            self.pkey = phonetic(romanize(clean_hangul(name)), english=False)
        else:
            self.dict_cov = 0.0
            self.clean = clean_text(name)
            self.pkey = phonetic(self.clean, english=True)
        self.raw_comp = compact(clean_hangul(name)) if self.hangul else ""
        self.comp = compact(self.clean)
        self.acomp = alpha(self.clean)
        self.tokens = frozenset(t for t in self.clean.split() if not t.isdigit())
        self.digits = digits_of(self.clean)
        self.pcomp = sound_key(self.pkey)
        self.skel = skeleton(self.pcomp)
        self.vkey = vowel_key(self.pcomp)
        self.g, self.ig = num(row.get("gross_tonnage")), num(row.get("international_gross_tonnage"))
        self.flag = row.get("flag_name") if isinstance(row.get("flag_name"), str) else ""
        self.vtype = row.get("vessel_type") if isinstance(row.get("vessel_type"), str) else ""


# --------------------------------------------------------------------- retrieval
def _cdist(queries: list[str], choices: list[str], scorer) -> np.ndarray:
    return process.cdist(queries, choices, scorer=scorer, dtype=np.uint8, workers=-1)


def name_sims(qs: list[Query], idx: CatalogIndex) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Name similarity to the whole catalog. Batched only for speed: row i depends on qs[i] alone.
    Hangul -> phonetic keys (+ dictionary translation); latin -> spelling + token order.
    Returns (max over keys, per-row list of per-key similarity vectors)."""
    sim = np.zeros((len(qs), len(idx.ids)), dtype=np.uint8)
    per_key: list[list[np.ndarray]] = [[] for _ in qs]
    h = [i for i, q in enumerate(qs) if q.hangul]
    n = [i for i, q in enumerate(qs) if not q.hangul]
    if h:
        keys = [_cdist([qs[i].pcomp for i in h], idx.pcomp_h, fuzz.ratio),
                _cdist([qs[i].vkey for i in h], idx.vkey_h, fuzz.ratio),
                _cdist([qs[i].skel for i in h], idx.skel_h, fuzz.ratio)]
        hd = [j for j, i in enumerate(h) if qs[i].dict_cov > 0]
        acomp = np.zeros_like(keys[0])
        if hd:
            acomp[hd] = _cdist([qs[h[j]].acomp for j in hd], idx.acomp, fuzz.ratio)
        merged = [keys[0], keys[1], acomp]
        if idx.has_ko:  # Hangul-to-Hangul against the LLM Korean spellings, jamo level
            keys.append(_cdist([qs[i].ko_j for i in h], idx.ko_j, fuzz.ratio))
            merged.append(keys[-1])
        if idx.has_ko2:  # per-key only: with tonnage present, extra look-alikes cost leaderboard rows
            keys.append(_cdist([qs[i].ko_j for i in h], idx.ko2_j, fuzz.ratio))
        sim[h] = np.maximum.reduce(merged)
        for j, i in enumerate(h):
            per_key[i] = [k[j] for k in keys] + ([acomp[j]] if qs[i].dict_cov > 0 else [])
    if n:
        keys = [_cdist([qs[i].comp for i in n], idx.comp, fuzz.ratio),
                _cdist([qs[i].clean for i in n], idx.clean, fuzz.token_sort_ratio)]
        sim[n] = np.maximum(*keys)
        for j, i in enumerate(n):
            per_key[i] = [k[j] for k in keys]
    return sim, per_key


def align_hits(q: Query, idx: CatalogIndex, limit: int = 15) -> np.ndarray:
    """Pinyin names whose reading options match the record syllable by syllable (at most one miss)."""
    hs = re.sub(r"\d", "", q.ko_k)
    if len(hs) < 2:
        return np.zeros(0, dtype=int)
    pool = set(idx.hj_pos.get((len(hs), 0, hs[0]), ())) | set(idx.hj_pos.get((len(hs), 1, hs[1]), ()))
    qd = "".join(re.findall(r"\d+", q.ko_k))
    scored = []
    for i in pool:
        a = hanja.align(q.ko_k, idx.hj_syl[i])
        if a is not None and a[0] >= len(hs) - 1:
            scored.append((a[0] + 0.5 * (idx.hj_syl[i][1] == qd), i))
    scored.sort(reverse=True)
    return np.asarray([i for _, i in scored[:limit]], dtype=int)


def retrieve(q: Query, sim: np.ndarray, idx: CatalogIndex, key_sims: list[np.ndarray] = (),
             top_name: int = 40, top_gt: int = 30) -> np.ndarray:
    """Stage 1: union of name Top-N and best names inside the ±5.5% GT window.
    Per-key Top-N (a vessel that only one key ranks high) is added only when the record has no
    tonnage: there it lifts Hangul recall@40 98.0% -> 100%, but with tonnage present the true vessel
    is already retrieved and the extra look-alikes cost Hangul rows on the leaderboard
    (D 0.98789 -> E1 0.98322 with per-key always on)."""
    sim = sim.astype(np.int16)
    picks = [np.argpartition(-sim, top_name)[:top_name]]
    if HANJA and q.hangul and q.ko_k:  # a record written by Sino-Korean reading (금성) or 환 (Maru)
        picks.append(np.asarray(idx.hj_index.get(q.ko_k, []), dtype=int))
        if idx.hj_flat_j:
            hits = process.extract(q.ko_j, idx.hj_flat_j, scorer=fuzz.ratio, limit=10)
            picks.append(idx.hj_flat_i[[h[2] for h in hits]])
        if q.maru_j:
            hits = process.extract(q.maru_j, idx.ko_j, scorer=fuzz.ratio, limit=10)
            picks.append(np.asarray([h[2] for h in hits], dtype=int))
        picks.append(align_hits(q, idx))
        picks.append(np.asarray(idx.mn_index.get(q.ko_k, []), dtype=int))
        if idx.mn_flat_j:
            hits = process.extract(q.ko_j, idx.mn_flat_j, scorer=fuzz.ratio, limit=10)
            picks.append(idx.mn_flat_i[[h[2] for h in hits]])
    if PERKEY_ALWAYS or not (np.isfinite(q.g) or np.isfinite(q.ig)):
        for ks in key_sims:
            picks.append(np.argpartition(-ks.astype(np.int16), top_name)[:top_name])
    win = np.union1d(idx.gt_window(q.g), idx.gt_window(q.ig))
    if len(win):
        k = min(top_gt, len(win))
        picks.append(win[np.argpartition(-sim[win], k - 1)[:k]])
    if len(win) and not q.hangul:  # renamed ships often keep a rare word: HAFNIA WREN -> BW Wren
        for t in q.tokens:
            post = idx.tok_post.get(t, ())
            if 0 < len(post) <= 2000:
                picks.append(np.intersect1d(np.asarray(post, dtype=int), win)[:top_gt])
    for g in {q.g, q.ig}:  # same-GT vessels even when renamed (name unrelated)
        same = np.asarray(idx.gt_same.get(float(g), []) if np.isfinite(g) else [], dtype=int)
        if len(same):
            k = min(top_gt, len(same))
            picks.append(same[np.argpartition(-sim[same], k - 1)[:k]])
    picks.append(np.asarray(idx.exact.get(q.comp, []), dtype=int))
    return np.unique(np.concatenate(picks)).astype(int)


# ---------------------------------------------------------------------- features
FEATURES = [
    "is_hangul", "dict_cov",
    "ratio", "partial", "tsort", "tset", "jw", "lev_norm", "exact", "prefix",
    "p_ratio", "p_jw", "p_partial", "skel_ratio", "len_ratio", "digits_eq", "digits_any",
    "gap_g", "gap_ig", "gap_min", "gt_exact", "gt_in5", "q_gt_missing", "c_gt_missing", "log_gt",
    "flag_p", "type_p", "flag_missing",
    "a_ratio", "a_jw", "tok_jacc", "tok_shared", "last_tok_eq", "n_same_gt", "idf_shared", "idf_max",
    "v_ratio", "v_jw", "digits_conflict",
    "ko_ratio", "ko_jw", "ko_tset", "ko_exact", "ko_missing",
    "ko2_ratio", "ko2_jw", "ko2_tset", "ko2_exact", "ko2_missing", "kob_ratio", "kob_exact",
    "hj_exact", "hj_ratio", "maru_ratio", "is_pinyin", "mn_exact", "mn_ratio",
    "r_ratio", "r_p_ratio", "d_ratio", "d_p_ratio", "d_gap", "n_cand", "d_ko", "d_kob", "d_hj",
    "emb_sim", "emb_rel",  # LaBSE meaning similarity, appended by Matcher.candidates (-1 without EMB)
]
EMB = None  # emb_feat.Embedder, set by script.py / train.py; None -> no embedding retrieval or features
EMB_TOP, EMB_TOL = 50, 0.10  # K3: 30 -> 50 (LB: K 10 -> K2 30 gained +0.0038)
CE = None  # ce_feat.CrossScorer, set by script.py; None -> ranker alone
CE_TOP, CE_W = 5, 0.25  # K3c: Hangul rows, ranker top-5 += 0.25 * cross-encoder (ce/ce_blend.py, out-of-fold)

# +1: more is better, -1: more is worse, 0: free. Keeps the ranker from trusting noise.
MONOTONE = {"ratio": 1, "partial": 1, "tsort": 1, "tset": 1, "jw": 1, "lev_norm": 1, "exact": 1,
            "prefix": 1, "p_ratio": 1, "p_jw": 1, "p_partial": 1, "skel_ratio": 1, "digits_eq": 1,
            "gap_g": -1, "gap_ig": -1, "gap_min": -1, "gt_exact": 1, "gt_in5": 1, "flag_p": 1,
            "type_p": 1, "a_ratio": 1, "a_jw": 1, "tok_jacc": 1, "tok_shared": 1, "last_tok_eq": 1, "idf_shared": 1, "idf_max": 1, "v_ratio": 1, "v_jw": 1, "digits_conflict": -1,
            "ko_ratio": 1, "ko_jw": 1, "ko_tset": 1, "ko_exact": 1, "d_ko": 1,
            "ko2_ratio": 1, "ko2_jw": 1, "ko2_tset": 1, "ko2_exact": 1, "kob_ratio": 1, "kob_exact": 1,
            "d_kob": 1, "hj_exact": 1, "hj_ratio": 1, "maru_ratio": 1, "d_hj": 1, "mn_exact": 1, "mn_ratio": 1,
            "r_ratio": -1, "r_p_ratio": -1, "d_ratio": 1, "d_p_ratio": 1, "d_gap": -1}


def _gap(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return 1.0
    return min(abs(a - b) / max(a, b), 1.0)


def _prefix(a: str, b: str) -> float:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b), 1)


class CondProb:
    """Smoothed P(catalog value | query value) learned from train labels."""

    def __init__(self, table: dict[str, dict[str, float]] | None = None):
        self.table = table or {}

    @classmethod
    def fit(cls, q_vals, c_vals, alpha: float = 1.0):
        cnt: dict[str, Counter] = defaultdict(Counter)
        for a, b in zip(q_vals, c_vals):
            cnt[a if isinstance(a, str) else ""][b] += 1
        table = {}
        for a, c in cnt.items():
            tot = sum(c.values()) + alpha * (len(c) + 1)
            table[a] = {b: (v + alpha) / tot for b, v in c.items()} | {"__other__": alpha / tot}
        return cls(table)

    def __call__(self, a: str, b: str) -> float:
        t = self.table.get(a)
        if t is None:
            return -1.0
        return t.get(b, t["__other__"])


def pair_features(q: Query, idx: CatalogIndex, cands: np.ndarray, flag_p: CondProb,
                  type_p: CondProb) -> np.ndarray:
    rows = []
    qlast = next((t for t in reversed(q.clean.split()) if not t.isdigit()), "")
    q_idf = max(sum(idx.idf.get(t, 12.0) for t in q.tokens), 1e-6)
    n_same = float(max(len(idx.gt_same.get(float(g), [])) if np.isfinite(g) else 0 for g in (q.g, q.ig)))
    use_ko = q.hangul and idx.has_ko  # otherwise the Korean features are a constant -1 for the row
    use_ko2 = q.hangul and idx.has_ko2
    use_hj = q.hangul and HANJA
    for i in cands:
        ko = ([fuzz.ratio(q.ko_j, idx.ko_j[i]), JaroWinkler.similarity(q.ko_j, idx.ko_j[i]),
               fuzz.token_set_ratio(q.ko_tok, idx.ko_tok[i]), float(bool(q.ko_k) and q.ko_k == idx.ko_k[i]),
               float(not idx.ko_k[i])] if use_ko else [-1.0] * 5)
        ko2 = ([fuzz.ratio(q.ko_j, idx.ko2_j[i]), JaroWinkler.similarity(q.ko_j, idx.ko2_j[i]),
                fuzz.token_set_ratio(q.ko_tok, idx.ko2_tok[i]), float(bool(q.ko_k) and q.ko_k == idx.ko2_k[i]),
                float(not idx.ko2_k[i])] if use_ko2 else [-1.0] * 5)
        kob = [max(ko[0], ko2[0]), max(ko[3], ko2[3])]  # the closer of the two spellings
        hj = ([float(q.ko_k in idx.hj_keys[i]), max((fuzz.ratio(q.ko_j, k) for k in idx.hj_jamo[i]), default=0.0),
               fuzz.ratio(q.maru_j, idx.ko_j[i]) if (q.maru_j and idx.maru[i]) else -1.0,
               float(bool(idx.hj_keys[i]))] if use_hj else [-1.0] * 4)
        al = hanja.align(q.ko_k, idx.hj_syl[i]) if use_hj else None
        if al is not None:  # alignment covers reading combinations the key list leaves out
            hj[0] = max(hj[0], float(al[1] == q.ko_k))
            hj[1] = max(hj[1], fuzz.ratio(q.ko_j, jamo(al[1])))
        # own features, not folded into hj_*: folding new signals in cost LB (0.91538 -> 0.91132)
        mn = ([float(q.ko_k in idx.mn_keys[i]), max((fuzz.ratio(q.ko_j, k) for k in idx.mn_jamo[i]), default=0.0)]
              if use_hj else [-1.0, -1.0])
        cp, ct = (idx.pcomp_h[i] if q.hangul else idx.pcomp[i]), idx.tokens[i]
        csk, cvk = (idx.skel_h[i], idx.vkey_h[i]) if q.hangul else (idx.skel[i], idx.vkey[i])
        # Hangul spelling vs English spelling is meaningless -> use the phonetic keys instead
        qc, cc = (q.pcomp, cp) if q.hangul else (q.comp, idx.comp[i])
        qa, ca = (q.pcomp, cp) if q.hangul else (q.acomp, idx.acomp[i])
        qt, ctx = (q.pkey, idx.pkey_h[i]) if q.hangul else (q.clean, idx.clean[i])
        shared = q.tokens & ct
        clast = next((t for t in reversed(idx.clean[i].split()) if not t.isdigit()), "")
        cgt = idx.gt[i]
        gg, gi = _gap(q.g, cgt), _gap(q.ig, cgt)
        rows.append([
            float(q.hangul), q.dict_cov,
            fuzz.ratio(qc, cc), fuzz.partial_ratio(qc, cc),
            fuzz.token_sort_ratio(qt, ctx), fuzz.token_set_ratio(qt, ctx),
            JaroWinkler.similarity(qc, cc), Levenshtein.normalized_similarity(qc, cc),
            float(qc == cc and bool(cc)), _prefix(qc, cc),
            fuzz.ratio(q.pcomp, cp), JaroWinkler.similarity(q.pcomp, cp), fuzz.partial_ratio(q.pcomp, cp),
            fuzz.ratio(q.skel, csk),
            min(len(qc), len(cc)) / max(len(qc), len(cc), 1),
            float(q.digits == idx.digits[i]), float(bool(q.digits) or bool(idx.digits[i])),
            gg, gi, min(gg, gi), float(cgt in (q.g, q.ig)), float(min(gg, gi) <= 0.051),
            float(not np.isfinite(q.g) and not np.isfinite(q.ig)), float(not np.isfinite(cgt)),
            np.log1p(cgt) if np.isfinite(cgt) else -1.0,
            flag_p(q.flag, idx.flag[i]), type_p(q.vtype, idx.vtype[i]), float(not q.flag),
            fuzz.ratio(qa, ca), JaroWinkler.similarity(qa, ca),
            len(q.tokens & ct) / max(len(q.tokens | ct), 1), len(q.tokens & ct),
            float(bool(qlast) and qlast == clast), n_same,
            sum(idx.idf.get(t, 0.0) for t in shared) / q_idf, max((idx.idf.get(t, 0.0) for t in shared), default=0.0),
            fuzz.ratio(q.vkey, cvk), JaroWinkler.similarity(q.vkey, cvk),
            float(bool(q.digits) and bool(idx.digits[i]) and q.digits != idx.digits[i]),
            *ko, *ko2, *kob, *hj, *mn,
        ])
    X = np.asarray(rows, dtype=np.float32)
    # within-query relative features (same row only -> row-independent)
    r, pr, gap, kr, kb, kh = X[:, 2], X[:, 10], X[:, 19], X[:, 39], X[:, 49], X[:, 52]
    rel = np.column_stack([
        (-r).argsort().argsort(), (-pr).argsort().argsort(),
        r - r.max(), pr - pr.max(), gap - gap.min(), np.full(len(X), len(X)), kr - kr.max(), kb - kb.max(), kh - kh.max(),
    ]).astype(np.float32)
    return np.hstack([X, rel])


# ------------------------------------------------------------------- artifacts
class BoosterAverage:
    """Mean score of several rankers that differ only in their random seed."""

    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)
def name_key(name: object) -> str:
    """Exact-lookup key for a record's vessel name (train spells each vessel identically)."""
    return compact(clean_hangul(name)) if is_hangul(name) else compact(clean_text(name))


class Matcher:
    """Bundles every learned artifact; `predict` scores each query row independently."""

    def __init__(self, idx: CatalogIndex, callsign_map: dict[str, str], token_dict: dict[str, str],
                 flag_p: CondProb, type_p: CondProb, booster=None,
                 name_map: dict[str, str] | None = None, callsign_name: dict[str, str] | None = None):
        self.idx, self.callsign_map, self.token_dict = idx, callsign_map, token_dict
        self.flag_p, self.type_p, self.booster = flag_p, type_p, booster
        self.name_map = name_map or {}            # train name key -> vessel_id
        self.callsign_name = callsign_name or {}  # train call_sign -> name key seen with it

    def _gt_ok(self, q: Query, vessel_id: str) -> bool:
        j = self.idx.id_to_index.get(vessel_id)
        if j is None:
            return False
        c = self.idx.gt[j]
        if not np.isfinite(c) or not (np.isfinite(q.g) or np.isfinite(q.ig)):
            return True
        return min(_gap(q.g, c), _gap(q.ig, c)) <= 0.055

    def candidates(self, rows: list[dict]):
        """Yield (query, candidate indices, feature matrix) per row; batching is for speed only."""
        qs = [Query(r, self.token_dict) for r in rows]
        sims, per_key = name_sims(qs, self.idx)
        for r, q, sim, ks in zip(rows, qs, sims, per_key):
            cands = retrieve(q, sim, self.idx, ks)
            qv = EMB.vec(r.get("vessel_name")) if (EMB is not None and q.hangul) else None
            if qv is not None:  # meaning neighbours (바다 공주 = Sea Princess) inside ±10% tonnage
                s = EMB.C @ qv
                g = q.g if np.isfinite(q.g) else q.ig
                win = np.flatnonzero(np.abs(self.idx.gt - g) <= EMB_TOL * g) if np.isfinite(g) else np.arange(len(s))
                if len(win):
                    k = min(EMB_TOP, len(win))
                    cands = np.union1d(cands, win[np.argpartition(-s[win], k - 1)[:k]]).astype(int)
            X = pair_features(q, self.idx, cands, self.flag_p, self.type_p)
            if qv is not None:
                es = EMB.C[cands] @ qv
                extra = np.column_stack([es, es - es.max()])
            else:
                extra = np.full((len(cands), 2), -1.0)
            yield q, cands, np.hstack([X, extra.astype(np.float32)])

    def predict(self, rows: list[dict], batch: int = 256, callsign_mode: str = "name_first",
                details: list[dict] | None = None, top_k: int = 10) -> list[str]:
        """callsign_mode:
        "off"        -> retrieval + reranker only
        "hard"       -> trust the train call_sign lookup blindly
        "guarded"    -> call_sign lookup only if that vessel is a retrieved candidate with consistent GT
        "name_first" -> 1) exact train name match (GT consistent)
                        2) call_sign lookup only if its train name is similar to this row's name
                        3) retrieval + reranker
        details: if a list is given, one dict per row is appended with the reranker's top_k
                 (ids, scores), whether the row is Hangul and whether a lookup overrode the ranker."""
        out: list[str] = []
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            for row, (q, cands, X) in zip(chunk, self.candidates(chunk)):
                sc = self.booster.predict(X)
                if CE is not None and q.hangul and len(cands) > 1:
                    top = np.argsort(-sc, kind="stable")[:CE_TOP]
                    sc = sc.copy()
                    sc[top] += CE_W * CE.score(row.get("vessel_name"), self.idx.names[cands[top]])
                best = str(self.idx.ids[cands[int(np.argmax(sc))]])
                ranker_best = best
                cs = row.get("call_sign")
                known = self.callsign_map.get(cs) if isinstance(cs, str) and callsign_mode != "off" else None
                if callsign_mode == "hard":
                    best = known or best
                elif callsign_mode == "guarded":
                    if known is not None and (cands == self.idx.id_to_index.get(known, -1)).any() \
                            and self._gt_ok(q, known):
                        best = known
                elif callsign_mode == "name_first":
                    key = name_key(row.get("vessel_name"))
                    by_name = self.name_map.get(key) if key else None
                    if by_name is not None and self._gt_ok(q, by_name):
                        best = by_name
                    elif known is not None and self._gt_ok(q, known) and \
                            fuzz.ratio(key, self.callsign_name.get(cs, "")) >= 85:
                        best = known
                out.append(best)
                if details is not None:
                    order = np.argsort(-sc, kind="stable")[:top_k]
                    details.append(dict(hangul=q.hangul, top_ids=[str(v) for v in self.idx.ids[cands[order]]],
                                        top_scores=[float(v) for v in sc[order]], overridden=best != ranker_best))
        return out
