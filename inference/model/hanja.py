"""Sino-Korean readings of pinyin vessel names, and the Japanese 丸 (Maru) <-> 환 convention.

Korean port records may write a Chinese vessel by the Korean reading of its characters instead
of by sound (Jin Cheng 金星 -> 금성, Hai Feng 海丰 -> 해풍), and a Japanese "... Maru" as "...환".
This module holds only general language knowledge: for each pinyin syllable, the Korean readings
of the characters commonly written with it (most common first), plus the initial-sound rule
(두음법칙). No vessel names, IDs or answers. Written by hand from general knowledge of hanja.
"""

from __future__ import annotations

import re
from itertools import product

# pinyin syllable -> Korean readings of common characters (most frequent in names first)
READINGS: dict[str, list[str]] = {
    "a": ["아"], "ai": ["애"], "an": ["안"], "ang": ["앙"], "ao": ["오"],
    "ba": ["팔", "파"], "bai": ["백", "배"], "ban": ["반"], "bang": ["방"], "bao": ["보", "포"],
    "bei": ["북", "패", "배"], "ben": ["본", "분"], "beng": ["붕"], "bi": ["비", "필", "벽"],
    "bian": ["변", "편"], "biao": ["표"], "bie": ["별"], "bin": ["빈"], "bing": ["병", "빙"],
    "bo": ["박", "파", "백", "발"], "bu": ["보", "부", "포", "불"],
    "cai": ["재", "채"], "can": ["찬", "참"], "cang": ["창"], "cao": ["조", "초"],
    "ce": ["책", "측"], "cen": ["잠"], "ceng": ["층"], "cha": ["차", "찰"], "chai": ["채"],
    "chan": ["산", "찬"], "chang": ["장", "창", "상"], "chao": ["조", "초"], "che": ["차", "철"],
    "chen": ["진", "신", "침"], "cheng": ["성", "정", "승", "징"], "chi": ["치", "적"],
    "chong": ["충", "중"], "chou": ["주"], "chu": ["출", "초", "처", "저"], "chuan": ["천", "선"],
    "chuang": ["창"], "chui": ["취"], "chun": ["춘", "순"], "chuo": ["작"], "ci": ["자", "차"],
    "cong": ["총", "종"], "cu": ["촉"], "cuan": ["찬"], "cui": ["취", "최"], "cun": ["촌", "존"],
    "cuo": ["착"],
    "da": ["대", "달", "답"], "dai": ["대"], "dan": ["단", "담"], "dang": ["당"], "dao": ["도"],
    "de": ["덕", "득"], "deng": ["등"], "di": ["지", "적", "제", "저"], "dian": ["전", "점"],
    "diao": ["조"], "die": ["첩"], "ding": ["정"], "dong": ["동"], "dou": ["두"],
    "du": ["도", "독", "두"], "duan": ["단"], "dui": ["대"], "dun": ["돈"], "duo": ["다", "탈"],
    "e": ["아", "액"], "en": ["은"], "er": ["이"],
    "fa": ["발", "법"], "fan": ["범", "번", "반"], "fang": ["방"], "fei": ["비"], "fen": ["분"],
    "feng": ["풍", "봉"], "fo": ["불"], "fu": ["복", "부"],
    "gai": ["개"], "gan": ["감", "간"], "gang": ["강", "항"], "gao": ["고"],
    "ge": ["각", "가", "격"], "gen": ["근"], "geng": ["경"], "gong": ["공"], "gou": ["구"],
    "gu": ["고", "곡"], "gua": ["과"], "guai": ["괴"], "guan": ["관"], "guang": ["광"],
    "gui": ["귀", "계"], "gun": ["곤"], "guo": ["국", "과"],
    "hai": ["해"], "han": ["한", "함"], "hang": ["항"], "hao": ["호"],
    "he": ["하", "화", "합"], "hei": ["흑"], "hen": ["흔"], "heng": ["항", "형"], "hong": ["홍"],
    "hou": ["후"], "hu": ["호"], "hua": ["화"], "huai": ["회"], "huan": ["환"], "huang": ["황"],
    "hui": ["휘", "회"], "hun": ["혼"], "huo": ["화", "활", "획"],
    "ji": ["길", "기", "제", "집", "급", "계", "극", "적"], "jia": ["가"],
    "jian": ["건", "견", "검", "간", "감"], "jiang": ["강", "장"], "jiao": ["교"],
    "jie": ["걸", "결", "개", "계", "첩"], "jin": ["금", "진", "근"], "jing": ["경", "정"],
    "jiong": ["형"], "jiu": ["구"], "ju": ["거", "국", "취"], "juan": ["권"], "jue": ["결", "각"],
    "jun": ["군", "준"],
    "kai": ["개"], "kang": ["강"], "ke": ["과", "가", "극", "객"],
    "keng": ["갱"], "kong": ["공"], "kou": ["구"], "kua": ["과"],
    "kuai": ["쾌"], "kuan": ["관"], "kuang": ["광"], "kui": ["규", "괴"], "kun": ["곤"],
    "kuo": ["확"],
    "lai": ["래"], "lan": ["란", "람"], "lang": ["랑"], "lao": ["로"],
    "le": ["락", "륵"], "lei": ["뢰"], "leng": ["랭"], "li": ["리", "력", "려", "립", "례"],
    "lian": ["련"], "liang": ["량"], "liao": ["료"], "lie": ["렬"], "lin": ["림", "린"],
    "ling": ["령"], "liu": ["류"], "long": ["룡", "륭"], "lou": ["루"], "lu": ["로", "륙", "록"],
    "luan": ["란"], "lue": ["략"], "lun": ["륜"], "luo": ["락", "라"], "lv": ["려", "률", "록"],
    "ma": ["마"], "mai": ["매"], "man": ["만"], "mang": ["망"], "mao": ["무", "모"],
    "mei": ["미", "매"], "men": ["문"], "meng": ["몽", "맹"], "mi": ["미", "밀"], "mian": ["면"],
    "miao": ["묘"], "min": ["민"], "ming": ["명"], "mo": ["막", "마", "묵"], "mou": ["모"],
    "mu": ["목", "모"],
    "na": ["나"], "nai": ["내"], "nan": ["남"], "nao": ["뇌"], "nei": ["내"], "neng": ["능"],
    "ni": ["니"], "nian": ["년"], "nie": ["섭"], "ning": ["녕"], "niu": ["우"], "nong": ["농"],
    "nuan": ["난"], "nuo": ["낙"],
    "ou": ["구"],
    "pai": ["배", "파"], "pan": ["반"], "pang": ["방"], "pao": ["포"], "pei": ["배"],
    "pen": ["분"], "peng": ["붕", "봉"], "pian": ["편"], "piao": ["표"],
    "pin": ["품", "빈"], "ping": ["평", "병"], "pu": ["보", "포"],
    "qi": ["기", "칠", "계", "제"], "qia": ["흡"], "qian": ["천", "건", "전", "겸", "잠"],
    "qiang": ["강"], "qiao": ["교"], "qie": ["절"], "qin": ["진", "친", "금", "근"],
    "qing": ["청", "경"], "qiong": ["경"], "qiu": ["추", "구"], "qu": ["구", "취"],
    "quan": ["전", "천", "권"], "que": ["작", "확"], "qun": ["군"],
    "ran": ["연"], "rang": ["양"], "rao": ["요"], "ren": ["인"], "reng": ["잉"],
    "ri": ["일"], "rong": ["영", "용"], "rou": ["유"], "ru": ["여", "유", "입"], "ruan": ["연"],
    "rui": ["서", "예"], "run": ["윤"], "ruo": ["약"],
    "sai": ["새"], "san": ["삼"], "sang": ["상"], "sao": ["소"],
    "sen": ["삼"], "seng": ["승"], "sha": ["사"], "shan": ["산", "선"], "shang": ["상"],
    "shao": ["소"], "she": ["사", "설"], "shen": ["심", "신"], "sheng": ["성", "승", "생"],
    "shi": ["시", "실", "석", "세", "사", "식"], "shou": ["수"], "shu": ["서", "수", "숙"],
    "shuang": ["쌍", "상"], "shui": ["수"], "shun": ["순"], "shuo": ["삭"], "si": ["사"],
    "song": ["송"], "sou": ["수"], "su": ["소", "속", "숙"], "suan": ["산"], "sui": ["수"],
    "sun": ["손"], "suo": ["소"],
    "ta": ["탑", "타"], "tai": ["태", "대"], "tan": ["탄", "담"], "tang": ["당"], "tao": ["도"],
    "teng": ["등"], "tian": ["천", "전"], "tiao": ["조"],
    "tie": ["철"], "ting": ["정"], "tong": ["통", "동"], "tou": ["투"], "tu": ["도", "토"],
    "tuan": ["단"], "tui": ["퇴"], "tun": ["둔"], "tuo": ["탁"],
    "wai": ["외"], "wan": ["만", "완"], "wang": ["왕", "망"], "wei": ["위", "미", "유"],
    "wen": ["문", "온"], "weng": ["옹"], "wo": ["옥", "와"], "wu": ["오", "무"],
    "xi": ["서", "희", "석", "식"], "xia": ["하"], "xian": ["선", "현", "험"], "xiang": ["상", "향"],
    "xiao": ["소", "효"], "xie": ["협", "해"], "xin": ["신", "흠", "심", "흔"],
    "xing": ["성", "흥", "행", "형"], "xiong": ["웅"], "xiu": ["수"], "xu": ["서", "욱", "허"],
    "xuan": ["선", "현"], "xue": ["설", "학"], "xun": ["순", "훈"],
    "ya": ["아"], "yan": ["연", "염", "안", "언", "암"], "yang": ["양"], "yao": ["요"],
    "ye": ["업", "야", "엽"], "yi": ["일", "의", "익", "이", "역", "예"], "yin": ["은", "인"],
    "ying": ["영", "응"], "yong": ["영", "용"], "you": ["우", "유"],
    "yu": ["우", "어", "유", "옥", "여", "욱", "예"], "yuan": ["원"], "yue": ["월", "악", "열"],
    "yun": ["운"],
    "zai": ["재"], "zan": ["찬"], "zang": ["장"], "zao": ["조"], "ze": ["택", "칙"],
    "zeng": ["증"], "zha": ["찰"], "zhai": ["채"], "zhan": ["전", "점"], "zhang": ["장"],
    "zhao": ["조", "소"], "zhe": ["절", "철"], "zhen": ["진", "정"], "zheng": ["정", "증"],
    "zhi": ["지", "치", "직", "제"], "zhong": ["중", "종"], "zhou": ["주"], "zhu": ["주", "축"],
    "zhuan": ["전"], "zhuang": ["장"], "zhui": ["추"], "zhun": ["준"], "zhuo": ["탁"],
    "zi": ["자"], "zong": ["종"], "zou": ["주"], "zu": ["조", "족"], "zuan": ["찬"], "zui": ["최"],
    "zun": ["존", "준"], "zuo": ["좌", "작"],
}
# syllables left out on purpose (ka, la, sa, ha, ti, te, re, pa, ...): rare in Chinese names but common in
# Indonesian/Japanese/Spanish ones (Timika, Salacia), which would otherwise be read as pinyin
_SYL = sorted(READINGS, key=len, reverse=True)

# 두음법칙: a word-initial ㄹ/ㄴ reading changes (리 -> 이, 룡 -> 용, 녕 -> 영, 로 -> 노 ...)
_INITIAL = {"라": "나", "락": "낙", "란": "난", "람": "남", "랑": "낭", "래": "내", "량": "양", "려": "여",
            "력": "역", "련": "연", "령": "영", "로": "노", "록": "녹", "뢰": "뇌", "료": "요", "룡": "용",
            "루": "누", "류": "유", "륙": "육", "륜": "윤", "리": "이", "린": "인", "림": "임",
            "녀": "여", "년": "연", "녕": "영", "니": "이", "랍": "납", "랭": "냉", "렬": "열",
            "략": "약", "륵": "늑", "례": "예", "립": "입", "률": "율", "륭": "융"}


def segment(token: str) -> list[str] | None:
    """Fewest-syllable split of a romanized token into pinyin syllables; None if impossible."""
    t = token.lower()
    best: list[list[str] | None] = [None] * (len(t) + 1)
    best[0] = []
    for i in range(len(t)):
        if best[i] is None:
            continue
        for s in _SYL:
            j = i + len(s)
            if t.startswith(s, i) and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + [s]
    return best[len(t)]


def _options(syls: list[str]) -> list[list[str]]:
    opts = []
    for k, s in enumerate(syls):
        o = list(dict.fromkeys(READINGS[s]))
        if k == 0:  # word-initial: both the rule-applied and the plain reading occur in records
            o = list(dict.fromkeys([_INITIAL.get(r, r) for r in o] + o))
        opts.append(o)
    return opts


def catalog_keys(name: object, max_keys: int = 8) -> list[str]:
    """Spacing-free Sino-Korean reading keys of a pinyin-looking catalog name, most likely first.
    'Jin Cheng 8' -> ['금성8', '진성8', '금정8', ...]; [] when the name does not read as pinyin."""
    if not isinstance(name, str):
        return []
    toks = re.findall(r"[A-Za-z]+|\d+", name)
    words = [t for t in toks if not t.isdigit() and t.lower() not in ("no",)]
    if not words or any(t.lower() in _ENGLISH for t in words):
        return []
    segs = [segment(t) for t in words]
    if any(s is None for s in segs) or sum(len(s) for s in segs) < 2:
        return []
    per_word = [_options(s) for s in segs]
    # best reading everywhere, then vary one syllable at a time (keeps the list short)
    base = [[o[0] for o in w] for w in per_word]
    variants = [base]
    for wi, w in enumerate(per_word):
        for si, o in enumerate(w):
            for alt in o[1:]:
                v = [list(x) for x in base]
                v[wi][si] = alt
                variants.append(v)
    digits = "".join(t for t in toks if t.isdigit())
    keys = ["".join("".join(w) for w in v) + digits for v in variants]
    return list(dict.fromkeys(keys))[:max_keys]


def catalog_syllables(name: object) -> tuple[tuple[tuple[str, ...], ...], str] | None:
    """Every syllable's reading options in order (words flattened) and the digits, for alignment:
    'Jin Cheng 8' -> ((('금', '진', '근'), ('성', '정', '승', '징')), '8'). None if not pinyin."""
    if not isinstance(name, str):
        return None
    toks = re.findall(r"[A-Za-z]+|\d+", name)
    words = [t for t in toks if not t.isdigit() and t.lower() not in ("no",)]
    if not words or any(t.lower() in _ENGLISH for t in words):
        return None
    segs = [segment(t) for t in words]
    if any(s is None for s in segs) or sum(len(s) for s in segs) < 2:
        return None
    opts = tuple(tuple(o) for s in segs for o in _options(s))
    return opts, "".join(t for t in toks if t.isdigit())


def align(ko_k: str, syl: tuple[tuple[tuple[str, ...], ...], str] | None) -> tuple[int, str] | None:
    """Position-by-position match of a spacing-free Hangul key against a name's reading options.
    Returns (syllables whose reading is one of the options, the closest reading string + digits), or
    None when the syllable counts differ. Covers every combination of readings, unlike the short
    key list of catalog_keys."""
    if syl is None:
        return None
    opts, digits = syl
    q = re.sub(r"\d", "", ko_k)
    if len(q) != len(opts):
        return None
    hit, out = 0, []
    for c, o in zip(q, opts):
        if c in o:
            hit += 1
            out.append(c)
        else:  # a typo: keep the reading that shares the most jamo with it
            out.append(max(o, key=lambda r: _jamo_overlap(c, r)))
    return hit, "".join(out) + digits


def _jamo_overlap(a: str, b: str) -> int:
    def parts(ch):
        v = ord(ch) - 0xAC00
        return (v // 588, (v // 28) % 21, v % 28) if 0 <= v < 11172 else (ch,)
    return sum(x == y for x, y in zip(parts(a), parts(b)))


def record_reading(name: str, rng) -> str:
    """A plausible Sino-Korean record for a pinyin catalog name (for synthetic training/validation):
    a random reading per syllable, words joined with or without a space."""
    toks = re.findall(r"[A-Za-z]+|\d+", name)
    out = []
    for t in toks:
        if t.isdigit():
            out.append(t)
            continue
        if t.lower() == "no":
            continue
        opts = _options(segment(t))
        out.append("".join(o[0] if rng.random() < 0.6 else o[int(rng.integers(len(o)))] for o in opts))
    sep = " " if rng.random() < 0.5 else ""
    return sep.join(out)


def maru_key(ko_key: str) -> str:
    """'가시마환10' -> '가시마마루10' (Japanese 丸 written by its Korean reading); '' if no 환 suffix."""
    k = re.sub(r"환(?=\d|$)", "마루", ko_key)
    return k if k != ko_key else ""


# common English words that also split into pinyin syllables (Sea = se+a, Ocean = o+ce+an ...)
_ENGLISH = {
    "sea", "ocean", "star", "bay", "lady", "pride", "glory", "sun", "can", "man", "gas", "ace",
    "hope", "queen", "king", "jade", "lion", "dragon", "pioneer", "leader", "spirit", "angel",
    "orient", "pacific", "marine", "express", "trader", "line", "liner", "one", "new", "golden", "gold",
    "silver", "blue", "green", "red", "white", "black", "grand", "great", "royal", "noble", "fortune",
    "harmony", "unity", "venus", "luna", "nova", "zen", "maru", "chem", "chemi", "tanker", "bulk",
    "ship", "cargo", "princess", "prince", "rose", "lily", "joy", "sky", "wind", "wave", "light",
}


# ----------------------------------------------------------------- English words written by meaning
# A record may translate common English words of a name instead of spelling them by sound
# (Ocean Star -> 해양성 / 해양별, Golden Glory -> 황금영광; also plurals and compounds: Seastar -> 해성). English word -> Korean words with that
# meaning, most common in names first. General vocabulary only; no vessel names or answers.
MEANINGS: dict[str, list[str]] = {
    "star": ["성", "별"], "sea": ["해", "바다"], "ocean": ["해양", "대양", "바다"], "pacific": ["태평양"],
    "atlantic": ["대서양"], "glory": ["영광"], "golden": ["황금", "금"], "gold": ["금", "황금"], "silver": ["은"],
    "new": ["신", "새"], "sun": ["태양", "해", "일"], "sunny": ["양지"], "sunrise": ["일출", "해돋이"],
    "moon": ["월", "달"], "sky": ["천", "하늘"], "island": ["도", "섬"], "bay": ["만"], "cape": ["곶"],
    "river": ["강", "하"], "lake": ["호"], "mount": ["산"], "port": ["항"], "bridge": ["교", "다리"],
    "east": ["동"], "eastern": ["동방", "동"], "west": ["서"], "south": ["남"], "southern": ["남방", "남"],
    "north": ["북"], "orient": ["동양"], "oriental": ["동양"], "blue": ["청", "푸른"],
    "green": ["녹", "녹색", "초록"], "red": ["홍", "적", "붉은"], "white": ["백", "흰"], "black": ["흑", "검은"],
    "pearl": ["진주"], "diamond": ["금강"], "jade": ["옥", "비취"], "crystal": ["수정"], "emerald": ["녹옥"],
    "ruby": ["홍옥"], "sapphire": ["청옥"], "coral": ["산호"], "amber": ["호박"], "crown": ["왕관"],
    "royal": ["왕실", "황실"], "king": ["왕"], "queen": ["여왕"], "prince": ["왕자"], "princess": ["공주"],
    "lady": ["숙녀"], "captain": ["선장"], "chief": ["대장"], "great": ["대", "위대한"], "grand": ["대", "웅대한"],
    "big": ["대", "큰"], "super": ["초"], "fortune": ["행운", "복"], "lucky": ["행운", "길"],
    "harmony": ["화합", "조화"], "peace": ["평화"], "hope": ["희망"], "dream": ["꿈", "몽"],
    "pride": ["긍지", "자랑"], "spirit": ["정신", "혼"], "victory": ["승리"], "success": ["성공"],
    "prosperity": ["번영"], "progress": ["진보", "발전"], "unity": ["단결", "통일"], "union": ["연합"],
    "liberty": ["자유"], "honor": ["명예"], "grace": ["은혜", "은총"], "faith": ["신앙", "믿음"], "wisdom": ["지혜"],
    "joy": ["기쁨", "희"], "confidence": ["신뢰"], "future": ["미래"], "vision": ["전망"], "power": ["력", "힘"],
    "bright": ["명", "광명"], "light": ["광", "빛"], "spring": ["춘", "봄"], "morning": ["아침", "조"],
    "wind": ["풍", "바람"], "breeze": ["미풍", "산들바람"], "wave": ["파", "파도"], "tide": ["조", "조수"],
    "dragon": ["용", "룡"], "phoenix": ["봉황", "불사조"], "tiger": ["호랑이", "호"], "lion": ["사자"],
    "eagle": ["독수리"], "falcon": ["매"], "hawk": ["매"], "swan": ["백조"], "dolphin": ["돌고래"],
    "pelican": ["사다새"], "rose": ["장미"], "lotus": ["연꽃", "연"], "flower": ["화", "꽃"],
    "pine": ["송", "소나무"], "galaxy": ["은하"], "arrow": ["화살"], "pioneer": ["개척자", "선구자"],
    "explorer": ["탐험가"], "navigator": ["항해사"], "voyager": ["항해자"], "leader": ["지도자"],
    "champion": ["우승자"], "warrior": ["전사"], "trader": ["무역", "상인"], "carrier": ["운반선"],
    "express": ["특급", "급행"], "highway": ["고속도로"], "city": ["시", "도시"], "one": ["일", "하나"],
    "seven": ["칠"], "brave": ["용감한", "용"], "fast": ["쾌속"], "global": ["세계"], "marine": ["해양", "해상"],
    "seaways": ["해로"], "seas": ["해", "바다"], "horizon": ["수평선", "지평선"], "gulf": ["만"], "front": ["전선"],
    "delta": ["삼각주"], "frontier": ["개척지", "변경"], "arctic": ["북극"], "polar": ["극지", "북극"],
    "harbour": ["항", "항구"], "harbor": ["항", "항구"], "coastal": ["연안"], "coast": ["해안", "연안"],
    "oceanic": ["해양"], "deep": ["심해", "깊은"], "islander": ["섬사람"], "isle": ["도", "섬"], "strait": ["해협"],
    "straits": ["해협"], "point": ["곶"], "park": ["공원"], "forest": ["숲", "삼림"], "desert": ["사막"],
    "earth": ["지구"], "world": ["세계"], "universe": ["우주"], "cosmos": ["우주"], "planet": ["행성"],
    "rainbow": ["무지개"], "ice": ["빙"], "water": ["수", "물"], "stream": ["류", "시냇물"], "reef": ["초", "암초"],
    "beach": ["해변"], "rock": ["암", "바위"], "peak": ["봉", "정상"], "summit": ["정상"], "mountain": ["산"],
    "hill": ["언덕"], "valley": ["계곡"], "canyon": ["협곡"], "meadow": ["초원"], "land": ["육지", "땅"],
    "sand": ["모래"], "storm": ["폭풍"], "thunder": ["천둥", "뇌"], "cloud": ["운", "구름"], "dawn": ["새벽", "여명"],
    "sunshine": ["햇빛", "일광"], "solar": ["태양"], "winter": ["동", "겨울"], "summer": ["하", "여름"],
    "midnight": ["자정"], "day": ["일", "날"], "time": ["시간"], "fjord": ["협만"], "tropic": ["열대"],
    "tropical": ["열대"], "northern": ["북방", "북"], "western": ["서방", "서"], "easterly": ["동풍"],
    "eclipse": ["일식"], "venus": ["금성"], "mercury": ["수성"], "mars": ["화성"], "jupiter": ["목성"],
    "saturn": ["토성"], "neptune": ["해왕성"], "uranus": ["천왕성"], "pluto": ["명왕성"], "polaris": ["북극성"],
    "aries": ["양자리"], "taurus": ["황소자리"], "leo": ["사자자리"], "virgo": ["처녀자리"], "libra": ["천칭자리"],
    "scorpio": ["전갈자리"], "sagittarius": ["궁수자리"], "capricorn": ["염소자리"], "aquarius": ["물병자리"],
    "pisces": ["물고기자리"], "stars": ["별", "성"], "stellar": ["별"], "comet": ["혜성"], "meteor": ["유성"],
    "orange": ["주황"], "yellow": ["황", "노란"], "violet": ["보라"], "indigo": ["남색"], "crimson": ["진홍"],
    "scarlet": ["주홍"], "azure": ["하늘색", "벽"], "topaz": ["황옥"], "opal": ["단백석"], "garnet": ["석류석"],
    "amethyst": ["자수정"], "aquamarine": ["남옥"], "jewel": ["보석"], "gem": ["보석"], "treasure": ["보물"],
    "platinum": ["백금"], "iron": ["철"], "steel": ["강철", "철"], "oil": ["유", "석유"], "mineral": ["광물"],
    "ore": ["광석"], "fuel": ["연료"], "chemical": ["화학"], "lily": ["백합"], "orchid": ["난", "난초"],
    "magnolia": ["목련"], "camellia": ["동백"], "sunflower": ["해바라기"], "maple": ["단풍"], "oak": ["참나무"],
    "cedar": ["삼나무"], "palm": ["야자"], "blossom": ["꽃"], "peony": ["모란"], "cherry": ["벚꽃", "앵두"],
    "sakura": ["벚꽃"], "ivy": ["담쟁이"], "laurel": ["월계수"], "azalea": ["진달래"], "flora": ["화"],
    "marlin": ["청새치"], "whale": ["고래"], "shark": ["상어"], "seal": ["물개"], "orca": ["범고래"],
    "seagull": ["갈매기"], "gull": ["갈매기"], "albatross": ["신천옹"], "heron": ["왜가리"], "crane": ["학", "두루미"],
    "osprey": ["물수리"], "kestrel": ["황조롱이"], "petrel": ["바다제비"], "tern": ["제비갈매기"], "egret": ["백로"],
    "stork": ["황새"], "owl": ["부엉이"], "robin": ["울새"], "bird": ["새", "조"], "leopard": ["표범"],
    "panther": ["흑표범"], "wolf": ["늑대"], "fox": ["여우"], "bear": ["곰"], "horse": ["마", "말"],
    "lynx": ["스라소니"], "unicorn": ["일각수"], "pegasus": ["천마"], "swift": ["칼새", "신속"], "bee": ["벌"],
    "seahorse": ["해마"], "sealion": ["바다사자"], "beluga": ["흰고래"], "cougar": ["퓨마"], "hawks": ["매"],
    "raven": ["까마귀"], "mariner": ["선원", "뱃사람"], "sailor": ["선원"], "admiral": ["제독"],
    "commander": ["사령관"], "master": ["선장", "주인"], "knight": ["기사"], "hero": ["영웅"], "guardian": ["수호자"],
    "defender": ["수호자"], "protector": ["보호자"], "sentinel": ["파수꾼"], "hunter": ["사냥꾼"], "fisher": ["어부"],
    "runner": ["주자"], "rider": ["기수"], "traveller": ["여행자"], "adventurer": ["모험가"],
    "discoverer": ["발견자"], "challenger": ["도전자"], "conqueror": ["정복자"], "achiever": ["성취자"],
    "innovator": ["혁신가"], "winner": ["승자"], "fighter": ["전사"], "patriot": ["애국자"],
    "friend": ["친구", "벗"], "brother": ["형제"], "brothers": ["형제"], "emperor": ["황제"], "empress": ["황후"],
    "duchess": ["공작부인"], "lord": ["주"], "president": ["대통령"], "professor": ["교수"], "miss": ["양"],
    "giant": ["거인"], "angel": ["천사"], "mermaid": ["인어"], "merchant": ["상인"], "surveyor": ["측량사"],
    "transporter": ["운송선"], "supporter": ["지원자"], "provider": ["공급자"], "supplier": ["공급자"],
    "constructor": ["건설자"], "diver": ["잠수부"], "scout": ["정찰"], "courage": ["용기"],
    "triumph": ["승리", "개선"], "freedom": ["자유"], "noble": ["고귀한", "귀"], "integrity": ["성실"],
    "prestige": ["명성"], "majestic": ["장엄한"], "majesty": ["위엄"], "trust": ["신뢰"], "luck": ["행운"],
    "advantage": ["우위"], "loyalty": ["충성"], "eternity": ["영원"], "eternal": ["영원"], "forever": ["영원"],
    "infinity": ["무한"], "legacy": ["유산"], "legend": ["전설"], "wealth": ["부"], "rich": ["부"],
    "beauty": ["미"], "melody": ["선율"], "symphony": ["교향곡"], "dignity": ["존엄"], "serenity": ["평온"],
    "excellence": ["탁월"], "destiny": ["운명"], "brilliance": ["광채"], "brilliant": ["찬란한"],
    "splendor": ["광휘"], "splendour": ["광휘"], "ambition": ["야망"], "innovation": ["혁신"],
    "friendship": ["우정"], "passion": ["열정"], "providence": ["섭리"], "justice": ["정의"], "valor": ["용맹"],
    "valour": ["용맹"], "bravery": ["용기"], "strength": ["힘"], "fidelity": ["충실"], "sincerity": ["성실"],
    "honesty": ["정직"], "virtue": ["미덕"], "fortitude": ["불굴"], "patience": ["인내"], "respect": ["존경"],
    "happiness": ["행복"], "happy": ["행복"], "blessing": ["축복"], "miracle": ["기적"], "wonder": ["경이"],
    "delight": ["기쁨"], "felicity": ["행복"], "bliss": ["축복"], "love": ["사랑", "애"], "promise": ["약속"],
    "wish": ["소원"], "inspiration": ["영감"], "tenacity": ["끈기"], "resolution": ["결의"], "resolve": ["결의"],
    "resilience": ["회복력"], "esteem": ["존중"], "independence": ["독립"], "elegant": ["우아한"],
    "glorious": ["영광스러운"], "precious": ["귀한"], "sweet": ["달콤한"], "strong": ["강한"], "mighty": ["강력한"],
    "superior": ["우수한"], "supreme": ["최고"], "imperial": ["제국"], "sovereign": ["주권"], "prime": ["최고"],
    "first": ["제일", "첫"], "best": ["최고"], "true": ["참"], "good": ["좋은", "선"], "smooth": ["순항"],
    "clean": ["청정"], "cool": ["시원한"], "calm": ["평온"], "fair": ["순풍"], "harvest": ["수확"],
    "abundance": ["풍요"], "flourish": ["번성"], "thriving": ["번성"], "prosper": ["번영"], "rising": ["상승"],
    "advance": ["전진"], "forward": ["전진"], "evolution": ["진화"], "genesis": ["창세"], "century": ["세기"],
    "millennium": ["천년"], "dynasty": ["왕조"], "empire": ["제국"], "kingdom": ["왕국"], "paradise": ["낙원"],
    "castle": ["성"], "palace": ["궁전"], "tower": ["탑"], "gate": ["문"], "road": ["길", "로"],
    "way": ["길", "도"], "journey": ["여정"], "voyage": ["항해"], "quest": ["탐색"], "odyssey": ["대장정"],
    "adventure": ["모험"], "challenge": ["도전"], "discovery": ["발견"], "venture": ["모험"],
    "enterprise": ["기업"], "alliance": ["동맹"], "united": ["연합"], "universal": ["보편"], "general": ["일반"],
    "national": ["국민"], "federal": ["연방"], "mega": ["거대"], "ultra": ["초"], "high": ["고"],
    "little": ["소"], "young": ["청년"], "twin": ["쌍둥이"], "magic": ["마법"], "fantasy": ["환상"],
    "dynamic": ["역동"], "flying": ["비행"], "fly": ["비행"], "speed": ["속도"], "thunderbird": ["천둥새"],
    "two": ["이", "둘"], "three": ["삼", "셋"], "four": ["사", "넷"], "five": ["오"], "nine": ["구"],
    "ten": ["십"], "bulk": ["산적"], "tanker": ["유조선"], "ferry": ["여객선"], "cargo": ["화물"],
    "shipping": ["해운"], "ship": ["선"], "transport": ["운송"], "logistics": ["물류"], "clipper": ["쾌속선"],
    "cruiser": ["순양함"], "barge": ["바지선"], "reefer": ["냉동선"],
}


def word_meanings(word: str) -> list[str]:
    """Translations of a word: the table, a plural (Stars -> 별), or a compound of table words
    (Seastar -> 해성, Goldenstar -> 황금성; each part >= 3 letters)."""
    w = word.lower()
    if w in MEANINGS:
        return MEANINGS[w]
    if len(w) > 4 and w.endswith("s") and w[:-1] in MEANINGS:
        return MEANINGS[w[:-1]]
    for i in range(3, len(w) - 2):
        a, b = w[:i], w[i:]
        ma = MEANINGS.get(a)
        mb = MEANINGS.get(b) or (MEANINGS.get(b[:-1]) if b.endswith("s") and len(b) > 3 else None)
        if ma and mb:
            return [x + y for x in ma[:2] for y in mb[:2]]
    return []


def meaning_words(name: object, ko: str = ""):
    """Per word of a catalog name: Korean words for it (meaning translations, then its sound spelling
    from catalog_ko), plus the digits. None unless some word has a meaning translation and every
    other word a sound spelling."""
    if not isinstance(name, str):
        return None
    toks = re.findall(r"[A-Za-z]+|\d+", name)
    words = [t for t in toks if not t.isdigit() and t.lower() not in ("no",)]
    if not words or not any(word_meanings(t) for t in words):
        return None
    ko_toks = [t for t in re.findall(r"[가-힣]+", ko or "") if t not in ("넘버", "호")]
    phon = ko_toks if len(ko_toks) == len(words) else [""] * len(words)
    phon = [p if len(w) / 3 <= len(p) <= len(w) else "" for w, p in zip(words, phon)]
    out = []
    for w, p in zip(words, phon):
        m = tuple(x for x in word_meanings(w) if x != p)  # a translation must differ from the sound
        if not m and not p:
            return None
        out.append((m, p))
    return tuple(out), "".join(t for t in toks if t.isdigit())


def meaning_keys(mw, limit: int = 12) -> list[str]:
    """Spacing-free keys with at least one word translated by meaning, most likely first."""
    if mw is None:
        return []
    words, digits = mw
    keys = []
    for combo in product(*[[(x, True) for x in m] + ([(p, False)] if p else []) for m, p in words]):
        if any(tr for _, tr in combo):
            keys.append("".join(x for x, _ in combo) + digits)
        if len(keys) >= limit:
            break
    return list(dict.fromkeys(keys))


def record_meaning(mw, rng) -> str:
    """Synthetic record for meaning_words: each translatable word by meaning (70%, at least one),
    the rest by sound; words joined with or without a space."""
    words, digits = mw
    tr = [bool(m) and (not p or rng.random() < 0.7) for m, p in words]
    if not any(tr):
        tr[int(rng.choice([i for i, (m, _) in enumerate(words) if m]))] = True
    out = [(m[0] if rng.random() < 0.6 else m[int(rng.integers(len(m)))]) if t else p
           for (m, p), t in zip(words, tr)]
    if digits:
        out.append(digits)
    return (" " if rng.random() < 0.5 else "").join(out)
