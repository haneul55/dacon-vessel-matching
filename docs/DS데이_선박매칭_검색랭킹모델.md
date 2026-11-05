# DS데이_선박매칭_검색랭킹모델

항만 입출항 기록 한 줄을 받아서, 약 7만 9천 척이 들어 있는 선박 명부에서 그 기록이 어떤 배인지 찾는 모델입니다. 최종 Public LB 점수는 0.95240이었습니다.

모델 구조를 먼저 설명하고, 그다음에 실제 코드를 단계별로 보여 드리겠습니다. 코드는 핵심 줄만 잘라 왔고, 전체 코드는 저장소의 `inference/`, `training/` 폴더에 있습니다.

- [모델 이름을 이렇게 지은 이유](#모델-이름을-이렇게-지은-이유)
- [전체 흐름](#전체-흐름)
- [기록 하나를 따라가며 보는 코드](#기록-하나를-따라가며-보는-코드)
- [학습 방법](#학습-방법)
- [코드 위치 정리](#코드-위치-정리)
- [결과와 아쉬운 점](#결과와-아쉬운-점)
- [예상 질의응답](#예상-질의응답)

---

## 모델 이름을 이렇게 지은 이유

겉으로 보면 기록을 보고 배를 맞히는 분류 문제 같지만, 정답이 될 수 있는 배가 78,595척이라 배마다 클래스를 두는 방식은 맞지 않습니다. 그래서 기록과 명부의 배를 짝지어 주는 매칭 문제로 보고 풀었고, 이름에 "선박매칭"을 넣었습니다.

"검색랭킹"은 모델의 뼈대를 뜻합니다. 명부 전체에서 후보를 먼저 검색해 오고, 학습한 모델이 그 후보들의 순위를 매기는 구조입니다. LaBSE나 cross-encoder 같은 자연어 모델도 쓰기는 했지만, 이 뼈대를 보조하는 역할이라서 이름에는 넣지 않았습니다.

---

## 전체 흐름

한 기록을 처리하는 순서는 다음과 같습니다.

1. 한글 선박명을 영문 명부와 비교할 수 있도록 발음, 자모, 한자음 형태로 바꿉니다.
2. 명부에서 비슷한 배를 100척 정도로 추립니다.
3. 추린 후보마다 68가지 특징을 계산합니다.
4. LightGBM 랭커가 후보 순위를 매기고, 한글 기록은 cross-encoder로 상위 5척을 한 번 더 채점합니다.
5. 마지막으로 호출부호 규칙으로 답을 점검합니다.

<p align="center"><img src="../assets/architecture.png" width="900" alt="모델 구조"></p>

명부 7만 9천 척을 전부 정밀하게 비교하면 시간이 너무 오래 걸립니다. 그래서 앞 단계에서는 빠르고 단순한 방법으로 범위를 줄이고, 뒤 단계에서만 무거운 모델을 씁니다. 검색 엔진이나 추천 시스템에서 많이 쓰는 retrieve → rerank 방식과 같습니다.

---

## 기록 하나를 따라가며 보는 코드

설명을 위해 `금성8호`라는 가상의 기록을 하나 만들었습니다. 실제 데이터에 있는 행은 아니고, 명부의 정답은 `Jin Cheng 8`이라고 가정했습니다.

```
vessel_name="금성8호", call_sign="CS_9F3A…", gross_tonnage=1234, flag_name="파나마"
```

### 1단계. 선박명을 비교 가능한 형태로 바꾸기

명부에는 영문 이름만 있기 때문에 한글 기록을 그대로 비교할 수는 없습니다. 그래서 기록 하나로 여러 종류의 키를 만듭니다. `금성8호`라면 발음 키는 `geum seong 8`, 자모 키는 `ㄱㅡㅁㅅㅓㅇ 8`이 됩니다. 중국 배 이름을 한자음으로 읽어 적은 경우에 대비해 한자음 키(`Jin Cheng 8`)도 만듭니다.

발음 키는 소리 나는 대로 적은 이름을 찾을 때 쓰고, 자모 키는 한 글자 오타가 있을 때 도움이 됩니다. `호`나 `제` 같은 글자는 빼고, 숫자는 따로 보관해 두었다가 뒤에서 비교합니다.

`inference/model/vessel_lib.py`의 `Query` 클래스

```python
class Query:
    def __init__(self, row, token_dict):
        name = row.get("vessel_name")
        self.hangul = is_hangul(name)
        self.ko_k = ko_key(name) if self.hangul else ""     # 한글 키
        self.ko_j = jamo(self.ko_k)                          # 자모로 분해
        if self.hangul:
            # 발음 키: 한글 → 로마자 → 발음 규칙
            self.pkey = phonetic(romanize(clean_hangul(name)), english=False)
        ...
        self.digits = digits_of(self.clean)                  # 숫자는 따로 보관
        self.g, self.ig = num(row.get("gross_tonnage")), num(row.get("international_gross_tonnage"))
```

이렇게 기록 한 줄을 `Query` 객체 하나로 정리해 두면, 이후 단계는 모두 이 객체만 가지고 작업합니다.

### 2단계. 후보 추리기

여러 기준으로 후보를 뽑아서 합칩니다. 기준은 다섯 가지입니다.

- 이름이 가장 비슷한 40척
- 톤수 ±5.5% 범위 안에서 이름이 가장 비슷한 30척
- 톤수가 정확히 같은 배
- 한자음·뜻 사전으로 찾은 배
- LaBSE로 뜻이 가까운 배 (최대 50척)

톤수가 같은 배를 따로 넣은 데는 이유가 있습니다. 이름을 바꾼 배는 이름 유사도로는 잡히지 않기 때문입니다. 마지막의 LaBSE 검색은 `해풍`을 `Sea Wind`로 번역해 적은 경우처럼, 발음으로는 전혀 연결되지 않는 이름을 찾기 위해 넣었습니다.

이 단계에서 가장 신경 쓴 것은 정답을 놓치지 않는 것입니다. 정답이 후보에 들어오지 못하면 뒤에서 아무리 잘 골라도 맞힐 수 없기 때문입니다.

`vessel_lib.py`의 `retrieve()`

```python
def retrieve(q, sim, idx, key_sims=(), top_name=40, top_gt=30):
    picks = [np.argpartition(-sim, top_name)[:top_name]]          # 이름 상위 40
    if HANJA and q.hangul and q.ko_k:
        picks.append(np.asarray(idx.hj_index.get(q.ko_k, [])))     # 한자음 사전
    win = np.union1d(idx.gt_window(q.g), idx.gt_window(q.ig))      # 톤수 ±5.5% 범위
    if len(win):
        k = min(top_gt, len(win))
        picks.append(win[np.argpartition(-sim[win], k - 1)[:k]])   # 범위 안 상위 30
    ...
    return np.unique(np.concatenate(picks))                        # 합집합
```

LaBSE 의미 검색은 `Matcher.candidates()` 안에서 이어집니다.

```python
qv = EMB.vec(r.get("vessel_name"))            # 한글 기록 → 768차원 벡터
s = EMB.C @ qv                                 # 명부 전체와 코사인 유사도
win = np.flatnonzero(np.abs(self.idx.gt - g) <= EMB_TOL * g)   # 톤수 ±10%
cands = np.union1d(cands, win[np.argpartition(-s[win], k - 1)[:k]])
```

### 3단계. 후보마다 특징 계산하기

남은 후보 하나하나에 대해, 기록과 얼마나 닮았는지를 68개 숫자로 계산합니다. 철자 유사도, 발음 유사도, 한자음 일치 여부, 숫자 일치 여부, 톤수 차이, 국적 확률, LaBSE 유사도 등이 들어갑니다. 후보가 N척이면 N × 68 크기의 표가 만들어집니다.

한 가지 주의할 점이 있었습니다. 한글 기록과 영문 이름은 철자를 직접 비교해도 의미가 없습니다. 그래서 한글 기록일 때는 철자 대신 발음 키끼리 비교하도록 했습니다.

`vessel_lib.py`의 `pair_features()` (특징 이름 목록은 `FEATURES`)

```python
qc, cc = (q.pcomp, cp) if q.hangul else (q.comp, idx.comp[i])   # 한글이면 발음 키로 비교

rows.append([
    float(q.hangul), q.dict_cov,
    fuzz.ratio(qc, cc), fuzz.partial_ratio(qc, cc),                # 철자 유사도
    fuzz.token_sort_ratio(qt, ctx), fuzz.token_set_ratio(qt, ctx),
    JaroWinkler.similarity(qc, cc), Levenshtein.normalized_similarity(qc, cc),
    ...                                                            # 발음·한자음·톤수·국적 등
])
```

### 4단계. 순위 매기기

특징 표를 LightGBM LambdaRank 모델에 넣어 후보별 점수를 받습니다. 한글 기록과 영문 기록은 틀리는 양상이 달라서 모델을 따로 두었습니다. 한글 기록은 한글 전용 모델 3개, 영문 기록은 영문 전용 모델 9개의 점수를 평균해서 씁니다. 각 모델은 랜덤 시드만 다르게 해서 학습했습니다.

한글 기록은 여기서 한 번 더 확인합니다. 상위 5척만 골라 cross-encoder로 다시 채점하고, 그 점수의 0.25배를 랭커 점수에 더합니다. cross-encoder는 기록과 후보 이름을 한 문장으로 이어서 BERT에 넣기 때문에, 두 이름을 글자 단위로 함께 보고 판단할 수 있습니다. 대신 속도가 느려서 후보 전체가 아니라 상위 5척에만 적용했습니다.

`vessel_lib.py`의 `Matcher.predict()`

```python
sc = self.booster.predict(X)                               # 랭커 점수
if CE is not None and q.hangul and len(cands) > 1:
    top = np.argsort(-sc, kind="stable")[:CE_TOP]          # 상위 5척
    sc[top] += CE_W * CE.score(row.get("vessel_name"),     # + 0.25 × cross-encoder
                               self.idx.names[cands[top]])
best = str(self.idx.ids[cands[int(np.argmax(sc))]])        # 1위 선택
```

한글 모델과 영문 모델을 나누는 부분은 `inference/script.py`에 있습니다. 특징의 0번 칸에 한글 여부가 들어 있어서 그 값을 보고 모델을 고릅니다.

```python
class RowTypeSwitch:
    def predict(self, X):
        if self.other is not None and len(X) and X[0, 0] != 1.0:
            return self.other.predict(X)     # 영문 기록 → 영문 모델
        return self.hangul.predict(X)        # 한글 기록 → 한글 모델
```

### 5단계. 호출부호로 점검하기

train 데이터를 살펴보다가 호출부호와 선박이 정확히 1:1로 대응한다는 것을 발견했습니다. 호출부호 5,851개에 배 5,851척이었습니다. 이 규칙이 test에도 적용된다고 보고 세 가지 후처리를 넣었습니다.

첫째, train에 있는 호출부호가 나오면 train에서 그 호출부호를 쓰던 배로 바로 정합니다. 둘째, train에 없는 호출부호라면 train에서 이미 다른 호출부호를 쓰고 있던 배는 후보에서 뺍니다. 셋째, test 안에서 같은 호출부호를 가진 기록끼리는 확률을 합쳐서 같은 배로 정합니다.

다만 test 데이터에서도 이 규칙이 맞을지는 확신할 수 없었습니다. 그래서 규칙을 적용하기 전에, 정확도가 높은 영문 기록에서 규칙과 모델 예측이 얼마나 어긋나는지 먼저 세어 보게 했습니다. 어긋나는 비율이 30%를 넘으면 그 규칙은 적용하지 않습니다.

`inference/script.py`의 `callsign_postprocess()`

```python
clash = sum(preds[i] != callsign_map[r["call_sign"]] for i, r, d in seen_eng)
hard = CS_HARD and clash <= 2 + 0.30 * len(seen_eng)       # 영문 기록으로 먼저 검증
if hard:
    for i, r, d in seen:
        out[i] = callsign_map[r["call_sign"]]               # train 호출부호 → 그 배로 확정
...
if exclude:
    keep = [j for j, v in enumerate(ids) if v not in known] # train 배 제외
p = np.exp((sc - sc.max()) / POOL_T)                        # 같은 호출부호끼리 합칠 확률
```

이 후처리만 넣었을 때 LB가 0.95147에서 0.95206으로 올랐습니다. 영문 랭커 앙상블을 3개에서 9개로 늘린 것과 함께 적용한 최종 제출이 0.95240입니다.

---

## 학습 방법

### 랭커

일반적인 분류 모델은 후보 하나하나가 정답인지 아닌지를 따로 판단합니다. 저희는 기록 하나에 딸린 후보 전체를 한 묶음으로 보고, 그 안에서 정답이 1위가 되도록 학습시키는 LambdaRank를 썼습니다. 결국 필요한 것은 "가장 그럴듯한 한 척"이기 때문입니다.

평가식이 한글 0.7, 영문 0.3이라 학습 가중치도 같은 비율로 맞췄습니다. 또 "유사도가 높을수록 점수도 높아야 한다"는 단조 제약을 걸어서, 학습 데이터에 우연히 생긴 패턴을 따라가지 않도록 했습니다.

`training/train.py`

```python
PARAMS = dict(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
              lambdarank_truncation_level=10,
              monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES])

def weights(hang, groups):                       # 한글 0.7 / 영문 0.3
    w = np.where(hang, 0.7 / max(hang.sum(), 1), 0.3 / max((~hang).sum(), 1)) * len(hang)
    return np.repeat(w, groups)

def fit_ranker(Xs, ys, hang):
    X = np.vstack(...); y = np.concatenate(...)
    g = np.asarray([len(ys[i]) for i in keep])   # 기록마다 후보 수 = 묶음 크기
    return BoosterAverage([lgb.LGBMRanker(**(PARAMS | dict(random_state=42 + s)))
                           .fit(X, y, group=g, sample_weight=w) for s in range(N_SEEDS)])
```

### 데이터 늘리기

train에서 한글 기록은 1,191행밖에 없었습니다. 그래서 두 가지 방법으로 데이터를 늘렸습니다.

하나는 `training/augment.py`에서 일부러 오타를 내거나, 띄어쓰기를 바꾸거나, 톤수를 지우는 방식입니다. 다른 하나는 `llm_synthetic/gen_synth.py`에서 Qwen2.5-7B에게 train 선박명을 여러 방식의 한글로 다시 써 달라고 하는 방식입니다. 소리대로, 한자음으로, 뜻으로, 섞어서, 줄여서 쓰는 다섯 가지 방식을 썼습니다. 대회 규칙상 train 데이터를 바탕으로 한 증강만 허용되어서 그 범위 안에서만 만들었습니다.

### 신경망 두 개

LaBSE와 cross-encoder는 둘 다 Kaggle GPU에서 학습했습니다.

LaBSE는 이름을 각각 벡터로 만들어 두고 벡터끼리 비교하는 bi-encoder입니다. 명부 전체를 미리 벡터로 바꿔 둘 수 있어서 빠르고, 그래서 후보 검색에 썼습니다. (한글 기록, 영문 선박명) 쌍으로 대조 학습을 해서, 같은 배는 벡터가 가까워지고 다른 배는 멀어지게 했습니다.

cross-encoder는 두 이름을 한 번에 읽고 바로 점수를 냅니다. 더 정확하지만 느려서 상위 5척을 다시 채점할 때만 씁니다. 학습할 때는 랭커가 헷갈려 했던 후보들을 오답으로 함께 넣어서, 비슷한 이름끼리 구별하는 능력을 키웠습니다.

```python
# labse_finetune/ft_labse.py — 배치 안에서 i번째 기록의 정답은 i번째 이름
loss = F.cross_entropy(logits, torch.arange(len(vids)))

# cross_encoder/ce_train.py — 정답(0번)과 헷갈리는 후보들 중에서 정답을 고르도록
loss += F.cross_entropy(logits[o:o + len(grp)][None], torch.zeros(1, dtype=torch.long))
```

### 검증 방식

train에는 같은 배가 평균 두 번 정도 나옵니다. 기록 단위로 검증 데이터를 나누면, 검증에 나온 배를 학습 때 이미 본 셈이 되어 점수가 실제보다 높게 나옵니다. 그래서 배 단위로 fold를 나누는 GroupKFold(5)를 썼습니다. 신경망과 보조 사전도 fold마다 따로 만들었습니다.

그래도 로컬 점수는 0.99가 넘는데 LB는 그보다 한참 낮았습니다. 그래서 `training/stress_eval.py`를 따로 만들었습니다. 검증 데이터에 오타, 톤수 삭제, 국적 삭제 같은 손상을 일부러 넣고 다시 채점해서, 어떤 상황에서 성능이 떨어지는지 확인했습니다.

---

## 코드 위치 정리

```
inference/                  대회 서버에서 실제로 실행되는 코드
  script.py                 시작점 run(), 한글/영문 모델 분기, 호출부호 후처리
  model/vessel_lib.py       1~4단계 전부 (Query, retrieve, pair_features, Matcher)
  model/hanja.py            병음과 한자음 대응표
  model/emb_feat.py         LaBSE 벡터 생성
  model/ce_feat.py          cross-encoder 채점
training/
  train.py                  랭커 학습, 교차검증, 제출 파일 생성
  augment.py                오타·톤수 손상 등 증강
  stress_eval.py            손상 조건별 검증
labse_finetune/             LaBSE 파인튜닝
cross_encoder/              cross-encoder 학습
llm_synthetic/              LLM 한글 기록 생성
```

| 단계 | 파일 | 함수 (줄 번호) |
|---|---|---|
| 시작점 | `inference/script.py` | `run()` (160) |
| 1단계 | `inference/model/vessel_lib.py` | `Query` (394) |
| 2단계 | `vessel_lib.py` | `name_sims()` (432), `retrieve()` (482), `Matcher.candidates()` (698) |
| 3단계 | `vessel_lib.py` | `FEATURES` (526), `pair_features()` (595) |
| 4단계 | `vessel_lib.py`, `script.py` | `Matcher.predict()` (720), `RowTypeSwitch` (24) |
| 5단계 | `inference/script.py` | `callsign_postprocess()` (68) |
| 학습 | `training/train.py` | `fit_ranker()` (101) |

처음 코드를 볼 때는 `run()`에서 시작해서 `Matcher.predict()`, `candidates()`, `Query`, `retrieve()`, `pair_features()` 순서로 따라가면 흐름을 잡기 쉽습니다.

---

## 결과와 아쉬운 점

| | 점수 | 한글 정답 | 영문 정답 |
|---|---|---|---|
| 기본 (영문 랭커 3개) | 0.95147 | 504 / 532 | 3,429 / 3,568 |
| + 영문 랭커 9개 앙상블 | 0.95181 | 504 / 532 | 3,433 / 3,568 |
| + 호출부호 후처리 | 0.95206 | 504 / 532 | 3,436 / 3,568 |
| 최종 | 0.95240 | 504 / 532 (94.7%) | 3,440 / 3,568 (96.4%) |

정답 개수는 점수를 평가식(0.7 × 한글/532 + 0.3 × 영문/3,568)에 넣어 거꾸로 계산했습니다.

가장 아쉬운 부분은 한글입니다. 여러 방법을 시도했지만 한글 정답은 끝까지 504개에서 늘지 않았습니다. 틀린 28건을 보면 그중 16건 정도는 정답이 상위 5척 안에도 들어오지 못했습니다. 표기가 너무 자유로워서 후보를 추리는 단계에서 이미 놓친 것입니다. 다음에 다시 한다면 LLM으로 검색 키를 더 다양하게 만들어 보거나, 더 큰 cross-encoder를 써 보고 싶습니다.

---

## 예상 질의응답

**Q. 딥러닝 하나로 하지 않고 트리 모델을 중심에 둔 이유는?**

A. 한글 기록이 1,191개뿐이라 큰 모델을 처음부터 학습시키기 어려웠습니다. 발음이나 한자음, 톤수처럼 사람이 아는 규칙은 특징으로 직접 넣는 편이 더 확실했습니다. 그래서 신경망은 후보 검색과 재채점을 돕는 역할로 붙였습니다.

**Q. 분류가 아니라 순위 학습을 쓴 이유는?**

A. 정답 후보가 78,595척이라 배마다 클래스를 둘 수 없었습니다. 또 저희에게 필요한 건 각 후보의 정답 확률보다 "누가 1위인가"였습니다. LambdaRank는 1위가 바뀌는 실수를 더 크게 벌점으로 주기 때문에 이 목적에 맞았습니다.

**Q. LaBSE와 cross-encoder를 둘 다 쓴 이유는?**

A. LaBSE는 빠르지만 덜 정확하고, cross-encoder는 정확하지만 느립니다. 그래서 넓은 범위는 LaBSE로 훑고, 좁혀진 상위 5척만 cross-encoder로 확인했습니다.

**Q. 호출부호 규칙이 test에서 안 맞으면?**

A. 규칙마다 영문 기록으로 먼저 검증합니다. 어긋나는 비율이 30%를 넘으면 그 규칙은 적용하지 않고 모델 예측을 그대로 씁니다.

**Q. 과적합은 어떻게 막았나?**

A. 배 단위로 fold를 나눠 검증했고, 보조 사전과 신경망도 fold마다 따로 만들었습니다. 단조 제약을 걸었고, 일부러 데이터를 손상시킨 스트레스 테스트로도 확인했습니다.
