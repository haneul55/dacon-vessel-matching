# 공부 계획: 모델 구조 + 코드

목표: 2027년 3월 면접에서 이 프로젝트를 **그림 없이 3분 안에 설명**하고, **코드 질문에 파일과 함수 이름으로 답할 수 있게** 되는 것입니다.
모델 구조 설명은 [model_structure.md](model_structure.md)에 있습니다. 이 문서는 무엇을 어떤 순서로 공부할지 정리한 체크리스트입니다.

| 주차 | 주제 | 끝나면 할 수 있는 것 |
|---|---|---|
| 1주 | 개념 공부 | 각 기법이 왜 필요한지 말로 설명하기 |
| 2주 | 추론 코드 읽기 (`inference/`) | 기록 한 줄이 어떤 함수를 거치는지 따라가기 |
| 3주 | 학습 코드 읽기 (`training/`, 신경망) | 모델을 어떻게 학습하고 검증했는지 설명하기 |
| 4주 | 직접 돌려 보기 + 면접 준비 | 데모 실행, 예상 질의응답 연습하기 |

---

## 1주차: 개념 공부

코드를 열기 전에 아래 개념부터 익혀 두십시오. 각 항목 아래의 질문에 답할 수 있으면 다음으로 넘어가도 됩니다.

### 1-1. 전체 구조
- [ ] **Retrieve → Rerank 구조**
  - 질문: 왜 78,595척을 전부 모델에 넣지 않고 후보를 먼저 줄입니까?
- [ ] **Entity matching / record linkage**
  - 질문: 서로 다른 데이터에서 같은 대상을 찾는 문제의 어려운 점은 무엇입니까?

### 1-2. 문자열 비교
- [ ] **Levenshtein 거리, Jaro-Winkler, token set ratio**
  - 질문: `GRAND VICCTORY`와 `GRAND VICTORY`의 편집 거리는 얼마입니까?
- [ ] **rapidfuzz 사용법** (`fuzz.ratio`, `process.cdist`)
- [ ] **한글 자모 분해**
  - 질문: 유니코드에서 `금`을 `ㄱ ㅡ ㅁ`으로 나누는 계산식은 무엇입니까?

### 1-3. Learning to Rank (가장 중요)
- [ ] **pointwise / pairwise / listwise의 차이**
- [ ] **NDCG 지표**
- [ ] **LambdaRank**
  - 질문: 왜 분류(정답이다/아니다)가 아니라 순위로 학습했습니까?
- [ ] **Gradient Boosting과 LightGBM 기본 원리**
- [ ] **monotone constraint**
  - 질문: 이 제약이 과적합을 어떻게 막습니까?

### 1-4. 신경망
- [ ] **BERT와 [CLS] 토큰**
- [ ] **Bi-encoder vs Cross-encoder** (면접 단골 질문)
  - 질문: 어느 쪽이 빠르고, 어느 쪽이 정확합니까? 이 프로젝트에서는 각각 어디에 썼습니까?
- [ ] **대조 학습(contrastive learning)과 in-batch negative**
- [ ] **hard negative mining**
- [ ] **LaBSE가 다국어 문장을 같은 공간에 놓는 원리**

### 1-5. 검증
- [ ] **GroupKFold와 데이터 누수**
  - 질문: 기록 단위로 KFold를 하면 점수가 왜 부풀려집니까?
- [ ] **OOF(out-of-fold) 예측**
  - 질문: cross-encoder 학습에 랭커의 OOF 후보를 쓴 이유는 무엇입니까?

추천 자료 (검색어): `LambdaMART explained`, `LightGBM lambdarank`, `sentence-transformers cross-encoder vs bi-encoder`, `LaBSE paper`, `GroupKFold data leakage`

---

## 2주차: 추론 코드 읽기 (`inference/`)

**읽는 순서는 실행 순서와 같습니다.** 진입점에서 시작해 함수를 하나씩 따라 들어가십시오.

### 2-1. 진입점 — `inference/script.py` (183줄)
- [ ] `run()` (160줄): 파일을 읽고, 예측하고, 저장하는 전체 흐름
- [ ] `load_matcher()` (38줄): 랭커, 임베딩, cross-encoder를 불러오는 부분
- [ ] `RowTypeSwitch` (24줄): 한글 행과 영문 행에 서로 다른 랭커를 쓰는 방법
- [ ] `callsign_postprocess()` (68줄): 호출부호 규칙 3개와 자기 검증 게이트
  - 연습: 게이트 조건 `clash <= 2 + 0.30 * len(...)`이 무슨 뜻인지 한 줄로 써 보기

### 2-2. 핵심 라이브러리 — `inference/model/vessel_lib.py` (763줄)
이 파일이 모델의 80%를 차지합니다. 한 번에 다 읽으려 하지 말고 아래 순서로 나누어 읽으십시오.

| 순서 | 위치 | 내용 | 그림 단계 |
|---|---|---|---|
| 1 | `Matcher.predict()` 720줄 | 전체 흐름. 이것부터 읽기 | 전체 |
| 2 | `Matcher.candidates()` 698줄 | 검색 → 특징 → 행렬 | B~D |
| 3 | `class Query` 394줄 | 기록 한 줄을 키로 바꾸기 | A |
| 4 | `clean_hangul` 73줄, `romanize` 95줄, `jamo` 279줄 | 정규화 세부 | A |
| 5 | `class CatalogIndex` 302줄 | 카탈로그를 미리 색인하기 | B |
| 6 | `name_sims()` 432줄, `retrieve()` 482줄 | 후보 검색 | B |
| 7 | `FEATURES` 526줄, `pair_features()` 595줄 | 특징 68개 | D |
| 8 | `BoosterAverage` 665줄 | 랭커 평균 | E |
| 9 | 737~739줄 | cross-encoder 재채점 (`CE_TOP`, `CE_W`) | E |

- [ ] 연습: `금성8호` 하나를 손으로 따라가며 `Query`가 만드는 키를 종이에 적어 보기
- [ ] 연습: `FEATURES` 68개를 1주차 표의 묶음(철자/발음/톤수…)으로 직접 분류해 보기

### 2-3. 보조 모듈
- [ ] `inference/model/hanja.py`: 병음 → 한자음 표와 `segment()`
- [ ] `inference/model/emb_feat.py`: `Embedder.vec()` (43줄). 한글 기록을 벡터로 바꾸는 부분
- [ ] `inference/model/ce_feat.py` (32줄): `CrossScorer.score()`. 짧으니 한 줄씩 다 이해하기

---

## 3주차: 학습 코드 읽기

### 3-1. 랭커 학습 — `training/train.py` (287줄)
- [ ] `main()` (197줄): 전체 학습 흐름
- [ ] `use_fold()` 47줄, `fit_aux()` 75줄: fold마다 따로 만드는 보조 정보 (누수 방지)
- [ ] `build()` (85줄): 학습용 (특징 행렬, 정답 라벨, 그룹) 만들기
- [ ] `make_aug()` (131줄): 증강 데이터를 섞는 방법
- [ ] `weights()` (96줄): 평가식 0.7 / 0.3을 학습 가중치로 옮긴 방법
- [ ] `PARAMS` (55줄), `fit_ranker()` (101줄): LambdaRank 설정
  - 연습: `group=g`가 무엇을 뜻하는지 설명해 보기
- [ ] `score()` (192줄): CV 점수 계산

### 3-2. 증강과 검증
- [ ] `training/augment.py` (112줄): `hangul_jamo_typo()`, `gt_noise()`, `no_gt()`
- [ ] `training/stress_eval.py`: `CONDITIONS`와 `hard_mix()`. 어떤 손상을 넣어서 시험했는지
- [ ] `analysis/stress_summary_K2.csv`: 결과표 읽는 법 (조건별 한글/영문 정확도)

### 3-3. 신경망 학습 (Kaggle GPU)
- [ ] `labse_finetune/ft_labse.py`
  - `train_encoder()` (108줄): 135줄 `cross_entropy(logits, arange(...))`가 in-batch 대조 학습입니다
  - `hard_negative()` (98줄): 헷갈리는 배를 같은 배치에 넣는 방법
- [ ] `cross_encoder/ce_train.py`
  - `class CrossEncoder` (36줄): BERT + 선형 헤드
  - `train_ce()` (62줄): 90줄의 listwise softmax 손실(정답 1개 vs 오답 여러 개)
- [ ] `llm_synthetic/gen_synth.py`: LLM 프롬프트와 표기 스타일 5종 (`STYLES`)

---

## 4주차: 직접 돌려 보기 + 면접 준비

### 4-1. 실행 준비
가중치는 저장소에 없으므로 PC에 있는 최종 zip에서 꺼내야 합니다.
1. `DS데이_선박매칭_검색랭킹모델.zip` (이전 이름 `9_submit_F_cs_Jfam_hard_v2.zip`)의 `model/` 폴더 내용을 `inference/model/`에 풀기
   - 가중치 파일은 `.gitignore`에 들어 있어서 실수로 올라가지 않습니다.
2. 대회 데이터(`train.csv`, `vessel_catalog.csv`)가 있는 폴더를 실행할 때 `--data <폴더>`로 알려 주기
   - 내 PC 기준 위치: `데이콘 경진대회 (진행중)/open_data/open`

### 4-2. 해 볼 것
- [ ] `python training/demo_model.py --name "테르모필래" --top 8`
  - 기록 하나의 후보와 점수를 화면에 보여 줍니다. 구조를 이해하는 데 가장 좋습니다.
  - 이 데모는 예전 모델(H4) 기준으로 만든 것이어서, 최종 모델 파일과 맞지 않으면 오류가 날 수 있습니다.
- [ ] `predict()` 안에 `print`를 넣어 후보 수, 1위 점수, cross-encoder 점수 차이를 직접 찍어 보기
- [ ] 특징 하나(예: `emb_sim`)를 0으로 바꿨을 때 예측이 어떻게 바뀌는지 보기 (특징 중요도 체감)
- [ ] LightGBM `feature_importance()`로 중요한 특징 상위 10개 뽑아 보기

### 4-3. 면접 준비
- [ ] 3분 설명 대본 쓰기: 문제 → 어려운 점 → 구조(A~F) → 결과 → 한계
- [ ] [model_structure.md](model_structure.md)의 예상 질의응답 4개를 소리 내어 연습하기
- [ ] "내가 한 부분"과 "팀원이 한 부분"을 분명히 나눠서 정리하기 (반드시 나오는 질문입니다)
- [ ] 실패한 시도 3개와 거기서 배운 점 정리하기. 예시:
  - 한자 표 확장(K6): 로컬 검증에서는 좋았지만 LB에서는 떨어졌습니다.
  - 키별 검색 항상 켜기(G4): 비슷한 이름이 늘어나 오히려 손해였습니다.
  - 톤수 노이즈 대응(G2): LB 하락의 원인이 아니었습니다.

---

## 핵심 숫자 (외워 두기)

| 항목 | 값 |
|---|---|
| 카탈로그 | 78,595척 |
| 후보 수 | 평균 74척 + 의미 검색 최대 50척 |
| 특징 | 68개 |
| 랭커 | LightGBM LambdaRank, 트리 600개, 한글 3개 / 영문 9개 평균 |
| Cross-encoder | 한글 행 상위 5개, 가중치 0.25 |
| 검증 | 선박 단위 GroupKFold(5) |
| test 구성 | 한글 532행, 영문 3,568행 |
| 최종 LB | 0.95240 (한글 504/532, 영문 3,440/3,568) |
| 점수 변화 | 0.95147 → 영문 앙상블 0.95181 / 후처리 0.95206 → 둘 다 0.95240 |
