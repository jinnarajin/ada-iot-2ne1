# 이번 주 팀 과제

## 일정

- 마감: **2026년 9월 15일 화요일**
- 대상: 팀원 전원
- 제출 위치: 이 저장소의 개인 작업 브랜치 또는 Pull Request

## 과제 1: 팀원별 떨림 분류기 1개

현재 팀이 만든 데이터로 각자 서로 다른 분류 모델을 하나씩 구현하고 평가한다.
이미 구현한 Random Forest는 선택 대상에서 제외한다.

선택 가능한 모델 예시:

- Logistic Regression
- RBF-SVM
- XGBoost 또는 LightGBM
- k-Nearest Neighbors
- 작은 1D CNN
- LSTM 또는 GRU
- raw signal과 주파수 표현을 결합한 모델

중복 모델을 피하기 위해 작업 시작 전에 담당 모델을 팀에 공유한다. 모델별 비교가
가능하도록 기존 전처리와 평가 규칙을 그대로 사용한다.

필수 조건:

- 입력은 가속도계와 자이로스코프 데이터를 사용함
- Random Forest를 사용하지 않음
- 50 Hz 리샘플링과 3초 window를 기본 조건으로 사용함
- 같은 recording의 window를 train과 test에 나누어 넣지 않음
- 사람 단위 Leave-One-Subject-Out 평가를 수행함
- balanced accuracy, sensitivity, specificity와 macro F1을 기록함
- 실행 방법과 필요한 패키지를 문서에 작성함

제출물:

1. 실행 가능한 학습·평가 코드
2. 모델과 전처리 방법을 설명하는 짧은 문서
3. fold별 및 평균 평가 결과
4. 기존 Random Forest 결과와의 간단한 비교
5. 잘못 분류한 사례와 한계에 대한 짧은 해석

## 과제 2: 팀원별 오픈소스 떨림 데이터셋 1개

각자 현재 저장소에 없는 오픈소스 떨림 데이터셋을 하나씩 조사해 가져온다.
다른 팀원과 같은 데이터셋을 선택하지 않는다.

데이터셋 선정 조건:

- 실제로 다운로드하거나 접근할 수 있음
- 가속도계 또는 자이로스코프 원시 시계열을 포함함
- tremor와 non-tremor를 구분할 수 있는 라벨 또는 근거가 있음
- 참가자, 세션 또는 recording을 구분할 수 있음
- 연구·재배포에 적용되는 라이선스와 이용 조건을 확인할 수 있음

데이터를 저장소에 직접 커밋하기 전 라이선스와 파일 크기를 확인한다. 재배포가
허용되지 않거나 데이터가 크면 원본 파일을 올리지 않고 다운로드 링크와 준비
스크립트만 제출한다. 환자 데이터인지 건강한 참가자의 모사 떨림인지 반드시
구분해서 기록한다.

제출물:

1. 데이터셋 이름, 공식 출처 URL과 관련 논문
2. 라이선스 및 재배포 가능 여부
3. 참가자 수, 라벨, 센서 종류, sampling rate와 recording 수
4. 파일 형식과 주요 column 설명
5. 다운로드 또는 준비 방법
6. 현재 공통 CSV schema로 변환하는 코드 또는 구체적인 변환 계획
7. 데이터 품질과 현재 데이터셋에 추가했을 때의 장점·한계

## 공통 제출 구조

각 팀원은 다음 구조를 권장 형식으로 사용한다.

```text
experiments/<name>/<model_name>/
  train.py
  README.md
  results.csv

data_sources/<name>/<dataset_name>/
  README.md
  download_or_prepare.py
```

원본 데이터는 라이선스와 용량을 확인하기 전까지 커밋하지 않는다. 생성된 대형
모델 파일, 가상환경, 캐시 파일도 커밋하지 않는다.

## 담당 현황

팀원 이름과 담당 모델·데이터셋은 중복을 막기 위해 정한 뒤 아래 표에 추가한다.

| 팀원 | 분류 모델 | 오픈소스 데이터셋 | 상태 |
| --- | --- | --- | --- |
| seojin | [k-Nearest Neighbors](../experiments/seojin/knn/README.md) | [PADS - Parkinsons Disease Smartwatch dataset](../data_sources/seojin/pads/README.md) | 분류기 완료, PADS 변환 코드 완료 |

### seojin 진행 내역 (2026-09-15)

- 과제 1: k-NN (k=5, distance weighting, B2 특징, 3초/50 Hz). LOSO 평균 balanced
  accuracy 0.950 (RF 0.992), cross-dataset 평균 0.976 (RF 0.990). 오분류는 전부 march의
  저진폭 simulated tremor. 실행법·fold별 결과·해석은
  [experiments/seojin/knn/README.md](../experiments/seojin/knn/README.md).
- 과제 2: PADS 조사 완료, 다운로드·변환·학습 완료. `data_sources/seojin/pads/sync_s3.py`가
  aws CLI 없이 S3 미러에서 `patients/`·`movement/`를 병렬로 내려받고(469명, 10,318
  recording, 1.2 GB), `scripts/prepare_pads.py`가 TXT를 공통 CSV schema로 변환한다
  (100 Hz, 첫 0.5초 vibration cue 제거, gyro rad/s→deg/s). 라벨 규칙: `Healthy` →
  non_tremor(79명), `Essential Tremor` 또는 `disease_comment`에 tremor/mixed type 언급 →
  tremor(193명), 나머지 무진전형 PD·근긴장이상·MS 등 197명은 떨림 근거가 없어 제외.
  결과는 `data/pads_manifest.csv`, `results/pads_recording_features.csv`(272명, 5,984
  recording). 변환된 CSV와 raw는 `.gitignore`로 제외된다.
- 과제 2 결과: 기존 RF B2 분류기를 PADS로 학습. 5-fold GroupKFold(subject-independent)
  recording AUC 0.79, balanced accuracy 0.72(임계값 0.68, 중첩 CV로 선택), subject 단위
  AUC 0.91 / balanced accuracy 0.79. 자체 dataset_a/b에 적용 시 balanced accuracy 0.87.
  task별로는 StretchHold·CrossArms 등 정적 과제가 0.85 이상, Entrainment 0.63으로
  kinetic 과제에서 healthy 동작을 tremor로 오탐하는 것이 주요 오류. 임계값은
  `tremor_classifier.py train --threshold`로 모델 아티팩트에 저장되며 `predict`가 사용한다.
- 추가로 팀원이 나눠 맡을 후보 6개를
  [data_sources/seojin/CANDIDATE_DATASETS_KO.md](../data_sources/seojin/CANDIDATE_DATASETS_KO.md)에
  정리했다. 추천: Levodopa Response Study(Synapse), Monipar(Zenodo, CC-BY, 35 MB).

## 완료 기준

과제는 코드나 링크만 공유하면 끝난 것으로 보지 않는다. 다른 팀원이 README의
명령을 실행해 결과를 재현할 수 있고, 데이터셋의 사용 조건과 라벨 의미를 확인할
수 있어야 완료로 처리한다.
