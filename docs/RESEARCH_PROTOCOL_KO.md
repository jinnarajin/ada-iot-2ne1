# 연구 프로토콜

## 연구 목적

서로 다른 샘플링 조건에서 수집된 소규모 IMU 데이터에서 주파수 정보를 명시적으로 표현하는 것이 참가자 변화와 데이터셋 수집 조건 변화에도 강건한 모사 떨림 탐지 성능을 향상하는지 확인한다.

핵심 연구 질문은 다음과 같다.

> 다른 sampling condition에서 수집된 소규모 IMU 데이터에 대해, 주파수 정보를 명시적으로 사용하는 모델이 subject-independent 및 cross-dataset simulated tremor detection을 개선하는가?

## 확인적 가설

- **H1 - 참가자 독립 성능:** 시간 영역 기준 모델에 주파수 특징을 추가하면 Leave-One-Subject-Out 평균 balanced accuracy가 향상된다.
- **H2 - 데이터셋 간 일반화:** 주파수 특징을 추가하면 두 방향의 cross-dataset balanced accuracy 평균이 향상된다.
- **H3 - 센서 기여도:** 자이로스코프 특징은 가속도계 특징과 상호 보완적인 정보를 제공한다.
- **H4 - 집계 방법:** 떨림 증거가 시간적으로 간헐적인 경우에 한해 attention pooling이 mean pooling보다 recording-level 성능을 향상한다.

H1과 H2를 주가설로, H3과 H4를 부가 분석으로 설정한다.

## 분석 단위

- **Recording:** 약 20초 길이의 CSV 파일 하나로, 데이터 분할과 최종 평가의 기본 단위이다.
- **Instance:** 특징 추출 또는 MIL 입력에 사용하는 3초 길이의 윈도우이다. 윈도우 간 중첩률은 50%로 설정한다.
- **Bag:** 하나의 recording에서 생성된 유효한 모든 윈도우의 집합이다.

서로 중첩된 윈도우는 상관된 관측값이며 독립 표본이 아니다. 따라서 최종 신뢰구간과 통계 검정은 윈도우가 아니라 recording 또는 participant 단위로 수행해야 한다.

## 전처리

1. CSV 스키마와 타임스탬프의 단조 증가 여부를 검증한다.
2. 가속도와 자이로스코프 채널을 공통 50 Hz 시간축으로 리샘플링한다.
3. 누락 구간을 기록하고, 사전에 정한 기준보다 큰 간격을 가로지르는 윈도우는 제거하거나 마스킹한다. 초기 기준은 100 ms로 설정한다.
4. 각 raw axis에서 윈도우별 평균을 제거한다.
5. 가속도계 축과 자이로스코프 축을 서로 다른 입력 그룹으로 취급한다. 중심화 이후 vector magnitude를 파생한다.
6. 확인적 실험에서는 `angle_*_deg`를 제외한다. 해당 값은 sensor fusion으로 생성된 방향 의존적 신호이며 각도 경계에서 불연속이 발생할 수 있다.
7. 스케일링과 학습되는 모든 전처리는 training data만 사용해 적합한다.

데이터셋 ID, 참가자 ID, 파일명, 타임스탬프 패턴, 원래 샘플링 속도는 분류기 입력으로 제공하지 않는다.

## 명시적 주파수 표현

균일하게 리샘플링한 각 센서의 축 및 magnitude 신호에 Welch power spectral density를 적용한다. 사전에 지정하는 주파수 특징은 다음과 같다.

- 3-12 Hz 구간의 log power
- 3-12 Hz 구간의 dominant frequency
- peak prominence 또는 peak-to-band-power ratio
- spectral entropy
- 3-6 Hz, 6-9 Hz, 9-12 Hz 구간의 power ratio

3-12 Hz 구간은 두 데이터셋에서 관측된 서로 다른 모사 떨림 주파수를 포괄하기 위해 사용한다. 다른 주파수 대역은 test 성능을 보고 선택하지 않으며, sensitivity analysis로만 보고한다.

## 모델 비교 단계

### B0 - 진폭 기반 sanity baseline

가속도 RMS와 자이로스코프 RMS만 사용하는 logistic regression을 학습한다. 이 모델은 모사 떨림의 단순 진폭 차이만으로 과제가 얼마나 설명되는지 확인하기 위한 기준이다.

### B1 - 시간 영역 baseline

RMS, variance, median absolute deviation, jerk RMS, zero-crossing 특징을 사용해 regularized logistic regression 또는 얕은 tree ensemble을 학습한다.

### B2 - 명시적 주파수 baseline

B1과 동일한 estimator 및 tuning budget을 사용하되, 사전에 지정한 주파수 특징을 추가한다. B1과 B2의 통제된 비교를 H1과 H2의 주검정으로 사용한다.

### D1 - 소형 raw-signal 모델

작은 dual-branch 1D CNN을 사용한다. 가속도계 윈도우와 자이로스코프 윈도우를 각각 처리한 다음 특징을 결합한다.

### D2 - dual-view frequency-aware 모델

Raw window encoder와 log-PSD 또는 spectrogram encoder의 표현을 결합한다. 모델 용량을 제한하고 파라미터 수를 보고한다.

### D3 - attention MIL

D1 또는 D2의 window encoder 뒤에 gated-attention pooling을 적용해 recording-level 결과를 출력한다. 같은 encoder를 사용하는 mean pooling 및 max pooling과 비교한다. Attention weight는 설명을 돕는 진단 자료이며 인과적 중요도의 증거로 해석하지 않는다.

## 평가 방법

### 참가자 독립 평가

6개의 Leave-One-Subject-Out fold를 사용한다. Held-out participant의 모든 recording과 window를 test data로 지정한다. Hyperparameter는 남은 참가자만 사용하는 nested grouped validation으로 선택하거나 실험 전에 고정해야 한다.

### 데이터셋 간 평가

다음 두 방향을 독립적으로 평가한다.

- `dataset_a`에서 학습 및 튜닝하고 `dataset_b`에서 한 번 평가한다.
- `dataset_b`에서 학습 및 튜닝하고 `dataset_a`에서 한 번 평가한다.

각 데이터셋 전체의 통계로 데이터셋별 정규화를 수행해서는 안 된다. 모든 정규화 통계는 해당 방향의 training dataset에서만 계산한다.

### 평가지표

주평가지표는 recording-level balanced accuracy이다.

다음 보조 지표도 함께 보고한다.

- sensitivity
- specificity
- macro F1
- AUROC
- AUPRC
- confusion matrix
- participant별 결과

가능한 경우 participant 단위로 재표집한 bootstrap confidence interval을 제시한다. 참가자가 6명뿐이므로 점근적 유의확률보다 effect size, fold별 결과 및 불확실성을 중심으로 해석한다.

## 필수 ablation 및 통제 실험

- 가속도계만 사용, 자이로스코프만 사용, 두 센서 모두 사용
- 시간 특징만 사용, 시간 특징과 주파수 특징을 함께 사용
- raw axis 사용, 중심화된 vector magnitude와 raw axis 함께 사용
- 2초, 3초, 5초 윈도우의 sensitivity analysis
- 동일한 encoder에서 mean, max, attention pooling 비교
- 잔존하는 domain information을 측정하기 위한 dataset-ID prediction control
- label shuffling 및 majority-class sanity check

## 결과 해석의 범위

이 데이터의 모든 떨림은 건강한 참가자가 자발적으로 모사한 것이다. 따라서 긍정적인 결과는 이 데이터가 나타내는 수집 조건에서 모사 떨림을 탐지하는 방법의 강건성을 지지한다.

다음과 같은 임상적 결론을 지지하지 않는다.

- 파킨슨병 또는 본태성 떨림 진단
- 실제 병적 떨림 탐지 성능
- 떨림 중증도 평가
- 임상 환경에서의 유효성

실제 병적 떨림으로 확장하려면 환자 데이터, 임상 기준 라벨, 다양한 일상 동작 및 독립적인 외부 검증 데이터가 추가로 필요하다.
