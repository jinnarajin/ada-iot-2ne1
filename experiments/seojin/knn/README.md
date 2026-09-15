# k-Nearest Neighbors 떨림 분류기 (seojin)

## 모델

`StandardScaler` → `KNeighborsClassifier(n_neighbors=5, weights="distance")`.
k와 가중치는 실행 전에 고정했고 test 결과로 조정하지 않았다.
scaler는 매 fold의 training 데이터에만 적합한다.

## 전처리와 평가

기존 파이프라인을 그대로 사용한다. 입력은 `results/recording_features_3s.csv`
(50 Hz 리샘플링, 3초 window 50% overlap, accelerometer + gyroscope, B2 time+frequency
특징을 recording 단위로 median/IQR 집계). window가 아니라 recording 단위로 예측하므로
같은 recording의 window가 train/test에 나뉘는 일은 없다.

- 6-fold Leave-One-Subject-Out
- cross-dataset 양방향 (dataset_a → dataset_b, dataset_b → dataset_a)
- 지표: balanced accuracy, sensitivity, specificity, macro F1 (+ AUROC, AUPRC)

## 실행

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_sensitivity_analysis.py   # recording_features_3s.csv 생성 (이미 있으면 생략)
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python experiments/seojin/knn/train.py
```

패키지는 저장소 `requirements.txt`와 동일하다 (numpy, pandas, scikit-learn). 추가 의존성 없음.
`--check` 플래그로 fold 개수와 확률 범위를 검사한다.

출력: `results.csv` (fold별 지표), `predictions.csv` (recording별 확률).

## 결과

### Leave-One-Subject-Out

| fold | balanced acc | sensitivity | specificity | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| BIBI | 1.000 | 1.000 | 1.000 | 1.000 |
| anna | 1.000 | 1.000 | 1.000 | 1.000 |
| euno | 1.000 | 1.000 | 1.000 | 1.000 |
| march | 0.700 | 0.400 | 1.000 | 0.670 |
| mu | 1.000 | 1.000 | 1.000 | 1.000 |
| seojin | 1.000 | 1.000 | 1.000 | 1.000 |
| **평균** | **0.950** (std 0.122) | 0.900 | 1.000 | 0.945 |

### Cross-dataset

| 방향 | balanced acc | sensitivity | specificity | macro F1 |
| --- | ---: | ---: | ---: | ---: |
| A → B | 0.988 | 0.975 | 1.000 | 0.988 |
| B → A | 0.964 | 0.943 | 0.986 | 0.964 |

## Random Forest와 비교 (B2, 3초)

| 분류기 | LOSO 평균 | LOSO std | cross-dataset 평균 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.992 | 0.020 | 0.990 |
| RBF-SVM | 0.992 | 0.020 | 0.977 |
| Logistic Regression | 0.975 | 0.061 | 0.989 |
| **kNN (k=5)** | 0.950 | 0.122 | 0.976 |

kNN은 네 모델 중 가장 낮고, 차이는 거의 전부 march fold 한 곳에서 나온다.

## 오분류 해석과 한계

- LOSO 오류 6건은 모두 march의 simulated tremor를 non-tremor로 판정한 것이다
  (확률 0.20~0.41). march는 다른 참가자보다 떨림 진폭이 작아 특징 공간에서
  non-tremor 군집에 가깝고, kNN은 training set에 march와 비슷한 사람이 없으면
  가장 가까운 이웃이 그대로 non-tremor가 된다. 트리·선형 모델은 주파수 특징 하나로
  경계를 그을 수 있지만 kNN은 모든 특징의 거리를 균등하게 쓰므로 진폭 계열 특징에
  끌려간다. RF도 같은 fold에서 1건 틀린 것과 같은 원인이다.
- B → A에서도 march 4건 + seojin non-tremor 1건 (0.595)이 틀렸다. 표본 40개짜리
  dataset_b만으로는 이웃 밀도가 부족하다.
- 참가자 6명, recording 220개라 fold 하나가 평균을 크게 흔든다. 순위 차이를 과해석하면 안 된다.
- kNN은 학습 데이터 전체를 들고 있어야 하므로 워치 온디바이스 추론에는 불리하다.
