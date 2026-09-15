# Histogram Gradient Boosting 떨림 분류기

Random Forest와 다른 boosting 계열의 `HistGradientBoostingClassifier`를 사용한다.
입력은 팀 데이터의 가속도계·자이로스코프를 50 Hz로 리샘플링하고 3초 window에서
계산한 B2 시간·주파수 특징이다. window 특징의 median/IQR을 recording별 한 행으로
집계한 뒤 사람 단위 LOSO로 평가하므로 한 recording이 train/test에 나뉘지 않는다.

## 실행

저장소 루트에서 Python 3.12 환경을 만든 후 실행한다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_baseline_analysis.py
.venv/bin/python experiments/hmuzn/hist_gradient_boosting/train.py
```

`results.csv`에는 6개 LOSO fold와 평균 balanced accuracy, sensitivity,
specificity, macro F1이 기록된다. `predictions.csv`는 오분류 recording 분석용이다.

실행 결과 평균은 balanced accuracy **0.9917**, sensitivity **0.9833**,
specificity **1.0000**, macro F1 **0.9916**이다. `march` fold의 tremor recording
`dataset_a:imu_20260721_222446` 한 건을 non-tremor로 오분류했다(확률 0.0022).

## 비교와 한계

동일 B2 특징의 기존 Random Forest 결과는 `results/nonlinear_metrics.csv`와 비교한다.
현재 Random Forest B2의 LOSO 평균도 네 지표가 각각 0.9917, 0.9833, 1.0000,
0.9916으로 동일하므로 이 작은 데이터에서는 우열을 주장할 수 없다.

## PADS 파일럿 학습

PADS 변환본이 `data/processed/dataset_c_pads/`에 있으면 다음 명령으로 동일 모델을
학습·평가한다.

```bash
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python \
  experiments/hmuzn/hist_gradient_boosting/train_pads.py
```

PADS 라벨은 recording/window에서 직접 관찰한 떨림 정답이 아니라 참가자 진단에서
도출한 weak label이다. 따라서 참가자를 train/test에 섞지 않고 두 라벨의 참가자를
각 fold에 균등 배분하는 5-fold 평가를 사용하며, 결과를 임상적 순간 떨림 탐지
성능으로 해석하지 않는다. 최종 모델은 `models/hgb_pads_strict.pkl`에 생성되며
git에는 커밋되지 않는다.

전체 strict 코호트는 160명(weak-label tremor 81명, healthy 79명), 1,600개
recording으로 구성된다. 5-fold 평가 결과와 해석은 `pads_results/summary.json`과
`pads_results/fold_metrics.csv`에 기록한다.
모델 hyperparameter는 고정했으며 test fold로 조정하지 않았다. 참가자가 6명뿐이고
모두 건강한 참가자의 모사 떨림이므로 임상적 tremor 탐지 성능으로 해석할 수 없다.
또한 recording 단위 집계는 떨림이 간헐적으로 나타나는 시점을 숨길 수 있다.
