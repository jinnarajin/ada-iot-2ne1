# AdaBoost baseline

`scripts/run_nonlinear_baselines.py`에 AdaBoost를 추가했다.
사용자가 작성한 설정인 `n_estimators=200`, `learning_rate=0.5`,
`random_state=42`를 사용하며, StandardScaler는 각 학습 fold에만 fit한다.
기존 B1 시간 특징과 B2 시간·주파수 특징, LOSO 및 양방향 cross-dataset
평가를 그대로 사용한다. 비교 그래프에도 AdaBoost를 표시한다.

## 실행

저장소 루트에서 Python 3.12로 실행한다.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_nonlinear_baselines.py
```

입력은 저장소의 `results/recording_features_3s.csv`이며, 그래프의 Logistic
Regression 비교에는 `results/baseline_metrics.csv`를 사용한다.
결과는 `results/nonlinear_metrics.csv`, `results/nonlinear_predictions.csv`,
`results/nonlinear_baselines.png`로 저장된다. 이 명령은 기존 결과를 덮어쓴다.

## 검증

로컬 scikit-learn 1.6.1 환경에서 기존 특징 파일로 전체 평가와 그래프 생성을
확인했다. 저장소의 고정 버전은 1.7.1이며, 고정 버전 환경에서의 재현은 별도로
확인해야 한다. 전체 48개 평가 행 중 AdaBoost는 16개이며 예측 확률은 모두
0~1 범위였다. 검증 출력은 별도 임시 폴더에 저장했다.

| 평가 | 특징 | AdaBoost 평균 balanced accuracy | Random Forest |
| --- | --- | ---: | ---: |
| LOSO | B1 | 0.9875 | 0.9792 |
| LOSO | B2 | 1.0000 | 0.9917 |
| Cross-dataset | B1 | 0.9839 | 0.9804 |
| Cross-dataset | B2 | 1.0000 | 0.9902 |

6명의 소규모 모사 떨림 데이터에 대한 결과이므로 실제 환자나 새로운 환경에서의
성능으로 일반화할 수 없다. PADS 샘플은 이 평가에 사용하지 않는다.
