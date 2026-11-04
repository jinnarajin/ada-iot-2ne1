# ABMIL 떨림 분류기 (seojin)

Ilse et al. (2018) Attention-based Deep MIL을 손목 IMU 떨림 탐지에 적용한 실험.
bag 하나에 라벨 하나만 주고, 어느 2초 윈도우가 떨림 근거인지는 모델이 attention으로 찾는다.

| 데이터 | bag | instance | 라벨 |
| --- | --- | --- | --- |
| 자체 모사 떨림 (dataset_a/b, 6명 220 recording) | 20초 녹음 1개 | 2초 윈도우 19개 (1초 hop, 50 Hz) | simulated_tremor / non_tremor |
| PADS (272명) | 피험자 1명 = 11과제 × 양손목 22녹음 | 2초 윈도우 236개 | Healthy → 0, Essential Tremor 또는 코멘트에 tremor/mixed type → 1 (`prepare_pads.py`와 동일 규칙) |

## 파일

| 파일 | 역할 |
| --- | --- |
| `prep.py` | 자체 데이터 → `data/bags.npz`. 50 Hz 리샘플, 1초 이동평균 고역통과, 윈도우 분할 |
| `abmil.py` | 모델(Conv1d 인코더 + gated attention + 선형 분류기), 학습, LOSO, 자체검증 |
| `viz.py` | hold-out 1명의 bag을 신호 + 스펙트로그램 + attention 배경으로 PNG |
| `pads_abmil.py` | PADS → `data/pads_bags.npz`, 5-fold 피험자 CV, 과제별 attention 합 |
| `export_pads_gui.py` | CV 결과를 뷰어용 `gui_pads/data.json`으로 |
| `gui_pads/` | **PADS Tremor Attention 뷰어** (`index.html` + `data.json` 13 MB). 피험자·과제 그룹 필터, 녹음별 attention, 윈도우별 3~12 Hz 파워 비율 |
| `data/pads_cv.npz` | 5-fold out-of-fold 예측 p와 attention A (커밋됨, 뷰어 재생성용) |

## 뷰어 바로 열기

```bash
cd experiments/seojin/abmil/gui_pads && python3 -m http.server 8000
```

브라우저에서 http://localhost:8000 . 학습·torch 없이 열린다 (`data.json` 포함 커밋).
파일을 더블클릭으로 열면 `data.json`을 못 읽으니 꼭 서버로 연다.

## 실행

```bash
pip install torch numpy pandas matplotlib          # torch는 CPU 빌드로 충분
python3 prep.py                                     # 자체 데이터 bag 생성 + 자체검증
python3 abmil.py                                    # 6-fold LOSO
python3 abmil.py march                              # 한 명 hold-out, bag별 attention 출력
python3 viz.py march                                # data/viz_march.png
```

PADS는 `data_sources/seojin/pads/raw/`에 원본이 있어야 한다 (`PADS_RAW` 환경변수로 변경 가능).

```bash
python3 pads_abmil.py                               # bag 생성(수 분) + 5-fold CV(약 20분, 결과 캐시)
python3 export_pads_gui.py                          # gui_pads/data.json 재생성 (모델 바꿨을 때만)
cd gui_pads && python3 -m http.server               # 브라우저에서 localhost:8000 → 바로 열림
```

CPU 3코어 기준 자체 데이터 LOSO 약 1분, PADS 5-fold 약 20분.

## 결과

### 자체 데이터 LOSO (balanced accuracy)

| hold-out | ABMIL | 비고 |
| --- | ---: | --- |
| BIBI, anna, euno, mu, seojin | 1.000 | |
| march | 0.600 | 저진폭·간헐적 떨림. RF·kNN도 같은 fold에서 틀림 |
| **평균** | **0.933** | RF 0.992, kNN 0.950 |

자체 데이터는 "떨림이면 20초 내내 떨었다"라 attention이 균등하고, MIL의 장점이 드러날 여지가 없다.
march만 attention이 특정 2초에 몰리는데, 수집 규칙과 어긋나므로 march 녹음은 재검토 대상이다.

### PADS 5-fold 피험자 CV (272명, 임계값 0.5 고정)

| 과제 그룹 | AUC | bal.acc | bal.acc (테스트에서 고른 최적 임계값, 낙관적) |
| --- | ---: | ---: | ---: |
| 전체 11과제 | 0.843 | 0.612 | - |
| 자세 4과제 (StretchHold, HoldWeight, LiftHold, CrossArms) | **0.845** | **0.698** | 0.788 (thr 0.70) |
| 운동성 4과제 (DrinkGlas, PointFinger, TouchNose, TouchIndex) | 0.810 | 0.601 | 0.740 (thr 0.71) |
| 안정 2과제 (Relaxed, RelaxedTask) | 0.745 | 0.501 | 0.706 (thr 0.64) |

기존 RF B2(subject AUC 0.91 / bal.acc 0.79)보다 낮다. 단 RF는 특징 공학 + 임계값 튜닝을 거쳤고 ABMIL은 25 epoch 기본 설정이다.

### 해석

- **attention은 안정 과제에 49%가 몰리지만, 실제로 가장 잘 가르는 건 자세 과제다.** 떨림군에 본태성진전·혼합형이 많아 자세 유지에서 떨림이 드러난다.
- **운동성 과제 중 쓸모 있는 건 DrinkGlas 하나** (attention 55%). PointFinger·TouchNose·TouchIndex는 1~3 Hz 동작이 지배해 떨림 대역 에너지가 거의 없다.
- **오분류는 전부 p 0.45~0.53에 몰려 있다.** 모델이 근거를 못 찾은 사람들이고, 떨림군이 2.4배 많아 기본값이 0.5 바로 위로 치우친다. 임계값 0.5 고정이 특이도를 깎는 주 원인.
- 뷰어에서 "오분류만" 필터로 보면 attention이 평평하고 3~12 Hz 파워 비율도 0.4 이하다. 즉 모델이 엉뚱한 곳을 본 게 아니라 볼 곳이 없었던 경우다.

## 다음

1. 임계값을 학습 fold 안에서 선택 (nested). 0.5 고정 → 약 0.7.
2. 0.45~0.55를 `uncertain`으로 보류. 로드맵의 세 상태 출력과 맞춘다.
3. 자세 4과제 + DrinkGlas 5과제로 bag을 좁혀 재학습. 수집 프로토콜도 "팔 뻗고 10초 유지"로 단순화 가능.
4. precision / sensitivity / specificity / F1을 fold별 평균 ± 표준편차로 보고 (Papadopoulos 2020 형식).

## 참고

- Ilse, Tomczak, Welling (2018). Attention-based Deep Multiple Instance Learning. ICML. arXiv:1802.04712
- Papadopoulos et al. (2020). Detecting Parkinsonian Tremor from IMU Data Collected In-The-Wild using Deep MIL. IEEE JBHI. arXiv:2005.04185
- Varghese et al. (2024). Machine Learning in the Parkinson's disease smartwatch dataset. npj Parkinson's Disease 10:9
