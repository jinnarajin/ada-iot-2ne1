# ada-iot-2ne1 · 손목 IMU 떨림 탐지

손목에 착용한 IMU(가속도계 + 자이로스코프) 신호로 **떨림(tremor) 유무를 실시간 추정**하는
웨어러블을 만드는 프로젝트입니다. 1안은 Apple Watch, 2안은 자체 IMU 센서 + 3D 프린팅 케이스입니다.

> ⚠️ 현재 모델은 연구용 프로토타입입니다. 자체 수집 데이터는 건강한 참가자가 **흉내 낸 떨림**이고,
> 환자 데이터(PADS)로는 아직 검증 초기 단계입니다. 의료 진단 용도로 쓸 수 없습니다.

---

## 1. 현황 한눈에 보기 (2026-10-20 기준)

| 영역 | 상태 | 핵심 수치 |
| --- | --- | --- |
| 자체 데이터셋 (모사 떨림, 6명 220 recording) | ✅ 완료 | 정리·manifest 생성 |
| 베이스라인 분석 (LR / SVM / RF / kNN) | ✅ 완료 | RF B2: LOSO 0.992, cross-dataset 0.990 |
| 학습·추론 CLI (`tremor_classifier.py`) | ✅ 완료 | `train` / `predict` |
| PADS 환자 데이터 연동 (469명, 1.5 GB) | ✅ 다운로드·변환·학습 완료 | RF: recording AUC 0.79, subject AUC 0.91 |
| 떨림 진폭 기준치 (PADS 건강군 기준) | ✅ 산출 | 안정시 p95 ≈ 12 mg, 자세 유지 ≈ 22 mg |
| 팀원별 분류기 (과제 1) | 🔄 진행 중 | seojin: kNN 완료 |
| 팀원별 오픈 데이터셋 (과제 2) | 🔄 진행 중 | seojin: PADS 완료, 후보 6개 조사 |
| Apple Watch 앱 / 온디바이스 추론 | ⏳ 예정 | 로드맵 문서 참고 |

### 지금까지 알게 된 것

- **진폭만으로도 모사 떨림은 거의 완벽히 구분됩니다.** 주파수 특징(B2)은 사람이 바뀔 때보다
  **수집 조건(기기·샘플링)이 바뀔 때** 더 도움이 됩니다.
- **실제 환자 데이터(PADS)는 훨씬 어렵습니다.** recording 단위 balanced accuracy 0.72.
  정적 과제(팔 뻗기, 팔짱)는 0.85 이상이지만, 동작 과제(Entrainment 등)에서 건강한 사람의
  움직임을 떨림으로 오탐하는 것이 주요 오류입니다.
- **헤르츠(피크 주파수)는 떨림 유무 판정 기준으로 쓰기 어렵습니다.** 건강군도 4~8 Hz에 피크가
  찍히고 파킨슨·본태성진전과 범위가 겹칩니다. **1차 판정은 3~12 Hz 대역 진폭(RMS)**, 종류 구분에만
  주파수를 보조로 쓰는 2단계 접근이 맞습니다.

---

## 2. 데이터

### 2-1. 자체 수집 (모사 떨림)

| ID | 출처 | 참가자 | 파일 | 샘플링 |
| --- | --- | ---: | ---: | --- |
| `dataset_a` | [jeongbaepsae/iOT-tremor-dataset](https://github.com/jeongbaepsae/iOT-tremor-dataset) | 4 | 140 | 약 100 Hz |
| `dataset_b` | [hmuzn/iot-tremor-sensor-dataset](https://github.com/hmuzn/iot-tremor-sensor-dataset) | 2 | 80 | 약 50 Hz, 타임스탬프 지터·결손 있음 |

recording 하나 = 약 20초 CSV. 라벨은 `simulated_tremor` / `non_tremor`.

공통 CSV 스키마:

```text
elapsed_ms,
acc_x_g,acc_y_g,acc_z_g,
gyro_x_dps,gyro_y_dps,gyro_z_dps,
angle_x_deg,angle_y_deg,angle_z_deg   # 분석에서는 제외 (자세 누설)
```

원본은 `data/raw/`에 두고 수정하지 않습니다. 데이터를 추가하면 manifest를 다시 만듭니다.

```bash
python3 scripts/build_manifest.py
```

### 2-2. PADS (실제 환자, PhysioNet)

Apple Watch Series 4 양손목, 100 Hz, 469명, 11개 신경학 과제. 상세 설명·라이선스·인용은
[data_sources/seojin/pads/README.md](data_sources/seojin/pads/README.md).

- **라벨 규칙**: `Healthy` → non_tremor (79명) / `Essential Tremor` 또는 진단 코멘트에 tremor 언급 → tremor (193명) /
  나머지 197명(무진전형 PD, MS 등)은 떨림 근거가 없어 **제외**.
- **라이선스 CC BY-NC-SA 4.0**, 용량이 커서 원본·변환 CSV는 커밋하지 않습니다 (`.gitignore`).

```bash
# 다운로드 (aws CLI 불필요, S3 미러 병렬)
python3 data_sources/seojin/pads/sync_s3.py
# 공통 스키마로 변환 + 특징 추출 → data/pads_manifest.csv, results/pads_recording_features.csv
.venv/bin/python scripts/prepare_pads.py
```

### 2-3. 추가 후보 데이터셋

팀원이 나눠 맡을 후보 6개 비교: [data_sources/seojin/CANDIDATE_DATASETS_KO.md](data_sources/seojin/CANDIDATE_DATASETS_KO.md).
추천 순서는 **Levodopa Response Study(Synapse)** → **Monipar(Zenodo, CC-BY, 35 MB)**.

---

## 3. 빠른 시작

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 베이스라인 재현

```bash
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_baseline_analysis.py      # LR, B0/B1/B2
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_sensitivity_analysis.py   # 센서·윈도우 ablation
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_nonlinear_baselines.py    # RBF-SVM, RF
```

### 분류기 학습 · 새 파일 판정

```bash
.venv/bin/python scripts/tremor_classifier.py train                       # 자체 데이터 → models/tremor_rf_b2.pkl
.venv/bin/python scripts/tremor_classifier.py predict path/to/new_imu.csv # simulated_tremor / non_tremor + 확률
```

PADS로 학습할 때는 임계값을 함께 저장합니다.

```bash
.venv/bin/python scripts/tremor_classifier.py train \
  --features results/pads_recording_features.csv \
  --model models/tremor_rf_pads.pkl --threshold 0.68
```

입력 CSV는 최소 3초, 위 7개 열(시간 + acc + gyro)이 필요합니다. 학습 데이터와 비슷한 20초 전후가 가장 좋습니다.
`models/*.pkl`은 Python pickle이므로 직접 만든 파일만 로드하세요.

---

## 4. 결과 요약

평가는 전부 **recording 단위**, 같은 recording의 윈도우는 train/test에 나뉘지 않습니다.
스케일링·특징 선택·임계값은 training fold에서만 적합합니다.

### 자체 데이터 (B2 = 시간 + 주파수 특징, 3초 윈도우, 50 Hz)

| 분류기 | LOSO 평균 | LOSO std | cross-dataset 평균 | 비고 |
| --- | ---: | ---: | ---: | --- |
| **Random Forest** | **0.992** | 0.020 | **0.990** | 현재 대표 모델 |
| RBF-SVM | 0.992 | 0.020 | 0.977 | |
| Logistic Regression | 0.975 | 0.061 | 0.989 | 해석 가능 기준 |
| kNN (k=5) | 0.950 | 0.122 | 0.976 | 오류는 전부 march의 저진폭 떨림 |

- 사람 6명이라 fold 하나가 평균을 크게 흔듭니다. 모델 간 순위 차이는 오분류 1~3건 수준이므로 과해석 금지.
- 자이로스코프만 써도 LOSO 0.99가 나옵니다 (2초 윈도우 time-only 0.9917).

### PADS (RF B2, 5-fold GroupKFold, 272명 5,984 recording)

| 단위 | AUC | balanced accuracy |
| --- | ---: | ---: |
| recording | 0.79 | 0.72 (임계값 0.68) |
| subject | 0.91 | 0.79 |

PADS 학습 모델을 자체 데이터에 적용하면 balanced accuracy 0.87.

### 떨림 진폭 기준치 (PADS 원본, 3~12 Hz 가속도 RMS, 단위 mg)

| 과제 | 건강군 p50 / p95 | 파킨슨 p50 / p95 | 본태성진전 p50 / p95 |
| --- | --- | --- | --- |
| Relaxed (안정) | 5.5 / 11.6 | 6.9 / 78.1 | 10.8 / 98.0 |
| StretchHold (팔 뻗기) | 7.8 / 22.0 | 10.5 / 53.9 | 25.4 / 216.9 |
| HoldWeight (무게 들기) | 9.2 / 23.0 | 12.0 / 53.0 | 28.8 / 202.7 |
| TouchNose (코 짚기, 동작) | 171.9 / 409.2 | 171.8 / 446.9 | 238.0 / 662.5 |

건강군 p95를 "정상 상한"으로 쓸 수 있습니다. 파킨슨은 중앙값은 정상과 비슷하고 **p95만 벌어집니다**
(일부 환자만 뚜렷한 떨림). 동작 과제는 움직임 자체가 같은 대역에 들어가 기준치로 부적합합니다.

상세 보고서:
[1차 분석](results/INITIAL_ANALYSIS_KO.md) ·
[민감도 분석](results/SENSITIVITY_ANALYSIS_KO.md) ·
[비선형 베이스라인](results/NONLINEAR_BASELINES_KO.md) ·
[kNN 실험](experiments/seojin/knn/README.md)

---

## 5. 팀 과제와 로드맵

- **이번 주 과제** (팀원별 분류기 1개 + 오픈 데이터셋 1개): [docs/WEEKLY_TASKS_2026-09-15_KO.md](docs/WEEKLY_TASKS_2026-09-15_KO.md)
  - 완료 기준: 다른 팀원이 README 명령으로 **재현**할 수 있고, 데이터셋 **사용 조건과 라벨 의미**를 확인할 수 있어야 함.
- **개발 로드맵** (Apple Watch → 자체 IMU 기기 fallback, 최종 출력 `tremor` / `non_tremor` / `uncertain`):
  [docs/APPLE_WATCH_ROADMAP_KO.md](docs/APPLE_WATCH_ROADMAP_KO.md)
- **연구 프로토콜** (가설 H1~H4, 전처리, 지표, ablation): [한국어](docs/RESEARCH_PROTOCOL_KO.md) · [English](docs/RESEARCH_PROTOCOL.md)

### 다음 단계

1. 각 팀원 분류기·데이터셋 결과를 `experiments/<이름>/`, `data_sources/<이름>/`에 추가
2. PADS 동작 과제 오탐 줄이기 (과제 인식 또는 동작 보상 특징)
3. 1D CNN · dual-view(raw + 주파수) 모델은 위 베이스라인이 끝난 뒤 비교
4. Apple Watch 수집 앱 + Core ML 변환 (타임박스 2일, 실패 시 2안 전환)

---

## 6. 저장소 구조

```text
data/            manifest.csv, pads_manifest.csv(생성물), raw/(자체 수집 원본)
data_sources/    팀원별 오픈 데이터셋 문서·다운로드 스크립트 (원본은 ignore)
docs/            로드맵, 연구 프로토콜, 주간 과제
experiments/     팀원별 분류기 실험 (코드 + README + 결과)
models/          학습된 .pkl (ignore)
results/         분석 보고서(.md), 지표(.csv), 그림(.png)
scripts/         manifest 생성, 베이스라인 분석, PADS 변환, 학습·추론 CLI
```

## 7. 범위와 한계

- 자체 데이터의 떨림은 전부 **의도적으로 흉내 낸 것**입니다. 파킨슨·본태성진전 진단이나 중증도 평가에 쓸 수 없습니다.
- PADS는 병원 환경 과제 수행 데이터라 일상생활(free-living) 신호와 다릅니다.
- 자체 데이터셋 원본 저장소에 라이선스 표기가 없습니다. 공개 배포 전 원 저작자 확인이 필요합니다.
- PADS는 CC BY-NC-SA 4.0으로 **비상업·동일조건 공유** 제약이 있습니다.
