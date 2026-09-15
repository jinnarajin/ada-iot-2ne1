# PADS - Parkinsons Disease Smartwatch dataset

## 선택 이유

PADS는 Parkinson's disease, differential diagnosis, healthy control 참가자를 대상으로
양쪽 손목의 Apple Watch Series 4에서 가속도계와 자이로스코프 원시 시계열을
수집한 공개 데이터셋이다. 우리 프로젝트가 Apple Watch 또는 손목형 IMU 기기로
떨림을 탐지하는 방향이므로 센서 위치, 센서 종류, 동작 과제가 잘 맞는다.

원본 데이터는 용량이 크고 비상업적 ShareAlike 라이선스이므로 저장소에 직접
커밋하지 않는다. 이 폴더에는 출처, 사용 조건, 다운로드 방법과 변환 계획만 둔다.

## 기본 정보

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | PADS - Parkinsons Disease Smartwatch dataset v1.0.0 |
| 공식 출처 | https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/ |
| DOI | https://doi.org/10.13026/m0w9-zx22 |
| 논문 | Varghese et al., Machine Learning in the Parkinson's disease smartwatch dataset, npj Parkinson's Disease, 2024 |
| 접근성 | PhysioNet Open Access |
| 라이선스 | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International |
| 원본 용량 | ZIP 735.0 MB, 압축 해제 약 1.4 GB |
| 참가자 수 | 469명 |
| 참가자 그룹 | Parkinson's disease, differential diagnoses, healthy controls |
| 기기 | 양손목 Apple Watch Series 4 |
| 센서 | 3축 accelerometer, 3축 gyroscope |
| sampling rate | 100 Hz |
| 동작 과제 | 11개 neurological assessment task, 10-20초 길이 |
| 파일 형식 | JSON metadata, TXT time series, binary/preprocessed files |

## 라벨과 tremor 근거

PADS는 단순한 `tremor`/`non_tremor` 이진 라벨 데이터셋은 아니다. 대신 다음 정보를
사용해 우리 프로젝트에 맞는 학습 문제를 만들 수 있다.

- `patients/patient_*.json`: Parkinson's disease, differential diagnosis, healthy control
  조건 라벨
- differential diagnosis에는 essential tremor, atypical Parkinsonism, secondary
  Parkinsonism, multiple sclerosis 등이 포함됨
- 동작 과제는 rest, postural, kinetic tremor 특성을 유발하도록 신경과 전문의가 설계
- 양쪽 손목의 accelerometer/gyroscope 원시 시계열이 있어 떨림 주파수 특징을 직접
  계산할 수 있음

초기 사용에서는 `Healthy`를 non-tremor 후보, `Parkinson's disease`와 tremor 관련
differential diagnosis를 tremor-risk 후보로 두고 탐색한다. 실제 tremor 유무가
recording 단위로 명확히 라벨링되어 있는지 원본 metadata와 논문을 더 확인한 뒤
최종 이진 라벨 규칙을 확정한다.

## 다운로드

원본 데이터는 커밋하지 않는다. 필요한 사람이 로컬에서 다음 중 하나로 내려받는다.

```bash
mkdir -p data_sources/seojin/pads/raw
wget -r -N -c -np \
  -P data_sources/seojin/pads/raw \
  https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/
```

또는 AWS CLI:

```bash
mkdir -p data_sources/seojin/pads/raw
aws s3 sync --no-sign-request \
  s3://physionet-open/parkinsons-disease-smartwatch/1.0.0/ \
  data_sources/seojin/pads/raw/
```

간단한 안내와 변환 계획 확인:

```bash
python data_sources/seojin/pads/download_or_prepare.py --help
python data_sources/seojin/pads/download_or_prepare.py summary
```

## 현재 공통 CSV schema로 변환 계획

우리 저장소의 기본 IMU schema는 다음과 같다.

```text
elapsed_ms,
acc_x_g,acc_y_g,acc_z_g,
gyro_x_dps,gyro_y_dps,gyro_z_dps,
angle_x_deg,angle_y_deg,angle_z_deg
```

PADS 변환 규칙:

| PADS | 공통 schema |
| --- | --- |
| sample index / 100 Hz | `elapsed_ms = index * 10` |
| `Accelerometer_X/Y/Z` | `acc_x_g`, `acc_y_g`, `acc_z_g` |
| `Gyroscope_X/Y/Z` in rad/s | `gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps`로 변환 |
| 없음 | `angle_x_deg`, `angle_y_deg`, `angle_z_deg`는 비워두거나 제외 |
| participant id | manifest의 `subject_id` |
| task name + wrist | manifest의 `recording_id`, `activity`, `device_location` |
| patient condition | manifest의 `source_label` 및 파생 `label_candidate` |

구현: `sync_s3.py`(S3 미러 병렬 다운로드, aws CLI 불필요)와 `scripts/prepare_pads.py`.
로컬 raw 경로는 `PADS_RAW` 환경변수로 바꿀 수 있다.

```bash
PADS_RAW=/path/to/pads/raw python3 data_sources/seojin/pads/sync_s3.py
.venv/bin/python scripts/prepare_pads.py
.venv/bin/python scripts/tremor_classifier.py train --features results/pads_recording_features.csv --model models/tremor_rf_pads.pkl --threshold 0.68
```

`data/pads_manifest.csv`와 `results/pads_recording_features.csv`를 생성한다.

확정 라벨 규칙: `Healthy` → non_tremor, `Essential Tremor` 또는 `disease_comment`에
tremor/mixed type 언급 → tremor, 나머지(무진전형 PD, 근긴장이상, MS 등)는 제외.
"Healthy 외 전부 tremor"로 두면 떨림 없는 환자가 tremor로 들어가 어떤 모델도
non_tremor 경계를 배우지 못한다.

100 Hz 원본은 우리 baseline pipeline에서 50 Hz로 리샘플링한다. 첫 0.5초는 Watch
vibration 알림 영향이 있을 수 있으므로 분석에서 제거하는 옵션을 둔다.

## 장점

- 실제 환자와 healthy control이 포함되어 현재 모사 떨림 데이터보다 현실성이 높음
- Apple Watch에서 수집되어 1안의 기기 환경과 잘 맞음
- 양손목 동시 수집이라 좌우 손목 차이를 분석할 수 있음
- rest, postural, kinetic task가 모두 있어 움직임에 강건한 모델 평가에 도움
- 참가자 수가 469명으로 현재 데이터보다 subject-level 평가에 유리함

## 한계와 주의점

- 라이선스가 CC BY-NC-SA 4.0이라 상업적 사용이 제한되고, 파생물 공유 조건이 있음
- 파일이 커서 원본 데이터를 GitHub에 직접 올리면 안 됨
- 이진 tremor 라벨이 바로 주어지는 구조가 아니므로 라벨링 규칙을 신중히 정해야 함
- PD, differential diagnosis, healthy control 분류가 tremor 유무와 완전히 같지 않음
- 병원 기반 assessment라 일상생활 free-living 데이터와는 차이가 있음
- 성별·질환군 불균형이 있어 balanced accuracy와 class balancing이 필요함

## 인용

Varghese, J., Brenner, A., Plagwitz, L., van Alen, C., Fujarski, M., & Warnecke, T.
(2024). PADS - Parkinsons Disease Smartwatch dataset (version 1.0.0). PhysioNet.
https://doi.org/10.13026/m0w9-zx22

Varghese, J., Brenner, A., Fujarski, M., van Alen, C. M., Plagwitz, L., & Warnecke,
T. (2024). Machine Learning in the Parkinson's disease smartwatch dataset. npj
Parkinson's Disease, 10, 9.

