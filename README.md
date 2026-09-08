# Cross-condition simulated tremor detection

## Research question

> 다른 sampling condition에서 수집된 소규모 IMU 데이터에 대해, 주파수 정보를 명시적으로 사용하는 모델이 subject-independent 및 cross-dataset simulated tremor detection을 개선하는가?

This repository studies binary detection of **simulated tremor versus non-tremor motion in healthy participants**. It does not contain measurements from patients and must not be presented as a clinical diagnosis dataset.

## Data

The repository combines two independently collected datasets while preserving their provenance:

| ID | Original source | Participants | Files | Nominal sampling condition |
| --- | --- | ---: | ---: | --- |
| `dataset_a` | [jeongbaepsae/iOT-tremor-dataset](https://github.com/jeongbaepsae/iOT-tremor-dataset) | 4 | 140 | approximately 100 Hz |
| `dataset_b` | [hmuzn/iot-tremor-sensor-dataset](https://github.com/hmuzn/iot-tremor-sensor-dataset) | 2 | 80 | approximately 50 Hz, with timestamp jitter and occasional gaps |

Each CSV contains elapsed time, tri-axial acceleration, tri-axial angular velocity, and derived orientation angles:

```text
elapsed_ms,
acc_x_g,acc_y_g,acc_z_g,
gyro_x_dps,gyro_y_dps,gyro_z_dps,
angle_x_deg,angle_y_deg,angle_z_deg
```

Raw files are immutable inputs under `data/raw/`. Run the manifest builder after adding or changing source data:

```bash
python3 scripts/build_manifest.py
```

The generated `data/manifest.csv` records dataset provenance, subject, label, row count, duration, and timestamp quality for every recording.

## Baseline analysis

Install the pinned dependencies with Python 3.12 and run the leakage-safe recording-level baseline analysis:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/ada_iot_mpl .venv/bin/python scripts/run_baseline_analysis.py
```

The initial results and interpretation are documented in [results/INITIAL_ANALYSIS_KO.md](results/INITIAL_ANALYSIS_KO.md). Sensor and window-length ablations are reported in [results/SENSITIVITY_ANALYSIS_KO.md](results/SENSITIVITY_ANALYSIS_KO.md). Fixed RBF-SVM and Random Forest baselines are reported in [results/NONLINEAR_BASELINES_KO.md](results/NONLINEAR_BASELINES_KO.md).

## Train and score a recording

After generating `results/recording_features.csv`, fit the selected Random Forest B2
baseline on all available recordings and score one or more new CSV files:

```bash
.venv/bin/python scripts/tremor_classifier.py train
.venv/bin/python scripts/tremor_classifier.py predict path/to/new_imu.csv
```

The command prints `simulated_tremor` or `non_tremor` and the simulated-tremor
probability. Input CSVs need at least 3 seconds of data and the seven timestamp,
accelerometer, and gyroscope columns documented above. A duration close to the
approximately 20-second training recordings is preferable because the model aggregates
3-second windows at recording level.

The resulting `models/tremor_rf_b2.pkl` is a local Python pickle; load only an artifact
you created or otherwise trust. This model detects the dataset's voluntarily simulated
tremor and is not a clinical diagnostic model.

## Experimental protocol

The primary comparison is:

1. interpretable time-domain features;
2. time-domain features plus explicit spectral features;
3. a compact raw-signal 1D CNN;
4. a dual-view model combining raw IMU windows and frequency representations;
5. gated-attention MIL when recording-level labels and intermittent windows justify it.

Primary evaluation:

- Leave-One-Subject-Out cross-validation across all six participants;
- train on `dataset_a`, test on `dataset_b`;
- train on `dataset_b`, test on `dataset_a`.

Windows from one recording must never be divided across train and test sets. Preprocessing statistics, feature selection, and thresholds must be fitted using the training fold only.

See [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) for hypotheses, preprocessing, metrics, and required ablations. A Korean version is available at [docs/RESEARCH_PROTOCOL_KO.md](docs/RESEARCH_PROTOCOL_KO.md).

## Scope and limitations

- All tremor is voluntarily simulated by healthy participants.
- The dataset supports methodological evaluation of IMU-based simulated-tremor detection, not Parkinson disease, essential tremor, severity scoring, or clinical diagnosis.
- With only six participants, subject-level uncertainty and per-fold results are more informative than a single random-split accuracy.
- The original repositories do not currently state a license. Redistribution and downstream use should be confirmed with the source owners before public release.
