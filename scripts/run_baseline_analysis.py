#!/usr/bin/env python3
"""Run recording-level EDA and leakage-safe simulated-tremor baselines."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.csv"
RESULTS = ROOT / "results"
FS = 50.0
WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = int(FS * WINDOW_SECONDS)
STEP_SAMPLES = WINDOW_SAMPLES // 2
MAX_GAP_MS = 100.0
SENSOR_COLUMNS = {
    "acc": ["acc_x_g", "acc_y_g", "acc_z_g"],
    "gyro": ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"],
}
MODEL_FEATURE_PREFIXES = {
    "B0_amplitude": ("acc__rms", "gyro__rms"),
    "B1_time": (
        "acc__rms",
        "acc__std",
        "acc__mad",
        "acc__jerk_rms",
        "acc__zcr",
        "gyro__rms",
        "gyro__std",
        "gyro__mad",
        "gyro__jerk_rms",
        "gyro__zcr",
    ),
    "B2_time_frequency": (
        "acc__",
        "gyro__",
    ),
}


def resample_recording(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    frame = pd.read_csv(path)
    time_ms = frame["elapsed_ms"].to_numpy(dtype=float)
    if len(time_ms) < 2 or np.any(np.diff(time_ms) <= 0):
        raise ValueError(f"Timestamps must be strictly increasing: {path}")

    uniform_ms = np.arange(time_ms[0], time_ms[-1] + 0.01, 1000.0 / FS)
    signals: dict[str, np.ndarray] = {}
    for sensor, columns in SENSOR_COLUMNS.items():
        axes = np.column_stack(
            [np.interp(uniform_ms, time_ms, frame[column].to_numpy(float)) for column in columns]
        )
        signals[sensor] = axes

    original_gap_ms = np.diff(time_ms)
    return uniform_ms, signals, original_gap_ms


def spectral_features(axes: np.ndarray) -> dict[str, float]:
    """Calculate rotationally robust spectra by summing the three axis PSDs."""
    frequency, power = welch(
        axes,
        fs=FS,
        window="hann",
        nperseg=len(axes),
        noverlap=0,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    power = power.sum(axis=1)
    tremor = (frequency >= 3.0) & (frequency <= 12.0)
    total = (frequency >= 0.5) & (frequency <= 20.0)
    tremor_power = np.trapezoid(power[tremor], frequency[tremor])
    total_power = np.trapezoid(power[total], frequency[total])
    band_power = power[tremor]
    normalized = band_power / (band_power.sum() + 1e-15)
    peak_index = int(np.argmax(band_power))

    output = {
        "log_power_3_12": float(np.log10(tremor_power + 1e-15)),
        "dominant_hz": float(frequency[tremor][peak_index]),
        "peak_ratio": float(band_power[peak_index] / (band_power.sum() + 1e-15)),
        "spectral_entropy": float(
            -np.sum(normalized * np.log(normalized + 1e-15)) / np.log(len(normalized))
        ),
        "tremor_total_ratio": float(tremor_power / (total_power + 1e-15)),
    }
    for low, high in ((3.0, 6.0), (6.0, 9.0), (9.0, 12.0)):
        selected = (frequency >= low) & (frequency < high if high < 12.0 else frequency <= high)
        selected_power = np.trapezoid(power[selected], frequency[selected])
        output[f"ratio_{int(low)}_{int(high)}"] = float(
            selected_power / (tremor_power + 1e-15)
        )
    return output


def window_features(axes: np.ndarray) -> dict[str, float]:
    axes = axes - axes.mean(axis=0, keepdims=True)
    signal = np.linalg.norm(axes, axis=1)
    centered = signal - signal.mean()
    median_value = np.median(signal)
    return {
        "rms": float(np.sqrt(np.mean(np.square(signal)))),
        "std": float(np.std(signal)),
        "mad": float(np.median(np.abs(signal - median_value))),
        "jerk_rms": float(np.sqrt(np.mean(np.square(np.diff(signal) * FS)))),
        "zcr": float(np.mean(np.signbit(centered[1:]) != np.signbit(centered[:-1]))),
        **spectral_features(axes),
    }


def recording_features(row: pd.Series) -> dict[str, object]:
    path = ROOT / row["relative_path"]
    uniform_ms, signals, original_gaps = resample_recording(path)
    starts = range(0, len(uniform_ms) - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    windows: list[dict[str, float]] = []
    rejected = 0

    raw_time = pd.read_csv(path, usecols=["elapsed_ms"])["elapsed_ms"].to_numpy(float)
    for start in starts:
        left = uniform_ms[start]
        right = uniform_ms[start + WINDOW_SAMPLES - 1]
        inside = (raw_time[:-1] >= left) & (raw_time[1:] <= right)
        if np.any(original_gaps[inside] > MAX_GAP_MS):
            rejected += 1
            continue
        feature_row: dict[str, float] = {}
        for sensor, axes in signals.items():
            values = window_features(axes[start : start + WINDOW_SAMPLES])
            feature_row.update({f"{sensor}__{name}": value for name, value in values.items()})
        windows.append(feature_row)

    if not windows:
        raise ValueError(f"No valid windows: {path}")

    window_frame = pd.DataFrame(windows)
    output: dict[str, object] = {
        "recording_id": row["recording_id"],
        "dataset_id": row["dataset_id"],
        "subject_id": row["subject_id"],
        "label": row["label"],
        "target": int(row["label"] == "tremor"),
        "valid_windows": len(windows),
        "rejected_windows": rejected,
    }
    for column in window_frame.columns:
        output[f"{column}__median"] = float(window_frame[column].median())
        output[f"{column}__iqr"] = float(
            window_frame[column].quantile(0.75) - window_frame[column].quantile(0.25)
        )
    return output


def feature_columns(frame: pd.DataFrame, model_name: str) -> list[str]:
    prefixes = MODEL_FEATURE_PREFIXES[model_name]
    return sorted(
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in prefixes)
    )


def score_predictions(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_and_score(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
) -> tuple[dict[str, float | int], np.ndarray]:
    columns = feature_columns(frame, model_name)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
    )
    model.fit(frame.loc[train_mask, columns], frame.loc[train_mask, "target"])
    probability = model.predict_proba(frame.loc[test_mask, columns])[:, 1]
    return score_predictions(frame.loc[test_mask, "target"].to_numpy(), probability), probability


def run_evaluations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    for subject in sorted(frame["subject_id"].unique()):
        test_mask = (frame["subject_id"] == subject).to_numpy()
        train_mask = ~test_mask
        for model_name in MODEL_FEATURE_PREFIXES:
            score, probability = fit_and_score(frame, train_mask, test_mask, model_name)
            metrics.append(
                {"evaluation": "LOSO", "fold": subject, "model": model_name, **score}
            )
            for index, value in zip(frame.index[test_mask], probability):
                predictions.append(
                    {
                        "evaluation": "LOSO",
                        "fold": subject,
                        "model": model_name,
                        "recording_id": frame.at[index, "recording_id"],
                        "target": frame.at[index, "target"],
                        "probability": value,
                    }
                )

    for train_dataset, test_dataset in (("dataset_a", "dataset_b"), ("dataset_b", "dataset_a")):
        train_mask = (frame["dataset_id"] == train_dataset).to_numpy()
        test_mask = (frame["dataset_id"] == test_dataset).to_numpy()
        fold = f"{train_dataset}_to_{test_dataset}"
        for model_name in MODEL_FEATURE_PREFIXES:
            score, probability = fit_and_score(frame, train_mask, test_mask, model_name)
            metrics.append(
                {"evaluation": "cross_dataset", "fold": fold, "model": model_name, **score}
            )
            for index, value in zip(frame.index[test_mask], probability):
                predictions.append(
                    {
                        "evaluation": "cross_dataset",
                        "fold": fold,
                        "model": model_name,
                        "recording_id": frame.at[index, "recording_id"],
                        "target": frame.at[index, "target"],
                        "probability": value,
                    }
                )

    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def create_plots(features: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"non_tremor": "#4C78A8", "tremor": "#E45756"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_features = [
        ("acc__rms__median", "Acceleration RMS"),
        ("gyro__rms__median", "Gyroscope RMS"),
        ("acc__dominant_hz__median", "Acceleration dominant frequency (Hz)"),
        ("gyro__dominant_hz__median", "Gyroscope dominant frequency (Hz)"),
    ]
    positions = {("dataset_a", "non_tremor"): 0, ("dataset_a", "tremor"): 1,
                 ("dataset_b", "non_tremor"): 3, ("dataset_b", "tremor"): 4}
    for axis, (column, title) in zip(axes.flat, plot_features):
        for (dataset, label), position in positions.items():
            values = features.loc[
                (features["dataset_id"] == dataset) & (features["label"] == label), column
            ]
            box = axis.boxplot(values, positions=[position], widths=0.7, patch_artist=True,
                               showfliers=False)
            box["boxes"][0].set_facecolor(colors[label])
            box["boxes"][0].set_alpha(0.75)
        axis.set_xticks([0.5, 3.5], ["Dataset A", "Dataset B"])
        axis.set_title(title)
        if "RMS" in title:
            axis.set_yscale("log")
    fig.suptitle("Recording-level signal characteristics", fontsize=14)
    fig.legend(
        handles=[
            Patch(facecolor=colors["non_tremor"], alpha=0.75, label="Non-tremor"),
            Patch(facecolor=colors["tremor"], alpha=0.75, label="Simulated tremor"),
        ],
        loc="lower center",
        ncol=2,
    )
    fig.subplots_adjust(bottom=0.1)
    fig.tight_layout()
    fig.savefig(RESULTS / "feature_distributions.png", dpi=180)
    plt.close(fig)

    loso = metrics[metrics["evaluation"] == "LOSO"]
    order = list(MODEL_FEATURE_PREFIXES)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, model in enumerate(order):
        subset = loso[loso["model"] == model].sort_values("fold")
        x = np.arange(len(subset)) + (offset - 1) * 0.22
        axis.bar(x, subset["balanced_accuracy"], width=0.22, label=model)
    subjects = sorted(loso["fold"].unique())
    axis.set_xticks(np.arange(len(subjects)), subjects)
    axis.set_ylim(0.0, 1.05)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Leave-One-Subject-Out performance")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "loso_balanced_accuracy.png", dpi=180)
    plt.close(fig)

    cross = metrics[metrics["evaluation"] == "cross_dataset"]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    directions = ["dataset_a_to_dataset_b", "dataset_b_to_dataset_a"]
    for offset, model in enumerate(order):
        subset = cross.set_index(["fold", "model"])
        values = [subset.loc[(direction, model), "balanced_accuracy"] for direction in directions]
        x = np.arange(len(directions)) + (offset - 1) * 0.22
        axis.bar(x, values, width=0.22, label=model)
    axis.set_xticks(np.arange(len(directions)), ["Dataset A → B", "Dataset B → A"])
    axis.set_ylim(0.8, 1.005)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Cross-dataset performance")
    axis.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(RESULTS / "cross_dataset_balanced_accuracy.png", dpi=180)
    plt.close(fig)


def write_summary(features: pd.DataFrame, metrics: pd.DataFrame) -> None:
    loso = metrics[metrics["evaluation"] == "LOSO"]
    cross = metrics[metrics["evaluation"] == "cross_dataset"]
    summary = {
        "n_recordings": int(len(features)),
        "n_subjects": int(features["subject_id"].nunique()),
        "valid_windows": int(features["valid_windows"].sum()),
        "rejected_windows": int(features["rejected_windows"].sum()),
        "loso_mean": loso.groupby("model")["balanced_accuracy"].mean().to_dict(),
        "loso_std": loso.groupby("model")["balanced_accuracy"].std().to_dict(),
        "cross_dataset_mean": cross.groupby("model")["balanced_accuracy"].mean().to_dict(),
        "cross_dataset_by_direction": {
            f"{row.fold}:{row.model}": row.balanced_accuracy for row in cross.itertuples()
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    RESULTS.mkdir(exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    features = pd.DataFrame([recording_features(row) for _, row in manifest.iterrows()])
    features.to_csv(RESULTS / "recording_features.csv", index=False)

    metrics, predictions = run_evaluations(features)
    metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "baseline_predictions.csv", index=False)
    create_plots(features, metrics)
    write_summary(features, metrics)

    print(metrics.to_string(index=False))
    print(f"\nResults written to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
