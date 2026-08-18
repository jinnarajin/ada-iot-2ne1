#!/usr/bin/env python3
"""Run sensor and window-length sensitivity analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_baseline_analysis as baseline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WINDOW_SECONDS = (2.0, 3.0, 5.0)
SENSORS = ("acc", "gyro", "both")
MODELS = ("B1_time", "B2_time_frequency")


def select_columns(frame: pd.DataFrame, model_name: str, sensor: str) -> list[str]:
    feature_prefixes = baseline.MODEL_FEATURE_PREFIXES[model_name]
    sensor_prefixes = ("acc__", "gyro__") if sensor == "both" else (f"{sensor}__",)
    return sorted(
        column
        for column in frame.columns
        if column.startswith(sensor_prefixes)
        and any(column.startswith(prefix) for prefix in feature_prefixes)
    )


def fit_and_score(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
    sensor: str,
) -> dict[str, float | int]:
    columns = select_columns(frame, model_name, sensor)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
    )
    model.fit(frame.loc[train_mask, columns], frame.loc[train_mask, "target"])
    probability = model.predict_proba(frame.loc[test_mask, columns])[:, 1]
    return baseline.score_predictions(frame.loc[test_mask, "target"].to_numpy(), probability)


def evaluate(frame: pd.DataFrame, window_seconds: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in sorted(frame["subject_id"].unique()):
        test_mask = (frame["subject_id"] == subject).to_numpy()
        for sensor in SENSORS:
            for model_name in MODELS:
                rows.append(
                    {
                        "window_seconds": window_seconds,
                        "evaluation": "LOSO",
                        "fold": subject,
                        "sensor": sensor,
                        "model": model_name,
                        **fit_and_score(frame, ~test_mask, test_mask, model_name, sensor),
                    }
                )

    for train_dataset, test_dataset in (("dataset_a", "dataset_b"), ("dataset_b", "dataset_a")):
        train_mask = (frame["dataset_id"] == train_dataset).to_numpy()
        test_mask = (frame["dataset_id"] == test_dataset).to_numpy()
        for sensor in SENSORS:
            for model_name in MODELS:
                rows.append(
                    {
                        "window_seconds": window_seconds,
                        "evaluation": "cross_dataset",
                        "fold": f"{train_dataset}_to_{test_dataset}",
                        "sensor": sensor,
                        "model": model_name,
                        **fit_and_score(frame, train_mask, test_mask, model_name, sensor),
                    }
                )
    return rows


def plot_results(metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axis, evaluation, title in zip(
        axes,
        ("LOSO", "cross_dataset"),
        ("LOSO mean across participants", "Cross-dataset mean across directions"),
    ):
        grouped = (
            metrics[metrics["evaluation"] == evaluation]
            .groupby(["window_seconds", "sensor", "model"])["balanced_accuracy"]
            .mean()
        )
        for sensor, linestyle in zip(SENSORS, (":", "--", "-")):
            for model, marker in zip(MODELS, ("o", "s")):
                values = [grouped.loc[(window, sensor, model)] for window in WINDOW_SECONDS]
                axis.plot(
                    WINDOW_SECONDS,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{sensor} / {model}",
                )
        axis.set_title(title)
        axis.set_xlabel("Window length (seconds)")
        axis.set_xticks(WINDOW_SECONDS)
        axis.set_ylim(0.75, 1.005)
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "sensitivity_analysis.png", dpi=180)
    plt.close(fig)


def main() -> None:
    manifest = pd.read_csv(baseline.MANIFEST)
    all_metrics: list[dict[str, object]] = []
    for window_seconds in WINDOW_SECONDS:
        baseline.WINDOW_SECONDS = window_seconds
        baseline.WINDOW_SAMPLES = int(baseline.FS * window_seconds)
        baseline.STEP_SAMPLES = baseline.WINDOW_SAMPLES // 2
        features = pd.DataFrame(
            [baseline.recording_features(row) for _, row in manifest.iterrows()]
        )
        features.to_csv(RESULTS / f"recording_features_{int(window_seconds)}s.csv", index=False)
        all_metrics.extend(evaluate(features, window_seconds))
        print(f"Completed {window_seconds:g}-second windows")

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(RESULTS / "sensitivity_metrics.csv", index=False)
    plot_results(metrics)
    summary = (
        metrics.groupby(["window_seconds", "evaluation", "sensor", "model"])[
            "balanced_accuracy"
        ]
        .mean()
        .round(4)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
