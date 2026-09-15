#!/usr/bin/env python3
"""Train and participant-disjointly evaluate HGB on converted PADS recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_baseline_analysis as baseline  # noqa: E402
from train import make_model  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data/processed/dataset_c_pads/manifest_dataset_c.csv"


def score(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "auroc": float(roc_auc_score(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def extract_features(manifest_path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    manifest = pd.read_csv(manifest_path)
    # PADS paths in the manifest are relative to this repository. The shared
    # extractor uses baseline.ROOT to resolve them.
    baseline.ROOT = ROOT
    rows = [baseline.recording_features(row) for _, row in manifest.iterrows()]
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HERE / "pads_results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = extract_features(
        args.manifest.resolve(), args.output_dir / "recording_features.csv"
    )
    columns = baseline.feature_columns(features, "B2_time_frequency")
    target = features["target"].to_numpy(dtype=int)
    groups = features["subject_id"].to_numpy()
    subject_labels = features.groupby("subject_id")["target"].first()
    n_splits = min(5, int(subject_labels.value_counts().min()))
    if n_splits < 2:
        raise SystemExit("Need at least two tremor and two control participants")

    # Split subjects—not recordings—within each class, then combine one chunk
    # from each class per fold. This guarantees both participant isolation and
    # approximately equal class counts in every test fold.
    rng = np.random.default_rng(42)
    subject_folds: list[list[str]] = [[] for _ in range(n_splits)]
    for label in (0, 1):
        label_subjects = subject_labels[subject_labels == label].index.to_numpy(copy=True)
        rng.shuffle(label_subjects)
        for fold, chunk in enumerate(np.array_split(label_subjects, n_splits)):
            subject_folds[fold].extend(chunk.tolist())
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for fold, test_subjects_for_fold in enumerate(subject_folds, start=1):
        test_mask = np.isin(groups, test_subjects_for_fold)
        test_index = np.flatnonzero(test_mask)
        train_index = np.flatnonzero(~test_mask)
        model = make_model()
        model.fit(features.iloc[train_index][columns], target[train_index])
        probability = model.predict_proba(features.iloc[test_index][columns])[:, 1]
        test_subjects = sorted(set(groups[test_index]))
        metrics.append({"fold": fold, "test_subjects": ";".join(test_subjects), **score(target[test_index], probability)})
        for index, value in zip(test_index, probability):
            predictions.append({
                "fold": fold,
                "recording_id": features.iloc[index]["recording_id"],
                "subject_id": groups[index],
                "target": target[index],
                "probability": float(value),
                "prediction": int(value >= 0.5),
            })

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "predictions.csv", index=False)

    final_model = make_model()
    final_model.fit(features[columns], target)
    model_path = ROOT / "models/hgb_pads_strict.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_columns": columns,
        "sampling_rate_hz": baseline.FS,
        "window_seconds": baseline.WINDOW_SECONDS,
        "label_scope": "participant-level weak label",
        "n_subjects": int(features["subject_id"].nunique()),
        "n_recordings": int(len(features)),
    }, model_path)

    summary = {
        "dataset": "PADS strict cohort",
        "n_subjects": int(features["subject_id"].nunique()),
        "n_recordings": int(len(features)),
        "n_splits": n_splits,
        "label_warning": "Participant-level diagnosis-derived weak labels; not per-window tremor ground truth.",
        "mean_metrics": {
            name: float(metrics_frame[name].mean())
            for name in ("balanced_accuracy", "sensitivity", "specificity", "macro_f1", "auroc")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Model: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
