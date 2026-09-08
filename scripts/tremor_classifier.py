#!/usr/bin/env python3
"""Train a recording-level simulated-tremor classifier and score new IMU CSVs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from run_baseline_analysis import ROOT, feature_columns, recording_features


DEFAULT_FEATURES = ROOT / "results" / "recording_features.csv"
DEFAULT_MODEL = ROOT / "models" / "tremor_rf_b2.pkl"
REQUIRED_COLUMNS = {
    "elapsed_ms",
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
}


def train(features_path: Path, model_path: Path) -> None:
    frame = pd.read_csv(features_path)
    columns = feature_columns(frame, "B2_time_frequency")
    if not columns:
        raise ValueError(f"No B2 features found in {features_path}")

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(frame[columns], frame["target"])
    artifact = {
        "model": model,
        "feature_columns": columns,
        "sampling_rate_hz": 50.0,
        "window_seconds": 3.0,
        "positive_class": "simulated_tremor",
        "training_recordings": len(frame),
        "training_subjects": int(frame["subject_id"].nunique()),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    print(f"Saved model: {model_path}")
    print(f"Training data: {len(frame)} recordings, {artifact['training_subjects']} subjects")


def extract_new_recording(path: Path) -> pd.DataFrame:
    time_frame = pd.read_csv(path, usecols=lambda column: column == "elapsed_ms")
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(REQUIRED_COLUMNS - header)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    if len(time_frame) < 2:
        raise ValueError("Input needs at least two timestamped samples")
    duration_ms = float(time_frame["elapsed_ms"].iloc[-1] - time_frame["elapsed_ms"].iloc[0])
    if duration_ms < 2980.0:
        raise ValueError(
            f"Input is {duration_ms / 1000.0:.2f}s; at least 3.00s is required"
        )

    row = pd.Series(
        {
            "recording_id": path.stem,
            "dataset_id": "inference",
            "subject_id": "unknown",
            "label": "unknown",
            # Joining an absolute path to ROOT preserves the absolute path.
            "relative_path": str(path.resolve()),
        }
    )
    return pd.DataFrame([recording_features(row)])


def predict(model_path: Path, csv_paths: list[Path]) -> None:
    # Pickle artifacts must only be loaded from a trusted local source.
    with model_path.open("rb") as file:
        artifact = pickle.load(file)

    columns = artifact["feature_columns"]
    model = artifact["model"]
    for path in csv_paths:
        features = extract_new_recording(path)
        missing = sorted(set(columns) - set(features.columns))
        if missing:
            raise ValueError(f"Model/input feature mismatch: {', '.join(missing)}")
        probability = float(model.predict_proba(features[columns])[0, 1])
        label = "simulated_tremor" if probability >= 0.5 else "non_tremor"
        print(f"{path}\t{label}\tprobability={probability:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run the recording-level simulated-tremor classifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="fit and save the classifier")
    train_parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)

    predict_parser = subparsers.add_parser("predict", help="score one or more IMU CSVs")
    predict_parser.add_argument("csv", type=Path, nargs="+")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args.features, args.model)
    else:
        predict(args.model, args.csv)


if __name__ == "__main__":
    main()
