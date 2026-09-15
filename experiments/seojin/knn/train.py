#!/usr/bin/env python3
"""k-Nearest Neighbors tremor classifier: LOSO + cross-dataset on the shared 3 s features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_baseline_analysis as baseline  # noqa: E402

FEATURES = ROOT / "results" / "recording_features_3s.csv"
OUT = Path(__file__).resolve().parent
FEATURE_SET = "B2_time_frequency"
K = 5  # fixed before running; no test-set tuning


def build_classifier():
    # ponytail: k fixed at 5, distance weighting; add inner CV over k if a real dataset needs it
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K, weights="distance"))


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = baseline.feature_columns(frame, FEATURE_SET)
    folds = [("LOSO", s, (frame["subject_id"] != s).to_numpy(), (frame["subject_id"] == s).to_numpy())
             for s in sorted(frame["subject_id"].unique())]
    for a, b in (("dataset_a", "dataset_b"), ("dataset_b", "dataset_a")):
        folds.append(("cross_dataset", f"{a}_to_{b}",
                      (frame["dataset_id"] == a).to_numpy(), (frame["dataset_id"] == b).to_numpy()))

    metrics, predictions = [], []
    for evaluation, fold, train_mask, test_mask in folds:
        clf = build_classifier().fit(frame.loc[train_mask, columns], frame.loc[train_mask, "target"])
        probability = clf.predict_proba(frame.loc[test_mask, columns])[:, 1]
        y_true = frame.loc[test_mask, "target"].to_numpy()
        metrics.append({"evaluation": evaluation, "fold": fold, "classifier": "kNN",
                        "feature_set": FEATURE_SET, **baseline.score_predictions(y_true, probability)})
        for index, p in zip(frame.index[test_mask], probability):
            predictions.append({"evaluation": evaluation, "fold": fold,
                                "recording_id": frame.at[index, "recording_id"],
                                "subject_id": frame.at[index, "subject_id"],
                                "target": int(frame.at[index, "target"]), "probability": float(p)})
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def main() -> None:
    frame = pd.read_csv(FEATURES)
    metrics, predictions = evaluate(frame)
    metrics.to_csv(OUT / "results.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)

    cols = ["balanced_accuracy", "sensitivity", "specificity", "macro_f1"]
    print(metrics[["evaluation", "fold"] + cols].to_string(index=False, float_format="%.4f"))
    loso = metrics[metrics["evaluation"] == "LOSO"]
    print("\nLOSO mean:", loso[cols].mean().round(4).to_dict())
    print("LOSO std balanced_accuracy:", round(float(loso["balanced_accuracy"].std()), 4))
    wrong = predictions[(predictions["probability"] >= 0.5) != (predictions["target"] == 1)]
    print("\nMisclassified:")
    print(wrong.to_string(index=False, float_format="%.3f") if len(wrong) else "none")


def _check() -> None:
    frame = pd.read_csv(FEATURES)
    metrics, predictions = evaluate(frame)
    assert set(metrics["evaluation"]) == {"LOSO", "cross_dataset"}
    assert len(predictions[predictions["evaluation"] == "LOSO"]) == len(frame)
    assert predictions["probability"].between(0, 1).all()
    assert metrics["balanced_accuracy"].between(0, 1).all()
    print("ok")


if __name__ == "__main__":
    _check() if "--check" in sys.argv else main()
