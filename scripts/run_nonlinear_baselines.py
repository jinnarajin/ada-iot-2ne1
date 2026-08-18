#!/usr/bin/env python3
"""Evaluate fixed RBF-SVM and Random Forest nonlinear baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import run_baseline_analysis as baseline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FEATURES = RESULTS / "recording_features_3s.csv"
FEATURE_SETS = ("B1_time", "B2_time_frequency")
CLASSIFIERS = ("RBF_SVM", "Random_Forest")


def build_classifier(name: str):
    if name == "RBF_SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        )
    if name == "Random_Forest":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"Unknown classifier: {name}")


def fit_and_score(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_set: str,
    classifier_name: str,
) -> tuple[dict[str, float | int], np.ndarray]:
    columns = baseline.feature_columns(frame, feature_set)
    classifier = build_classifier(classifier_name)
    classifier.fit(frame.loc[train_mask, columns], frame.loc[train_mask, "target"])
    probability = classifier.predict_proba(frame.loc[test_mask, columns])[:, 1]
    return baseline.score_predictions(
        frame.loc[test_mask, "target"].to_numpy(), probability
    ), probability


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    for subject in sorted(frame["subject_id"].unique()):
        test_mask = (frame["subject_id"] == subject).to_numpy()
        for feature_set in FEATURE_SETS:
            for classifier_name in CLASSIFIERS:
                score, probability = fit_and_score(
                    frame, ~test_mask, test_mask, feature_set, classifier_name
                )
                metrics.append(
                    {
                        "evaluation": "LOSO",
                        "fold": subject,
                        "classifier": classifier_name,
                        "feature_set": feature_set,
                        **score,
                    }
                )
                for index, value in zip(frame.index[test_mask], probability):
                    predictions.append(
                        {
                            "evaluation": "LOSO",
                            "fold": subject,
                            "classifier": classifier_name,
                            "feature_set": feature_set,
                            "recording_id": frame.at[index, "recording_id"],
                            "target": frame.at[index, "target"],
                            "probability": value,
                        }
                    )

    for train_dataset, test_dataset in (
        ("dataset_a", "dataset_b"),
        ("dataset_b", "dataset_a"),
    ):
        train_mask = (frame["dataset_id"] == train_dataset).to_numpy()
        test_mask = (frame["dataset_id"] == test_dataset).to_numpy()
        fold = f"{train_dataset}_to_{test_dataset}"
        for feature_set in FEATURE_SETS:
            for classifier_name in CLASSIFIERS:
                score, probability = fit_and_score(
                    frame, train_mask, test_mask, feature_set, classifier_name
                )
                metrics.append(
                    {
                        "evaluation": "cross_dataset",
                        "fold": fold,
                        "classifier": classifier_name,
                        "feature_set": feature_set,
                        **score,
                    }
                )
                for index, value in zip(frame.index[test_mask], probability):
                    predictions.append(
                        {
                            "evaluation": "cross_dataset",
                            "fold": fold,
                            "classifier": classifier_name,
                            "feature_set": feature_set,
                            "recording_id": frame.at[index, "recording_id"],
                            "target": frame.at[index, "target"],
                            "probability": value,
                        }
                    )

    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def plot_metrics(metrics: pd.DataFrame) -> None:
    logistic = pd.read_csv(RESULTS / "baseline_metrics.csv")
    logistic = logistic[logistic["model"].isin(FEATURE_SETS)].copy()
    logistic["classifier"] = "Logistic_Regression"
    logistic["feature_set"] = logistic["model"]
    combined = pd.concat([logistic, metrics], ignore_index=True)
    summary = (
        combined.groupby(["evaluation", "classifier", "feature_set"])[
            "balanced_accuracy"
        ]
        .mean()
        .reset_index()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    classifiers = ("Logistic_Regression", "RBF_SVM", "Random_Forest")
    width = 0.36
    for axis, evaluation, title in zip(
        axes,
        ("LOSO", "cross_dataset"),
        ("LOSO mean", "Cross-dataset directional mean"),
    ):
        subset = summary[summary["evaluation"] == evaluation].set_index(
            ["classifier", "feature_set"]
        )
        x = np.arange(len(classifiers))
        for offset, feature_set in zip((-width / 2, width / 2), FEATURE_SETS):
            values = [
                subset.loc[(classifier, feature_set), "balanced_accuracy"]
                for classifier in classifiers
            ]
            axis.bar(x + offset, values, width=width, label=feature_set)
        axis.set_xticks(x, ["Logistic", "RBF-SVM", "Random Forest"])
        axis.set_ylim(0.75, 1.005)
        axis.set_title(title)
        axis.set_ylabel("Balanced accuracy")
        axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "nonlinear_baselines.png", dpi=180)
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(FEATURES)
    metrics, predictions = evaluate(frame)
    metrics.to_csv(RESULTS / "nonlinear_metrics.csv", index=False)
    predictions.to_csv(RESULTS / "nonlinear_predictions.csv", index=False)
    plot_metrics(metrics)
    print(
        metrics.groupby(["evaluation", "classifier", "feature_set"])[
            "balanced_accuracy"
        ]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )
    print("\nCross-dataset directions")
    print(
        metrics[metrics["evaluation"] == "cross_dataset"][[
            "fold", "classifier", "feature_set", "balanced_accuracy",
            "sensitivity", "specificity",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
