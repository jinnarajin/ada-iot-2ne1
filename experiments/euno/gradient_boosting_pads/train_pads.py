#!/usr/bin/env python3
"""Evaluate the gradient boosting classifier on the PADS dataset alone.

This is the open-data counterpart to ``train.py``. Nothing from ``dataset_a`` or
``dataset_b`` is used: the model is trained and tested only on PADS, so the
question it answers is "does this feature set and model separate real patient
tremor from healthy controls", not "does simulated tremor transfer".

Preprocessing is imported unchanged from ``scripts/run_baseline_analysis.py``
(50 Hz resampling, 3-second windows, recording-level aggregation), so the
features are exactly the ones used for the team's own recordings.

Two differences from ``train.py`` are forced by the dataset itself:

* With 160 participants, leave-one-subject-out would mean 160 refits per model.
  Subject-grouped stratified 5-fold cross-validation is the standard analogue and
  gives the same guarantee that no participant appears in both train and test.
* Parkinsonian tremor is frequently one-sided, and PADS labels the participant,
  not the wrist. Recording-level scoring therefore counts a non-shaking wrist of
  a patient as a miss. A participant-level view is reported alongside, taking the
  maximum probability over that participant's recordings, which matches the
  clinical reading "tremor present in at least one hand".
* The fixed 0.5 threshold is badly placed for this data: both classes sit above
  it, so the participant-level view scores 0.63 despite an AUROC of 0.96. The
  decision threshold is therefore also chosen, using an inner subject-grouped
  split of the training fold only. Both the fixed and the selected threshold are
  reported so the size of that effect stays visible.

Usage:
    python experiments/euno/gradient_boosting/train_pads.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
# the fixed hyperparameters live in the sibling gradient boosting experiment
sys.path.insert(0, str(ROOT / "experiments" / "euno" / "gradient_boosting"))

import run_baseline_analysis as baseline  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PADS_DIR = ROOT / "data" / "processed" / "dataset_c_pads"
PADS_MANIFEST = PADS_DIR / "manifest_dataset_c.csv"
FEATURE_CACHE = RESULTS / "pads_recording_features_3s.csv"
FEATURE_SETS = ("B1_time", "B2_time_frequency")
CLASSIFIERS = ("LightGBM", "XGBoost")
N_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 42


def build_features(force: bool) -> pd.DataFrame:
    """Extract recording-level features, caching the result."""
    if FEATURE_CACHE.exists() and not force:
        return pd.read_csv(FEATURE_CACHE)
    if not PADS_MANIFEST.exists():
        raise SystemExit(
            f"{PADS_MANIFEST} not found. Run "
            "data_sources/euno/pads/download_or_prepare.py first."
        )
    manifest = pd.read_csv(PADS_MANIFEST)
    print(f"Extracting features from {len(manifest)} PADS recordings ...", flush=True)
    rows: list[dict[str, object]] = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        features = baseline.recording_features(row)
        features["pads_task"] = row["pads_task"]
        features["pads_wrist"] = row["pads_wrist"]
        features["pads_condition"] = row["pads_condition"]
        rows.append(features)
        if position % 200 == 0 or position == len(manifest):
            print(f"  {position}/{len(manifest)}", flush=True)
    frame = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    frame.to_csv(FEATURE_CACHE, index=False)
    return frame


def score_at(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    """Confusion-based metrics at an explicit threshold, plus threshold-free ones."""
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n_test": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    """Threshold that maximises balanced accuracy on the given predictions."""
    candidates = np.unique(np.round(probability, 3))
    scores = [
        balanced_accuracy_score(target, (probability >= value).astype(int))
        for value in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def inner_out_of_fold(
    frame: pd.DataFrame, train_index: np.ndarray, columns: list[str], classifier_name: str
) -> pd.DataFrame:
    """Out-of-fold probabilities inside the training fold.

    The threshold must not be chosen on predictions the model has already fitted,
    because a boosted tree is overconfident on its own training data. An inner
    subject-grouped split produces honest probabilities within the training fold,
    and the threshold picked on those transfers to the outer test fold without
    ever touching it.
    """
    import train as gb  # sibling experiment: fixed hyperparameters

    subset = frame.iloc[train_index]
    inner = StratifiedGroupKFold(
        n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    probability = np.zeros(len(subset), dtype=float)
    for inner_train, inner_test in inner.split(
        subset, subset["target"], groups=subset["subject_id"]
    ):
        model = gb.build_classifier(classifier_name)
        model.fit(subset.iloc[inner_train][columns], subset.iloc[inner_train]["target"])
        probability[inner_test] = model.predict_proba(
            subset.iloc[inner_test][columns]
        )[:, 1]
    return pd.DataFrame(
        {
            "subject_id": subset["subject_id"].to_numpy(),
            "target": subset["target"].to_numpy(),
            "probability": probability,
        }
    )


def by_participant(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: the maximum probability over their recordings.

    Parkinsonian tremor is often unilateral and intermittent, so a participant
    counts as detected when any recording crosses the threshold.
    """
    return frame.groupby("subject_id", as_index=False).agg(
        target=("target", "first"), probability=("probability", "max")
    )


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-grouped stratified 5-fold cross-validation."""
    import train as gb  # sibling experiment: fixed hyperparameters

    subjects = frame["subject_id"].to_numpy()
    targets = frame["target"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame, targets, groups=subjects), start=1
    ):
        train_mask = np.zeros(len(frame), dtype=bool)
        train_mask[train_index] = True
        test_mask = ~train_mask
        for feature_set in FEATURE_SETS:
            columns = baseline.feature_columns(frame, feature_set)
            for classifier_name in CLASSIFIERS:
                model = gb.build_classifier(classifier_name)
                model.fit(
                    frame.loc[train_mask, columns], frame.loc[train_mask, "target"]
                )
                probability = model.predict_proba(frame.loc[test_mask, columns])[:, 1]

                test = pd.DataFrame(
                    {
                        "subject_id": frame.loc[test_mask, "subject_id"].to_numpy(),
                        "target": frame.loc[test_mask, "target"].to_numpy(),
                        "probability": probability,
                    }
                )
                inner = inner_out_of_fold(
                    frame, train_index, columns, classifier_name
                )

                recording_threshold = best_threshold(
                    inner["target"].to_numpy(), inner["probability"].to_numpy()
                )
                inner_participant = by_participant(inner)
                participant_threshold = best_threshold(
                    inner_participant["target"].to_numpy(),
                    inner_participant["probability"].to_numpy(),
                )
                test_participant = by_participant(test)

                for level, table, selected in (
                    ("recording", test, recording_threshold),
                    ("participant", test_participant, participant_threshold),
                ):
                    for rule, threshold in (
                        ("fixed_0.5", 0.5),
                        ("selected", selected),
                    ):
                        metrics.append(
                            {
                                "level": level,
                                "rule": rule,
                                "fold": fold,
                                "classifier": classifier_name,
                                "feature_set": feature_set,
                                "n_test_subjects": int(table["subject_id"].nunique())
                                if level == "recording"
                                else int(len(table)),
                                **score_at(
                                    table["target"].to_numpy(),
                                    table["probability"].to_numpy(),
                                    threshold,
                                ),
                            }
                        )

                for index, value in zip(frame.index[test_mask], probability):
                    predictions.append(
                        {
                            "fold": fold,
                            "classifier": classifier_name,
                            "feature_set": feature_set,
                            "recording_id": frame.at[index, "recording_id"],
                            "subject_id": frame.at[index, "subject_id"],
                            "pads_task": frame.at[index, "pads_task"],
                            "pads_wrist": frame.at[index, "pads_wrist"],
                            "pads_condition": frame.at[index, "pads_condition"],
                            "target": int(frame.at[index, "target"]),
                            "probability": float(value),
                            "recording_threshold": recording_threshold,
                            "participant_threshold": participant_threshold,
                        }
                    )
        print(f"  fold {fold}/{N_SPLITS} done", flush=True)

    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def per_task(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recording-level performance split by movement task."""
    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["classifier", "feature_set", "pads_task"])
    for (classifier, feature_set, task), group in grouped:
        score = baseline.score_predictions(
            group["target"].to_numpy(), group["probability"].to_numpy()
        )
        rows.append(
            {
                "classifier": classifier,
                "feature_set": feature_set,
                "pads_task": task,
                **score,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["classifier", "feature_set", "balanced_accuracy"], ascending=[True, True, False]
    )


def by_condition(predictions: pd.DataFrame) -> pd.DataFrame:
    """Detection rate per clinical condition, for the best configuration."""
    best = predictions[
        (predictions["classifier"] == "LightGBM")
        & (predictions["feature_set"] == "B2_time_frequency")
    ]
    aggregated = best.groupby(["subject_id", "pads_condition", "target"])[
        "probability"
    ].max().reset_index()
    aggregated["detected"] = (aggregated["probability"] >= 0.5).astype(int)
    summary = aggregated.groupby(["pads_condition", "target"]).agg(
        participants=("subject_id", "nunique"),
        flagged_as_tremor=("detected", "sum"),
        median_probability=("probability", "median"),
    ).reset_index()
    summary["rate"] = (summary["flagged_as_tremor"] / summary["participants"]).round(3)
    return summary


def plot_summary(metrics: pd.DataFrame) -> None:
    summary = (
        metrics.groupby(["level", "rule", "classifier", "feature_set"])[
            "balanced_accuracy"
        ]
        .mean()
        .reset_index()
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    width = 0.36
    panels = (
        ("recording", "Recording level"),
        ("participant", "Participant level (max over recordings)"),
    )
    for axis, (level, title) in zip(axes, panels):
        subset = summary[summary["level"] == level].set_index(
            ["rule", "classifier", "feature_set"]
        )
        labels, groups = [], []
        for rule, rule_label in (("fixed_0.5", "0.5 고정"), ("selected", "임계값 선택")):
            for classifier in CLASSIFIERS:
                labels.append(f"{classifier}\n{rule_label}")
                groups.append((rule, classifier))
        x = np.arange(len(groups))
        for offset, feature_set in zip((-width / 2, width / 2), FEATURE_SETS):
            values = [
                subset.loc[(rule, classifier, feature_set), "balanced_accuracy"]
                for rule, classifier in groups
            ]
            bars = axis.bar(x + offset, values, width=width, label=feature_set)
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylim(0.5, 1.02)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("Balanced accuracy")
        axis.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "PADS only — subject-grouped 5-fold cross-validation (160 participants)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(HERE / "pads_only_results.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-features", action="store_true", help="Ignore the feature cache."
    )
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", message=".*valid feature names.*")

    frame = build_features(arguments.rebuild_features)
    print(
        f"\n{len(frame)} recordings, {frame['subject_id'].nunique()} participants, "
        f"{int(frame['valid_windows'].sum())} valid windows, "
        f"{int(frame['rejected_windows'].sum())} rejected"
    )
    print(frame.groupby("label")["recording_id"].count().to_string())

    all_metrics, predictions = evaluate(frame)
    all_metrics.to_csv(HERE / "pads_only_results.csv", index=False)
    predictions.to_csv(HERE / "pads_only_predictions.csv", index=False)

    task_table = per_task(predictions)
    task_table.to_csv(HERE / "pads_only_by_task.csv", index=False)
    condition_table = by_condition(predictions)
    condition_table.to_csv(HERE / "pads_only_by_condition.csv", index=False)

    summary = (
        all_metrics.groupby(["level", "rule", "classifier", "feature_set"])
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            auroc_mean=("auroc", "mean"),
            threshold_mean=("threshold", "mean"),
        )
        .reset_index()
        .round(4)
    )
    summary.to_csv(HERE / "pads_only_summary.csv", index=False)
    plot_summary(all_metrics)

    (HERE / "pads_only_summary.json").write_text(
        json.dumps(
            {
                "n_recordings": int(len(frame)),
                "n_participants": int(frame["subject_id"].nunique()),
                "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS}) grouped by participant",
                "results": summary.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Per movement task (recording level) ===")
    print(
        task_table[
            ["classifier", "feature_set", "pads_task", "balanced_accuracy",
             "sensitivity", "specificity", "auroc"]
        ].round(4).to_string(index=False)
    )
    print("\n=== Per clinical condition (LightGBM B2, participant level) ===")
    print(condition_table.to_string(index=False))
    print(f"\nArtefacts written to {HERE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
