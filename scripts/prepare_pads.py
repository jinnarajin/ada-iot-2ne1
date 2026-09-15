#!/usr/bin/env python3
"""Convert locally downloaded PADS recordings to the common IMU CSV schema and extract B2 features.

Label rule: Healthy -> non_tremor; Essential Tremor or a disease_comment mentioning tremor /
mixed type -> tremor; all other conditions (akinetic/rigid PD, dystonia, MS, ...) are excluded
because the recordings carry no tremor evidence.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_baseline_analysis import ROOT, recording_features

PADS = ROOT / "data_sources" / "seojin" / "pads"
RAW = Path(os.environ.get("PADS_RAW", ROOT.parent / "data_sources" / "seojin" / "pads" / "raw")) / "physionet.org" / "files" / "parkinsons-disease-smartwatch" / "1.0.0"
PREPARED = PADS / "prepared"
MANIFEST = ROOT / "data" / "pads_manifest.csv"
FEATURES = ROOT / "results" / "pads_recording_features.csv"
FS_HZ = 100.0
DROP_SECONDS = 0.5  # Watch vibration cue at recording start
COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


def convert(txt: Path) -> Path:
    subject, task, wrist = txt.stem.split("_")
    raw = np.loadtxt(txt, delimiter=",")[int(FS_HZ * DROP_SECONDS):, 1:]
    raw[:, 3:] = np.degrees(raw[:, 3:])  # rad/s -> deg/s
    frame = pd.DataFrame(raw, columns=COLUMNS)
    frame.insert(0, "elapsed_ms", np.arange(len(frame)) * (1000.0 / FS_HZ))
    out = PREPARED / subject / f"{task}_{wrist}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def label_for(patient: dict) -> str | None:
    if patient["condition"] == "Healthy":
        return "non_tremor"
    if patient["condition"] == "Essential Tremor" or re.search(
        r"tremor|mixed type", patient["disease_comment"], re.IGNORECASE
    ):
        return "tremor"
    return None


def main() -> None:
    rows = []
    for txt in sorted((RAW / "movement" / "timeseries").glob("*.txt")):
        subject = txt.stem.split("_")[0]
        patient = json.loads((RAW / "patients" / f"patient_{subject}.json").read_text())
        label = label_for(patient)
        if label is None:
            continue
        try:
            out = convert(txt)
        except ValueError:  # partially downloaded file
            print(f"skip (truncated): {txt.name}")
            continue
        rows.append(
            {
                "recording_id": f"pads:{txt.stem}",
                "dataset_id": "pads",
                "subject_id": subject,
                "label": label,
                "condition": patient["condition"],
                "relative_path": str(out.relative_to(ROOT)),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)
    features = pd.DataFrame([recording_features(row) for _, row in manifest.iterrows()])
    features.to_csv(FEATURES, index=False)
    print(f"{len(manifest)} recordings, {manifest.subject_id.nunique()} subjects -> {FEATURES}")
    print(manifest.groupby(["subject_id", "condition", "label"]).size())


if __name__ == "__main__":
    main()
