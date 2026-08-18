#!/usr/bin/env python3
"""Build a recording-level manifest without modifying raw data."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "manifest.csv"
EXPECTED_COLUMNS = [
    "elapsed_ms",
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "angle_x_deg",
    "angle_y_deg",
    "angle_z_deg",
]


def parse_identity(dataset_id: str, folder: str) -> tuple[str, str]:
    if dataset_id == "dataset_a":
        if folder.startswith("nontremor_"):
            return folder.removeprefix("nontremor_"), "non_tremor"
        if folder.startswith("tremor_"):
            return folder.removeprefix("tremor_"), "tremor"
    elif dataset_id == "dataset_b":
        if folder.endswith("_still"):
            return folder.removesuffix("_still"), "non_tremor"
        if folder.endswith("_tremor"):
            return folder.removesuffix("_tremor"), "tremor"
    raise ValueError(f"Cannot infer subject and label from {dataset_id}/{folder}")


def inspect_recording(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
        timestamps = [float(row["elapsed_ms"]) for row in reader]

    if len(timestamps) < 2:
        raise ValueError(f"Recording has fewer than two rows: {path}")

    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    positive_deltas = [delta for delta in deltas if delta > 0]
    if not positive_deltas:
        raise ValueError(f"Recording has no increasing timestamps: {path}")

    return {
        "n_rows": len(timestamps),
        "duration_s": round((timestamps[-1] - timestamps[0]) / 1000, 6),
        "median_dt_ms": round(median(positive_deltas), 6),
        "min_dt_ms": round(min(positive_deltas), 6),
        "max_dt_ms": round(max(positive_deltas), 6),
        "non_increasing_steps": sum(delta <= 0 for delta in deltas),
        "gaps_over_100ms": sum(delta > 100 for delta in positive_deltas),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(RAW_ROOT.glob("dataset_*/*/*.csv")):
        dataset_id = path.relative_to(RAW_ROOT).parts[0]
        folder = path.parent.name
        subject_id, label = parse_identity(dataset_id, folder)
        rows.append(
            {
                "recording_id": f"{dataset_id}:{path.stem}",
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "label": label,
                "relative_path": path.relative_to(ROOT).as_posix(),
                **inspect_recording(path),
            }
        )

    if not rows:
        raise RuntimeError(f"No CSV files found below {RAW_ROOT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} recordings to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
