#!/usr/bin/env python3
"""Parallel sync of PADS patients/ and movement/ from the public S3 mirror (no aws CLI needed)."""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "parkinsons-disease-smartwatch/1.0.0/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEST = Path(os.environ.get("PADS_RAW", Path(__file__).resolve().parent / "raw")) / "physionet.org" / "files"
SUBDIRS = ("patients/", "movement/")  # ponytail: skip preprocessed/questionnaire; add if needed


def list_keys(prefix: str) -> list[tuple[str, int]]:
    keys, token = [], ""
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000{token}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        keys += [(c.find(NS + "Key").text, int(c.find(NS + "Size").text)) for c in root.iter(NS + "Contents")]
        nxt = root.find(NS + "NextContinuationToken")
        if nxt is None:
            return keys
        token = "&continuation-token=" + urllib.parse.quote(nxt.text)


def fetch(item: tuple[str, int]) -> str:
    key, size = item
    out = DEST / key
    if out.exists() and out.stat().st_size == size:
        return ""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as r, tmp.open("wb") as f:
                f.write(r.read())
            tmp.rename(out)
            return key
        except Exception as e:  # noqa: BLE001
            err = e
    return f"FAILED {key}: {err}"


def main() -> None:
    items = [k for d in SUBDIRS for k in list_keys(PREFIX + d)]
    print(f"{len(items)} files, {sum(s for _, s in items) / 1e6:.0f} MB", flush=True)
    done = 0
    with ThreadPoolExecutor(16) as pool:
        for i, msg in enumerate(pool.map(fetch, items), 1):
            if msg.startswith("FAILED"):
                print(msg, flush=True)
            elif msg:
                done += 1
            if i % 500 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print(f"DONE downloaded={done}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
