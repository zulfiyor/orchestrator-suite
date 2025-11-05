# Program: Capturing Disk — legacy CSV adapter
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import json

from ..models import Plan, Waypoint


def emit_legacy_csv(plan_json_path: str, out_csv: str) -> None:
    """Emit simple legacy CSV with MOVE/DWELL/SNAP rows.

    Format:
        MOVE,x,y
        DWELL,ms,
        SNAP,,
    """

    payload = json.loads(Path(plan_json_path).read_text(encoding="utf-8"))
    rows: list[list[str]] = []
    for w in payload["path"]:
        if w["type"] == "move":
            rows.append(["MOVE", f"{w['x']:.3f}", f"{w['y']:.3f}"])
        elif w["type"] == "dwell":
            rows.append(["DWELL", str(int(w.get("ms", 0))), ""])  # noqa: PLE1142
        elif w["type"] == "trigger":
            rows.append(["SNAP", "", ""])  # noqa: PLE1142

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# Created by Dr. Z. Bakhtiyorov
