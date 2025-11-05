# Program: Capturing Disk — G-code adapter
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models import Plan, Waypoint


def emit_gcode(plan: Plan | None = None, plan_json_path: str | None = None) -> str:
    """Convert plan to a G-code string.

    Notes:
        - Uses absolute positioning (G90) in millimeters (G21).
        - Camera trigger is left as a comment `; SNAP` to be mapped by firmware/M-code.
        - Dwell uses G4 P<ms> if supported.
    """

    if plan is None and plan_json_path is None:
        raise ValueError("Provide plan or plan_json_path")

    if plan is None:
        payload = json.loads(Path(plan_json_path).read_text(encoding="utf-8"))
        # Minimal typed reconstruction
        from ..models import DiskParams, ScanParams

        plan = Plan(
            disk=DiskParams(**payload["disk"]),
            scan=ScanParams(**payload["scan"]),
            path=[Waypoint(**w) for w in payload["path"]],
        )

    lines: list[str] = []
    lines.append("; Capturing Disk — auto-generated G-code")
    lines.append("G21 ; millimeters")
    lines.append("G90 ; absolute positioning")

    feed = max(1.0, plan.scan.feed_mm_min)
    for w in plan.path:
        if w.type == "move" and w.x is not None and w.y is not None:
            lines.append(f"G1 X{w.x:.3f} Y{w.y:.3f} F{feed:.1f}")
        elif w.type == "dwell" and w.ms is not None:
            lines.append(f"G4 P{int(w.ms)}")
        elif w.type == "trigger":
            lines.append("; SNAP")

    return "\n".join(lines) + "\n"


# Created by Dr. Z. Bakhtiyorov
