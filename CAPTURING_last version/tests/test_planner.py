# Program: Capturing Disk — tests
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import json
from pathlib import Path

from capturing_disk.models import Plan
from capturing_disk.disk_planner_xy import save_plan_json


def test_save_plan_json(tmp_path: Path) -> None:
    from capturing_disk.models import DiskParams, ScanParams, Waypoint

    plan = Plan(
        disk=DiskParams(outer_mm=100.0, inner_mm=10.0, center_xy_mm=(0.0, 0.0), pixel_size_mm=0.05),
        scan=ScanParams(pattern="raster", step_mm=1.0, dwell_ms=100, feed_mm_min=1000.0),
        path=[Waypoint(type="move", x=0.0, y=0.0), Waypoint(type="dwell", ms=100), Waypoint(type="trigger", mode="camera")],
    )
    out = tmp_path / "plan.json"
    save_plan_json(plan, str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "disk" in payload and "scan" in payload and "path" in payload
    assert len(payload["path"]) == 3

# Created by Dr. Z. Bakhtiyorov
