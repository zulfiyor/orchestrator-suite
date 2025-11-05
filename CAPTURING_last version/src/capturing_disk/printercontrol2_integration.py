# Program: Capturing Disk — printercontrol2 integration helpers
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .adapters.gcode_adapter import emit_gcode


def launch_planner_subprocess(image_path: str, out_json: str | None = None) -> str:
    """
    Launch GUI planner in a subprocess and return the generated plan path.

    Ensures local 'src' is on PYTHONPATH so 'capturing_disk' can be imported
    even when the package is not installed system-wide.
    """
    if out_json is None:
        out_json = str(Path("out/plan.json").absolute())

    cmd = [
        sys.executable,
        "-m",
        "capturing_disk.gui_planner",
        "--image",
        image_path,
        "--out-json",
        out_json,
    ]

    # Build environment with PYTHONPATH that includes local 'src' if we run from a src checkout
    env = os.environ.copy()
    try:
        import importlib.util

        spec = importlib.util.find_spec("capturing_disk")
        if spec and spec.submodule_search_locations:
            # spec.submodule_search_locations[0] -> ...\src\capturing_disk
            pkg_dir = Path(list(spec.submodule_search_locations)[0])
            src_dir = str(pkg_dir.parent)  # ...\src
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")
    except Exception:
        # non-fatal: if anything goes wrong, just proceed with current env
        pass

    proc = subprocess.run(
        cmd,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Planner failed (code {proc.returncode}).\n\nSTDERR:\n{proc.stderr}"
        )

    return out_json


def plan_to_gcode(plan_json_path: str) -> str:
    """Convert a plan JSON to G-code string for immediate sending."""
    return emit_gcode(plan=None, plan_json_path=plan_json_path)



# Example usage inside your printercontrol2.py:
# plan_path = launch_planner_subprocess(image_path)
# gcode = plan_to_gcode(plan_path)
# send_to_printer(gcode)

# Created by Dr. Z. Bakhtiyorov
