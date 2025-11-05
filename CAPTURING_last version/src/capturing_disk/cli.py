# Program: Capturing Disk — CLI
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .adapters.gcode_adapter import emit_gcode
from .disk_planner_xy import plan_from_image, save_plan_json

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def plan(
    image: str = typer.Option(..., help="Path to disk image"),
    out_json: str = typer.Option("out/plan.json", help="Output plan JSON"),
    out_gcode: Optional[str] = typer.Option(None, help="Optional G-code output"),
    pixel_size_mm: float = typer.Option(0.05, help="Pixel size in mm"),
    outer_mm: Optional[float] = typer.Option(None, help="Outer diameter in mm (optional)"),
    inner_mm: Optional[float] = typer.Option(None, help="Inner diameter in mm (optional)"),
    cx: float = typer.Option(0.0, help="Center X in mm"),
    cy: float = typer.Option(0.0, help="Center Y in mm"),
    pattern: str = typer.Option("raster", help="Pattern: raster|spiral"),
    step_mm: float = typer.Option(0.5, help="Step in mm"),
    dwell_ms: int = typer.Option(250, help="Dwell per point in ms"),
    feed_mm_min: float = typer.Option(1200.0, help="Feed rate in mm/min"),
):
    """Plan path from an image and optionally emit G-code."""

    p = plan_from_image(
        image_path=image,
        pixel_size_mm=pixel_size_mm,
        outer_diameter_mm=outer_mm,
        inner_diameter_mm=inner_mm,
        center_xy_mm=(cx, cy),
        pattern=pattern,
        step_mm=step_mm,
        dwell_ms=dwell_ms,
        feed_mm_min=feed_mm_min,
    )
    save_plan_json(p, out_json)
    console.print(f"[green]Plan saved:[/] {out_json}")

    if out_gcode:
        gcode = emit_gcode(plan=p)
        Path(out_gcode).parent.mkdir(parents=True, exist_ok=True)
        Path(out_gcode).write_text(gcode, encoding="utf-8")
        console.print(f"[green]G-code saved:[/] {out_gcode}")


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Created by Dr. Z. Bakhtiyorov
