# Program: Capturing Disk — core models
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple


Pattern = Literal["raster", "spiral"]


@dataclass(slots=True)
class DiskParams:
    """Disk geometry parameters in millimeters.

    Attributes:
        outer_mm: Outer diameter of the disk in mm.
        inner_mm: Inner (hole) diameter in mm.
        center_xy_mm: Center coordinates (x, y) in mm in printer frame.
        pixel_size_mm: Pixel size (mm per pixel) for the provided image.
    """

    outer_mm: float
    inner_mm: float
    center_xy_mm: Tuple[float, float]
    pixel_size_mm: float


@dataclass(slots=True)
class ScanParams:
    """Scanning strategy and motion parameters.

    Attributes:
        pattern: Raster or spiral pattern.
        step_mm: Step between sampling points or lines in mm.
        dwell_ms: Dwell time at each capture point.
        feed_mm_min: Feed rate for G1 moves in mm/min.
    """

    pattern: Pattern
    step_mm: float
    dwell_ms: int
    feed_mm_min: float


@dataclass(slots=True)
class Waypoint:
    """A single path waypoint or action."""

    type: Literal["move", "dwell", "trigger"]
    x: Optional[float] = None
    y: Optional[float] = None
    ms: Optional[int] = None
    mode: Optional[str] = None


@dataclass(slots=True)
class Plan:
    """Full plan to be consumed by the printer interface or adapters."""

    disk: DiskParams
    scan: ScanParams
    path: List[Waypoint]


# Created by Dr. Z. Bakhtiyorov
