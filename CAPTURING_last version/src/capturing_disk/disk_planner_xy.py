# Program: Capturing Disk — XY planner
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

import json
from dataclasses import asdict
from math import cos, pi, sin
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .models import DiskParams, Plan, ScanParams, Waypoint


def detect_disk_circles(image_path: str) -> Tuple[int, int, int, int, int]:
    """Detect outer and inner circles of a disk using Hough transform.

    Args:
        image_path: Path to a top-view disk image.

    Returns:
        (cx, cy, r_outer, cx2, r_inner) — integer pixel coordinates/radii.

    Notes:
        If inner circle is not found, r_inner is set to 0 and cx2==cx.
    """

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    img_blur = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=img.shape[0] // 8,
        param1=100,
        param2=30,
        minRadius=max(10, int(min(img.shape) * 0.05)),
        maxRadius=int(min(img.shape) * 0.49),
    )

    if circles is None:
        raise RuntimeError("No circles detected. Try better contrast or params.")

    circles = np.uint16(np.around(circles[0]))
    # Outer circle = the largest radius
    outer = max(circles, key=lambda c: c[2])

    # Inner circle = among smaller ones near the center
    candidates = sorted(circles, key=lambda c: c[2])
    inner = None
    for c in candidates:
        d = np.hypot(c[0] - outer[0], c[1] - outer[1])
        if d < outer[2] * 0.2 and c[2] < outer[2] * 0.5:
            inner = c
            break

    cx, cy, r_outer = int(outer[0]), int(outer[1]), int(outer[2])
    if inner is None:
        return cx, cy, r_outer, cx, 0
    return cx, cy, r_outer, int(inner[0]), int(inner[2])


def raster_points_in_annulus(
    center_mm: Tuple[float, float],
    r_inner_mm: float,
    r_outer_mm: float,
    step_mm: float,
) -> Iterable[Tuple[float, float]]:
    """Generate raster grid points within annulus.

    The grid spans a bounding square and filters points by annulus mask.
    """

    (cx, cy) = center_mm
    # Bounding box
    x_min = cx - r_outer_mm
    x_max = cx + r_outer_mm
    y_min = cy - r_outer_mm
    y_max = cy + r_outer_mm

    y = y_min
    toggle = False
    while y <= y_max + 1e-9:
        row: List[Tuple[float, float]] = []
        x = x_min
        while x <= x_max + 1e-9:
            r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if r_inner_mm <= r <= r_outer_mm:
                row.append((x, y))
            x += step_mm
        # Zig-zag to minimize travel
        if toggle:
            row.reverse()
        toggle = not toggle
        for p in row:
            yield p
        y += step_mm


def spiral_points_in_annulus(
    center_mm: Tuple[float, float], r_inner_mm: float, r_outer_mm: float, step_mm: float
) -> Iterable[Tuple[float, float]]:
    """Generate spiral path within annulus."""

    (cx, cy) = center_mm
    # Archimedean spiral r = a + b*theta
    a = r_inner_mm
    b = step_mm / (2 * pi)
    theta = 0.0
    r = a
    while r <= r_outer_mm:
        x = cx + r * cos(theta)
        y = cy + r * sin(theta)
        yield (x, y)
        theta += step_mm / max(r, 1e-6)
        r = a + b * theta


def plan_from_image(
    image_path: str,
    pixel_size_mm: float,
    outer_diameter_mm: float | None,
    inner_diameter_mm: float | None,
    center_xy_mm: Tuple[float, float] = (0.0, 0.0),
    pattern: str = "raster",
    step_mm: float = 0.5,
    dwell_ms: int = 250,
    feed_mm_min: float = 1200.0,
) -> Plan:
    """Build plan from a top-view image with automatic circle detection.

    If diameters are not provided, they are derived from Hough circles and pixel size.
    """

    cx, cy, r_out_px, _, r_in_px = detect_disk_circles(image_path)

    if outer_diameter_mm is None:
        outer_diameter_mm = 2.0 * r_out_px * pixel_size_mm
    if inner_diameter_mm is None:
        inner_diameter_mm = 2.0 * r_in_px * pixel_size_mm

    disk = DiskParams(
        outer_mm=outer_diameter_mm,
        inner_mm=inner_diameter_mm,
        center_xy_mm=center_xy_mm,
        pixel_size_mm=pixel_size_mm,
    )
    scan = ScanParams(
        pattern="raster" if pattern not in {"raster", "spiral"} else pattern,
        step_mm=step_mm,
        dwell_ms=dwell_ms,
        feed_mm_min=feed_mm_min,
    )

    r_outer = disk.outer_mm / 2.0
    r_inner = disk.inner_mm / 2.0

    if scan.pattern == "raster":
        pts = raster_points_in_annulus(disk.center_xy_mm, r_inner, r_outer, scan.step_mm)
    else:
        pts = spiral_points_in_annulus(disk.center_xy_mm, r_inner, r_outer, scan.step_mm)

    path: List[Waypoint] = []
    first = True
    for (x, y) in pts:
        path.append(Waypoint(type="move", x=float(x), y=float(y)))
        path.append(Waypoint(type="dwell", ms=scan.dwell_ms))
        path.append(Waypoint(type="trigger", mode="camera"))
        first = False

    return Plan(disk=disk, scan=scan, path=path)


def save_plan_json(plan: Plan, out_json: str) -> None:
    """Serialize plan to JSON."""

    payload = {
        "disk": asdict(plan.disk),
        "scan": asdict(plan.scan),
        "path": [asdict(w) for w in plan.path],
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# Created by Dr. Z. Bakhtiyorov
