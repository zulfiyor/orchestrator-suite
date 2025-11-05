# Program: Orchestrator Event Models
# Version: 0.2.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Lightweight dataclasses for status and control events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Tuple


@dataclass(slots=True)
class CaptureRequest:
    path: str
    autofocus: bool = True
    region: Optional[Tuple[int, int, int, int]] = None


@dataclass(slots=True)
class CaptureResponse:
    ok: bool
    filepath: Optional[str] = None
    error: Optional[str] = None


@dataclass(slots=True)
class StatusEvent:
    ts: datetime
    source: Literal["printer", "camera", "planner", "orchestrator"]
    level: Literal["info", "warn", "error"]
    message: str


# Created by Dr. Z. Bakhtiyorov
