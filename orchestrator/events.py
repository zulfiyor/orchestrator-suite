# Program: Orchestrator Event Models
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Pydantic models for status and control events."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Tuple

from pydantic import BaseModel


class CaptureRequest(BaseModel):
    path: str
    autofocus: bool = True
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class CaptureResponse(BaseModel):
    ok: bool
    filepath: Optional[str] = None
    error: Optional[str] = None


class StatusEvent(BaseModel):
    ts: datetime
    source: Literal["printer", "camera", "planner", "orchestrator"]
    level: Literal["info", "warn", "error"]
    message: str


# Created by Dr. Z. Bakhtiyorov
