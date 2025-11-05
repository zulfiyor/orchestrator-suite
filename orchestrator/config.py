# Program: Orchestrator Config Utilities
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Shared configuration loading and save-dir propagation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class AppConfig:
    save_dir: Path
    stabilization_ms: int = 500
    capture_timeout_s: int = 10
    max_retries: int = 3
    keepalive_interval_s: int = 5


def load_config(path: Path) -> AppConfig:
    """Load YAML config with sane defaults."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        save_dir=Path(data.get("save_dir", "./captures")).resolve(),
        stabilization_ms=int(data.get("stabilization_ms", 500)),
        capture_timeout_s=int(data.get("capture_timeout_s", 10)),
        max_retries=int(data.get("max_retries", 3)),
        keepalive_interval_s=int(data.get("keepalive_interval_s", 5)),
    )


# Created by Dr. Z. Bakhtiyorov
