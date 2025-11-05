# Program: Printer Retry Logic Tests
# Version: 0.2.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Ensure retry behaviour works without HTTP clients."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.api import Orchestrator
from orchestrator.clients import CameraModule
from orchestrator.config import AppConfig


class FlakyCamera(CameraModule):
    """Camera that fails the first N captures before succeeding."""

    def __init__(self, save_dir: Path, failures: int) -> None:
        super().__init__(save_dir=save_dir)
        self.failures = failures

    def capture(self, region=None, filename=None):  # type: ignore[override]
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("temporary lens issue")
        return super().capture(region=region, filename=filename)


def test_retry_succeeds_after_transient_failure(tmp_path: Path) -> None:
    cfg = AppConfig(save_dir=tmp_path, stabilization_ms=0, max_retries=2)
    camera = FlakyCamera(save_dir=tmp_path, failures=1)
    orchestrator = Orchestrator(cfg, camera=camera)
    orchestrator.prep_camera()

    result = orchestrator.start_from_printer(steps=1)
    attempt = result.captures[0]
    assert attempt.path is not None
    assert attempt.attempts == 2
    assert not result.warnings


def test_retry_emits_warning_after_exhaustion(tmp_path: Path) -> None:
    cfg = AppConfig(save_dir=tmp_path, stabilization_ms=0, max_retries=1)
    camera = FlakyCamera(save_dir=tmp_path, failures=3)
    orchestrator = Orchestrator(cfg, camera=camera)
    orchestrator.prep_camera()

    outcome = orchestrator.start_from_printer(steps=1)
    attempt = outcome.captures[0]
    assert attempt.path is None
    assert attempt.error is not None
    assert outcome.warnings


# Created by Dr. Z. Bakhtiyorov
