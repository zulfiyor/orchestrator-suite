# Program: Orchestrator Integration Tests
# Version: 0.2.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Validate that the orchestrator wires planner calls from inside the printer."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.api import Orchestrator
from orchestrator.clients import CameraModule, PlannerModule, PrinterModule
from orchestrator.config import AppConfig


@pytest.fixture()
def orchestrator(tmp_path: Path) -> Orchestrator:
    cfg = AppConfig(save_dir=tmp_path, stabilization_ms=0, max_retries=2)
    camera = CameraModule(save_dir=tmp_path)
    planner = PlannerModule(save_dir=tmp_path)
    printer = PrinterModule(
        camera=camera,
        planner=planner,
        stabilization_sleep=lambda _: None,
        max_retries=cfg.max_retries,
        stabilization_ms=cfg.stabilization_ms,
        save_dir=tmp_path,
    )
    orchestrator = Orchestrator(cfg, camera=camera, planner=planner, printer=printer)
    return orchestrator


def test_printer_calls_planner(orchestrator: Orchestrator, tmp_path: Path) -> None:
    orchestrator.set_save_dir(tmp_path)
    orchestrator.prep_camera()
    result = orchestrator.start_from_printer(steps=3)

    assert len(result.plan_steps) == 3
    assert result.captures and len(result.captures) == 3
    assert orchestrator.planner.start_calls == 1
    assert all(attempt.path is not None for attempt in result.captures)


def test_planner_button_delegates_to_printer(orchestrator: Orchestrator, tmp_path: Path) -> None:
    orchestrator.prep_camera()
    output = orchestrator.start_from_planner(steps=1)
    assert len(output.captures) == 1
    assert orchestrator.planner.start_calls == 1
    assert output.captures[0].path is not None


# Created by Dr. Z. Bakhtiyorov
