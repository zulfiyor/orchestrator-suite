# Program: Orchestrator API Tests
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Basic tests for orchestrator endpoints with mocked clients."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import sys
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.api import create_app
from orchestrator.config import AppConfig


@pytest.fixture()
def app() -> Iterator[TestClient]:
    cfg = AppConfig(save_dir=Path("./captures").resolve())
    with patch("orchestrator.api.CameraClient") as camera_mock, patch(
        "orchestrator.api.PrinterClient"
    ) as printer_mock, patch("orchestrator.api.PlannerClient") as planner_mock:
        camera_mock.return_value.capture.return_value = {"ok": True, "filepath": "demo"}
        camera_mock.return_value.prepare.return_value = None
        printer_mock.return_value.start.return_value = None
        planner_mock.return_value.set_save_dir.return_value = None
        test_app = create_app(cfg, "http://cam", "http://printer", "http://planner")
        client = TestClient(test_app)
        yield client


def test_prep_camera(app: TestClient) -> None:
    response = app.post("/prep/camera")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_sequence_runs(app: TestClient) -> None:
    response = app.post("/run/sequence", json={"steps": 2})
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert len(data["frames"]) == 2


# Created by Dr. Z. Bakhtiyorov
