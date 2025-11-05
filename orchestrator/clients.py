# Program: Orchestrator HTTP Clients
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Thin HTTP clients for printer, camera, and planner services with retry."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import requests


class BaseClient:
    def __init__(self, base_url: str, retries: int = 3, backoff_s: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.backoff_s = backoff_s

    def _post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                return requests.post(url, json=json, timeout=timeout)
            except Exception:  # pragma: no cover - network failures
                attempt += 1
                if attempt > self.retries:
                    raise
                time.sleep(self.backoff_s * attempt)


class CameraClient(BaseClient):
    def prepare(self) -> None:
        self._post("/camera/prepare")

    def keepalive(self) -> None:
        self._post("/camera/keepalive")

    def set_save_dir(self, path: str) -> None:
        self._post("/camera/set_save_dir", json={"path": path})

    def capture(
        self,
        path: str,
        autofocus: bool = True,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"path": path, "autofocus": autofocus, "region": region}
        return self._post("/camera/capture", json=payload).json()


class PrinterClient(BaseClient):
    def start(self) -> None:
        self._post("/printer/start")

    def capture_at_stop(self) -> None:
        self._post("/printer/capture_at_stop")

    def set_save_dir(self, path: str) -> None:
        self._post("/printer/set_save_dir", json={"path": path})


class PlannerClient(BaseClient):
    def start_shooting(self) -> None:
        self._post("/planner/start_shooting")

    def set_save_dir(self, path: str) -> None:
        self._post("/planner/set_save_dir", json={"path": path})


# Created by Dr. Z. Bakhtiyorov
