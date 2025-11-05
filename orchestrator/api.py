# Program: Orchestrator API (FastAPI)
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""FastAPI glue-service that coordinates printer, planner, and camera."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI

from .clients import CameraClient, PlannerClient, PrinterClient
from .config import AppConfig


def create_app(cfg: AppConfig, cam_url: str, prn_url: str, pln_url: str) -> FastAPI:
    app = FastAPI(title="Orchestrator")

    camera = CameraClient(cam_url)
    printer = PrinterClient(prn_url)
    planner = PlannerClient(pln_url)

    @app.post("/prep/camera")
    async def prep_camera() -> dict[str, object]:
        camera.prepare()
        camera.set_save_dir(str(cfg.save_dir))
        return {"ok": True}

    @app.post("/sync/save_dir")
    async def sync_save_dir(path: Optional[str] = None) -> dict[str, str | bool]:
        save_dir = Path(path) if path else cfg.save_dir
        printer.set_save_dir(str(save_dir))
        camera.set_save_dir(str(save_dir))
        planner.set_save_dir(str(save_dir))
        return {"ok": True, "save_dir": str(save_dir)}

    @app.post("/run/capture_once")
    async def capture_once(region: Optional[Tuple[int, int, int, int]] = None) -> dict[str, object]:
        await asyncio.sleep(cfg.stabilization_ms / 1000.0)
        resp = camera.capture(path=str(cfg.save_dir), region=region)
        return resp

    @app.post("/run/sequence")
    async def coordinated_sequence(
        steps: int,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> dict[str, object]:
        results = []
        printer.start()
        for _ in range(steps):
            await asyncio.sleep(cfg.stabilization_ms / 1000.0)
            res = camera.capture(path=str(cfg.save_dir), region=region)
            results.append(res)
        return {"ok": all(r.get("ok", False) for r in results), "frames": results}

    return app


# Created by Dr. Z. Bakhtiyorov
