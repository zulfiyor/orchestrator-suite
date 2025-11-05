# Program: Printer Adapter Service (FastAPI)
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Wrap printer controls with HTTP endpoints for orchestration."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

app = FastAPI(title="Printer Adapter")
SAVE_DIR = Path("./captures").resolve()


@app.post("/printer/start")
async def start() -> dict[str, bool]:
    return {"ok": True}


@app.post("/printer/set_save_dir")
async def set_save_dir(payload: dict[str, str]) -> dict[str, str | bool]:
    global SAVE_DIR
    SAVE_DIR = Path(payload["path"]).resolve()
    return {"ok": True, "save_dir": str(SAVE_DIR)}


@app.post("/printer/capture_at_stop")
async def capture_at_stop() -> dict[str, bool]:
    return {"ok": True}


# Created by Dr. Z. Bakhtiyorov
