# Program: Planner Adapter Service (FastAPI)
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Expose disc-planner controls to orchestrator HTTP clients."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

app = FastAPI(title="Planner Adapter")
SAVE_DIR = Path("./captures").resolve()


@app.post("/planner/start_shooting")
async def start_shooting() -> dict[str, bool]:
    return {"ok": True}


@app.post("/planner/set_save_dir")
async def set_save_dir(payload: dict[str, str]) -> dict[str, str | bool]:
    global SAVE_DIR
    SAVE_DIR = Path(payload["path"]).resolve()
    return {"ok": True, "save_dir": str(SAVE_DIR)}


# Created by Dr. Z. Bakhtiyorov
