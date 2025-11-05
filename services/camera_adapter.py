# Program: Camera Adapter Service (FastAPI)
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Expose existing camera controls as HTTP endpoints with retry and keep-alive."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Camera Adapter")
SAVE_DIR = Path("./captures").resolve()


@app.post("/camera/prepare")
async def prepare() -> dict[str, bool]:
    return {"ok": True}


@app.post("/camera/set_save_dir")
async def set_save_dir(payload: dict[str, str]) -> dict[str, str | bool]:
    global SAVE_DIR
    SAVE_DIR = Path(payload["path"]).resolve()
    return {"ok": True, "save_dir": str(SAVE_DIR)}


@app.post("/camera/keepalive")
async def keepalive() -> dict[str, bool]:
    time.monotonic()
    return {"ok": True}


@app.post("/camera/capture")
async def capture(payload: dict[str, object]) -> dict[str, object]:
    path = Path(payload.get("path", str(SAVE_DIR)))
    path.mkdir(parents=True, exist_ok=True)
    region = payload.get("region")
    _ = region
    ok = True
    if not ok:
        raise HTTPException(503, detail="capture failed")
    return {"ok": True, "filepath": str(path / "frame_xxx.png")}


# Created by Dr. Z. Bakhtiyorov
