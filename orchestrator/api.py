# Program: Direct Orchestrator Runtime
# Version: 0.2.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Direct in-process orchestration without intermediate microservices."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

from .clients import CameraModule, PlannerModule, PrinterModule, PrintJobResult
from .config import AppConfig, load_config


class Orchestrator:
    """Coordinate printer, planner, and camera inside a single runtime."""

    def __init__(
        self,
        config: AppConfig,
        *,
        camera: Optional[CameraModule] = None,
        planner: Optional[PlannerModule] = None,
        printer: Optional[PrinterModule] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.config = config
        self.camera = camera or CameraModule(save_dir=config.save_dir)
        self.planner = planner or PlannerModule(save_dir=config.save_dir)
        sleep_callable = sleep_fn or (lambda _: None)
        self.printer = printer or PrinterModule(
            camera=self.camera,
            planner=self.planner,
            stabilization_sleep=sleep_callable,
            max_retries=config.max_retries,
            stabilization_ms=config.stabilization_ms,
            save_dir=config.save_dir,
        )
        self.set_save_dir(config.save_dir)

    def set_save_dir(self, path: Path) -> Path:
        resolved = path.resolve()
        self.camera.set_save_dir(resolved)
        self.planner.set_save_dir(resolved)
        self.printer.set_save_dir(resolved)
        return resolved

    def prep_camera(self) -> None:
        self.camera.prepare()

    def keep_camera_alive(self) -> None:
        self.camera.keepalive()

    def start_from_printer(
        self,
        *,
        steps: int,
        region: Optional[tuple[int, int, int, int]] = None,
    ) -> PrintJobResult:
        return self.printer.start(steps=steps, region=region)

    def start_from_planner(
        self,
        *,
        steps: int,
        region: Optional[tuple[int, int, int, int]] = None,
    ) -> PrintJobResult:
        return self.start_from_printer(steps=steps, region=region)


def create_default_orchestrator(config_path: Optional[Union[Path, str]] = None) -> Orchestrator:
    """Helper to create an orchestrator by loading YAML configuration."""

    cfg_path: Optional[Path]
    if config_path is None:
        cfg_path = None
    else:
        cfg_path = Path(config_path).expanduser().resolve()
    cfg = load_config(cfg_path) if cfg_path is not None else AppConfig(save_dir=Path("./captures").resolve())
    orchestrator = Orchestrator(config=cfg)
    return orchestrator


# Created by Dr. Z. Bakhtiyorov
