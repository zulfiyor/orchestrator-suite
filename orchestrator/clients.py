# Program: Integrated Controller Modules
# Version: 0.2.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""In-process controllers for printer, planner, and camera cooperation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence


class CaptureError(RuntimeError):
    """Raised when the camera cannot produce a capture frame."""


@dataclass
class CameraModule:
    """Minimal camera controller keeping the device awake and capturing frames."""

    save_dir: Path
    capture_handler: Optional[Callable[[Path, Optional[tuple[int, int, int, int]], str], Path]] = None
    prepared: bool = False
    keepalive_count: int = 0
    capture_count: int = 0

    def set_save_dir(self, path: Path) -> None:
        self.save_dir = path.resolve()

    def prepare(self) -> None:
        self.prepared = True

    def keepalive(self) -> None:
        self.keepalive_count += 1
        self.prepared = True

    def capture(
        self,
        region: Optional[tuple[int, int, int, int]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        if not self.prepared:
            raise CaptureError("Camera is not prepared for capturing.")
        self.capture_count += 1
        target_name = filename or f"frame_{self.capture_count:04d}.png"
        target_path = (self.save_dir / target_name).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if self.capture_handler is not None:
            return self.capture_handler(target_path, region, target_name)

        target_path.touch(exist_ok=True)
        return target_path


@dataclass
class PlannerModule:
    """Simple disc planner facade that prepares capture filenames."""

    save_dir: Path
    start_calls: int = 0

    def set_save_dir(self, path: Path) -> None:
        self.save_dir = path.resolve()

    def outline(self) -> Path:
        outline_path = (self.save_dir / "planner_outline.json").resolve()
        outline_path.parent.mkdir(parents=True, exist_ok=True)
        outline_path.write_text("{}", encoding="utf-8")
        return outline_path

    def provide_path(self) -> Path:
        return self.save_dir

    def start_shooting(self, steps: int) -> Sequence[str]:
        self.start_calls += 1
        return tuple(f"frame_{self.start_calls}_{index:04d}.png" for index in range(steps))


@dataclass
class CaptureAttempt:
    """Result of a single capture attempt."""

    path: Optional[Path]
    attempts: int
    error: Optional[str] = None


@dataclass
class PrintJobResult:
    """Aggregate outcome of a coordinated print/capture run."""

    plan_steps: Sequence[str]
    captures: list[CaptureAttempt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PrinterModule:
    """Printer façade that calls the planner and triggers the camera directly."""

    camera: CameraModule
    planner: PlannerModule
    stabilization_sleep: Callable[[float], None]
    max_retries: int
    stabilization_ms: int
    save_dir: Path

    def set_save_dir(self, path: Path) -> None:
        resolved = path.resolve()
        self.save_dir = resolved
        self.camera.set_save_dir(resolved)
        self.planner.set_save_dir(resolved)

    def start(
        self,
        steps: int,
        region: Optional[tuple[int, int, int, int]] = None,
    ) -> PrintJobResult:
        if steps <= 0:
            return PrintJobResult(plan_steps=())

        if not self.camera.prepared:
            self.camera.prepare()

        plan_steps = self.planner.start_shooting(steps)
        result = PrintJobResult(plan_steps=plan_steps)

        for index, plan_name in enumerate(plan_steps):
            self._wait_for_stabilization()
            attempt = self._capture_with_retry(plan_name, region)
            result.captures.append(attempt)
            if attempt.error is not None:
                result.warnings.append(
                    f"Capture for step {index} failed after {attempt.attempts} attempts: {attempt.error}"
                )
        return result

    def _wait_for_stabilization(self) -> None:
        delay = max(self.stabilization_ms / 1000.0, 0.0)
        if delay:
            self.stabilization_sleep(delay)

    def _capture_with_retry(
        self,
        filename: str,
        region: Optional[tuple[int, int, int, int]],
    ) -> CaptureAttempt:
        attempts = 0
        last_error: Optional[Exception] = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                path = self.camera.capture(region=region, filename=filename)
                return CaptureAttempt(path=path, attempts=attempts)
            except Exception as exc:  # noqa: BLE001 - propagate warnings
                last_error = exc
        error_message = str(last_error) if last_error is not None else "Unknown capture failure"
        return CaptureAttempt(path=None, attempts=attempts, error=error_message)


# Created by Dr. Z. Bakhtiyorov
