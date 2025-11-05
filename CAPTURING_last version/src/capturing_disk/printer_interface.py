# Program: Capturing Disk — abstract printer interface
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Institutions: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class PrinterPort(Protocol):
    """Minimal protocol of a printer connection (e.g., serial)."""

    def write(self, data: bytes) -> int:  # pragma: no cover - transport specific
        ...


class Printer(ABC):
    """Abstract printer for sending commands."""

    @abstractmethod
    def send_gcode(self, gcode: str) -> None:  # pragma: no cover - transport specific
        """Send a block of G-code to the device."""
        raise NotImplementedError


# Created by Dr. Z. Bakhtiyorov
