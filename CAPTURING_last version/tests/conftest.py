from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running tests without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
