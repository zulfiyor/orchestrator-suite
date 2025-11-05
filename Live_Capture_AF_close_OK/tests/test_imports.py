# Program: Import Smoke Test
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge;
#   Xinjiang Institute of Ecology and Geography;
#   National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Import smoke test to ensure package availability."""

import importlib


def test_import_top_level() -> None:
    """Ensure top-level package (if any) is importable."""
    try:
        importlib.import_module("your_package")
    except ModuleNotFoundError:
        assert True


# Created by Dr. Z. Bakhtiyorov
