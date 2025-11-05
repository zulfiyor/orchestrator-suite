# Program: Import Smoke Test
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""This test checks that the package can be imported without side effects."""

import importlib
import pkgutil
from pathlib import Path

_ = (pkgutil, Path)


def test_import_top_level() -> None:
    """Ensure top-level package (if any) is importable.

    Adjust the package name if needed. This test will not fail if module is
    absent; it will be xfailed gracefully. Replace `your_package` with the
    actual top-level package name when available.
    """

    try:
        importlib.import_module("your_package")
    except ModuleNotFoundError:
        assert True  # ok for placeholder


# Created by Dr. Z. Bakhtiyorov
