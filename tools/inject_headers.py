# Program: Repository Header Injector
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Insert standardized headers into Python files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

HEADER_TEMPLATE = """# Program: {program}\n# Version: {version}\n# Author: Dr. Zulfiyor Bakhtiyorov\n# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan\n# Year: {year}\n# License: MIT License\n\n"""
FOOTER = "\n# Created by Dr. Z. Bakhtiyorov\n"


def iter_python_files(path: Path) -> Iterable[Path]:
    for py_file in path.rglob("*.py"):
        if py_file.name == Path(__file__).name:
            continue
        yield py_file


def inject_header(path: Path, year: int, version: str) -> None:
    original = path.read_text(encoding="utf-8")
    header = HEADER_TEMPLATE.format(program=path.stem.replace("_", " ").title(), version=version, year=year)
    if original.startswith("# Program:"):
        body = original.split("\n", 6)[6]
        updated = header + body
    else:
        updated = header + original
    if not updated.rstrip().endswith("# Created by Dr. Z. Bakhtiyorov"):
        updated = updated.rstrip() + FOOTER
    path.write_text(updated + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject standardized headers into Python files.")
    parser.add_argument("--path", type=Path, default=Path("."), help="Target directory")
    parser.add_argument("--year", type=int, default=2025, help="Copyright year")
    parser.add_argument("--version", type=str, default="0.1.0", help="Version string")
    args = parser.parse_args()

    for py_file in iter_python_files(args.path):
        inject_header(py_file, args.year, args.version)


if __name__ == "__main__":
    main()

# Created by Dr. Z. Bakhtiyorov
