# Program: Header & Footer Injector for Python Files
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge;
#   Xinjiang Institute of Ecology and Geography;
#   National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Prepend a standardized header block and append a developer signature line.

- Header includes: Program name (derived from filename), version, author, affiliations,
  year, and license. The program name is dynamic per file.
- Footer line: "Created by Dr. Z. Bakhtiyorov" (ensured at EOF).
- Safe idempotency: the script skips files that already contain our header marker.
- Keeps logic and architecture intact (no code reordering here).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

HEADER_MARKER = "# Program:"
FOOTER_LINE = "# Created by Dr. Z. Bakhtiyorov"


def build_header(program: str, year: int, license_name: str) -> str:
    lines = [
        f"# Program: {program}",
        "# Version: 0.1.0",
        "# Author: Dr. Zulfiyor Bakhtiyorov",
        (
            "# Affiliations: University of Cambridge; "
            "Xinjiang Institute of Ecology and Geography; "
            "National Academy of Sciences of Tajikistan"
        ),
        f"# Year: {year}",
        f"# License: {license_name} License",
        "",
    ]
    return "\n".join(lines)


def iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        yield path


def has_header(text: str) -> bool:
    return text.lstrip().startswith(HEADER_MARKER)


def ensure_footer(text: str) -> str:
    stripped = text.rstrip("\n")
    if stripped.endswith(FOOTER_LINE):
        return stripped + "\n"
    return stripped + "\n\n" + FOOTER_LINE + "\n"


def inject(path: Path, year: int, license_name: str) -> None:
    src = path.read_text(encoding="utf-8")
    if not has_header(src):
        program = path.stem.replace("_", " ").title()
        header = build_header(program, year, license_name)
        src = header + src
    src = ensure_footer(src)
    path.write_text(src, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject standardized headers/footers into Python files."
    )
    parser.add_argument("--path", type=Path, default=Path("."))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--license", dest="license_name", type=str, default="MIT")
    args = parser.parse_args()

    for py_file in iter_py_files(args.path):
        inject(py_file, args.year, args.license_name)


if __name__ == "__main__":
    main()

# Created by Dr. Z. Bakhtiyorov
