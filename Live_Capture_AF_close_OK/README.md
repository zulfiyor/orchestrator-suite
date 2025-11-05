# canon-remote

Internationalized, linted, and tested Python project. Comments and docstrings
are provided in English (Google-style), keeping logic and architecture intact.

## Features
- English-only comments and docstrings
- MIT License, citation metadata
- CI: Ruff, Black, mypy, pytest
- Pre-commit hooks

## Installation
```bash
pip install -e .[dev]
```

## Usage

Your original module entry points remain unchanged. This repository does **not**
rename functions or classes. Only comments/docstrings are translated.

### UI & Accessibility
- Global English-only headers and Google-style docstrings across all modules.
- Added Help/About buttons/menus (Tkinter/PyQt/Web recipes in `ui/`).
- Scrollable content areas with mouse-wheel support for small windows.
- Footer developer signature: `Created by Dr. Z. Bakhtiyorov`.

## How to cite

See `CITATION.cff`.

## Contributing

Run linters and tests before committing:

```bash
pre-commit install
pre-commit run --all-files
pytest
```
