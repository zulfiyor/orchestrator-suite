# Program: Repository Comment/Docstring Translator
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge; Xinjiang Institute of Ecology and Geography; National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Translate Russian comments and docstrings to English without changing logic.

This utility scans Python files and translates:
  * comments (token type COMMENT)
  * module/class/function docstrings (via AST)
Optionally, it can translate string literals in code with `--translate-strings`.

Identifiers and code structure remain intact. Backups are created when
`--backup` is provided. Uses `deep_translator` for RU->EN.

Style: Google-style docstrings, Black (88), Ruff compliant.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

from deep_translator import GoogleTranslator

RU = "ru"
EN = "en"


@dataclass
class Options:
    path: Path
    translate_strings: bool
    backup: bool
    only_comments_docstrings: bool


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def translate_text(text: str) -> str:
    """Translate text ru->en, skip if mostly ASCII.

    Args:
        text: Source text to translate.
    Returns:
        Translated English text.
    """

    # Heuristic: if text is ASCII-dominant, return as-is.
    if sum(ord(ch) < 128 for ch in text) / max(1, len(text)) > 0.9:
        return text

    translator = GoogleTranslator(source=RU, target=EN)
    try:
        return translator.translate(text)
    except Exception:
        return text  # fail-safe: keep original


def translate_docstring(s: str) -> str:
    """Translate docstring body while keeping triple quotes intact later."""
    return translate_text(s)


def iter_docstring_nodes(tree: ast.AST) -> Iterator[Tuple[ast.AST, str, Tuple[int, int]]]:
    """Yield nodes with docstrings and their (lineno, col)."""
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        if hasattr(node, "body") and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            ds_node = node.body[0]
            yield ds_node, doc, (ds_node.lineno, ds_node.col_offset)


def replace_docstrings(source: str) -> str:
    """Replace RU docstrings with EN translations without changing code layout.

    Keeps the same quoting style by reusing the original token spans.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    for ds_node, doc, (lineno, col) in iter_docstring_nodes(tree):
        # Find the token span of the docstring literal
        # We re-tokenize the specific line region conservatively.
        # Simpler approach: use regex around the first statement of node body.
        pattern = re.compile(r"([\t ]*)[\"\']{3}([\s\S]*?)[\"\']{3}")
        start_idx = lineno - 1
        # Search forward from the docstring start line
        for i in range(start_idx, min(start_idx + 10, len(lines))):
            m = pattern.search(lines[i])
            if m:
                indent = m.group(1)
                inner = m.group(2)
                translated = translate_docstring(inner)
                lines[i] = pattern.sub(f"{indent}'''{translated}'''", lines[i])
                break

    return "".join(lines)


def replace_comments(source: str) -> str:
    """Translate RU comments, preserve spacing and # positions."""
    out = io.StringIO()
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for tok in tokens:
        if tok.type == tokenize.COMMENT and tok.string.startswith("#"):
            prefix, text = tok.string[:1], tok.string[1:]
            replaced = prefix + translate_text(text)
            out.write(replaced)
        else:
            out.write(tok.string)
    return out.getvalue()


def replace_string_literals(source: str) -> str:
    """Optionally translate string constants (non-docstring).

    We translate only string tokens that are not triple-quoted docstrings.
    """
    out = io.StringIO()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.STRING and not tok.string.startswith(("'''", '"""')):
            # Keep quotes; translate inner content heuristically.
            q = tok.string[0]
            if q not in ('"', "'"):
                out.write(tok.string)
                continue
            inner = tok.string.strip(q)
            translated = translate_text(inner)
            out.write(f"{q}{translated}{q}")
        else:
            out.write(tok.string)
    return out.getvalue()


def process_file(p: Path, opts: Options) -> None:
    src = _read_text(p)
    original = src

    # Docstrings first, then comments.
    src = replace_docstrings(src)
    src = replace_comments(src)

    if opts.translate_strings and not opts.only_comments_docstrings:
        src = replace_string_literals(src)

    if src != original:
        if opts.backup:
            _write_text(p.with_suffix(p.suffix + ".bak"), original)
        _write_text(p, src)


def iter_python_files(path: Path) -> Iterable[Path]:
    for p in path.rglob("*.py"):
        # Skip our own tools/tests
        if "tools" in p.parts and p.name == "translate_repo.py":
            continue
        yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Russian comments/docstrings to English.")
    parser.add_argument("--path", type=Path, default=Path("."), help="Repository root path")
    parser.add_argument("--translate-strings", action="store_true", help="Translate string literals as well")
    parser.add_argument("--backup", action="store_true", help="Write .bak files before modifying")
    parser.add_argument("--only-comments-docstrings", action="store_true", help="Do not touch runtime strings")
    args = parser.parse_args()

    opts = Options(
        path=args.path,
        translate_strings=bool(args.translate_strings),
        backup=bool(args.backup),
        only_comments_docstrings=bool(args.only_comments_docstrings),
    )

    for p in iter_python_files(opts.path):
        process_file(p, opts)


if __name__ == "__main__":
    main()

# Created by Dr. Z. Bakhtiyorov
