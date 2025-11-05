# Program: Repository Comment/Docstring Translator
# Version: 0.1.0
# Author: Dr. Zulfiyor Bakhtiyorov
# Affiliations: University of Cambridge;
#   Xinjiang Institute of Ecology and Geography;
#   National Academy of Sciences of Tajikistan
# Year: 2025
# License: MIT License

"""Translate Russian comments and docstrings to English without changing logic.

This utility scans Python files and translates:
  * comments (token type COMMENT)
  * module/class/function docstrings (via AST)
Optionally, it can translate string literals in code with ``--translate-strings``.

Identifiers and code structure remain intact. Backups are created when
``--backup`` is provided. Uses ``deep_translator`` for RU->EN when available.

Style: Google-style docstrings, Black (88), Ruff compliant.
"""

from __future__ import annotations

import argparse
import ast
import re
import tokenize
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

try:  # Lazy import; tool should still work (no-op) without the dependency.
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    GoogleTranslator = None  # type: ignore

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
    """Translate text to English using Google Translate when available."""
    if not text.strip():
        return text
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(1, len(text))
    if ascii_ratio > 0.9:
        return text
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=RU, target=EN).translate(text)
    except Exception:
        return text


def iter_docstring_nodes(tree: ast.AST) -> Iterator[tuple[ast.AST, ast.Constant]]:
    """Yield AST nodes that contain docstrings."""
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            if node.body and isinstance(node.body[0], ast.Expr):
                expr = node.body[0]
                if isinstance(expr.value, ast.Constant) and isinstance(
                    expr.value.value, str
                ):
                    yield node, expr.value


def replace_docstrings(source: str) -> str:
    """Translate docstrings while keeping surrounding quotes intact."""
    tree = ast.parse(source)
    replacements: list[tuple[int, int, str]] = []
    for _node, const in iter_docstring_nodes(tree):
        literal = ast.get_source_segment(source, const)
        if literal is None:
            continue
        translated = translate_text(const.value)
        if translated == const.value:
            continue
        quote_match = re.match(r"([urbfURBF]*)(['\"]{3})(.*)(['\"]{3})", literal, re.S)
        if quote_match:
            prefix, quote, _body, suffix = quote_match.groups()
            new_literal = f"{prefix}{quote}{translated}{suffix}"
        else:
            new_literal = repr(translated)
        replacements.append((const.lineno, const.end_lineno, new_literal))

    if not replacements:
        return source

    lines = source.splitlines()
    for start, end, new_literal in sorted(replacements, reverse=True):
        lines[start - 1 : end] = [new_literal]
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


COMMENT_RE = re.compile(r"#(\s*)(.*)")


def replace_comments(source: str) -> str:
    """Translate comment tokens while preserving spacing."""
    buf = StringIO()
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            match = COMMENT_RE.match(tok.string)
            if match:
                spacing, body = match.groups()
                translated = translate_text(body)
                buf.write(f"#{spacing}{translated}")
            else:
                buf.write(tok.string)
        else:
            buf.write(tok.string)
    return buf.getvalue()


STRING_RE = re.compile(r"([urbfURBF]*)(['\"])(.*)(['\"])", re.S)


def replace_string_literals(source: str) -> str:
    """Translate non-docstring string literals."""
    buf = StringIO()
    tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    for tok in tokens:
        if tok.type == tokenize.STRING and not tok.string.startswith(("'''", '"""')):
            match = STRING_RE.match(tok.string)
            if not match:
                buf.write(tok.string)
                continue
            prefix, quote, body, suffix = match.groups()
            translated = translate_text(body)
            buf.write(f"{prefix}{quote}{translated}{suffix}")
        else:
            buf.write(tok.string)
    return buf.getvalue()


def process_file(p: Path, opts: Options) -> None:
    """Run translation passes on a single Python file."""
    src = _read_text(p)
    original = src
    src = replace_docstrings(src)
    src = replace_comments(src)
    if opts.translate_strings and not opts.only_comments_docstrings:
        src = replace_string_literals(src)
    if src != original:
        if opts.backup:
            _write_text(p.with_suffix(p.suffix + ".bak"), original)
        _write_text(p, src)


def iter_python_files(path: Path) -> Iterable[Path]:
    """Iterate through Python files in the given path."""
    for p in path.rglob("*.py"):
        if p.resolve() == Path(__file__).resolve():
            continue
        yield p


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Translate Russian comments/docstrings to English."
    )
    parser.add_argument(
        "--path", type=Path, default=Path("."), help="Repository root path"
    )
    parser.add_argument(
        "--translate-strings",
        action="store_true",
        help="Translate string literals as well",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Write .bak files before modifying"
    )
    parser.add_argument(
        "--only-comments-docstrings",
        action="store_true",
        help="Do not touch runtime strings",
    )
    args = parser.parse_args(argv)

    opts = Options(
        path=args.path,
        translate_strings=bool(args.translate_strings),
        backup=bool(args.backup),
        only_comments_docstrings=bool(args.only_comments_docstrings),
    )

    for py_path in iter_python_files(opts.path):
        process_file(py_path, opts)


if __name__ == "__main__":
    main()

# Created by Dr. Z. Bakhtiyorov
