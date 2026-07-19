"""Fail closed on placeholder Python implementations and task markers."""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

MARKER_PATTERN = re.compile(r"\b(?:TODO|FIXME)\b")


def _python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in ("src", "tests", "scripts"):
        directory = root / relative
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    return sorted(set(files))


def _scan_file(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    findings: list[dict[str, object]] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.COMMENT:
            continue
        for match in MARKER_PATTERN.finditer(token.string):
            findings.append(
                {
                    "kind": "task_marker",
                    "line": token.start[0],
                    "value": match.group(0),
                }
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            findings.append({"kind": "pass_statement", "line": node.lineno})
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, (ast.Call, ast.Name))
            and (
                isinstance(node.exc, ast.Name)
                and node.exc.id == "NotImplementedError"
                or isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "NotImplementedError"
            )
        ):
            findings.append(
                {"kind": "not_implemented_error", "line": node.lineno}
            )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and value.value is Ellipsis:
                    findings.append(
                        {"kind": "ellipsis_body", "line": node.lineno}
                    )
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = _python_files(root)
    findings: list[dict[str, object]] = []
    for path in files:
        for finding in _scan_file(path):
            findings.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    **finding,
                }
            )
    report = {
        "files_scanned": len(files),
        "finding_count": len(findings),
        "findings": findings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(findings))


if __name__ == "__main__":
    sys.exit(main())
