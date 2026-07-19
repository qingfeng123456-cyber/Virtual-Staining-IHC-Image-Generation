"""Cross-platform filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the source checkout root for editable and direct execution."""

    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory."""

    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to an explicit base without changing cwd."""

    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path(base) / value if base is not None else Path.cwd() / value
    return value.resolve()


def relative_or_absolute(path: str | Path, root: str | Path) -> str:
    """Represent a path relative to root when possible."""

    value = Path(path).resolve()
    base = Path(root).resolve()
    try:
        return value.relative_to(base).as_posix()
    except ValueError:
        return str(value)

