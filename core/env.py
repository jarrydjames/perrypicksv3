"""Environment loading helpers for scripts and workers."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable


_ROOT = Path(__file__).resolve().parents[1]


def _candidate_env_files(search_from: Path | None = None) -> Iterable[Path]:
    """Yield likely .env file locations in priority order."""
    seen: set[Path] = set()

    env_file = os.getenv("ENV_FILE")
    if env_file:
        candidate = Path(env_file).expanduser().resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

    if search_from is None:
        cwd = Path.cwd()
    else:
        cwd = search_from if search_from.is_dir() else search_from.parent

    for parent in [cwd] + list(cwd.parents):
        candidate = (parent / ".env").resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

    repo_env = (_ROOT / ".env").resolve()
    if repo_env not in seen:
        yield repo_env


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return None

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]

    return key, value


def load_environment(search_from: Path | None = None, override: bool = False) -> Path | None:
    """Load environment variables from the first discovered .env file.

    Returns the path of the loaded file, or None if no file is found.
    """
    for env_file in _candidate_env_files(search_from=search_from):
        if not env_file.exists() or not env_file.is_file():
            continue

        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value
        return env_file

    return None
