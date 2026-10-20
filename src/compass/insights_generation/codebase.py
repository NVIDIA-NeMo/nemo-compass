# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only, repository-confined tools for code-aware Problem validation."""

import os
from pathlib import Path
from typing import Annotated

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill

_BLOCKED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_BLOCKED_FILENAMES = frozenset({".npmrc", ".pypirc", "credentials.json"})
_BLOCKED_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_MAX_FILE_BYTES = 1_000_000
_MAX_LISTED_FILES = 200
_MAX_MATCHES = 100
_MAX_READ_LINES = 400


class CodebaseTools(Skill):
    """Search and read files inside one codebase without modifying it."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError(f"code_base must be an existing directory: {root}")

    @hidden
    def _is_blocked(self, path: Path) -> bool:
        relative = path.relative_to(self._root)
        if any(part in _BLOCKED_DIRECTORIES for part in relative.parts):
            return True
        name = path.name.lower()
        return (
            name == ".env"
            or name.startswith(".env.")
            or name in _BLOCKED_FILENAMES
            or path.suffix.lower() in _BLOCKED_SUFFIXES
        )

    @hidden
    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"path escapes the configured codebase: {path}")
        if self._is_blocked(resolved):
            raise ValueError(f"path is not available to code validation: {path}")
        return resolved

    @hidden
    def _iter_files(self, start: Path) -> list[Path]:
        if start.is_file():
            return [start]
        if not start.is_dir():
            raise FileNotFoundError(f"path does not exist: {start.relative_to(self._root)}")

        files: list[Path] = []
        for directory, directory_names, file_names in os.walk(start, followlinks=False):
            current = Path(directory)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if name not in _BLOCKED_DIRECTORIES and not (current / name).is_symlink()
            )
            for name in sorted(file_names):
                path = current / name
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if not resolved.is_relative_to(self._root) or self._is_blocked(resolved):
                    continue
                files.append(resolved)
        return files

    @hidden
    def _read_text(self, path: Path) -> str | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if len(data) > _MAX_FILE_BYTES or b"\0" in data:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def list_files(
        self,
        pattern: Annotated[
            str,
            spec(description="Optional repository-relative glob, such as src/**/*.py"),
        ] = "**/*",
    ) -> list[str]:
        """List matching readable files, relative to the codebase root."""

        matches = [
            str(path.relative_to(self._root))
            for path in self._iter_files(self._root)
            if pattern == "**/*" or path.relative_to(self._root).match(pattern)
        ]
        return matches[:_MAX_LISTED_FILES]

    def search_code(
        self,
        query: Annotated[str, spec(description="Literal text to search for")],
        path: Annotated[
            str,
            spec(description="Repository-relative file or directory to search"),
        ] = ".",
    ) -> list[dict[str, str | int]]:
        """Search readable UTF-8 code for literal text, case-insensitively."""

        if not query:
            raise ValueError("query must not be empty")
        needle = query.casefold()
        matches: list[dict[str, str | int]] = []
        for file_path in self._iter_files(self._resolve(path)):
            text = self._read_text(file_path)
            if text is None:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if needle not in line.casefold():
                    continue
                matches.append(
                    {
                        "path": str(file_path.relative_to(self._root)),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                if len(matches) == _MAX_MATCHES:
                    return matches
        return matches

    def read_file(
        self,
        path: Annotated[str, spec(description="Repository-relative file path")],
        start_line: Annotated[int, spec(description="First line to read, starting at 1")] = 1,
        end_line: Annotated[
            int | None,
            spec(description="Last line to read; defaults to at most 400 lines"),
        ] = None,
    ) -> str:
        """Read a numbered excerpt from one UTF-8 text file."""

        file_path = self._resolve(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"file does not exist: {path}")
        if start_line < 1:
            raise ValueError("start_line must be at least 1")
        if end_line is not None and end_line < start_line:
            raise ValueError("end_line must not be before start_line")

        text = self._read_text(file_path)
        if text is None:
            raise ValueError(f"file is binary, non-UTF-8, or larger than {_MAX_FILE_BYTES} bytes")
        lines = text.splitlines()
        final_line = min(end_line or start_line + _MAX_READ_LINES - 1, len(lines))
        final_line = min(final_line, start_line + _MAX_READ_LINES - 1)
        width = len(str(final_line))
        return "\n".join(
            f"{line_number:>{width}}| {lines[line_number - 1]}"
            for line_number in range(start_line, final_line + 1)
        )


__all__ = ["CodebaseTools"]
