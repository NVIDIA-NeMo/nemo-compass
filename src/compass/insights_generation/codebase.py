# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only, repository-confined tools for code-aware Problem validation."""

import os
import sys
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill

from compass.telemetry_phases import codebase_operation

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
_MAX_CACHE_BYTES = 64 * 1024 * 1024
_READ_PATHS: ContextVar[set[str] | None] = ContextVar("codebase_read_paths", default=None)


class CodebaseTools(Skill):
    """Search and read files inside one codebase without modifying it."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError(f"code_base must be an existing directory: {root}")
        self._texts: OrderedDict[Path, tuple[tuple[int, ...], str]] = OrderedDict()
        self._listings: dict[Path, tuple[dict[Path, tuple[int, ...]], list[Path]]] = {}
        self._cache_stats = dict(hits=0, bytes_read=0, cache_bytes=0)

    @hidden
    def _stamp(self, path: Path) -> tuple[int, ...]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size

    @hidden
    @contextmanager
    def capture_reads(self) -> Iterator[set[str]]:
        """Associate successful source reads with one concurrent investigation."""
        paths: set[str] = set()
        token = _READ_PATHS.set(paths)
        try:
            yield paths
        finally:
            _READ_PATHS.reset(token)

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

        if start in self._listings:
            directories, files = self._listings[start]
            try:
                if all(self._stamp(path) == stamp for path, stamp in directories.items()):
                    self._cache_stats["hits"] += 1
                    return files
            except OSError:
                pass

        files: list[Path] = []
        directories = {}
        for directory, directory_names, file_names in os.walk(start, followlinks=False):
            current = Path(directory)
            directories[current] = self._stamp(current)
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
        if len(self._listings) >= 8:
            self._listings.clear()
        self._listings[start] = directories, files
        return files

    @hidden
    def _read_text(self, path: Path) -> str | None:
        try:
            path = self._resolve(str(path))
            stamp = self._stamp(path)
            cached = self._texts.pop(path, None)
            if cached:
                self._cache_stats["cache_bytes"] -= sys.getsizeof(cached[1])
                if cached[0] == stamp:
                    self._texts[path] = cached
                    self._cache_stats["cache_bytes"] += sys.getsizeof(cached[1])
                    self._cache_stats["hits"] += 1
                    return cached[1]
            if stamp[-1] > _MAX_FILE_BYTES:
                return None
            with path.open("rb") as source:
                data = source.read(_MAX_FILE_BYTES + 1)
            self._cache_stats["bytes_read"] += len(data)
        except (OSError, ValueError):
            return None
        if len(data) > _MAX_FILE_BYTES or b"\0" in data:
            return None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        size = sys.getsizeof(text)
        while self._texts and self._cache_stats["cache_bytes"] + size > _MAX_CACHE_BYTES:
            self._cache_stats["cache_bytes"] -= sys.getsizeof(self._texts.popitem(last=False)[1][1])
        if size <= _MAX_CACHE_BYTES:
            self._texts[path] = stamp, text
            self._cache_stats["cache_bytes"] += size
        return text

    @codebase_operation
    def list_files(
        self,
        pattern: Annotated[
            str,
            spec(description="Optional repository-relative glob, such as src/**/*.py"),
        ] = "**/*",
    ) -> list[str]:
        """List the first 200 matching paths. If 200 return, narrow the glob for more.

        Prefer a relevant directory or filename pattern over a repository-wide list.
        """

        matches = [
            str(path.relative_to(self._root))
            for path in self._iter_files(self._root)
            if pattern == "**/*" or path.relative_to(self._root).match(pattern)
        ]
        return matches[:_MAX_LISTED_FILES]

    @codebase_operation
    def search_code(
        self,
        query: Annotated[str, spec(description="Literal text to search for")],
        path: Annotated[
            str,
            spec(description="Repository-relative file or directory to search"),
        ] = ".",
    ) -> list[dict[str, str | int]]:
        """Search UTF-8 code for a case-insensitive literal, not a regex.

        Search a known file or directory first; expand path only if needed.
        """

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

    @codebase_operation
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
        if (paths := _READ_PATHS.get()) is not None:
            paths.add(str(file_path.relative_to(self._root)))
        lines = text.splitlines()
        final_line = min(end_line or start_line + _MAX_READ_LINES - 1, len(lines))
        final_line = min(final_line, start_line + _MAX_READ_LINES - 1)
        width = len(str(final_line))
        return "\n".join(
            f"{line_number:>{width}}| {lines[line_number - 1]}"
            for line_number in range(start_line, final_line + 1)
        )


__all__ = ["CodebaseTools"]
